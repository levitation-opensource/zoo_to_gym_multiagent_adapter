# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Repository: https://github.com/levitation-opensource/zoo_to_gym_multiagent_adapter

import os
import multiprocessing
import psutil

import torch

import gymnasium as gym
from pettingzoo import AECEnv, ParallelEnv


def saferecv(pipe, source_process):
    """
    pipe.recv() may hang when the other side dies, and not throw EOFError()
    as the documentation seems to promise. This function here attempts to
    work around that. Though in certain corner cases even this function
    here would not help.
    """
    while True:
        if pipe.poll(timeout=1):
            return pipe.recv()
        elif (
            isinstance(source_process, psutil.Process)
            and not source_process.is_running()
        ):  # parent process as psutil.Process
            raise EOFError()
        # elif isinstance(source_process, multiprocessing.Process) and not source_process.is_alive():   # child process as multiprocessing.Process
        #    raise EOFError()


class MultiAgentZooToGymAdapterGymSide(gym.Env):
    """
    A wrapper that transforms a PettingZoo environment with multiple agents
    into multiple single-agent Gymnasium environments by using threading.
    Each single-agent Gymnasium environment will run in its own thread.
    Both Zoo ParallelEnv and Zoo AECEnv (sequential env) are supported.
    In case of AECEnv, the agent's step observation is returned AFTER all
    other agents have taken their step as well.
    """

    def __init__(
        self, pipe, agent_id, checkpoint_filename, observation_space, action_space
    ):
        super().__init__()

        parent_pid = os.getppid()
        self.parent_process = psutil.Process(
            parent_pid
        )  # cannot get multiprocessing.Process for the parent, need to use psutil.Process

        (receiver_pipe, sender_pipe) = pipe
        self.receiver_pipe = receiver_pipe
        self.sender_pipe = sender_pipe

        self.agent_id = agent_id
        self.checkpoint_filename = checkpoint_filename

        self.observation_space = observation_space
        self.action_space = action_space

    def reset(self, seed=None, options=None, *args, **kwargs):
        """
        Reset the environment.
        Return: initial observation and an info dict
        """

        # send info to the main thread
        if not self.parent_process.is_running():
            raise BrokenPipeError()
        self.sender_pipe.send(["reset", seed, options, args, kwargs])

        # receive info from the main thread
        data = saferecv(self.receiver_pipe, self.parent_process)
        if data[0] == "reset_result":
            (_, observation, info) = data
            return observation, info
        elif data[0] == "force_termination":
            raise Exception("Forced termination")
        else:
            raise ValueError(
                "Unexpected response received from parent: " + str(data[0])
            )

    def step(self, action):
        """
        Take one step in the environment using the provided action.
        Return: observation, reward, done, truncated, info
        """

        # send info to the main thread
        if not self.parent_process.is_running():
            raise BrokenPipeError()
        self.sender_pipe.send(["step", action])

        # receive info from the main thread
        data = saferecv(self.receiver_pipe, self.parent_process)
        if data[0] == "step_result":
            (_, observation, reward, terminated, truncated, info) = data
            return observation, reward, terminated, truncated, info
        elif data[0] == "force_termination":
            raise Exception("Forced termination")
        else:
            raise ValueError(
                "Unexpected response received from parent: " + str(data[0])
            )

    def render(self, mode="human"):
        raise NotImplementedError()

    def close(self):
        """
        Close the environment.
        """
        pass

    def save_or_return_model(self, model, filename_timestamp_sufix_str=None):
        """
        Called outside of SB3 model, after SB3 training ends normally
        """
        if (
            self.checkpoint_filename is not None
        ):  # some SB3 models do not support pickling, thus it is better to save them directly from the training thread
            filename = self.checkpoint_filename
            if filename_timestamp_sufix_str is not None:
                filename += "-" + filename_timestamp_sufix_str

            model.save(filename)

            if not self.parent_process.is_running():
                raise BrokenPipeError()
            self.sender_pipe.send(["model_saved", filename])

        else:
            if not self.parent_process.is_running():
                raise BrokenPipeError()
            self.sender_pipe.send(["return_model", model])

    def terminate_with_exception(self, ex):
        """
        Called outside of SB3 model, after SB3 training ends with an exception
        """
        if not self.parent_process.is_running():
            raise BrokenPipeError()
        self.sender_pipe.send(["return_exception", ex])


class MultiAgentZooToGymAdapterZooSide(gym.Env):
    """
    A wrapper that transforms a PettingZoo environment with multiple agents
    into multiple single-agent Gymnasium environments by using threading.
    Each single-agent Gymnasium environment will run in its own thread.
    Both Zoo ParallelEnv and Zoo AECEnv (sequential env) are supported.
    In case of AECEnv, the agent's step observation is returned AFTER all
    other agents have taken their step as well.
    """

    def __init__(self, zoo_env, cfg):
        super().__init__()

        self.env = zoo_env
        self.cfg = cfg
        self.agent_ids = zoo_env.agents

        # self.observation_spaces = self.env.observation_spaces
        # self.action_spaces = self.env.action_spaces

    # TODO: automatically use singleagent wrapper if there is actually only one agent, do not create a subprocess in this case.
    def train(
        self,
        num_total_steps,
        agent_thread_entry_point,
        model_constructor,
        terminate_all_agents_when_one_excepts=True,
        checkpoint_filenames=None,
        seed=None,
        options=None,
        *args,
        **kwargs,
    ):
        """
        Reset the environment. Return initial observation and an info dict.
        """

        gpu_count = torch.cuda.device_count()
                  
        if gpu_count > 0:
            try:
                # per documentation, this method should be called only once
                multiprocessing.set_start_method("spawn", force=True)  # https://github.com/pytorch/pytorch/issues/40403
            except:
                pass

        # start Gym threads via agent_thread_entry_point, send (pipe, gpu_index, num_total_steps, model_constructor, agent_id, cfg, observation_space, action_space) to each
        self.agent_processes = []
        for agent_index, agent_id in enumerate(self.agent_ids):
            checkpoint_filename = (
                checkpoint_filenames[agent_id]
                if checkpoint_filenames is not None
                else None
            )

            observation_space = self.env.observation_spaces[agent_id]
            action_space = self.env.action_spaces[agent_id]

            # TODO: run some threads on CPU if the available GPU-s do not support the required amount of agent threads
            gpu_index = agent_index % gpu_count if gpu_count > 0 else None

            (receiver, thread_side_sender) = multiprocessing.Pipe(duplex=False)
            (thread_side_receiver, sender) = multiprocessing.Pipe(duplex=False)
            agent_thread_args = (
                (
                    thread_side_receiver,
                    thread_side_sender,
                ),  # NB! sender and receiver are swapped here
                gpu_index,
                num_total_steps,
                model_constructor,
                agent_id,
                checkpoint_filename,
                self.cfg,
                observation_space,
                action_space,
            )

            # TODO: Redirect stdio in such a manner that multi-line output from child processes does not get mixed in case of a race condition when multiple processes output at the same time. Though this is not urgent, currently it seems it does not get mixed up.
            # proc = await asyncio.create_subprocess_exec(
            #    "python",
            #    *args,
            #    env=env,
            #    stdout=asyncio.subprocess.PIPE,
            #    stderr=asyncio.subprocess.STDOUT,
            # )

            agent_process = multiprocessing.Process(
                target=agent_thread_entry_point, args=agent_thread_args
            )
            agent_process.start()
            psutil_agent_process = psutil.Process(agent_process.pid)

            training_complete = False
            is_in_reset = (
                False  # calling reset at the beginning of the episode is required
            )
            agent_process_data = [
                agent_id,
                agent_process,
                psutil_agent_process,
                is_in_reset,
                training_complete,
                (receiver, sender),
            ]  # TODO: use namedtuple
            self.agent_processes.append(agent_process_data)

        # / for agent_id in self.agent_ids:

        if isinstance(self.env, ParallelEnv):
            models, exceptions = self.parallel_env_main_loop(
                terminate_all_agents_when_one_excepts, seed, options, *args, **kwargs
            )
        elif isinstance(self.env, AECEnv):
            models, exceptions = self.sequential_env_main_loop(
                terminate_all_agents_when_one_excepts, seed, options, *args, **kwargs
            )  # TODO

        # Wait for the agent processes to shut down
        # It might be necessary in order to avoid GPU resources becoming overcommitted.
        for agent_process_data in self.agent_processes:
            (
                agent_id,
                agent_process,
                psutil_agent_process,
                is_in_reset,
                training_complete,
                (receiver, sender),
            ) = agent_process_data
            agent_process.join()  # Wait for the process to shut down. Note, psutil.Process does not have join() method, therefore using multiprocessing.Process here.

        return models, exceptions

    def parallel_env_main_loop(
        self,
        terminate_all_agents_when_one_excepts=True,
        seed=None,
        options=None,
        *args,
        **kwargs,
    ):
        models = {}
        exceptions = {}
        while len(models) + len(exceptions) < len(
            self.agent_processes
        ):  # loop over episodes until all agents have completed with a model or exception
            # receive resets from Gym threads
            for agent_process_data in self.agent_processes:
                (
                    agent_id,
                    agent_process,
                    psutil_agent_process,
                    is_in_reset,
                    training_complete,
                    (receiver, sender),
                ) = agent_process_data

                if (
                    is_in_reset or training_complete
                ):  # this agent has already called reset or finished training
                    continue
                # elif not psutil_agent_process.is_running():  # the process died unexpectedly
                #    exceptions[agent_id] = Exception("The agent process died unexpectedly")
                #    agent_process_data[4] = True  # training_complete = True
                #    continue

                # get command from Gym thread
                try:
                    data = saferecv(
                        receiver, psutil_agent_process
                    )  # TODO: handle processes that die during recv
                except EOFError:
                    exceptions[agent_id] = Exception(
                        "The agent process died unexpectedly"
                    )
                    agent_process_data[4] = True  # training_complete = True
                    continue

                if data[0] == "reset":
                    (_, seed, options, thread_reset_args, thread_reset_kwargs) = data
                    # we will ignore any data that was provided in the reset arguments
                    agent_process_data[3] = True  # is_in_reset = True

                elif data[0] == "model_saved":
                    (_, checkpoint_filename) = data
                    models[agent_id] = checkpoint_filename
                    agent_process_data[4] = True  # training_complete = True

                elif data[0] == "return_model":
                    (_, model) = data
                    models[agent_id] = model
                    agent_process_data[4] = True  # training_complete = True

                elif data[0] == "return_exception":
                    (_, ex) = data
                    exceptions[agent_id] = ex
                    agent_process_data[4] = True  # training_complete = True

                else:
                    raise ValueError(
                        "Unexpected command received from agent: " + str(data[0])
                    )

            # / for agent_process_data in self.agent_processes:

            # did remaining active agents send their models or exceptions during the reset loop above?
            if len(models) + len(exceptions) == len(self.agent_processes):
                break

            # call reset command on the Zoo environment
            if seed is not None:
                (observations, infos) = self.env.reset(
                    seed=seed, options=options, *args, **kwargs
                )
            else:
                (observations, infos) = self.env.reset(options=options, *args, **kwargs)

            # send reset results to Gym threads
            for agent_process_data in self.agent_processes:
                (
                    agent_id,
                    agent_process,
                    psutil_agent_process,
                    is_in_reset,
                    training_complete,
                    (receiver, sender),
                ) = agent_process_data

                if training_complete:
                    continue
                elif (
                    not psutil_agent_process.is_running()
                ):  # the process died unexpectedly
                    exceptions[agent_id] = Exception(
                        "The agent process died unexpectedly"
                    )
                    agent_process_data[4] = True  # training_complete = True
                    continue

                assert is_in_reset

                # send result to Gym thread
                observation = observations[agent_id]
                info = infos[agent_id]
                try:
                    sender.send(
                        ["reset_result", observation, info]
                    )  # Until the buffer is full, it is okay to send to a terminated process. On the next loop the code will check for is_running() condition anyway.
                except BrokenPipeError:
                    exceptions[agent_id] = Exception(
                        "The agent process died unexpectedly"
                    )
                    agent_process_data[4] = True  # training_complete = True
                    continue

            # / for agent_process_data in self.agent_processes:

            # run episodes and steps until all Gym threads are done
            while True:  # loop over steps
                # receive step actions or other commands from Gym threads
                actions = {}
                resets = set()
                for agent_process_data in self.agent_processes:
                    (
                        agent_id,
                        agent_process,
                        psutil_agent_process,
                        is_in_reset,
                        training_complete,
                        (receiver, sender),
                    ) = agent_process_data

                    if (
                        training_complete
                    ):  # TODO: send random actions for agents with completed training?
                        continue
                    # elif not psutil_agent_process.is_running():  # the process died unexpectedly
                    #    exceptions[agent_id] = Exception(
                    #        "The agent process died unexpectedly"
                    #    )
                    #    agent_process_data[4] = True  # training_complete = True
                    #    continue

                    # repeat reset handling loop
                    while (
                        True
                    ):  # may need to loop if the Gym environment sends multiple resets in a sequence
                        # get command from Gym thread
                        try:
                            data = saferecv(
                                receiver, psutil_agent_process
                            )  # TODO: handle processes that die during recv
                        except EOFError:
                            exceptions[agent_id] = Exception(
                                "The agent process died unexpectedly"
                            )
                            agent_process_data[4] = True  # training_complete = True
                            break  # break repeat reset handling loop

                        if data[0] == "reset":
                            (
                                _,
                                seed,
                                options,
                                thread_reset_args,
                                thread_reset_kwargs,
                            ) = data
                            # we will ignore any data that was provided in the reset arguments

                            if (
                                is_in_reset
                            ):  # if previous command was also reset, then ignore the new reset
                                # send previous reset result again to the Gym thread
                                observation = observations[agent_id]
                                info = infos[agent_id]
                                try:
                                    sender.send(["reset_result", observation, info])
                                    continue  # try again to get a step command
                                except BrokenPipeError:
                                    exceptions[agent_id] = Exception(
                                        "The agent process died unexpectedly"
                                    )
                                    agent_process_data[
                                        3
                                    ] = True  # training_complete = True
                                    break  # break repeat reset handling loop
                            else:
                                agent_process_data[3] = True  # is_in_reset = True
                                resets.add(agent_id)
                                break  # break repeat reset handling loop

                        elif data[0] == "step":
                            agent_process_data[3] = False  # is_in_reset = False
                            (_, action) = data
                            actions[agent_id] = action
                            break  # break repeat reset handling loop

                        elif data[0] == "model_saved":
                            (_, checkpoint_filename) = data
                            models[agent_id] = checkpoint_filename
                            agent_process_data[4] = True  # training_complete = True
                            break  # break repeat reset handling loop

                        elif data[0] == "return_model":
                            (_, model) = data
                            models[agent_id] = model
                            agent_process_data[4] = True  # training_complete = True
                            break  # break repeat reset handling loop

                        elif data[0] == "return_exception":
                            (_, ex) = data
                            exceptions[agent_id] = ex
                            agent_process_data[4] = True  # training_complete = True
                            break  # break repeat reset handling loop

                        else:
                            raise ValueError(
                                "Unexpected command received from agent: "
                                + str(data[0])
                            )

                    # / while True:   # may need to loop if the Gym environment sends multiple resets in a sequence

                # / for agent_process_data in self.agent_processes:

                if not any(actions):
                    # all remaining active agents called reset or returned a model or exception at the same time
                    break  # go to outer loop over episodes

                elif terminate_all_agents_when_one_excepts and any(exceptions):
                    # we will below force-terminate all other training threads, so nothing needs to be done here (after which the agent will trigger an exception and exit the thread/process)
                    pass

                # TODO: Make difference between episode end due to truncation and dead agents. In case of dead agents, DO NOT reset other agents.
                elif any(resets) or any(exceptions) or any(models):
                    # TODO: this could be configurable: If at least one Gym thread decides to reset or finish training (with an exception or a model result), then send termination responses to all other threads and wait for them to call reset

                    # TODO: Maybe there is no need to get new observations from the environment in this case?
                    (
                        observations,
                        rewards,
                        terminations,
                        truncations,
                        infos,
                    ) = self.env.step(actions)

                    # Sutton and Barto uses terminal states to specifically refer to special states whose values are 0, states at the end of the MDP. This is not true for a truncation where the value of the final state need not be 0.
                    # https://github.com/openai/gym/pull/2752
                    # terminate all agents regardless of what the environment responded
                    terminations = {agent_id: True for agent_id in terminations.keys()}

                else:
                    # call step on the Zoo environment
                    (
                        observations,
                        rewards,
                        terminations,
                        truncations,
                        infos,
                    ) = self.env.step(actions)

                # First check that all processes are still alive. This is important so that exceptions dict will be populated and thus the other processes can be sent termination commands via step_result if one of them had died unexpectedly.
                for agent_process_data in self.agent_processes:
                    (
                        agent_id,
                        agent_process,
                        psutil_agent_process,
                        is_in_reset,
                        training_complete,
                        (receiver, sender),
                    ) = agent_process_data

                    if (
                        not psutil_agent_process.is_running()
                    ):  # the process died unexpectedly
                        exceptions[agent_id] = Exception(
                            "The agent process died unexpectedly"
                        )
                        agent_process_data[4] = True  # training_complete = True

                # send results to Gym threads
                for agent_process_data in self.agent_processes:
                    (
                        agent_id,
                        agent_process,
                        psutil_agent_process,
                        is_in_reset,
                        training_complete,
                        (receiver, sender),
                    ) = agent_process_data

                    if training_complete:
                        continue

                    try:
                        if terminate_all_agents_when_one_excepts and any(exceptions):
                            sender.send(
                                ["force_termination"]
                            )  # the agent will later respond with a return_exception command
                        elif (
                            is_in_reset
                        ):  # NB! This condition is checked only after forced termination branch above is skipped. In other words, agents in reset also should get forced termination if needed.
                            continue  # reset results will be sent later when all agents have called reset
                        else:
                            observation = observations[agent_id]
                            reward = rewards[agent_id]
                            terminated = terminations[agent_id]
                            truncated = truncations[agent_id]
                            info = infos[agent_id]

                            # send results to Gym threads
                            sender.send(
                                [
                                    "step_result",
                                    observation,
                                    reward,
                                    terminated,
                                    truncated,
                                    info,
                                ]
                            )  # Until the buffer is full, it is okay to send to a terminated process. On the next loop the code will check for is_running() condition anyway.
                    except BrokenPipeError:
                        exceptions[agent_id] = Exception(
                            "The agent process died unexpectedly"
                        )
                        agent_process_data[4] = True  # training_complete = True
                        continue

                # / for agent_process_data in self.agent_processes:

                if terminate_all_agents_when_one_excepts and any(exceptions):
                    break  # the training threads were sent force-termination commands (the agent will trigger an exception and exit the thread/process)
                elif any(resets) or any(exceptions) or any(models):
                    break  # TODO: this could be configurable: The agents were sent Gym terminated statuses above (the agent continues running, but starts a new episode by calling reset command next)

            # / while True: # loop over steps

        # / while len(models) + len(exceptions) < len(self.agent_processes): # loop over episodes

        return models, exceptions

    # def sequential_env_main_loop(self, terminate_all_agents_when_one_excepts=True, seed, options, *args, **kwargs):

    # TODO
