# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Repository: https://github.com/levitation-opensource/zoo_to_gym_multiagent_adapter

from typing import List, NamedTuple, Optional, Tuple

import numpy as np
import os
import datetime

from zoo_to_gym_multiagent_adapter.singleagent_zoo_to_gym_adapter import (
    SingleAgentZooToGymAdapter,
)

import torch
from stable_baselines3 import DQN


def is_json_serializable(item: Any) -> bool:
    return False


def sb3_agent_train_thread_entry_point(
    pipe,
    gpu_index,
    num_total_steps,
    model_constructor,
    agent_id,
    checkpoint_filename,
    cfg,
    observation_space,
    action_space,
):
    # activate selected GPU
    select_gpu(gpu_index)

    env_wrapper = MultiAgentZooToGymAdapterGymSide(
        pipe, agent_id, checkpoint_filename, observation_space, action_space
    )
    try:
        model = model_constructor(env_wrapper, cfg)
        model.learn(total_timesteps=num_total_steps)
        filename_timestamp_sufix_str = datetime.datetime.now().strftime(
            "%Y_%m_%d_%H_%M_%S_%f"
        )
        env_wrapper.save_or_return_model(model, filename_timestamp_sufix_str)
    except (
        Exception
    ) as ex:  # NB! need to catch exception so that the env wrapper can signal the training ended
        info = str(ex) + os.linesep + traceback.format_exc()
        env_wrapper.terminate_with_exception(info)
        print(info)


# need separate function outside of class in order to init multi-model training threads
def dqn_model_constructor(env, cfg):
    return DQN(
        env,
        verbose=1,
        device=torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),  # Note, CUDA-based CPU performance is much better than Torch-CPU mode.
    )


class DQNAgent:

    def __init__(
        self,
        agent_id: str,
        env: ParallelEnv,
        cfg,
        test_mode: bool = False,
    ) -> None:
        self.id = agent_id
        self.cfg = cfg
        self.env = env
        self.test_mode = test_mode
        self.last_action = None
        self.model = None  # for single-model scenario
        self.models = None  # for multi-model scenario
        self.exceptions = None  # for multi-model scenario
        self.model_constructor = None  # for multi-model scenario

        stable_baselines3.common.save_util.is_json_serializable = is_json_serializable  # The original function throws many "Pythonic" exceptions which make debugging in Visual Studio too noisy since VS does not have capacity to filter out handled exceptions

        self.model_constructor = dqn_model_constructor

        if (
            self.env.num_agents == 1 or self.test_mode
        ):  # during test, each agent has a separate in-process instance with its own model and not using threads/subprocesses
            env = SingleAgentZooToGymAdapter(env, self.id)
            self.model = self.model_constructor(env, cfg)
        else:
            pass  # multi-model training will be automatically set up by the base class when self.model is None. These models will be saved to self.models and there will be only one agent instance in the main process. Actual agents will run in threads/subprocesses because SB3 requires Gym interface.

    # called during test
    def get_action(
        self,
        observation,
    ) -> Optional[int]:
        """Given an observation, ask your model what to do. 
        Called during test only, not during training."""

        # TODO: Make sure your observation image is in the channel-first format.
        action, _states = self.model.predict(
            observation, deterministic=True
        )  # TODO: config setting for "deterministic" parameter
        action = np.asarray(
            action
        ).item()  # SB3 sends actions in wrapped into an one-item array for some reason. np.asarray is also able to handle lists.

        return action

    def train(self, num_total_steps):
        if self.model is not None:  # single-model scenario
            self.model.learn(total_timesteps=num_total_steps)
        else:
            checkpoint_filenames = self.get_checkpoint_filenames(
                include_timestamp=False
            )
            env_wrapper = MultiAgentZooToGymAdapterZooSide(self.env, self.cfg)
            self.models, self.exceptions = env_wrapper.train(
                num_total_steps=num_total_steps,
                agent_thread_entry_point=sb3_agent_train_thread_entry_point,
                model_constructor=self.model_constructor,
                terminate_all_agents_when_one_excepts=True,
                checkpoint_filenames=checkpoint_filenames,
            )

        if self.exceptions:
            raise Exception(str(self.exceptions))

    def get_checkpoint_filenames(self, include_timestamp=True):
        checkpoint_filenames = {}

        filename_timestamp_sufix_str = datetime.datetime.now().strftime(
            "%Y_%m_%d_%H_%M_%S_%f"
        )

        for agent_id in self.env.possible_agents:
            filename = "checkpoint-" + agent_id
            if include_timestamp:
                filename += "-" + filename_timestamp_sufix_str
            checkpoint_filenames[agent_id] = filename

        return checkpoint_filenames

    def save_model(self):
        checkpoint_filenames = self.get_checkpoint_filenames(include_timestamp=True)
        models = {self.id: self.model} if self.model is not None else self.models

        for agent_id, model in models.items():
            if not isinstance(
                model, str
            ):  # model can contain a path to an already saved model
                checkpoint_filename = checkpoint_filenames[agent_id]
                model.save(checkpoint_filename)

    def init_model(
        self,
        checkpoint: Optional[str] = None,
    ):
        if checkpoint:
            # NB! torch.cuda.device_count() > 0 is needed here since SB3 does not support CPU-based CUDA device during model load() or set_parameters() for some reason
            use_cuda = torch.cuda.is_available() and torch.cuda.device_count() > 0
            device = torch.device("cuda" if use_cuda else "cpu")

            # Warning: load() re-creates the model from scratch, it does not update it in-place! For an in-place load use set_parameters() instead.
            self.model.set_parameters(
                checkpoint, device=device
            )  # device argument in needed in case the model is loaded to CPU. SB3 seems to be buggy in that regard that it will crash during model load() or set_parameters() if Torch-CPU device is not explicitly specified.

def main():
  
    env = construct_your_multi_agent_zoo_env_here()
    cfg = {}


    # Training. During training there is only one agent object which acts as a proxy/trainer for training all agents in a multi-agent environment.

    agent = DQNAgent(
        agent_id="agent_trainer",
        env=env,
        cfg=cfg,
        test_mode=False,
    )
    agent.init_model()
    agent.train(num_total_steps)
    agent.save_model()


    # Testing

    observations, infos = env.reset()

    for agent_id in env.possible_agents:
        agent = DQNAgent(
            agent_id=agent_id,
            env=env,
            cfg=cfg,
            test_mode=True,
        )
        model_file = glob.glob("checkpoint-" + agent_id)[0]
        agent.init_model(model_file)


    while env.agents:

        actions = {}
        for agent_id in env.agents:
            observation = observations[agent_id]
            actions[agent_id] = agent.get_action(observation)

        observations, rewards, terminations, truncations, infos = env.step(actions)


if __name__ == "__main__":
    main()
