
# Zoo to Gym Multi-Agent Adapter


## Overview

This wrapper allows you to convert a **PettingZoo** environment to a **Gym** environment while supporting multiple agents (MARL). Gym's default setup doesn't easily support multi-agent environments, but this wrapper resolves that by running each agent in its own process and sharing the environment across those processes.


## Features

* Converts a **PettingZoo** environment to a **Gym**-compatible environment.
* **Multi-agent** support by running agents in separate processes.
* Thus enables using Gym-based RL algorithms with **PettingZoo** environments. 
* There are two adapters:
    * Single-agent adapter - Runs in the same process.
	* Multi-agent adapter - Runs each agents in a separate subprocess.


## Installation

Currently you can install the wrapper from the source:


### From Source:

```
pip install git+https://github.com/biological-alignment-benchmarks/zoo_to_gym_multiagent_adapter@main#egg=zoo_to_gym_multiagent_adapter
```


## Usage

For an example, see 

[example/usage_example.py](example/usage_example.py)

Alternatively, for a more comprehensive usage example, see "Biologically and economically aligned multi-objective multi-agent gridworld-based AI safety benchmarks" repo:

https://github.com/biological-alignment-benchmarks/biological-alignment-gridagents-benchmarks/tree/main/gridagents/agents/dqn_agent.py
<br>and
<br> https://github.com/biological-alignment-benchmarks/biological-alignment-gridagents-benchmarks/tree/main/gridagents/agents/sb3_base_agent.py


## How It Works

* The wrapper handles the conversion of the PettingZoo environment API into a Gym-compatible API.
* When there are multiple agents, each agent runs in its own process, and the wrapper ensures synchronization and data sharing between them, allowing you to train multi-agent RL scenarios using Gym.
* This solves the issue of Gym's default lack of native support for multiple agents in a single environment.


## Current project state

Ready to use. Is actively developed further.


## Contributions

Contributions are welcome! Feel free to fork the repository, open issues, and submit pull requests.


## License

This project is licensed under the Mozilla Public License 2.0 - see the [LICENSE.txt](LICENSE.txt) file for details. You are free to use, modify, and distribute this code under the terms of this license.

**Attribution Requirement**: If you use this library, please cite the source as follows:

Roland Pihlakas. From homeostasis to resource sharing: Biologically and economically aligned multi-objective multi-agent gridworld-based AI safety benchmarks. Arxiv, a working paper, September 2024 (https://arxiv.org/abs/2410.00081).

Original upstream repository: [https://github.com/biological-alignment-benchmarks/zoo_to_gym_multiagent_adapter](https://github.com/biological-alignment-benchmarks/zoo_to_gym_multiagent_adapter)



