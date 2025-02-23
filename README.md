
# Zoo to Gym Multi-Agent Adapter


## Overview

This wrapper allows you to convert a **PettingZoo** environment to a **Gym** environment while supporting multiple agents. Gym's default setup doesn't easily support multi-agent environments, but this wrapper resolves that by running each agent in its own process and sharing the environment across those processes.


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
pip install git+https://github.com/levitation-opensource/zoo_to_gym_multiagent_adapter@main#egg=zoo_to_gym_multiagent_adapter
```


## Usage

TODO

For an example, see 
<br> https://github.com/aintelope/biological-compatibility-benchmarks/tree/main/aintelope/agents/dqn_agent.py
<br>and
<br> https://github.com/aintelope/biological-compatibility-benchmarks/tree/main/aintelope/agents/sb3_base_agent.py


## How It Works

* The wrapper handles the conversion of the PettingZoo environment API into a Gym-compatible API.
* When there are multiple agents, each agent runs in its own process, and the wrapper ensures synchronization and data sharing between them, allowing you to train multi-agent RL scenarios using Gym.
* This solves the issue of Gym's default lack of native support for multiple agents in a single environment.


## Contributions

Contributions are welcome! Feel free to fork the repository, open issues, and submit pull requests.


## License

This project is licensed under the Mozilla Public License Version 2.0 - see the [LICENSE.txt](LICENSE.txt) file for details.
