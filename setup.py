from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

try:
    import setuptools
except ImportError:
    from ez_setup import use_setuptools

    use_setuptools()
    import setuptools

setuptools.setup(
    name="zoo_to_gym_multiagent_adapter",
    version="0.9.0",
    description="Zoo to Gym Multi-Agent Adapter",
    long_description=(
        "Extended, multi-agent and multi-objective environments based on DeepMind's "
        "AI Safety Gridworlds. "
        "This is a suite of reinforcement learning environments illustrating "
        "various safety properties of intelligent agents. "
        "It is made compatible with OpenAI's Gym and Gymnasium "
        "and Farama Foundation PettingZoo."
    ),
    url="https://github.com/levitation-opensource/zoo_to_gym_multiagent_adapter",
    author="Roland Pihlakas",
    author_email="roland@simplify.ee",
    license="Mozilla Public License Version 2.0",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Environment :: Console",
        "Environment :: Console :: Curses",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Topic :: Games/Entertainment :: Arcade",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Testing",
    ],
    keywords=(
        "ai "
        "artificial intelligence "
        "game engine "
        "gridworld "
        "gym "
        "gymnasium "
        "marl "
        "multi-agent "
        "pettingzoo "
        "reinforcement learning "
        "rl "
    ),
    install_requires=[
      "gymnasium",
      "PettingZoo",
      "psutil" 
    ],
    packages=setuptools.find_packages(),
    zip_safe=True,
    entry_points={},
    package_data={},
)
