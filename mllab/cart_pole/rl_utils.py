#!/usr/bin/env python

import numpy as np
import gym

from stable_baselines import A2C, ACER, ACKTR, DDPG, GAIL, PPO1, PPO2, TRPO
from stable_baselines.deepq import DQN
from stable_baselines.common import set_global_seeds

g_stable_agents = [A2C, ACER, ACKTR, PPO2, DQN, PPO1, TRPO]

def create_agent(agnet_type):
    for a in g_stable_agents:
        if a.__name__==agnet_type:
            return a
    raise ValueError("could not find {} in stable_baselines agents".format(agnet_type))

def make_env(env_id, rank, seed=0):
    """
    Utility function for multiprocessed env.
    
    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environment you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = gym.make(env_id)
        env.seed(seed + rank)
        return env
    set_global_seeds(seed)
    return _init