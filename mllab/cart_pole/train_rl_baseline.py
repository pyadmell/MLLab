#!/usr/bin/env python

import os
import gym
import numpy as np
from abc import ABCMeta

from evaluate_rl_baseline import evaluate
from rl_utils import g_stable_agents, make_env, create_agent

from stable_baselines.common import policies as common_policies
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines import deepq

def train(env, agnet, policy, steps):
    model = agent(env=env,
                  policy=policy,
                  verbose=0)
    model.learn(steps)
    return model

def main():
    env_id = "CartPole-v1"
    num_cpu = 8  # Number of processes to use
    training_steps = int(1e4)
    agent_types = [m.__name__ for m in g_stable_agents]

    for agent_type in agent_types:
        model_name = "{0}-{1}".format(agent_type, env_id)

        print(model_name)
        policy = common_policies.MlpPolicy
        if agent_type == "DQN":
            policy = deepq.MlpPolicy
        # Create the vectorized environment
        env = DummyVecEnv([make_env(env_id, 0)])
        """
        if agent_type in ["DQN","PPO1","TRPO"]:
            env = DummyVecEnv([make_env(env_id, 0)])
        else:
            env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])
        """
        agent = create_agent(agent_type)
        model=train(env,agent,policy,training_steps)
        model.save(os.path.join('./models',model_name))
        del model #To make sure model is saved
        for e in env.envs:
            e.close()
            del e
        env.close()
        del env

if __name__ == '__main__':
    main()
    print("done!")