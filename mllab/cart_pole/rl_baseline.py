#!/usr/bin/env python

import os
import gym
import numpy as np
from abc import ABCMeta

from stable_baselines.common import policies as common_policies
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines.common import set_global_seeds
from stable_baselines import A2C, ACER, ACKTR, DDPG, GAIL, PPO1, PPO2, TRPO
from stable_baselines.deepq import DQN
from stable_baselines import deepq

stable_agents = [A2C, ACER, ACKTR, DQN, PPO1, PPO2, TRPO]

def create_agent(agnet_type):
    for a in stable_agents:
        if a.__name__==agnet_type:
            return a
    raise ValueError("could not find {} in stable_baselines agents".format(agnet_type))

def train(env, agnet, policy, steps):
    model = agent(env=env,
                  policy=policy,
                  verbose=0)
    model.learn(steps)
    return model

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

def evaluate(env,model,num_episodes=10, num_steps=100):
    """
    Evaluate a RL agent
    :param model: (BaseRLModel object) the RL Agent
    :param num_episodes: (int) number of episodes to evaluate the agent
    :return: (float) Mean reward
    """
    episode_rewards = [[[0.0] for _ in range(int(num_episodes//env.num_envs))] for _ in range(env.num_envs)]
    for e in range(int(num_episodes//env.num_envs)):
        obs = env.reset()
        for s in range(evaluation_steps_per_episode):
            # _states are only useful when using LSTM policies
            actions, _states = model.predict(obs)
            # here, action, rewards and dones are arrays
            # because we are using vectorized env
            obs, rewards, dones, info = env.step(actions)

            # Stats
            for i in range(env.num_envs):
                episode_rewards[i][e][-1] += rewards[i]
                if dones[i]:
                    episode_rewards[i][e].append(0.0)


    mean_rewards =  [0.0 for _ in range(env.num_envs)]
    std_rewards =  [0.0 for _ in range(env.num_envs)]
    n_episodes = 0
    for i in range(len(episode_rewards)):
        for e in range(len(episode_rewards[i])):
            mean_rewards_per_env = np.mean(episode_rewards[i][e])
            std_rewards_per_env = np.std(episode_rewards[i][e])
        mean_rewards[i] = np.mean(mean_rewards_per_env)
        std_rewards[i] = np.sum(std_rewards_per_env)

    # Compute mean reward
    mean_reward = round(np.mean(mean_rewards), 1)
    std_reward = round(np.sum(std_rewards), 1)
    print("Mean reward:", mean_reward, "Std reward:", std_reward)

    return mean_reward

if __name__ == '__main__':
    env_id = "CartPole-v1"
    num_cpu = 8  # Number of processes to use
    training_steps = int(1e6)
    evaluation_steps_per_episode = 500
    evaluation_episodes = num_cpu * 10
    agent_types = [m.__name__ for m in stable_agents]

    for agent_type in agent_types:
        model_name = "{0}-{1}".format(agent_type, env_id)

        print(model_name)
        policy = common_policies.MlpPolicy
        if agent_type == "DQN":
            policy = deepq.MlpPolicy
        # Create the vectorized environment
        if agent_type in ["DQN","PPO1","TRPO"]:
            env = DummyVecEnv([make_env(env_id, 0)])
        else:
            env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])
        agent = create_agent(agent_type)
        model=train(env,agent,policy,training_steps)
        model.save(os.path.join('./models',model_name))
        del model #To make sure model is saved
        model = agent.load(os.path.join('./models',model_name))
        evaluate(env,model,evaluation_episodes)
        env.close()
        del env