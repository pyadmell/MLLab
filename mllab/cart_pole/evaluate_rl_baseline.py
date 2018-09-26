#!/usr/bin/env python

import os
import gym
import numpy as np

from rl_utils import g_stable_agents, make_env, create_agent

from stable_baselines.common import policies as common_policies
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines import deepq

def evaluate(env,model,num_episodes=10, num_steps=100, render=False):
    """
    Evaluate a RL agent
    :param model: (BaseRLModel object) the RL Agent
    :param num_episodes: (int) number of episodes to evaluate the agent
    :return: (float) Mean reward
    """
    episode_rewards = [0.0 for _ in range(num_episodes)]
    for e in range(num_episodes):
        obs = env.reset()
        for s in range(num_steps):
            # _states are only useful when using LSTM policies
            actions, _states = model.predict(obs)
            # here, action, rewards and dones are arrays
            # because we are using vectorized env
            obs, rewards, dones, info = env.step(actions)

            # Stats
            episode_rewards[e] += rewards[0]
            if True in dones:
                break
            else:
                if render:
                    env.render()

    # Compute mean reward
    mean_reward = round(np.mean(episode_rewards), 1)
    std_reward = round(np.std(episode_rewards), 1)
    print("Num Episodes: {0} \t Num Steps per Ep: {1} \t Mean reward: {2} \t Std reward: {3}".format(num_episodes, 
                                                                                                     num_steps, 
                                                                                                     mean_reward, 
                                                                                                     std_reward))

    return mean_reward

def main():
    env_id = "CartPole-v1"
    num_cpu = 1  # Number of processes to use
    evaluation_steps_per_episode = 500
    evaluation_episodes = 10
    render = True
    agent_types = [m.__name__ for m in g_stable_agents]
    for agent_type in agent_types:
        model_name = "{0}-{1}".format(agent_type, env_id)

        print(model_name)
        policy = common_policies.MlpPolicy
        if agent_type == "DQN":
            policy = deepq.MlpPolicy
        # Create the vectorized environment
        env = DummyVecEnv([make_env(env_id=env_id,
                                    rank=0,
                                    seed=0)])
        agent = create_agent(agent_type)
        model = agent.load(os.path.join('./models',model_name))
        evaluate(env=env,
                 model=model,
                 num_episodes=evaluation_episodes,
                 num_steps=evaluation_steps_per_episode,
                 render=render)
        for e in env.envs:
            e.close()
            del e
        env.close()
        del env

if __name__ == '__main__':
    main()
    print("done!")