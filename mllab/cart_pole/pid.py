#!/usr/bin/env python

import gym
from gym import wrappers, logger
import math
import argparse

class PIDAgent(object):
    """A naive pid implementation"""
    def __init__(self, config, action_space):
        self.config = config
        self.err_tetha_sum = 0.0
        self.err_x_sum = 0.0
        self.x_sum_max = 0.05

    def act(self, states, reward, done):
        err_x = states[0]
        err_dot_x = states[1]
        err_theta = states[2]
        err_dot_theta = states[3]

        u_1 = self.config['theta']['kp']*err_theta + self.config['theta']['kd']*err_dot_theta

        self.err_x_sum += err_x
        self.err_x_sum = min(self.x_sum_max,self.err_x_sum)
        self.err_x_sum = max(-self.x_sum_max,self.err_x_sum)

        u_2 = self.config['x']['kp']*err_x + self.config['x']['kd']*err_dot_x + self.config['x']['ki']*self.err_x_sum
        pid = self.config['theta']['c']*u_1 + self.config['x']['c']*u_2

        return 0 if pid>0 else 1

def get_desired_x(traj_type, n):
    
    if traj_type=='sinusoidal':
        desired_amp = 1.0
        return desired_amp*math.sin(2.0*math.pi*10.0*score/500.0)
    elif traj_type=='switching':
        desired_x = [-1.0,1.0]
        return desired_x[n%len(desired_x)]
    elif traj_type=='center':
        return 0.0
    else:
        # Let's make it centered for now
        return 0.0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Controlling Cart-pole problem using PID.')
    parser.add_argument('--cart-traj-type', metavar='cart_traj_type', type=str, nargs='?', choices=['center', 'switching', 'sinusoidal'], default='center',
                   help='the type of desired trajecotry for the cart')
    args = parser.parse_args()
    print(args.cart_traj_type)

    """Test pid agent on OpenAI gym cart-pol-v1"""
    outdir = '/tmp/pid-agent-results'
    #logger.set_level(logger.WARN)
    env = gym.make('CartPole-v1')
    #env = wrappers.Monitor(env, directory=outdir, force=True)

    config = dict(theta=dict(),x=dict())
    config['theta']['desired'] = 0.2
    config['theta']['c'] = 0.8
    config['theta']['kp'] = 1.0
    config['theta']['kd'] = 0.75#0.3
    config['theta']['ki'] = 0.0
    
    config['x']['desired'] = 1.5
    config['x']['c'] = 0.2
    config['x']['kp'] = 1.05
    config['x']['kd'] = 1.95
    config['x']['ki'] = 0.5

    agent = PIDAgent(config, env.action_space)

    states = env.reset()
    err = [-s for s in states]
    run_count = 20
    avg_score = 0.0
    for n in range(run_count):
        score = 0
        done = False
        reward = 0.0
        states = env.reset()
        des_x = 0.0
        des_theta = 0.0
        while not done:
            action = agent.act(err, reward, done)
            states, reward, done, _ = env.step(action)
            last_des_x = des_x
            des_x = get_desired_x(args.cart_traj_type, n)
            last_des_theta = des_theta
            des_theta = 0.0*math.sin(2.0*math.pi*20.0*score/500.0)
            err[0] = des_x-states[0]
            err[1] = (des_x-last_des_x)-states[1]
            err[2] = des_theta-states[2]
            err[3] = (des_theta-last_des_theta)-states[3]
            #print(err[0])
            score += reward
            env.render()
        avg_score += score
        print("run #{}: score = {}".format(n+1,score))
    env.close()
    
    print("===================================")
    print("average score after {} runs = {}".format(n+1,avg_score/(n+1)))
    print("===================================")
