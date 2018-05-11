import gym
from gym import wrappers, logger

class PIDAgent(object):
    """A naive pid implementation"""
    def __init__(self, config, action_space):
        self.config = config


    def act(self, states, reward, done):
        x = states[0]
        x_dot = states[1]
        theta = states[2]
        theta_dot = states[3]
        
        err_theta = self.config['theta']['desired']-theta
        err_dot_theta = -theta_dot
        
        err_x = self.config['x']['desired']-x
        err_dot_x = -x_dot
        
        u_1 = self.config['theta']['kp']*err_theta + self.config['theta']['kd']*err_dot_theta
        u_2 = self.config['x']['kp']*err_x + self.config['x']['kd']*err_dot_x
        pid = self.config['theta']['c']*u_1 + self.config['x']['c']*u_2
        
        return 0 if pid>0 else 1


if __name__ == '__main__':
    """Test pid agent on OpenAI gym cart-pol-v1"""
    outdir = '/tmp/pid-agent-results'
    #logger.set_level(logger.WARN)
    env = gym.make('CartPole-v1')
    #env = wrappers.Monitor(env, directory=outdir, force=True)
    
    config = dict(theta=dict(),x=dict())
    config['theta']['desired'] = 0.0
    config['theta']['c'] = 0.8
    config['theta']['kp'] = 1.0
    config['theta']['kd'] = 0.5
    config['theta']['ki'] = 0.0
    
    config['x']['desired'] = 1.5
    config['x']['c'] = 0.2
    config['x']['kp'] = 0.7
    config['x']['kd'] = 1.0
    config['x']['ki'] = 0.0
    
    states = env.reset()
    desired_x = [-1.5,0.0,1.5]
    run_count = 10
    for n in range(run_count):
        score = 0
        done = False
        reward = 0.0
        config['x']['desired'] = desired_x[n%len(desired_x)]
        agent = PIDAgent(config, env.action_space)
        states = env.reset()
        while not done:
            action = agent.act(states, reward, done)
            states, reward, done, _ = env.step(action)
            print(states[0])
            score += reward
            env.render()
            #print(score)
    env.close()
    print("**********************")
    print("**********************")
    print(score)
    print("**********************")
    print("**********************")
