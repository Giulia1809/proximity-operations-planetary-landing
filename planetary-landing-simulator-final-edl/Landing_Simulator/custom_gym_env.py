import gym
from gym import spaces
import numpy as np
import main_a

#Define a Custom environment that follows gym interface --> define a class inheriting from gym.Env class

class Custom_lander_env(gym.Env):
  metadata = {'render.modes': ['human']}
  
  def __init__(self, ):
    super(Custom_lander_env, self).__init__()
    
    #define action and observation space: they must be gym.spaces objects -> continuous:
    #action is 4 floats - 4 thrusters: 0 = off, ..., up to 50=max power
    self.action_space = spaces.Box(0, 50, (4,), dtype=np.float)
    
    #image + lidar (100 rays) as input:
    self.observation_space = spaces.Box(low=0, high=255+100, shape=(HEIGHT, WIDTH, N_CHANNELS, N_RAYS), dtype=np.float32)
    #self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
    
    def step(self, action):
    #execute 1 time-step within the env
    
    def reset(self):
    #reset the state of the environment to an initial state
    
    def render(self, mode='human', close=False):
    #render the environment to the screen
