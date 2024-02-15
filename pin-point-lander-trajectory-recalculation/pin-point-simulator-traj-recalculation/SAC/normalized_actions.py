#import torch
import numpy as np
import gym

class NormalizedActions(gym.ActionWrapper):
  def action(self, action):
    low = self.env.action_space.low[0]
    high = self.env.action_space.high[0]
    
    action = low + (action + 1) * 0.5 * (high - low)
    action = np.clip(action, low, high)
    
  def _reverse_action(self, action):
    low = self.env.action_space.low[0]
    high = self.env.action_space.high[0]
    
    action = 2 * (action - low) / (high - low) - 1
    action = np.clip(action, low, high)
   
    return action
