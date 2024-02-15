#import torch
import numpy as np
import gym

class NormalizedActions(gym.ActionWrapper):
  def action(self, action):
    low = self.action_space.low
    high = self.action_space.high
    
    action = low + (action + 1) * 0.5 * (high - low)
    action = np.clip(action, low, high)
    
  def _revers_action(self, action):
    low = self.action_space.low
    high = self.action_space.high
    
    action = 2 * (action - low) / (high - low) - 1
    action = np.clip(action, low, high)
   
    return action
