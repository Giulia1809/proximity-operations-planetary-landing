#import torch
import numpy as np
import math
import random
#import itertools


class ReplayBuffer:
  def __init__(self, capacity):
    self.capacity = capacity
    self.buffer = []
    self.position = 0
    
  def push(self, state, action, reward, next_state, done):
    if len(self.buffer) < self.capacity:
      self.buffer.append(None)
      #self.buffer.append([0])
    self.buffer[self.position] = (state, action, reward, next_state, done)
    self.position = (self.position + 1) % self.capacity 
    
  def sample(self, batch_size):
    #torch.cuda.empty_cache()
    batch = random.sample(self.buffer, batch_size)
    #print(batch)
    #print('len batch = ', len(batch))
    state, action, reward, next_state, done = map(np.stack, zip(*batch))
    return state, action, reward, next_state, done
    
  def __len__(self):
    return len(self.buffer)
