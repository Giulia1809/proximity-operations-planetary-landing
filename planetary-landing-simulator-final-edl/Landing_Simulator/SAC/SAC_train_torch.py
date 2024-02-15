import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

import numpy as np
import matplotlib.pyplot as plt
import gym #with pip installation already imports also box2d
import math
import random

from replay_buffer import ReplayBuffer
#from normalized_actions import NormalizedActions

#Load simplified environment - no atmospheric disturbances:
#import lander_gym_env
#from lander_gym_env import LanderGymEnv

#Load environment with gusts:
import lander_gym_env_with_gusts
from lander_gym_env_with_gusts import LanderGymEnv

print('OK! All imports successful!')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device set to : ' + str(torch.cuda.get_device_name(device)))


# Define Networks:

class ValueNetwork(nn.Module):
  def __init__(self, state_dim, hidden_dim, init_w=3e-3):
    super(ValueNetwork, self).__init__()
    
    self.linear1 = nn.Linear(state_dim, hidden_dim)
    self.linear2 = nn.Linear(hidden_dim, hidden_dim)
    self.linear3 = nn.Linear(hidden_dim, 1)
    
    self.linear3.weight.data.uniform_(-init_w, init_w)
    self.linear3.bias.data.uniform_(-init_w, init_w)
    
  def forward(self, state):
    x = nn.functional.relu(self.linear1(state))
    x = nn.functional.relu(self.linear2(x))
    x = self.linear3(x)
    return x
    
class SoftQNetwork(nn.Module):
  def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3):
    super(SoftQNetwork, self).__init__()
    
    self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
    self.linear2 = nn.Linear(hidden_size, hidden_size)
    self.linear3 = nn.Linear(hidden_size, 1)
    
    self.linear3.weight.data.uniform_(-init_w, init_w)
    self.linear3.bias.data.uniform_(-init_w, init_w)
    
  def forward(self, state, action):
    x = torch.cat([state, action], 1)
    x = nn.functional.relu(self.linear1(x))
    x = nn.functional.relu(self.linear2(x))
    x = self.linear3(x)
    return x
    
class PolicyNetwork(nn.Module):
  def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3, log_std_min=-20, log_std_max=2):
    super(PolicyNetwork, self).__init__()
    
    self.log_std_min = log_std_min
    self.log_std_max = log_std_max
    
    self.linear1 = nn.Linear(num_inputs, hidden_size)
    self.linear2 = nn.Linear(hidden_size, hidden_size)
    
    self.mean_linear = nn.Linear(hidden_size, num_actions)
    self.mean_linear.weight.data.uniform_(-init_w, init_w)
    self.mean_linear.bias.data.uniform_(-init_w, init_w)
    
    self.log_std_linear = nn.Linear(hidden_size, num_actions)
    self.log_std_linear.weight.data.uniform_(-init_w, init_w)
    self.log_std_linear.bias.data.uniform_(-init_w, init_w)
    
  def forward(self, state):
    x = nn.functional.relu(self.linear1(state))
    x = nn.functional.relu(self.linear2(x))
    mean = self.mean_linear(x)
    log_std = self.log_std_linear(x)
    log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
    return mean, log_std
    
  def evaluate(self, state, epsilon=1e-6):
    mean, log_std = self.forward(state)
    std = log_std.exp()
    normal = Normal(0, 1)
    z = normal.sample()
    action = torch.tanh(mean + std*z.to(device))
    log_prob = Normal(mean, std).log_prob(mean + std*z.to(device)) - torch.log(1 - action.pow(2) + epsilon)
    return action, log_prob, z, mean, log_std
    
  def get_action(self, state):
    state = torch.FloatTensor(state).unsqueeze(0).to(device)
    mean, log_std = self.forward(state)
    std = log_std.exp()
    normal = Normal(0, 1)
    z = normal.sample().to(device)
    action = torch.tanh(mean + std*z)
    action = action.cpu()
    return action[0]
    
  
def update(batch_size, gamma=0.99, soft_tau=1e-2):
  state, action, reward, next_state, done = replay_buffer.sample(batch_size)
  
  state = torch.FloatTensor(state).to(device)
  next_state = torch.FloatTensor(next_state).to(device)
  action = torch.FloatTensor(action).to(device)
  reward = torch.FloatTensor(reward).unsqueeze(1).to(device)
  done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)
  
  predicted_q_value1 = soft_q_net1(state, action)
  predicted_q_value2 = soft_q_net2(state, action)
  predicted_value = value_net(state)
  new_action, log_prob, epsilon, mean, log_std = policy_net.evaluate(state)
  
  #Q Function Training:
  target_value = target_value_net(next_state)
  target_q_value = reward + (1 - done) * gamma * target_value
  q_value_loss1 = soft_q_criterion1(predicted_q_value1, target_q_value.detach())
  q_value_loss2 = soft_q_criterion2(predicted_q_value2, target_q_value.detach())
  
  soft_q_optimizer1.zero_grad()
  q_value_loss1.backward()
  soft_q_optimizer1.step()
  
  soft_q_optimizer2.zero_grad()
  q_value_loss2.backward()
  soft_q_optimizer2.step()
  
  #Value Function Training:    
  predicted_new_q_value = torch.min(soft_q_net1(state, new_action), soft_q_net2(state, new_action))
  target_value_func = predicted_new_q_value - log_prob
  value_loss = value_criterion(predicted_value, target_value_func.detach())
  
  value_optimizer.zero_grad()
  value_loss.backward()
  value_optimizer.step()
  
  #Policy Function Training
  policy_loss = (log_prob - predicted_new_q_value).mean()
  policy_optimizer.zero_grad()
  policy_loss.backward()
  policy_optimizer.step()  
  
  for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
    target_param.data.copy_(target_param.data * (1.0 - soft_tau) + param.data * soft_tau)

def plot(frame_idx, rewards):
  plt.figure(figsize=(20,5))
  plt.subplot(131)
  plt.title('frame: %s. reward: %s' % (frame_idx, rewards[-1]))
  plt.plot(rewards)
  plt.show()
    
# Initialize and Run executable:

if __name__ == '__main__':
  
  env = LanderGymEnv(renders=False)
  #env = NormalizedActions(env)  
  print('OK! Environment configuration successful!')
  state_dim = env.observation_space.shape[0]
  print("Size of state space -> {}".format(state_dim))
  action_dim = env.action_space.shape[0]
  print("Size of action space -> {}".format(action_dim))
  upper_bound = env.action_space.high[0]
  lower_bound = env.action_space.low[0]
  print("max value of action -> {}".format(upper_bound))
  print("min value of action -> {}".format(lower_bound))
  hidden_dim = 256
  
  value_net = ValueNetwork(state_dim, hidden_dim).to(device)
  target_value_net = ValueNetwork(state_dim, hidden_dim).to(device)
  soft_q_net1 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)
  soft_q_net2 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)
  policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)
  
  for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
    target_param.data.copy_(param.data)
    
  value_criterion = nn.MSELoss()
  soft_q_criterion1 = nn.MSELoss()
  soft_q_criterion2 = nn.MSELoss()
  
  value_lr = 3e-4
  soft_q_lr = 3e-4
  policy_lr = 3e-4
  
  value_optimizer = optim.Adam(value_net.parameters(), lr=value_lr)
  soft_q_optimizer1 = optim.Adam(soft_q_net1.parameters(), lr=soft_q_lr)
  soft_q_optimizer2 = optim.Adam(soft_q_net2.parameters(), lr=soft_q_lr)
  policy_optimizer = optim.Adam(policy_net.parameters(), lr=policy_lr)
  
  replay_buffer_size=100000
  replay_buffer = ReplayBuffer(replay_buffer_size)
  
  #Define Training Hyperparameters:
  max_frames = 120
  max_steps = 500
  frame_idx = 0
  rewards = [] 
  avg_reward_list = []
  batch_size = 128
  #total_episodes = 10
  
  #Train with episodes:
  while frame_idx < max_frames:
    state = env.reset()
    episode_reward = 0
    print('frame_idx = ', frame_idx)

    
    for step in range(max_steps):
      if frame_idx > 50:
        action = policy_net.get_action(state).detach()
        next_state, reward, done, _ = env.step(action.numpy())
      elif frame_idx == 50:
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
      else: 
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
      
      replay_buffer.push(state, action, reward, next_state, done)
    
      state = next_state
      episode_reward += reward
      frame_idx += 1
    
      if len(replay_buffer) > batch_size:
        update(batch_size)
      
      if frame_idx % 1000 == 0:
        plot(frame_idx, rewards)
      
      if done:
        break
      
    rewards.append(episode_reward)
    avg_reward = np.mean(rewards[-100:])
    print("Frame * {} * Avg Reward is ==> {}".format(frame_idx, avg_reward))
    avg_reward_list.append(avg_reward)
    
  torch.save(value_net.state_dict(), 'SAC_weights/weights_value_net.pt')
  torch.save(target_value_net.state_dict(), 'SAC_weights/weights_target_value_net.pt')
  torch.save(soft_q_net1.state_dict(), 'SAC_weights/weights_soft_q_net1.pt')
  torch.save(soft_q_net2.state_dict(), 'SAC_weights/weights_soft_q_net2.pt')
  torch.save(policy_net.state_dict(), 'SAC_weights/policy_net.pt')
    
  plt.plot(avg_reward_list)
  plt.xlabel("Episodes")
  plt.ylabel("Avg. Episodic Reward")
  plt.show()
  plt.savefig('plot.png')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
