import math
import time
import numpy as np
import random
import os
import pybullet as p
import pybullet_data
#from pybullet_utils import bullet_client as bc
import bullet_client as bc
import gym
from gym import spaces
from gym.utils import seeding
#from lander import Lander
import lander

RENDER_HEIGHT = 720
RENDER_WIDTH = 960

#Define lander environment class, inheriting from gym.Env and override methods:
class LanderGymEnv(gym.Env):
  metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second':50}
  
  def __init__(self, urdfRoot = pybullet_data.getDataPath(), actionRepeat=50, isEnableSelfCollision=True, isDiscrete=False, renders=True, recalculateSpot=False):
  
    print("init")
    self._timeStep = 0.01
    self._urdfRoot = urdfRoot
    #self._cubeStartPos = cubeStartPos
    #self._cubeStartOrientation = cubeStartOrientation
    self._actionRepeat = actionRepeat
    self._isEnableSelfCollision = isEnableSelfCollision
    self._observation = []
    self._envStepCounter = 0
    self._renders = renders
    self._isDiscrete = isDiscrete
    self._recalculateSpot = recalculateSpot
    if self._renders:
      self._p = bc.BulletClient(connection_mode=p.GUI)
    else:
      self._p = bc.BulletClient()
    
    self.seed()
  
    #observationDim =  #the length of self.getExtendedObservation)
    observation_high = np.inf
    if (isDiscrete):
      self.action_space = spaces.Discrete(3)
    else:
      action_dim = 3
      self.action_bound = 15
      action_high = np.array([self.action_bound] * action_dim) # = array([15, 15, 15, 15])
      self.action_space = spaces.Box(-action_high, action_high, (action_dim,), dtype=np.float32)
    #shape=256*256*4 + 100 + 3 + 4
    shape = 128*128*4 + 100 + 3 + 4 + 3 + 3 + 2 #RGBA + lidar ray scans + pos + or(quatern) + lin vel + ang vel + x_spot, y_spot
    self.observation_space = spaces.Box(-observation_high, observation_high, (shape,), dtype=np.float32)
    self.viewer = None
    
  def reset(self):
    self._p.resetSimulation()
    self._p.setTimeStep(self._timeStep)
    
    x = random.uniform(-15, 15)
    y = random.uniform(-15, 15)
    z = 20
    
    self._cubeStartPos = [x,y,z]
    self._cubeStartOrientation = self._p.getQuaternionFromEuler([0,0,0])
    self._p.setGravity(0, 0, -1.62) #moon
    #self._p.setGravity(0, 0, -3.711) #mars
    #self._p.setGravity(0, 0, -1.352) #titan
    self._lander = lander.Lander(self._p, urdfRootPath=self._urdfRoot, timeStep=self._timeStep, cubeStartPos=self._cubeStartPos, cubeStartOrientation=self._cubeStartOrientation, renders=self._renders)
    
    #Initialize landing spot start position:
    self.x_spot = random.uniform(-15, 15)
    self.y_spot = random.uniform(-15, 15)
    self.z_spot = 0
    self.x_correction = random.uniform(-3, +3)
    self.y_correction = random.uniform(-3, +3)
    
    self._spotStartPos = [self.x_spot, self.y_spot, self.z_spot]
    self.spotObject = self._p.loadURDF("landing_spot.urdf", self._spotStartPos) #load landing spot
    
    self.envStepCounter = 0
    
    for i in range(100):
      self._p.stepSimulation()
    self._observation = self.getExtendedObservation()
    return np.array(self._observation)
    
  def __del__(self):
    self._p = 0
    
  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]
    
  def getExtendedObservation(self):
    self._observation = [] #self._lander.getObservation()
    cubepos, cubeorn = self._p.getBasePositionAndOrientation(self._lander.landerUniqueId)
    print('z cube pos = ', cubepos[2])
    lin_vel, ang_vel = self._lander.get_base_velocity()
    RGB_img, _, _ = self._lander.RGB_camera()
    lidar_results = self._lander.Lidar()
    #spotposition, _ = self._p.getBasePositionAndOrientation(self.spotObject)
    z_lander = cubepos[2]
    z_lander_rec = 10
    if (round(z_lander) < z_lander_rec):
      rec = True
    else: 
      rec = False
    if (self._recalculateSpot & rec):
      #self._p.removeBody(self.spotObject)
      self._spotStartPos = [self.x_spot+self.x_correction, self.y_spot+self.y_correction, self.z_spot]
    else: 
      #self._p.removeBody(self.spotObject)
      self._spotStartpos = [self.x_spot, self.y_spot, self.z_spot]
    self.spotObject = self._p.loadURDF("landing_spot.urdf", self._spotStartPos) #load landing spot
    spotpos = self._spotStartPos
    print('spotpos = ', spotpos[0:2])
    
    self._observation.extend(cubepos)
    self._observation.extend(cubeorn)
    self._observation.extend(lin_vel)
    self._observation.extend(ang_vel)
    self._observation.extend(np.ravel(RGB_img))
    self._observation.extend(np.ravel(lidar_results))
    #self._observation.extend([0] * 100)
    self._observation.extend(spotpos[0:2])
    
    #print("self._observation = ", self._observation)
    
    return self._observation
    
  def step(self, action):
    if (self._renders):
      basePos, baseOrn = self._p.getBasePositionAndOrientation(self._lander.landerUniqueId)
      lin_vel, ang_vel = self._lander.get_base_velocity()
      
    if (self._isDiscrete):
      thrust = [0, 0, 0, 15, 15, 15]
      thrusters = thrust[action]
      realaction = [thrusters]
    else:
      realaction = action
      
    self._lander.applyAction(realaction)
    
    for i in range(self._actionRepeat):
      self._p.stepSimulation()
      if self._renders:
        time.sleep(self._timeStep)
      self._observation = self.getExtendedObservation()
      
      if self._termination():
        break
      self._envStepCounter += 1
    reward = self._reward()
    done = self._termination()
    
    return np.array(self._observation), reward, done, {}
    
  def render(self, mode='human', close=False):
    if mode!= "rgb_array":
      return np.array([])
    basePos, baseOrn = self._p.getBasePositionAndOrientation(self._lander.landerUniqueId)
    lin_vel, ang_vel = self._lander.get_base_velocity()
    view_matrix = self._p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=basePos, distance=self._cam_dist, yaw=self._cam_yaw, pitch=self._cam_pitch, roll=0, upAxisIndex=2)
    proj_matrix = self._p.computeProjectionMatrixFOV(fov=60, aspect=float(RENDER_WIDTH)/RENDER_HEIGHT, nearVal=0.1, farVal=100)
    (_, _, px, _, _) = self.getCameraImage(width=RENDER_WIDTH, height=RENDER_HEIGHT, viewMatrix=view_matrix, projectionMatrix=proj_matrix, renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)
    
    rgb_array = np.array(px)
    rgb_array = rgb_array[:, :, :3]
    return rgb_array
    
  def _termination(self):
    contact = self._lander.detect_contact()
    #linear_velocity, angular_velocity = self._lander.get_base_velocity()
    stop = False
    #if ((contact == 1) and (np.linalg.norm(linear_velocity) <= 0.1) and (np.linalg.norm(angular_velocity) <= 0.1)):
    if (contact == 1):
      stop = True
    return stop
  
  #Reward function - can be customizable:    
  def _reward(self):
    total_links_contact = self._lander.detect_links_contact()
    #tot_thr = self._lander.actionApplied()
    if (total_links_contact == 0):
      reward_1 = -100
    elif (total_links_contact == 1):
      reward_1 = +50
    elif (total_links_contact == 2):
      reward_1 = +100
    elif (total_links_contact == 3):
      reward_1 = +200
    else:
      reward_1 = +300
    
    _, base_com_or = self._p.getBasePositionAndOrientation(self._lander.landerUniqueId) #is a quaternion
    base_or_euler = self._p.getEulerFromQuaternion(base_com_or) #roll pitch yaw
    roll = base_or_euler[0] * 180 / math.pi #roll in deg
    pitch = base_or_euler[1] * 180 / math.pi #pitch in deg
    #yaw = base_or_euler[2] * 180 / math.pi #yaw in deg
    if ((abs(roll) < 45) and (abs(pitch) < 45)):
      reward_2 = +15
    else:
      reward_2 = -15
    
   #TODO: Add here ad-hoc reward for velocities/pin-point landing/fuel consumption:
    lin_vel, ang_vel = self._lander.get_base_velocity() 
    z_vel = lin_vel[0]
    print('abs vel along z = ', abs(z_vel))
    if abs(z_vel) < 0.1: # & abs(z_vel > 0.01)):
      reward_3 = + 15
    elif abs(z_vel) < 0.01:
      reward_3 = + 30
    else: 
      reward_3 = - 15
    
    #print('lin vel = ', lin_vel)
    #print('ang vel = ', ang_vel)
     
    reward = reward_1 + reward_2 + reward_3
    print("reward: ", reward)
    #print("roll, pitch = ", roll, pitch)
    #print("tot_thr = ", tot_thr)
    return reward
    
    
  
