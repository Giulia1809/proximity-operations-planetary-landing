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
import lander_with_gusts

RENDER_HEIGHT = 720
RENDER_WIDTH = 960

class LanderGymEnv(gym.Env):
  metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second':50}
  
  def __init__(self, urdfRoot = pybullet_data.getDataPath(), actionRepeat=50, isEnableSelfCollision=True, isDiscrete=False, renders=True):
  
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
    if self._renders:
      self._p = bc.BulletClient(connection_mode=p.GUI)
    else:
      self._p = bc.BulletClient()
    
    self.seed()
  
    #observationDim =  #the length of self.getExtendedObservation)
    observation_high = np.inf
    if (isDiscrete):
      self.action_space = spaces.Discrete(4)
    else:
      action_dim = 4
      self.action_bound = 15
      action_high = np.array([self.action_bound] * action_dim) # = array([15, 15, 15, 15])
      self.action_space = spaces.Box(0, action_high, (4,), dtype=np.float32)
    shape=256*256*4 + 100 + 3 + 4
    self.observation_space = spaces.Box(-observation_high, observation_high, (shape,), dtype=np.float32)
    self.viewer = None
    
  def reset(self):
    self._p.resetSimulation()
    self._p.setTimeStep(self._timeStep)
    
    #terrainObject = self._p.loadURDF("terrain.urdf")
    
    x = random.uniform(-10, 10)
    y = random.uniform(-10, 10)
    z = 10
    self._cubeStartPos = [x,y,z]
    self._cubeStartOrientation = self._p.getQuaternionFromEuler([0,0,0])
    #self._p.setGravity(0, 0, -1.62) #moon
    #self._p.setGravity(0, 0, -3.711) #mars
    self._p.setGravity(0, 0, -1.352) #titan
    self._lander = lander_with_gusts.Lander(self._p, urdfRootPath=self._urdfRoot, timeStep=self._timeStep, cubeStartPos=self._cubeStartPos, cubeStartOrientation=self._cubeStartOrientation, renders=self._renders)
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
    self._observation = []
    cubepos, cubeorn = self._p.getBasePositionAndOrientation(self._lander.landerUniqueId)
    RGB_img, _, _ = self._lander.RGB_camera()
    lidar_results = self._lander.Lidar()
    
    self._observation.extend(cubepos)
    self._observation.extend(cubeorn)
    self._observation.extend(np.ravel(RGB_img))
    self._observation.extend(np.ravel(lidar_results))
    #self._observation.extend([0] * 100)
    
    #print("self._observation = ", self._observation)
    
    return self._observation
    
  def step(self, action):
    if (self._renders):
      basePos, baseOrn = self._p.getBasePositionAndOrientation(self._lander.landerUniqueId)
      
    if (self._isDiscrete):
      thrust = [0, 0, 0, 0, 15, 15, 15, 15]
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
    #return contact=1, abs(linear_velocity) <= [0.01, 0.01, 0.01], abs(angular_velocity) <= [0.01, 0.01, 0.01]
  
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
      
    reward = reward_1 + reward_2
    print("reward: ", reward)
    #print("roll, pitch = ", roll, pitch)
    #print("tot_thr = ", tot_thr)
    return reward
    
    
  
