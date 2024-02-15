#Run this script to view Lander in the chosen Environment - verify cameras. 
#This script includes sensors: RGB camera, Depth camera, Segmentation Mask (+ inputs from position and velocity)

import pybullet as p
import time
import pybullet_data
import random
import numpy as np
import math

physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0,0,-1.62) #lunar gravity set along -z
planeId = p.loadURDF("terrain.urdf") #load terrain
#planeId_2 = p.loadURDF("plane_background.urdf")

x = random.uniform(-20, 20)
y = random.uniform(-20, 20)
z = 20
cubeStartPos = [x,y,z]
cubeStartOrientation = p.getQuaternionFromEuler([0,0,0])
boxId = p.loadURDF("lander_1.urdf",cubeStartPos, cubeStartOrientation) #load rover and its starting pose

fov, aspect, nearplane, farplane = 45, 1.0, 0.01, 100
projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, nearplane, farplane)
def RGB_camera():
  com_pos, com_or, _, _, _, _ = p.getLinkState(boxId, 5, computeForwardKinematics=True)
  cubeOr = p.getMatrixFromQuaternion(com_or)
  cubeOr = np.array(cubeOr).reshape(3, 3)
  init_camera_vector = (0, 0, -0.2)
  init_up_vector = (0, 0, 1)
  #rotated vectors:
  camera_vector = cubeOr.dot(init_camera_vector)
  up_vector = cubeOr.dot(init_up_vector)
  view_matrix = p.computeViewMatrix(com_pos, com_pos+0.1*camera_vector, up_vector)
  RGB_img = p.getCameraImage(256, 256, view_matrix, projection_matrix)
  return RGB_img, com_pos
  
while True:
  
  p.stepSimulation()
   
  #RGB_camera()
  _, com_pos = RGB_camera()
  #print("com_pos = ", com_pos)
  
#p.disconnect()

