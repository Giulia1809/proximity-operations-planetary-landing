import math
import numpy as np
import random
#import os

class Lander:

  def __init__(self, bullet_client, cubeStartPos, cubeStartOrientation, renders, urdfRootPath='', timeStep=0.01):
    self.numRays = 100
    self.rayIds = [-1] * self.numRays
    
    self.cubeStartPos = cubeStartPos
    self.cubeStartOrientation = cubeStartOrientation
    self._renders = renders
    #self.urdfRootPath = urdfRootPath
    self.timeStep = timeStep
    self._p = bullet_client
    self._p.setAdditionalSearchPath(urdfRootPath)
    self.reset()
    
  def reset(self):
    self.terrainObject = self._p.loadURDF("terrain.urdf")
    #self.terrainObject2 = self._p.loadURDF("plane_background.urdf") #Uncomment for Titan env
    lander = self._p.loadURDF("lander_1.urdf", self.cubeStartPos, self.cubeStartOrientation)
    self.landerUniqueId = lander
    
    self.baseForce = 0.
    self.force1Diff = 0.
    self.force2Diff = 0.
    self.force3Diff = 0.
    self.force4Diff = 0.
    
    self.gustDiff = 0.
    
    self.force1 = [0., 0., self.force1Diff + self.baseForce]
    self.force2 = [0., 0., self.force2Diff + self.baseForce]
    self.force3 = [0., 0., self.force3Diff + self.baseForce]
    self.force4 = [0., 0., self.force4Diff + self.baseForce]
    
    self.gust1 = [self.gustDiff, 0., 0.]
    self.gust2 = [0., self.gustDiff, 0.]
    
    self._p.applyExternalForce(lander, -1, self.force1, [.25, 0, 0], flags=self._p.LINK_FRAME)
    self._p.applyExternalForce(lander, -1, self.force2, [0, .25, 0], flags=self._p.LINK_FRAME)
    self._p.applyExternalForce(lander, -1, self.force3, [-.25, 0, 0], flags=self._p.LINK_FRAME)
    self._p.applyExternalForce(lander, -1, self.force4, [0, -.25, 0], flags=self._p.LINK_FRAME)
    
    self.nThrusters = 4
    self.maxForce = 15
    self.minForce = 0
    self.forceIncrement = 2
    
    #Introduce gust disturbances acting normal to lander z-axis:
    
    self._p.applyExternalForce(lander, -1, self.gust1, [0, 0, .10], flags=self._p.LINK_FRAME)
    self._p.applyExternalForce(lander, -1, self.gust2, [0, 0, .28], flags=self._p.LINK_FRAME)
    
  def getActionDimension(self):
    return self.nThrusters
    
  def getObservationDimension(self):
    return len(self.getObservation())
    
  def RGB_camera(self):
    fov = 45.0
    aspect = 1.0
    nearplane = 0.01
    farplane = 100
    projectionMatrix = self._p.computeProjectionMatrixFOV(fov, aspect, nearplane, farplane)
    com_pos, com_or, _, _, _, _ = self._p.getLinkState(self.landerUniqueId, 5, computeForwardKinematics=True)
    base_com_pos, base_com_or = self._p.getBasePositionAndOrientation(self.landerUniqueId)
    #print ("com_or = ", com_or)
    cubeOr = self._p.getMatrixFromQuaternion(com_or)
    cubeOr = np.array(cubeOr).reshape(3, 3)
    baseOr = self._p.getMatrixFromQuaternion(base_com_or)
    baseOr = np.array(baseOr).reshape(3, 3)
    init_camera_vector = (0, 0, -0.2)
    init_up_vector = (0, 0, 1)
    #rotated vectors:
    camera_vector = cubeOr.dot(init_camera_vector)
    up_vector = cubeOr.dot(init_up_vector)
    to_vector = baseOr.dot([self.cubeStartPos[0], self.cubeStartPos[1], -self.cubeStartPos[2]])
    view_matrix = self._p.computeViewMatrix(com_pos, to_vector, up_vector)
    RGB_img = self._p.getCameraImage(256, 256, view_matrix, projectionMatrix)
    return RGB_img[2], camera_vector, com_pos
    
  def Lidar(self):
    com_pos, _, _, _, _, _ = self._p.getLinkState(self.landerUniqueId, 5, computeForwardKinematics=True)
    com_pos_init = [com_pos[0] - 0.25, com_pos[1] - 0.25, com_pos[2] - 0.25]
    
    rayLen = 13
    rayHitColor = [0, 1, 0]
    rayMissColor = [1, 0, 0]
    replaceLines = True
    
    rayFrom = []
    rayTo = []
    
    for i in range(self.numRays):   
      rayFrom.append(com_pos_init)
      rayTo.append([0.25*rayLen * math.sin (2. * math.pi * float(i) / self.numRays) + self.cubeStartPos[0],
                    0.25*rayLen * math.cos (2. * math.pi * float(i) / self.numRays) + self.cubeStartPos[1],
                    0])
    lidar_results = self._p.rayTestBatch(rayFrom, rayTo)
    if self._renders:
      for i in range(self.numRays):
        hitObjectUid = lidar_results[i][0]
        if (hitObjectUid < 0):
          hitPosition = rayTo[i]
          rayColor = rayMissColor
        else:
          hitPosition = lidar_results[i][3]
          rayColor = rayHitColor
      self.rayIds[i] = self._p.addUserDebugLine(rayFrom[i], hitPosition, rayColor, replaceItemUniqueId=self.rayIds[i])

    hit_fraction_elems = [elem[2] for elem in lidar_results]
    return hit_fraction_elems
    
  def detect_contact(self):
    #lander = self._p.loadURDF("lander_1.urdf", self.cubeStartPos, self.cubeStartOrientation)
    #self.landerUniqueId = lander
    self.contact_color = [1, 0, 0, 1]
    if self._p.getContactPoints(self.landerUniqueId):
      contact = 1
      self._p.changeVisualShape(self.landerUniqueId, -1, rgbaColor=self.contact_color)
      for i in range(6):
        self._p.changeVisualShape(self.landerUniqueId, i, rgbaColor=self.contact_color)
    else:
      contact = 0
    return contact
    
  def detect_links_contact(self):
    if self._p.getContactPoints(self.landerUniqueId, self.terrainObject, 1):
      link_contact_1 = 1
      self._p.changeVisualShape(self.landerUniqueId, 1, rgbaColor=[1, 0, 1, 1])
    else:
      link_contact_1 = 0
    if self._p.getContactPoints(self.landerUniqueId, self.terrainObject, 2):
      link_contact_2 = 1
      self._p.changeVisualShape(self.landerUniqueId, 2, rgbaColor=[1, 0, 1, 1])
    else:
      link_contact_2 = 0
    if self._p.getContactPoints(self.landerUniqueId, self.terrainObject, 3):
      link_contact_3 = 1
      self._p.changeVisualShape(self.landerUniqueId, 3, rgbaColor=[1, 0, 1, 1])
    else:
      link_contact_3 = 0
    if self._p.getContactPoints(self.landerUniqueId, self.terrainObject, 4):
      link_contact_4 = 1
      self._p.changeVisualShape(self.landerUniqueId, 4, rgbaColor=[1, 0, 1, 1])
    else:
      link_contact_4 = 0
    total_links_contact = link_contact_1 + link_contact_2 + link_contact_3 + link_contact_4      
    return total_links_contact

  def get_base_velocity(self):
    linear_velocity, angular_velocity = self._p.getBaseVelocity(self.landerUniqueId)
    linear_velocity = np.asarray(linear_velocity)
    angular_velocity = np.asarray(angular_velocity)
    return linear_velocity, angular_velocity
    
  def getObservation(self):
    observation = []
    pos, orn = self._p.getBasePositionAndOrientation(self.landerUniqueId)
    RGB_img, _, _, = RGB_camera()
    lidar_results = Lidar()
    
    observation.extend(list(pos))
    observation.extend(list(orn))
    observation.extend(list(RBG_img))
    observation.extend(list(lidar_results))
    
    return observation
    
  def applyAction(self, thrusterCommands):
  
    print("thrusterCommands:", thrusterCommands)
    #self.landerUniqueId = lander
  
    #self.baseForce = 0.
    #self.force1Diff = 0.
    #self.force2Diff = 0.
    #self.force3Diff = 0.
    #self.force4Diff = 0.
    
    self.force1Diff = thrusterCommands[0] * self.forceIncrement
    self.force2Diff = thrusterCommands[1] * self.forceIncrement
    self.force3Diff = thrusterCommands[2] * self.forceIncrement
    self.force4Diff = thrusterCommands[3] * self.forceIncrement
    
    self.force1 = [0., 0., self.force1Diff + self.baseForce]
    self.force2 = [0., 0., self.force2Diff + self.baseForce]
    self.force3 = [0., 0., self.force3Diff + self.baseForce]
    self.force4 = [0., 0., self.force4Diff + self.baseForce]
    
    self.gustDiff = random.uniform(0, 2)
    self.gust1 = [self.gustDiff*random.uniform(0, 1), 0., 0.]
    self.gust2 = [0., self.gustDiff*random.uniform(0, 1), 0.]
   
    self._p.applyExternalForce(self.landerUniqueId, -1, self.force1, [.25, 0, 0], flags=self._p.LINK_FRAME)
    self._p.applyExternalForce(self.landerUniqueId, -1, self.force2, [0, .25, 0], flags=self._p.LINK_FRAME)
    self._p.applyExternalForce(self.landerUniqueId, -1, self.force3, [-.25, 0, 0], flags=self._p.LINK_FRAME)
    self._p.applyExternalForce(self.landerUniqueId, -1, self.force4, [0, -.25, 0], flags=self._p.LINK_FRAME)
    
    #Introduce gust disturbances acting normal to lander z-axis:
    
    self._p.applyExternalForce(self.landerUniqueId, -1, self.gust1, [0, 0, .10], flags=self._p.LINK_FRAME)
    self._p.applyExternalForce(self.landerUniqueId, -1, self.gust2, [0, 0, .28], flags=self._p.LINK_FRAME)
    
    print('gust1 value: ', self.gust1)
    print('gust2 value: ', self.gust2) 
    
