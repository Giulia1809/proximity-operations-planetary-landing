import pybullet as p
import time
import pybullet_data
import random
import numpy as np
import math

useGui = True

if useGui:
  physicsClient = p.connect(p.GUI) #or p.DIRECT for non-graphical version
else:
  physicsClient = p.connect(p.DIRECT)
  
#p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, "lunar_lander.mp4")
 
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0,0,-1.62) #/1000) #gravity set along -z
planeId = p.loadURDF("terrain.urdf") #load terrain
#planeId_2 = p.loadURDF("plane_background.urdf")

#Initialize landing spot start position:
x_spot = random.uniform(-10, 10)
y_spot = random.uniform(-10, 10)
z_spot = 0
spotStartPos = [x_spot, y_spot, z_spot]
spotId = p.loadURDF("landing_spot.urdf", spotStartPos) #load landing spot

x = random.uniform(-10, 10)
y = random.uniform(-10, 10)
z = 20
cubeStartPos = [x,y,z]
cubeStartOrientation = p.getQuaternionFromEuler([0,0,0])
boxId = p.loadURDF("lander_1.urdf",cubeStartPos, cubeStartOrientation) #load rover and its starting pose

contact_color = [1., 0., 0., 1.]

fov, aspect, nearplane, farplane = 45, 1.0, 0.01, 100
projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, nearplane, farplane)

def RGB_camera():
  com_pos, com_or, _, _, _, _ = p.getLinkState(boxId, 5, computeForwardKinematics=True)
  base_com_pos, base_com_or = p.getBasePositionAndOrientation(boxId)
  #print ("com_or = ", com_or)
  #com_or_vec = [com_or[0], com_or[1], com_or[2]]
  cubeOr = p.getMatrixFromQuaternion(com_or)
  cubeOr = np.array(cubeOr).reshape(3, 3)
  baseOr = p.getMatrixFromQuaternion(base_com_or)
  baseOr = np.array(baseOr).reshape(3, 3)
  #axis_angle_from_quat = p.getAxisAngleFromQuaternion(com_or)
  #orientation_vec = axis_angle_from_quat[0]
  #print("orientation_com_vector =", orientation_vec)
  #print("CubeOr = ", cubeOr)
  init_camera_vector = (0, 0, -0.2)
  #init_camera_vector = (0, 0, -1.3)
  init_up_vector = (0, 0, 1)
  #rotated vectors:
  camera_vector = cubeOr.dot(init_camera_vector)
  up_vector = cubeOr.dot(init_up_vector)
  to_vector = baseOr.dot([x, y, -z])
  #to_vector = [com_pos[0], com_pos[1], com_pos[2] - 2]
  #view_matrix = p.computeViewMatrix(com_pos, com_pos+0.1*camera_vector, up_vector)
  view_matrix = p.computeViewMatrix(com_pos, to_vector, up_vector)
  RGB_img = p.getCameraImage(256, 256, view_matrix, projection_matrix)
  return RGB_img, camera_vector, com_pos
  #return camera_vector, com_pos
  
#p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
#p.configureDebugViualizer(p.COV_ENABLE_RENDERING, 0)

#def Lidar():
numRays = 100
rayIds = [-1] * numRays
rayLen = 13
rayHitColor = [0, 1, 0]
rayMissColor = [1, 0, 0]
replaceLines = True
    
if (not useGui): 
  timingLog = p.startStateLogging(p.STATE_LOGGING_PROFILE_TIMINGS, "rayCastBench.json")

baseForce = 0.
force1Diff = 0.
force2Diff = 0.
force3Diff = 0.
#force4Diff = 0.
    
while True:
  rayFrom = []
  rayTo = []
  
  #Trigger keyboard events: to control thrusters
  keys = p.getKeyboardEvents()
  for k,v in keys.items():
    if k == p.B3G_UP_ARROW and (v & p.KEY_WAS_TRIGGERED):
      force1Diff = 5/1000
    if k == p.B3G_UP_ARROW and (v & p.KEY_WAS_RELEASED):
      force1Diff = 0.
    if k == p.B3G_LEFT_ARROW and (v & p.KEY_WAS_TRIGGERED):
      force2Diff = 5/1000
    if k == p.B3G_LEFT_ARROW and (v & p.KEY_WAS_RELEASED):
      force2Diff = 0.
    if k == p.B3G_RIGHT_ARROW and (v & p.KEY_WAS_TRIGGERED):
      force3Diff = 5/1000
    if k == p.B3G_RIGHT_ARROW and (v & p.KEY_WAS_RELEASED):
      force3Diff = 0.
          
    if k == ord('n') and (v & p.KEY_IS_DOWN):
      baseForce = max(0, baseForce - 0.1)
      print("BaseForce = {}".format(baseForce))
    if k == ord('m') and (v & p.KEY_IS_DOWN):
      baseForce = min(50, baseForce + 0.1)
      print("Baseforce = {}".format(baseForce))
  
  def thruster_forces(baseForce, force1Diff, force2Diff, force3Diff):    
    force1 = [0., 0., force1Diff + baseForce] #up arrow / z
    force2 = [0., force2Diff + baseForce, 0.] #left arrow / y
    force3 = [force3Diff + baseForce, 0., 0.] #right arrow / x
  
    p.applyExternalForce(boxId, -1, force1, [0, 0, 0], flags=p.LINK_FRAME)
    p.applyExternalForce(boxId, -1, force2, [0, 0, 0], flags=p.LINK_FRAME)
    p.applyExternalForce(boxId, -1, force3, [0, 0, 0], flags=p.LINK_FRAME)
    return force1, force2, force3 
  force1, force2, force3 = thruster_forces(baseForce, force1Diff, force2Diff, force3Diff)
    
  def detect_contact():
    if p.getContactPoints(boxId):
      contact = 1
      p.changeVisualShape(boxId, -1, rgbaColor=contact_color)
      for i in range(6):
        p.changeVisualShape(boxId, i, rgbaColor=contact_color)
    else:
      contact = 0
    return contact
    
  contact = detect_contact()
  print("contact = ", contact)
  
  def detect_link_1_contact():
    #print("getcontactpoint 1= ", p.getContactPoints(1))
    #print("getcontactpoint 2= ", p.getContactPoints(2))
    if p.getContactPoints(boxId, planeId, 1):
      link_contact_1 = 1
      p.changeVisualShape(boxId, 1, rgbaColor=[1, 0, 1, 1])
      #print("link 1 if")
    else:
      link_contact_1 = 0
      #print("link 1 else")
    return link_contact_1
  link_contact_1 = detect_link_1_contact()
  
  def detect_link_2_contact():
    if p.getContactPoints(boxId, planeId, 2):
      link_contact_2 = 1
      p.changeVisualShape(boxId, 2, rgbaColor=[1, 0, 1, 1])
      #print("link 2 if")
    else:
      link_contact_2 = 0
      #print("link 2 else")
    return link_contact_2
  link_contact_2 = detect_link_2_contact()
  #print("link_2_contact = ", link_contact_2)
  
  def detect_link_3_contact():
    if p.getContactPoints(boxId, planeId, 3):
      link_contact_3 = 1
      p.changeVisualShape(boxId, 3, rgbaColor=[1, 0, 1, 1])
      #print("link 3 if")
    else:
      link_contact_3 = 0
      #print("link 3 else")
    return link_contact_3
  link_contact_3 = detect_link_3_contact()
  #print("link_3_contact = ", link_contact_3)
  
  def detect_link_4_contact():
    if p.getContactPoints(boxId, planeId, 4):
      link_contact_4 = 1
      p.changeVisualShape(boxId, 4, rgbaColor=[1, 0, 1, 1])
      #print("link 4 if")
    else:
      link_contact_4 = 0
      #print("link 4 else")
    return link_contact_4
  link_contact_4 = detect_link_4_contact()
  #print("link_4_contact = ", link_contact_4)
  
  def return_total_links_contact(link_contact_1, link_contact_2, link_contact_3, link_contact_4):
    links_contact = link_contact_1 + link_contact_2 + link_contact_3 + link_contact_4
    return links_contact
  links_contact = return_total_links_contact(link_contact_1, link_contact_2, link_contact_3, link_contact_4)
  print("links_contact = ", links_contact)
  
  p.stepSimulation()
 # p.setRealTimeSimulation(0)
  
  #RGB_camera()
  _, _, com_pos = RGB_camera()
  #print("com_pos = ", com_pos)
  
  # When not using camera:
  #com_pos, com_or, _, _, _, _ = p.getLinkState(boxId, 5, computeForwardKinematics=True)
  
  com_pos_init = [com_pos[0] - 0.25, com_pos[1] - 0.25, com_pos[2] - 0.25]
  #com_pos_init = [com_pos[0], com_pos[1], com_pos[2] - 0.25]
  
  for i in range(numRays):   
    rayFrom.append(com_pos_init)
    #rayTo.append([0.25*rayLen * math.sin(2. * math.pi * float(i) / numRays) + x, 0.25*rayLen * math.cos(2. * math.pi * float(i) / numRays) + y, 1])
    rayTo.append([0.25*rayLen * math.sin (2. * math.pi * float(i) / numRays) + x, 0.25*rayLen * math.cos (2. * math.pi * float(i) / numRays) + y, 0])

  #results = p.rayTestBatch(rayFrom, rayTo)#, i+1)
  
  def return_lidar_results():
    results = p.rayTestBatch(rayFrom, rayTo)
    return results
  results = return_lidar_results()
  #print("lidar_results = ", lidar_results)
  
  def get_base_position_and_orientation():
    basePos, baseOrn = p.getBasePositionAndOrientation(boxId)
    return basePos, baseOrn
  basePos, baseOrn = get_base_position_and_orientation()
  
  def get_euler_fromorientation():
    basePos, baseOrn = p.getBasePositionAndOrientation(boxId)
    euler = p.getEulerFromQuaternion(baseOrn)
    return euler
  euler = get_euler_fromorientation()
  print("euler = ", euler)
  
  def get_base_velocity():
    linear_velocity, angular_velocity = p.getBaseVelocity(boxId)
    return linear_velocity, angular_velocity
  linear_velocity, angular_velocity = get_base_velocity()
  linear_velocity = np.asarray(linear_velocity)
  angular_velocity = np.asarray(angular_velocity)
  #print("linear_velocity = ", linear_velocity, "abs_lin_vel = ", abs(linear_velocity), "angular_velocity = ", angular_velocity)
  
  if (useGui):
         
    for i in range(numRays):
      hitObjectUid = results[i][0]
      if (hitObjectUid < 0):
        hitPosition = rayTo[i]
        rayColor = rayMissColor
      else:
        hitPosition = results[i][3]
        rayColor = rayHitColor
      rayIds[i] = p.addUserDebugLine(rayFrom[i], hitPosition, rayColor, replaceItemUniqueId=rayIds[i])
      
        
  if (not useGui):
    p.stopStateLogging(timingLog)
  #print("lidar results = ", results)
  
#p.disconnect()

