#!/usr/bin/env python
# coding: utf-8

# Import urdf to pybullet and check it look ok

import pybullet as p
import pybullet_data
import os
import numpy as np
import time
physicsClient = p.connect(p.GUI)



p.resetSimulation() # remove all objects from the world and reset the world to initial conditions. (not needed here but kept for example)
p.configureDebugVisualizer(p.COV_ENABLE_GUI,0,physicsClientId=physicsClient)

p.setGravity(0,0,-9.81)
planeId = p.loadURDF(os.path.join(pybullet_data.getDataPath(),
        "plane100.urdf"))
startPosition=[0,0,0.3] # high enough that nothing touches the ground
startPosition=[0,0,1] # high enough that nothing touches the ground
startOrientationRPY = [0,0,0]
startOrientation = p.getQuaternionFromEuler(startOrientationRPY)   

robotID = p.loadURDF(
             'llllll.urdf', # 'crab_urdf/crab_model.urdf', #
                   basePosition=startPosition, baseOrientation=startOrientation,
                   flags= (p.URDF_MAINTAIN_LINK_ORDER 
                    | p.URDF_USE_SELF_COLLISION
                    | p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES))

# count all joints, including fixed ones
num_joints_total = p.getNumJoints(robotID,
                physicsClientId=physicsClient)

box_id = p.createCollisionShape(
    p.GEOM_BOX,
    halfExtents=[.05, .05, 0.1])
visual_id = p.createVisualShape(
    p.GEOM_BOX,
    rgbaColor = [0, 1, 0, 1],
    halfExtents=[.05, .05, 0.1])
block_ID = p.createMultiBody(
    baseMass=0,
    baseCollisionShapeIndex=box_id,
    baseVisualShapeIndex = visual_id,
    basePosition=[0.1, 0., 0.])
# do not need to compute collisions between these
p.setCollisionFilterPair(planeId, block_ID, -1, -1, 0)


# Each body is part of a group. It collides with other bodies if their group 
# matches the mask, and vise versa.
collisionFilterGroup = 1
collisionFilterMask = 1
for i in range(-1,num_joints_total):
    p.setCollisionFilterGroupMask(robotID, i,collisionFilterGroup,collisionFilterMask)


rayFromPositions = [[0,0,3], [0.1,0,3], [0,0.1,3]]
rayToPositions = [[0,0,-3], [0.1,0,-3], [0,0.1,-3]]
out = p.rayTestBatch(rayFromPositions=rayFromPositions,
    rayToPositions=rayToPositions,
    collisionFilterMask=0) 
    # collisionFilterMask=2) 
# only test hits if the bitwise and between
# collisionFilterMask and body collision filter group is
# non-zero. See setCollisionFilterGroupMask on how
# to modify the body filter mask/group.
print('Ray hit:')
print([out[i][3][-1] for i in range(len(rayFromPositions))])
print([out[i][2] for i in range(len(rayFromPositions))])


moving_joint_names = []
moving_joint_inds = []
moving_joint_types = []
moving_joint_limits = []
moving_joint_centers = []
for j_ind in range(num_joints_total):
    j_info = p.getJointInfo(robotID, 
        j_ind)
    if j_info[2] != (p.JOINT_FIXED):
        moving_joint_inds.append(j_ind)
        moving_joint_names.append(j_info[1])
        moving_joint_types.append(j_info[2])
        j_limits = [j_info[8], j_info[9]]
        j_center = (j_info[8] + j_info[9])/2
        if j_limits[1]<=j_limits[0]:
            j_limits = [-np.inf, np.inf]
            j_center = 0
        moving_joint_limits.append(j_limits)
        moving_joint_centers.append(j_center)

num_joints = len(moving_joint_names)
for i in range(num_joints):
    center = moving_joint_centers[i]
    jind = moving_joint_inds[i]
    p.resetJointState( bodyUniqueId=robotID, 
        jointIndex = jind,
        targetValue= center, 
        targetVelocity = 0 )
    
for j in moving_joint_inds:
#     p.setJointMotorControl2(robotID, j, p.VELOCITY_CONTROL, force=0)
    p.enableJointForceTorqueSensor(robotID, j, 1)
    
# steps to take with no control input before starting up
# This drops to robot down to a physically possible resting starting position
for i in range(300):
    # p.setJointMotorControlArray(robotID, moving_joint_inds,
    #                           p.TORQUE_CONTROL,
    #                            forces = -10)
#     p.setJointMotorControlArray(robotID, moving_joint_inds,
#                               p.POSITION_CONTROL,
#                               moving_joint_centers)   

    p.setJointMotorControlArray(
            bodyUniqueId=robotID, 
            jointIndices=moving_joint_inds, 
            controlMode=p.VELOCITY_CONTROL,
            targetVelocities = -5*np.ones(num_joints)*np.sin(i/10))
            # targetVelocities = -2*np.ones(num_joints))

    p.stepSimulation()    
    joint_states = p.getJointStates(robotID,
            moving_joint_inds,
            physicsClientId=physicsClient)



    out = p.rayTestBatch(rayFromPositions=rayFromPositions,
        rayToPositions=rayToPositions,
        collisionFilterMask=2) 
    # only test hits if the bitwise and between
    # collisionFilterMask and body collision filter group is
    # non-zero. See setCollisionFilterGroupMask on how
    # to modify the body filter mask/group.
    print('Ray hit:')
    print([out[i][3][-1] for i in range(len(rayFromPositions))])
    # print([out[i][2] for i in range(len(rayFromPositions))])



    time.sleep(2./240)



while True:
    p.stepSimulation()    
    time.sleep(2./240)




