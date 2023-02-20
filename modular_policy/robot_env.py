'''
Source code for paper "Learning modular robot control policies" in Transactions on Robotics
Julian Whitman, Dec. 2022. 

Simulation environment for the modular robots and helper functions

'''
import numpy as np
import torch
import pybullet #as p
import pybullet_data
import pybullet_utils.bullet_client as bc
# used to support multiple instances, see
# https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/gym/pybullet_utils/examples/multipleScenes.py#L13
import os
import time

pi = np.pi

# utilities
cwd = os.path.dirname(os.path.realpath(__file__))
urdf_dir = os.path.join(os.path.dirname(cwd), 'urdf')

# initialization
class robot_env:
    def __init__(self, 
        show_GUI = False):
        self.show_GUI = show_GUI
        if show_GUI:
            # self.physicsClient = p.connect(p.GUI)# p.DIRECT for non-graphical version
            p = bc.BulletClient(connection_mode=pybullet.GUI)
            self.physicsClient = p._client


        else:
            # initialize physics engine when environment is made    
            # self.physicsClient = p.connect(p.DIRECT)# p.DIRECT for non-graphical version
            p = bc.BulletClient(connection_mode=pybullet.DIRECT)
            self.physicsClient = p._client


        
        p.resetDebugVisualizerCamera(2,0,-25,[0,0,0],physicsClientId=self.physicsClient) # I like this view
        p.configureDebugVisualizer(p.COV_ENABLE_GUI,0,physicsClientId=self.physicsClient)
        # turn off shadows
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS,0,physicsClientId=self.physicsClient)
        
        self.time_step = 1./240. # default 1/240 ~= 0.004
        self.n_time_steps_per_step = 20 # 20/240 = 0.08
        self.dt = self.time_step*self.n_time_steps_per_step
        self.robotID = None # reserved for the pybullet object id
        self.sim_speed_factor = 1
        self.p = p
        self.loaded_urdf = None
        self.follow_with_camera = False # follow the robot with camera

        self.foot_friction = 1.2
        self.wheel_friction = [0.8,0.8,1.6]

        # self.foot_friction = 1.0
        # self.wheel_friction = [0.8,0.8,1.4]
        self.plane_friction = 0.8

    # reset terrain
    def reset_terrain(self, plane_transparent=False):
        p = self.p
        p.resetSimulation(self.physicsClient) # remove all objects from the world and reset the world to initial conditions. (not needed here but kept for example)
        p.setGravity(0,0,-9.81,physicsClientId=self.physicsClient)
        if plane_transparent:
            self.planeId = p.loadURDF(
                os.path.join(urdf_dir, 'plane_white.urdf'),
                physicsClientId=self.physicsClient)
            p.changeVisualShape(self.planeId, -1, rgbaColor=[1, 1, 1, 0])

        else:
            self.planeId = p.loadURDF(os.path.join(pybullet_data.getDataPath(),
                "plane100.urdf"),
                physicsClientId=self.physicsClient)

        if self.plane_friction is not None:  
            p.changeDynamics(bodyUniqueId=self.planeId, 
                        linkIndex=-1, lateralFriction=self.plane_friction, 
                        physicsClientId=self.physicsClient)

        p.setTimeStep(self.time_step, physicsClientId=self.physicsClient)
        self.loaded_urdf = None
        self.robotID = None
        
        # dyn_info = p.getDynamicsInfo(bodyUniqueId=self.planeId,linkIndex=-1,
        #             physicsClientId=self.physicsClient)
        # print('Plane friction: ' + str(dyn_info[1]))

    def remove_robot(self):
        p = self.p
        if self.robotID is not None:    
            p.removeBody(self.robotID, physicsClientId=self.physicsClient)
        self.loaded_urdf = None

    def reset_debug_items(self):
        p = self.p
        if self.show_GUI:
            # clear debug lines etc
            p.removeAllUserDebugItems(physicsClientId=self.physicsClient)

        #     p.addUserDebugLine(lineFromXYZ=[0,0,0.01],
        #         lineToXYZ=[0.25,0,0.01], lineColorRGB=[1,0,0],physicsClientId=self.physicsClient)
        #     p.addUserDebugLine(lineFromXYZ=[0,0,0.01],
        #         lineToXYZ=[0,0.25,0.01], lineColorRGB=[0,1,0],)
        #     p.addUserDebugLine(lineFromXYZ=[0,0,0],
        #         lineToXYZ=[0,0,0.25], lineColorRGB=[0,0,1],physicsClientId=self.physicsClient)

        self.arrow_ids = []
        self.arrow_data = None


    # reset robot
    def reset_robot(self, randomize_start=True,
        randomize_xyyaw = False,
        urdf_name = 'wnwwnw', start_xyyaw = None):
        p = self.p

        self.reset_debug_items()

        startPosition=[0,0,0.3] # high enough that nothing touches the ground
        startOrientationRPY = [0,0,0]
        if randomize_xyyaw:
            startPosition[0]+= (np.random.rand()*2-1)*0.05
            startPosition[1]+= (np.random.rand()*2-1)*0.05
            startOrientationRPY[2] += (np.random.rand()*2-1)*pi/8

        # allow for starting yaw to be overridden by user
        if start_xyyaw is not None: 
            startPosition[0] = start_xyyaw[0]
            startPosition[1] = start_xyyaw[1]
            startOrientationRPY[2] = start_xyyaw[2]

        startOrientation = p.getQuaternionFromEuler(startOrientationRPY)   

        if self.robotID is not None and (urdf_name == self.loaded_urdf):
        # reload same robot but need to reset its pose
            p.resetBaseVelocity(
                self.robotID,
                linearVelocity=[0,0,0],
                angularVelocity=[0,0,0],
                physicsClientId=self.physicsClient
                )
            p.resetBasePositionAndOrientation(
                self.robotID,
                posObj=startPosition,
                ornObj=startOrientation,
                physicsClientId=self.physicsClient
                )

            for i in range(self.num_joints):
                jind = self.moving_joint_inds[i]
                p.resetJointState( bodyUniqueId=self.robotID, 
                    jointIndex = jind,
                    targetValue= self.moving_joint_centers[i], 
                    targetVelocity = 0,
                    physicsClientId=self.physicsClient )


        else:
            # remove robot if needed
            if self.loaded_urdf is not None:
                self.remove_robot()
            self.loaded_urdf = urdf_name

            self.robotID = p.loadURDF(
                        os.path.join(urdf_dir, urdf_name + '.urdf'),
                           basePosition=startPosition, baseOrientation=startOrientation,
                           flags= (p.URDF_MAINTAIN_LINK_ORDER 
                            | p.URDF_USE_SELF_COLLISION
                            | p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES),
                           physicsClientId=self.physicsClient)
              # 

            # count all joints, including fixed ones
            num_joints_total = p.getNumJoints(self.robotID,
                            physicsClientId=self.physicsClient)

            # segment out into moving and fixed joints
            self.moving_joint_inds = []
            self.joint_names = []
            self.moving_joint_limits = []
            self.link_names = []
            self.moving_joint_names = []
            self.moving_joint_types = []
            self.moving_joint_limits = []
            self.moving_joint_centers = []
            self.moving_joint_max_torques = []
            self.moving_joint_max_velocities = []
            self.foot_wheel_link_inds = []

            for j_ind in range(num_joints_total):
                j_info = p.getJointInfo(self.robotID, 
                    j_ind, physicsClientId=self.physicsClient)
                self.joint_names.append(j_info[1])
                if j_info[2] != (p.JOINT_FIXED):
                    self.moving_joint_inds.append(j_ind)
                    j_limits = [j_info[8], j_info[9]]
                    j_center = (j_info[8] + j_info[9])/2
                    if j_limits[1]<=j_limits[0]:
                        j_limits = [-np.inf, np.inf]
                        j_center = 0
                    self.moving_joint_limits.append(j_limits)
                    self.moving_joint_centers.append(j_center)
                    self.moving_joint_names.append(j_info[1])
                    self.moving_joint_max_torques.append(j_info[10])
                    self.moving_joint_max_velocities.append(j_info[11])
                    self.moving_joint_types.append(j_info[2])
                link_name = str(j_info[12])
                self.link_names.append(link_name)


                # raise friction value for feet
                if link_name.find('foot/INPUT')>=0:
                    p.changeDynamics(bodyUniqueId=self.robotID, 
                                    linkIndex=j_ind, lateralFriction=self.foot_friction, 
                                    physicsClientId=self.physicsClient)
                    self.foot_wheel_link_inds.append(j_ind)
                # raise friction value for wheel
                elif link_name.find('wheel/INPUT')>=0:
                    p.changeDynamics(bodyUniqueId=self.robotID, 
                                    linkIndex=j_ind, lateralFriction=1.0, 
                                    anisotropicFriction=self.wheel_friction,
                                    physicsClientId=self.physicsClient)
                    self.foot_wheel_link_inds.append(j_ind)
                # raise friction for chassis, in case wheels or ground rub into it
                elif link_name.find('chassis')>=0:
                    p.changeDynamics(bodyUniqueId=self.robotID, 
                                    linkIndex=j_ind, lateralFriction=1.2, 
                                    physicsClientId=self.physicsClient)

            # throttle torque max and vel
            # this is needed since the "max" vel listed is the unloaded max,
            # and "max" torque is the instantaneous and not continuous max
            # self.moving_joint_max_torques    = np.array(self.moving_joint_max_torques)*0.6
            # self.moving_joint_max_velocities = np.array(self.moving_joint_max_velocities)*0.6
            self.moving_joint_max_torques    = np.array(self.moving_joint_max_torques)*0.8
            self.moving_joint_max_velocities = np.array(self.moving_joint_max_velocities)


            self.num_joints = len(self.moving_joint_inds)
            self.num_links = len(self.link_names) # does not include the root link
            
            self.total_mass = 0
            self.link_masses = []
            for l in range(-1, self.num_links): # -1 gets the root
                dyn_info = p.getDynamicsInfo(bodyUniqueId=self.robotID, 
                                            linkIndex=l, 
                                            physicsClientId=self.physicsClient )
                nominal_mass = dyn_info[0]
                self.link_masses.append(nominal_mass)
                self.total_mass += nominal_mass
            # generate the types from this robot urdf name code
            modules_types = [0] # module type 0 is chassis, on all robots
            chassis_attachments = []
            limb_attachments = []
            ii = 1
            module_joint_is_on = [] # which module is this joint on?
            joint_index_on_module = [] # how far along on the chain is this joint on this module?
            action_indexes =[0,0] # no actions on base

            for i in range(6):
                letter  = urdf_name[i]
                if letter=='l':
                    modules_types.append(1) # module type 1 is a leg
                    chassis_attachments.append(ii)
                    limb_attachments.append([0]) # attached to chassis
                    module_joint_is_on.extend([i,i,i]) # three joints on leg
                    joint_index_on_module.extend([0,1,2])
                    action_indexes.append(action_indexes[-1]+3) # the next three actions are for this module
                    ii+=1

                elif letter=='w':
                    modules_types.append(2) # module type 2 is a wheel
                    chassis_attachments.append(ii)
                    module_joint_is_on.extend([i,i]) # two joints on wheel
                    joint_index_on_module.extend([0,1])
                    limb_attachments.append([0]) # attached to chassis
                    action_indexes.append(action_indexes[-1]+2) # the next two actions are for this module
                    ii+=1

                elif letter=='n':
                    chassis_attachments.append(None)
                    # no actions are for this module


            # example attachments = [[1,2,3,4], [0], [0], [0], [0]]
            attachments = [chassis_attachments] + limb_attachments

            self.modules_types = modules_types
            self.attachments = attachments
            self.module_joint_is_on = module_joint_is_on
            self.joint_index_on_module = joint_index_on_module
            self.action_indexes = action_indexes

        # set to a centered initial joint angle with a little noise
        # print(len(self.moving_joint_inds))
        if randomize_start:
            joint_noise = pi/8
        else:
            joint_noise = 0
        for i in range(self.num_joints):
            center = self.moving_joint_centers[i]
            jind = self.moving_joint_inds[i]
            p.resetJointState( bodyUniqueId=self.robotID, 
                jointIndex = jind,
                targetValue= center + np.random.uniform(-1,1)*joint_noise, 
                targetVelocity = 0,
                physicsClientId=self.physicsClient )

        # set commands to be zero
        p.setJointMotorControlArray(
            bodyUniqueId=self.robotID, 
            jointIndices=self.moving_joint_inds, 
            controlMode=p.VELOCITY_CONTROL,
            targetVelocities = np.zeros(self.num_joints),
            forces = self.moving_joint_max_torques,
            physicsClientId=self.physicsClient)

        # steps to take with no control input before starting up
        # This drops to robot down to a physically possible resting starting position
        for i in range(100):
            p.stepSimulation(physicsClientId=self.physicsClient)
            # if self.show_GUI:
            #     time.sleep(self.time_step/self.sim_speed_factor)

        # set joints to random small velocities after its dropped to the ground
        if randomize_start:
            joint_noise = 0.1
            joint_states = p.getJointStates(self.robotID,
                                self.moving_joint_inds,
                                physicsClientId=self.physicsClient)
            for i in range(self.num_joints):
                theta = joint_states[i][0]
                jind = self.moving_joint_inds[i]
                max_vel = self.moving_joint_max_velocities[i]
                p.resetJointState( bodyUniqueId=self.robotID, 
                    jointIndex = jind,
                    targetValue= theta, 
                    targetVelocity = np.random.uniform(-1,1)*joint_noise*max_vel,
                    physicsClientId=self.physicsClient )


        self.update_state()

        # only need to get measurement stds once at reset
        self.measurement_stds = self.get_measurement_stds()

        env_state_init = self.get_state()
        self.module_state_len = []
        for s in env_state_init:
            self.module_state_len.append(len(s))
        self.module_action_len = list(np.diff(self.action_indexes))


    def add_joint_noise(self):
        # note: can't add joint position noise after setting state because it could cause ground penetration
        p = self.p
        
        # joint_angle_noise = np.pi/8
        joint_vel_noise = 0.25

        # for i in range(self.num_joints):
        #     center = self.moving_joint_centers[i]
        #     jind = self.moving_joint_inds[i]
        #     p.resetJointState( bodyUniqueId=self.robotID, 
        #         jointIndex = jind,
        #         targetValue= self.joint_angles[i],# + np.random.uniform(-1,1)*joint_angle_noise, 
        #         targetVelocity = self.joint_vels[i] + np.random.uniform(-1,1)*joint_vel_noise,
        #         physicsClientId=self.physicsClient )

        # set to a random joint velocity command and take a few steps
        p.setJointMotorControlArray(
            bodyUniqueId=self.robotID, 
            jointIndices=self.moving_joint_inds, 
            controlMode=p.VELOCITY_CONTROL,
            targetVelocities = (self.moving_joint_max_velocities*
                            np.random.uniform(-1,1,self.num_joints)
                            *joint_vel_noise),
            forces = self.moving_joint_max_torques,
            physicsClientId=self.physicsClient)
        for sim_step in range(self.n_time_steps_per_step):
            p.stepSimulation(physicsClientId=self.physicsClient)

        self.update_state()


    # update state of the robot, broken up by modules
    def update_state(self):
        p = self.p
        linkWorldPosition, linkWorldOrientationQuat = p.getBasePositionAndOrientation(
                        bodyUniqueId=self.robotID,physicsClientId=self.physicsClient)
        linkWorldOrientation = p.getEulerFromQuaternion(linkWorldOrientationQuat)
        self.worldLinkLinearVelocity, self.worldLinkAngularVelocity  = p.getBaseVelocity(
                        bodyUniqueId=self.robotID, physicsClientId=self.physicsClient)
        self.pos_xyz = linkWorldPosition
        self.pos_rpy = linkWorldOrientation

        rotmat = p.getMatrixFromQuaternion(linkWorldOrientationQuat)
        self.rotmat = rotmat
        self.z_axis = [rotmat[6], rotmat[7], rotmat[8]]

        # joint angles and vels
        current_joint = 0 # counter needed to handle arbitrary number of joints on each module type
        self.joint_angles = np.array([])
        self.joint_vels = np.array([])
        self.joint_torques = np.array([])

        joint_states = p.getJointStates(self.robotID,
                                        self.moving_joint_inds,
                                        physicsClientId=self.physicsClient)


        module_state_list = []

        # go through the modules and get the appropriate sensor data for each one
        for i in range(len(self.modules_types)):
            module_state = np.array([])

            if self.modules_types[i]==0: # chassis
                module_state = np.concatenate([
                        self.pos_xyz,
                        self.pos_rpy,
                        self.worldLinkLinearVelocity,
                        self.worldLinkAngularVelocity])

            elif self.modules_types[i]==1: # get joint angles on legs 
                for leg_j in range(3):
                    joint_state = joint_states[current_joint]
                    theta = joint_state[0]
                    dtheta = joint_state[1]#/self.moving_joint_max_velocities[current_joint]
                    tau = joint_state[3]#/self.moving_joint_max_torques[current_joint]
                    # appliedJointMotorTorque is index 3
                    self.joint_angles=np.append(self.joint_angles,theta)
                    self.joint_vels=np.append(self.joint_vels,dtheta)
                    self.joint_torques=np.append(self.joint_torques,tau)
                    current_joint+=1
                    module_state=np.append(module_state,theta)
                    module_state=np.append(module_state,dtheta)

            elif self.modules_types[i]==2: # wheel module
                # first joint on wheel is a revolute
                joint_state = joint_states[current_joint]
                theta1 = joint_state[0]
                dtheta1 = joint_state[1]#/self.moving_joint_max_velocities[current_joint]
                tau1 = joint_state[3]#/self.moving_joint_max_torques[current_joint]
                self.joint_angles=np.append(self.joint_angles,theta1)
                self.joint_vels=np.append(self.joint_vels,dtheta1)
                self.joint_torques=np.append(self.joint_torques,tau1)
                current_joint+=1
                module_state=np.append(module_state,theta1)
                module_state=np.append(module_state,dtheta1)

                # second joint is continuous
                # wheel only reports is speed, since its position doesnt matter
                joint_state = joint_states[current_joint]
                # theta 2 is irrelavant for a wheel so I am removing it. dtheta2 is useful.
                theta2 = joint_state[0]
                dtheta2 = joint_state[1]#/self.moving_joint_max_velocities[current_joint]
                tau2 = joint_state[3]#/self.moving_joint_max_torques[current_joint]
                self.joint_angles=np.append(self.joint_angles,theta2)
                # keep track of theta2 for bookkeeping
                self.joint_vels=np.append(self.joint_vels,dtheta2)
                self.joint_torques=np.append(self.joint_torques,tau2)
                current_joint+=1
                module_state=np.append(module_state,dtheta2)

            module_state_list.append(module_state)

        self.full_state = module_state_list

    # Manually set the state of the robot.
    # It might be faster to vectorize the setting of the joint angles
    # Input must be broken up into states for each module
    def set_state(self, module_states):
        p = self.p
        current_joint = 0

        # go through the modules and set each one
        for i in range(len(self.modules_types)):
            state = module_states[i]
            if self.modules_types[i]==0: # chassis
                p.resetBasePositionAndOrientation(
                    self.robotID, 
                    state[0:3], p.getQuaternionFromEuler(state[3:6]))#,self.physicsClient)
                p.resetBaseVelocity(                    
                    self.robotID, 
                    state[6:9], state[9:12])#,self.physicsClient)

            elif self.modules_types[i]==1: # get joint angles on legs 
                for leg_j in range(3):
                    theta = state[2*leg_j]
                    dtheta = state[2*leg_j+1]
                    p.resetJointState(self.robotID, 
                        self.moving_joint_inds[current_joint],
                        theta, dtheta)#, self.physicsClient)
                    current_joint+=1

            elif self.modules_types[i]==2: # wheel module
                # first joint on wheel is a revolute
                theta1 = state[0]
                dtheta1 = state[1]
                p.resetJointState(self.robotID, 
                        self.moving_joint_inds[current_joint],
                        theta1, dtheta1)#, self.physicsClient)
                current_joint+=1

                # second joint is continuous
                # wheel only reports is speed, since its position doesnt matter
                dtheta2 = state[2]
                theta2 = self.joint_angles[current_joint] # leave wheel angle alone
                p.resetJointState(self.robotID, 
                        self.moving_joint_inds[current_joint],
                        theta2, dtheta2)#, self.physicsClient)
                current_joint+=1

        self.update_state()


    # Return the state of the robot, broken up by modules
    def get_state(self): 
        return self.full_state

    def divide_action_to_modules(self,u):
        u_div = []
        for i in range(len(self.modules_types)):
            u_div.append( u[self.action_indexes[i]:self.action_indexes[i+1]] )
        return u_div

    def divide_action_to_modules_batch(self,u):
        u_div = []
        for i in range(len(self.modules_types)):
            u_div.append( u[:,self.action_indexes[i]:self.action_indexes[i+1]] )
        return u_div

    # step function advances simulation and updates state
    # assumes u is on [-1,1] and scales it by joint max.
    # u is not broken up by module, it is just a vector.
    def step(self, u):
        p = self.p
        # velocity control, will use as much torque as needed to reach that velocity
        p.setJointMotorControlArray(
            bodyUniqueId=self.robotID, 
            jointIndices=self.moving_joint_inds, 
            controlMode=p.VELOCITY_CONTROL,
            targetVelocities = u*self.moving_joint_max_velocities,
            forces=self.moving_joint_max_torques,
            physicsClientId=self.physicsClient)

        for sim_step in range(self.n_time_steps_per_step):
            p.stepSimulation(physicsClientId=self.physicsClient)
            if self.show_GUI:
                if len(self.arrow_ids)>0:
                    linkWorldPosition, linkWorldOrientationQuat = p.getBasePositionAndOrientation(
                        bodyUniqueId=self.robotID,physicsClientId=self.physicsClient)
                    self.pos_xyz = linkWorldPosition
                    self.draw_body_arrows()

                if self.follow_with_camera and self.show_GUI:
                    linkWorldPosition, linkWorldOrientationQuat = p.getBasePositionAndOrientation(
                        bodyUniqueId=self.robotID,physicsClientId=self.physicsClient)
                    self.pos_xyz = linkWorldPosition
                    # p.resetDebugVisualizerCamera(2,0,-30,[1+self.pos_xyz[0],self.pos_xyz[1],0.2],
                    p.resetDebugVisualizerCamera(1.5,0,-30,[self.pos_xyz[0],self.pos_xyz[1],0.2],
                    physicsClientId=self.physicsClient) 

                time.sleep(self.time_step/self.sim_speed_factor)

        self.update_state()

    # sets position control commands for a timestep
    def step_pos_control(self, pos):
        p = self.p

        p.setJointMotorControlArray(
            self.robotID, 
            jointIndices=self.moving_joint_inds, 
            controlMode=p.POSITION_CONTROL,
            targetPositions= pos,
            forces=self.moving_joint_max_torques,
            physicsClientId=self.physicsClient)

        for sim_step in range(self.n_time_steps_per_step):
            p.stepSimulation(physicsClientId=self.physicsClient)
            if self.show_GUI:
                time.sleep(self.time_step/self.sim_speed_factor)

        self.update_state()

    # Get the robot Center of Mass in world frame
    def get_center_of_mass(self):
        p = self.p
        link_positions = []
        link_positions.append(self.pos_xyz) # note: must have done an update state first so this is correct
        for li in range(self.num_links): 
            link_state = self.p.getLinkState(bodyUniqueId=self.robotID, 
                                            linkIndex=li, 
                                            physicsClientId=self.physicsClient )
            link_positions.append( link_state[0] )
        COM = np.asarray(self.link_masses*np.matrix(link_positions)/self.total_mass)[0]
        self.COM = COM # cache for later possible use
        return COM



    # other utilities for within pybullet
    def draw_line(self, from_pt, to_pt, color=[0,0,0] ):
        p = self.p
         # from, to, color
        p.addUserDebugLine(lineFromXYZ=from_pt,
            lineToXYZ=to_pt,
            lineColorRGB = color,
            lifeTime=0,# keeps it around forever
            physicsClientId = self.physicsClient)

    def start_video_log(self, fileName='traj.mp4'):
        p = self.p
        self.logID = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, fileName=fileName)

    def stop_video_log(self):
        p = self.p
        p.stopStateLogging(self.logID)


    def alt_tripod_positions(self, t, amplitude = 0.6, period = 1.25):
        phases = np.array([0,1,0,1,0,1])*np.pi
        amplitudes = np.array([1,1,1,1,1,1])*amplitude
        
        pos_div = []
        current_joint = 0
        for i in range(6):
            if self.loaded_urdf[i]=='l':
                th = (amplitudes[i]*np.heaviside(
                        np.sin(t*2*np.pi/period - phases[i]),0)*
                        np.sin(t*2*np.pi/period - phases[i]))
                pos_i = np.array([0,th, th])  
                pos_i += self.moving_joint_centers[current_joint:current_joint+3]
                current_joint+=3
            elif self.loaded_urdf[i]=='w': 
                pos_i = np.array([])
                current_joint+=2
            elif self.loaded_urdf[i]=='n': 
                pos_i = np.array([])

            pos_div.append(pos_i)
        return pos_div

    def alt_tripod_velocities(self, t, amplitude = 0.6, period = 1.25):
        phases = np.array([0,1,0,1,0,1])*np.pi
        amplitudes = np.array([1,1,1,1,1,1])*amplitude
        
        v_div = []
        current_joint = 0
        for i in range(6):
            if self.loaded_urdf[i]=='l':
                dth = (amplitudes[i]*
                        np.heaviside(np.sin(t*2*np.pi/period - phases[i]),0)*
                        np.cos(t*2*np.pi/period - phases[i])*2*np.pi/period)
                v_i = np.array([0,dth, dth])  
                v_i += self.moving_joint_centers[current_joint:current_joint+3]
                current_joint+=3
            elif self.loaded_urdf[i]=='w': 
                v_i = np.array([])
                current_joint+=2
            elif self.loaded_urdf[i]=='n': 
                v_i = np.array([])

            v_div.append(v_i)
        return v_div


    def leg_sine_wave(self, t, period=1, amplitude=1, phase=0):
        out = np.zeros(3)
        out[1] = amplitude*np.sin(t*2*pi/period + phase)
        out[2] = amplitude*np.sin(t*2*pi/period + phase)
        return out

    def alt_tripod_wave(self, t, period=1, amplitude=1):
        # only works for 6-leg robot
        out = np.zeros((6,3))
        for limb_number in range(6):

            amplitudes = np.array([0, amplitude, amplitude]) # change amplitude[0] to 0 to make it step in place
            phases = np.array([pi/2,0,0])
            offsets = np.array([0,0,0])
            # set phase for alternating tripod
            if np.mod(limb_number,2)==0:
                phases += 0 + pi/2
            else:
                phases += pi+ pi/2

            out[limb_number,0] = amplitudes[0]*np.sin(t*2*pi/period - phases[0]) + offsets[0]
            # out[limb_number,1] = amplitudes[1]*np.heaviside(np.sin(t*2*pi/period - phases[1]),0)*np.sin(t*2*pi/period - phases[1]) + offsets[1]
            # out[limb_number,2] = amplitudes[2]*np.heaviside(np.sin(t*2*pi/period - phases[2]),0)*np.sin(t*2*pi/period - phases[2]) + offsets[2]
            out[limb_number,1] = amplitudes[1]*np.sin(t*2*pi/period - phases[1]) + offsets[1]
            out[limb_number,2] = amplitudes[2]*np.sin(t*2*pi/period - phases[2]) + offsets[2]

        out = out.flatten()
        return out

    def get_measurement_stds(self):
    # returns the standard deviation of the measurement noise,
    # (standard deviation of the distribution, not variance)
    # which is assumed to be zero mean.
    # this was approximated from the measured robot data, and assumes it's stationary
        measurement_stds = []
        for i in range(len(self.modules_types)):
            if self.modules_types[i]==0: # chassis
                measurement_stds.append(
                    torch.tensor([1e-04, 1e-04, 1e-04, # xyz
                        2e-03, 2e-03, 2e-03, # rpy
                        1e-04,1e-04,1e-04, # vxyz
                        5e-3, 5e-3, 5e-3 # w_xyz
                        ], dtype=torch.float32) )

            elif self.modules_types[i]==1: # get joint angles on legs 
                measurement_stds.append(
                    torch.tensor([2e-5, 5e-2, 2e-5, 5e-2, 2e-5, 5e-2],
                      dtype=torch.float32) )
            elif self.modules_types[i]==2: # wheel module
                measurement_stds.append(
                    torch.tensor([2e-5, 5e-2, 5e-2],
                      dtype=torch.float32) )   
   

        return measurement_stds
    # example sensor noise 
# [tensor([0.0000e+00, 0.0000e+00, 1.3288e-05, 6.3331e-05, 5.6818e-04, 2.5309e-02,
#          0.0000e+00, 0.0000e+00, 0.0000e+00, 2.1432e-03, 2.0409e-03, 1.9623e-03]),
#  tensor([3.9633e-05, 1.6361e-02, 4.2375e-05, 1.6408e-02, 4.2369e-05, 1.6663e-02]),
#  tensor([4.1850e-05, 1.6528e-02, 3.7394e-05, 1.4178e-02, 4.4773e-05, 1.7578e-02]),
#  tensor([3.9314e-05, 1.4942e-02, 4.2634e-05, 1.6680e-02, 4.2178e-05, 1.6600e-02]),
#  tensor([3.9898e-05, 1.5943e-02, 4.1142e-05, 1.5948e-02, 4.1863e-05, 1.6125e-02]),
#  tensor([4.6039e-05, 1.7552e-02, 3.5823e-05, 1.4376e-02, 4.7222e-05, 1.8835e-02]),
#  tensor([3.7567e-05, 1.4525e-02, 3.9617e-05, 1.5166e-02, 4.0091e-05, 1.4704e-02])]

    def draw_body_arrows(self, vects=None, lineColors=None):
    # Draws arrow from the body frame and replaces them at each step call
    
        start_pt = np.array(self.pos_xyz)
        start_pt[2] = start_pt[2] + 0.05
        p = self.p


        if vects is None:
            (vects, lineColors) = self.arrow_data

        
        arrow_yaws = []
        arrow_head_len = []
        for vect in vects:
            arrow_yaws.append(np.arctan2(vect[1], vect[0]))
            if np.sqrt(vect[0]**2 + vect[1]**2)>0.05:
                arrow_head_len.append(0.05)
            else:
                arrow_head_len.append(0)

        arrow_head_angle = 3*np.pi/4

        n_arrows = len(vects)
        self.arrow_data = (vects, lineColors)
        if len(self.arrow_ids)>0:
            for i in range(n_arrows):
                yaw = arrow_yaws[i]
                lineTo = start_pt + np.array(vects[i])
                arrow_head1 = lineTo + np.array(
                    [np.cos(yaw+arrow_head_angle),
                     np.sin(yaw+arrow_head_angle),0]
                     )*arrow_head_len[i]
                arrow_head2 = lineTo + np.array(
                    [np.cos(yaw-arrow_head_angle),
                     np.sin(yaw-arrow_head_angle),0]
                     )*arrow_head_len[i]

                self.arrow_id_list = self.arrow_ids[i]
                self.arrow_id_list[0] = p.addUserDebugLine(
                    lineFromXYZ = start_pt,
                    lineToXYZ=lineTo, lineColorRGB=lineColors[i],
                    replaceItemUniqueId = self.arrow_id_list[0],
                    lineWidth = 2,
                    physicsClientId=self.physicsClient)
                self.arrow_id_list[1] = p.addUserDebugLine(
                    lineFromXYZ = lineTo,
                    lineToXYZ=arrow_head1, lineColorRGB=lineColors[i],
                    replaceItemUniqueId = self.arrow_id_list[1],
                    lineWidth = 2,
                    physicsClientId=self.physicsClient)
                self.arrow_id_list[2] = p.addUserDebugLine(
                    lineFromXYZ = lineTo,
                    lineToXYZ=arrow_head2, lineColorRGB=lineColors[i],
                    replaceItemUniqueId = self.arrow_id_list[2],
                    lineWidth = 2,
                    physicsClientId=self.physicsClient)
        else:
            for i in range(n_arrows):
                yaw = arrow_yaws[i]
                lineTo = np.array(vects[i]) + start_pt
                arrow_head1 = lineTo + np.array(
                    [np.cos(yaw+arrow_head_angle),
                     np.sin(yaw+arrow_head_angle),0]
                     )*arrow_head_len[i]
                arrow_head2 = lineTo + np.array(
                    [np.cos(yaw-arrow_head_angle),
                     np.sin(yaw-arrow_head_angle),0]
                     )*arrow_head_len[i]
                self.arrow_id0 = p.addUserDebugLine(
                    lineFromXYZ = start_pt,
                    lineToXYZ=lineTo, lineColorRGB=lineColors[i],
                    lineWidth = 2,
                    physicsClientId=self.physicsClient)
                self.arrow_id1 = p.addUserDebugLine(
                    lineFromXYZ = lineTo,
                    lineToXYZ=arrow_head1, lineColorRGB=lineColors[i],
                    lineWidth = 2,
                    physicsClientId=self.physicsClient)
                self.arrow_id2 = p.addUserDebugLine(
                    lineFromXYZ = lineTo,
                    lineToXYZ=arrow_head2, lineColorRGB=lineColors[i],
                    lineWidth = 2,
                    physicsClientId=self.physicsClient)
                self.arrow_ids.append([self.arrow_id0, self.arrow_id1, self.arrow_id2])


    def set_friction(self, plane_friction=None, wheel_friction=None, foot_friction=None):
        if plane_friction is not None:
            self.plane_friction = plane_friction
            p.changeDynamics(bodyUniqueId=self.planeId, 
                linkIndex=-1, lateralFriction=self.plane_friction, 
                physicsClientId=self.physicsClient)

        if foot_friction is not None:
            self.foot_friction = foot_friction

        if wheel_friction is not None:
            self.wheel_friction = wheel_friction

        # count all joints, including fixed ones
        num_joints_total = p.getNumJoints(self.robotID,
                        physicsClientId=self.physicsClient)
        for j_ind in range(num_joints_total):
            j_info = p.getJointInfo(self.robotID, 
                j_ind, physicsClientId=self.physicsClient)
            link_name = str(j_info[12])
            # raise friction value for feet
            if link_name.find('foot/INPUT')>=0:
                p.changeDynamics(bodyUniqueId=self.robotID, 
                                linkIndex=j_ind, lateralFriction=self.foot_friction, 
                                physicsClientId=self.physicsClient)
                self.foot_wheel_link_inds.append(j_ind)
            # raise friction value for wheel
            elif link_name.find('wheel/INPUT')>=0:
                p.changeDynamics(bodyUniqueId=self.robotID, 
                                linkIndex=j_ind, lateralFriction=1.0, 
                                anisotropicFriction=self.wheel_friction,
                                physicsClientId=self.physicsClient)


    def get_ff_torque_approx(self):
    # return a rough approximation of the ff torques for legs
    # uses a list of the points that are all low and within 0.02m of the lowest foot/wheel
        

        foot_wheel_states = self.p.getLinkStates(self.robotID,
                        self.foot_wheel_link_inds,
                        computeForwardKinematics=1,
                        physicsClientId=self.physicsClient)
        x_limbs = [s[4][0] for s in foot_wheel_states] # worldLinkFramePosition is in index 4
        y_limbs = [s[4][1] for s in foot_wheel_states] # worldLinkFramePosition is in index 4
        z_limbs = [s[4][2] for s in foot_wheel_states] # worldLinkFramePosition is in index 4
        radius = 0.105
        foot_diam = 0.04
        x_limbs = np.array(x_limbs)
        y_limbs = np.array(y_limbs)
        z_limbs = np.array(z_limbs)
        # take the radius distance down from wheels (approximates low point)
        n_limbs = len(self.modules_types[1:])
        for i in range(n_limbs):
            mt = self.modules_types[i+1]
            if mt == 1:
                z_limbs[i] = z_limbs[i] - foot_diam
            elif mt == 2:
                z_limbs[i] = z_limbs[i] - radius
        z_limbs[z_limbs<0] = 0

        # print(z_limbs)
        lowest_ind = np.argmin(z_limbs)
        lowest_val = z_limbs[lowest_ind]
        nearby_lowest_inds = np.where( np.abs(z_limbs - lowest_val)<0.02)[0]
        # print(nearby_lowest_inds)
        # tau = R x F
        F = -self.total_mass*9.81 / len(nearby_lowest_inds)
        R = np.sqrt((x_limbs - self.pos_xyz[0])**2 + (y_limbs - self.pos_xyz[1])**2)
        
        RF = R*F
        tau_approx = np.zeros(n_limbs)*np.nan
        tau_approx[nearby_lowest_inds] = RF[nearby_lowest_inds]
        # print(tau_approx)

        tau_out = []
        for i in range(n_limbs):
            mt = self.modules_types[i+1]
            if mt == 1:
                tau_out.extend([np.nan, tau_approx[i], np.nan])
            elif mt == 2:
                tau_out.extend([np.nan, np.nan])

        return tau_out


    # get a list of all contact points of the robot with the ground in world frame
    def get_contacts(self):
        # query the simulation for the contact points
        contact_positions = []
        # contact_forces= [] # see pybullet guide. just getting normal force for now.
        for k in range(len(self.link_names)):
            contact_points = self.p.getContactPoints(bodyA=self.planeId, bodyB=self.robotID, 
                                               linkIndexA=-1, linkIndexB=k,
                                               physicsClientId=self.physicsClient)
            if len(contact_points)>0:
                # each link could potentially have multiple contact points.
                # here add them each onto the list seperately
                for point in contact_points:
                    # force = point[9]
                    location = point[6] # positionOnB, vec3, list of 3 floats. contact position on B, in Cartesian world coordinates
                    # contact_forces.append(force)
                    contact_positions.append(location)
        self.contact_positions = contact_positions # cache for later possible use
        return contact_positions

    # Get the distance of the mean contact position from the center of mass.
    # Generally a good measure of static stability, although not flawless.
    def static_stability(self):
        contact_positions = self.get_contacts()
        COM = self.get_center_of_mass()
        # todo: might want to weight this by some nominal quasi-static p-inv of masses
        print(np.mean(contact_positions,0))
        print(COM)
        diff_in_plane = np.mean(contact_positions,0)[0:2] -  COM[0:2]
        return np.linalg.norm(diff_in_plane)


# Demonstration of robot_env usage
if __name__ == '__main__':

    # Utilties needed to compute cost matrix
    from planning_utils import create_cost_mats2, get_pos_control_inds, slew_rate_penalty
    from utils import combine_state, to_tensors

    ## Initialize environment
    env = robot_env(show_GUI = True) # brings GUI to screen
    # env = robot_env(show_GUI = False) # can run headless without GUI
    env.reset_terrain()
    urdf_options = ['llllll', 'wnwwnw', 'lwllwl']
    # Note that the urdf name must be made up of six characters, each of which is 'l', 'n', or 'w'
    # for leg, none, or wheel on the six ports.
    # This environment is particular to those module types.
    device = torch.device('cpu')

    for i_reset in range(6): # Iterate through the robot to add at each reset

        urdf_name = urdf_options[i_reset % len(urdf_options)]
        # Select which urdf to load. It will load ../urdf/<urdf_name>.urdf
        print('Resetting to robot ' +urdf_name)
        env.reset_robot(urdf_name=urdf_name, randomize_start=False) 
        
        ## This section is used to compute the cost of a motion. 
        # It is not the same as a reward function call used in model-free RL 
        # because it is used in TrajOpt.
        # The cost matrix must be recomputed for each new trajectory and new robot 
        T = 20 # time horizon for cost computation
        env_state_init = env.get_state()
        module_state_len = []
        for s in env_state_init:
            module_state_len.append(len(s))
        module_action_len = list(np.diff(env.action_indexes))
        env_state_init = combine_state(to_tensors(env_state_init)).to(device)
        leg_pos_inds, leg_control_inds, wheel_steer_inds, wheel_control1_inds, wheel_control2_inds = get_pos_control_inds(
            env.modules_types, module_state_len, module_action_len)
        x_init = env_state_init.to(device)
        delta_xyyaw_des = np.array([[1,0,0]])
        last_u = None
        n_state= np.sum(module_state_len)
        n_ctrl = env.num_joints
        # Create quadratic cost matrices, using batch size of 1
        C, c =create_cost_mats2([0], device, T, 1, env,
                     env_state_init, n_state, n_ctrl,
                     leg_pos_inds, leg_control_inds,
                     wheel_steer_inds, wheel_control1_inds, wheel_control2_inds,
                     last_u = last_u, slew_rate_penalty = slew_rate_penalty,
                     xyyaw_start = x_init[:,[0,1,5]].detach().cpu(), 
                     delta_xyyaw_des = delta_xyyaw_des  )
        print('sizes: C ' + str(C.shape) + ', c ' + str(c.shape))


        ## Run some random actions and compute the cost
        net_cost = 0
        for i_step in range(T):
            u = torch.rand(n_ctrl, device=device)*2-1 # random action. Actions must be on [-1,1]
            env.step(u) 
            state = env.get_state() # after each step, it runs update_state internally and caches the state so you can get_state whenever you want
            x = combine_state(to_tensors(state)).to(device).squeeze() # states come in broken up by module, so cat them here
            xu = torch.cat([x, u]) # cat state and action to pass into cost mat
            if i_step==0: print('sizes: x ' + str(x.shape) + ', u ' + str(u.shape))
            # cost = (1/2)*x^T*C*x + c^T*x 
            net_cost += torch.dot(xu, torch.matmul(C[i_step,0], xu))/2 + torch.dot(c[i_step,0], xu) # multiply cost mat by xu
        print('Net cost for reset ' + str(i_reset) + ': ' + str(net_cost.item()))
