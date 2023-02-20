#!/usr/bin/env python
# coding: utf-8
'''

Source code for paper "Learning modular robot control policies" in Transactions on Robotics
Julian Whitman, Dec. 2022. 


Run the policy on robot
This will require very specific robots built from HEBI actuators
But we have included it here as an example for how learning modular policies 
can be applied to real robots.
'''

import hebi
import time
import pybullet as p
import torch
import numpy as np
import itertools
import json
from scipy.spatial.transform import Rotation as R
pi = np.pi
from robot_env import robot_env
import pgnn_control as pgnnc
from utils import create_control_inputs2

# Joystick control, if no joystick is there, robot will just go forward
USE_JOY = True
if USE_JOY:
    from vrjoystick import init_joystick, read
    joy = init_joystick()
    
    
# specify robot type (unfortunately, robot cannot yet self-identify its design)
urdf_name = 'llllll'

# path to the pytorch neural network weights
folder = 'saved/with_tripod/'



# if not('group' in locals() or 'group' in globals()):

# create robot group
lookup = hebi.Lookup()
if urdf_name == 'llllll':
    names = [        
        'X-80363', 'X-80312', 'X-00055',
        'X-80430', 'X-80335', 'X-00039',
        'X-80433', 'X-80454', 'X-00547',
        'X-80381', 'X-80234', 'X-00042', 
        'X-80455', 'X-80323', 'X-00058', 
        'X-80263', 'X-80257', 'X-00529',
     ]

elif urdf_name == 'wllllw': 
    names = ['X-00039','X-80430',
            'X-80433', 'X-80454', 'X-00547',
            'X-80363', 'X-80312', 'X-00055',
            'X-80263', 'X-80257', 'X-00529',
            'X-80381', 'X-80234', 'X-00042', 
            'X-00058', 'X-80455'] 

elif urdf_name == 'lnwwnl':
    names = ['X-80263', 'X-80257', 'X-00529', # leg 1
         'X-00547', 'X-80433', 
          'X-00042', 'X-80381', 
          'X-80363', 'X-80312', 'X-00055'
     ] 

elif urdf_name == 'wnwwnw': 
    names = ['X-00039','X-80430',
         'X-00547', 'X-80433', 
          'X-00042', 'X-80381', 
                 'X-00058', 'X-80455']
elif urdf_name == 'lwwwnl': 
    names = [    'X-80263', 'X-80257', 'X-00529', # leg 1
       'X-00058', 'X-80455',
         'X-00547', 'X-80433', 
          'X-00042', 'X-80381', 
          'X-80363', 'X-80312', 'X-00055'] # leg 5 ]

elif urdf_name == 'lwwwwl': 
    names = [    'X-80263', 'X-80257', 'X-00529', # leg 1
       'X-00058', 'X-80455',
         'X-00547', 'X-80433', 
          'X-00042', 'X-80381', 
            'X-00039','X-80430',
          'X-80363', 'X-80312', 'X-00055'] # leg 5 ]
             
elif urdf_name == 'wlwwlw':
    names = ['X-00039','X-80430',
                 'X-80363', 'X-80312', 'X-00055',
         'X-00547', 'X-80433', 
          'X-00042', 'X-80381', 
                     'X-80263', 'X-80257', 'X-00529',# leg 5
                 'X-00058', 'X-80455']
elif urdf_name == 'wwllww':
    names = ['X-00039','X-80430',
            'X-00547', 'X-80433', 
            'X-80363', 'X-80312', 'X-00055',
            'X-80263', 'X-80257', 'X-00529',
            'X-00042', 'X-80381', 
            'X-00058', 'X-80455']
elif urdf_name == 'wnllnw':
    names = ['X-00039','X-80430',
            'X-80363', 'X-80312', 'X-00055',
            'X-80263', 'X-80257', 'X-00529',
            'X-00058', 'X-80455'] 
    
elif urdf_name == 'llwwll': 
    names = [
        'X-80363', 'X-80312', 'X-00055',
        'X-80433', 'X-80454', 'X-00547',
        'X-00039','X-80430',
        'X-00058', 'X-80455',
        'X-80381', 'X-80234', 'X-00042', 
        'X-80263', 'X-80257', 'X-00529',
        ] 
elif urdf_name == 'lwllwl': 
    names = [
        'X-80363', 'X-80312', 'X-00055',
                'X-00039','X-80430',
        'X-80433', 'X-80454', 'X-00547',
        'X-80381', 'X-80234', 'X-00042', 
                'X-00058', 'X-80455',
        'X-80263', 'X-80257', 'X-00529',
        ] 
elif urdf_name == 'lnllnl': 
    names = [
        'X-80363', 'X-80312', 'X-00055',
        'X-80433', 'X-80454', 'X-00547',
        'X-80381', 'X-80234', 'X-00042', 
        'X-80263', 'X-80257', 'X-00529',
        ] 

# create group, the robot object
group = lookup.get_group_from_names('*', names)
time.sleep(0.1)
print(group)
group_info = hebi.GroupInfo(group.size)
group_info = group.request_info()

# create command and feedback structures
group_command = hebi.GroupCommand(group.size)
group_feedback = hebi.GroupFeedback(group.size)

# IO board is onboard to get orientation IMU data from body
io_group = lookup.get_group_from_names('*',['IO'] )
time.sleep(0.1)
print(io_group)
io_feedback = hebi.GroupFeedback(io_group.size)




# make environment so that it can create module types, kinematics. 
# We won't use the step function here though, its just used as a way to get info out of the urdf.

# env = robot_env(show_GUI = True)
env = robot_env(show_GUI = False)
env.reset_terrain()
env.reset_robot(urdf_name=urdf_name, randomize_start=False)

modules_types = env.modules_types
attachments = env.attachments
n_modules = len(modules_types)
module_action_len = list(np.diff(env.action_indexes))
print('made env')




# set gains on the actuators
def create_gains(modules_types):
    gains = dict()
    gains['vkp'] = []
    gains['vkd'] = []
    gains['ekp'] = []
    gains['ekd'] = []
    gains['pos_lim_max']= []
    gains['pos_lim_min']= []
    gains['pos_lowpass']= []
    gains['velocity_feed_forward'] = []
    gains['effort_feed_forward'] = []
    gains['velocity_output_lowpass'] = []
    gains['velocity_target_lowpass'] = []
    gains['velocity_feed_forward'] = []
    gains['effort_target_lowpass'] = []
    gains['effort_output_lowpass'] = []
   
    
#     current_joint = 0 # counter needed to handle arbitrary number of joints on each module type
    # go through the modules and get the appropriate sensor data for each one
    for i in range(len(modules_types)):
        if modules_types[i]==1: # leg module

            gains['vkp'].append([.075,.075, .075])  # 0.1 is too high! 0.05 seems ok
            gains['vkd'].append([.00, .00, .00]) 
            gains['ekp'].append([0,0,0]) 
            gains['ekd'].append([0,0,0]) 
            gains['pos_lim_min'].append([pi/6, -1.5, pi/6])
            gains['pos_lim_max'].append([5*pi/6, 0.8, 2.5])
            gains['velocity_output_lowpass'].append([0.8,0.8,0.8])
            gains['velocity_target_lowpass'].append([0.4,0.8,0.7])
            gains['velocity_feed_forward'].append([1,1,1])
            gains['effort_feed_forward'].append([0.3,0.8,0.7])
            gains['effort_target_lowpass'].append([0.3,0.3,0.3])
            gains['effort_output_lowpass'].append([0.8,0.8,0.8])         
    
    
        elif modules_types[i]==2: # wheel
            gains['vkp'].append([.02, .075])
            gains['vkd'].append([0.000, 0.000])
            gains['ekp'].append([0,0]) 
            gains['ekd'].append([0,0]) 
            gains['pos_lim_min'].append([-pi/4, -np.inf])
            gains['pos_lim_max'].append([pi/4, np.inf])
            gains['velocity_output_lowpass'].append([0.8,0.2])
            gains['velocity_target_lowpass'].append([0.6,0.075])
            gains['velocity_feed_forward'].append([1,1])
            gains['effort_feed_forward'].append([1,0.25])
            gains['effort_target_lowpass'].append([0.4,0.03])
            gains['effort_output_lowpass'].append([0.8,0.1])

    for key in gains:
        gains[key] = list(itertools.chain(*gains[key]))
    return gains 

# create command and feedback structures
group_command = hebi.GroupCommand(group.size)
group_feedback = hebi.GroupFeedback(group.size)

gains = create_gains(modules_types)
# print(gains)
group_command.velocity_kp = gains['vkp']
group_command.velocity_kd = gains['vkd']
group_command.effort_kp   = gains['ekp'] 
group_command.effort_kd   = gains['ekd']
group_command.position_limit_max = gains['pos_lim_max']
group_command.position_limit_min = gains['pos_lim_min']
group_command.velocity_output_lowpass = gains['velocity_output_lowpass']
group_command.velocity_target_lowpass = gains['velocity_target_lowpass']
group_command.velocity_feed_forward  = gains['velocity_feed_forward']
group_command.effort_feed_forward  = gains['effort_feed_forward']
group_command.effort_target_lowpass  = gains['effort_target_lowpass']
group_command.effort_output_lowpass  = gains['effort_output_lowpass']
group.command_lifetime = 200 # ms
gains_set = group.send_command_with_acknowledgement(group_command)
if gains_set:
    print('gains set.')
else:
    print('failed to set gains')




def convert_to_pi(q):
    # function change all joint angle values to between -pi and pi
    q_mod = np.mod( q, 2*np.pi)
    q_new =  np.where(q_mod > np.pi, q_mod-2*np.pi, q_mod)
    return q_new


# function to get IMU data from IO board and convert to correct formats
def get_IMU_data(io_feedback):
    orientation_quat = io_feedback.orientation[0]
    q = [orientation_quat[3], orientation_quat[0], orientation_quat[1], orientation_quat[2]]
    # q = orientation_quat
    angular_vel = io_feedback.gyro[0] # matches correctly as is
    r = R.from_quat(q)
    orientation_rpy = r.as_euler('xyz', degrees=False)

    # change frames to match my convention
    orientation_rpy[0] = - orientation_rpy[0] + pi
    if orientation_rpy[0]>pi:
        orientation_rpy[0]-=2*pi
    orientation_rpy[2] = -orientation_rpy[2]
    
    return orientation_rpy, angular_vel

# Output a list of sensor data for the various modules,
# in the form [[data_module_1], [data_module_2], ...]
def get_sensor_data(fbk, io_fbk):
    xyz = [0,0,1]
    v_xyz = [0,0,0]
    rpy, worldLinkAngularVelocity = get_IMU_data(io_fbk)
    rpy[-1] = 0
    
    sensor_data = []
    current_joint = 0 # counter needed to handle arbitrary number of joints on each module type

    # go through the modules and get the appropriate sensor data for each one
    for i in range(len(modules_types)):
        
        if modules_types[i]==0:
            data = xyz + rpy.tolist() + v_xyz + worldLinkAngularVelocity.tolist()   # keeping this shorter makes it easier to learn and transfer to reality
#             data = rpy.tolist()[0:2] + worldLinkAngularVelocity.tolist()   # keeping this shorter makes it easier to learn and transfer to reality
            sensor_data.append(torch.tensor(data, dtype=torch.float32))
        
        elif modules_types[i]==1: # leg module
            # get joint angle on legs 
            angles_now = []
            joint_vel_now = []
            data = []
            for leg_j in range(3):
                theta = fbk.position[current_joint]
                dtheta = fbk.velocity[current_joint]
                data.append(theta)
                data.append(dtheta)
                current_joint+=1
            # combine into sensor data tensor
            sensor_data.append( torch.tensor(data, dtype=torch.float32) )

        elif modules_types[i]==2: # wheel
            # first joint on wheel is a revolute
            theta1 = fbk.position[current_joint]
            dtheta1 = fbk.velocity[current_joint]
            theta1 = convert_to_pi(theta1).tolist()
            current_joint+=1

            # second joint is continuous
            theta2 = fbk.position[current_joint]
            dtheta2 = fbk.velocity[current_joint]
            theta2 = convert_to_pi(theta2).tolist()
            current_joint+=1
            # wheel only reports is speed, since its position doesnt matter
            data = [theta1, dtheta1, dtheta2]
            sensor_data.append(  torch.tensor(data, dtype=torch.float32) )

    return sensor_data

def get_robot_fbk(group_feedback, io_feedback):
    
    n_feedback_attempts = 0
    
    # get robot feedback
    fbk = group_feedback
    group_feedback = group.get_next_feedback(timeout_ms=1, reuse_fbk=group_feedback)
    if (group_feedback == None): # occasionally a feedback might be dropped. worse over wifi.
        n_feedback_attempts+=1
        print('no robot feedback came, attempt ' + str(n_feedback_attempts))
        group_feedback = fbk
    else:
        n_feedback_attempts = 0
        
    if n_feedback_attempts>0:
        print('Attempts: ' + str(n_feedback_attempts) + ', Min Voltage: ' + np.array2string(np.min(group_feedback.voltage), precision=2 ) + 
         ' Total current: ' +  np.array2string(np.sum(group_feedback.motor_current) , precision=2 ))
    
    # get IO feedback
    io_fbk = io_feedback
    io_feedback = io_group.get_next_feedback(timeout_ms=0.5,reuse_fbk=io_feedback)
    if (io_feedback == None):
        print('no io feedback came')
        io_feedback = io_fbk
        
    return group_feedback, io_feedback
    

# # function to load network parameters
device = torch.device("cpu")
def init_GNN():
    
    save_dict = torch.load( PATH, map_location=lambda storage, loc: storage)
    gnn_state_dict= save_dict['gnn_state_dict'] 
    internal_state_len = save_dict['internal_state_len'] 
    message_len = save_dict['message_len'] 
    hidden_layer_size = save_dict['hidden_layer_size']
    goal_len =3

    gnn_nodes = pgnnc.create_GNN_nodes(internal_state_len, 
                                       message_len, hidden_layer_size,
                                       device, goal_len=goal_len, body_input = True)
    pgnnc.load_state_dicts(gnn_nodes, save_dict['gnn_state_dict']) 

    # create module containers for the nodes.
    # modules are the graph neural network node instantiations.
    modules = []
    for i in range(n_modules):
        modules.append(pgnnc.Module(i, gnn_nodes[modules_types[i]], device))
    
    return modules

# read joystick and convert to body velocity commands for network inptus
def get_goals():
    vxy_scale = 1.6
    vyaw_scale = 1.6
    if USE_JOY:
        buttons = None
        axes, buttons, povs = read(joy)
        axes = np.array(axes)
        axes[np.abs(axes)<0.01] = 0
        desired_xyyaw = np.array([-axes[1]*vxy_scale,
                                  -axes[0]*vxy_scale,
                                  -axes[2]*vyaw_scale])
    else: # no joystick, just go forward
        desired_xyyaw =  np.array([1,0, 0])
        
    return desired_xyyaw, buttons

# forward pass of neural network returns joint velocity command and feedforward torques
def run_GNN(modules, node_inputs):

    for module in modules: # this prevents the LSTM in the GNN nodes from 
        # learning relations over time, only over internal prop steps.
        module.reset_hidden_states(1) 

    with torch.no_grad():
        out_mean, out_var = pgnnc.run_propagations(
            modules, attachments, 2, node_inputs, device)
        u_out_mean = []
        tau_out_mean = []
        for mm in range(n_modules):
            u_out_mean.append(out_mean[mm][:,:module_action_len[mm]])
            tau_out_mean.append(out_mean[mm][:,module_action_len[mm]:])
        u_np= torch.cat(u_out_mean,-1).squeeze().numpy()
        tau_np= torch.cat(tau_out_mean,-1).squeeze().numpy()

    return u_np, tau_np





# Interpolate to starting position
t_end = 2
group_command = hebi.GroupCommand(group.size)
group_feedback = None
while group_feedback is None:
    print('getting fbk')
    group_feedback = group.get_next_feedback(reuse_fbk=group_feedback)
initial_angles = group_feedback.position

# angles based on joint centers
final_angles = np.array(env.moving_joint_centers )
# Note: this will cause the continuous joints to go to zero, 
# which might be a longer journey than the time alloted

print('Moving to starting position:')
pos_cmd = np.zeros(group.size)
start_time = time.time()
time_now = start_time
while time_now - start_time < t_end:
    for i in range(len(initial_angles)):
        angle_i = np.interp(time_now - start_time, [0, t_end], [initial_angles[i], final_angles[i]])
        pos_cmd[i] = angle_i
    group_command.position = pos_cmd
#     print(pos_cmd)
    group.send_command(group_command)
    time.sleep(0.01)
    time_now = time.time()
    
    
# waut for joystick button press to start
start_time = time.time()
time_now = start_time
waiting = True
# while time_now - start_time < 1:
print('press a button 9 to start')
while waiting and USE_JOY:
    group.send_command(group_command)
    time.sleep(0.01)
    time_now = time.time()
    desired_xyyaw, buttons = get_goals()
    if buttons[8]==1:
        waiting = False
    

    
print("Moved to initial point:")
print(np.round(final_angles,2))

# sending NAN to actuators makes them ignore that part of the lowlevel pid loop.
group_command.position =[np.nan]*group.size
group_command.velocity =[np.nan]*group.size
group_command.effort =[np.nan]*group.size

joint_lower_pos_limits = np.array(gains['pos_lim_min'])#np.array(env.moving_joint_limits)[:,0]
joint_upper_pos_limits = np.array(gains['pos_lim_max'])#np.array(env.moving_joint_limits)[:,1]
joint_max_velocities = np.array(env.moving_joint_max_velocities)
joint_max_torques = np.array(env.moving_joint_max_torques)

# load the policy
running = True
initial_yaw = 0
start_time = time.time()
dt = 20./240.
step = 0
modules = init_GNN()
last_state = None

print('Press a button 10, 11, or 12 to kill')


# MAIN LOOP
while running:
    iter_start_time = time.time()
    
    # get joystick command
    desired_xyyaw, buttons = get_goals()
    if buttons[9]==1 or buttons[10]==1 or buttons[11]==1:
        print(buttons)
        running = False
        break
        
    # query robot for sensor feedback
    group_feedback, io_feedback = get_robot_fbk(group_feedback, io_feedback)
    
    # convert to sensor data structure
    sensor_data = get_sensor_data(group_feedback, io_feedback)
    
    if torch.abs(sensor_data[0][0])>np.pi/2 or torch.abs(sensor_data[0][1])>np.pi/2:
        print('ROBOT FLIPPED')
        break
    
    
    if last_state is None: # only true at first time step
        last_state = sensor_data
    
    node_inputs = create_control_inputs2(state,sensor_data,
                           torch.tensor(desired_xyyaw, dtype=torch.float32),
                           rotate_goals = False)
    

    action, torque_ff = run_GNN(modules, node_inputs)
#     print('Action: ' + str(np.round(action,2)))
    
    # convert action to joint commands
    current_pos = group_feedback.position[:]
    j_velocities = action*joint_max_velocities
    j_torques = torque_ff*joint_max_torques
    
    # track how long the iteration is taking
    iter_end_time = time.time()
    time_elapsed = iter_end_time - start_time
    iter_time_elapsed = iter_end_time- iter_start_time
#     print('Iteration time elapsed: ' + str(np.round(iter_time_elapsed,3)))
    
        
    # make sure we are not sending outside joint limits
    new_pos = current_pos + j_velocities*dt
    new_pos_clip = np.clip(new_pos, joint_lower_pos_limits+0.05, joint_upper_pos_limits-0.05)
    j_velocities = (new_pos - current_pos)/dt

    
    ## In case the position is over the soft limit, return to it like a spring
    over_pos_limit = np.where(group_feedback.position > joint_upper_pos_limits)[0]
    under_pos_limit = np.where(group_feedback.position < joint_lower_pos_limits)[0]
    j_velocities[over_pos_limit]  = -10*(group_feedback.position[over_pos_limit]  
                                          - joint_upper_pos_limits[over_pos_limit])
    j_velocities[under_pos_limit] = -10*(group_feedback.position[under_pos_limit] 
                                          - joint_lower_pos_limits[under_pos_limit])
    j_torques[over_pos_limit]  = -5*(group_feedback.position[over_pos_limit]  
                                          - joint_upper_pos_limits[over_pos_limit])
    j_torques[under_pos_limit] = -5*(group_feedback.position[under_pos_limit] 
                                          - joint_lower_pos_limits[under_pos_limit])    
    
#     if np.any(over_pos_limit) or np.any(under_pos_limit):
#         running = False
#         break
    
    # send data to robot
    group_command.position =[np.nan]*group.size # I found sending positions was not great
    group_command.velocity = j_velocities # sending velocities with FF torque worked best
    group_command.effort = j_torques
    group.send_command(group_command)

    
    # Sleep to keep iteration frequency constant. 
    # This assumes that each forward pass takes the same amount of time.
    if iter_time_elapsed<=dt:
        time.sleep(dt - iter_time_elapsed)
#         pass
    else:
        # slow loop happens if the feedback gets dropped too much or robot does not respond 
        print('Slow loop, '+ str(iter_time_elapsed))
        
    step+=1
    


print('done')

# send nans after we're done so that robot collapses down
group_command.position =[np.nan]*group.size
group_command.velocity =[np.nan]*group.size
group_command.effort =[np.nan]*group.size
group.send_command(group_command)






