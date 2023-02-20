#!/usr/bin/env python
# coding: utf-8
'''
Source code for paper "Learning modular robot control policies" in Transactions on Robotics
Julian Whitman, Dec. 2022. 

Run modular policy on robot in simulation.
The control inputs stripped to frame with (x y yaw) removed
Uses the trained policy from the velocity input
Uses previous observation as input for next action

If you have a pygame compatible gamepad/joystick, you could change USE_JOY to True and drive the robot around.

'''

# load libraries
import torch
from robot_env import robot_env
import numpy as np
import pgnn_control as pgnnc
from utils import to_tensors, combine_state, wrap_to_pi, rotate, create_control_inputs

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

# init env
# use the env to get the size of the inputs and outputs
env = robot_env(show_GUI = True)
env.reset_terrain()

    
# USE_JOY = True 
USE_JOY = False
if USE_JOY:
    from vrjoystick import init_joystick, read
    joy = init_joystick()
    

folder = 'saved/with_tripod1/'
# folder = 'saved/no_tripod1/'
PATH =  'multidesign_control_iter3.pt'
PATH = folder + PATH

save_dict = torch.load( PATH, map_location=lambda storage, loc: storage)
urdf_names = save_dict['urdf_names']
print(urdf_names)
urdf_name = 'wnwwnw'
urdf_name = 'wwwwww'
urdf_name = 'llllll'

gnn_state_dict= save_dict['gnn_state_dict'] 
internal_state_len = save_dict['internal_state_len'] 
message_len = save_dict['message_len'] 
hidden_layer_size = save_dict['hidden_layer_size']
goal_len =3

print(save_dict['comment'])

env.reset_robot(urdf_name=urdf_name, randomize_start=False)

attachments = env.attachments
modules_types = env.modules_types
print('attachments: ' + str(attachments))
print('modules_types: ' + str(modules_types))
n_modules = len(modules_types)

env_state_init = env.get_state()
module_state_len = []
for s in env_state_init:
    module_state_len.append(len(s))

state_len= np.sum(module_state_len)
action_len = env.num_joints
module_action_len = list(np.diff(env.action_indexes))

module_sa_len = module_state_len+ module_action_len

gnn_nodes = pgnnc.create_GNN_nodes(internal_state_len, 
                                   message_len, hidden_layer_size,
                                   device, goal_len=goal_len, body_input = True)
pgnnc.load_state_dicts(gnn_nodes, save_dict['gnn_state_dict']) 

# create module containers for the nodes
modules = []
for i in range(n_modules):
    modules.append(pgnnc.Module(i, gnn_nodes[modules_types[i]], device))



env.p.resetDebugVisualizerCamera(2.1,0,-65,[0,0,0.2],physicsClientId=env.physicsClient) 
# env.p.resetDebugVisualizerCamera(2.1,0,-89.999,[0,0,0.2],physicsClientId=env.physicsClient) 
env.reset_robot(urdf_name=urdf_name, randomize_start=False, start_xyyaw=[0,0,0])
# vid_fname=urdf_name+'_steer_vel.mp4'
# env.start_video_log(fileName=vid_fname)

robot_button_mapping = [0,1,2,3,4,8,9,10]
n_buttons_to_check = len(robot_button_mapping)
vxy_scale = 20*(20/240)*0.314/0.75
vyaw_scale = 20*(20/240)*1.1/0.75
robot_ind = 0
env_state = env.get_state()
last_states = [smm.to(device) for smm in to_tensors(env_state)]
tau_list = []
u_list = []
step = 0
buttons = np.zeros(12)
while True:
    buttons = np.zeros(12)
    if USE_JOY:
        axes, buttons, povs = read(joy)
        axes = np.array(axes)
        axes[np.abs(axes)<0.01] = 0
        desired_xyyaw = np.array([-axes[1]*vxy_scale,
                                  -axes[0]*vxy_scale,
                                  -axes[2]*vyaw_scale])
        if buttons[9]==1 or buttons[8]==1:
            break
        
        if np.any(buttons[0:n_buttons_to_check]) or buttons[10] or buttons[11]:
            if np.any(buttons[0:n_buttons_to_check]):
                pressed = np.where(buttons[0:n_buttons_to_check])[0][0]
                urdf_name = urdf_names[robot_button_mapping[pressed]]
            else:
                ind = urdf_names.index(urdf_name)
                urdf_name = urdf_names[(ind+1) % len(urdf_names)]

            env.reset_terrain()
            env.reset_robot(urdf_name=urdf_name, randomize_start=False, start_xyyaw=[0,0,0])
            attachments = env.attachments
            modules_types = env.modules_types
            n_modules = len(modules_types)
            env_state_init = env.get_state()
            module_state_len = []
            for s in env_state_init:
                module_state_len.append(len(s))
            state_len= np.sum(module_state_len)
            action_len = env.num_joints
            module_action_len = list(np.diff(env.action_indexes))
            module_sa_len = module_state_len+ module_action_len
            # create module containers for the nodes
            modules = []
            for i in range(n_modules):
                modules.append(pgnnc.Module(i, gnn_nodes[modules_types[i]], device))
            tau_list = []
            u_list = []
            last_states = [smm.to(device) for smm in to_tensors(env_state_init)]
            buttons = np.zeros(12)

    else:
        if step<50:
            desired_xyyaw =  np.array([1,0, 0])
            print('New heading: ' + str(desired_xyyaw))
        elif step<100:
            desired_xyyaw =  np.array([0,1, 0])
            print('New heading: ' + str(desired_xyyaw))
        elif step<150:
            desired_xyyaw =  np.array([0,0, 1])
            print('New heading: ' + str(desired_xyyaw))
        elif step<200:
            desired_xyyaw =  np.array([0.5,0.5, 1])
            print('New heading: ' + str(desired_xyyaw))
        else: # switch robot
            step = 0
            robot_ind = (robot_ind+1) % len(urdf_names)
            urdf_name = urdf_names[robot_ind]
            env.reset_terrain()
            env.reset_robot(urdf_name=urdf_name, randomize_start=False, start_xyyaw=[0,0,0])
            attachments = env.attachments
            modules_types = env.modules_types
            n_modules = len(modules_types)
            env_state_init = env.get_state()
            module_state_len = []
            for s in env_state_init:
                module_state_len.append(len(s))
            state_len= np.sum(module_state_len)
            action_len = env.num_joints
            module_action_len = list(np.diff(env.action_indexes))
            module_sa_len = module_state_len+ module_action_len
            # create module containers for the nodes
            modules = []
            for i in range(n_modules):
                modules.append(pgnnc.Module(i, gnn_nodes[modules_types[i]], device))
            tau_list = []
            u_list = []
            last_states = [smm.to(device) for smm in to_tensors(env_state_init)]
            buttons = np.zeros(12)
            print('New robot ' + urdf_name + ' and new heading: ' + str(desired_xyyaw))
            desired_xyyaw =  np.array([1,0, 0])


    chassis_yaw = env.pos_rpy[-1]
    vect1 = np.array([desired_xyyaw[0]*np.cos(chassis_yaw) - desired_xyyaw[1]*np.sin(chassis_yaw),
            desired_xyyaw[0]*np.sin(chassis_yaw) + desired_xyyaw[1]*np.cos(chassis_yaw),
            0] )
    vect2 = np.array([np.cos(desired_xyyaw[2]/2+chassis_yaw),
                      np.sin(desired_xyyaw[2]/2+chassis_yaw), 
                      0])*np.abs(desired_xyyaw[2])
    env.draw_body_arrows([vect1*0.5/vxy_scale, 
                          0.5*vect2/vyaw_scale],
                         [[0,0,0], [0,0,1]])

    
    env_state = env.get_state()
    states = [smm.to(device) for smm in to_tensors(env_state)]


    # heading and yaw here are wrt body frame
    goals = torch.tensor(desired_xyyaw, dtype=torch.float32, device=device).unsqueeze(0)

    node_inputs = create_control_inputs(last_states, goals, rotate_goals = False)
    

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
        action = torch.cat(u_out_mean,-1).squeeze().numpy()
        tau =  torch.cat(tau_out_mean,-1).squeeze().cpu().numpy()
        tau_list.append(tau)
        u_list.append(action)
    
    env.step(action)
    

    
    if np.dot([0,0,1], env.z_axis)<0:
        env_yaw = env.pos_rpy[-1]
        env.reset_robot(urdf_name=urdf_name, randomize_start=True)
        env_state = env.get_state()
        states = [smm.to(device) for smm in to_tensors(env_state)]
        
    if np.sqrt(env.pos_xyz[0]**2+ env.pos_xyz[1]**2)>2:
        env_state = env.get_state()
        env_state[0][0:2] = 0 
        env.set_state(env_state)
    
    last_states = [s.clone() for s in states]
    step+=1

# env.stop_video_log()



## Could plot out a set of joint torques or velocity commands
# import matplotlib
# import matplotlib.pyplot as plt

# # tau_array = np.stack(tau_list)
# # all_joint_inds = list(range(env.num_joints))
# # jnames = []
# # for i in all_joint_inds[1::3]:
# #     jnames.append(str(i))
# #     plt.plot(tau_array[:,i])
# # plt.legend(jnames)

# u_array = np.stack(u_list)
# all_joint_inds = list(range(env.num_joints))
# jnames = []
# for i in all_joint_inds:
#     jnames.append(str(i))
#     plt.plot(u_array[:,i])
# plt.legend(jnames)

# # Could convert video to slow down to correct speed with ffmpeg
# # import subprocess
# # cmd_status = subprocess.call(['ffmpeg', '-i',
# #                   vid_fname, '-filter:v',
# #                   'setpts=2.0*PTS',
# #                   '2x_' + vid_fname, '-y'])






