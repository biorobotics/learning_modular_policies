#!/usr/bin/env python
# coding: utf-8
'''
Source code for paper "Learning modular robot control policies" in Transactions on Robotics
Julian Whitman, Dec. 2022. 


'''

# load libraries
import torch
from robot_env import robot_env
import numpy as np
import pgnn_control as pgnnc
from utils import to_tensors, combine_state, wrap_to_pi, rotate, create_control_inputs
import cProfile, pstats, io

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

# init env
# use the env to get the size of the inputs and outputs
env = robot_env(show_GUI = False)
# env = robot_env(show_GUI = True)
env.reset_terrain()

folder = 'saved/with_tripod1/'
# folder = 'saved/no_tripod/'
PATH =  'multidesign_control_iter3.pt'
PATH = folder + PATH

save_dict = torch.load( PATH, map_location=lambda storage, loc: storage)
urdf_names = save_dict['urdf_names']
print(urdf_names)
urdf_name = 'wnwwnw'
# urdf_name = 'llllll'

gnn_state_dict= save_dict['gnn_state_dict'] 
internal_state_len = save_dict['internal_state_len'] 
message_len = save_dict['message_len'] 
hidden_layer_size = save_dict['hidden_layer_size']
goal_len =3

# print(save_dict['comment'])

env.reset_robot(urdf_name=urdf_name, randomize_start=False)

attachments = env.attachments
modules_types = env.modules_types
# print('attachments: ' + str(attachments))
# print('modules_types: ' + str(modules_types))
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



# env.p.resetDebugVisualizerCamera(2.1,0,-65,[0,0,0.2],physicsClientId=env.physicsClient) 
# env.p.resetDebugVisualizerCamera(2.1,0,-89.999,[0,0,0.2],physicsClientId=env.physicsClient) 
env.reset_robot(urdf_name=urdf_name, randomize_start=False, start_xyyaw=[0,0,0])
# vid_fname=urdf_name+'_steer_vel.mp4'
# env.start_video_log(fileName=vid_fname)

# robot_button_mapping = [0,1,2,3,4,8,9,10]
# n_buttons_to_check = len(robot_button_mapping)
vxy_scale = 20*(20/240)*0.314/0.75
vyaw_scale = 20*(20/240)*1.1/0.75
robot_ind = 0
env_state = env.get_state()
last_states = [smm.to(device) for smm in to_tensors(env_state)]
tau_list = []
u_list = []
step = 0
desired_xyyaw =  np.array([1,0, 0])


pr = cProfile.Profile()
pr.enable()

while step<500:

    chassis_yaw = env.pos_rpy[-1]
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
    
    
    last_states = [s.clone() for s in states]
    step+=1


pr.disable()
s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
ps.print_stats()

with open(urdf_name+'_profile.txt', 'w+') as f:
    f.write(s.getvalue())
