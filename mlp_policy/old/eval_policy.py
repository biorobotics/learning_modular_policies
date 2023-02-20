'''
evaluate how well the policy is working compared to the optimized trajectories

This version uses a cost function that updates every 10 steps, like the costs used 
in the mpc planning stage.

'''

import torch
import numpy as np
import pgnn_control as pgnnc
from utils import get_sampleable_inds, sample_memory, wrap_to_pi
import os
# cwd = os.path.dirname(os.path.realpath(__file__))
from robot_env import robot_env
from utils import to_tensors, combine_state, wrap_to_pi, rotate
from planning_utils import get_pos_control_inds, create_cost_mats2

# device = torch.device("cpu")

# compute the quadratic cost of the trajectory with slew rate
def cost_fn(states, actions, C, c, slew_rate_penalty):
    T = states.shape[0]
    batch_size = states.shape[1]
    # quadratic cost
    actions0 = torch.cat([actions, torch.zeros_like(actions[0]).unsqueeze(0)],0)
    sa = torch.cat((states,actions0),-1)
    sa_C_sa = torch.matmul(torch.matmul(sa.unsqueeze(1),C), sa.unsqueeze(-1))
    c_cs = torch.matmul(c.unsqueeze(1), sa.unsqueeze(-1))
    cost = (0.5*sa_C_sa + c_cs).sum()
    # slew rate cost
    cost_slew = 0.5*(actions[1:,:] - actions[:-1,:]).pow(2).sum(dim=0).sum(dim=-1) 
    cost += cost_slew*slew_rate_penalty
    return cost

# evaluate the policy performance on a single design
def eval_policy(urdf, trial_costs,
    modules,
    run_lens, states_memory, actions_memory, goal_memory,
    step_memory, slew_rate_penalty, device,
    n_runs_max = 100):
    
    env = robot_env(show_GUI = False)
    # env = robot_env(show_GUI = True)
    env.reset_terrain()
    env.reset_robot(urdf_name=urdf, randomize_start=False)

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



    trial_costs = []
    n_runs = len(run_lens)
    if n_runs_max < n_runs:
        n_runs = n_runs_max

    leg_pos_inds, leg_control_inds, wheel_steer_inds, wheel_control1_inds, wheel_control2_inds = get_pos_control_inds(
            modules_types, module_state_len, module_action_len)
    n_exec = 10 # how many steps between the cost function changes
    env_state_init = combine_state(to_tensors(env_state_init)).to(device)

    T = n_exec
    batch_size = 1
    n_state, n_ctrl = state_len, action_len
                
    for run_index in range(n_runs):
    # if True:
    #     run_index = 80
        cost_plan_total = 0
        cost_policy_total = 0
        last_u_policy = None
        last_u_plan = None
        
        # sample a run from the memory
        run_index = np.random.choice(n_runs)
        # check all
        sampled_run = run_index
        run_len = run_lens[sampled_run]
        states_run = states_memory[sampled_run]
        actions_run = actions_memory[sampled_run]
        goals_run = goal_memory[sampled_run]
        steps_run = step_memory[sampled_run]
        state_start = [s[0] for s in states_run]
    #     print('started at state ' + str(state_start))
        states_planned = combine_state(states_run)
        actions_planned = combine_state(actions_run)

        env.set_state(state_start)

        # create a trial, where the policy is simulated
        states_policy_list = []
        actions_policy_list = []

        for t in range(run_len-1):
            goals_world = goals_run[:,t].unsqueeze(0)
            env_state = env.get_state()
            states = [smm.to(device) for smm in to_tensors(env_state)]
            state_policy = combine_state(states) 
            states_policy_list.append(state_policy )
            
            if np.mod(t, n_exec)==0: # happens at t=0, 10, ... 50
                if t>0: # add on cost of most recent trajectory segment
                    actions_policy = torch.stack(actions_policy_list,0)
                    states_policy = torch.cat(states_policy_list,0)
                    cost_plan = cost_fn(states_planned[t-n_exec:t,:],
                                        actions_planned[t-n_exec:t-1,:],
                                        C_plan.squeeze(1), c_plan.squeeze(1), 
                                        slew_rate_penalty)
                    cost_policy = cost_fn(states_policy[t-n_exec:t,:],
                                        actions_policy[t-n_exec:t-1,:],
                                        C_policy.squeeze(1), c_policy.squeeze(1), 
                                        slew_rate_penalty)
    #                 print(cost_plan, cost_policy)
                    cost_plan_total = cost_plan_total + cost_plan
                    cost_policy_total = cost_policy_total + cost_policy

                
                # compute cost function for simulated policy and sampled trial
                delta_xyyaw_des = goals_world # goal is constant over full run in mpc dataset 
                state_start = [s[t] for s in states_run]
                xyyaw_start_plan = state_start[0][[0,1,5]].detach().cpu().unsqueeze(0)
                xyyaw_start_policy = state_policy[0][[0,1,5]].detach().cpu().unsqueeze(0)
    #             print('xyyaw_start_plan ' + str(xyyaw_start_plan))
    #             print('xyyaw_start_policy ' + str(xyyaw_start_policy))
                

                C_plan, c_plan = create_cost_mats2(
                                     [steps_run[t]], device, T, batch_size, env,
                                     env_state_init, n_state, n_ctrl,
                                     leg_pos_inds, leg_control_inds,
                                     wheel_steer_inds, wheel_control1_inds, wheel_control2_inds,
                                     last_u = last_u_plan, 
                                     slew_rate_penalty = slew_rate_penalty,
                                     xyyaw_start = xyyaw_start_plan, 
                                     delta_xyyaw_des = delta_xyyaw_des )
                C_policy, c_policy = create_cost_mats2(
                                     [steps_run[t]], device, T, batch_size, env,
                                     env_state_init, n_state, n_ctrl,
                                     leg_pos_inds, leg_control_inds,
                                     wheel_steer_inds, wheel_control1_inds, wheel_control2_inds,
                                     last_u = last_u_policy, 
                                     slew_rate_penalty = slew_rate_penalty,
                                     xyyaw_start = xyyaw_start_policy, 
                                     delta_xyyaw_des = delta_xyyaw_des )



            ### change to body frame the goal heading and state
            # goals_world[x,y] are recorded in world frame. shift to body frame here.
            # (TODO Note: could preprocess this part)
            chassis_state = states[0]
            chassis_yaw = chassis_state[:,5]
            sin0 = torch.sin(chassis_yaw)
            cos0 = torch.cos(chassis_yaw)
            z0 = torch.zeros_like(cos0)
            R0_t = torch.stack( [ torch.stack([cos0, sin0, z0]),
                              torch.stack([-sin0,  cos0, z0]),
                              torch.stack([z0, z0,   torch.ones_like(cos0)])
                              ]).permute(2,0,1)
            # form input as zrp+ [vx,vy,vz,wx,wy,wz]_Body + [q, qdot] 
            chassis_state_body = torch.cat([chassis_state[:,2:5],
                                    rotate(R0_t,chassis_state[:,6:9]),
                                    rotate(R0_t,chassis_state[:,9:12])],-1)
            node_inputs = [chassis_state_body] + states[1:]
            R0_t_xy = torch.stack( [ torch.stack([cos0, sin0]),
                                     torch.stack([-sin0,cos0])]).permute(2,0,1)
            goals_body0 = rotate(R0_t_xy, goals_world[:,0:2])
            goals_body1 = wrap_to_pi(goals_world[:,-1]) # probably don't need to actually wrap to pi, but for safety I do anyway
            # goals_world[-1] is a delta for turn angle 
            goals = torch.cat([goals_body0, 
                               goals_body1.unsqueeze(1)
                              ],-1)    

            ### pass into control network
            # overwrite v_xyz with zeros to effectively remove it from input
            # node_inputs[0][3:6] = 0

            # # overwrite z with zero to remove if effectively
            # node_inputs[0][0] = 0 
            
            # remove v_xyz, its noisy and prone to drift
            node_inputs[0] = torch.cat([node_inputs[0][:,:3], node_inputs[0][:,6:]],1)

            # remove z, its hard to estimate and not critical
            node_inputs[0] = node_inputs[0][:,1:]


            for module in modules: # this prevents the LSTM in the GNN nodes from 
                # learning relations over time, only over internal prop steps.
                module.reset_hidden_states(1) 

            with torch.no_grad():
                node_inputs[0] = torch.cat([node_inputs[0], goals],1)
                out_mean, out_var = pgnnc.run_propagations(
                    modules, attachments, 2, node_inputs, device)
                u_out_mean = []
                tau_out_mean = []
                for mm in range(n_modules):
                    u_out_mean.append(out_mean[mm][:,:module_action_len[mm]])
                    tau_out_mean.append(out_mean[mm][:,module_action_len[mm]:])
                u_combined = torch.cat(u_out_mean,-1).squeeze()
                u_np = u_combined.numpy()

            actions_policy_list.append(u_combined)   
            last_u_policy = u_combined.unsqueeze(0)
            last_u_plan = actions_planned[t,:].unsqueeze(0)
            
            # execute control action
            env.step(u_np)
            
            if (t<run_len-1) and env.show_GUI:
                xyz_before = states_run[0][t,0:3].numpy()
                xyz_after = states_run[0][t+1,0:3].numpy()

                env.draw_line( [xyz_before[0],xyz_before[1],0.01],
                       [xyz_after[0], xyz_after[1],0.01],
                         color=[0,0,0])



        print('Plan vs Policy cost:' + str(np.round(cost_plan_total.item()))
              + ', ' +
             str(np.round(cost_policy_total.item())))

        trial_costs.append([cost_plan_total.item(), cost_policy_total.item()])

    # return trial_costs



