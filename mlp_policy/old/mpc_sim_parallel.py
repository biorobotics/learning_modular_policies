#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''
Use simulation as model to do mpc
'''
import torch
from robot_env import robot_env
import numpy as np
import os
import sys
import warnings
from mpc import mpc
from mpc.mpc import QuadCost
from planning_utils import get_pos_control_inds, create_cost_mats2
from planning_utils import fd_func_jac, fd_func_autodiff
from utils import combine_state, to_tensors, divide_state
from copy import copy

if __name__ == '__main__':
    device = torch.device('cpu')
    urdf = 'llllll'
    show_GUI = True
    env = robot_env(show_GUI = show_GUI)
    env.reset_terrain()
    env.reset_robot(urdf_name=urdf, randomize_start=False)

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


    num_processes = 2
    pool = torch.multiprocessing.Pool(processes=num_processes)
    envs_fd = []
    for i in range(sum(module_sa_len)):
        env_fd = robot_env(show_GUI = False)
        env_fd.reset_terrain()
        env_fd.reset_robot(urdf_name=urdf, randomize_start=False)
        envs_fd.append(env_fd)


    def div_state(x_in, module_state_len):
        x_out  = []
        ind = 0
        for i in range(len(module_state_len)):
            l = module_state_len[i]
            x_out.append(x_in[ind:ind+l])
            ind+=l
        return x_out

    class fd_func_sim_jac(torch.nn.Module):
        # pass as input and output state and action without outer wrapping
        def __init__(self, env_fdjac):
            super(fd_func_sim_jac, self).__init__()
            self.finite_diff_delta = 1e-3
            self.env_jac = env_fdjac

        def forward(self, state, action):

            batch_size = state.shape[0]
            n_state = state.shape[1]
            state_next = torch.zeros((batch_size, n_state))
            for ib in range(batch_size):
                state0 = state[ib,:].detach().numpy()
                action0 = action[ib,:].detach().numpy()
                state_div = div_state(state0, self.env_jac.module_state_len)
                self.env_jac.set_state(state_div)
                self.env_jac.step(action0)
                state_next[ib,:] = combine_state(to_tensors(self.env_jac.get_state()))
            
    #         print(state_next.shape)
            return state_next

        def forward_env(self, state, action, env_now):

            batch_size = state.shape[0]
            n_state = state.shape[1]
            state_next = torch.zeros((batch_size, n_state))
            for ib in range(batch_size):
                state0 = state[ib,:].detach().numpy()
                action0 = action[ib,:].detach().numpy()
                state_div = div_state(state0, env_now.module_state_len)
                env_now.set_state(state_div)
                env_now.step(action0)
                state_next[ib,:] = combine_state(to_tensors(env_now.get_state()))
            
    #         print(state_next.shape)
            return state_next


        def grad_input(self, state, action):
            delta = self.finite_diff_delta

    #         print('grad_input')
    #         print(str(state.shape),str(action.shape))
            batch_size = state.shape[0]

            n_state = state.shape[-1]
            n_ctrl = action.shape[-1]
            dfds = torch.zeros((batch_size,n_state,n_state), dtype=torch.float32)
            dfda = torch.zeros((batch_size,n_state,n_ctrl), dtype=torch.float32)
            for ib in range(batch_size):

                state_now = state[ib,:].detach().numpy()
                action_now = action[ib,:].detach().numpy()
                self.env_jac.set_state(
                    div_state(state_now, self.env_jac.module_state_len))
                self.env_jac.step(action_now)
                state_next = combine_state(to_tensors(self.env_jac.get_state()))

                pool_inputs = []

                for i in range(n_state):
                    state_perturbed = copy(state_now)
                    state_perturbed[i] += delta
                    # self.env_jac.set_state(
                    #     div_state(state_perturbed, self.env_jac.module_state_len))
                    # self.env_jac.step(action_now)
                    pool_inputs.append([state_perturbed,action_now, envs_fd[i]])


                    # state_perturbed = copy(state_now)
                    # state_perturbed[i] -= delta
                    # self.env_jac.set_state(
                    #     div_state(state_perturbed, self.env_jac.module_state_len))
                    # self.env_jac.step(action_now)
                    # state_next_perturbed2 = combine_state(to_tensors(self.env_jac.get_state()))
                    # dfds[ib,:,i] = (state_next_perturbed -
                    #                              state_next_perturbed2)/(2*delta)

                for i in range(n_ctrl):
                    action_perturbed = copy(action_now)
                    action_perturbed[i] += delta
                    # self.env_jac.set_state(
                    #     div_state(state_now, self.env_jac.module_state_len))
                    # self.env_jac.step(action_perturbed)
                    pool_inputs.append([state_now,
                                action_perturbed, envs_fd[n_state+i]])

                    # action_perturbed = copy(action_now)
                    # action_perturbed[i] -= delta
                    # self.env_jac.set_state(
                    #     div_state(state_now, self.env_jac.module_state_len))
                    # self.env_jac.step(action_perturbed)
                    # state_next_perturbed2 =combine_state(to_tensors(self.env_jac.get_state()))
                    # dfda[ib,:,i] = (state_next_perturbed -
                    #                              state_next_perturbed2)/(2*delta)

                pool_out = pool.starmap(self.forward_env, pool_inputs)
                print('joining pool')
                pool.join()
                print('after join pool')
                for i in range(n_state):
                    state_next_perturbed = pool_out[i]
                    dfds[ib,:,i] =  (state_next_perturbed -  state_next)/delta
                for i in range(n_ctrl):
                    state_next_perturbed = pool_out[n_state+i]
                    dfda[ib,:,i] =  (state_next_perturbed -  state_next)/delta


            return dfds, dfda
        
    env_jac = robot_env(show_GUI = False)
    env_jac.reset_terrain()
    env_jac.reset_robot(urdf_name=urdf, randomize_start=False)


    # In[3]:


    import time
    res_times = []



    # mpc-style control parameters
    n_envs = 1
    batch_size, n_state, n_ctrl = n_envs, state_len, action_len
    T = 20 # T is the planning horizon

    n_replans = 4
    n_execute = 10 # number of steps to execute after each replan
    total_steps = n_replans*n_execute
    slew_rate_penalty = 40 

    leg_pos_inds, leg_control_inds, wheel_steer_inds, wheel_control1_inds, wheel_control2_inds = get_pos_control_inds(
        modules_types, module_state_len, module_action_len)

    env_state_init = combine_state(to_tensors(env_state_init)).to(device)

    # The upper and lower control bounds. All ones since environment rescales them.
    u_lower = -torch.ones(T, batch_size, n_ctrl, device=device)
    u_upper =  torch.ones(T, batch_size, n_ctrl, device=device)


    gradient_method = mpc.GradMethods.ANALYTIC # actually does finite diff, we rewrote it for finer control


    dt = env.dt
    speed_scale_yaw = (T*dt)*np.pi/2
    speed_scale_xy = (T*dt)

    # create the test direction goals.
    # these will be popped and used as the goal directions
    # on the first few planning runs.
    test_goals = [[speed_scale_xy*0.75,0,0],
                  [0,speed_scale_xy*0.75,0],
                  [0,0,speed_scale_yaw*0.75],
                  [-speed_scale_xy*0.75,0,0],
                  [0,-speed_scale_xy*0.75,0],
                  [0,0,-speed_scale_yaw*0.75]]
    n_runs = 1

    # states_memory = []
    # actions_memory = []
    # torques_memory = []
    # run_lens = []
    # goal_memory = []
    # step_memory = [] # keep track of the time step, so that the tripod can be regulated
    envs = [env]
    for i_traj in range(n_runs):

        u_init = torch.zeros(T, batch_size, n_ctrl, device=device)

        t_list = [ [] for i in range(n_envs) ]
        state_list = [ [] for i in range(n_envs) ]
        # state_list_tensors =[ [] for i in range(n_envs) ]
        goal_tensors =[ [] for i in range(n_envs) ]
        action_list = [ [] for i in range(n_envs) ]
        torques_list = [ [] for i in range(n_envs) ]
        last_u = None


        delta_xyyaw_des = np.array(test_goals)

        sampled_steps = np.zeros(n_envs)
        
        for i in range(len(envs)):

     # set the robot to default zero pose
            envs[i].reset_robot(urdf_name = urdf, randomize_start=False,
                start_yaw = 0)

            env_state = envs[i].get_state()
            env_state_combined = combine_state(to_tensors(env_state))
            state_list[i].append(env_state)
            # state_list_tensors[i].append(env_state_combined)

            # get initial torques 
            tau = envs[i].joint_torques / envs[i].moving_joint_max_torques
            tau_div =  envs[i].divide_action_to_modules(tau)
            torques_list[i].append(tau_div)

            goal_tensors[i].append(torch.tensor(delta_xyyaw_des[i,:], dtype=torch.float32))
                

            # track whether each sim env robot has flipped over, and stop tracking it if so
            robot_alive = [True]*n_envs

            # print('Replan ', end = '')
            for replan in range(n_replans):
                start_time = time.time()

                if replan ==0:
                    # lqr_iter = 10 # need larger for initial soln
                    lqr_iter = 15 # ?
                else:
                    # lqr_iter = 5 # too high makes lqr take too long
                    lqr_iter = 8 # ?

                # print( str(replan), end = ',')

                x_init = []
                for i in range(n_envs):
                    env_state_i = to_tensors(envs[i].get_state())
                    x_init.append( combine_state(env_state_i))
                x_init = torch.cat(x_init,0).to(device)


                start_steps = np.array([replan*n_execute]*batch_size) + sampled_steps
                C, c =create_cost_mats2(start_steps, device, T, batch_size, env,
                             env_state_init, n_state, n_ctrl,
                             leg_pos_inds, leg_control_inds,
                             wheel_steer_inds, wheel_control1_inds, wheel_control2_inds,
                             last_u = last_u, slew_rate_penalty = slew_rate_penalty,
                             xyyaw_start = x_init[:,[0,1,5]].detach().cpu(), 
                             delta_xyyaw_des = delta_xyyaw_des  )

                with torch.no_grad():
                    print('starting mpc')
                    x_lqr, u_lqr, objs_lqr = mpc.MPC(
                        n_state=n_state,
                        n_ctrl=n_ctrl,
                        T=T,
                        u_lower=u_lower, 
                        u_upper=u_upper,
                        u_init = u_init,
                        lqr_iter=lqr_iter,
                        grad_method=gradient_method,
                        verbose=1,
                        backprop=False,
                        exit_unconverged=False,
                        slew_rate_penalty=slew_rate_penalty
                    )(x_init, QuadCost(C, c), 
                        fd_func_sim_jac(env_jac)
                     )
                    
                end_time = time.time()
                time_delta = end_time-start_time
                print(time_delta)
                res_times.append(time_delta)


                # check for nans
                if torch.isnan(u_lqr).any():
                    print('NaN detected')
                    break


                for i in range(n_envs):            
                    for t in range(n_execute):
                        if robot_alive[i]:
                            xyz_before = envs[i].pos_xyz 
                            u = u_lqr[t,i,:]
                            u_np = u.detach().cpu().numpy()
                            envs[i].step(u_np)
                            env_state = envs[i].get_state()
                            state_list[i].append(env_state)
                            goal_tensors[i].append(torch.tensor(delta_xyyaw_des[i,:], dtype=torch.float32))
                            u_div = envs[i].divide_action_to_modules(u_np)
                            action_list[i].append(u_div)
                            # scale by max torque
                            tau = envs[i].joint_torques / envs[i].moving_joint_max_torques
                            tau_div =  envs[i].divide_action_to_modules(tau)
                            torques_list[i].append(tau_div)
                            xyz_after = envs[i].pos_xyz 
                            rpy_after = envs[i].pos_rpy 
                            if envs[0].show_GUI:
                                # draw line for where robot has gone
                                if i==0:
                                    line_color = [0,0,0]
                                else:
                                    line_color = [1,0,0]
                                envs[0].draw_line( [xyz_before[0],xyz_before[1],0.01],
                                           [xyz_after[0], xyz_after[1],0.01],
                                             color=line_color)
                                # draw line for its desired heading
                                vxy_scale = 2
                                vyaw_scale = 1.5
                                desired_xyyaw = delta_xyyaw_des[0,:]
                                vect1 = np.array([desired_xyyaw[0],
                                        desired_xyyaw[1],
                                        0] )
                                vect2 = np.array([np.cos(desired_xyyaw[2]),
                                                  np.sin(desired_xyyaw[2]), 
                                                  0])*np.abs(desired_xyyaw[2])
                                envs[0].draw_body_arrows([vect1*0.5/vxy_scale, 
                                                      0.5*vect2/vyaw_scale],
                                                     [[0,0,0], [0,0,1]])

                            # stop stepping this robot if it flipped over       
                            if np.dot([0,0,1], envs[i].z_axis)<0:
                                robot_alive[i] = False

                    last_u = u_lqr[t] # set for use in next slew rate cost


                    # scroll through long term initialization 
                    ind = replan*n_execute
                    u_init[T-n_execute:,:,:] = 0 

                    # use the remainder of the control inputs to warm start the next replan
                    u_init[:(T-n_execute),:,:] = u_lqr[n_execute:,:,:].detach()

    pool.close()

    # In[4]:


    print(len(state_list))
    print(len(state_list[0]))
    print(len(state_list[0][0]))
    chassis_states = [s[0] for s in state_list[0]]
    print(len(chassis_states))
    chassis_states = np.vstack(chassis_states)
    print('End state', chassis_states[-1,0])
    print('Total time', np.sum(res_times))

    # results:
    # End state 0.8818973398891625
    # Total time 1284.1354298591614


    # In[ ]:




