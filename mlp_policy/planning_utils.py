'''
Source code for paper "Learning modular robot control policies" in Transactions on Robotics
MLP comparisons
Julian Whitman, Dec. 2022. 

'''

import torch
import numpy as np
from utils import divide_state, divide_action, state_to_fd_input, wrap_to_pi
from utils import from_body_frame_batch, state_add_batch, combine_state

# define the cost weights here so that they can be consistently referenced by any function
w_control_cost = 0.01
w_z_height = 5
z_target = 0.23 # meters, target z height
w_roll = 30 # up from 15
w_pitch = 25
w_yaw_track = 25
w_xy_track = 110 # up from 35 10/29/2020, helps move them forward faster
w_wheel_steer_angle = 0.4
# w_wheel_steer_angle = 0.75

w_tripod = 5 # track alternating tripod, using env function
w_center = 0 # penalize leg angles far from center

# w_tripod = 0
# w_center = 3

tripod_amplitude = 0.65
tripod_period = 1.25 # used this up to 10/28
# tripod_period = 1.15 # slightly faster

slew_rate_penalty = 7


# base the max speed possible on max speed of wheels, with a small addition (1./0.75)=1.33 factor
# wheel max angular vel is 3.14, radius 0.1
# max linear speed of car is then v = omega*r = 0.314 m/s 
# in T*dt seconds that is 0.314*T*dt m
# body radius is around 0.1760m (chassis port)+ 0.11m (wheel module) = 0.286m
# wheel contact linear speed 0.314 m/s as above
# omega_body = v_wheel/r_body = 0.314m/s /0.286m = 1.1rad/s
# in T*dt seconds that is 1.1*T*dt
T= 20
dt = 20./240.
speed_scale_xy = (T*dt)*0.314*(1./0.75)
speed_scale_yaw = (T*dt)*1.1*(1./0.75)
# these define the max commanded body displacements per time span

cost_weights = dict()
cost_weights['w_control_cost']=w_control_cost
cost_weights['w_z_height']=w_z_height
cost_weights['z_target']=z_target

cost_weights['w_roll']=w_roll
cost_weights['w_pitch']=w_pitch
cost_weights['w_yaw_track']=w_yaw_track
cost_weights['w_xy_track']=w_xy_track
cost_weights['w_wheel_steer_angle']=w_wheel_steer_angle
cost_weights['w_tripod']=w_tripod
cost_weights['w_center']=w_center
cost_weights['slew_rate_penalty']=slew_rate_penalty
cost_weights['tripod_amplitude']=tripod_amplitude
cost_weights['tripod_period']=tripod_period
cost_weights['speed_scale_xy']=speed_scale_xy
cost_weights['speed_scale_yaw']=speed_scale_yaw



def create_cost_mats2(start_steps, device_mpc, T, batch_size, env,
                     env_state_init, n_state, n_ctrl,
                     leg_pos_inds, leg_control_inds,
                     wheel_steer_inds, wheel_control1_inds, wheel_control2_inds,
                     last_u = None, slew_rate_penalty = slew_rate_penalty,
                     xyyaw_start = [0,0,0], delta_xyyaw_des = [0,0,0] ):
    # start step is an int used to find an index in desired_states
    # start_state is a combined state tensor [batch_size x state_len]
    # last_u must be [batch_size x n_cntrl]
    #  minimizes:    sum_t 0.5 tau_t^T C_t tau_t + c_t^T tau_t
    # Note the 0.5 before the C term!

    # penalize distance from z=2, via
    # penalize (z-cz)**2 --> z**2 - 2*cz*z + cz**2 
    # --> C = wz*2, c = -2*cz*wz
    # cz = 2,     # wz = 2
    # quad_diag[2] = wz*2
    # lin_part[2] = -2*cz*wz
    n_sc = n_state + n_ctrl


    C = torch.zeros(T, batch_size, n_sc, n_sc, device=device_mpc)
    c = torch.zeros(T, batch_size, n_sc,   device=device_mpc)
    for j in range(batch_size):

        # break up  deltaxy_des into interpolated parts
        x_des = np.interp(range(T), [0,T-1], 
            [xyyaw_start[j,0], xyyaw_start[j,0]+delta_xyyaw_des[j,0]])
        y_des = np.interp(range(T), [0,T-1], 
            [xyyaw_start[j,1], xyyaw_start[j,1]+delta_xyyaw_des[j,1]])
        yaw_des = np.interp(range(T), [0,T-1], 
            [xyyaw_start[j,2], xyyaw_start[j,2]+delta_xyyaw_des[j,2]])


        # compute what fraction of the max displacement is being commanded
        # in this element of the batch. used to scale up gait-style amplitude
        # dx_max_frac = delta_xyyaw_des[j,0]/(0.75*speed_scale_xy)
        # dy_max_frac = delta_xyyaw_des[j,1]/(0.75*speed_scale_xy)
        # dyaw_max_frac = delta_xyyaw_des[j,2]/(0.75*speed_scale_yaw)
        # max_frac = np.max([dx_max_frac,dy_max_frac,dyaw_max_frac])
        # # rescale so that max_frac is max 1. this means that for 75% max speed and up it will be at max amplitude
        # max_frac = np.clip(max_frac, 0, 1)
        # print('Max frac: '  + str(max_frac))

        for i in range(T):
            quad_diag = torch.zeros(n_sc, device=device_mpc)
            lin_part = torch.zeros(n_sc, device=device_mpc)
            
            quad_diag[n_state:] = w_control_cost # penalty on controls
            
            # lower wheel spinning control
            # quad_diag[n_state + np.array(wheel_control1_inds)] *= 0.25

            # alter wheel steering penalty
            # quad_diag[n_state + np.array(wheel_control2_inds)] *= 0.75
        
            # track x
            z_d = torch.tensor(x_des[i], dtype=torch.float32, device=device_mpc)
            quad_diag[0] += w_xy_track*2
            lin_part[0] += -2*z_d*w_xy_track

            # track y
            z_d = torch.tensor(y_des[i], dtype=torch.float32, device=device_mpc)
            quad_diag[1] += w_xy_track*2
            lin_part[1] += -2*z_d*w_xy_track
        
            # keep z height near 0.25
            z_d = torch.tensor(z_target, dtype=torch.float32, device=device_mpc)
            quad_diag[2] += w_z_height*2
            lin_part[2] += -2*z_d*w_z_height

            # keep roll near 0
            z_d = torch.tensor(0, dtype=torch.float32, device=device_mpc)
            quad_diag[3] += w_roll*2
            lin_part[3] += -2*z_d*w_roll

            # keep pitch near 0
            z_d = torch.tensor(0, dtype=torch.float32, device=device_mpc)
            quad_diag[4] += w_pitch*2
            lin_part[4] += -2*z_d*w_pitch
            
            # track yaw
            z_d = torch.tensor(yaw_des[i], dtype=torch.float32, device=device_mpc)
            quad_diag[5] += w_yaw_track*2
            lin_part[5] += -2*z_d*w_yaw_track
            

            
            # keep wheel steer near zero
            z_d = env_state_init[0,wheel_steer_inds]
            quad_diag[wheel_steer_inds] += w_wheel_steer_angle*2
            lin_part[wheel_steer_inds] += -2*z_d*w_wheel_steer_angle
            
            # track alternating tripod, using env function
            # scale weight to number of legs
            # w_tripod = int(len(leg_pos_inds)/3.)
            # w_tot = 2*w_xy_track + w_yaw_track


#             pos_legs = env.alt_tripod_positions(
#                 env.dt*(start_step+i), amplitude = 0.6, period = 1.25)
            # pos_legs = pos_func(start_steps[j] + i)
            step = start_steps[j] + i
            pos_legs = env.alt_tripod_positions(
                    env.dt*step, 
                    # amplitude = tripod_amplitude*max_frac, 
                    amplitude = tripod_amplitude, 
                    period = tripod_period)

            pos_legs = np.concatenate(pos_legs)
            x_d = torch.tensor(pos_legs, dtype=torch.float32, device=device_mpc)
            quad_diag[leg_pos_inds] += w_tripod*2
            lin_part[leg_pos_inds] += -2*x_d*w_tripod

            # scale center so it drops with number of legs
            # w_center = 6 - w_tripod

            # penalize leg angles far from center
            x_d = env_state_init[0,leg_pos_inds]
            quad_diag[leg_pos_inds] += w_center*2
            lin_part[leg_pos_inds] += -2*x_d*w_center


            # reduce for first joint
            quad_diag[leg_pos_inds[::3]] *= 0.5
            lin_part[leg_pos_inds[::3]] *= 0.5
            # w0 = 3
            # x_d = torch.tensor(pos_legs[::3], dtype=torch.float32, device=device_mpc)
            # quad_diag[leg_pos_inds[::3]] = w0*2
            # lin_part[leg_pos_inds[::3]] = -2*x_d*w0

            C[i, j,  :, :] = torch.diag(quad_diag)
            c[i, j,  :] = lin_part # reward every x

    
    # slew rate from last control input
    # || u - u_last||^2_2 =  sum( (u_i-u_last_i)^2 )
    # = sum( u_i^2 - 2*u_i*u_last_i + u_last_i^2 ) 
    if (last_u is not None) and (slew_rate_penalty>0):
        for j in range(batch_size):
            quad_diag = torch.zeros(n_sc, device=device_mpc)
            lin_part = torch.zeros(n_sc, device=device_mpc)
            quad_diag[n_state:] =  2*slew_rate_penalty
            lin_part[n_state:] =  -2*slew_rate_penalty*last_u[j,:]
            C[0, j,  :, :] += torch.diag(quad_diag)
            c[0, j,  :] += lin_part 
    
    return C, c
        
        
def compare_velocities(states_memory, goal_memory, run_lens,n_execute, T ):
    metric_totals = []
    metric_baselines = []
    for run_choice in range(len(run_lens)):
        metric_runs = []
        metric_baseline = []

        # take the run only chassis module is needed
        states_chassis = states_memory[run_choice][0]
        goals = goal_memory[run_choice]
        run_len = run_lens[run_choice]
        # for each block of n_exec time steps,
        # compute the waypoints desired and the distance from them


        for i in range(run_len-1):
            # (Assumes that all the goals in an n_exec window are the same.)
            # Every n_exec steps, compute the next few desired waypoints,
            # and find difference of the achieved from desired


            # at the last few steps it might not be a full n_exec.
            # compute dist travelled in that interval
            # If the there are not enough time steps left in the run for 
            # a full interval, do not attempt to calculate the velocity match.
            if np.mod(i, n_execute)==0 and not(i+n_execute >=run_len):
                interval_len = n_execute
        
                # dx_des / dy_des / dyaw_des is how far we wanted to go in n_exec steps
                dx_des = np.interp(interval_len, [0,T-1], 
                    [0, goals[0,i]])
                dy_des = np.interp(interval_len, [0,T-1], 
                    [0, goals[1,i]])
                dyaw_des = np.interp(interval_len, [0,T-1], 
                    [0, goals[2,i]])

                # dx / dy / dyaw is how far we actually went in n_exec steps
                dx = states_chassis[(i+interval_len),0] - states_chassis[i,0]
                dy = states_chassis[(i+interval_len),1] - states_chassis[i,1]
                dyaw = states_chassis[(i+interval_len),5] - states_chassis[i,5]
                dyaw = wrap_to_pi(dyaw) # in case it went over the pi/-pi boundary

                dx_diff = (dx.numpy() - dx_des)
                dy_diff = (dy.numpy() - dy_des)
                dyaw_diff = (dyaw.numpy() - dyaw_des)

                # what if the robot hadn't moved at all? what is the baseline 
                dx_baseline = dx_des
                dy_baseline = dy_des
                dyaw_baseline = dyaw_des
                

                # weighting factors should match what was used during planning,
                # since this defines the relative importance of each direction
                w_tot = 2*w_xy_track + w_yaw_track

                # print(i,(i+interval_len), dx_des, dy_des,dyaw_des,dx,dy,dyaw )
                # print(w_xy_track,w_yaw_track, w_tot)
                # print(dx_diff,dy_diff,dyaw_diff)

                metric_runs.append( 
                    np.sqrt( (w_xy_track*dx_diff/w_tot)**2 + 
                             (w_xy_track*dy_diff/w_tot)**2 + 
                             (w_yaw_track*dyaw_diff/w_tot)**2 ))
                metric_baseline.append( 
                    np.sqrt( (w_xy_track*dx_baseline/w_tot)**2 + 
                             (w_xy_track*dy_baseline/w_tot)**2 + 
                             (w_yaw_track*dyaw_baseline/w_tot)**2 ))
        # print('metric_runs', metric_runs)
        # print('metric_baseline', metric_baseline)
        if len(metric_runs)>0:
            metric_totals.append( np.mean( metric_runs) )
            metric_baselines.append(  np.mean(metric_baseline))

        # print('total: ', np.mean(metric_totals))
    return np.mean(metric_totals), np.mean(metric_baselines)
    






def get_pos_control_inds(modules_types, module_state_len, module_action_len):
    # find the indexes of the position variables in the legs
    leg_module_inds = np.where(np.array(modules_types)==1)[0]
    leg_pos_inds = []
    leg_vel_inds = []
    # assumes the first module is not a leg (should be a base)
    for leg in leg_module_inds:
        leg_inds = list(range(np.cumsum(module_state_len)[leg-1],
                   np.cumsum(module_state_len)[leg]))
        leg_pos_inds += leg_inds[::2]
        leg_vel_inds += leg_inds[1::2]
    # print('Leg pos inds: ' + str(leg_pos_inds))    


    # find the indexes of the rolling and steering joints in the wheels
    wheel_control1_inds = []
    wheel_control2_inds = []
    wheel_steer_inds = []
    wheel_module_inds = np.where(np.array(modules_types)==2)[0]
    for wheel_ind in wheel_module_inds:
        wheel_a_inds = list(range(np.cumsum(module_action_len)[wheel_ind-1],
                   np.cumsum(module_action_len)[wheel_ind]))
        wheel_control1_inds.append(wheel_a_inds[0])
        wheel_control2_inds.append(wheel_a_inds[1])
        
        wheel_s_inds = list(range(np.cumsum(module_state_len)[wheel_ind-1],
                   np.cumsum(module_state_len)[wheel_ind]))
        wheel_steer_inds.append(wheel_s_inds[0])
    # print('wheel rolling control inds: ' + str(wheel_control_inds))    
    # print('wheel steering pos inds: ' + str(wheel_steer_inds))    
      

    # find the indexes of the control for legs
    leg_control_inds = []
    for leg_ind in leg_module_inds:
        inds = list(range(np.cumsum(module_action_len)[leg_ind-1],
                   np.cumsum(module_action_len)[leg_ind]))
        leg_control_inds += inds
    # print('leg control inds: ' + str(leg_control_inds))    

    return leg_pos_inds, leg_control_inds, wheel_steer_inds, wheel_control1_inds, wheel_control2_inds





class fd_func_shared_trunk(torch.nn.Module):


 def __init__(self, module_sa_len, attachments,
             device, model_network, design_index,
             finite_diff_delta = 1e-4):
    super(fd_func_shared_trunk, self).__init__()
    self.module_state_len = module_sa_len[:int(len(module_sa_len)/2)]
    self.module_action_len = module_sa_len[int(len(module_sa_len)/2):]
    self.attachments = attachments
    self.model_network = model_network
    self.device = device
    self.design_index = design_index
    self.finite_diff_delta = finite_diff_delta

 def forward(self, state, action):
     # state = torch tensor size [batch_size, state_len]
     # action = torch tensor size [batch_size, action_len]
     
     with torch.no_grad():
         batch_size = state.shape[0]
         # divide up, since the functions operate on a module level
         state = divide_state(state, self.module_state_len)
         action = divide_action(action, self.module_action_len)

         fd_input, R0_t = state_to_fd_input(state)
         
         delta_fd_list = []
         delta_fd_var_list = []

         fd_input = torch.cat(fd_input,1)
         actions_in = torch.cat(action,1)
         state_delta_est_mean, state_delta_est_var = self.model_network(
            fd_input, actions_in, self.design_index)
         delta_fd_list.append(state_delta_est_mean)
         delta_fd_var_list.append(state_delta_est_var)

         delta_fd = torch.cat(delta_fd_list)
         delta_fd_approx = torch.mean(delta_fd,0).unsqueeze(0)
         delta_fd_var = torch.cat(delta_fd_var_list)
         delta_fd_var_mean = torch.mean(delta_fd_var,0).unsqueeze(0)

         # divide MLP output divided up into modules
         delta_fd_approx = divide_state(state_delta_est_mean, self.module_state_len)

         state_next_approx = from_body_frame_batch(state, delta_fd_approx)
        
         # combine it so that fd_func operates on tensors
         state_next_approx = combine_state(state_next_approx)    
     return state_next_approx
 
 # return the gradient approximated by manually set fd
 def grad_input(self, state, action):
     finite_diff_delta = self.finite_diff_delta
     device = self.device

     batch_size = state.shape[0]
     n_ctrl = action.shape[-1]
     n_state = state.shape[-1]

     # compute batch with varied actions
     state_rep = state.repeat(n_ctrl,1)
     action_rep = action.repeat(n_ctrl,1)
     eye_rep_ctrl = torch.eye(n_ctrl,device=device).repeat_interleave(batch_size, dim=0)
     action_perturbed = action_rep + eye_rep_ctrl*finite_diff_delta
     state_next_a_perturbed = self.forward(state_rep, action_perturbed)

     # compute batch with varied states
     state_rep = state.repeat(n_state,1)
     action_rep = action.repeat(n_state,1)
     eye_rep_state = torch.eye(n_state,device=device).repeat_interleave(batch_size, dim=0)
     state_perturbed = state_rep + eye_rep_state*finite_diff_delta
     state_next_s_perturbed = self.forward(state_perturbed, action_rep)

     # forward differencing
     # compute with non-perturbed states and actions
     state_next = self.forward(state, action)
     delta_ss = (state_next_s_perturbed - state_next.repeat(n_state,1))/finite_diff_delta
     delta_sa = (state_next_a_perturbed - state_next.repeat(n_ctrl,1))/finite_diff_delta


     dfds = delta_ss.view(n_state, batch_size, -1).permute(1,2,0)
     dfda = delta_sa.view(n_ctrl, batch_size, -1).permute(1,2,0)

             # return seperate grad mats
     return dfds, dfda

class fd_func_hardware_conditioned(torch.nn.Module):


 def __init__(self, module_sa_len, attachments,
             device, model_network, design_input,
             finite_diff_delta = 1e-4):
    super(fd_func_hardware_conditioned, self).__init__()
    self.module_state_len = module_sa_len[:int(len(module_sa_len)/2)]
    self.module_action_len = module_sa_len[int(len(module_sa_len)/2):]
    self.attachments = attachments
    self.model_network = model_network
    self.device = device
    self.fd_input_lens = design_input[0]
    self.action_lens = design_input[1]
    self.fd_output_lens = design_input[2]
    self.limb_types = design_input[3]
    self.finite_diff_delta = finite_diff_delta

 def forward(self, state, action):
     # state = torch tensor size [batch_size, state_len]
     # action = torch tensor size [batch_size, action_len]
     
     with torch.no_grad():
         batch_size = state.shape[0]
         # divide up, since the functions operate on a module level
         state = divide_state(state, self.module_state_len)
         action = divide_action(action, self.module_action_len)

         fd_input, R0_t = state_to_fd_input(state)
         
         delta_fd_list = []
         delta_fd_var_list = []

         fd_input = torch.cat(fd_input,1)
         actions_in = torch.cat(action,1)

         state_delta_est_mean, state_delta_est_var = self.model_network(
                    torch.split(fd_input, self.fd_input_lens, dim=-1),
                    torch.split(actions_in, self.action_lens, dim=-1),
                    self.fd_output_lens, self.limb_types)
         delta_fd_list.append(state_delta_est_mean)
         delta_fd_var_list.append(state_delta_est_var)

         delta_fd = torch.cat(delta_fd_list)
         delta_fd_approx = torch.mean(delta_fd,0).unsqueeze(0)
         delta_fd_var = torch.cat(delta_fd_var_list)
         delta_fd_var_mean = torch.mean(delta_fd_var,0).unsqueeze(0)

         # divide MLP output divided up into modules
         delta_fd_approx = divide_state(state_delta_est_mean, self.module_state_len)

         state_next_approx = from_body_frame_batch(state, delta_fd_approx)
        
         # combine it so that fd_func operates on tensors
         state_next_approx = combine_state(state_next_approx)    
     return state_next_approx
 
 # return the gradient approximated by manually set fd
 def grad_input(self, state, action):
     finite_diff_delta = self.finite_diff_delta
     device = self.device

     batch_size = state.shape[0]
     n_ctrl = action.shape[-1]
     n_state = state.shape[-1]

     # compute batch with varied actions
     state_rep = state.repeat(n_ctrl,1)
     action_rep = action.repeat(n_ctrl,1)
     eye_rep_ctrl = torch.eye(n_ctrl,device=device).repeat_interleave(batch_size, dim=0)
     action_perturbed = action_rep + eye_rep_ctrl*finite_diff_delta
     state_next_a_perturbed = self.forward(state_rep, action_perturbed)

     # compute batch with varied states
     state_rep = state.repeat(n_state,1)
     action_rep = action.repeat(n_state,1)
     eye_rep_state = torch.eye(n_state,device=device).repeat_interleave(batch_size, dim=0)
     state_perturbed = state_rep + eye_rep_state*finite_diff_delta
     state_next_s_perturbed = self.forward(state_perturbed, action_rep)

     # forward differencing
     # compute with non-perturbed states and actions
     state_next = self.forward(state, action)
     delta_ss = (state_next_s_perturbed - state_next.repeat(n_state,1))/finite_diff_delta
     delta_sa = (state_next_a_perturbed - state_next.repeat(n_ctrl,1))/finite_diff_delta


     dfds = delta_ss.view(n_state, batch_size, -1).permute(1,2,0)
     dfda = delta_sa.view(n_ctrl, batch_size, -1).permute(1,2,0)

             # return seperate grad mats
     return dfds, dfda
 