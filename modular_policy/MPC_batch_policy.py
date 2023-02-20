'''
Source code for paper "Learning modular robot control policies" in Transactions on Robotics
Julian Whitman, Dec. 2022. 

Uses the last state input delayed by one step (not current state) 
to control for policy warm start

Runs MPC with learned model: plan for a few steps, 
execute for fewer steps, repeat. 
Uses the policy to generate plan initial seeds.

'''
import torch
from robot_env import robot_env
import numpy as np
import os, sys, gc, time
import warnings
sys.path.append('../mpc')
from mpc import mpc
from mpc.mpc import QuadCost
from planning_utils import get_pos_control_inds, create_cost_mats2
from planning_utils import fd_func_jac, fd_func_autodiff, cost_weights
from utils import combine_state, to_tensors, state_to_fd_input, from_body_frame_batch
from utils import wrap_to_pi, rotate, create_control_inputs, divide_action
import pgnn_control as pgnnc
import pgnn
import logging 
import torch.multiprocessing as multiprocessing
import traceback
from copy import deepcopy

cwd = os.path.dirname(os.path.realpath(__file__))
# logging.info('Working in: ' + cwd)
warnings.filterwarnings("ignore", category=UserWarning)
# logging.info('ignoring UserWarning warnings')

slew_rate_penalty = cost_weights['slew_rate_penalty']
speed_scale_xy = cost_weights['speed_scale_xy']
speed_scale_yaw = cost_weights['speed_scale_yaw']

# n_envs  = number of simulations to run at the same time
def plan_batch(worker_num, urdf, gnn_nodes_model_in, gnn_nodes_control_in, 
    devices, mpc_save_path,
    n_envs = 2, n_runs = 10, show_GUI = False):


    # set up logging. since this gets run as parallel processes,
    # each process gets its own log so that they don't get jumbled.
    # import logging # putting this here should refresh the log settings each time it's run
    if mpc_save_path is not None:
        folder = os.path.dirname(mpc_save_path)
        log_name = os.path.splitext(mpc_save_path)[0]
        log_path = os.path.join(folder, log_name+'_log.log')
        logging.basicConfig(level=logging.INFO,
                                format='%(message)s', 
                                filename=log_path,
                                filemode='w')
        console = logging.StreamHandler()
        # console.setLevel(logging.INFO)
        console.setLevel(logging.DEBUG) # this might make it not print to screen?
        formatter = logging.Formatter('%(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)




    current_process = multiprocessing.current_process()
    try:
        dev_ind = int(np.mod(current_process._identity[0], len(devices)))
    except:
        dev_ind = 0
    device = devices[dev_ind]
    logging.info('Starting ' + os.path.realpath(__file__) + ' ' + urdf +' ' +  
        ' worker:' + str(worker_num) + ' ' + 
        str(current_process.name) + ' ' + str(current_process._identity)
        + ' on ' + str(device))
    

    
    # ensure that all the networks are on the desired device.
    # from documentation of "to" function:
    # If the self Tensor already has the correct torch.dtype and
    # torch.device, then self is returned. Otherwise, the returned
    # tensor is a copy of self with the desired torch.dtype and torch.device.
    gnn_nodes_model = gnn_nodes_model_in
    for node in gnn_nodes_model:
        node = node.to(device)
    del gnn_nodes_model_in

    gnn_nodes_control = gnn_nodes_control_in
    if gnn_nodes_control is not None:
        for node in gnn_nodes_control:
            node = node.to(device)
    del gnn_nodes_control_in


    # make the MPC verbosity show up if the gui is up also
    if show_GUI:
        mpc_verbosity = 1
        np.random.seed(worker_num)
    else:
        mpc_verbosity = -1

    # if we passed in None for gnn_nodes_control, then use zeros as initial seeds.
    if gnn_nodes_control is None:
        use_policy = False
    else:
        use_policy = True


    mpc_save_dict = dict()

    env = robot_env(show_GUI = show_GUI)
    env.reset_terrain()
    env.reset_robot(urdf_name=urdf, randomize_start=False)
    attachments = env.attachments
    modules_types = env.modules_types
    # logging.info('attachments: ' + str(attachments))
    # logging.info('modules_types: ' + str(modules_types))
    n_modules = len(modules_types)

    env_state_init = env.get_state()
    module_state_len = []
    for s in env_state_init:
        module_state_len.append(len(s))

    state_len= np.sum(module_state_len)
    action_len = env.num_joints
    module_action_len = list(np.diff(env.action_indexes))
    module_sa_len = module_state_len+ module_action_len

    # This isn't needed to run, really, but is left so that the same utilities
    # can be used with other functions.
    network_options = dict()
    network_options['output'] = 'Probabilistic'
    network_options['frame'] = 'Body'
    network_options['n_ensemble'] = 1
    network_options['type'] = 'GNN'
    fd_networks = None
    gnn_nodes_model_set = [gnn_nodes_model]
    modules_model = []
    for i in range(n_modules):
        modules_model.append(pgnn.Module(i, 
            gnn_nodes_model[modules_types[i]], device))
    modules_set = [modules_model]
    
    if use_policy:
        # logging.info('Using Policy for traj opt initial seed')
        # assemble module data structure for control
        modules_control = []
        for i in range(n_modules):
            modules_control.append(pgnnc.Module(i, 
                gnn_nodes_control[modules_types[i]], device))
    # else:
        # logging.info('Using Zeros for traj opt initial seed')


    # initialize a set of environments
    envs = [env]# start with the one already open
    for ind_env in range(1,n_envs):
        envs.append( robot_env(show_GUI = False) )
        envs[-1].reset_terrain()
    start_yaws = np.array([0]*n_envs)

    # mpc-style control parameters
    batch_size, n_state, n_ctrl = n_envs, state_len, action_len
    T = 20 # T is the planning horizon

    n_replans = 4
    n_execute = 10 # number of steps to execute after each replan



    leg_pos_inds, leg_control_inds, wheel_steer_inds, wheel_control1_inds, wheel_control2_inds = get_pos_control_inds(
        modules_types, module_state_len, module_action_len)

    env_state_init = combine_state(to_tensors(env_state_init)).to(device)

    delta_u = 1.0 # The amount each component of the controls is allowed to change in each LQR iteration.


    gradient_method = mpc.GradMethods.ANALYTIC # actually does finite diff, we rewrote it for finer control
    # gradient_method = mpc.GradMethods.AUTO_DIFF

    if gradient_method == mpc.GradMethods.ANALYTIC:
        fd_func = fd_func_jac
    elif gradient_method == mpc.GradMethods.AUTO_DIFF:
        fd_func = fd_func_autodiff
    else:
        logging.info('invalid gradient_method')    

    dt = env.dt
    # speed_scale_yaw = (T*dt)*np.pi/2
    # speed_scale_xy = (T*dt)

    # create the test direction goals.
    # these will be popped and used as the goal directions
    # on the first few planning runs.
    test_goals = [[speed_scale_xy*0.75,0,0],
                  [0,speed_scale_xy*0.75,0],
                  [0,0,speed_scale_yaw*0.75],
                  [-speed_scale_xy*0.75,0,0],
                  [0,-speed_scale_xy*0.75,0],
                  [0,0,-speed_scale_yaw*0.75]]
    # as a result, the first 6 runs in this can be used to compute the 
    # velocity matching metric for the current learned model


    states_memory = []
    actions_memory = []
    torques_memory = []
    run_lens = []
    goal_memory = []
    step_memory = [] # keep track of the time step, so that the tripod can be regulated



    for i_traj in range(n_runs):

        try: # if an error happens, want to move on to next i_traj


            # The upper and lower control bounds. All ones since environment rescales them.
            u_lower = -torch.ones(T, batch_size, n_ctrl, device=device)
            u_upper =  torch.ones(T, batch_size, n_ctrl, device=device)

            # We will plan for a batch of environments all at onces, since the operations are vectorizable.
            # But, the first time, use only one env, so that future runs can draw from it to get initial seeds
            # with some common grounding.
            # sampleable_runs = np.where(np.asarray(run_lens)>=T)[0]
            sampleable_runs = np.where(np.asarray(run_lens)>T/2)[0] # don't sample from runs where it flipped over immediately


            t_list = [ [] for i in range(batch_size) ]
            state_list = [ [] for i in range(batch_size) ]
            goal_tensors =[ [] for i in range(batch_size) ]
            action_list = [ [] for i in range(batch_size) ]
            torques_list = [ [] for i in range(batch_size) ]
            last_u = None

            # draw random heading and speed
            # These will form the x,y,yaw setpoint to try to reach over
            # each set of T time steps.
            # if T =20, and dt = 20/240, T*dt = 400/240 = 1.66
            # dx = delta_x_meters/(T*dt seconds)
            # dx*T*dt = delta_x_meter_per_second
            # max of 1m/s or 1rad/s use max of (T*dt) as max

            heading = (np.random.rand(batch_size)*2-1)*np.pi
            speed = np.clip(np.random.rand(batch_size)*2,0,1)*speed_scale_xy
            # ^ ensures more samples are at max speed, the rest lower speed
            turn = np.clip(np.random.normal(0,0.3*speed_scale_yaw, 
                                        size=batch_size),
                            -speed_scale_yaw,speed_scale_yaw )
            # ^ ensures most samples have low turning speed, but some have high
            dx = speed*np.cos(heading)
            dy = speed*np.sin(heading)
            delta_xyyaw_des = np.stack([dx, dy, turn], 1)
            # logging.info('delta_xyyaw_des: ' + str(np.round(delta_xyyaw_des,2)))

            # use the test goals first, until they are gone
            used_test_goals = False
            if len(test_goals)>0:
                for ib in range(batch_size):
                    if len(test_goals)>0:
                        test_goal = np.array(test_goals.pop())
                        delta_xyyaw_des[ib,:] = test_goal
                        used_test_goals = True



            sampled_steps = np.zeros(batch_size)

            last_states_for_policy = []
            for i in range(batch_size):

                if len(states_memory)>0 and not(used_test_goals) and len(sampleable_runs)>0:
                    try:
                        # pick a random state from the buffer and set that as the 
                        ind_mem1 = np.random.choice(sampleable_runs) # index of run, which has to be at least T long
                        # ind_mem2 = np.random.choice(states_memory[ind_mem1][0].shape[0]) # index
                        ind_mem2 = np.random.choice(states_memory[ind_mem1][0].shape[0]-1) # index used for state0, then ind+1 for state1

                        #  of timestep within run, which has to be at least T away from the end of the run
                        last_state_i = deepcopy([s[ind_mem2][:] for s in states_memory[ind_mem1]])
                        sampled_state = deepcopy([s[ind_mem2+1][:] for s in states_memory[ind_mem1]])
                        # need to make a copy  ^ otherwise setting 0 later will mess with the data
                        sampled_actions = deepcopy([a[ind_mem2:ind_mem2+T][:] for a in actions_memory[ind_mem1]])
                        sampled_steps[i] = step_memory[ind_mem1][ind_mem2]
                        sampled_state[0][0:2] = 0. # set x/y to zero  
                        envs[i].set_state(sampled_state)
                        envs[i].reset_debug_items()
                        envs[i].add_joint_noise()

                    except:
                        envs[i].reset_robot(urdf_name = urdf, randomize_start=False,
                            start_xyyaw = [0,0,start_yaws[i]])
                        sampled_steps[i] = 0
                        last_state_i = envs[i].get_state()

                else: # set the robot to default zero pose
                    envs[i].reset_robot(urdf_name = urdf, randomize_start=False,
                            start_xyyaw = [0,0,start_yaws[i]])
                    sampled_steps[i] = 0
                    last_state_i = envs[i].get_state()

                env_state = envs[i].get_state()
                env_i_state_tensors = to_tensors(env_state)
                state_list[i].append(env_state)
                last_states_for_policy.append(last_state_i)

                # get initial torques 
                tau = envs[i].joint_torques / envs[i].moving_joint_max_torques
                tau_div =  envs[i].divide_action_to_modules(tau)
                torques_list[i].append(tau_div)

                goal_tensors[i].append(torch.tensor(delta_xyyaw_des[i,:], dtype=torch.float32))
            

            # the default initial u will be zero. Depending on the iteration and step,
            # This can be overwritten by
            # - nothing
            # - the control sequence starting from the sampled start state in memory
            # - the current control policy rolled out with the current learned model 
            u_init = torch.zeros(T, batch_size, n_ctrl, device=device)

            # track whether each sim env robot has flipped over, and stop tracking it if so
            robot_alive = [True]*batch_size

            for replan in range(n_replans):

                ### use the policy to create initial control sequence
                if use_policy:
                    # Get sim state at start of replan
                    states_for_policy = []
                    for i in range(batch_size):
                        env_state = envs[i].get_state()
                        states_for_policy.append(env_state)

                    # state_approx starts out as the real intial state, then gets updated based on 
                    # model prediction state deltas
                    state_approx = [torch.tensor( np.stack(s),dtype=torch.float32, device=device)
                                     for s in list(zip(*states_for_policy)) ]
                    last_state = [torch.tensor( np.stack(s),dtype=torch.float32, device=device)
                                     for s in list(zip(*last_states_for_policy)) ]

                    u_policy = []

                    # for this planning set, keep goals constant over the episode
                    goals_world = torch.tensor(delta_xyyaw_des,dtype=torch.float32, device=device)
                    for t in range(T):
                        

                        ### change to body frame the goal heading and state
                        node_inputs_control = create_control_inputs(
                                last_state, goals_world)


                        for module in modules_control: # this prevents the LSTM in the GNN nodes from 
                            # learning relations over time, only over internal prop steps.
                            module.reset_hidden_states(batch_size) 

                        with torch.no_grad():

                            # use the learned policy fully on the first replan,
                            # but only for the last n_exec steps on subsequent replans,
                            # since we have the previous plan remainder as an initial guess.
                            if (replan==0 or t>=n_execute):
                                # Use GNN to find intitial control seed for this time step
                                out_mean, out_var = pgnnc.run_propagations(
                                    modules_control, attachments, 2, node_inputs_control, device)
                                u_out_mean = []
                                for mm in range(n_modules):
                                    u_out_mean.append(out_mean[mm][:,:module_action_len[mm]])

                                u_combined = torch.cat(u_out_mean,-1).squeeze()
                            else:
                                # use previous iteration remainder as control seed for this time step
                                u_combined = u_init[t,:,:]
                                u_out_mean = divide_action(u_combined, module_action_len)
                                
                            # store policy output for use later as initial planning seed
                            u_policy.append( u_combined)
                            
                            last_state = [s.clone() for s in state_approx] # keep for next time step

                            ### use forward dynamics approx to get next state.
                            if t<(T-1): # no need to do forward pass of model for final action
                                for module in modules_model: # must reset module lstm state
                                    module.reset_hidden_states(batch_size) 

                                # process states to move them to vehicle frame
                                fd_input_approx, R_t = state_to_fd_input(state_approx) # for recursive estimation

                                # pass through network
                                node_inputs_model = [torch.cat([s,a],1) for (s,a) in zip(fd_input_approx, u_out_mean)]
                                state_delta_est_mean, state_delta_var = pgnn.run_propagations(
                                    modules_model, attachments, 2, node_inputs_model, device)
                                
                                state_approx = from_body_frame_batch(state_approx, state_delta_est_mean)
                    # stack up the time steps
                    u_init = torch.stack(u_policy,0)


                ### Occasionally, the planning function will fail because there is a singular matrix inversion.
                # this seems to be a side effect of using learned dynamics to plan.
                # Rather than address it directly, for now, we throw out that trial, but save the
                # data so that later iterations will have more accurate estimates of the dynamics in that region.
                try:

                    if replan == 0:
                        # lqr_iter = 10 # need larger for initial soln
                        lqr_iter = 12 # ok
                    else:
                        # lqr_iter = 5 # too high makes lqr take too long
                        lqr_iter = 9 # ok
                    
                    x_init = []
                    for i in range(batch_size):
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

                    if gradient_method == mpc.GradMethods.AUTO_DIFF:
                        x_init.requires_grad = True # this allows AUTO_DIFF to work
                        x_lqr, u_lqr, objs_lqr = mpc.MPC(
                            n_state=n_state,
                            n_ctrl=n_ctrl,
                            T=T,
                            u_lower=u_lower, 
                            u_upper=u_upper,
                            u_init = u_init,
                            lqr_iter=lqr_iter,
                            grad_method=gradient_method,
                            verbose=mpc_verbosity,
                            backprop=False,
                            exit_unconverged=False,
                            slew_rate_penalty=slew_rate_penalty,
                            delta_u = delta_u
                        )(x_init, QuadCost(C, c), 
                            fd_func(network_options, module_sa_len, attachments,
                                device, fd_networks, modules_set, gnn_nodes_model_set
                            ))
                    else:
                        with torch.no_grad():
                            x_lqr, u_lqr, objs_lqr = mpc.MPC(
                                n_state=n_state,
                                n_ctrl=n_ctrl,
                                T=T,
                                u_lower=u_lower, 
                                u_upper=u_upper,
                                u_init = u_init,
                                lqr_iter=lqr_iter,
                                grad_method=gradient_method,
                                verbose=mpc_verbosity,
                                backprop=False,
                                exit_unconverged=False,
                                slew_rate_penalty=slew_rate_penalty,
                                delta_u = delta_u
                            )(x_init, QuadCost(C, c), 
                                fd_func(network_options, module_sa_len, attachments,
                                    device, fd_networks, modules_set, gnn_nodes_model_set
                                ))


                    # check for nans
                    if torch.isnan(u_lqr).any():
                        logging.info('NaN detected')
                        logging.info(' gave up on ' + urdf + ' plan ' + str(i_traj) + '/' + str(int(n_runs)))
                        break

                    last_states_for_policy = []
                    for i in range(batch_size):            
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

                                # stop stepping this robot if it flipped over. 
                                # But keep it in the batch to keep sizes consistent.     
                                if np.dot([0,0,1], envs[i].z_axis)<0:
                                    robot_alive[i] = False

                            # keep track of the next-to-last state in the replan sequence,
                            # so that it can be passed into the policy as the previous observation
                            # this must happen even if the robot flipped over, so that the dimensionare are consistent
                            if t==n_execute-2: # t = [0... n_ex-1] so n_ex-2 is next to last.
                                last_states_for_policy.append(env_state)


                    last_u = u_lqr[t] # set for use in next slew rate cost

                    if not use_policy:
                        # If we use_policy, the policy roll-out will overwrite the remainder of u_init
                        # Otherwise use 0 for long term initialization 
                        ind = replan*n_execute
                        u_init[T-n_execute:,:,:] = 0 

                    # use the remainder of the control inputs to warm start the next replan
                    u_init[:(T-n_execute),:,:] = u_lqr[n_execute:,:,:].detach()

                except: # occasionally if the model is really bad, we
                 # can get errors in the trajopt. In that case, abort it and try again.
                    logging.info("Error in planning: " + str(sys.exc_info()[0]) + ' worker ' + str(worker_num))
                    traceback.print_exc()
                    logging.info(' gave up on ' + urdf + ' plan ' + str(i_traj) + '/' + str(int(n_runs)))
                    break


        except: 
            # can get errors from gpu driver if used at high capacity.
            logging.info("Error: " + str(sys.exc_info()[0]) + ' worker ' + str(worker_num))
            traceback.print_exc()
            logging.info(' gave up on ' + urdf + ' plan ' + str(i_traj) + '/' + str(int(n_runs)))
            time.sleep(0.5)

        for i in range(batch_size):
        # add on a NaN for last action so that the states and action lists 
        # are the same length
            action_now = []
            for ai in range(len(module_action_len)):
                na =module_action_len[ai]
                action_now.append(np.ones(na)*np.nan) 
            action_list[i].append(action_now)  

            t_list[i] = np.array(range(n_replans*n_execute +1 )) + sampled_steps[i]
                 # keep track of what time step was used for tripod

        if np.mod(i_traj,10)==0:
            logging.info('  Planned ' + urdf + ' traj ' + str(i_traj) + '/' + str(int(n_runs)))

        # stack up and record 
        for i in range(batch_size):
            state_list_tensors = [torch.tensor( np.stack(s),dtype=torch.float32)
                                     for s in list(zip(*state_list[i])) ]
            action_list_tensors = [torch.tensor( np.stack(a),dtype=torch.float32)
                                     for a in list(zip(*action_list[i])) ]
            torques_list_tensors = [torch.tensor( np.stack(a),dtype=torch.float32)
                                     for a in list(zip(*torques_list[i])) ]
            run_len = len(state_list[i])
            run_lens.append(run_len)
            states_memory.append(state_list_tensors)
            actions_memory.append(action_list_tensors)
            torques_memory.append(torques_list_tensors)
            step_memory.append(t_list[i])
            goal_memory.append(torch.stack(goal_tensors[i],-1))



        if np.mod(i_traj,5)==0 and mpc_save_path is not None:
            # Save data periodically
            # action_list state_list are both lists of divided np
            # mpc_save_dict is a dict()
            mpc_save_dict['comment'] = ''
            mpc_save_dict['states_memory'] = states_memory
            mpc_save_dict['actions_memory'] = actions_memory
            mpc_save_dict['torques_memory'] = torques_memory
            mpc_save_dict['goal_memory'] = goal_memory
            mpc_save_dict['run_lens'] = run_lens
            mpc_save_dict['step_memory'] = step_memory
            mpc_save_dict['urdf'] = urdf
            mpc_save_dict['attachments'] = attachments
            mpc_save_dict['modules_types'] = modules_types
            mpc_save_dict['module_sa_len'] = module_sa_len
            mpc_save_dict['slew_rate_penalty'] = slew_rate_penalty
            mpc_save_dict['plan_horizon'] = T
            mpc_save_dict['mpc_n_executed'] = n_execute
            mpc_save_dict['speed_scale_xy'] = speed_scale_xy
            mpc_save_dict['speed_scale_yaw'] = speed_scale_yaw 
            torch.save(mpc_save_dict, mpc_save_path)



    if mpc_save_path is not None:
        # Save data after all plans done
        # action_list state_list are both lists of divided np
        # mpc_save_dict is a dict()
        mpc_save_dict['comment'] = ''
        mpc_save_dict['states_memory'] = states_memory
        mpc_save_dict['actions_memory'] = actions_memory
        mpc_save_dict['torques_memory'] = torques_memory
        mpc_save_dict['goal_memory'] = goal_memory
        mpc_save_dict['run_lens'] = run_lens
        mpc_save_dict['step_memory'] = step_memory
        mpc_save_dict['urdf'] = urdf
        mpc_save_dict['attachments'] = attachments
        mpc_save_dict['modules_types'] = modules_types
        mpc_save_dict['module_sa_len'] = module_sa_len
        mpc_save_dict['slew_rate_penalty'] = slew_rate_penalty
        mpc_save_dict['plan_horizon'] = T
        mpc_save_dict['mpc_n_executed'] = n_execute
        mpc_save_dict['speed_scale_xy'] = speed_scale_xy
        mpc_save_dict['speed_scale_yaw'] = speed_scale_yaw 

        
        torch.save(mpc_save_dict, mpc_save_path)
    logging.info(' Finished and saved ' + urdf + ' trajectories ')

if __name__ == '__main__':
    import pgnn
    folder = 'mbrl_test'

    urdf = 'llllll'
    # urdf = 'wnwwnw'

    # torch.multiprocessing.set_start_method('spawn') # needed for CUDA drivers in parallel
    seq_len = 10
    mbrl_iter = 1
        # # load previous weights if desired
    # check if file exists, if so load it.
    model_fname = 'multidesign_pgnn_ms'+ str(int(seq_len))+ '_iter' + str(int(mbrl_iter)) + '.pt'
    model_fname = os.path.join(folder, model_fname)
    # if os.path.exists(model_fname):
    logging.info('Loading weights from ' + model_fname)
    save_dict = torch.load( model_fname, map_location=lambda storage, loc: storage)
    preload_model = True
    internal_state_len = save_dict['internal_state_len']
    message_len= save_dict['message_len']
    hidden_layer_size= save_dict['hidden_layer_size']

    gnn_nodes_model = pgnn.create_GNN_nodes(internal_state_len, 
                    message_len, hidden_layer_size, 
                    torch.device('cpu'), body_input = True)

    if preload_model:
        pgnn.load_state_dicts(gnn_nodes_model, save_dict['gnn_state_dicts'])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    mpc_save_path  = None
    plan_batch(0, urdf, gnn_nodes_model, None,
                [device], mpc_save_path,
                n_envs = 1, n_runs = 3,
                show_GUI=True)