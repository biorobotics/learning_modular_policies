'''
Runs MPC with learned model: plan for a few steps, 
execute for fewer steps, repeat. 

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
from utils import combine_state, to_tensors
import pgnn

from copy import deepcopy
cwd = os.path.dirname(os.path.realpath(__file__))
print('Working in: ' + cwd)
warnings.filterwarnings("ignore", category=UserWarning)
print('ignoring UserWarning warnings')

# n_envs  = number of simulations to run at the same time
# n_runs = per env. 
def plan_batch(worker_num, urdf, gnn_nodes, device, mpc_save_path,
    n_envs = 2, n_runs = 10, show_GUI = False):


    # set up logging. since this gets run as parallel processes,
    # each process gets its own log so that they don't get jumbled.
    import logging # putting this here should refresh the log settings each time it's run
    # TODO: check this works ok
    folder = os.path.dirname(mpc_save_path)
    log_name = os.path.splitext(mpc_save_path)[0]
    log_path = os.path.join(folder, log_name+'_log.log')
    logging.basicConfig(level=logging.INFO,
                            format='%(message)s', 
                            filename=log_path,
                            filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    logging.info('Starting ' + os.path.realpath(__file__))
    print = logging.info # set print to be logging so I don't need to replace all instances in the file

    for gnn_node in gnn_nodes: # share memory for later use by multiprocessing
        gnn_node.share_memory()

    mpc_save_dict = dict()
    np.random.seed(worker_num)
    print(urdf + ' worker ' + str(worker_num) + 
        ' on GPU device ' + str(device))


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

    # This isn't needed to run, really, but is left so that the same utilities
    # can be used with other functions.
    network_options = dict()
    network_options['output'] = 'Probabilistic'
    network_options['frame'] = 'Body'
    network_options['n_ensemble'] = 1
    network_options['type'] = 'GNN'
    fd_networks = None
    gnn_nodes_set = [gnn_nodes]
    modules = []
    for i in range(n_modules):
        modules.append(pgnn.Module(i, gnn_nodes[modules_types[i]], device))
    modules_set = [modules]
       

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
    total_steps = n_replans*n_execute

    # slew_rate_penalty = 75 #100 high # 50 ok #10 meh
    # slew_rate_penalty = 50 # higher than 50 impedes ability for wheels to change direction on a dime
    # slew_rate_penalty = 35 # seems all right
    slew_rate_penalty = 40 # good


    leg_pos_inds, leg_control_inds, wheel_steer_inds, wheel_control1_inds, wheel_control2_inds = get_pos_control_inds(
        modules_types, module_state_len, module_action_len)

    env_state_init = combine_state(to_tensors(env_state_init)).to(device)

    # The upper and lower control bounds. All ones since environment rescales them.
    u_lower = -torch.ones(T, batch_size, n_ctrl, device=device)
    u_upper =  torch.ones(T, batch_size, n_ctrl, device=device)
    delta_u = 0.5 # The amount each component of the controls is allowed to change in each LQR iteration.


    method = 'DDP'
    gradient_method = mpc.GradMethods.ANALYTIC # actually does finite diff, we rewrote it for finer control
    # gradient_method = mpc.GradMethods.AUTO_DIFF

    if gradient_method == mpc.GradMethods.ANALYTIC:
        fd_func = fd_func_jac
    elif gradient_method == mpc.GradMethods.AUTO_DIFF:
        fd_func = fd_func_autodiff
    else:
        print('invalid gradient_method')    

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
    # as a result, the first 6 runs in this can be used to compute the 
    # velocity matching metric for the current learned model


    states_memory = []
    actions_memory = []
    torques_memory = []
    run_lens = []
    goal_memory = []
    step_memory = [] # keep track of the time step, so that the tripod can be regulated

    for i_traj in range(n_runs):

        u_init = torch.zeros(T, batch_size, n_ctrl, device=device)

        t_list = [ [] for i in range(n_envs) ]
        state_list = [ [] for i in range(n_envs) ]
        # state_list_tensors =[ [] for i in range(n_envs) ]
        goal_tensors =[ [] for i in range(n_envs) ]
        action_list = [ [] for i in range(n_envs) ]
        torques_list = [ [] for i in range(n_envs) ]
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
        turn = np.clip(np.random.normal(0,0.2*speed_scale_yaw, 
                                    size=batch_size),
                        -speed_scale_yaw,speed_scale_yaw )
        # ^ ensures most samples have low turning speed, but some have high
        dx = speed*np.cos(heading)
        dy = speed*np.sin(heading)
        delta_xyyaw_des = np.stack([dx, dy, turn], 1)
        # print('delta_xyyaw_des: ' + str(np.round(delta_xyyaw_des,2)))

        # use the test goals first, until they are gone
        used_test_goals = False
        if len(test_goals)>0:
            for ib in range(batch_size):
                if len(test_goals)>0:
                    test_goal = np.array(test_goals.pop())
                    delta_xyyaw_des[ib,:] = test_goal
                    used_test_goals = True



        sampled_steps = np.zeros(n_envs)
        for i in range(n_envs):

            if len(states_memory)>0 and not(used_test_goals):
                # pick a random state from the buffer and set that as the 
                ind_mem1 = np.random.choice(len(states_memory)) # index of run
                ind_mem2 = np.random.choice(states_memory[ind_mem1][0].shape[0]) # index of timestep within run
                sampled_state = deepcopy([s[ind_mem2][:] for s in states_memory[ind_mem1]])
                # need to make a copy  ^ otherwise setting 0 later will mess with the data
                sampled_steps[i] = step_memory[ind_mem1][ind_mem2]
                sampled_state[0][0:2] = 0. # set x/y to zero  
                envs[i].set_state(sampled_state)
                envs[i].reset_debug_items()
                envs[i].add_joint_noise()

            else: # set the robot to default zero pose
                envs[i].reset_robot(urdf_name = urdf, randomize_start=False,
                        start_yaw = start_yaws[i])
                sampled_steps[i] = 0

            env_state = envs[i].get_state()
            env_state_combined = combine_state(to_tensors(env_state))
            state_list[i].append(env_state)
            # state_list_tensors[i].append(env_state_combined)

            # get initial torques 
            tau = envs[i].joint_torques / envs[i].moving_joint_max_torques
            tau_div =  envs[i].divide_action_to_modules(tau)
            torques_list[i].append(tau_div)

            goal_tensors[i].append(torch.tensor(delta_xyyaw_des[i,:], dtype=torch.float32))
            
        ### Occasionally, the planning function will fail because there is a singular matrix inversion.
        ## this seems to be a side effect of using learned dynamics, 
        ## rather than address it directly, for now, we throw out that trial.
        try:

            # track whether each sim env robot has flipped over, and stop tracking it if so
            robot_alive = [True]*n_envs

            # print('Replan ', end = '')
            for replan in range(n_replans):

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
                        verbose=-1,
                        backprop=False,
                        exit_unconverged=False,
                        slew_rate_penalty=slew_rate_penalty,
                        delta_u = delta_u
                    )(x_init, QuadCost(C, c), 
                        fd_func(network_options, module_sa_len, attachments,
                            device, fd_networks, modules_set, gnn_nodes_set
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
                            verbose=-1,
                            backprop=False,
                            exit_unconverged=False,
                            slew_rate_penalty=slew_rate_penalty,
                            delta_u = delta_u
                        )(x_init, QuadCost(C, c), 
                            fd_func(network_options, module_sa_len, attachments,
                                device, fd_networks, modules_set, gnn_nodes_set
                            ))


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


            for i in range(n_envs):
            # add on a NaN for last action so that the states and action lists 
            # are the same length
                action_now = []
                last_action = action_list[i][-1]
                for ai in range(len(last_action)):
                    na = len(last_action[ai])
                    action_now.append(np.ones(na)*np.nan) 
                action_list[i].append(action_now)  

                t_list[i] = np.array(range(n_replans*n_execute +1 )) + sampled_steps[i]
                     # keep track of what time step was used for tripod


            for i in range(n_envs):
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

            if np.mod(i_traj,5)==0:
                print('  planned ' + urdf + ' plan ' + str(i_traj) + '/' + str(int(n_runs)))

        except: # occasionally if the model is really bad,
         # can get errors in the DDP planner. In that case, abort it and try again.
            print("Unexpected error: ", sys.exc_info()[0])
            print(' gave up on ' + urdf + ' plan ' + str(i_traj) + '/' + str(int(n_runs)))



        if np.mod(i_traj,10)==0:
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

    torch.cuda.empty_cache()

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
    print('  saved ' + urdf + ' plans ')

