#!/usr/bin/env python
# coding: utf-8

'''
Source code for paper "Learning modular robot control policies" in Transactions on Robotics
MLP comparisons
Julian Whitman, Dec. 2022. 

Run mbrl, with **shared trunk architecture** for model and policy:
1) generate random rollout data
2) learn dynamics from all rollout data collected
3) plan with learned model to generate mpc rollout data. 
 -- This version uses policy to generate intitial seed
4) learn to clone plans into policy
5) simulate policy to validate and gather policy rollout data
- return to (2) 
- repeat


'''

# import libraries
import torch
from robot_env import robot_env
import numpy as np
from datetime import datetime
from generate_random_rollouts import generate_random_rollouts
from utils import get_sampleable_inds
from planning_utils import compare_velocities, w_tripod, cost_weights
import gc, os
import logging

from shared_MLP_model import shared_trunk_model
from shared_MLP_policy import shared_trunk_policy
from train_model_MLP import train_model
from train_control_MLP import train_control
from MPC_batch_policy_MLP import plan_batch
from apply_policy_MLP import make_goal_memory, apply_policy
from shared_MLP_utils import get_in_out_lens

cwd = os.path.dirname(os.path.realpath(__file__))


def npstr(input): # a shortcut to print numpy arrays
    return np.array2string(input,precision=3)


if __name__ == '__main__':


    # Flags and settings

    USE_MULTIPROCESS = True 
    USE_MULTIPROCESS = False # flag to use parallized version.
    # There are then some worker parameters that need be tuned for a specific
    # machine if this is se to true.


    # Which GPU device to use, if any
    # backprop_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    backprop_devices = [torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                       torch.device("cuda:1" if torch.cuda.is_available() else "cpu")]

    # backprop_devices = [torch.device("cuda:0" if torch.cuda.is_available() else "cpu")]*2
    # backprop_devices = [torch.device("cuda:2" if torch.cuda.is_available() else "cpu"),
    #                    torch.device("cuda:3" if torch.cuda.is_available() else "cpu")]

    # for real training
    num_mbrl_iters = 3 # number of alteration between plan and learning
    seq_len = 10 # select sequence length for multistep loss. 10 is good
    
    # Settings to test if everything runs without full training
    # num_mbrl_iters = 0 # number of alteration between plan and learning
    # seq_len = 2 # select sequence length for multistep loss. 10 is good
    
    # which design urdfs are going to be used in training:
    urdf_names = ['llllll', 'lnwwnl', 'llwwll', 'lnllnl', 
                  'lwllwl', 'lwwwwl', 'wlwwlw', 'wwllww', 
                  'wwwwww', 'wnllnw', 'wllllw', 'wnwwnw']
    # urdf_names = ['llllll', 'wnwwnw'] # smaller test case

    start_time = datetime.now()
    start_time_str = datetime.strftime(start_time, '%Y%m%d_%H%M')
    folder = os.path.join(cwd, 'saved/shared_trunk_tripod_trial2')   



    if USE_MULTIPROCESS:
        # # set to spawn processes
        torch.multiprocessing.set_start_method('spawn') # needed for CUDA drivers in parallel
        torch.multiprocessing.set_sharing_strategy('file_system') # might be needed for opening and closing many files
        manager = torch.multiprocessing.Manager()

    if not(os.path.exists(folder)):
        os.mkdir(folder)
        print('Created folder ' + folder)
    else:
        print('Using folder ' + folder)

    # set up logging
    log_path = os.path.join(folder, 'mbrl_log_' + start_time_str + '.log')

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

    def print_time():
        time_now = datetime.now()
        time_now_str = datetime.strftime(time_now, '%Y%m%d_%H%M')    
        logging.info('Time now: ' + time_now_str)
    print_time()

    if w_tripod>0:
        logging.info('Using tripod penalty')
    else:
        logging.info('NOT using tripod penalty')
    logging.info('Cost weights: ' + str(cost_weights))



    ### First phase: Gather random rollouts
    random_rollouts = dict()
    measurement_stds = dict()
    num_joints = dict()
    module_state_len = dict()
    for urdf in urdf_names:
        random_rollouts[urdf] = dict()
        for key in ['states_memory','actions_memory','run_lens','attachments','modules_types']:
            random_rollouts[urdf][key] = list()

        # make env, which will be only temporarily used to extract some data
        env = robot_env(show_GUI = False)
        env.reset_terrain()
        env.reset_robot(urdf_name=urdf, randomize_start=False)
        measurement_stds[urdf] = env.measurement_stds
        num_joints[urdf] = env.num_joints
        # logging.info(urdf + ' measurement_stds: ' + str(measurement_stds[urdf]))
        env_state_init = env.get_state()
        module_state_len[urdf] = []
        for s in env_state_init:
            module_state_len[urdf].append(len(s))

    logging.info('foot friction: ' + str(env.foot_friction))
    logging.info('wheel friction: ' + str(env.wheel_friction))
    logging.info('plane friction: ' + str(env.plane_friction))

    del env # remove the environment to save memory, since main does not use them anymore

    # Note that multiprocessing doesn't work properly inside a jupyter notebook
    # join processes. The main purpose of join() is to ensure that a child process has completed before the main process does anything that depends on the work of the child process.
    cpu_count = torch.multiprocessing.cpu_count()
    if cpu_count > 20:
        num_processes = 25
    elif cpu_count > 10:
        num_processes = 18
    elif cpu_count ==8:
        num_processes = 8
    else:
        num_processes = 4
    
    num_rollouts_per_joint = 300. # setting used in full run
    logging.info('num_rollouts_per_joint: ' + str(num_rollouts_per_joint))


    for urdf in urdf_names:
        # scale to get more rollouts for robots with more joints
        # it will take more data to cover their state space.
        num_rollouts_per_process = int(
            (num_rollouts_per_joint/num_processes)*num_joints[urdf]
            )
        # more is better but will end up taking up more memory.

        # check if file exists, if so load it.
        random_rollouts_fname = urdf  + '_random_rollouts.ptx'
        random_rollouts_fname = os.path.join(folder, random_rollouts_fname)
        if os.path.exists(random_rollouts_fname):
        # if False: # force to recreate it
            random_rollouts[urdf] = torch.load(random_rollouts_fname)
            logging.info('Loading rollouts from ' + random_rollouts_fname)

        else: # if it does not exist yet, create it.

            if USE_MULTIPROCESS: # Multiprocess version
                # use manager lists to allow passing from main process to child processes
                random_rollouts_p_list = []
                for p_num in range(num_processes):
                    random_rollouts_p = manager.dict()
                    random_rollouts_p['state_lists'] = manager.list()
                    random_rollouts_p['action_lists'] = manager.list()
                    random_rollouts_p['run_lens'] = manager.list()
                    random_rollouts_p['attachments'] = manager.list()
                    random_rollouts_p['modules_types'] = manager.list()
                    random_rollouts_p_list.append(random_rollouts_p)

                # spawn processes
                processes = []
                for p_num in range(num_processes):
                    p = torch.multiprocessing.Process(target=generate_random_rollouts, 
                                  args=(p_num,urdf,
                                    random_rollouts_p_list[p_num], 
                                    num_rollouts_per_process))
                    logging.info(urdf + ' starting process ' + str(p_num) + 
                        ' doing ' + str(num_rollouts_per_process) + ' rollouts')
                    p.start()
                    processes.append(p)
                for p_num in range(num_processes):
                    processes[p_num].join()

                # Collect and convert managed lists to normal list before saving
                random_rollouts[urdf]['states_memory'] = []
                random_rollouts[urdf]['actions_memory'] = []
                random_rollouts[urdf]['run_lens'] = []
                for p_num in range(num_processes):
                    random_rollouts_p = random_rollouts_p_list[p_num]
                    for state_list in random_rollouts_p['state_lists']:
                        state_list_tensors = [torch.tensor( np.stack(s),dtype=torch.float32)
                             for s in list(zip(*state_list)) ]
                        random_rollouts[urdf]['states_memory'].append(state_list_tensors)

                    for action_list in random_rollouts_p['action_lists']:
                        action_list_tensors = [torch.tensor( np.stack(a),dtype=torch.float32)
                             for a in list(zip(*action_list)) ]
                        random_rollouts[urdf]['actions_memory'].append(action_list_tensors)

                    random_rollouts[urdf]['run_lens'] += list(random_rollouts_p_list[p_num]['run_lens'])
                    random_rollouts[urdf]['attachments'] = list(random_rollouts_p_list[p_num]['attachments'])
                    random_rollouts[urdf]['modules_types'] = list(random_rollouts_p_list[p_num]['modules_types'])

                del random_rollouts_p_list

            else: # Single process verion. Takes much longer.
                random_rollouts_p = dict()
                random_rollouts_p['state_lists'] = list()
                random_rollouts_p['action_lists'] = list()
                random_rollouts_p['run_lens'] = list()
                random_rollouts_p['attachments'] = list()
                random_rollouts_p['modules_types'] = list()
                logging.info(urdf + ' running ')
                generate_random_rollouts(0,
                                         urdf,
                                         random_rollouts_p, 
                                         num_rollouts_per_process)

                random_rollouts[urdf]['states_memory'] = []
                random_rollouts[urdf]['actions_memory'] = []
                random_rollouts[urdf]['run_lens'] = []
                for state_list in random_rollouts_p['state_lists']:
                    state_list_tensors = [torch.tensor( np.stack(s),dtype=torch.float32)
                         for s in list(zip(*state_list)) ]
                    random_rollouts[urdf]['states_memory'].append(state_list_tensors)

                for action_list in random_rollouts_p['action_lists']:
                    action_list_tensors = [torch.tensor( np.stack(a),dtype=torch.float32)
                         for a in list(zip(*action_list)) ]
                    random_rollouts[urdf]['actions_memory'].append(action_list_tensors)

                random_rollouts[urdf]['run_lens'] += list(random_rollouts_p['run_lens'])
                random_rollouts[urdf]['attachments'] = list(random_rollouts_p['attachments'])
                random_rollouts[urdf]['modules_types'] = list(random_rollouts_p['modules_types'])

                del random_rollouts_p

            # Save to file so that if we re-run later we can skip this step
            torch.save(random_rollouts[urdf], random_rollouts_fname)



    # Trim down number of rollouts used. As needed, if the runs have more data than needed
    for urdf in urdf_names:
        n_random_runs_used = int(num_rollouts_per_joint*num_joints[urdf])
        if len(random_rollouts[urdf]['run_lens'])>=n_random_runs_used:
            logging.info('Trimming ' + urdf + ' to '
             + str(n_random_runs_used) + ' random runs')
            random_rollouts[urdf]['states_memory'] = random_rollouts[urdf]['states_memory'][:n_random_runs_used]
            random_rollouts[urdf]['actions_memory'] = random_rollouts[urdf]['actions_memory'][:n_random_runs_used]
            random_rollouts[urdf]['run_lens'] = random_rollouts[urdf]['run_lens'][:n_random_runs_used]
        else:
            logging.info(urdf + ' num random runs: ' + str(len(random_rollouts[urdf]['run_lens'])))

    # garbage collect the unused data to manage memory 
    gc.collect()

    print_time()


    ### Learn dynamics from all rollout data collected
    logging.info('Starting dynamics learning')

    # Depending on the length of the multistep sequence we want,
    # only some indexes of the full set of states collected can be sampled.
    sampleable_inds = dict()
    batch_size_model = 500 # default batch size
    batch_sizes = dict()
    states_memory_tensors = dict()
    actions_memory_tensors = dict()
    modules_types = dict()
    attachments = dict()

            
    for urdf in urdf_names:

        states_memory = random_rollouts[urdf]['states_memory']
        actions_memory = random_rollouts[urdf]['actions_memory']
        run_lens = random_rollouts[urdf]['run_lens']
        modules_types[urdf] = random_rollouts[urdf]['modules_types']
        attachments[urdf]  = random_rollouts[urdf]['attachments']
        
        # concatenate data to a long list
        states_memory_tensors[urdf] = [torch.cat(s,0) for s in list(zip(*states_memory)) ]
        actions_memory_tensors[urdf] = [torch.cat(s,0) for s in list(zip(*actions_memory)) ]


        sampleable_inds[urdf] = get_sampleable_inds(
            run_lens, seq_len)
        n_sampleable = len(sampleable_inds[urdf])
        batch_sizes[urdf] = batch_size_model
        if batch_sizes[urdf] > n_sampleable:
            batch_sizes[urdf] = n_sampleable
            
    mbrl_iter = 0

    # Initialize model network and optimizer
    # # load previous weights if desired
    # check if file exists, if so load it.
    model_fname = 'shared_trunk_ms'+ str(int(seq_len))+ '_iter' + str(int(mbrl_iter)) + '.pt'
    model_fname = os.path.join(folder, model_fname)
    if os.path.exists(model_fname):
        logging.info('Loading weights from ' + model_fname)
        save_dict = torch.load( model_fname, map_location=lambda storage, loc: storage)
        preload_model = True
        n_hidden_layers = save_dict['n_hidden_layers']
        hidden_layer_size= save_dict['hidden_layer_size']
    else:
        n_hidden_layers = 6
        hidden_layer_size = 300
        preload_model = False

    fd_input_lens, fd_output_lens, policy_input_lens,action_lens,limb_types = get_in_out_lens(urdf_names)

    fd_input_lens_sums = [sum(s) for s in fd_input_lens]
    fd_output_lens_sums = [sum(s) for s in fd_output_lens]
    action_lens_sums = [sum(a) for a in action_lens]
    policy_input_lens_sums = [sum(s) for s in policy_input_lens]
    logging.info('fd_input_lens_sums, action_lens_sums, policy_input_lens_sums,fd_output_lens_sums: ' + 
        str(fd_input_lens_sums) + ', ' +
        str(action_lens_sums) +', ' +
        str(policy_input_lens_sums) +', ' +
        str(fd_output_lens_sums))

    n_training_steps = 9000
    weight_decay = 1e-4
    lr_init = 1e-3
    model_network = shared_trunk_model(
        fd_input_lens_sums, action_lens_sums, 
        fd_output_lens_sums, 
        n_hidden_layers, hidden_layer_size)
    optimizer_model = torch.optim.Adam(model_network.parameters(), 
                        lr=lr_init,
                        weight_decay= weight_decay)
    for param_group in optimizer_model.param_groups:
        param_group['lr'] = lr_init

    if preload_model:
        model_network.load_state_dict(save_dict['state_dict'])

    model_network.share_memory() # share memory for later use by multiprocessing

    # count number of parameters    
    num_nn_params= sum(p.numel() for p in model_network.parameters())
    logging.info('Num NN params model_network: ' + str(num_nn_params))


    if not(preload_model):
        logging.info('Training model')
        n_designs_per_step = 6
        train_model(model_fname, urdf_names, n_training_steps, 
                model_network, optimizer_model,
                states_memory_tensors, actions_memory_tensors, 
                modules_types,  attachments, module_state_len,
                sampleable_inds, batch_sizes, seq_len,
                backprop_devices[0], weight_decay, 
                n_designs_per_step, None, 0, True, False)

    print_time()

    mpc_rollouts = dict()
    vel_metric = dict()
    vel_metric_baseline = dict()
    for urdf in urdf_names:
        mpc_rollouts[urdf] = dict()
        # initialize empty lists for these entries
        for key in ['states_memory', 'actions_memory', 'torques_memory',
                     'goal_memory', 'run_lens','step_memory']:
            mpc_rollouts[urdf][key] = list()
        vel_metric[urdf] = list()
        vel_metric_baseline[urdf] = list()



    ## Create control GNN
    control_save_path = 'shared_trunk_control_iter0.pt'
    control_save_path = os.path.join(folder, control_save_path)

    if os.path.exists(control_save_path):
        logging.info('Loading weights from ' + control_save_path)
        save_dict_control = torch.load( control_save_path, map_location=lambda storage, loc: storage)
        preload_control = True
        n_hidden_layers = save_dict_control['n_hidden_layers']
        hidden_layer_size= save_dict_control['hidden_layer_size']
    else:
        n_hidden_layers = 6
        hidden_layer_size = 350
        preload_control = False

    # Graph neural network creation
    goal_len = 3
    policy_network = shared_trunk_policy(
        policy_input_lens_sums, action_lens_sums, 
        goal_len, n_hidden_layers, hidden_layer_size)
    weight_decay_control = 1e-4
    optim_lr_control = 3e-3 
    optimizer_control = torch.optim.Adam(
                policy_network.parameters(),
                lr=optim_lr_control,
                weight_decay= weight_decay_control)  
    policy_network.share_memory()

    # # load previous weights if they exist, otherwise, save the initial weights
    if preload_control:
        policy_network.load_state_dict(save_dict_control['state_dict'])
    else:
        state_dict = policy_network.state_dict()
        save_dict_control = dict()
        save_dict_control['comment'] = 'initial network weights, untrained'
        save_dict_control['state_dict'] =  state_dict
        save_dict_control['n_hidden_layers'] = policy_network.n_hidden_layers
        save_dict_control['hidden_layer_size'] = policy_network.hidden_layer_size
        torch.save(save_dict_control,  control_save_path,
                _use_new_zipfile_serialization=False)


#### ----- Main MBRL loop ----- ####

    for mbrl_iter in range(1,num_mbrl_iters+1):
        logging.info('*** Starting mbrl iter ' + str(mbrl_iter) + ' ***')
        print_time()

        torch.cuda.empty_cache()


        # Pass in None as the controller on the first iteration since it
        # outputs nonsense before it is trained.
        # In the plan_batch, this will result in the initial seed for the control
        # in trajectory optimization will be zeros, rather than control policy.
        if mbrl_iter>1:
            policy_network_input = policy_network
            logging.info('Using Policy for traj opt initial seed')
        else:
            policy_network_input = None
            logging.info('Using Zeros for traj opt initial seed')
        
        # OVERWRITE FOR TEST: do not use gnn for control init
        # policy_network_input = None

                    
        ### Plan with learned model to generate mpc rollout data
        ## This runs the data generation script which saves to files.
        ## Then the files all get loaded, whether they were just created or 
        ## if they existed previously.
        if not(USE_MULTIPROCESS):# or len(urdf_names)==1:
            logging.info('running mpc serially')
            mpc_rollouts_now = dict()
            if mbrl_iter == num_mbrl_iters:
                n_runs = 125 # need more data to train controller on last run
            else:
                n_runs = 100
            
            # n_runs = 3 # ## For debug only, small number of runs

            for design_index in range(len(urdf_names)):
                urdf = urdf_names[design_index]
                mpc_save_path  = urdf  + '_mpc_rollouts_iter' + str(int(mbrl_iter)) + '.ptx'
                mpc_save_path = os.path.join(folder, mpc_save_path)
                if not(os.path.exists(mpc_save_path)):

                    plan_batch(0, urdf, design_index,
                        model_network, policy_network,                         
                        [backprop_devices[0]], mpc_save_path,
                        n_envs = 8, n_runs = n_runs,
                        show_GUI=False)
                    # the plans are saved at the end of each plan_batch process

        else:
            # Multiprocess version.
            # use pool starmap to do all designs with one process each. 
            # TODO: Might be able to distribute more efficiently later
            # how to use multiple GPUS efficiently?
            # how to handle gpu memory running out inside spawned processes?

            # Use multiple gpus or cpus to do the gnn forward faster
            # for quad workstation:
            if torch.cuda.device_count()==4:
                # devices = [ torch.device('cuda:0') ]*3 + [ torch.device('cuda:1') ]*3 + \
                #         [ torch.device('cuda:2') ]*3 + [ torch.device('cuda:3') ]*3
                devices = [ torch.device('cuda:0') , torch.device('cuda:1') ,
                            torch.device('cuda:2') , torch.device('cuda:3') ]*2
            elif torch.cuda.device_count()==2:
                devices = [ torch.device('cuda:0') ,torch.device('cuda:1') ]*2 
            else:
                devices = [ torch.device('cuda:0') ]*2

            num_processes = len(devices)


            if mbrl_iter == num_mbrl_iters:
                n_runs_per_process = 100 # use more data to train controller on last run
            else:
                n_runs_per_process =  75

            # n_runs_per_process = 3 ## For debug only

            n_envs_per_process = 10
            logging.info('n_envs_per_process: ' + str(n_envs_per_process))
            pool_inputs = []
            mpc_rollouts_now = dict()
            ind = 0
            for i in range(len(urdf_names)):
                urdf = urdf_names[i]
                mpc_save_path  = urdf  + '_mpc_rollouts_iter' + str(int(mbrl_iter)) + '.ptx'
                mpc_save_path = os.path.join(folder, mpc_save_path)

                if not(os.path.exists(mpc_save_path)):
                    pool_inputs.append([ind, urdf,  i,
                        model_network, policy_network,      
                        devices, 
                        mpc_save_path,
                        n_envs_per_process, 
                        n_runs_per_process])
                    ind+=1 # iterates through the devices
                else:
                    logging.info(mpc_save_path + ' from file')
            if len(pool_inputs)>0:
                logging.info('starting plan_batch pool')
                with torch.multiprocessing.Pool(processes=num_processes) as pool:
                    pool.starmap(plan_batch, pool_inputs)
                    # the plans are saved at the end of each plan_batch process
        
        gc.collect()
        torch.cuda.empty_cache()



        # Loading rollouts from file instead of passing them around as arguments
        # costs a few seconds of time, but sidesteps some multiprocessing difficulties
        # that arise from trying to pass tensors and save to file within children.
        for urdf in urdf_names:
            mpc_save_path  = urdf  + '_mpc_rollouts_iter' + str(int(mbrl_iter)) + '.ptx'
            mpc_save_path = os.path.join(folder, mpc_save_path)
            logging.info('Loading rollouts from ' + mpc_save_path)
            mpc_rollouts_now[urdf] = torch.load( mpc_save_path, map_location=lambda storage, loc: storage)

        logging.info('Plan vel metrics: ')

        for urdf in urdf_names:
            vm, vm_baseline = compare_velocities( # only evaluate the first 6 which are the test directions
                                mpc_rollouts_now[urdf]['states_memory'][0:6], 
                                mpc_rollouts_now[urdf]['goal_memory'][0:6],
                                mpc_rollouts_now[urdf]['run_lens'][0:6],
                                mpc_rollouts_now[urdf]['mpc_n_executed'],
                                mpc_rollouts_now[urdf]['plan_horizon'])
            # logging.info(urdf + ' vel metric: ' + str(vm))
            vel_metric[urdf].append(vm)
            vel_metric_baseline[urdf].append(vm_baseline)
            vm_rescaled = (np.array(vm_baseline)-np.array(vm)
                           )/np.array(vm_baseline)
            logging.info(urdf + ': ' + npstr(vm) + ' baseline ' + npstr(vm_baseline) + 
                ', rescaled: ' + npstr(vm_rescaled))

            # add new data to the mpc_rollouts data collection
            for key in ['states_memory', 'actions_memory', 'torques_memory',
                     'goal_memory', 'run_lens','step_memory']:
                mpc_rollouts[urdf][key] += mpc_rollouts_now[urdf][key]
            mpc_rollouts[urdf]['attachments'] = mpc_rollouts_now[urdf]['attachments']
            mpc_rollouts[urdf]['modules_types'] = mpc_rollouts_now[urdf]['modules_types']
            mpc_rollouts[urdf]['module_sa_len'] = mpc_rollouts_now[urdf]['module_sa_len']
            mpc_rollouts[urdf]['slew_rate_penalty'] = mpc_rollouts_now[urdf]['slew_rate_penalty']


        vel_save_path = os.path.join(folder, 'results' + start_time_str + '.csv')
        results_matrix = []
        names_text = ''
        for urdf in urdf_names:
            # results_matrix.append(vel_metric[urdf])
            vm_rescaled = ( (np.array(vel_metric_baseline[urdf]) 
                            - np.array(vel_metric[urdf]))
                            /np.array(vel_metric_baseline[urdf]) )
            results_matrix.append(vm_rescaled)
            names_text = names_text + urdf + ',' 


        with open(vel_save_path, 'w') as fp:
            fp.write(names_text + '\n')
            np.savetxt(fp, results_matrix, delimiter=',')

            # # plot all the velocity measurement metrics
            # fig, axs = plt.subplots(1, 1)
            # for urdf in urdf_names:
            #     vm_rescaled = (np.array(vel_metric_baseline[urdf] )
            #          -np.array(vel_metric[urdf])
            #        )/np.array(vel_metric_baseline[urdf] )
            #     axs.plot( vm_rescaled, 'o--')
            #     # axs.plot(vel_metric[urdf], 'o--')
            # axs.legend(urdf_names)
            # axs.set_xlabel('Iteration number')
            # axs.set_ylabel('Velocity matching metric')
            # plt.draw()
            # fname = os.path.join(folder, 'vel_metric' + start_time_str+ '.pdf')
            # plt.savefig(fname, facecolor='w', edgecolor='w', format='pdf')



        gc.collect()
        torch.cuda.empty_cache()
        print_time()

        if mbrl_iter < num_mbrl_iters:
        ### Add mpc rollouts to dataset and retrain model
        # (unless this was the last iteration, in which case no need)
            new_data_start_inds = dict()
            for urdf in urdf_names:

                states_memory = random_rollouts[urdf]['states_memory'] + mpc_rollouts[urdf]['states_memory']
                actions_memory = random_rollouts[urdf]['actions_memory'] + mpc_rollouts[urdf]['actions_memory']
                run_lens = random_rollouts[urdf]['run_lens'] + mpc_rollouts[urdf]['run_lens']

                # concatenate data to a long list
                states_memory_tensors[urdf] = [torch.cat(s,0) for s in list(zip(*states_memory)) ]
                actions_memory_tensors[urdf] = [torch.cat(s,0) for s in list(zip(*actions_memory)) ]



                sampleable_inds[urdf] = get_sampleable_inds(
                    run_lens, seq_len)
                n_sampleable = len(sampleable_inds[urdf])
                batch_sizes[urdf] = batch_size_model
                if batch_sizes[urdf] > n_sampleable:
                    batch_sizes[urdf] = n_sampleable

                # Get the part of the sampleable_inds that correspond to the 
                # newest data, so that it can be used more often than the old data
                # in the model retraining
                n_new_runs = len(mpc_rollouts_now[urdf]['run_lens'])
                new_data_start_inds[urdf] = ( len(sampleable_inds[urdf]) 
                    - np.sum(np.array(mpc_rollouts_now[urdf]['run_lens'])
                        -(seq_len-1) ) )


            model_fname = 'shared_trunk_ms'+ str(int(seq_len))+ '_iter' + str(int(mbrl_iter))+'.pt'
            model_fname = os.path.join(folder, model_fname)
            p_train_model = None


            if os.path.exists(model_fname):
                logging.info('Loading weights from ' + model_fname)
                save_dict = torch.load( model_fname, map_location=lambda storage, loc: storage)
                model_network.load_state_dict(save_dict['state_dict'])
            else:

                # Recreate optimizer for model since the old one may have grads on the wrong device
                optimizer_model = torch.optim.Adam(model_network.parameters(), 
                        lr=lr_init/4,
                        weight_decay= weight_decay)

                n_training_steps = 1000
                frac_new_data = 0.1

                logging.info('Retraining model at iter ' + str(int(mbrl_iter)) 
                    + ',   frac_new_data = ' + str(frac_new_data))
                if USE_MULTIPROCESS:
                    logging.info('Running train_model in parallel process')
                    # NOTE: if it quits here, the shared memory might have run out.
                    # if using a docker container, for instance, expand shared memory with --shm-size=10g
                    p_train_model = torch.multiprocessing.Process(
                        target=train_model, 
                        args = (model_fname, urdf_names, n_training_steps, 
                            model_network, optimizer_model,
                            states_memory_tensors, actions_memory_tensors, 
                            modules_types,  attachments,module_state_len,
                            sampleable_inds, batch_sizes, seq_len,
                            backprop_devices[1], weight_decay, 
                            12, new_data_start_inds,
                            frac_new_data, False,True,))

                    p_train_model.start()
                else:
                    train_model(model_fname, urdf_names, n_training_steps, 
                        model_network, optimizer_model,
                        states_memory_tensors, actions_memory_tensors, 
                        modules_types,  attachments,module_state_len,
                        sampleable_inds, batch_sizes, seq_len,
                        backprop_devices[0], weight_decay, 
                        12, new_data_start_inds,
                        frac_new_data, False,True)



        ### Imitation learn (behavioral clone) trajectories into policy
        logging.info('Preparing to train control')
        print_time()

        batch_size_control = 500 # default batch size for control 
        batch_sizes_control = dict()
        sampleable_inds = dict()
        states_memory_tensors = dict()
        actions_memory_tensors = dict() 
        torques_memory_tensors = dict()
        goal_memory_tensors = dict()
        module_sa_len = dict()

        for urdf in urdf_names:     

            # Only use most recent rollouts to train policy.
            # older rollouts would come from models not trained on the newly guided data.

            states_memory = mpc_rollouts_now[urdf]['states_memory'] 
            actions_memory = mpc_rollouts_now[urdf]['actions_memory'] 
            torques_memory = mpc_rollouts_now[urdf]['torques_memory'] 
            goal_memory = mpc_rollouts_now[urdf]['goal_memory'] 
            run_lens = mpc_rollouts_now[urdf]['run_lens'] 
            step_memory = mpc_rollouts_now[urdf]['step_memory'] 

            attachments[urdf] = mpc_rollouts_now[urdf]['attachments'] 
            modules_types[urdf] = mpc_rollouts_now[urdf]['modules_types'] 
            module_sa_len[urdf] = mpc_rollouts_now[urdf]['module_sa_len'] 
            sampleable_inds[urdf] = get_sampleable_inds(run_lens, 3) 
            # ^ will use state0, state1, action1, torque2
            # (last state),(current state),(action),(torque after action)
            n_sampleable = len(sampleable_inds[urdf])
            batch_sizes_control[urdf] = batch_size_control
            if batch_sizes_control[urdf] > n_sampleable:
                batch_sizes_control[urdf] = n_sampleable

            states_memory_tensors[urdf]= [torch.cat(s,0) for s in list(zip(*states_memory)) ]
            actions_memory_tensors[urdf]= [torch.cat(s,0) for s in list(zip(*actions_memory)) ]
            torques_memory_tensors[urdf]= [torch.cat(s,0) for s in list(zip(*torques_memory)) ]
            goal_memory_tensors[urdf] = torch.cat(goal_memory,-1).permute(1,0)

        n_training_steps_control = 7000
        if mbrl_iter == 1 :      
            n_training_steps_control += 1000
            # takes some steps to warm up the gnn
        elif mbrl_iter == num_mbrl_iters:
            n_training_steps_control += 1000 # fine tune for final iter

            # Reset optimizer learning rate. it will decay during training
            for param_group in optimizer_control.param_groups:
                param_group['lr'] = optim_lr_control/2 
                # lower than the original since its warm started


        control_save_path = 'shared_trunk_control_iter' + str(int(mbrl_iter))+'.pt'
        control_save_path = os.path.join(folder, control_save_path)

        ## Load previous weights if they exist
        if os.path.exists(control_save_path):
            logging.info('Loading weights from ' + control_save_path)
            save_dict = torch.load( control_save_path, map_location=lambda storage, loc: storage)
            policy_network.load_state_dict(save_dict['state_dict'])

        else:
            logging.info('Training control ' + control_save_path)     
            n_designs_per_step = len(urdf_names)

            train_control(control_save_path,
                    batch_sizes_control, 
                    n_training_steps_control, backprop_devices[0],
                    optimizer_control, urdf_names, sampleable_inds,
                    states_memory_tensors, actions_memory_tensors, 
                    torques_memory_tensors, goal_memory_tensors,
                    measurement_stds, module_sa_len, 
                    policy_network, modules_types, attachments,
                    torque_loss_weight = 0.25,
                    n_designs_per_step = n_designs_per_step)

        print_time()

        
        logging.info('Training control done.')
        logging.info('Control vel metrics:')
        ### Simulate policy to validate and gather policy rollout data
        # First make some direction goals, then send them in to apply policy
        goal_memory = make_goal_memory(41, device=torch.device('cpu')) # 10*4 + 1
        T = 20
        n_execute = 10
        for design_index in range(len(urdf_names)):
            urdf = urdf_names[design_index]
            apply_policy_save_path = os.path.join(folder, urdf + '_apply_policy_iter' + 
                str(int(mbrl_iter)) + '.ptx')
            if not os.path.exists(apply_policy_save_path):
                # logging.info('Loading weights from ' + control_save_path)
                apply_policy(urdf, design_index, goal_memory,
                  policy_network, torch.device('cpu'), 
                  apply_policy_save_path, show_GUI=False)

            apply_policy_save_dict = torch.load(apply_policy_save_path,
                     map_location=lambda storage, loc: storage)
            vm, vmb = compare_velocities(
                    apply_policy_save_dict['states_memory'],
                    apply_policy_save_dict['goal_memory'], 
                    apply_policy_save_dict['run_lens'],
                    n_execute, T )
            vm_rescaled = (vmb-vm)/vmb
            logging.info(urdf + ': ' + npstr(vm) + ' baseline ' + npstr(vmb) + 
                ', rescaled: ' + npstr(vm_rescaled))

        if p_train_model is not None:
            p_train_model.join() # wait to make sure model retrain is done before moving on
            save_dict = torch.load( model_fname, map_location=lambda storage, loc: storage)
            model_network.load_state_dict(save_dict['state_dict'])  

    logging.info('Loop done after ' +str(num_mbrl_iters) + ' iterations.')
