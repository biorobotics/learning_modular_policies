#!/usr/bin/env python
# coding: utf-8

'''
Source code for paper "Learning modular robot control policies" in Transactions on Robotics
Julian Whitman, Dec. 2022. 

Learn models with different options, for model learning comparison experiment

'''

# import libraries
import torch
from robot_env import robot_env
import numpy as np
from pmlp import pmlp
import pgnn
from utils import get_sampleable_inds, sample_memory
from utils import to_body_frame_batch, from_body_frame_batch
from utils import state_diff_batch, state_to_fd_input, state_add_batch
from utils import divide_state, to_device, detach_list, clip_grads
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

import os
cwd = os.path.dirname(os.path.realpath(__file__))
# cwd = ''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
folder = os.path.join(cwd, 'learned_models')
if not(os.path.exists(folder)):
    os.mkdir(folder)
    print('Created folder ' + folder )
else:
    print('Using folder '  + folder)
    
start_time = datetime.now()
start_time_str = datetime.strftime(start_time, '%Y%m%d_%H%M')

RUN_MLP = True
RUN_SINGLE_GNN = True
RUN_MULTI_GNN = True


# load dataset and gather module configs

urdf_names = ['wnwwnw', 'llllll', 'llwwll']

envs = dict()
run_lens = dict()
states_memory_tensors = dict()
actions_memory_tensors = dict()
module_sa_len = dict()
modules_types = dict()
attachments = dict()

for urdf in urdf_names:

    env = robot_env(show_GUI = False)
    env.reset_terrain()
    env.reset_robot(urdf_name=urdf, randomize_start=False)
    attachments[urdf] = env.attachments
    modules_types[urdf] = env.modules_types
    print('attachments: ' + str(attachments[urdf]))
    print('modules_types: ' + str(modules_types[urdf]))
    n_modules = len(modules_types[urdf])
    envs[urdf] = env
    state = env.get_state()
    module_state_len = []
    for s in state:
        module_state_len.append(len(s))
    module_action_len = list(np.diff(env.action_indexes))
    state_len = sum(module_state_len)
    module_sa_len[urdf] = module_state_len+ module_action_len



for urdf in urdf_names:
    # load dataset
    file_names = []
    folder = 'random_rollouts/'
    found = True
    fname_test = os.path.join(cwd, folder+urdf+'_random_rollouts.ptx')
    if os.path.isfile(fname_test):
        file_names.append(fname_test)
    print('Found files ')
    print(str(file_names))

    states_memory = []
    actions_memory = []
    run_lens[urdf] = []

    for fname in file_names:
        print('loading ' + fname )
        data_in = torch.load(fname)
        states_memory += data_in['states_memory']
        actions_memory += data_in['actions_memory']
        run_lens[urdf] += data_in['run_lens']
        del data_in

    states_memory_tensors[urdf] = [torch.cat(s,0) for s in list(zip(*states_memory)) ]
    actions_memory_tensors[urdf] = [torch.cat(s,0) for s in list(zip(*actions_memory)) ]

print('loaded and merged data')
    
batch_size_default = 500 # default batch size


# select options:
n_training_steps = 30000

condition_tuples = [(100, 2),
                    (1000, 2),
                    (5000, 2)]

#                     (100, 10),
#                     (1000, 10),
#                     (5000, 10)]

# # select data regime (number of data samples to use, low mid or high)
# n_rollouts_to_use = 100 # Low
# n_rollouts_to_use = 1000 # mid
# n_rollouts_to_use = 5000 # high

# # select sequence length for multistep loss
# seq_len = 10
# # seq_len = 2

for condition_tuple in condition_tuples:
    n_rollouts_to_use = condition_tuple[0]
    seq_len = condition_tuple[1]
    condition_run_folder = os.path.join(folder,'runs', start_time_str+ '_' + str(n_rollouts_to_use) + '_' + str(seq_len))
    if not(os.path.exists(condition_run_folder)):
        os.mkdir(condition_run_folder)
        print('Created folder ' + condition_run_folder )
    else:
        print('Using folder '  + condition_run_folder)
    



# train P-MLP
if RUN_MLP:    
    for urdf in urdf_names:

        for condition_tuple in condition_tuples:
            n_rollouts_to_use = condition_tuple[0]
            seq_len = condition_tuple[1]

            print('--- Running condition: ' 
                  + 'n_rollouts=' + str(n_rollouts_to_use)
                  + 'seq_len=' + str(seq_len))

            comment_str = '_pmlp_' + urdf + str(n_rollouts_to_use) + '_' + str(seq_len)
            writer = SummaryWriter(log_dir = os.path.join(folder,'runs',
                        start_time_str+ '_' + str(n_rollouts_to_use) + '_' + str(seq_len)),
                        comment=comment_str)

            # depending on the length of the multistep sequence we want,
            # only some indexes of the full set of states collected can be sampled.
            sampleable_inds = dict()
            batch_sizes = dict()

            sampleable_inds[urdf] = get_sampleable_inds(
                run_lens[urdf][:n_rollouts_to_use], seq_len)
            n_sampleable = len(sampleable_inds[urdf])
            batch_sizes[urdf] = batch_size_default
            if batch_sizes[urdf] > n_sampleable:
                batch_sizes[urdf] = n_sampleable
            print(urdf + ' using ' + str(n_rollouts_to_use) + ' out of Rollouts ' + str(len(run_lens[urdf])))



            batch_size = batch_sizes[urdf]
            n_modules = len(modules_types[urdf])
            module_state_len = module_sa_len[urdf][:n_modules]
            # initialize network and optimizer
            input_len = sum(module_sa_len[urdf]) - 3
            output_len = sum(module_state_len)
            hidden_layer_size = 300
            n_hidden_layers = 5
            fd_network = pmlp(input_len = input_len, output_len=output_len,
                n_hidden_layers = n_hidden_layers, hidden_layer_size=hidden_layer_size
                ).to(device)
            weight_decay = 1e-4
            optimizer =  torch.optim.Adam(fd_network.parameters(),lr=1e-3, weight_decay = weight_decay) 

            num_nn_params=0
            for p in fd_network.parameters():
                nn=1
                for s in list(p.size()):
                    nn = nn*s
                num_nn_params += nn
            print('Num NN params: ' + str(num_nn_params))

            for training_step in range( n_training_steps):

                if np.mod(training_step,5000 )==0 and training_step>10000:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = param_group['lr']/2
                        print( 'LR: ' + str(param_group['lr']) )

                # sample without replacement from the full memory, depending on what is sampleable
                state_seq, action_seq, sampled_inds = sample_memory(
                                states_memory_tensors[urdf], 
                                actions_memory_tensors[urdf],
                                sampleable_inds[urdf], seq_len, batch_size)

                loss = 0
                state_approx = to_device(state_seq[0],device) # initial state input is the first in sequence
                for seq in range(seq_len-1): # for multistep loss, go through the sequence

                    # process states to move them to vehicle frame
                    fd_input_real, delta_fd_real = to_body_frame_batch(state_seq[seq], state_seq[seq+1])
                    fd_input_approx, R_t = state_to_fd_input(state_approx) # for recursive estimation

                    # pass through network
                    fd_input = torch.cat(fd_input_approx,1).to(device)
                    actions_in = torch.cat(action_seq[seq],1).to(device)
                    delta_fd = torch.cat(delta_fd_real,1).to(device)
                    state_delta_est_mean, state_delta_est_var = fd_network(fd_input, actions_in)

                    # compute loss for this step in sequence
                    loss += torch.sum(
                         (state_delta_est_mean - delta_fd)**2/state_delta_est_var
                         + torch.log(state_delta_est_var)
                         )/batch_size/(seq_len-1)   

                    # transform back to world frame advance to next sequence step
                    if seq_len>2:
                        # divide MLP output divided up into modules
                        delta_fd_approx = divide_state(state_delta_est_mean, module_state_len)
                        # update recursive state estimation for multistep loss  
                        state_approx = from_body_frame_batch(state_approx, delta_fd_approx)


                # backprop and optimizer step 
                loss_np = loss.detach().cpu().numpy()
                fd_network.zero_grad()
                loss.backward()
                optimizer.step()

                writer.add_scalar('Train' + '/Loss_pmlp', loss_np, training_step)


                # periodically save the model

                if np.mod(training_step,500)==0:
                    PATH = ('learned_models/' + urdf + '_pmlp_r' + str(int(n_rollouts_to_use)) + 
                            '_ms'+ str(int(seq_len))+'.pt')
                    PATH = os.path.join(cwd, PATH)
                    fd_network_state_dict=fd_network.state_dict()
                    torch.save({'fd_network_state_dict':fd_network_state_dict,
                        'fd_network_input_len':fd_network.input_len,
                        'fd_network_output_len':fd_network.output_len,
                        'fd_network_n_hidden_layers':fd_network.n_hidden_layers,
                        'fd_network_hidden_layer_size':fd_network.hidden_layer_size,
                        'n_rollouts_to_use':n_rollouts_to_use,
                        'seq_len':seq_len,
                        'batch_size':batch_size,
                        'urdf':urdf,
                        'num_nn_params':num_nn_params,
                        'weight_decay':weight_decay
                        },  PATH)  
                    print('Training losses at iter ' + 
                            str(training_step) + ': ' + 
                            str(np.round(loss_np,2)))
            del fd_input, actions_in, delta_fd, fd_network, loss, optimizer
            torch.cuda.empty_cache()


# train P-GNN
# urdf = urdf_to_train
if RUN_SINGLE_GNN:
    for urdf in urdf_names:
        for condition_tuple in condition_tuples:

            n_rollouts_to_use = condition_tuple[0]
            seq_len = condition_tuple[1]

            comment_str = '_pgnn_' + urdf + str(n_rollouts_to_use) + '_' + str(seq_len)
            writer = SummaryWriter(log_dir = os.path.join(folder,'runs',
                        start_time_str+ '_' + str(n_rollouts_to_use) + '_' + str(seq_len)),
                        comment=comment_str)


            # depending on the length of the multistep sequence we want,
            # only some indexes of the full set of states collected can be sampled.
            sampleable_inds = dict()
            batch_sizes = dict()
        #     for urdf in urdf_names:
            sampleable_inds[urdf] = get_sampleable_inds(run_lens[urdf][:n_rollouts_to_use], seq_len)
            n_sampleable = len(sampleable_inds[urdf])
            batch_sizes[urdf] = batch_size_default
            if batch_sizes[urdf] > n_sampleable:
                batch_sizes[urdf] = n_sampleable
            print(urdf + ' using ' + str(n_rollouts_to_use) + ' out of Rollouts ' + str(len(run_lens[urdf])))


            batch_size = batch_sizes[urdf]
            n_modules = len(modules_types[urdf])
            module_state_len = module_sa_len[urdf][:n_modules]

            # initialize network and optimizer
            internal_state_len = 100
            message_len = 50
            hidden_layer_size = 250
            weight_decay = 1e-4
            gnn_nodes = pgnn.create_GNN_nodes(internal_state_len, message_len, hidden_layer_size, 
                            device, body_input = True)
            optimizer = torch.optim.Adam(pgnn.get_GNN_params_list(gnn_nodes), 
                                         lr=1e-3,
                                weight_decay= weight_decay)# create module containers for the nodes
            modules = []
            for i in range(n_modules):
                modules.append(pgnn.Module(i, gnn_nodes[modules_types[urdf][i]], device))


            num_nn_params=0
            for p in pgnn.get_GNN_params_list(gnn_nodes):
                nn=1
                for s in list(p.size()):
                    nn = nn*s
                num_nn_params += nn

            print('Num NN params: ' + str(num_nn_params))


            for training_step in range(n_training_steps):

                if np.mod(training_step,5000 )==0 and training_step>10000:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = param_group['lr']/2
                        print( 'LR: ' + str(param_group['lr']) )

                # sample without replacement from the full memory, depending on what is sampleable
                state_seq, action_seq, sampled_inds = sample_memory(
                                states_memory_tensors[urdf], actions_memory_tensors[urdf],
                                sampleable_inds[urdf], seq_len, batch_size)

                loss = 0
                state_approx = to_device(state_seq[0],device) # initial state input is the first in sequence
                for seq in range(seq_len-1): # for multistep loss, go through the sequence

                    for module in modules: # must reset module lstm state
                        module.reset_hidden_states(batch_size) 

                    # process states to move them to vehicle frame
                    fd_input_real, delta_fd_real = to_body_frame_batch(state_seq[seq], state_seq[seq+1])
                    fd_input_approx, R_t = state_to_fd_input(state_approx) # for recursive estimation

                    # pass through network
                    fd_input   = to_device(fd_input_approx, device) 
                    actions_in = to_device(action_seq[seq], device)
                    delta_fd   = to_device(delta_fd_real, device) 
                    node_inputs = [torch.cat([s,a],1) for (s,a) in zip(fd_input, actions_in)]
                    state_delta_est_mean, state_delta_var = pgnn.run_propagations(
                        modules, attachments[urdf], 2, node_inputs, device)

                    # compute loss for this step in sequence
                    for mm in range(len(state_delta_est_mean)):
                        loss += torch.sum(
                            (state_delta_est_mean[mm] - delta_fd[mm])**2/state_delta_var[mm] + 
                            torch.log(state_delta_var[mm]) 
                                        )/batch_size/(seq_len-1)

                    # transform back to world frame advance to next sequence step
                    if seq_len>2:
                        # update recursive state estimation for multistep loss
                        # GNN output is already divided up into modules
                        delta_fd_approx = state_delta_est_mean
                        state_approx = from_body_frame_batch(state_approx, delta_fd_approx)



                # backprop and optimizer step 
                loss_np=(loss.detach().cpu().numpy())
                pgnn.zero_grads(gnn_nodes)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                writer.add_scalar('Train' + '/Loss_pgnn', loss_np, training_step)


                # periodically save the model
                if np.mod(training_step,500)==0:
                    PATH = ('learned_models/' + urdf + '_pgnn_r' + str(int(n_rollouts_to_use)) + 
                            '_ms'+ str(int(seq_len))+'.pt')
                    PATH = os.path.join(cwd, PATH)
                    gnn_state_dicts=(pgnn.get_state_dicts(gnn_nodes))
                    save_dict = dict()
                    save_dict['gnn_state_dicts'] =  gnn_state_dicts
                    save_dict['internal_state_len'] = gnn_nodes[0].internal_state_len
                    save_dict['message_len'] = gnn_nodes[0].message_len
                    save_dict['hidden_layer_size'] = gnn_nodes[0].hidden_layer_size
                    save_dict['n_rollouts_to_use']=n_rollouts_to_use
                    save_dict['seq_len']=seq_len
                    save_dict['batch_size']=batch_size
                    save_dict['urdf']=urdf
                    save_dict['weight_decay'] = weight_decay
                    save_dict['num_nn_params'] = num_nn_params

                    torch.save(save_dict,  PATH)

                    print('Training losses at iter ' + 
                        str(training_step) + ': ' + 
                        str(np.round(loss_np,2)))

            del fd_input, actions_in, delta_fd,  loss, optimizer
            torch.cuda.empty_cache()



if RUN_MULTI_GNN:
    # train P-GNN, with multiple designs. this will be N_des times slower
    for condition_tuple in condition_tuples:
        n_rollouts_to_use = condition_tuple[0]
        seq_len = condition_tuple[1]


        comment_str = '_pgnn_multi_' + str(n_rollouts_to_use) + '_' + str(seq_len)
        writer = SummaryWriter(log_dir = os.path.join(folder,'runs',
                    start_time_str+ '_' + str(n_rollouts_to_use) + '_' + str(seq_len)),
                    comment=comment_str)


        # depending on the length of the multistep sequence we want,
        # only some indexes of the full set of states collected can be sampled.
        sampleable_inds = dict()
        batch_sizes = dict()
        for urdf in urdf_names:
            sampleable_inds[urdf] = get_sampleable_inds(run_lens[urdf][:n_rollouts_to_use], seq_len)
            n_sampleable = len(sampleable_inds[urdf])
            batch_sizes[urdf] = batch_size_default
            if batch_sizes[urdf] > n_sampleable:
                batch_sizes[urdf] = n_sampleable
            print(urdf + ' using ' + str(n_rollouts_to_use) + ' out of Rollouts ' + str(len(run_lens[urdf])))


        # initialize network and optimizer
        internal_state_len = 100
        message_len = 50
        hidden_layer_size = 250
        weight_decay = 1e-4
        gnn_nodes = pgnn.create_GNN_nodes(internal_state_len, message_len, hidden_layer_size, 
                        device, body_input = True)
        optimizer = torch.optim.Adam(pgnn.get_GNN_params_list(gnn_nodes), 
                                     lr=1e-3,
                            weight_decay= weight_decay)# create module containers for the nodes


        modules = dict()
        for urdf in urdf_names:
            modules[urdf] = []
            n_modules = len(modules_types[urdf])
            for i in range(n_modules):
                modules[urdf].append(pgnn.Module(i, gnn_nodes[modules_types[urdf][i]], device))

        num_nn_params=0
        for p in pgnn.get_GNN_params_list(gnn_nodes):
            nn=1
            for s in list(p.size()):
                nn = nn*s
            num_nn_params += nn
        print('Num NN params: ' + str(num_nn_params))


        for training_step in range(n_training_steps):

            if np.mod(training_step,5000 )==0 and training_step>10000:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['lr']/2
                    print( 'LR: ' + str(param_group['lr']) )

            # accumulate gradients accros designs but not loss
            optimizer.zero_grad()
            loss_tot_np = 0
            for urdf in urdf_names:
                batch_size = batch_sizes[urdf]

                # sample without replacement from the full memory, depending on what is sampleable
                state_seq, action_seq, sampled_inds = sample_memory(
                                states_memory_tensors[urdf], actions_memory_tensors[urdf],
                                sampleable_inds[urdf], seq_len, batch_size)

                loss = 0 # accumulate loss for a single design accross the multistep sequence
                state_approx = to_device(state_seq[0],device) # initial state input is the first in sequence
                for seq in range(seq_len-1): # for multistep loss, go through the sequence

                    for module in modules[urdf]: # must reset module lstm state
                        module.reset_hidden_states(batch_size) 

                    # process states to move them to vehicle frame
                    fd_input_real, delta_fd_real = to_body_frame_batch(state_seq[seq], state_seq[seq+1])
                    fd_input_approx, R_t = state_to_fd_input(state_approx) # for recursive estimation

                    # pass through network
                    fd_input   = to_device(fd_input_approx, device) 
                    actions_in = to_device(action_seq[seq], device)
                    delta_fd   = to_device(delta_fd_real, device) 
                    node_inputs = [torch.cat([s,a],1) for (s,a) in zip(fd_input, actions_in)]
                    state_delta_est_mean, state_delta_var = pgnn.run_propagations(
                        modules[urdf], attachments[urdf], 2, node_inputs, device)

                    # compute loss for this step in sequence
                    for mm in range(len(state_delta_est_mean)):
                        loss += torch.sum(
                            (state_delta_est_mean[mm] - delta_fd[mm])**2/state_delta_var[mm] + 
                            torch.log(state_delta_var[mm]) 
                                        )/batch_size/(seq_len-1)

                    # transform back to world frame advance to next sequence step
                    if seq_len>2:
                        # update recursive state estimation for multistep loss
                        # GNN output is already divided up into modules
                        delta_fd_approx = state_delta_est_mean
                        state_approx = from_body_frame_batch(state_approx, delta_fd_approx)

                # after multistep sequence, add loss for this design onto full loss for tracking
                loss_np=(loss.detach().cpu().numpy())
                loss_tot_np += loss_np

                # backward for each design to keep compute tree smaller
                loss.backward()

            # optimizer step once we have accumulated grads for all designs
            optimizer.step()
            writer.add_scalar('Train' + '/Loss_pgnn_multidesign', loss_tot_np, training_step)


            # periodically save the model
            if np.mod(training_step,100)==0:
                PATH = ('learned_models/' + 'multidesign_pgnn_r' + str(int(n_rollouts_to_use)) + 
                        '_ms'+ str(int(seq_len))+'.pt')
                PATH = os.path.join(cwd, PATH)
                gnn_state_dicts=(pgnn.get_state_dicts(gnn_nodes))
                save_dict = dict()
                save_dict['gnn_state_dicts'] =  gnn_state_dicts
                save_dict['internal_state_len'] = gnn_nodes[0].internal_state_len
                save_dict['message_len'] = gnn_nodes[0].message_len
                save_dict['hidden_layer_size'] = gnn_nodes[0].hidden_layer_size
                save_dict['n_rollouts_to_use']=n_rollouts_to_use
                save_dict['seq_len']=seq_len
                save_dict['batch_sizes']=batch_sizes
                save_dict['urdf_names']=urdf_names
                save_dict['weight_decay'] = weight_decay
                save_dict['num_nn_params'] = num_nn_params
                torch.save(save_dict,  PATH)

                print('Training losses at iter ' + 
                    str(training_step) + ': ' + 
                    str(np.round(loss_tot_np,2)))

        del fd_input, actions_in, delta_fd
        torch.cuda.empty_cache()



# evaluation: load validation data set
states_memory_validation =dict()
actions_memory_validation =dict()
run_lens_validation =dict()
for urdf in urdf_names:
    
    file_names = []
    folder = os.path.join(cwd, 'random_rollouts')
    found = True
    fname_test = os.path.join(folder,urdf+'_random_rollouts_validation.ptx')
    if os.path.isfile(fname_test):
        file_names.append(fname_test)
    print('Found files ')
    print(str(file_names))
    
    states_memory_validation[urdf] = []
    actions_memory_validation[urdf] = []
    run_lens_validation[urdf] = []

    for fname in file_names:
        print('loading ' + fname )
        data_in = torch.load(fname)
        states_memory_validation[urdf] += data_in['states_memory']
        actions_memory_validation[urdf] += data_in['actions_memory']
        run_lens_validation[urdf] += data_in['run_lens']
        del data_in

#     states_memory_tensors[urdf] = [torch.cat(s,0) for s in list(zip(*states_memory)) ]
#     actions_memory_tensors[urdf] = [torch.cat(s,0) for s in list(zip(*actions_memory)) ]

print('loaded and merged data')




# compute constant prediction baseline:
# If we were to always predict that delta_fd = 0, 
# what would the error be? Used as a baseline.
# urdf = urdf_names[0]
const_pred_diff_list = dict()
for urdf in urdf_names:
    const_pred_diff_list[urdf] = []
    for run_index in range(len(run_lens_validation[urdf])):

        state_seq0 = [s[:-1] for s in states_memory_validation[urdf][run_index]]
        state_seq1 = [s[1:] for s in states_memory_validation[urdf][run_index]]
        action_seq = [a[:-1] for a in actions_memory_validation[urdf][run_index]]

        # process states to move them to vehicle frame
        fd_input_real, delta_fd_real = to_body_frame_batch(state_seq0, state_seq1)
        delta_fd = torch.cat(delta_fd_real,1)

        const_pred_diff_list[urdf].append(delta_fd)
    print(urdf, ' Constant pred baseline:')
    print( torch.abs(torch.cat(const_pred_diff_list[urdf],0)).sum(-1).mean() )




# Evalute MLP accuracy
for urdf in urdf_names:

    for condition_tuple in condition_tuples:
    
        seq_len = condition_tuple[1]
        n_rollouts_to_use = condition_tuple[0]


        # load network
        PATH = os.path.join(cwd,'learned_models',
                            urdf + '_pmlp_r' +
                            str(int(n_rollouts_to_use)) + 
                            '_ms'+ str(int(seq_len))+'.pt')

        if os.path.exists(PATH):
            save_dict = torch.load( PATH)#, map_location=lambda storage, loc: storage)
            input_len, output_len = save_dict['fd_network_input_len'], save_dict['fd_network_output_len']
            n_hidden_layers = save_dict['fd_network_n_hidden_layers']
            hidden_layer_size = save_dict['fd_network_hidden_layer_size']

            fd_network = pmlp(input_len = input_len, output_len=output_len,
                n_hidden_layers = n_hidden_layers, hidden_layer_size=hidden_layer_size
                ).to(device)
            fd_network.load_state_dict(save_dict['fd_network_state_dict'])
            fd_network.eval()


            diff_list = []
            diff_list_by_run = []
            for run_index in range(len(run_lens_validation[urdf])):
                with torch.no_grad():
                    state_seq0 = [s[:-1] for s in states_memory_validation[urdf][run_index]]
                    state_seq1 = [s[1:] for s in states_memory_validation[urdf][run_index]]
                    action_seq = [a[:-1] for a in actions_memory_validation[urdf][run_index]]

                    # process states to move them to vehicle frame
                    fd_input_real, delta_fd_real = to_body_frame_batch(state_seq0, state_seq1)

                    # pass through network
                    fd_input = torch.cat(fd_input_real,1).to(device)
                    actions_in = torch.cat(action_seq,1).to(device)
                    delta_fd = torch.cat(delta_fd_real,1).to(device)
                    state_delta_est_mean, state_delta_est_var = fd_network(fd_input, actions_in)

                    diff = (delta_fd - state_delta_est_mean)
                    diff_list_by_run.append(torch.abs(diff).sum(-1).mean())


                diff_list.append(diff)
            diff_list_all = torch.cat(diff_list,0)
            print(urdf + ' ' + str(condition_tuple) + ' MLP differences:')
            diffs_abs = torch.abs(diff_list_all).sum(-1)

            print('Mean: ' + str(diffs_abs.mean().item() ))
            print('Std: ' + str(diffs_abs.std().item()))

#     print('MLP relative to const baseline mean:')
#     const_pred = torch.abs(torch.cat(const_pred_diff_list[urdf],0)).sum(-1).to(device)
#     diff_rel = diffs_abs/const_pred
#     print('Mean: ' + str(diff_rel.mean().item() ))
#     print('Std: ' + str(diff_rel.std().item()))
#     # print(np.mean(diff_list/const_pred_diff_list[urdf]))

#     print(urdf + ' MLP differences by run:')
#     diffs_sums = torch.stack(diff_list_by_run,-1)
#     print('Mean: ' + str(diffs_sums.mean().item() ))
#     print('Std: ' + str(diffs_sums.std().item()))
del fd_network, fd_input, actions_in, delta_fd
torch.cuda.empty_cache()




# Evalute GNN accuracy for one design
# urdf = urdf_names[0]
for urdf in urdf_names:
    for condition_tuple in condition_tuples:
        seq_len = condition_tuple[1]
        n_rollouts_to_use =  condition_tuple[0]

        # load network
        PATH = os.path.join(cwd,'learned_models', urdf + '_pgnn_r' + str(int(n_rollouts_to_use)) + 
                '_ms'+ str(int(seq_len))+'.pt')

        if os.path.exists(PATH):

            save_dict = torch.load( PATH)#, map_location=lambda storage, loc: storage)
            internal_state_len = save_dict['internal_state_len']
            message_len= save_dict['message_len']
            hidden_layer_size= save_dict['hidden_layer_size']

            gnn_nodes = pgnn.create_GNN_nodes(internal_state_len, message_len, hidden_layer_size, 
                            device, body_input = True)
            pgnn.load_state_dicts(gnn_nodes, save_dict['gnn_state_dicts'])
            for gnn_node in gnn_nodes:
                gnn_node.eval()

            modules = dict()
            modules[urdf] = []
            n_modules = len(modules_types[urdf])
            for i in range(n_modules):
                modules[urdf].append(pgnn.Module(i, gnn_nodes[modules_types[urdf][i]], device))


            diff_list = dict()
            diff_list_by_run =  dict()

            diff_list[urdf] = []
            diff_list_by_run[urdf] = []
            for run_index in range(len(run_lens_validation[urdf])):
                batch_size = run_lens_validation[urdf][run_index]-1
                with torch.no_grad():
                    state_seq0 = [s[:-1] for s in states_memory_validation[urdf][run_index]]
                    state_seq1 = [s[1:]  for s in states_memory_validation[urdf][run_index]]
                    action_seq = [a[:-1] for a in actions_memory_validation[urdf][run_index]]

                    for module in modules[urdf]: # must reset module lstm state
                        module.reset_hidden_states(batch_size) 

                    # process states to move them to vehicle frame
                    fd_input_real, delta_fd_real = to_body_frame_batch(state_seq0, state_seq1)

                    # pass through network
                    fd_input   = to_device(fd_input_real, device) 
                    actions_in = to_device(action_seq, device)
                    node_inputs = [torch.cat([s,a],1) for (s,a) in zip(fd_input, actions_in)]
                    state_delta_est_mean, state_delta_var = pgnn.run_propagations(
                        modules[urdf], attachments[urdf], 2, node_inputs, device)

                    # cat to one tensor and take difference
                    state_delta_est_mean = torch.cat(state_delta_est_mean,-1)
                    delta_fd   = torch.cat(to_device(delta_fd_real, device),-1)
                    diff = (delta_fd - state_delta_est_mean)

                diff_list_by_run[urdf].append(torch.abs(diff).sum(-1).mean())   
                diff_list[urdf].append(diff)

            print('------------------------------')    
            diff_list_all = torch.cat(diff_list[urdf],0)
            print(urdf + ' ' + str(condition_tuple)+ ' GNN differences:')
            diffs_abs = torch.abs(diff_list_all).sum(-1)

            print('Mean: ' + str(diffs_abs.mean().item() ))
            print('Std: ' + str(diffs_abs.std().item()))

#         print(urdf + ' GNN relative to const baseline mean:')
#         const_pred = torch.abs(torch.cat(const_pred_diff_list[urdf],0)).sum(-1).to(device)
#         diff_rel = diffs_abs/const_pred
#         print('Mean: ' + str(diff_rel.mean().item() ))
#         print('Std: ' + str(diff_rel.std().item()))

#         print(urdf + ' GNN differences by run:')
#         diffs_sums = torch.stack(diff_list_by_run[urdf],-1)
#         print('Mean: ' + str(diffs_sums.mean().item() ))
#         print('Std: ' + str(diffs_sums.std().item()))




# Evalute GNN accuracy for all designs
for condition_tuple in condition_tuples:
    seq_len = condition_tuple[1]
    n_rollouts_to_use = condition_tuple[0]
    
    # load network

    PATH = os.path.join(cwd,
            'learned_models', 'multidesign_pgnn_r' + str(int(n_rollouts_to_use)) + 
            '_ms'+ str(int(seq_len))+'.pt')

    print(PATH)
    save_dict = torch.load( PATH)#, map_location=lambda storage, loc: storage)
    internal_state_len = save_dict['internal_state_len']
    message_len= save_dict['message_len']
    hidden_layer_size= save_dict['hidden_layer_size']

    gnn_nodes = pgnn.create_GNN_nodes(internal_state_len, message_len, hidden_layer_size, 
                    device, body_input = True)
    pgnn.load_state_dicts(gnn_nodes, save_dict['gnn_state_dicts'])
    for gnn_node in gnn_nodes:
        gnn_node.eval()

        
    modules = dict()
    for urdf in urdf_names:
        modules[urdf] = []
        n_modules = len(modules_types[urdf])
        for i in range(n_modules):
            modules[urdf].append(pgnn.Module(i, gnn_nodes[modules_types[urdf][i]], device))


    diff_list = dict()
    diff_list_by_run =  dict()

    for urdf in urdf_names:
        diff_list[urdf] = []
        diff_list_by_run[urdf] = []
        for run_index in range(len(run_lens_validation[urdf])):
            batch_size = run_lens_validation[urdf][run_index]-1
            with torch.no_grad():
                state_seq0 = [s[:-1] for s in states_memory_validation[urdf][run_index]]
                state_seq1 = [s[1:]  for s in states_memory_validation[urdf][run_index]]
                action_seq = [a[:-1] for a in actions_memory_validation[urdf][run_index]]

                for module in modules[urdf]: # must reset module lstm state
                    module.reset_hidden_states(batch_size) 

                # process states to move them to vehicle frame
                fd_input_real, delta_fd_real = to_body_frame_batch(state_seq0, state_seq1)

                # pass through network
                fd_input   = to_device(fd_input_real, device) 
                actions_in = to_device(action_seq, device)
                node_inputs = [torch.cat([s,a],1) for (s,a) in zip(fd_input, actions_in)]
                state_delta_est_mean, state_delta_var = pgnn.run_propagations(
                    modules[urdf], attachments[urdf], 2, node_inputs, device)

                # cat to one tensor and take difference
                state_delta_est_mean = torch.cat(state_delta_est_mean,-1)
                delta_fd   = torch.cat(to_device(delta_fd_real, device),-1)
                diff = (delta_fd - state_delta_est_mean)

            diff_list_by_run[urdf].append(torch.abs(diff).sum(-1).mean())   
            diff_list[urdf].append(diff)

        print('------------------------------')    
        diff_list_all = torch.cat(diff_list[urdf],0)
        print(urdf + ' GNN differences:')
        diffs_abs = torch.abs(diff_list_all).sum(-1)

        print('Mean: ' + str(diffs_abs.mean().item() ))
        print('Std: ' + str(diffs_abs.std().item()))

#         print(urdf + ' GNN relative to const baseline mean:')
#         const_pred = torch.abs(torch.cat(const_pred_diff_list[urdf],0)).sum(-1).to(device)
#         diff_rel = diffs_abs/const_pred
#         print('Mean: ' + str(diff_rel.mean().item() ))
#         print('Std: ' + str(diff_rel.std().item()))

#         print(urdf + ' GNN differences by run:')
#         diffs_sums = torch.stack(diff_list_by_run[urdf],-1)
#         print('Mean: ' + str(diffs_sums.mean().item() ))
#         print('Std: ' + str(diffs_sums.std().item()))



