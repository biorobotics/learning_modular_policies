'''
Source code for paper "Learning modular robot control policies" in Transactions on Robotics
MLP comparisons
Julian Whitman, Dec. 2022. 

Train MLP, with multiple designs
'''

from utils import sample_memory, sample_memory_old_new, to_device
from utils import to_body_frame_batch, from_body_frame_batch, state_add_batch, state_to_fd_input
from utils import divide_state
import numpy as np
import torch
import gc
import logging
import os
from shared_MLP_utils import get_in_out_lens

def train_model(fname, urdf_names, n_training_steps,
    model_network, optimizer,
    states_memory_tensors, actions_memory_tensors,
    modules_types,  attachments, module_state_len,
    sampleable_inds, batch_sizes, seq_len,
    device, weight_decay, 
    n_designs_per_step ,
    new_data_start_inds,
    frac_new_data,
    seq_len_anneal, new_log):

    # n_designs_per_step = 6,
    # new_data_start_inds = None,
    # frac_new_data = 0.05,
    # seq_len_anneal = False
    # frac_new_data is the fraction that the "new data", which starts at 
    # sampleable_inds[new_data_start_inds:], will be used in each sampled batch.
    # This is to allow for the batch to have a fixed ration of new and old data when retraining.
    fd_input_lens, fd_output_lens, policy_input_lens, action_lens,limb_types = get_in_out_lens(urdf_names)

    # # ensure that all the networks are on the desired device.
    model_network = model_network.to(device)

    if new_log:
        # # set up logging
        folder = os.path.dirname(fname)
        log_name = os.path.splitext(fname)[0]
        log_path = os.path.join(folder, log_name+'_log.log')
        logging.basicConfig(level=logging.INFO,
                                format='%(message)s', 
                                filename=log_path,
                                filemode='w')
        console = logging.StreamHandler()
        # console.setLevel(logging.INFO)
        console.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)




    # writer = SummaryWriter(comment=log_name)

    ms_discount = 0.95 # exponential decay on 


    # how many designs to sample each step?
    if len(urdf_names)<n_designs_per_step:
        n_designs_per_step = len(urdf_names)

    logging.info('n_designs_per_step: ' + str(n_designs_per_step))


    # keep track of when the n_designs_per_step changes at a training step
    n_designs_per_step_record = []
    n_designs_per_step_record.append([0,n_designs_per_step])

    # keep track of when the learning rate changes at a training step
    lr_record = []
    for param_group in optimizer.param_groups:
        logging.info( 'LR: ' + str(param_group['lr']) )
        lr_record.append([0, param_group['lr']])

    # keep track of when seq_len is annealed
    if seq_len==2:
        seq_len_anneal = False
    seq_len_record = []
    if seq_len_anneal:
        seq_len_now = 2
        seq_len_now = 3
        logging.info('Seq len now: ' + str(seq_len_now))
    else:
        seq_len_now = seq_len
    seq_len_record.append([0,seq_len_now])

    for training_step in range(n_training_steps):

        # decay learning rate
        if np.mod(training_step,2000 )==0 and training_step>=6000:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr']/2
                logging.info( 'Model LR: ' + str(param_group['lr']) )
                lr_record.append([training_step, param_group['lr']])

        # increase number of design sampled as steps increase
        if ( np.mod(training_step,1500 )==0 and 
             n_designs_per_step<len(urdf_names) and
             training_step>0 ):
             # training_step>0 ):
            n_designs_per_step+=1
            n_designs_per_step_record.append([training_step,n_designs_per_step])

        # pick design indexes
        if n_designs_per_step<len(urdf_names):
            design_inds = np.sort(
            np.random.choice(
                len(urdf_names),n_designs_per_step,
                replace=False))
        else:
            design_inds = list(range(len(urdf_names)))

        # anneal seq_len
        if seq_len_anneal:
            if ( np.mod(training_step,1000 )==0 
                and training_step>0
                and seq_len_now<seq_len):
                seq_len_now+=1
                logging.info('Seq len at step ' + str(training_step) 
                    + ' = ' + str(seq_len_now))
                seq_len_record.append([training_step,seq_len_now])


        # accumulate gradients accros designs but not loss
        optimizer.zero_grad()
        loss_tot_np = 0
        losses_np = np.zeros(len(urdf_names))
        # for urdf in urdf_names:
        for des_ind in design_inds:
            urdf = urdf_names[des_ind]

            batch_size = batch_sizes[urdf]
            # sample without replacement from the full memory, depending on what is sampleable
            if new_data_start_inds is not None:
                # sample without replacement from "old" data and "new" data seperately
                # TODO: calculating batch sizes can be done up front
                sample_inds_old_data = sampleable_inds[urdf][:new_data_start_inds[urdf]]
                sample_inds_new_data = sampleable_inds[urdf][new_data_start_inds[urdf]:]
                batch_size_new_data = int(batch_size*frac_new_data)
                if len(sample_inds_new_data)<batch_size_new_data:
                    batch_size_new_data = len(sample_inds_new_data)
                batch_size_old_data = batch_size-batch_size_new_data

                state_seq, action_seq, sampled_inds = sample_memory_old_new(
                    states_memory_tensors[urdf], actions_memory_tensors[urdf],
                    sample_inds_old_data, sample_inds_new_data,
                    seq_len_now, batch_size_old_data, batch_size_new_data )

            else:
                state_seq, action_seq, sampled_inds = sample_memory(
                            states_memory_tensors[urdf], actions_memory_tensors[urdf],
                            sampleable_inds[urdf], 
                            seq_len_now, batch_size)

            # nan_found= False
            # for s in state_seq:
            #     for s2 in s:
            #         if torch.any(torch.isnan(s2)):
            #             nan_found = True
            #             # logging.info('nan found in ' + urdf + ' states at step ' + str(training_step))
            #             # logging.info(sampled_inds)
            # assert(nan_found==False, 'nan found in ' + urdf + ' states sampled')
            # for s in action_seq:
            #     for s2 in s:
            #         if torch.any(torch.isnan(s2)):
            #             # logging.info('nan found in ' + urdf + ' actions at step ' + str(training_step))
            #             # logging.info(sampled_inds)
            #             nan_found = True
            # assert(nan_found==False, 'nan found in ' + urdf + ' actions sampled')


            loss = 0 # accumulate loss for a single design accross the multistep sequence
            state_approx = to_device(state_seq[0],device) # initial state input is the first in sequence

            for seq in range(seq_len_now-1): # for multistep loss, go through the sequence

                # process states to move them to vehicle frame
                fd_input_real, delta_fd_real = to_body_frame_batch(
                    state_seq[seq], state_seq[seq+1])
                fd_input_approx, R_t = state_to_fd_input(state_approx) # for recursive estimation

                # pass through network
                fd_input   = to_device(fd_input_approx, device) 
                actions_in = to_device(action_seq[seq], device)
                delta_fd   = to_device(delta_fd_real, device) 
                fd_cat = torch.cat(fd_input,-1)
                act_cat =  torch.cat(actions_in,-1)
                if model_network.type == 'shared_trunk':
                    # logging.info(str(fd_cat.shape))
                    # logging.info(str(act_cat.shape))
                    # node_inputs = [torch.cat([s,a],1) for (s,a) in zip(fd_input, actions_in)]
                    state_delta_est_mean, state_delta_var = model_network(
                        fd_cat,act_cat,des_ind)
                    # state_delta_est_mean, state_delta_var = pgnn.run_propagations(
                    #     modules[urdf], attachments[urdf], 2, node_inputs, device)
                else:
                    # logging.info(str(fd_output_lens[des_ind]))
                    # logging.info(str([a.shape for a in fd_input]))
                    # logging.info(str([a.shape for a in actions_in]))
                    state_delta_est_mean, state_delta_var = model_network(
                        torch.split(fd_cat, fd_input_lens[des_ind], dim=-1),
                        torch.split(act_cat, action_lens[des_ind], dim=-1),
                        fd_output_lens[des_ind],
                        limb_types[des_ind])
                    


                # compute loss for this step in sequence
                delta_fd_cat = torch.cat(delta_fd, -1)
                delta_fd_len = delta_fd_cat.shape[-1]
                loss += (torch.sum(
                                (state_delta_est_mean - delta_fd_cat)**2/state_delta_var + 
                                torch.log(state_delta_var) 
                                )/batch_size/(seq_len_now-1)
                        )*(ms_discount**seq)

                # transform back to world frame advance to next sequence step
                if seq_len_now>2:
                    # update recursive state estimation for multistep loss
                    # GNN output is already divided up into modules
                    state_delta_est_mean = divide_state(
                        state_delta_est_mean, module_state_len[urdf])
                    state_approx = from_body_frame_batch(
                        state_approx, state_delta_est_mean)
            

            # after multistep sequence, add loss for this design onto full loss for tracking
            loss_np=(loss.detach().cpu().numpy())
            loss_np = loss_np/delta_fd_len # divide by number of vars when recording for writer
            losses_np[des_ind] = loss_np # leave the loss for each design undivided by n_designs
            loss_tot_np += loss_np/n_designs_per_step

            # divide by number of designs to keep gradients at the same scale
            loss = loss/n_designs_per_step


            # writer.add_scalar('Train' + '/Loss_' + urdf, loss_np, training_step)

            # backward for each design to keep backprop tree smaller
            # backward() accumulates gradients each time until zero_grad 
            # is called. saves gpu memory at the cost of some time.
            loss.backward()

        # optimizer step once we have accumulated grads for all designs
        optimizer.step()
        # writer.add_scalar('Train' + '/Loss_pgnn_multidesign', loss_tot_np, training_step)


        # periodically save the model
        if np.mod(training_step,200)==0 or (training_step==n_training_steps-1) and training_step>0:
            save_dict = dict()
            


            save_dict['state_dict'] =  model_network.state_dict()
            save_dict['n_hidden_layers'] = model_network.n_hidden_layers
            save_dict['hidden_layer_size'] =  model_network.hidden_layer_size
            save_dict['seq_len']=seq_len_now
            save_dict['batch_sizes']=batch_sizes
            save_dict['urdf_names']=urdf_names
            save_dict['weight_decay'] = weight_decay
            save_dict['ms_discount'] = ms_discount
            save_dict['n_designs_per_step_record'] = n_designs_per_step_record
            save_dict['lr_record'] = lr_record
            save_dict['seq_len_record'] = seq_len_record

            save_dict['new_data_start_inds'] = new_data_start_inds
            save_dict['frac_new_data'] = frac_new_data


            torch.save(save_dict,  fname,
                _use_new_zipfile_serialization=False)

            logging.info(
                ('Model losses at iter ' + 
                str(training_step) + ': Net ' +
                str(np.round(loss_tot_np,1)) + 
                ' ' + str(np.round(losses_np,1)) ).replace('\n', '')
                )

    logging.info('Done training model')        
    del fd_input, actions_in, delta_fd, loss
    del state_delta_est_mean, state_delta_var, state_approx

    model_network = model_network.to(torch.device('cpu'))

    gc.collect()
    torch.cuda.empty_cache()

