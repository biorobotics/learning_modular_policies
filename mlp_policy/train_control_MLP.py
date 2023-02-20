'''
Source code for paper "Learning modular robot control policies" in Transactions on Robotics
MLP comparisons
Julian Whitman, Dec. 2022. 

Train a MLP for control based on the data collected during mpc

Use last state in observation

'''
import torch
import numpy as np
from utils import get_sampleable_inds, sample_memory, wrap_to_pi
import os
from robot_env import robot_env
from utils import rotate, create_control_inputs
import logging
from shared_MLP_utils import get_in_out_lens
from datetime import datetime


    
def train_control(control_save_path,
        batch_sizes, n_training_steps, device,
        optimizer, urdf_names, sampleable_inds,
        states_memory_tensors, actions_memory_tensors, 
        torques_memory_tensors, goal_memory_tensors,
        measurement_stds, module_sa_len, 
        policy_network, modules_types, attachments,
        torque_loss_weight = 1,
        n_designs_per_step = 6):
    
    fd_input_lens, fd_output_lens, policy_input_lens,action_lens,limb_types = get_in_out_lens(urdf_names)

    # # ensure that all the networks are on the desired device.
    policy_network = policy_network.to(device)


    # how many designs to sample each step?
    if len(urdf_names)<n_designs_per_step:
        n_designs_per_step = len(urdf_names)
    logging.info('Control n_designs_per_step: ' + str(n_designs_per_step))

    # keep track of when the n_designs_per_step changes at a training step
    n_designs_per_step_record = []
    n_designs_per_step_record.append([0,n_designs_per_step])

    # keep track of when the learning rate changes at a training step
    lr_record = []
    for param_group in optimizer.param_groups:
        logging.info( 'Control LR: ' + str(param_group['lr']) )
        lr_record.append([0, param_group['lr']])

    for training_step in range(n_training_steps+1):
        
        if np.mod(training_step, 1000)==0 and training_step>=3000:
            for param_group in optimizer.param_groups:
                if param_group['lr']>3e-5:
                    # half the learning rate periodically
                    param_group['lr'] = param_group['lr']/2.
                    logging.info( 'Control LR: ' + str(param_group['lr']) )
                    lr_record.append([training_step, param_group['lr']])


        # increase number of design sampled as steps increase
        if ( np.mod(training_step,1500 )==0 and 
             n_designs_per_step<len(urdf_names) and
             training_step>0 ):
            n_designs_per_step+=1
            n_designs_per_step_record.append([training_step,n_designs_per_step])

        
        if n_designs_per_step<len(urdf_names):
            design_inds = np.sort(
            np.random.choice(
                len(urdf_names),n_designs_per_step,
                replace=False))
        else:
            design_inds = list(range(len(urdf_names)))


        optimizer.zero_grad()

        loss = 0
        losses_np = np.zeros(len(urdf_names))

        # for urdf in urdf_names:
        for des_ind in design_inds:
            urdf = urdf_names[des_ind]

            batch_size = batch_sizes[urdf]
            
            n_modules = len(modules_types[urdf])
            module_action_len = module_sa_len[urdf][n_modules:]
            module_state_len = module_sa_len[urdf][:n_modules]

            # can't sample all actions as last one in each sequence has nan actions.
            sampled_inds = sampleable_inds[urdf][
                            np.random.choice(len(sampleable_inds[urdf]), 
                                         batch_size, replace=False)]
            # raw data is in world frame
            states0 = [smm[sampled_inds].to(device) for smm in states_memory_tensors[urdf]]
            # states1 = [smm[sampled_inds+1].to(device) for smm in states_memory_tensors[urdf]]
            actions = [amm[sampled_inds+1].to(device) for amm in actions_memory_tensors[urdf]]
            # torques are actually the "next state" torques used for feedforward
            torques = [amm[sampled_inds+2].to(device) for amm in torques_memory_tensors[urdf]]
            goals_world = goal_memory_tensors[urdf][sampled_inds+1,:].to(device)
            
            # add on white noise
            for si in range(len(states0)):
                noise = torch.distributions.Normal(0.0, measurement_stds[urdf][si])
                states0[si] += noise.sample((batch_size,)).to(device)



            # # goals_world[x,y] are recorded in world frame. shift to body frame here.
            inputs, goals = create_control_inputs(states0, goals_world)
            inputs =  torch.cat(inputs,-1)
            if policy_network.type=='shared_trunk':
                # logging.info('states0 shapes ' + str([str(s.shape) for s in states0]))
                # logging.info(str(inputs.shape) + ' , ' + str(goals.shape))
                u_mean, u_var, tau_mean, tau_var = policy_network(
                   inputs, goals, des_ind)
            else:
                u_mean, u_var, tau_mean, tau_var = policy_network(
                   torch.split(inputs, policy_input_lens[des_ind], dim=-1),
                    goals, action_lens[des_ind],limb_types[des_ind])
                
            u_out_mean = torch.split(u_mean, module_action_len, dim=-1)
            u_out_var = torch.split(u_var, module_action_len, dim=-1)
            t_out_mean = torch.split(tau_mean, module_action_len, dim=-1)
            t_out_var = torch.split(tau_var, module_action_len, dim=-1)

            loss_m = 0
            sum_module_action_len = sum(module_action_len)
            for mm in range(n_modules):
                # backprop appears to treat empty tensors poorly, so make sure its not empty
                if module_action_len[mm]>0:
                    # loss for velocity command
                    loss_v = torch.sum(
                        (u_out_mean[mm]
                         - actions[mm])**2/u_out_var[mm] + 
                        torch.log(u_out_var[mm]) 
                                    )/batch_size
                    loss_m += loss_v#/sum_module_action_len # divide out the number of actions

                    # loss for torque value
                    losses_tau = ( (t_out_mean[mm] 
                            - torques[mm])**2/t_out_var[mm]+ 
                            torch.log(t_out_var[mm]) )
                    # There is a bug in pybullet where for some joints it returns zeros
                    # for the joint torque (exactly 0). 
                    # I could not fix this bug in pybullet, so,
                    # To help with this, ignore those entries.
                    losses_tau[torques[mm]==0] = 0 # overwrite entries where the torque is buggy
                    loss_tau = torch.sum(losses_tau)/batch_size

                    loss_m += loss_tau*torque_loss_weight

            loss_m_np = loss_m.detach().cpu().numpy()
            losses_np[des_ind] = loss_m_np # leave the loss for each design undivided by n_designs
            loss += loss_m/n_designs_per_step
            
            # writer.add_scalar('Train' + '/Loss_' + urdf, loss_m_np, training_step)

            

        # accumulate loss and loss grads over the different designs before backward
        loss.backward() 

        # optimizer takes step based on accumulated grads
        optimizer.step()
        loss_np = loss.detach().cpu().numpy()
        
        # writer.add_scalar('Train' + '/Loss_total', loss_np, training_step)

        if np.mod(training_step,200)==0 or (training_step==n_training_steps):

            logging.info(
                (str(training_step) + ': ' 
                  + str(np.round(losses_np,1)) + ' Net: ' 
                  + np.array2string(loss_np,precision=1)).replace('\n', '')
                  )
                  # + str(np.round(loss_np,1)))

        if np.mod(training_step,500)==0 or (training_step==n_training_steps) and training_step>0:


            save_dict = dict()
            save_dict['comment'] = ''
            save_dict['state_dict'] =  policy_network.state_dict()
            save_dict['n_hidden_layers'] = policy_network.n_hidden_layers
            save_dict['hidden_layer_size'] = policy_network.hidden_layer_size
            save_dict['urdf_names'] = urdf_names # urdfs used in training
            save_dict['training_step']= training_step
            save_dict['batch_size'] = batch_size
            save_dict['torque_loss_weight'] = torque_loss_weight
            save_dict['n_designs_per_step_record'] = n_designs_per_step_record
            save_dict['lr_record'] = lr_record

            torch.save(save_dict,  control_save_path,
                _use_new_zipfile_serialization=False)

    logging.info('done')
    policy_network = policy_network.to(torch.device('cpu'))
