'''
Train a gnn for control based on the data collected during mpc

This version uses the state and last state in observation

'''
import torch
import numpy as np
import pgnn_control2 as pgnnc
from utils import get_sampleable_inds, sample_memory, wrap_to_pi
import os
# cwd = os.path.dirname(os.path.realpath(__file__))
from robot_env import robot_env
from utils import rotate, create_control_inputs2
import logging
# from torch.utils.tensorboard import SummaryWriter

from datetime import datetime

def print_time():
    time_now = datetime.now()
    time_now_str = datetime.strftime(time_now, '%Y%m%d_%H%M%S')    
    print('Time now: ' + time_now_str)
    
def train_control(control_save_path,
        batch_sizes, n_training_steps, device,
        optimizer, urdf_names, sampleable_inds,
        states_memory_tensors, actions_memory_tensors, 
        torques_memory_tensors, goal_memory_tensors,
        measurement_stds, module_sa_len, 
        gnn_nodes_in, modules_types, attachments,
        torque_loss_weight = 1,
        n_designs_per_step = 6):
    

    # # ensure that all the networks are on the desired device.
    gnn_nodes = gnn_nodes_in
    for node in gnn_nodes:
        node = node.to(device)


    # set up logging
    # folder = os.path.dirname(control_save_path)
    # log_name = os.path.splitext(control_save_path)[0]
    # log_path = os.path.join(folder, log_name+'_log.log')
    # logging.basicConfig(level=logging.INFO,
    #                         format='%(message)s', 
    #                         filename=log_path,
    #                         filemode='w')
    # console = logging.StreamHandler()
    # console.setLevel(logging.INFO)
    # formatter = logging.Formatter('%(message)s')
    # console.setFormatter(formatter)
    # logging.getLogger('').addHandler(console)
    # logging.info('Starting ' + os.path.realpath(__file__))
    print = logging.info # set print to be logging so I don't need to replace all instances in the file

    print('Control training using both last action and current action')

    modules = dict()
    for urdf in urdf_names:
        modules[urdf] = []
        n_modules = len(modules_types[urdf])
        for mi in range(n_modules):
            modules[urdf].append(pgnnc.Module(mi,
                    gnn_nodes[modules_types[urdf][mi]], device))


    # how many designs to sample each step?
    if len(urdf_names)<n_designs_per_step:
        n_designs_per_step = len(urdf_names)
    print('n_designs_per_step: ' + str(n_designs_per_step))

    # keep track of when the n_designs_per_step changes at a training step
    n_designs_per_step_record = []
    n_designs_per_step_record.append([0,n_designs_per_step])

    # keep track of when the learning rate changes at a training step
    lr_record = []
    for param_group in optimizer.param_groups:
        print( 'LR: ' + str(param_group['lr']) )
        lr_record.append([0, param_group['lr']])

    # print_time()
    for training_step in range(n_training_steps+1):
        
    #     if training_step == 10000 or training_step == 20000 or training_step == 30000:
        # if np.mod(training_step, 10000)==0 and training_step>10000:
        if np.mod(training_step, 2500)==0 and training_step>=10000:
            for param_group in optimizer.param_groups:
                if param_group['lr']>2e-5:
                    # half the learning rate periodically
                    param_group['lr'] = param_group['lr']/2.
                    print( 'LR: ' + str(param_group['lr']) )
                    lr_record.append([training_step, param_group['lr']])


        # increase number of design sampled as steps increase
        if ( np.mod(training_step,2500 )==0 and 
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
            states1 = [smm[sampled_inds+1].to(device) for smm in states_memory_tensors[urdf]]
            actions = [amm[sampled_inds+1].to(device) for amm in actions_memory_tensors[urdf]]
            # torques are actually the "next state" torques used for feedforward
            torques = [amm[sampled_inds+2].to(device) for amm in torques_memory_tensors[urdf]]
            goals_world = goal_memory_tensors[urdf][sampled_inds+1,:].to(device)
            
            # add on white noise
            for si in range(len(states0)):
                noise = torch.distributions.Normal(0.0, measurement_stds[urdf][si])
                states0[si] += noise.sample((batch_size,)).to(device)

                # FOR TEST ONLY: pass in zeros for states0
                # states0[si] =  torch.zeros_like(states0[si])
            # # add on white noise
            # for si in range(len(states1)):
                # noise = torch.distributions.Normal(0.0, measurement_stds[urdf][si])
                states1[si] += noise.sample((batch_size,)).to(device)



            # # goals_world[x,y] are recorded in world frame. shift to body frame here.
            node_inputs = create_control_inputs2(states0, states1, goals_world)
            # node_inputs = create_control_inputs2(states1, states1, goals_world)

            

            for module in modules[urdf]: # this prevents the LSTM in the GNN nodes from 
                # learning relations over time, only over internal prop steps.
                module.reset_hidden_states(batch_size) 
            

            u_out_mean, u_out_var = pgnnc.run_propagations(
                modules[urdf], attachments[urdf], 2, node_inputs, device)
            loss_m = 0
            sum_module_action_len = sum(module_action_len)
            for mm in range(n_modules):
                # backprop appears to treat empty tensors poorly, so make sure its not empty
                if module_action_len[mm]>0:
                    # loss for velocity command
                    loss_v = torch.sum(
                        (u_out_mean[mm][:,:module_action_len[mm]]
                         - actions[mm])**2/u_out_var[mm][:,:module_action_len[mm]] + 
                        torch.log(u_out_var[mm][:,:module_action_len[mm]]) 
                                    )/batch_size
                    loss_m += loss_v#/sum_module_action_len # divide out the number of actions

                    # loss for torque value
                    losses_tau = ( (u_out_mean[mm][:,module_action_len[mm]:]
                             - torques[mm])**2/u_out_var[mm][:,module_action_len[mm]:] + 
                            torch.log(u_out_var[mm][:,module_action_len[mm]:]) )
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
            # print_time()

            print(
                (str(training_step) + ': ' 
                  + str(np.round(losses_np,1)) + ' Net: ' 
                  + np.array2string(loss_np,precision=1)).replace('\n', '')
                  )
                  # + str(np.round(loss_np,1)))

        if np.mod(training_step,500)==0 or (training_step==n_training_steps) and training_step>0:

            gnn_state_dict = pgnnc.get_state_dicts(gnn_nodes)

            save_dict = dict()
            save_dict['comment'] = ''
            save_dict['gnn_state_dict'] =  gnn_state_dict
            save_dict['internal_state_len'] = gnn_nodes[0].internal_state_len
            save_dict['message_len'] = gnn_nodes[0].message_len
            save_dict['hidden_layer_size'] = gnn_nodes[0].hidden_layer_size
            save_dict['urdf_names'] = urdf_names # urdfs used in training
            save_dict['training_step']= training_step
            save_dict['batch_size'] = batch_size
            save_dict['torque_loss_weight'] = torque_loss_weight
            save_dict['n_designs_per_step_record'] = n_designs_per_step_record
            save_dict['lr_record'] = lr_record

            torch.save(save_dict,  control_save_path)
    print('done')
    for node in gnn_nodes:
        node = node.to(torch.device('cpu'))


if __name__ == '__main__':
    from robot_env import robot_env




    print('Preparing to train control')
    # urdf_names = ['llllll','wnwwnw']
    urdf_names = ['llllll', 'lnwwnl', 'llwwll', 'lnllnl',
              'lwllwl', 'lwwwwl', 'wlwwlw', 'wwllww' ,
              'wwwwww', 'wnllnw', 'wllllw', 'wnwwnw']
    # urdf_names = ['wnwwnw']

    device = torch.device("cuda:0")

    folder = 'mbrl_v5_test14'
    cwd = os.path.dirname(os.path.realpath(__file__))
    folder = os.path.join(cwd, folder)   

    # Loading them all from file instead of passing them around as arguments
    # costs a few seconds of time, but sidesteps some multiprocessing difficulties
    # that arise from trying to pass tensors and save to file within children.
    mpc_rollouts_now = dict()
    measurement_stds = dict()
    for urdf in urdf_names:
        mpc_save_path  = urdf  + '_mpc_rollouts_iter3.ptx'
        mpc_save_path = os.path.join(folder, mpc_save_path)
        print('Loading rollouts from ' + mpc_save_path)
        mpc_rollouts_now[urdf] = torch.load( mpc_save_path, map_location=lambda storage, loc: storage)

        # make env, which will be only temporarily used to extract some data
        env = robot_env(show_GUI = False)
        env.reset_terrain()
        env.reset_robot(urdf_name=urdf, randomize_start=False)
        measurement_stds[urdf] = env.measurement_stds
        del env

    batch_size_control = 500 # default batch size for control (reduced to save gpu ram)
    batch_sizes_control = dict()

    sampleable_inds = dict()
    states_memory_tensors = dict()
    actions_memory_tensors = dict() 
    torques_memory_tensors = dict()
    goal_memory_tensors = dict()
    module_sa_len = dict()
    modules_types = dict()
    attachments = dict()
    print('reshaping memory')
    for urdf in urdf_names:     
        # print(urdf)
        # only use most recent rollouts to train policy.
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

        states_memory_tensors[urdf]= [torch.cat(s,0).share_memory_() for s in list(zip(*states_memory)) ]
        actions_memory_tensors[urdf]= [torch.cat(s,0).share_memory_() for s in list(zip(*actions_memory)) ]
        torques_memory_tensors[urdf]= [torch.cat(s,0).share_memory_() for s in list(zip(*torques_memory)) ]
        goal_memory_tensors[urdf] = torch.cat(goal_memory,-1).permute(1,0).share_memory_()
        
    n_training_steps_control = 50000


    print('creating network')
    control_save_path = 'multidesign_control_iter1_test2.pt'
    control_save_path = os.path.join(folder, control_save_path)

  

    internal_state_len = 100
    message_len = 50
    hidden_layer_size = 350
    preload_control = False

    # print('Loading weights from ' + control_save_path)
    # save_dict_control = torch.load( control_save_path, map_location=lambda storage, loc: storage)
    # preload_control = True
    # internal_state_len = save_dict_control['internal_state_len']
    # message_len= save_dict_control['message_len']
    # hidden_layer_size= save_dict_control['hidden_layer_size']


    # set up logging
    folder = os.path.dirname(control_save_path)
    log_name = os.path.splitext(control_save_path)[0]
    log_path = os.path.join(folder, log_name+'5_log.log')
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



    # print('***Note: for test, only use state1 and ignore state0')
    # print('***Note: for test, zeros as states0')
    # print('***Note: doubled noise applied to states0')



    print('Policy internal_state_len,message_len,hidden_layer_size = ' + 
        str([internal_state_len,message_len,hidden_layer_size]))

    # Graph neural network creation
    goal_len = 3
    gnn_nodes_control = pgnnc.create_GNN_nodes(internal_state_len, 
                    message_len, hidden_layer_size, 
                    torch.device('cpu'), goal_len=goal_len, body_input= True) 

    # pgnnc.load_state_dicts(gnn_nodes_control, save_dict_control['gnn_state_dict'])


    weight_decay_control = 1e-4
    optim_lr_control = 3e-3 
    # optim_lr = 1e-3 
    optimizer_control = torch.optim.Adam(
                            pgnnc.get_GNN_params_list(gnn_nodes_control),
                            lr=optim_lr_control,
                            weight_decay= weight_decay_control)  


    print('Training control ' + control_save_path)     
    n_designs_per_step = len(urdf_names)

    # reset optimizer learning rate to original, since may decay during training
    for param_group in optimizer_control.param_groups:
        param_group['lr'] = optim_lr_control

    train_control(control_save_path,
            batch_sizes_control, 
            n_training_steps_control, device,
            optimizer_control, urdf_names, sampleable_inds,
            states_memory_tensors, actions_memory_tensors, 
            torques_memory_tensors, goal_memory_tensors,
            measurement_stds, module_sa_len, 
            gnn_nodes_control, modules_types, attachments,
            torque_loss_weight = 0.25,
            n_designs_per_step = n_designs_per_step)

    from planning_utils import compare_velocities
    from apply_policy2 import make_goal_memory, apply_policy


    print('Training control done.')
    print('Control vel metrics:')
    ### simulate policy to validate and gather policy rollout data
    # make some direction goals
    goal_memory = make_goal_memory(41, device=torch.device('cpu')) # 10*4 + 1
    T = 20

    for urdf in urdf_names:
        apply_policy_save_path = os.path.join(folder, urdf + '_apply_policy_iter1.ptx')
        # if not os.path.exists(apply_policy_save_path):
            # print('Loading weights from ' + control_save_path)
        apply_policy(urdf, goal_memory,
          gnn_nodes_control, torch.device('cpu'), 
          apply_policy_save_path, show_GUI=False)
          # zero_out_last_states = True)

        apply_policy_save_dict = torch.load(apply_policy_save_path,
                 map_location=lambda storage, loc: storage)
        vm, vmb = compare_velocities(
                apply_policy_save_dict['states_memory'],
                apply_policy_save_dict['goal_memory'], 
                apply_policy_save_dict['run_lens'],
                10, T )
        vm_rescaled = (vmb-vm)/vmb
        print(urdf + ': ' + str(vm) + ' baseline ' + str(vmb) + 
            ', rescaled: ' + str(vm_rescaled))

# if __name__ == '__main__':
#     from robot_env import robot_env

#     print('Preparing to train control')
#     # urdf_names = ['llllll','wnwwnw']
#     urdf_names = ['llllll', 'lnwwnl', 'llwwll', 'lnllnl',
#               'lwllwl', 'lwwwwl', 'wlwwlw', 'wwllww' ,
#               'wwwwww', 'wnllnw', 'wllllw', 'wnwwnw']
#     device = torch.device("cuda:0")

#     folder = 'mbrl_v5_test3B'
#     cwd = os.path.dirname(os.path.realpath(__file__))
#     folder = os.path.join(cwd, folder)   


#     torch.multiprocessing.set_start_method('spawn') # needed for CUDA drivers in parallel
#     torch.multiprocessing.set_sharing_strategy('file_system') # might be needed for opening and closing many files

#     # Loading them all from file instead of passing them around as arguments
#     # costs a few seconds of time, but sidesteps some multiprocessing difficulties
#     # that arise from trying to pass tensors and save to file within children.
#     mpc_rollouts_now = dict()
#     measurement_stds = dict()
#     for urdf in urdf_names:
#         mpc_save_path  = urdf  + '_mpc_rollouts_iter3.ptx'
#         mpc_save_path = os.path.join(folder, mpc_save_path)
#         print('Loading rollouts from ' + mpc_save_path)
#         mpc_rollouts_now[urdf] = torch.load( mpc_save_path, map_location=lambda storage, loc: storage)

#         # make env, which will be only temporarily used to extract some data
#         env = robot_env(show_GUI = False)
#         env.reset_terrain()
#         env.reset_robot(urdf_name=urdf, randomize_start=False)
#         measurement_stds[urdf] = env.measurement_stds
#         del env

#     batch_size_control = 500 # default batch size for control (reduced to save gpu ram)
#     batch_sizes_control = dict()

#     sampleable_inds = dict()
#     states_memory_tensors = dict()
#     actions_memory_tensors = dict() 
#     torques_memory_tensors = dict()
#     goal_memory_tensors = dict()
#     module_sa_len = dict()
#     modules_types = dict()
#     attachments = dict()
#     print('reshaping memory')
#     for urdf in urdf_names:     
#         # print(urdf)
#         # only use most recent rollouts to train policy.
#         # older rollouts would come from models not trained on the newly guided data.

#         states_memory = mpc_rollouts_now[urdf]['states_memory'] 
#         actions_memory = mpc_rollouts_now[urdf]['actions_memory'] 
#         torques_memory = mpc_rollouts_now[urdf]['torques_memory'] 
#         goal_memory = mpc_rollouts_now[urdf]['goal_memory'] 
#         run_lens = mpc_rollouts_now[urdf]['run_lens'] 
#         step_memory = mpc_rollouts_now[urdf]['step_memory'] 

#         attachments[urdf] = mpc_rollouts_now[urdf]['attachments'] 
#         modules_types[urdf] = mpc_rollouts_now[urdf]['modules_types'] 
#         module_sa_len[urdf] = mpc_rollouts_now[urdf]['module_sa_len'] 
#         sampleable_inds[urdf] = get_sampleable_inds(run_lens, 3)
#                     # ^ will use state0, state1, action1, torque2
#             # (last state),(current state),(action),(torque after action)
#         n_sampleable = len(sampleable_inds[urdf])
#         batch_sizes_control[urdf] = batch_size_control
#         if batch_sizes_control[urdf] > n_sampleable:
#             batch_sizes_control[urdf] = n_sampleable

#         states_memory_tensors[urdf]= [torch.cat(s,0).share_memory_() for s in list(zip(*states_memory)) ]
#         actions_memory_tensors[urdf]= [torch.cat(s,0).share_memory_() for s in list(zip(*actions_memory)) ]
#         torques_memory_tensors[urdf]= [torch.cat(s,0).share_memory_() for s in list(zip(*torques_memory)) ]
#         goal_memory_tensors[urdf] = torch.cat(goal_memory,-1).permute(1,0).share_memory_()
        
#         n_training_steps_control = 30000


#     print('creating network')
#     control_save_path = 'multidesign_control_serial.pt'
#     control_save_path = os.path.join(folder, control_save_path)

  
#     internal_state_len = 50
#     message_len = 50
#     hidden_layer_size = 100#250
#     preload_control = False

#     # Graph neural network creation
#     goal_len = 3
#     gnn_nodes_control = pgnnc.create_GNN_nodes(internal_state_len, 
#                     message_len, hidden_layer_size, 
#                     torch.device('cpu'), goal_len=goal_len, body_input= True) 

#     weight_decay_control = 1e-4
#     optim_lr_control = 3e-3 
#     # optim_lr = 1e-3 
#     optimizer_control = torch.optim.Adam(
#                             pgnnc.get_GNN_params_list(gnn_nodes_control),
#                             lr=optim_lr_control,
#                             weight_decay= weight_decay_control)  


#     print('Training control ' + control_save_path)     
#     n_designs_per_step = len(urdf_names)

#     # reset optimizer learning rate to original, since may decay during training
#     for param_group in optimizer_control.param_groups:
#         param_group['lr'] = optim_lr_control

#     train_control(control_save_path,
#             batch_sizes_control, 
#             n_training_steps_control, device,
#             optimizer_control, urdf_names, sampleable_inds,
#             states_memory_tensors, actions_memory_tensors, 
#             torques_memory_tensors, goal_memory_tensors,
#             measurement_stds, module_sa_len, 
#             gnn_nodes_control, modules_types, attachments,
#             torque_loss_weight = 0.25,
#             n_designs_per_step = n_designs_per_step)


