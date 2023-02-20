'''
Apply the learned control policy to a series of goal velocities

This version uses only last state in observation


'''
import torch
import numpy as np
import pgnn_control as pgnnc
from robot_env import robot_env
from utils import to_tensors, combine_state, wrap_to_pi, rotate, create_control_inputs
import os
cwd = os.path.dirname(os.path.realpath(__file__))


from planning_utils import cost_weights 
speed_scale_xy = cost_weights['speed_scale_xy']
speed_scale_yaw = cost_weights['speed_scale_yaw']


# import logging
# print = logging.info

def apply_policy(urdf, goal_memory, 
    gnn_nodes_control, device, 
    save_path, show_GUI=False, video_name=None,
    sim_speed_factor = 1,
    zero_out_last_states = False,
    overhead=True):


    # print('Applying policy to ' + urdf)
    # create environment
    env = robot_env(show_GUI = show_GUI)
    env.reset_terrain()
    env.reset_robot(urdf_name=urdf, randomize_start=False)
    if overhead:
        env.p.resetDebugVisualizerCamera(1.5,0,-89.999,
            [0,0,0.2],physicsClientId=env.physicsClient) 
    else:
        env.p.resetDebugVisualizerCamera(1.1,0,-45,
            [0,-0.7,0.2],physicsClientId=env.physicsClient) 



    if show_GUI:
        env.sim_speed_factor = sim_speed_factor # run it faster for visualization
        env.p.configureDebugVisualizer( env.p.COV_ENABLE_MOUSE_PICKING, 0,physicsClientId=env.physicsClient)
        env.p.configureDebugVisualizer( env.p.COV_ENABLE_KEYBOARD_SHORTCUTS, 0,physicsClientId=env.physicsClient)

    attachments = env.attachments
    modules_types = env.modules_types
    n_modules = len(modules_types)
    env_state_init = env.get_state()
    module_state_len = []
    for s in env_state_init:
        module_state_len.append(len(s))
    state_len= np.sum(module_state_len)
    action_len = env.num_joints
    module_action_len = list(np.diff(env.action_indexes))
    module_sa_len = module_state_len+ module_action_len

    modules_control = []
    n_modules = len(modules_types)
    for mi in range(n_modules):
        modules_control.append(pgnnc.Module(mi,
            gnn_nodes_control[modules_types[mi]], device))

    states_memory = []
    actions_memory = []
    run_lens = []

    vxy_scale = 1
    vyaw_scale = 1

    # for each goal direction simulate a run
    for i in range(len(goal_memory)):
        env.reset_robot(urdf_name=urdf, randomize_start=False)
        states_list = []
        actions_list = []
        goals_run = goal_memory[i]
        run_len = goals_run.shape[-1]

        if video_name is not None:
            if show_GUI:
                env.start_video_log(video_name+str(i) +'.mp4')
            else:
                print('Cannot log video without showing GUI')

        env_state = env.get_state()
        last_states = [smm.to(device) for smm in to_tensors(env_state)]

        for t in range(run_len-1):

            # draw arrows for body direction before stepping
            desired_xyyaw = goals_run[:,t].cpu().numpy()
            vect1 = np.array([desired_xyyaw[0],
                    desired_xyyaw[1],
                    0] )
            chassis_yaw = env.pos_rpy[-1]
            vect2 = np.array([np.cos(desired_xyyaw[2]/2+chassis_yaw),
                      np.sin(desired_xyyaw[2]/2+chassis_yaw), 
                      0])*np.abs(desired_xyyaw[2])
            env.draw_body_arrows([vect1/vxy_scale, 
                                  0.5*vect2/vyaw_scale],
                                 [[0,0,0], [0,0,1]])


            goals_world = goals_run[:,t].unsqueeze(0)

            env_state = env.get_state()
            states = [smm.to(device) for smm in to_tensors(env_state)]
            states_list.append(env_state )

            ### change to body frame the goal heading and state
            node_inputs = create_control_inputs(last_states, goals_world)

            # node_inputs = create_control_inputs2(states, states, goals_world) # for debug
            # print('last state')
            # print(last_states)
            # print('state')
            # print(states)
            # print('node_inputs')
            # print(node_inputs)



            for module in modules_control: # this prevents the LSTM in the GNN nodes from 
                # learning relations over time, only over internal prop steps.
                module.reset_hidden_states(1) 

            with torch.no_grad():
                out_mean, out_var = pgnnc.run_propagations(
                    modules_control, attachments, 2, node_inputs, device)
                u_out_mean = []
                tau_out_mean = []
                for mm in range(n_modules):
                    u_out_mean.append(out_mean[mm][:,:module_action_len[mm]])
                    tau_out_mean.append(out_mean[mm][:,module_action_len[mm]:])
                u_combined = torch.cat(u_out_mean,-1).squeeze()
                action = u_combined.cpu().numpy()
                u_div = env.divide_action_to_modules(action)

            actions_list.append(u_div)   


            # execute control action
            env.step(action)

            last_states = [s.clone() for s in states]

            # stop if it flips over
            if np.dot([0,0,1], env.z_axis)<0:
                break

    #         if (t<run_len-1) and env.show_GUI:
    #             xyz_before = states_run[0][t,0:3].numpy()
    #             xyz_after = states_run[0][t+1,0:3].numpy()

    #             env.draw_line( [xyz_before[0],xyz_before[1],0.01],
    #                    [xyz_after[0], xyz_after[1],0.01],
    #                      color=[0,0,0])


            # if video_name is not None:
            #     env.take_snapshot(video_name + str(i))


        # done with this run
        env_state = env.get_state()
        states_list.append(env_state )
        
        # add NaN as last action
        action_now = []
        last_action = actions_list[-1]
        for ai in range(len(last_action)):
            na = len(last_action[ai])
            action_now.append(np.ones(na)*np.nan) 
        actions_list.append(action_now)  

        state_list_tensors = [torch.tensor( np.stack(s),dtype=torch.float32)
                                 for s in list(zip(*states_list)) ]
        action_list_tensors = [torch.tensor( np.stack(a),dtype=torch.float32)
                                 for a in list(zip(*actions_list)) ]

        states_memory.append(state_list_tensors)
        actions_memory.append(action_list_tensors)
        run_lens.append(len(states_list))

        if video_name is not None and show_GUI:
            env.stop_video_log()


    # Save data after all done
    save_dict = dict()
    save_dict['states_memory'] = states_memory
    save_dict['actions_memory'] = actions_memory
    save_dict['goal_memory'] = goal_memory
    save_dict['run_lens'] = run_lens

    torch.save(save_dict, save_path)
    # print('  saved ' + urdf + ' applied policy ')

def make_goal_memory(n_steps, device=torch.device('cpu')):

    T = 20
    dt = 20./240.
    # speed_scale_yaw = (T*dt)*np.pi/2
    # speed_scale_xy = (T*dt)
    speed_scales = np.array([speed_scale_xy, speed_scale_xy, speed_scale_yaw ])
    # Uses 75% top speed as the goal speed
    speed_scales = speed_scales*0.75
    # print(speed_scales )

    goal_memory = []
    directions = [[1,0,0], [0,1,0],
                  [-1,0,0], [0,-1,0],
                  [0,0,1], [0,0,-1]]
    for i in range(len(directions)):
        direction = np.array(directions[i])*speed_scales
        des_xyyaw = torch.tensor(direction, dtype=torch.float32, device=device)
        goal_tensors = []
        for j in range(n_steps):
            goal_tensors.append(des_xyyaw)
        goal_memory.append(torch.stack(goal_tensors,-1))

    # for ii in range(3):
    #     des_xyyaw = torch.zeros(3, dtype=torch.float32)
    #     des_xyyaw[ii] = speed_scales[ii]


    # for ii in range(3):
    #     des_xyyaw = torch.zeros(3, dtype=torch.float32)
    #     des_xyyaw[ii] = -speed_scales[ii]
    #     goal_tensors = []
    #     for j in range(n_steps):
    #         goal_tensors.append(des_xyyaw)
    #     goal_memory.append(torch.stack(goal_tensors,-1))

    return goal_memory


if __name__ == '__main__':
    from planning_utils import compare_velocities
    folder = 'mbrl_v5_test11_car'
    device = torch.device('cpu')

    urdf_names = ['wnwwnw']
    print('Control vel metrics:')
    ### simulate policy to validate and gather policy rollout data
    # make some direction goals
    goal_memory = make_goal_memory(41, device=torch.device('cpu')) # 10*4 + 1
    T = 20

    for urdf in urdf_names:

        # load up a learned policy to test
        # # load previous weights if desired
        control_save_path = 'multidesign_control_iter1_test.pt'

        fname = os.path.join(folder, control_save_path)
        if os.path.exists(fname):
            print('Loading weights from ' + fname)
            save_dict_control = torch.load( fname)#, map_location=lambda storage, loc: storage)

            gnn_nodes_control = pgnnc.create_GNN_nodes(save_dict_control['internal_state_len'], 
                        save_dict_control['message_len'], save_dict_control['hidden_layer_size'], 
                        device, goal_len=3, body_input= True) 

            pgnnc.load_state_dicts(gnn_nodes_control, save_dict_control['gnn_state_dict'])

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




    # # make some direction goals
    # goal_memory = make_goal_memory(41) # 10*4 + 1
        
    # # load up a learned policy to test
    # # # load previous weights if desired
    # fname = 'mbrl_v3_test7/multidesign_control_iter5.pt'
    # if os.path.exists(fname):
    #     print('Loading weights from ' + fname)
    #     save_dict_control = torch.load( fname)#, map_location=lambda storage, loc: storage)

    #     gnn_nodes_control = pgnnc.create_GNN_nodes(save_dict_control['internal_state_len'], 
    #                 save_dict_control['message_len'], save_dict_control['hidden_layer_size'], 
    #                 device, goal_len=3, body_input= True) 

    #     pgnnc.load_state_dicts(gnn_nodes_control, save_dict_control['gnn_state_dict'])

    # T = 20
    # # urdf = 'wnwwnw'
    # urdf_names = [ 'lnllnl', 'llllll', 'wnwwnw']
    # # urdf_names = ['llwlll', 'lnwlnl', 'wlllnw' , 'lllwnw']
    # for urdf in urdf_names:

    #     # save_path = os.path.join(cwd, 'mbrl_v3_test8/' + 
    #         # urdf + '_mpc_rollouts_iter1.ptx')

    #     save_path = os.path.join(cwd, urdf + '_apply_policy.ptx')
    #     apply_policy(urdf, goal_memory,
    #       gnn_nodes_control, device, save_path, show_GUI=True)

    #     save_dict = torch.load(save_path, map_location=lambda storage, loc: storage)
    #     vel_metric, vel_metric_baseline = compare_velocities(
    #             save_dict['states_memory'],
    #             save_dict['goal_memory'], 
    #             save_dict['run_lens'],
    #             10, T )
    #     print(urdf + ': ' + str(vel_metric) + ' baseline ' + str(vel_metric_baseline))
