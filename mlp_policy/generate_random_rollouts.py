'''

Source code for paper "Learning modular robot control policies" in Transactions on Robotics
MLP comparisons
Julian Whitman, Dec. 2022. 

Randomly intitialize robots with differnet designs and send them random commands,
To collect data used for initial model learning.


'''

import numpy as np
import torch
import pybullet as p
import pybullet_data
import os
from robot_env import robot_env
from scipy import interpolate

# Generate random trajectories with splines
def spline_generator(num_joints, n_steps, n_sample_pts):
    x = np.linspace(0, n_steps, n_sample_pts) # lower res
    # y = np.random.uniform(-1,1,(n_sample_pts,num_joints))
    y = np.random.normal(0,0.75,(n_sample_pts,num_joints)) # this will skew to higher control
    xnew = np.arange(0,n_steps) # higher res
    ynew = np.zeros((n_steps,num_joints))
    for i in range(num_joints):
        tck = interpolate.splrep(x, y[:,i], s=0)
        ynew[:,i] = interpolate.splev(xnew, tck, der=0)
    return ynew




# assumes robot_rollouts is a dict with fields:
# states_memory, actions_memory, run_lens
# attachments, modules_types
#
def generate_random_rollouts(p_num,urdf_name, robot_rollouts, n_runs, 
    save_file_name=None, show_GUI=False):
    
    env = robot_env(show_GUI = show_GUI)
    # env = robot_env(show_GUI = True)

    randomize_start = True
    env.reset_terrain()
    env.reset_robot(urdf_name=urdf_name, 
        randomize_start = randomize_start)

    attachments = env.attachments
    modules_types = env.modules_types
    n_modules = len(modules_types)

    robot_rollouts['attachments'] = attachments
    robot_rollouts['modules_types'] = modules_types

    states_memory = []
    actions_memory = []
    run_lens = []

    n_steps = 100

    for i in range(n_runs):
        # if np.mod(i,100)==0:
        #     print('Sim run ' + str(i))
        state_list = []
        action_list = []
        randomize_start = True

        env.reset_robot(urdf_name=urdf_name, 
            randomize_start = randomize_start)
    
        u_list = spline_generator(env.num_joints, n_steps, int(n_steps/10))


        state_now = env.get_state()
        state_list.append(state_now)
        for step in range(n_steps):
            u = u_list[step]

            env.step(u)
            state_now = env.get_state()
            # end loop if the robot went sideways
            if np.dot([0,0,1], env.z_axis)<0:
                break
            else:
                state_list.append(state_now)
                u_div = env.divide_action_to_modules(u)
                action_list.append(u_div)
        
        # add on a NaN for last action so that the states and action lists 
        # are the same length
        action_now = []
        last_action = action_list[-1]
        for ai in range(len(last_action)):
            na = len(last_action[ai])
            action_now.append(np.ones(na)*np.nan) 
        action_list.append(action_now)        

        # sanity check that they are the same length now
        assert(len(state_list) == len(action_list))

        run_len = len(state_list)
        robot_rollouts['run_lens'].append(run_len)

        # REMOVE: convert to tensors
        # state_list_tensors = [torch.tensor( np.stack(s),dtype=torch.float32)
        #                          for s in list(zip(*state_list)) ]
        # action_list_tensors = [torch.tensor( np.stack(a),dtype=torch.float32)
        #                          for a in list(zip(*action_list)) ]

        # robot_rollouts['states_memory'].append(state_list_tensors)
        # robot_rollouts['actions_memory'].append(action_list_tensors)


        # Can't easily pass a list of tensors between processes with a manager,
        # so I will convert these to tensors in the main script
        robot_rollouts['state_lists'].append(state_list)
        robot_rollouts['action_lists'].append(action_list)

        if np.mod(i,100)==0:
            print(urdf_name + ' worker ' + str(p_num) + ' did run ' 
                + str(i) + ' / ' + str(n_runs))
            if save_file_name is not None and i>0:
                gen_to_file(robot_rollouts, save_file_name)
    print(urdf_name + ' worker ' + str(p_num) + ' done')

    if save_file_name is not None:
        gen_to_file(robot_rollouts, save_file_name)

def gen_to_file(random_rollouts_p, random_rollouts_fname):

    random_rollouts = dict()
    # collect and convert managed lists to normal list before saving
    random_rollouts['states_memory'] = []
    random_rollouts['actions_memory'] = []

    for state_list in random_rollouts_p['state_lists']:
        state_list_tensors = [torch.tensor( np.stack(s),dtype=torch.float32)
             for s in list(zip(*state_list)) ]
        random_rollouts['states_memory'].append(state_list_tensors)

    for action_list in random_rollouts_p['action_lists']:
        action_list_tensors = [torch.tensor( np.stack(a),dtype=torch.float32)
             for a in list(zip(*action_list)) ]
        random_rollouts['actions_memory'].append(action_list_tensors)

    random_rollouts['run_lens'] = list(random_rollouts_p['run_lens'])
    random_rollouts['attachments'] = list(random_rollouts_p['attachments'])
    random_rollouts['modules_types'] = list(random_rollouts_p['modules_types'])

    # save to file so that if we re-run later we can skip this step
    print('saving ' + random_rollouts_fname)
    torch.save(random_rollouts, random_rollouts_fname)


if __name__ == '__main__':

    VALIDATION = True # false to do many runs, true to do only a few

    torch.manual_seed(0)
    np.random.seed(0)

    cwd = os.path.dirname(os.path.realpath(__file__))

    folder = 'random_rollouts'
    # folder = 'random_rollouts_test'
    folder = os.path.join(cwd, folder)   

    if not(os.path.exists(folder)):
        os.mkdir(folder)
        print('Created folder ' + folder)
    else:
        print('Using folder ' + folder)

    urdf_names = ['llllll', 'llwwll', 'lwllwl', 'lwwwwl', 
                  'lnllnl', 'lnwwnl', 'wlwwlw', 'wwllww', 
                  'wwwwww', 'wnllnw', 'wllllw', 'wnwwnw']
    # urdf_names = ['llllll', 'wnwwnw']
    
    manager = torch.multiprocessing.Manager()
    # use a multiprocess pool to generate all random rollouts
    random_rollouts_p_list = dict()
    for urdf in urdf_names:
        random_rollouts_p = manager.dict()
        random_rollouts_p['state_lists'] = manager.list()
        random_rollouts_p['action_lists'] = manager.list()
        random_rollouts_p['run_lens'] = manager.list()
        random_rollouts_p['attachments'] = manager.list()
        random_rollouts_p['modules_types'] = manager.list()
        random_rollouts_p_list[urdf] = random_rollouts_p

    num_processes = 4
    pool_inputs = []
    show_GUI = False # set to False for large scale. Can turn to true to get videos of 
    # show_GUI = True # set to False for large scale. Can turn to true to get videos of 
    # the random action sequences.

    for i in range(len(urdf_names)):
        urdf = urdf_names[i]

        if VALIDATION:
            if show_GUI:
                n_runs = 10
            else:
                n_runs = 100
            save_file_name = os.path.join(folder, urdf + '_random_rollouts_validation.ptx')

        else:
            n_runs = 5000
            save_file_name = os.path.join(folder, urdf + '_random_rollouts.ptx')

        pool_inputs.append([i, urdf, 
            random_rollouts_p_list[urdf], 
            n_runs, save_file_name, show_GUI])
        # generate_random_rollouts(p_num,urdf_name, robot_rollouts, n_runs, save_file_name)

    print('starting pool')
    with torch.multiprocessing.Pool(processes=num_processes) as pool:
        pool.starmap(generate_random_rollouts, pool_inputs)
