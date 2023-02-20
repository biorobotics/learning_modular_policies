'''
Source code for paper "Learning modular robot control policies" in Transactions on Robotics
MLP comparisons
Julian Whitman, Dec. 2022. 

'''
import numpy as np
import torch


def wrap_to_pi(angle):
    return torch.remainder(angle + np.pi,  np.pi*2) - np.pi


# helper functions to move to and from normalized data
def to_normalized(tensor_in, tensor_mean, tensor_std ):
    if len(tensor_in)>0: # no data means no need to normalize
        return (tensor_in - tensor_mean)/tensor_std    
    else:
        return tensor_in

def from_normalized(tensor_in, tensor_mean, tensor_std ):
    if len(tensor_in)>0:
        return (tensor_in*tensor_std) + tensor_mean 
    else:
        return tensor_in

def rotate(rotmats, vecs):
    return  torch.matmul(rotmats, vecs.unsqueeze(-1)).squeeze(2)

def state_to_fd_input(state0):
  # strips off the x,y,yaw and transforms data to a yaw-less frame
  
    chassis_state = state0[0]

    sin0 = torch.sin(chassis_state[:,5])
    cos0 = torch.cos(chassis_state[:,5])
    z0 = torch.zeros_like(cos0)

    R0_t = torch.stack( [ torch.stack([cos0, sin0, z0]),
                      torch.stack([-sin0,  cos0, z0]),
                      torch.stack([z0, z0,   torch.ones_like(cos0)])
                      ]).permute(2,0,1)

    # form fd input zrp+ [vx,vy,vz,wx,wy,wz]_Body + [q, qdot] 

    chassis_state_body = torch.cat([chassis_state[:,2:5],
                            rotate(R0_t,chassis_state[:,6:9]),
                            rotate(R0_t,chassis_state[:,9:12])],-1)
    fd_input = [chassis_state_body] + state0[1:]


    return fd_input, R0_t


def to_body_frame_batch(state0, state1):
    fd_input, R0_t = state_to_fd_input(state0)
    chassis_state0 = state0[0]
    chassis_state1 = state1[0]

    # form fd output: [delta (xyz), delta (rpy), delta(v), delta(omega)]_body + delta[q, qdot] 
    delta_xyz = chassis_state1[:,0:3] - chassis_state0[:,0:3]
    delta_rpy = chassis_state1[:,3:6] - chassis_state0[:,3:6]

    # delta_rpy = torch.asin(torch.sin(delta_rpy)) # moves to be on [-pi,pi]
    delta_rpy = wrap_to_pi(delta_rpy)
    delta_v =     chassis_state1[:,6:9] - chassis_state0[:,6:9]
    delta_omega = chassis_state1[:,9:12] - chassis_state0[:,9:12]
    delta_fd_output = [
            torch.cat([rotate(R0_t,delta_xyz),
              delta_rpy, # does not need to be transformed since rp are already in body frame
              rotate(R0_t,delta_v),
              rotate(R0_t,delta_omega)] ,-1)
              ]

    # diff between non-chassis modules
    for i in range(1,len(state0)):
        delta_fd_output.append( state1[i] - state0[i] )

    return fd_input, delta_fd_output


def from_body_frame_batch(state0, delta_fd):
    chassis_state0 = state0[0]
    chassis_delta_fd  = delta_fd[0]
    sin0 = torch.sin(chassis_state0[:,5])
    cos0 = torch.cos(chassis_state0[:,5])
    z0 = torch.zeros_like(cos0)

    R0 = torch.stack( [ torch.stack([cos0, -sin0, z0]),
                  torch.stack([sin0,  cos0, z0]),
                  torch.stack([z0, z0,   torch.ones_like(cos0)])
                  ]).permute(2,0,1)
    state1_est = [ 
        torch.cat([
        chassis_state0[:,0:3] + rotate(R0,chassis_delta_fd[:,0:3]),
        chassis_state0[:,3:6] + chassis_delta_fd[:,3:6], # see note below
        chassis_state0[:,6:9] + rotate(R0, chassis_delta_fd[:,6:9]),
        chassis_state0[:,9:12] + rotate(R0, chassis_delta_fd[:,9:12])],-1)
        ]

        # NOTE previously used 
                # wrap_to_pi(chassis_state0[:,3:6] + chassis_delta_fd[:,3:6]),
#  but I notice that when predicting state rollouts it is better if we keep those rollouts 
# without wrap around yaw. This is so the desired delta and achieved delta can be compared in the cost.


        # Used to use:
        # torch.asin(torch.sin( chassis_state0[:,3:6] + chassis_delta_fd[:,3:6] )),
        # but, this caused NaNs in recursive gradients. changed to remainder instead 
        # which is functionally the same but has simpler grads.

    # diff between non-chassis modules
    for i in range(1,len(state0)):
        state1_est.append( state0[i] + delta_fd[i] )

    return state1_est

# utilities to clean up the normal state add and subtract
def state_diff_batch(state0, state1):
    delta_fd_output = [s1-s0 for (s1,s0) in zip(state1, state0)]
    return state0, delta_fd_output

def state_add_batch(state0, delta_fd):
    state1 = [s0+ds for (s0,ds) in zip(state0, delta_fd)]
    return state1


# wrap these all up into functions

def get_sampleable_inds(run_lens, seq_len=2):

    # seq_len = 3 # 2 = standard (state, action, next_state)
    # sampleable_inds is the list of inds we can pick to have valid sequences of seq_len
    sampleable_inds = [] 
    start_index = 0
    for run_len in run_lens:
        sampleable_inds +=  list(range(
                start_index, 
                start_index+run_len-seq_len+1
                ) ) # the range function, if given a negative interval,
        # will return an empty list, so this filters out automatically 
        # any runs that are too short to get a full sequence out of.
        start_index+=run_len

    sampleable_inds = torch.tensor(sampleable_inds, 
                        dtype= torch.long) # torch.long is an int64
    return sampleable_inds


def sample_memory(states_memory_tensors,actions_memory_tensors,
    sampleable_inds, seq_len, batch_size=5):
    # conditions: 
    # states_memory_tensors and actions_memory_tensors are lists of tensors,
    # where each entry in the list has the full set of states and action from the loaded memory.
    
    # sample from the tensor memories and create new views with seq_len
    # sampled_inds = sampleable_inds # sample all of them to start testing
    sampled_inds = sampleable_inds[np.random.choice(
        len(sampleable_inds), batch_size, replace=False)]

    sampled_ranges = sampled_inds.repeat((seq_len,1))
    for si in range(seq_len):
        sampled_ranges[si] += si

    state_seq = []
    action_seq =[]
    for si in range(seq_len):
        smi = [smm[sampled_ranges[si]] for smm in states_memory_tensors]
        ami = [amm[sampled_ranges[si]] for amm in actions_memory_tensors]
        state_seq.append(smi)
        action_seq.append(ami) # the last action in the seq will get thrown out
        
    return state_seq, action_seq, sampled_inds


def sample_memory_old_new(states_memory_tensors,actions_memory_tensors,
    sampleable_inds_old, sampleable_inds_new,
    seq_len, batch_size_old=5, batch_size_new=5 ):
    # conditions: 
    # states_memory_tensors and actions_memory_tensors are lists of tensors,
    # where each entry in the list has the full set of states and action from the loaded memory.
    
    # sample from the tensor memories and create new views with seq_len
    # sampled_inds = sampleable_inds # sample all of them to start testing
    sampled_inds_old = sampleable_inds_old[np.random.choice(
        len(sampleable_inds_old), batch_size_old, replace=False)]
    sampled_inds_new = sampleable_inds_new[np.random.choice(
        len(sampleable_inds_new), batch_size_new, replace=False)]
    sampled_inds = torch.cat([sampled_inds_old, sampled_inds_new])

    sampled_ranges = sampled_inds.repeat((seq_len,1))
    for si in range(seq_len):
        sampled_ranges[si] += si

    state_seq = []
    action_seq =[]
    for si in range(seq_len):
        smi = [smm[sampled_ranges[si]] for smm in states_memory_tensors]
        ami = [amm[sampled_ranges[si]] for amm in actions_memory_tensors]
        state_seq.append(smi)
        action_seq.append(ami) # the last action in the seq will get thrown out
        
    return state_seq, action_seq, sampled_inds

def to_device(list_in,device):
    list_out = []
    for item in list_in:
        list_out.append(item.to(device))
    return list_out
def to_tensors(x_in):
    x_out = []
    for x in x_in:
        x_out.append(torch.tensor(x, dtype=torch.float32).unsqueeze(0))
    return x_out

def detach_list(x_in):
    for x in x_in:
        x = x.detach()


def divide_state(x_in, module_state_len):
    x_out  = []
    ind = 0
    for i in range(len(module_state_len)):
        l = module_state_len[i]
        x_out.append(x_in[:,ind:ind+l])
        ind+=l
    return x_out

def divide_action(a_in, module_action_len):
    return divide_state(a_in, module_action_len)

def combine_state(x_in):
    return torch.cat(x_in,1)

def clip_grads_GNN(nodes, grad_clip):
    for node in nodes:
        # clamp the gradients
        for param in node.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

def clip_grads(nnetwork, grad_clip):
    # clamp the gradients
    for param in nnetwork.parameters():
        if param.grad is not None:
            param.grad.data.clamp_(-grad_clip, grad_clip)


def create_control_inputs(state_approx, goals_input, rotate_goals = True):

    ### change to body frame the goal heading and state
    # goals_world[x,y] are recorded in world frame. shift to body frame here.
    chassis_state = state_approx[0]
    chassis_yaw = chassis_state[:,5]
    sin0 = torch.sin(chassis_yaw)
    cos0 = torch.cos(chassis_yaw)
    z0 = torch.zeros_like(cos0)
    R0_t = torch.stack( [ torch.stack([cos0, sin0, z0]),
                      torch.stack([-sin0,  cos0, z0]),
                      torch.stack([z0, z0,   torch.ones_like(cos0)])
                      ]).permute(2,0,1)


    chassis_state_body = torch.cat([chassis_state[:,3:5],
                            rotate(R0_t,chassis_state[:,9:12])],-1)
    node_inputs_control = [chassis_state_body] + state_approx[1:]


    if rotate_goals:
        # goals are transformed to body frame using the last state in the sequence
        R0_t_xy = torch.stack( [ torch.stack([cos0, sin0]),
                                 torch.stack([-sin0,cos0])]).permute(2,0,1)
        goals_body0 = rotate(R0_t_xy, goals_input[:,0:2])
        goals_body1 = wrap_to_pi(goals_input[:,-1]) # probably don't need to actually wrap to pi, but for safety I do anyway
        # goals_input[-1] is a delta for turn angle 
        goals = torch.cat([goals_body0, 
                           goals_body1.unsqueeze(1)
                          ],-1)    
    else:
        goals = goals_input


    # node_inputs_control[0] = torch.cat([node_inputs_control[0], goals],1)

    return node_inputs_control, goals

