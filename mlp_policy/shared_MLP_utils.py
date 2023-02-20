'''
Source code for paper "Learning modular robot control policies" in Transactions on Robotics
MLP comparisons
Julian Whitman, Dec. 2022. 

'''

def get_in_out_lens(urdf_names):
    # Some dimensionality information needed for the shared trunk
    max_fd_input_lens = [9, 6, 6, 6, 6, 6, 6]
    max_fd_output_lens = [12, 6, 6, 6, 6, 6, 6]
    max_action_lens = [0, 3, 3, 3, 3, 3, 3] # outputs vel
    fd_input_lens, fd_output_lens = [], []
    action_lens, policy_input_lens = [], []
    limb_types = []
    for urdf in urdf_names:
        fd_input_lens_i = [12-3] # remove xyyaw
        fd_output_lens_i = [12]
        action_lens_i = [0]
        policy_input_lens_i = [12-3-4] # remove xyyaw, also z, v_xyz
        limb_types_i = []
        for letter in urdf:
            if letter=='l':
                fd_input_lens_i.append(6)
                fd_output_lens_i.append(6)
                policy_input_lens_i.append(6)
                action_lens_i.append(3)
                limb_types_i.append(1)
            elif letter=='w':
                fd_input_lens_i.append(3)
                fd_output_lens_i.append(3)
                policy_input_lens_i.append(3)
                action_lens_i.append(2)
                limb_types_i.append(2)
            elif letter== 'n':
                fd_input_lens_i.append(0)
                fd_output_lens_i.append(0)
                policy_input_lens_i.append(0)
                action_lens_i.append(0) 
                limb_types_i.append(0)
        fd_input_lens.append(fd_input_lens_i)
        fd_output_lens.append(fd_output_lens_i)
        policy_input_lens.append(policy_input_lens_i)
        action_lens.append(action_lens_i)
        limb_types.append(limb_types_i)
    return fd_input_lens, fd_output_lens, policy_input_lens,action_lens,limb_types