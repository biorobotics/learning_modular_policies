'''
Source code for paper "Learning modular robot control policies" in Transactions on Robotics
MLP comparisons
Julian Whitman, Dec. 2022. 

MLP for multiple robot designs


'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class shared_trunk_policy(torch.nn.Module):
    def __init__(self, 
        state_lens, action_lens, goal_len,
        n_hidden_layers=2, 
        hidden_layer_size=250
        ):
        super(shared_trunk_policy, self).__init__()   
        self.state_lens = state_lens
        self.goal_len = goal_len
        self.action_lens = action_lens
        self.type = 'shared_trunk'
        self.n_hidden_layers = n_hidden_layers   
        self.hidden_layer_size = hidden_layer_size
        assert len(state_lens) == len(action_lens), 'Number of states and actions to input and output must match!' 
        # Create a set of input layers and output layers, each with different dims
        self.input_layers = torch.nn.ModuleList()
        for i in range(len(state_lens)):
            self.input_layers.append(
                nn.Linear(state_lens[i] + goal_len, self.hidden_layer_size))


        self.hidden_layers = torch.nn.ModuleList()
        for i in range(n_hidden_layers):
            self.hidden_layers.append( nn.Linear(self.hidden_layer_size,self.hidden_layer_size) )

        self.output_layers_mean = torch.nn.ModuleList()
        self.output_layers_var = torch.nn.ModuleList()
        self.output_layers_tmean = torch.nn.ModuleList()
        self.output_layers_tvar = torch.nn.ModuleList()
        for i in range(len(action_lens)):
            self.output_layers_mean.append(
                nn.Linear(self.hidden_layer_size, action_lens[i]))
            self.output_layers_var.append(
                nn.Linear(self.hidden_layer_size, action_lens[i]))
            self.output_layers_tmean.append(
                nn.Linear(self.hidden_layer_size, action_lens[i]))
            self.output_layers_tvar.append(
                nn.Linear(self.hidden_layer_size, action_lens[i]))
        # self.output_layer_mean = torch.nn.Linear(self.hidden_layer_size , output_len)
        # self.output_layer = torch.nn.Linear(self.hidden_layer_size , output_len)

        self.max_logvar = nn.Parameter(torch.tensor(1, dtype=torch.float32) / 2.0)
        self.min_logvar = nn.Parameter(torch.tensor(-1, dtype=torch.float32) * 10.0)

      
    def forward(self,s, g, index): 
        # The input and output layers used depend on the robot index
        x = F.relu(self.input_layers[index](torch.cat([s,g], -1))) # transform to hidden_layer size
        for i in range(self.n_hidden_layers):
            x = F.relu(self.hidden_layers[i](x))
        mu = self.output_layers_mean[index](x)
        logvar = self.output_layers_var[index](x)

        mut = self.output_layers_tmean[index](x)
        logvart = self.output_layers_tvar[index](x)

        # Avoids exploding std
        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)
        var = torch.exp(logvar)

        logvart = self.max_logvar - F.softplus(self.max_logvar - logvart)
        logvart = self.min_logvar + F.softplus(logvart - self.min_logvar)
        vart = torch.exp(logvart)

        # vel_mean, vel_var, ff_tau_mean, ff_tau_var
        return mu, var, mut, vart


class hardware_conditioned_policy(torch.nn.Module):
    def __init__(self, 
        max_state_lens, max_action_lens, goal_len,
        n_hidden_layers=2, 
        hidden_layer_size=250,
        num_limb_types = 3, num_limbs = 6
        ):
        # max_state_lens = list, e.g. [12, 6, 6, 6, 6, 6]
        # max_action_lens = list, e.g. [0, 3, 3, 3,3, 3]
        # goal_len = integer
        # Assumes there is a base with a fixed number of modules,
        # i.e. base + 6 limbs
        # Each module has an upper bound (max) on the number of states and actions
        # the rest will be zero padded.
        # limb types are [none, leg, wheel]

        super(hardware_conditioned_policy, self).__init__()   
        self.max_state_lens = max_state_lens
        self.goal_len = goal_len
        self.max_action_lens = max_action_lens
        self.n_hidden_layers = n_hidden_layers   
        self.hidden_layer_size = hidden_layer_size
        self.num_limb_types= num_limb_types
        self.num_limbs= num_limbs
        self.type = 'hardware_conditioned'

        assert len(max_state_lens) == len(max_action_lens), 'Number of states and actions to input and output must match!' 
        

        # Create  input layers and output layers
        self.input_layer = nn.Linear(
            sum(max_state_lens) + goal_len + num_limb_types*num_limbs,
             self.hidden_layer_size)


        self.hidden_layers = torch.nn.ModuleList()
        for i in range(n_hidden_layers):
            self.hidden_layers.append( 
                nn.Linear(self.hidden_layer_size,
                          self.hidden_layer_size) )

        self.output_layer_mean = torch.nn.Linear(self.hidden_layer_size , sum(max_action_lens))
        self.output_layer_var = torch.nn.Linear(self.hidden_layer_size , sum(max_action_lens))

        self.output_layer_tmean = torch.nn.Linear(self.hidden_layer_size , sum(max_action_lens))
        self.output_layer_tvar = torch.nn.Linear(self.hidden_layer_size , sum(max_action_lens))

        self.max_logvar = nn.Parameter(torch.tensor(1, dtype=torch.float32) / 2.0)
        self.min_logvar = nn.Parameter(torch.tensor(-1, dtype=torch.float32) * 10.0)

      
    def forward(self,state_list, g, action_len, limb_type_list): 
        # state_list = a list of input, [[batch x ns1], [batch x ns2]...]
        # g = goals [batch x goal_len]
        # action_len = list, example [12, 6,3,6,3,6], len=num_modules
        # assumes state_list is broken up by module.
        # for any that are less than the max, add zeros.
        # This assumes that state list len = len(max_state_lens)
        # limb_type_list = [1,0,2,0,1] where 0=no limb, 1= type1, etc.
        # as long as the limb type order is consistent, it does not matter which one is index 1 or 2 
        batch_size = g.shape[0]
        s = []
        for i in range(len(state_list)):
            if state_list[i].shape[-1] < self.max_state_lens[i]:
                z = torch.zeros([batch_size , 
                    self.max_state_lens[i] - state_list[i].shape[-1]],
                    dtype=torch.float32, device = g.device)
                s.append( torch.cat([state_list[i], z], -1) )
            else:
                s.append(state_list[i])
        s = torch.cat(s, -1)

        # create one-hot encoding of module types
        limbs_onehot = torch.zeros(batch_size, self.num_limb_types, self.num_limbs,
                    dtype=torch.float32, device= g.device)
        for i in range(self.num_limbs):
            limbs_onehot[:, limb_type_list[i], i] = 1
        # print(limbs_onehot)
        # The input and output layers used depend on the robot index
        x = F.relu(self.input_layer(torch.cat([s,g, 
                limbs_onehot.view(batch_size, self.num_limb_types*self.num_limbs)
                ], -1))) # transform to hidden_layer size
        for i in range(self.n_hidden_layers):
            x = F.relu(self.hidden_layers[i](x))
        mu = self.output_layer_mean(x)
        logvar = self.output_layer_var(x)
        mut = self.output_layer_tmean(x)
        logvart = self.output_layer_tvar(x)

        # Avoids exploding std
        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)
        var = torch.exp(logvar)

        logvart = self.max_logvar - F.softplus(self.max_logvar - logvart)
        logvart = self.min_logvar + F.softplus(logvart - self.min_logvar)
        vart = torch.exp(logvart)

        # divide up outputs and remove extra entries
        res = []
        for output in [mu, var, mut, vart]:
            output_split = torch.split(output, self.max_action_lens, dim=-1)
            output_out = []
            for i in range(len(action_len)):
                # print(output_split[i].shape)
                # if output_split[i].shape[-1] > action_len[i]:
                    # TODO: is there clean way to take the first few elements of the last dimension?
                output_out.append(output_split[i][:,:action_len[i]])
                # else:
                #     output_out.append(output_split[i])
            res.append( torch.cat(output_out, -1) )


        # vel_mean, vel_var, ff_tau_mean, ff_tau_var
        mu, var, mut, vart = res
        return mu, var, mut, vart


if __name__ == '__main__':
    # Test networks
    urdf_names= [ 'llllll', 'lwllwl','wnwwnw']
    max_state_lens = [9, 6, 6, 6, 6, 6, 6]
    max_action_lens = [0, 3, 3, 3, 3, 3, 3]
    state_lens, action_lens = [], []
    limb_types = []
    for urdf in urdf_names:
        state_lens_i = [9]
        action_lens_i = [0]
        limb_types_i = []
        for letter in urdf:
            if letter=='l':
                state_lens_i.append(6)
                action_lens_i.append(3)
                limb_types_i.append(1)
            elif letter=='w':
                state_lens_i.append(4)
                action_lens_i.append(2)
                limb_types_i.append(2)
            elif letter== 'n':
                state_lens_i.append(0)
                action_lens_i.append(0)   
                limb_types_i.append(0)
        limb_types.append(limb_types_i)
        state_lens.append(state_lens_i)
        action_lens.append(action_lens_i)
    state_lens_sums = [sum(s) for s in state_lens]
    action_lens_sums = [sum(a) for a in action_lens]

    n_hidden_layers = 6
    hidden_layer_size = 350
    # pgnn_control was around 780k params

    goal_len = 3
    trunk_NN = shared_trunk_policy(
        state_lens_sums, action_lens_sums, goal_len,
        n_hidden_layers= n_hidden_layers, 
        hidden_layer_size = hidden_layer_size)
    # count number of parameters    
    num_nn_params= sum(p.numel() for p in trunk_NN.parameters())
    print('Num NN params trunk_NN: ' + str(num_nn_params))
    hc_NN = hardware_conditioned_policy(
        max_state_lens, max_action_lens, goal_len,
        n_hidden_layers= n_hidden_layers, 
        hidden_layer_size = hidden_layer_size)
    num_nn_params= sum(p.numel() for p in hc_NN.parameters())
    print('Num NN params hc_NN: ' + str(num_nn_params))
    batch_size = 10
    g = torch.zeros(batch_size, goal_len)
    for i in range(len(urdf_names)):
        print('urdf_name: ' + str(urdf_names[i]))
        print('state_len: ' + str(state_lens[i]))
        print('action_len: ' + str(action_lens[i]))
        s = torch.ones(batch_size, state_lens_sums[i])
        out_mu, out_var, _,_ = trunk_NN(s,g,i)
        print('trunk out: ' + str(out_mu.shape))
        s_div = torch.split(s, state_lens[i], dim=-1)
        out_mu, out_var, _,_ =hc_NN(s_div, g, action_lens[i], limb_types[i])
        print('hc onehot: ' + str(limb_types[i]))
        print('hc out: ' + str(out_mu.shape))

