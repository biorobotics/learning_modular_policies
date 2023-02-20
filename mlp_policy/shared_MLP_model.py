'''
Source code for paper "Learning modular robot control policies" in Transactions on Robotics
MLP comparisons
Julian Whitman, Dec. 2022. 

MLP network architectures (shared trunk and hardware-conditioned)
 for multiple robot designs

'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class shared_trunk_model(torch.nn.Module):
    def __init__(self, 
        state_input_lens, action_lens, state_output_lens,
        n_hidden_layers=2, 
        hidden_layer_size=250
        ):
        super(shared_trunk_model, self).__init__()   
        self.state_input_lens = state_input_lens
        self.state_output_lens = state_output_lens
        self.action_lens = action_lens
        self.n_hidden_layers = n_hidden_layers   
        self.hidden_layer_size = hidden_layer_size
        self.type = 'shared_trunk'
        assert len(state_input_lens) == len(action_lens), 'Number of states and actions to input and output must match!' 
        # Create a set of input layers and output layers, each with different dims
        self.input_layers = torch.nn.ModuleList()
        for i in range(len(state_input_lens)):
            self.input_layers.append(
                nn.Linear(state_input_lens[i] + action_lens[i],
                 self.hidden_layer_size))


        self.hidden_layers = torch.nn.ModuleList()
        for i in range(n_hidden_layers):
            self.hidden_layers.append( nn.Linear(self.hidden_layer_size,self.hidden_layer_size) )

        self.output_layers_mean = torch.nn.ModuleList()
        self.output_layers_var = torch.nn.ModuleList()
        for i in range(len(action_lens)):
            self.output_layers_mean.append(
                nn.Linear(self.hidden_layer_size, 
                    state_output_lens[i]))
            self.output_layers_var.append(
                nn.Linear(self.hidden_layer_size, 
                    state_output_lens[i]))

        # self.output_layer_mean = torch.nn.Linear(self.hidden_layer_size , output_len)
        # self.output_layer = torch.nn.Linear(self.hidden_layer_size , output_len)

        self.max_logvar = nn.Parameter(torch.tensor(1, dtype=torch.float32) / 2.0)
        self.min_logvar = nn.Parameter(torch.tensor(-1, dtype=torch.float32) * 10.0)

      
    def forward(self,s, a, index): 
        # The input and output layers used depend on the robot index
        x = F.relu(self.input_layers[index](torch.cat([s,a], -1))) # transform to hidden_layer size
        for i in range(self.n_hidden_layers):
            x = F.relu(self.hidden_layers[i](x))
        mu = self.output_layers_mean[index](x)
        logvar = self.output_layers_var[index](x)

        # Avoids exploding std
        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)
        var = torch.exp(logvar)

        return mu, var


class hardware_conditioned_model(torch.nn.Module):
    def __init__(self, 
        max_state_input_lens, max_action_lens,
        max_state_output_lens,
        n_hidden_layers=2, 
        hidden_layer_size=250,
        num_limb_types = 3, num_limbs = 6
        ):
        # max_state_input_lens = list, e.g. [9, 6, 6, 6, 6, 6]
        # max_state_output_lens = list, e.g. [12, 6, 6, 6, 6, 6]
        # max_action_lens = list, e.g. [0, 3, 3, 3,3, 3]
        # Assumes there is a fixed number of modules,
        # i.e. base + 6 limbs
        # Each module has an upper bound (max) on the number of states and actions
        # the rest will be zero padded.
        # limb types are [none, leg, wheel]

        super(hardware_conditioned_model, self).__init__()   
        self.max_state_input_lens = max_state_input_lens
        self.max_state_output_lens = max_state_output_lens
        self.max_action_lens = max_action_lens
        self.n_hidden_layers = n_hidden_layers   
        self.hidden_layer_size = hidden_layer_size
        assert len(max_state_input_lens) == len(max_action_lens), 'Number of states and actions to input and output must match!' 
        self.num_limb_types= num_limb_types
        self.num_limbs= num_limbs
        self.type = 'hardware_conditioned'

        # Create input layer
        self.input_layer = nn.Linear(
            sum(max_state_input_lens) + sum(max_action_lens)  + num_limb_types*num_limbs,
             self.hidden_layer_size)


        self.hidden_layers = torch.nn.ModuleList()
        for i in range(n_hidden_layers):
            self.hidden_layers.append( 
                nn.Linear(self.hidden_layer_size,
                          self.hidden_layer_size) )

        self.output_layer_mean = torch.nn.Linear(self.hidden_layer_size,
         sum(max_state_output_lens))
        self.output_layer_var = torch.nn.Linear(self.hidden_layer_size,
         sum(max_state_output_lens))

        self.max_logvar = nn.Parameter(torch.tensor(1, dtype=torch.float32) / 2.0)
        self.min_logvar = nn.Parameter(torch.tensor(-1, dtype=torch.float32) * 10.0)

      
    def forward(self,state_list, action_list, 
                state_output_lens, limb_type_list): 
        # state_list = a list of input, [[batch x ns1], [batch x ns2]...]
        # action_len = list, example [12, 6,3,6,3,6], len=num_modules
        # assumes state_list is broken up by module.
        # for any that are less than the max, add zeros.
        # This assumes that state list len = len(max_state_lens)
        batch_size = state_list[0].shape[0]
        device = state_list[0].device
        s = []
        for i in range(len(state_list)):
            if state_list[i].shape[-1] < self.max_state_input_lens[i]:
                z = torch.zeros([ batch_size, 
                    self.max_state_input_lens[i] - state_list[i].shape[-1]],
                    dtype=torch.float32, device = device)
                s.append( torch.cat([state_list[i], z], -1) )
            else:
                s.append(state_list[i])
        s = torch.cat(s, -1)
        a = []
        for i in range(len(action_list)):
            if action_list[i].shape[-1] < self.max_action_lens[i]:
                z = torch.zeros([ batch_size, 
                    self.max_action_lens[i] - action_list[i].shape[-1]],
                    dtype=torch.float32, device = device)
                a.append( torch.cat([action_list[i], z], -1) )
            else:
                a.append(action_list[i])
        a = torch.cat(a, -1)

        # create one-hot encoding of module types
        limbs_onehot = torch.zeros(batch_size, self.num_limb_types, self.num_limbs,
                    dtype=torch.float32, device= device)
        for i in range(self.num_limbs):
            limbs_onehot[:, limb_type_list[i], i] = 1

        # The input and output layers used depend on the robot index
        x = torch.cat([s,a, 
                limbs_onehot.view(batch_size, self.num_limb_types*self.num_limbs)
                ], -1)
        # print(x.shape)
        x = F.relu(self.input_layer(x)) # transform to hidden_layer size
        for i in range(self.n_hidden_layers):
            x = F.relu(self.hidden_layers[i](x))
        mu = self.output_layer_mean(x)
        logvar = self.output_layer_var(x)

        # Avoids exploding std
        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)
        var = torch.exp(logvar)

        # divide up outputs and remove extra entries
        var_split = torch.split(var, self.max_state_output_lens, dim=-1)
        mu_split = torch.split(mu, self.max_state_output_lens, dim=-1)
        # print(mu_split)
        mu_out = []
        var_out = []
        for i in range(len(state_output_lens)):
            # if mu_split[i].shape[-1] > state_output_lens[i]:
                # # TODO: is there cleaner way to take the first few elements of the last dimension?
                # This assumes there is a batch dim first
            mu_out.append(mu_split[i][:,:state_output_lens[i]])
            var_out.append(var_split[i][:,:state_output_lens[i]])
            # else:
            #     mu_out.append(mu_split[i])
            #     var_out.append(var_split[i] )
        mu = torch.cat(mu_out, -1)
        var = torch.cat(var_out, -1)

        return mu, var



if __name__ == '__main__':
    # Test networks
    urdf_names= [ 'llllll', 'lwllwl','wnwwnw']
    max_state_input_lens = [9, 6, 6, 6, 6, 6, 6]
    max_state_output_lens = [12, 6, 6, 6, 6, 6, 6]
    max_action_lens = [0, 3, 3, 3, 3, 3, 3]
    state_input_lens, state_output_lens, action_lens = [], [], []
    limb_types = []

    for urdf in urdf_names:
        state_input_lens_i = [9]
        state_output_lens_i = [12]
        action_lens_i = [0]
        limb_types_i = []

        for letter in urdf:
            if letter=='l':
                state_input_lens_i.append(6)
                state_output_lens_i.append(6)
                action_lens_i.append(3)
                limb_types_i.append(1)

            elif letter=='w':
                state_input_lens_i.append(4)
                state_output_lens_i.append(4)
                action_lens_i.append(2)
                limb_types_i.append(2)

            elif letter== 'n':
                state_input_lens_i.append(0)
                state_output_lens_i.append(0)
                action_lens_i.append(0) 
                limb_types_i.append(0)

        state_input_lens.append(state_input_lens_i)
        state_output_lens.append(state_output_lens_i)
        action_lens.append(action_lens_i)
        limb_types.append(limb_types_i)

    state_input_lens_sums = [sum(s) for s in state_input_lens]
    state_output_lens_sums = [sum(s) for s in state_output_lens]
    action_lens_sums = [sum(a) for a in action_lens]
    n_hidden_layers = 6
    hidden_layer_size = 300
    # pgnn was around 600k params
    trunk_NN = shared_trunk_model(
        state_input_lens_sums, action_lens_sums, 
        state_output_lens_sums, n_hidden_layers, hidden_layer_size)
    # count number of parameters    
    num_nn_params= sum(p.numel() for p in trunk_NN.parameters())
    print('Num NN params trunk_NN: ' + str(num_nn_params))
    hc_NN = hardware_conditioned_model(
        max_state_input_lens, max_action_lens, 
        max_state_output_lens, n_hidden_layers, hidden_layer_size)
    num_nn_params= sum(p.numel() for p in hc_NN.parameters())
    print('Num NN params hc_NN: ' + str(num_nn_params))
    batch_size = 10
    for i in range(len(urdf_names)):
        print('urdf_name: ' + str(urdf_names[i]))
        print('state_input_lens: ' + str(state_input_lens[i]))
        print('state_output_lens: ' + str(state_output_lens[i]))
        print('action_len: ' + str(action_lens[i]))
        s = torch.ones(batch_size, state_input_lens_sums[i])
        a = torch.ones(batch_size, action_lens_sums[i])
        print('state/action in sizes ' + str(s.shape) + ' ' + str(a.shape))
        out_mu, out_var = trunk_NN(s,a,i)
        print('trunk out: ' + str(out_mu.shape))
        s_div = torch.split(s, state_input_lens[i], dim=-1)
        a_div = torch.split(a, action_lens[i], dim=-1)
        out_mu, out_var =hc_NN(s_div, a_div,
            state_output_lens[i],limb_types[i] )
        print('hc onehot: ' + str(limb_types[i]))
        print('hc out: ' + str(out_mu.shape))

