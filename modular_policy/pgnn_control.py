'''

Source code for paper "Learning modular robot control policies" in Transactions on Robotics
Julian Whitman, Dec. 2022. 

Functions for probabilistic graph nerual networks (PGNN) for policy learning.



'''
import torch
from LSTMCell import LSTMCell

# GNN Node base class. The node has a type, for bookkeeping. 
# By default, all module types have the same layout of their NN, but there is the option to change it
# by adding subclasses with different methods.
class Node(torch.nn.Module):

    # Define all the different layer types and their sizes in this base class.
    # subclasses have the option to change these for their specific module needs.
    def __init__(self, my_type, internal_state_len, message_len, 
                 input_len, output_len, num_ports,
                 hidden_layer_size = 100, lstm_hidden_size = 50):
        super(Node, self).__init__()        

        self.type = my_type
        self.internal_state_len = internal_state_len
        self.message_len = message_len
        self.input_len = input_len
        self.output_len = output_len
        self.hidden_layer_size = hidden_layer_size
        self.lstm_hidden_size = lstm_hidden_size
        self.num_ports = num_ports

        # input function layers
        self.input_linear = torch.nn.Linear(input_len, internal_state_len )

        # if self.output_len>0:
        self.output_mean = torch.nn.Linear(internal_state_len, output_len )
        self.output_var = torch.nn.Linear(internal_state_len, output_len )
        # initialize_weights([self.input_linear, 
        #     self.output_mean, self.output_var])


        # message output function layers, for number of ports
        self.message_out_linear1_list = torch.nn.ModuleList()
        self.message_out_linear2_list = torch.nn.ModuleList()
        for port in range(self.num_ports):
            self.message_out_linear1_list.append( torch.nn.Linear(internal_state_len, hidden_layer_size ) )
            self.message_out_linear2_list.append( torch.nn.Linear(hidden_layer_size, message_len ) )
        # initialize_weights(self.message_out_linear1_list)
        # initialize_weights(self.message_out_linear2_list)

        # status update function layers
        self.update_lstm = LSTMCell(internal_state_len+message_len*self.num_ports, self.lstm_hidden_size)

        # self.update_lstm = torch.nn.LSTM(internal_state_len+message_len*self.num_ports, self.lstm_hidden_size, 1)
        # initialize with small weights in LSTM 
        # for layer_p in self.update_lstm._all_weights:
        #     for p in layer_p:
        #         if 'weight' in p:
        #             torch.nn.init.normal_(self.update_lstm.__getattr__(p), 0.0, 0.1)
        # self.update_linear = torch.nn.Linear(self.lstm_hidden_size, internal_state_len )
        self.update_linear = torch.nn.Linear(self.lstm_hidden_size, hidden_layer_size )
        # self.update_linear2 = torch.nn.Linear(hidden_layer_size, internal_state_len )
        self.update_linear2 = torch.nn.Linear(hidden_layer_size, hidden_layer_size )
        self.update_linear3 = torch.nn.Linear(hidden_layer_size, internal_state_len )
        # initialize_weights([self.update_linear])

        self.max_logvar = torch.nn.Parameter(torch.ones(1, output_len, dtype=torch.float32) / 2.0)
        self.min_logvar = torch.nn.Parameter(- torch.ones(1, output_len, dtype=torch.float32) * 10.0)


    # input function: h^0 = F(obs) for observation from the world obs
    def run_input_function(self, obs): # precondition: obs is a torch tensor
        x = self.input_linear(obs)
        x = torch.nn.functional.relu(x)
        return x

    # output message function m = M(h). 
    def compute_messages_out_function(self,h,ports_occupied):
        messages = []
        for i in range(self.num_ports):
            if ports_occupied[i] is not None:
                m = self.message_out_linear1_list[i](h)
                m = torch.nn.functional.relu(m)
                m = self.message_out_linear2_list[i](m)
                messages.append(m)
            else:
                messages.append(None)
        return messages

    # transform each input message depending on which port it is
    def compute_aggregation_function(self,messages_in):
        aggregated_message = torch.cat(messages_in,dim=-1)
        return aggregated_message


    # action output is a mu = O(h)
    def compute_output_function(self,h):
        mu = self.output_mean(h)
        logvar = self.output_var(h)

        # Avoids exploding std
        logvar = self.max_logvar - torch.nn.functional.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + torch.nn.functional.softplus(logvar - self.min_logvar)
        var = torch.exp(logvar)

        return mu, var

    # internal state update h^{t+1} = U(h^t, m_bar)  
    def status_update_function(self,h_mbar, lstm_hidden_hc_tuple):

        lstm_hidden_hc_tuple = self.update_lstm(
                h_mbar, lstm_hidden_hc_tuple )
        h = lstm_hidden_hc_tuple[0]
        h2 = self.update_linear(h) 
        h2 = torch.nn.functional.relu(h2)
        h2 = self.update_linear2(h2) 
        h2 = torch.nn.functional.relu(h2)
        h2 = self.update_linear3(h2) 
        return h2, lstm_hidden_hc_tuple
        

## Each module type gets a class which determines its number of inputs and outputs.
# It also provides the opportunity to modify the internal functions as needed per module.
class Chassis(Node): 
    def __init__(self, internal_state_len, message_len, hidden_layer_size,
     goal_len=0, body_input=False):
        input_len = 12 # no goal, not in body frame
        input_len -= 4 # take out pos_z, v_xyz
        input_len += goal_len
        if body_input:
            input_len = input_len-3
        super().__init__("Chassis", internal_state_len, message_len,
                 input_len, 0, 6, hidden_layer_size=hidden_layer_size) 
# my_type, internal_state_len, message_len, input_len=12, output_len=12, num_ports

    # overwrite output function since it does not have any outputs
    def compute_output_function(self,h):
        num_dims = list(h.shape)
        num_dims[-1] = 0
        return torch.FloatTensor(size=num_dims).to(h.device), torch.FloatTensor(size=num_dims).to(h.device)

class Wheel(Node): # sensors, actuators, ports
    # (reads joint angle, outputs joint velocity)
    def __init__(self, internal_state_len, message_len, hidden_layer_size):
        super().__init__("Wheel", internal_state_len, message_len,
            3, 2*2, 1, hidden_layer_size=hidden_layer_size)
        # one angle, two vels = 3. two actuators

class Leg(Node): # sensors 2, actuators 2, ports 1
    # (reads joint angle, outputs joint velocity)
    def __init__(self, internal_state_len, message_len, hidden_layer_size):
        super().__init__("Leg", internal_state_len, message_len,
         6, 3*2, 1, hidden_layer_size=hidden_layer_size)
        # three angle, three vels = 6. three actuators

# Module class, each module added to the robot is one of these. The module has a GNN node, which is shared 
# among all modules of the same type, but each module has an internal state and messages which are not shared.
class Module():
    def __init__(self, my_id, my_gnn, device):
        self.my_id = my_id
        self.my_gnn  = my_gnn # module gets initialized with a pointer to the NN 
        self.internal_state  = torch.zeros(my_gnn.internal_state_len, device=device)
        self.lstm_hidden_hc_tuple = (torch.zeros(my_gnn.lstm_hidden_size, device=device),
            torch.zeros(my_gnn.lstm_hidden_size, device=device))

        self.messages_in  = []
        self.messages_out = []
        self.aggregated_messages  = torch.zeros(my_gnn.message_len, device=device)
        self.device = device
        self.num_ports = my_gnn.num_ports
        
    # input function: h^0 = F(x) for observation from the world
    def run_input_function(self, observation): # precondition: x is a torch tensor.
        self.internal_state  = self.my_gnn.run_input_function(observation)

    # output message function m = M(h). 
    def compute_messages_out(self, ports_occupied):
        am_out = self.my_gnn.compute_messages_out_function(self.internal_state, ports_occupied)
        return am_out

    
    # message aggregation m_bar = A(m1, m2...). All modules use the same aggregation function.
    # This function must be able to accept variable number of inputs 
    def aggregate_messages(self): 
        self.aggregated_messages = self.my_gnn.compute_aggregation_function(self.messages_in)

    # action output is a mu = O(h)
    def compute_output(self):
        mu, var = self.my_gnn.compute_output_function(self.internal_state)
        return mu, var

            
    # internal state update h^{t+1} = U(h^t, m_bar)  
    def status_update(self):
        self.internal_state, self.lstm_hidden_hc_tuple  = self.my_gnn.status_update_function(
                torch.cat([self.internal_state, self.aggregated_messages], dim=-1),
                 self.lstm_hidden_hc_tuple)

    def reset_hidden_states(self, batch_size=1):
        self.lstm_hidden_hc_tuple = (
            torch.zeros(batch_size, self.my_gnn.lstm_hidden_size, device=self.device),
            torch.zeros(batch_size, self.my_gnn.lstm_hidden_size, device=self.device) )


# creates a neural network for each module type. These nodes are shared by all modules of that type.
def create_GNN_nodes(internal_state_len, message_len, hidden_layer_size, 
    device, goal_len=0, body_input=False):
    shared_nodes = []

    # append each type manually here for the different types of modules
    # Module type 0: body
    # internal_state_len, message_len, hidden_layer_size
    shared_nodes.append(Chassis(internal_state_len=internal_state_len, 
                            message_len=message_len, 
                            hidden_layer_size=hidden_layer_size,
                            goal_len=goal_len, body_input=body_input).to(device))
    # Module type 1: leg
    shared_nodes.append(Leg(internal_state_len=internal_state_len, 
                            message_len=message_len, 
                            hidden_layer_size=hidden_layer_size).to(device))
   # module type 2: wheel
    shared_nodes.append(Wheel(internal_state_len=internal_state_len, 
                            message_len=message_len, 
                            hidden_layer_size=hidden_layer_size).to(device))
    # later more node types here

    return shared_nodes

def get_GNN_params_list(GNN_nodes):
    params_list = list()
    for GNN_node in GNN_nodes:
        params_list+=list(GNN_node.parameters())
    return params_list

def zero_grads(GNN_nodes):
    for GNN_node in GNN_nodes:
        GNN_node.zero_grad()

def get_state_dicts(GNN_nodes):
    save_dict = dict()
    for GNN_node in GNN_nodes:
        save_dict[GNN_node.type] = GNN_node.state_dict()
    return save_dict

def load_state_dicts(GNN_nodes, save_dict):
    for GNN_node in GNN_nodes:
        GNN_node.load_state_dict(save_dict[GNN_node.type])


# Runs propogation steps on the GNN.
# precondition: len(sensor_data) = len(modules_list)
def run_propagations(modules_list, attachments, num_propagation_steps, node_inputs, device):
    # modules_list is a list of Module objects
    # node_inputs is a list where each entry contains the [state,action] input to that node
    # Node_types = [Node_ID_1_type, Node_ID_2_type…]
    # Edge = [(parent_ID, parent_port), child_ID] 
    # Edges = [Edge_1, Edge_2…]
    # Attachments = [ (child_ID_on_module_ID0_port0, child_ID_on_module_ID0_port1...), (child_ID_on_module_ID1_port0, child_ID_on_module_ID1_port0...), …]
    # example:
    # Attachments = [(None, 1,2), (0),(0)] # module 0 has (1,2) on its ports (1,2), and module 1,2 have zeros on their inputs but no outputs.
    
    n_modules = len(modules_list)
    batch_size = node_inputs[0].shape[0] # ASSUMES that all node_inputs are same first shape len

    # start by taking input observation to create internal state.
    for i in range(n_modules):
        modules_list[i].run_input_function(node_inputs[i].to(device))
        # print('Input ** ' + modules_list[i].actor_GNN.type + ' ' + str(modules_list[i].my_id) + ' **')
        # print(sensor_data[i])
        # print(modules_list[i].state_actor)

    # next take some propagation steps, 
    for step in range(num_propagation_steps): # multiple propogations
        # reset input message buffers
        all_messages_out = []
        # each module computes an output message for each of its ports
        for i in range(n_modules):
            module =  modules_list[i]
            module.messages_in = []
            m_a_list = module.compute_messages_out(attachments[i])
            all_messages_out.append(m_a_list) # output messages from each module
            module.messages_out = m_a_list

        # send messages 
        for i in range(n_modules):
            module =  modules_list[i]
            ports_occupied = attachments[i] 
            # e.g. [None, 1, 3] means module is recieving from nothing on its input and 1 and 3 on outputs
            # e.g. [2, 1, 3] means module is recieving from module 2 on its input and 1 and 3 on outputs
            # print('ports_occupied ' +str(ports_occupied))
            for port_num in range(len(ports_occupied)):
                attach_ind = ports_occupied[port_num]
                if attach_ind is not None:
                    # print('port num: ' +str(port_num))
                    # print('attach_ind: ' +str(attach_ind))
                    if port_num==0: # port0 is the input port, so all modules should have at least this port
                        # print('attachments[attach_ind].index(i) ' + str(attachments[attach_ind].index(i)))

                        # use the location of module_i in attachments[attach_ind]
                        module.messages_in.append(all_messages_out[attach_ind][attachments[attach_ind].index(i)])
                        # print(str(i) + ' from ' + str(attach_ind) + ' port ' + str(attachments[attach_ind].index(i)))

                    else:
                        module.messages_in.append(all_messages_out[attach_ind][0])
                        # print(str(i) + ' from ' + str(attach_ind) + ' port 0')
                else:
                    module.messages_in.append( 
                        torch.zeros(batch_size,module.my_gnn.message_len, device=device))


        # aggregation of messages, internal state update
        last_step = (step == num_propagation_steps-1)
        for module in modules_list:
            # don't bother updating internal state if this node has not outputs and
            # is on its last update prop, since that info would not be used
            # (should buy a small efficiency gain since it's one less forward pass)
            if not(last_step and module.my_gnn.output_len==0):
                module.aggregate_messages()
                module.status_update()

    # run output function to get actions out
    outputs_mu = [] 
    outputs_var = [] 
    for module in modules_list:
        mu,var = module.compute_output()
        outputs_mu.append(mu)
        outputs_var.append(var)

        # print('Output ** ' + module.actor_GNN.type + ' ' + str(module.my_id) + ' **')
        # print(module.messages_in_actor)
        # print(messages_out_actor)
        # print(module.aggregated_messages_actor)
        # print(module.state_actor)

    # return torch.cat(actions_out), torch.cat(sigmas_out), torch.cat(V_out)
    return outputs_mu, outputs_var
