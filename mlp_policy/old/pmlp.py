import torch
import torch.nn as nn
import torch.nn.functional as F

# multi-layer perceptron general class
class pmlp(torch.nn.Module):
    def __init__(self, 
        input_len=1, output_len=1, n_hidden_layers=2, hidden_layer_size=500):
        super(pmlp, self).__init__()   
        self.input_len=  input_len
        self.output_len=output_len
        self.n_hidden_layers = n_hidden_layers   
        self.hidden_layer_size = hidden_layer_size
        self.input_layer = torch.nn.Linear(input_len, self.hidden_layer_size )
        self.hidden_layers = torch.nn.ModuleList()
        for i in range(n_hidden_layers):
            self.hidden_layers.append( nn.Linear(self.hidden_layer_size,self.hidden_layer_size) )
        self.output_layer_mean = torch.nn.Linear(self.hidden_layer_size , output_len)
        self.output_layer_var = torch.nn.Linear(self.hidden_layer_size , output_len)

        self.max_logvar = nn.Parameter(torch.ones(1, output_len, dtype=torch.float32) / 2.0)
        self.min_logvar = nn.Parameter(- torch.ones(1, output_len, dtype=torch.float32) * 10.0)


    def forward(self,s,u): # concatenates states and controls
        x = torch.cat((s, u), dim = -1)
        x = F.relu(self.input_layer(x))
        for i in range(self.n_hidden_layers):
            x = F.relu(self.hidden_layers[i](x))
        mu = self.output_layer_mean(x)
        logvar = self.output_layer_var(x)

        # Avoids exploding std
        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)
        var = torch.exp(logvar)
        # var = F.softplus(var)
        return mu, var