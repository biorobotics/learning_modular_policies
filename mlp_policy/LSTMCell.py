'''
Code adapted from:
https://github.com/emadRad/lstm-gru-pytorch/blob/master/lstm_gru.ipynb
'''
import torch
import torch.nn as nn
# import torchvision.transforms as transforms
# import torchvision.datasets as dsets
# from torch.autograd import Variable
# from torch.nn import Parameter
# from torch import Tensor
import torch.nn.functional as F
import math

class LSTMCell(nn.Module):

    """
    An implementation of Hochreiter & Schmidhuber:
    'Long-Short Term Memory' cell.
    http://www.bioinf.jku.at/publications/older/2604.pdf

    """

    def __init__(self, input_size, hidden_size, bias=True):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.x2h = nn.Linear(input_size, 4 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)
        self.reset_parameters()



    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)
    
    def forward(self, x, hidden):
        
        hx, cx = hidden
        
        x = x.view(-1, x.size(1))
        
        gates = self.x2h(x) + self.h2h(hx)
    
        gates = gates.squeeze()
        
        # ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1) # this breaks when batch_size=1
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, dim=-1) # appears to work for all batch_size

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)
        

        cy = torch.mul(cx, forgetgate) +  torch.mul(ingate, cellgate)        

        hy = torch.mul(outgate, torch.tanh(cy))
        
        return (hy, cy)