import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
class Relational_Embeding(Module):
    """

    """

    def __init__(self, input_dim, output_dim, hr=2,bias=True):
        super(Relational_Embeding, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.W = nn.Parameter(torch.FloatTensor(input_dim, input_dim))
        #self.W = nn.Parameter(nn.init.xavier_uniform(torch.Tensor(input_dim, int(input_dim / hr)).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), gain=np.sqrt(2.0)))
        self.H = nn.Parameter(torch.FloatTensor(input_dim, int(input_dim / hr)))
        self.fc0 = nn.Linear(int(input_dim / hr), input_dim)
        #self.fc0.weight = torch.nn.init.xavier_normal(self.fc0.weight, gain=np.sqrt(2.0))
        self.fc1 = nn.Linear(input_dim, output_dim)
        #self.fc1.weight = torch.nn.init.xavier_normal(self.fc1.weight, gain=np.sqrt(2.0))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(1))
        self.W.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.H.size(1))
        self.H.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.fc0.weight.size(1))
        self.fc0.weight.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.fc1.weight.size(1))
        self.fc1.weight.data.uniform_(-stdv, stdv)



    def forward(self, input, im_inds):
        '''

        Args:
            input: shape: num_obj, hid_dim
            adj:   shape: num_obj, num_obj

        Returns:

        '''

        #print('input: ',input)
        rel1_ = F.sigmoid(torch.mm(torch.mm(input, self.W), input.permute(1, 0)))
        #print('rel1_: ', rel1_)
        rel1_exp = torch.exp(rel1_)
        rel1_exp = rel1_exp * (im_inds[:,None]==im_inds[None,:]).type_as(rel1_exp)
        #print('rel1_exp: ', rel1_exp)
        #print('(torch.sum(rel1_exp, -1)+1e-8): ',(torch.sum(rel1_exp, -1)+1e-8)[:,None])
        rel1 = rel1_exp / (torch.sum(rel1_exp, -1)+1e-8)[:,None]
        #print('rel1: ', rel1)
        output1 = input + self.fc0(torch.mm(rel1, torch.mm(input, self.H)))
        output2 = self.fc1(output1)
        return output1, output2
