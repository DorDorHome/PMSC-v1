import math
import random
import functools
import operator

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function

from op import FusedLeakyReLU, fused_leaky_relu, upfirdn2d, conv2d_gradfix

class EqualLinear(nn.Module):
    """
    FC layers used in the mapping network from z to w,
    Also used in the affine map (1-layer) from w to each s (style layers)

    Hyperparameter settings:
    lr_mul
    
    """
    def __init__(
        self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul)) #weights are scaled by the inverse of lr_mul.

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        else:
            self.bias = None

        self.activation = activation
        if in_dim == 0:
            self.scale=lr_mul
        else:
            self.scale = (1 / math.sqrt(in_dim)) * lr_mul # shape (out_dim)
        self.lr_mul = lr_mul

    def forward(self, input):
        if self.activation=="fused_lrelu":
            # debug:
            # print('shape of input in EqualLinear:', input.shape)
            # print('shape of the weight in EqualLinear: ', self.weight.shape)
            out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)
        elif self.activation == 'tanh':
            out = F.linear(
                input, self.weight * self.scale, bias=self.bias * self.lr_mul
            )
            out = torch.tanh(out)

        elif self.activation == 'sqrt_softmax': 
            # before going through softmax, first multiply the style output by a constant
            out = F.linear(input, self.weight * self.scale, bias=self.bias * self.lr_mul) # shape: (batch dim, out_dim )
            out = F.softmax(out, dim= 1) # shape: (batch_dim, out_dim)
            out = torch.sqrt(out)
            
        else: # used in first layer, and ToRBG layers
            out = F.linear(
                input, self.weight * self.scale, bias=self.bias * self.lr_mul
            )

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})"
        )

class TwoLayerHypernetwork(nn.Module):
    """
    A two layer FC.
    w_prev
    
    """
    def __init__(self, in_dim_prev, in_dim_current, out_dim, middle_dim, bias_init = 0, lr_mul = 1, hidden_layer_activation = None, final_activation= None):
        super().__init__()
        self.FC1 = EqualLinear(in_dim=in_dim_prev, out_dim=middle_dim, bias = True, bias_init = bias_init, lr_mul = lr_mul, activation= hidden_layer_activation)
        self.FC2 = EqualLinear(in_dim=middle_dim, out_dim=out_dim*(in_dim_current+1), bias = True, bias_init = bias_init, lr_mul = lr_mul, activation= final_activation)
        

    def forward(self, prev_correlated_w, current_w):

        # first FC1 and FC2 takes the prev_correlated_w as input:

        hyper_out1 = self.FC1(prev_correlated_w)# shape (batch, middle_dim)
        hyper_out1 = self.FC2(hyper_out1)#shape (batch, out_dim*in_dim_current)

        hyper_out1 = hyper_out1.view(current_w.shape[0], in_dim_current + 1, out_dim) # shape (batch, in_dim_current + 1, out_dim )
        current_w = current_w.view(current_w.shape[0], 1,-1) # shape (batch, 1, in_dim_current)
        out = torch.matmul(current_w, hyper_out1[:, :in_dim_current, out_dim])       + hyper_out1[:, in_dim_current, out_dim]     # shape ( batch, )
        out = out.view(current_w.shape[0], -1)

        return out




print(torch.zeros(3,4,5)[:,:3,:].shape)