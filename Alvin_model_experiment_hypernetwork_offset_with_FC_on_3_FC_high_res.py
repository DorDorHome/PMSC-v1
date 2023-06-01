# this is changed from hypternetwork_with_FC file

# This model was adapted from Alvin_model_experiment_changed_mapping_network_with_reduced_RGB_style_dim
# the difference is that an extra 2layer FC network is added to the first 
# __________________________________________________
# This model is changed from Alvin_model_
# 
# experiment_changed_mapping_network_with_style_projection_to_sphere_scaled.py

# the main difference is the the projection to sphere is cancelled. the only difference from the standard model is 
# the changes made in the mapping network.


# This model is further changed from Alvin_model_experiment_with_style_projection_to_sphere_scaled.py

# 1. The mapping network (z to w) is cancelled, replaced by a series of w, generated for each conv layer
# 2. each w (one for each conv layer) is set to have dimension 32
# 2.2 this requires a change in StyleEqualLinear.
# 2.3 the ToRGB layer needs to be changed, with corresponding input from the w. But how?
# 3. from each w, mapped to s for that layer directly. This is achieved within StyleEqualLinear. No need to set again within MyNewGenerator
# 4. however, from the second layer onward, a 2-layered FC network is used to map w_i to another input into the affine map for w_(i+1)
# 

#_______________________________________________________
# this is updated from Alvin_model_experiment, with the following changes made:

#8. in NewModulatedConv2d, compared to style_projection_to_sphere, I forgot to add back the scale when activation is not used. This is corrected


# 4.spherical projection is used in place of softmax
# 5. a bias term is added back to the affine map for style space 
# 6. when applying StyleEqualLinear in NewModulatedConv2d, bias is init to zero
# 7. In view of the self.scale 


# copy from model.py and then adapt:
# 1. Generator was changed to MyGenerator
# 2. ModulatedConv2d changed to NewModulatedConv2d, with actual changes
# 3. StyleConv changed to NewStyleConv

# within NewModulatedConv, demodulate is changed to be over each in_channel, out_channel pair

# within MyGenerator, the followings were changed:


import math
import random
import functools
import operator

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function

from op import FusedLeakyReLU, fused_leaky_relu, upfirdn2d, conv2d_gradfix


class PixelNorm(nn.Module):

    """
    (added by Alvin:)
    This layer normalize the length (l_2 norm) of each channel to be the size of the image.
    
    
    """
    def __init__(self):
        super().__init__()

    def forward(self, input):
        # note that torch.rsqrt means reciprocal of the square root.
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8) 


def make_kernel(k):
    """
    (added by Alvin)
    for 1-dim tensor with entries x_i,
    This function first creates a 2d tensor (matrix) with entries x_i*x_j
    Followed by normalizing the values so adding all entries =1 

    For higher dimensional tensor, 
    this just normalizing the values so adding all entries =1 
    
    
    """
    k = torch.tensor(k, dtype=torch.float32)

    if k.ndim == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()

    return k


class Upsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel) * (factor ** 2)
        self.register_buffer("kernel", kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=self.factor, down=1, pad=self.pad)

        return out


class Downsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel)
        self.register_buffer("kernel", kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=1, down=self.factor, pad=self.pad)

        return out


class Blur(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()

        kernel = make_kernel(kernel)

        if upsample_factor > 1:
            kernel = kernel * (upsample_factor ** 2)

        self.register_buffer("kernel", kernel)

        self.pad = pad

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, pad=self.pad)

        return out


class EqualConv2d(nn.Module):
    # this is different from the one used in Generator
    # used in discriminator only
    def __init__(
        self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(out_channel, in_channel, kernel_size, kernel_size)
        )
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))

        else:
            self.bias = None

    def forward(self, input):
        out = conv2d_gradfix.conv2d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},"
            f" {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})"
        )




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

class TwoLayerFC(nn.Module):
    """
    Created by Alvin, to get the correction of the affine map from the style of the previous layer.
    
    """
    def __init__(self, in_dim, out_dim, middle_dim, bias_init = 0, lr_mul = 1, hidden_layer_activation = None, final_activation= None):
        super().__init__()
        self.FC1 = EqualLinear(in_dim=in_dim, out_dim=middle_dim, bias = True, bias_init = bias_init, lr_mul = lr_mul, activation= hidden_layer_activation)
        self.FC2 = EqualLinear(in_dim=middle_dim, out_dim=out_dim, bias = True, bias_init = bias_init, lr_mul = lr_mul, activation= final_activation)

    def forward(self, input):
        """
        input is expected to be a tensor of shape (batch, in_dim) , in_dim is expected to be 32
        
        
        """
        out = self.FC1(input)
        out = self.FC2(out)
        return out

class ThreeLayer_FC_with_2FC_Hypernetwork(nn.Module):
    """
    A two layer FC.
    w_prev
    
    """
    def __init__(self, in_dim_prev, in_dim_current, out_dim, first_hidden_dim_main, second_hidden_dim_main, hidden_dim_hyper ,bias_init = 0, lr_mul_main = 1, lr_mul_offset = 1, scale_of_offset = 0.01, hidden_layer_activation = None, final_activation= None):
        super().__init__()
        self.in_dim_prev = in_dim_prev
        self.in_dim_current = in_dim_current

        self.first_hidden_dim_main=first_hidden_dim_main
        self.second_hidden_dim_main= second_hidden_dim_main

        self.hidden_dim_hyper = hidden_dim_hyper

        self.out_dim = out_dim

        self.hidden_layer_activation = hidden_layer_activation
        self.final_activation = final_activation

        self.lr_mul_main = lr_mul_main
        self.lr_mul_offset = lr_mul_offset

        self.scale_of_offset = scale_of_offset

        # these represent the main forward pass of mapping from current layer latent (w, 32-dimension) to stylespace (s)
        self.FC_main_1 = EqualLinear(in_dim=in_dim_current, out_dim=first_hidden_dim_main, bias = True, bias_init = bias_init, lr_mul = lr_mul_main, activation= hidden_layer_activation)
        self.FC_main_2 = EqualLinear(in_dim = first_hidden_dim_main, out_dim = second_hidden_dim_main,  bias = True, bias_init = bias_init, lr_mul = lr_mul_main, activation= hidden_layer_activation)
        self.weight_FC_main_3 = nn.Parameter(torch.randn(out_dim, second_hidden_dim_main).div_(lr_mul_main))
        # bias is not reqired, as it will be absorbed by the hypernetwork
        #self.bias_FC_main_2 = nn.parameter()


       

        # these represent the correction to the map above, according to the values of the previous main forward pass
        # the correction should be on the parameters of the last layer (FC_main_2)
        self.FC_offset_1 = EqualLinear(in_dim=in_dim_prev, out_dim= hidden_dim_hyper, bias = True, bias_init = bias_init, lr_mul = lr_mul_offset, activation= hidden_layer_activation)
        self.FC_offset_2 = EqualLinear(in_dim= hidden_dim_hyper, out_dim=out_dim*(second_hidden_dim_main+1), bias = True, bias_init = bias_init, lr_mul = lr_mul_offset, activation= final_activation)
        
        # we also need to correct for the input variance
        self.scale_main_3 = 1/math.sqrt(second_hidden_dim_main)*lr_mul_main
        
    def forward(self, prev_correlated_w, current_w):
        """
        current_w: shape (batch, in_dim_current) 
        prev_correlated_w: shape (batch, in_dim_prev)
        
        
        """

        # the offset network takes prev_correlated_w as input:

        off_set_to_main_3 = self.FC_offset_1(prev_correlated_w)#shape (batch, middle_dim_hyper)
        off_set_to_main_3 = self.FC_offset_2(off_set_to_main_3)# shape (batch, out_dim*(middle_dim_main+1))
        off_set_to_main_3 = self.scale_of_offset*off_set_to_main_3.view(prev_correlated_w.shape[0], self.out_dim, self.second_hidden_dim_main +1 ) # shape(batch, out_dim, middle_dim_main +1)
        # print('!!warning: please set off_set !!!!!!!!!!!!! ')
        # off_set_to_main_2 = torch.zeros(prev_correlated_w.shape[0], self.out_dim, self.middle_dim_main +1 ).to('cuda:0')

        # the main network takes the current_w as input:
        # print('batch size = ', current_w.shape[0])
        # first layer of main
        main_out  = self.FC_main_1(current_w)

        # 2nd layer of main:
        main_out  = self.FC_main_2(main_out).view(current_w.shape[0], self.second_hidden_dim_main, 1)
        # shape (batch, middle_dim_main)
        # second layer of main should be have its weights altered by the output of the offset:
        
        # first alter the weight of the 2nd main FC:
        weight_FC_3 = self.weight_FC_main_3 + off_set_to_main_3[:, :, :self.second_hidden_dim_main]# shape (batch, out_dim, middle_dim_main)

        # forward pass of the 2nd FC:
        if self.final_activation=="fused_lrelu":
            # print('shape of main_out:', main_out.shape)
            # print('shape of weight_FC_2:', weight_FC_2.shape)
            # print(' shape of torch.mul(weight_FC_2, main_out): ', torch.matmul(weight_FC_2, main_out).shape)
            
            # print("shape of torch.mul(weight_FC_2, main_out).view(current_w.shape[0], self.out_dim) is", torch.matmul(weight_FC_2, main_out).view(current_w.shape[0], self.out_dim).shape)
            # print('shape of  off_set_to_main_2[:,:,self.middle_dim_main] is ',  off_set_to_main_2[:,:,self.middle_dim_main].shape)
            # main_out = F.linear(main_out, weight_FC_2*self.scale_main_2)
            main_out = torch.matmul(weight_FC_3, main_out).view(current_w.shape[0], self.out_dim) + off_set_to_main_3[:,:,self.second_hidden_dim_main]*self.lr_mul_main
            main_out = F.leaky_relu(main_out)


            # main_out = fused_leaky_relu(main_out,  off_set_to_main_2[:, :, self.middle_dim_main]*self.lr_mul_main)

        elif self.final_activation=='tanh':
            main_out = torch.matmul(weight_FC_3, main_out).view(current_w.shape[0], self.out_dim) + off_set_to_main_3[:,:,self.second_hidden_dim_main]*self.lr_mul_main
            
            main_out = torch.tanh(main_out)


        # elif self.final_activation=='sqrt_softmax':
        #     main_out = torch.matmul(weight_FC_2, main_out).view(current_w.shape[0], self.out_dim) + off_set_to_main_2[:,:,self.middle_dim_main]*self.lr_mul_main # shape: (batch dim, out_dim )
        #     main_out = F.softmax(main_out, dim= 1) # shape: (batch_dim, out_dim)
        #     main_out = torch.sqrt(main_out)

        else:

            main_out = torch.matmul(weight_FC_3, main_out).view(current_w.shape[0], self.out_dim) + off_set_to_main_3[:,:,self.second_hidden_dim_main]*self.lr_mul_main



        return main_out




class ThreeLayerFC(nn.Module):
    
    """
    Created by Alvin, to get the correction of the affine map from the style of the previous layer.
    
    """
    def __init__(self, in_dim, out_dim, hidden_dim_1st, hidden_dim_2nd, bias_init = 0, lr_mul = 1, hidden_layer_activation = None, final_activation = None):
        super().__init__()
        self.FC1 = EqualLinear(in_dim=in_dim, out_dim=hidden_dim_1st, bias = True, bias_init = bias_init, lr_mul = lr_mul, activation= hidden_layer_activation)
        self.FC2 = EqualLinear(in_dim= hidden_dim_1st, out_dim= hidden_dim_2nd, bias = True, bias_init = bias_init, lr_mul = lr_mul, activation= hidden_layer_activation)
        self.FC3 = EqualLinear(in_dim=hidden_dim_2nd, out_dim= out_dim, bias = True, bias_init = bias_init, lr_mul = lr_mul, activation= final_activation )


    def forward(self, input):
        """
        input is expected to be a tensor of shape (batch, in_dim) , in_dim is expected to be 32
        
        
        """
        # print('forward pass of 3FC model.')
        out = self.FC1(input)
        out = self.FC2(out)
        out = self.FC3(out)
        return out




class ModulatedConv2d(nn.Module):
    def __init__(
        self,
        in_channel, # number of input channels  
        out_channel, # number of output channels
        kernel_size, # size of each filter. In stylegan, this is set to 3 in style conv, and 1 in ToRGB layer
        style_dim, # dimension of the style, (?): expected 
        demodulate=True,
        upsample=False,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
        fused=True,
    ):
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample
        self.downsample = downsample

        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1

            self.blur = Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor)

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            self.blur = Blur(blur_kernel, pad=(pad0, pad1))

        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in) # the scale is to adjust for the expected variance of output
        self.padding = kernel_size // 2 # padding to preserve spatial size

        # initialize the weights for all filters:
        self.weight = nn.Parameter(
            torch.randn(1, out_channel, in_channel, kernel_size, kernel_size)
        )

        # define a FC layer to map from style to each input channel:
        self.modulation = EqualLinear(style_dim, in_channel, bias_init=1) #lr_mul is set to default (1), no activation, bias_init changed to 1 (from default zero)

        self.demodulate = demodulate
        print('New stylemodulatedcov layer created, demulate is ', demodulate)
        self.fused = fused

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, "
            f"upsample={self.upsample}, downsample={self.downsample})"
        )

    def forward(self, input, style):
        batch, in_channel, height, width = input.shape
        # print('forward pass of ModulatedConv2d.')
        # print('batch size of input:', batch)
        # print('batch size of style:', style.shape[0])

        if not self.fused:  # not the default
            # first, scale the weights and 
            weight = self.scale * self.weight.squeeze(0)
            style = self.modulation(style) # output shape: (batch, in_channel)

            if self.demodulate:
                w = weight.unsqueeze(0) * style.view(batch, 1, in_channel, 1, 1)
                dcoefs = (w.square().sum((2, 3, 4)) + 1e-8).rsqrt()

            input = input * style.reshape(batch, in_channel, 1, 1)

            if self.upsample:
                weight = weight.transpose(0, 1)
                out = conv2d_gradfix.conv_transpose2d(
                    input, weight, padding=0, stride=2
                )
                out = self.blur(out)

            elif self.downsample:
                input = self.blur(input)
                out = conv2d_gradfix.conv2d(input, weight, padding=0, stride=2)
 

            return out


        # when fused = True (the default): 

        # transform the input style, of size (batch, style_dim),
        # to (batch, in_channel) via the FC layer (self.modulation)
        # finally, get a view of it with a different shape
        # print('shape of style before modulation:', style.shape)
        # print('shape of style after modulation:', self.modulation(style).shape)
        style = self.modulation(style).view(batch, 1, in_channel, 1, 1) # shape ( batch, 1, in_channel, 1, 1)
        # adjust the weight. each channel is adjusted by
        # 1. self.scale, which tries to preserve variance of the output
        # 2. style, which multiples filers at each channel by a scalar. 
        weight = self.scale * self.weight * style

        if self.demodulate: # default is True
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)

        weight = weight.view(
            batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
        )

        if self.upsample:
            input = input.view(1, batch * in_channel, height, width)
            weight = weight.view(
                batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size
            )
            weight = weight.transpose(1, 2).reshape(
                batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size
            )
            out = conv2d_gradfix.conv_transpose2d(
                input, weight, padding=0, stride=2, groups=batch
            )
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
            out = self.blur(out)

        elif self.downsample:
            input = self.blur(input)
            _, _, height, width = input.shape
            input = input.view(1, batch * in_channel, height, width)
            out = conv2d_gradfix.conv2d(
                input, weight, padding=0, stride=2, groups=batch
            )
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        else:
            input = input.view(1, batch * in_channel, height, width)


            # Conv2d. Note that 
            # weight has shape (batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size)
            
            out = conv2d_gradfix.conv2d(
                input, weight, padding=self.padding, groups=batch
            )
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        return out


class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1))


    def forward(self, image, noise=None):
        """"
        (added by Alvin)
        input:  image, shape (batch, n_channels, height, width)
        _____________________________________________________
        ______________________________________________________
        For each single image (corresponding one image in the batch), 
        same noise is apply to each channel.

        But for different images (in the same batch), different noise is used.        
        """
        if noise is None:
            batch, _, height, width = image.shape
            noise = image.new_empty(batch, 1, height, width).normal_()

        return image + self.weight * noise


class ConstantInput(nn.Module):
    def __init__(self, channel, size=4):
        super().__init__()

        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, input):
        batch = input.shape[0]
        out = self.input.repeat(batch, 1, 1, 1)

        return out


class StyledConv(nn.Module):
    """
    added by Alvin:
    This represents the typical conv layers in the stylegan2

    
    """



    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        upsample=False,
        blur_kernel=[1, 3, 3, 1],
        demodulate=True,
    ):
        super().__init__()

        self.conv = ModulatedConv2d(
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=upsample,
            blur_kernel=blur_kernel,
            demodulate=demodulate,
        )

        self.noise = NoiseInjection()
        # self.bias = nn.Parameter(torch.zeros(1, out_channel, 1, 1))
        # self.activate = ScaledLeakyReLU(0.2)
        self.activate = FusedLeakyReLU(out_channel)

    def forward(self, input, style, noise=None):
        out = self.conv(input, style)
        out = self.noise(out, noise=noise)
        # out = out + self.bias
        out = self.activate(out)

        return out


class ToRGB(nn.Module):
    def __init__(self, in_channel, style_dim, upsample=True, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        if upsample:
            self.upsample = Upsample(blur_kernel)

        self.conv = ModulatedConv2d(in_channel, 3, 1, style_dim, demodulate=False)# , activation_for_style= None)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, input, style, skip=None):
        out = self.conv(input, style)
        out = out + self.bias

        if skip is not None:
            skip = self.upsample(skip)

            out = out + skip

        return out

        
    


class MyNewGenerator(nn.Module):
    def __init__(
        self,
        size,
        # note that the style_dim below is not the dimension of the latent space.
        w_style_dim, # dimension of each layer of the w space. This should be set to 32 in this new model
        n_neurons_middle_layer, # number of neurons in the middle layer in the FC layers fom w to the correction of the next affine map
        n_neurons_for_middle_layer_hypernet,# number of neurons in the middle layer in the FC hypernet layers fom w to the correction of the next affine map
        scale_of_offset = 0.01,
        channel_multiplier=2,
        blur_kernel=[1, 3, 3, 1],
        lr_mlp=0.01,
        mapping_network_hidden_layer_activation = "fused_lrelu",
        mapping_network_final_layer_activation = "fused_lrelu"
    ):
        super().__init__()
        self.size = size

        # the dimension of 'style' at each style layer.
        # this is a global hyperparameter for the model
        # i.e. each layer has the same style dimension
        # they will be mapped to a suitable dimension by an affine map:
        self.w_style_dim = w_style_dim 



        # # Setup the mapping network from z to w:
        # layers = [PixelNorm()]

        # for i in range(n_mlp):
        #     layers.append(
        #         EqualLinear(
        #             style_dim, style_dim, lr_mul=lr_mlp, activation="fused_lrelu"
        #         )
        #     )

        # # this is the FC layers(with n_mlp =8 in the paper)
        # # used to map z space into w space: 
        # self.style = nn.Sequential(*layers)

        self.channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }
 
        self.input = ConstantInput(self.channels[4])  # initialize a constant input with dim 512 x 4 x4 
        self.conv1 = StyledConv(
            self.channels[4], self.channels[4], 3, w_style_dim, blur_kernel=blur_kernel
        ) # style_dim 32, dimension of w, into S space, self.channels[4] = 512
        self.to_rgb1 = ToRGB(self.channels[4], w_style_dim, upsample=False)

        self.log_size = int(math.log(size, 2))
        self.num_layers = (self.log_size - 2) * 2 + 1

        # (added by Alvin) initialize a ModuleList, with submodules to be added below:
        # this is the most crucial layers, to be used below.
        print('setting up FC_from_w_to_correction_for_next')
        self.FC_from_w_to_correction_for_next = nn.ModuleList()

        # initialize a ModuleList, with submodules to be added below:
        # this contains all the conv layers (with StyleConv) with no upsampling:
        self.convs = nn.ModuleList()

        # initlize a Module list, with submodules to be added below:
        # this contains alll the conv layers with upsampling:
        self.upsamples = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.noises = nn.Module()

        in_channel = self.channels[4]

        # for the noises:
        for layer_idx in range(self.num_layers):
            res = (layer_idx + 5) // 2
            shape = [1, 1, 2 ** res, 2 ** res]
            self.noises.register_buffer(f"noise_{layer_idx}", torch.randn(*shape))


        # extra added layer:
        # this is the crucial, additional FC layers from w to the correction to the next affine map.
        # first add the first FC layer, which we defined to have more neurons.
        self.FC_from_w_to_correction_for_next.append(
                ThreeLayerFC(in_dim =w_style_dim , out_dim = w_style_dim, hidden_dim_1st = n_neurons_middle_layer, hidden_dim_2nd = n_neurons_middle_layer, bias_init = 0, hidden_layer_activation = mapping_network_hidden_layer_activation, final_activation = mapping_network_final_layer_activation))
                #(in_dim= style_dim, out_dim= style_dim, middle_dim=  2*n_neurons_middle_layer, bias_init=0, activation = "fused_lrelu")
            #)

        # the next 9 layers use hypernetwork:
        for i in range(0, 8):
            self.FC_from_w_to_correction_for_next.append(
                ThreeLayer_FC_with_2FC_Hypernetwork(in_dim_prev= w_style_dim, in_dim_current=w_style_dim, out_dim= w_style_dim,
                first_hidden_dim_main= n_neurons_middle_layer,
                second_hidden_dim_main= n_neurons_middle_layer,
                hidden_dim_hyper= n_neurons_for_middle_layer_hypernet,
                lr_mul_main= 1,
                lr_mul_offset= 1,
                scale_of_offset= scale_of_offset,
                bias_init=0, hidden_layer_activation = mapping_network_hidden_layer_activation, final_activation = mapping_network_final_layer_activation)
            )

        # the remaining layers use 2-layer FC to implement, to save computations:
        for i in range(8, 12 ):
            self.FC_from_w_to_correction_for_next.append(
                ThreeLayer_FC_with_2FC_Hypernetwork(in_dim_prev= w_style_dim, in_dim_current=w_style_dim, out_dim= w_style_dim,
                first_hidden_dim_main= int(n_neurons_middle_layer/4),
                second_hidden_dim_main= int(n_neurons_middle_layer/4),
                hidden_dim_hyper= n_neurons_for_middle_layer_hypernet,
                lr_mul_main= 1,
                lr_mul_offset= 1,
                scale_of_offset= scale_of_offset,
                bias_init=0, hidden_layer_activation = mapping_network_hidden_layer_activation, final_activation = mapping_network_final_layer_activation)
            )        

        for i in range(12, self.num_layers):
            self.FC_from_w_to_correction_for_next.append(
                ThreeLayer_FC_with_2FC_Hypernetwork(in_dim_prev= w_style_dim, in_dim_current=w_style_dim, out_dim= w_style_dim,
                first_hidden_dim_main= int(n_neurons_middle_layer/16),
                second_hidden_dim_main= int(n_neurons_middle_layer/16),
                hidden_dim_hyper= n_neurons_for_middle_layer_hypernet,
                lr_mul_main= 1,
                lr_mul_offset= 1,
                scale_of_offset= scale_of_offset,
                bias_init=0, hidden_layer_activation = mapping_network_hidden_layer_activation, final_activation = mapping_network_final_layer_activation)
            )       



        # setting up the main, StyleConv
        for i in range(3, self.log_size + 1):
            
            #pick the number of channels from the self.channels dictionary above:
            out_channel = self.channels[2 ** i]

            # first a StyleConv layer with upsampling:
            self.convs.append(
                StyledConv(
                    in_channel,
                    out_channel,
                    3, # kernel size
                    w_style_dim,
                    upsample=True,
                    blur_kernel=blur_kernel,
                )
            )
            # then, a StyleConv layer without upsampling:
            self.convs.append(
                StyledConv(
                    out_channel, out_channel, 3, w_style_dim, blur_kernel=blur_kernel
                )
            )

            self.to_rgbs.append(ToRGB(out_channel, w_style_dim))


            in_channel = out_channel

        # calculate the number of latent vectors needed:
        self.n_latent = self.log_size * 2 - 2

    def make_noise(self):
        device = self.input.input.device

        noises = [torch.randn(1, 1, 2 ** 2, 2 ** 2, device=device)]

        for i in range(3, self.log_size + 1):
            for _ in range(2):
                noises.append(torch.randn(1, 1, 2 ** i, 2 ** i, device=device))

        return noises

    def mean_correlated_latent(self, n_latent):
        """
        added by Alvin:
        (changed:)
        to get the mean corrected latent, from the concatenated w, to a mean in S space, as a list.



        Original:
        to get the mean_latent in the w space.
        n_latent: the number of latent to be used to calculate the mean.
        
        
        
        """


        # latent_in = torch.randn(
        #     n_latent, self.style_dim, device=self.input.input.device
        # )
        # latent = self.style(latent_in).mean(0, keepdim=True)

        #return latent
        return style

    def get_correlated_latent(self, latent): # originally, 'input' is used in place of 'latent'
        """
        changed:
        from the concatenated w, output a list of correlated_latent.
        latent is assumed to be from the w space (the first latent space.)

        input:
        latent (w): a single tensor of shape (batch, self.style_dim*self.num_layers)
        
        output:
        correlated_latent, a list of length self.num_layers, each element of shape (batch, 32)


        ___________________________
        original:

        get the latent in w spacet
        input is assumed to be from the z space.
        
        
        """

        #check that latent is in the correct shape:
        assert latent.shape[1] == self.w_style_dim*(self.num_layers+1), "latent dimension disagrees with number of layers"       
        # assert latent.requires_grad == True, 'latent do not require'
        # next, we turn latent into a list of vectors of shape(batch, 32, )
        # latent_list= []
        # for i in range(self.num_layers+1):
        #     latent_list.append(latent[:, i*self.style_dim:(i+1)*self.style_dim)])
        
        # assert len(latent_list)==self.num_layers+1, 'not enough latent vectors to match number of layers'
         # from the second latent onward, we use a 2-FC layer to transform it
        # for i in range(self.num_layers+1):
        #     converted_latent = self.FC_from_w_to_correction_for_next[i](latent_list[i])
        #     FC2_latent.append(converted_latent)

        # # print('number of layers =', self.num_layers)
        # # print('length of correlated_latent =', len(correlated_latent))
        # assert len(FC2_latent)==self.num_layers+1, 'output length mismatch'

        # # finally we need to concatenate the result with latent_list, except the first vector
        # correlated_latent = [FC2_latent[0]]

        # for i, (t1, t2) in enumerate(zip(latent_list, FC2_latent)):
        #     if i>0:
        #         t_to_add = torch.cat([t1, t2], 1)

        converted_latent_list= []
        # first convert the first w_style_dim of the style vector:
        current_correlated_latent = self.FC_from_w_to_correction_for_next[0](latent[:, :self.w_style_dim])
        converted_latent_list.append(current_correlated_latent)
        for i in range(1, self.num_layers+1):
            if i > 0:
                # print('layer:', i)
                #input_for_next_FC = torch.cat([current_correlated_latent, latent[:, i*self.w_style_dim:(i+1)*self.w_style_dim]], 1)
                
                next_correlated_latent = self.FC_from_w_to_correction_for_next[i](current_correlated_latent, latent[:, i*self.w_style_dim:(i+1)*self.w_style_dim])
                converted_latent_list.append(next_correlated_latent)
                #set current_correlated_latent:
                current_correlated_latent = next_correlated_latent

        # print(len(converted_latent_list))
        # print(self.num_layers+1)

        assert len(converted_latent_list)==self.num_layers+1, 'not enough latent vectors to match number of layers'

        correlated_latent = torch.cat(converted_latent_list, dim =1)


        return correlated_latent

        # return self.style(input)
    def forward(
        self,
        latent, # New: this is now the new w = (w1, w2, .... w(self.num_layers)) shape (batch, 32*self.num_layers)
        #styles, # to simplify notation, we use 'latent' instead
        return_correlated_latent=False,
        # inject_index=None, # New: this is not used
        truncation=1, # if other than one, truncation trick is used
        truncation_correlated_latent=None, # if truncation is used, the truncation_style to be used as starting vector
        input_is_correlated_latent=False, #If True, styles is taken to be a list from w space. Otherwise, from z space
        noise=None,
        randomize_noise=True,
    ):
        # if not input_is_latent:# default is false, meaning it will go through the following:
        #     # if 'styles' is not from w space, first map them from z space into w space 
        #     styles = [self.style(s) for s in styles]

        # first we convert the latent input (expected to be a single tensor of shape (batch, 32*self.num_layers))
        # into a correlated latent:
        correlated_latent = self.get_correlated_latent(latent)


        if noise is None:
            # if noise is not provided, generate noise:
            if randomize_noise:
                noise = [None] * self.num_layers
            else: # if randomize_noise is set to False, take the noises saved in the buffer
                    # during initialization of self.noises.
                noise = [
                    getattr(self.noises, f"noise_{i}") for i in range(self.num_layers)
                ]

        #  # apply truncation trick when truncation is not 1.
        # if truncation < 1:
        #     style_t = []

        #     for style in styles:
        #         style_t.append(
        #             truncation_latent + truncation * (style - truncation_latent)
        #         )

        #     styles = style_t

         # if a single latent w/style is given
        # if len(styles) < 2:
        #     inject_index = self.n_latent

        #     if styles[0].ndim < 3:
        #         latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)

        #     else:
        #         latent = styles[0]

        # # if more than one style is given: 
        # else:
        #     if inject_index is None:
        #         # randomly choose an integer, for crossover two styles:
        #         inject_index = random.randint(1, self.n_latent - 1)

        #     latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
        #     latent2 = styles[1].unsqueeze(1).repeat(1, self.n_latent - inject_index, 1)

        #     latent = torch.cat([latent, latent2], 1)

        # go through the self.input. This doesn't do anything but the copy the fixed input n_sample times!
        out = self.input(correlated_latent)# result has nothing to do with the content of latent!

        # print('shape of correlated_latent[:,0:self.w_style_dim]:', correlated_latent[:,0:self.w_style_dim].shape)
        # print('shape of out:', out.shape)
        out = self.conv1(out, correlated_latent[:,0:self.w_style_dim], noise=noise[0])
        skip = self.to_rgb1(out, correlated_latent[:, self.w_style_dim:2*self.w_style_dim])

        i = 1
        for conv1, conv2, noise1, noise2, to_rgb in zip(
            self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_rgbs
        ):
            out = conv1(out, correlated_latent[:, i*self.w_style_dim:(i+1)*self.w_style_dim], noise=noise1)
            out = conv2(out, correlated_latent[:, (i + 1)*self.w_style_dim:(i+2)*self.w_style_dim], noise=noise2)
            skip = to_rgb(out, correlated_latent[:, (i + 2)*self.w_style_dim:(i+3)*self.w_style_dim], skip)

            i += 2

        image = skip

        if return_correlated_latent:
            return image, correlated_latent

        else:
            return image, None

# used only in the discriminator network:
class ConvLayer(nn.Sequential):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
        bias=True,
        activate=True,
    ):
        layers = []

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            layers.append(Blur(blur_kernel, pad=(pad0, pad1)))

            stride = 2
            self.padding = 0

        else:
            stride = 1
            self.padding = kernel_size // 2

        layers.append(
            EqualConv2d(
                in_channel,
                out_channel,
                kernel_size,
                padding=self.padding,
                stride=stride,
                bias=bias and not activate,
            )
        )

        if activate:
            layers.append(FusedLeakyReLU(out_channel, bias=bias))

        super().__init__(*layers)

# used only in the Discriminator network
class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        self.conv1 = ConvLayer(in_channel, in_channel, 3)
        self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=True)

        self.skip = ConvLayer(
            in_channel, out_channel, 1, downsample=True, activate=False, bias=False
        )

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)

        skip = self.skip(input)
        out = (out + skip) / math.sqrt(2)

        return out


class Discriminator(nn.Module):
    def __init__(self, size, channel_multiplier=2, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }
        convs = [ConvLayer(3, channels[size], 1)]

        log_size = int(math.log(size, 2))

        in_channel = channels[size]

        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]

            convs.append(ResBlock(in_channel, out_channel, blur_kernel))

            in_channel = out_channel

        self.convs = nn.Sequential(*convs)

        self.stddev_group = 4
        self.stddev_feat = 1

        self.final_conv = ConvLayer(in_channel + 1, channels[4], 3)
        self.final_linear = nn.Sequential(
            EqualLinear(channels[4] * 4 * 4, channels[4], activation="fused_lrelu"),
            EqualLinear(channels[4], 1),
        )

    def forward(self, input):
        out = self.convs(input)

        batch, channel, height, width = out.shape
        group = min(batch, self.stddev_group)
        stddev = out.view(
            group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
        )
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], 1)

        out = self.final_conv(out)

        out = out.view(batch, -1)
        out = self.final_linear(out)

        return out

