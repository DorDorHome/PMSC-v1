# changed MyNewGeneraor to allow for path regularization


# adapted from Alvin_model_experiment_changed_mapping_network_multi_stage_Gaussian_with_double_activations.py
# 
#  #
#____________________________________
# with double activations inside Gaussian stage.
# this model was adapted from Alvin_model_experiment_changed_mapping_network_Multi_Gaussian_with_reduced_RGB_style_dim.py,
# with only activations function added in the mean and covariance function.
#
#  This model was adapted from Alvin_model_experiment_changed_mapping_network_2FC_with_reduced_RGB_style_dim
#
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
    def __init__(self, in_dim, out_dim, middle_dim, bias_init = 0, lr_mul = 1, activation = None):
        super().__init__()
        self.FC1 = EqualLinear(in_dim=in_dim, out_dim=middle_dim, bias = True, bias_init = bias_init, lr_mul = lr_mul, activation= activation)
        self.FC2 = EqualLinear(in_dim=middle_dim, out_dim=out_dim, bias = True, bias_init = bias_init, lr_mul = lr_mul, activation= activation)

    def forward(self, input):
        """
        input is expected to be a tensor of shape (batch, in_dim) , in_dim is expected to be 32
        
        
        """
        out = self.FC1(input)
        out = self.FC2(out)
        return out

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
        out = self.FC1(input)
        out = self.FC2(out)
        out = self.FC3(out)
        return out    

class GRU_map_with_prev_style(nn.Module):
    """
    
    Created by Alvin. This is a network modelled by GRU that, given a 'prev_style_hidden' from the previous layer( h1) and current input (w2)
    It creates:
    1. hidden state h2 called 'current_style_hidden', based on current w style input w2, and on prev_style_hidden h1
    2. an output style (s2) 

    when w2 is sampled from Gaussian N(0, I), this gives a vector sample from:
    s2 ~ N(\mu(s1), \Lambda(s1)^T \Lambda(s1))
    """

    def __init__(self, prev_style_hidden_dim, current_w_in_dim, current_out_style_dim, bias= True, final_activation = None ):
        super().__init__()
    # the function that map s1 to a mean:
        self.prev_style_hidden_dim = prev_style_hidden_dim  # dimension of hidden in the GRU cell. In particular, h1
        self.current_out_style_dim = current_out_style_dim # dimension of output style vector, s2
        self.current_w_in_dim = current_w_in_dim # dimension of w style input, w2
        self.activation = final_activation # activation function, Expected to be 'None'
        self.GRU = nn.GRUCell(input_size = current_w_in_dim, hidden_size = prev_style_hidden_dim, bias = bias )
        self.affine = EqualLinear(in_dim = prev_style_hidden_dim, out_dim = current_out_style_dim, activation = final_activation )    # this represents the final stage of the map, using h2 to calculates s2

        # the function that map s1 to a matrix of shape (current_out_style_dim, current_in_dim ) (but flattened)
        # self.transform_matrix = EqualLinear(in_dim = prev_transformed_style_dim, out_dim = current_out_style_dim*current_w_in_dim, activation = activation) 
        

    def forward(self, h1, w2):
        """
        h1: initial values of hidden state, prev_style_hidden 
        it should have a dimension of (batch, prev_style_hidden_dim)
        
        w2: the input 


        ___________________
        ________________
        output:

        h2: the new hidden state, also having a dimension of (batch, prev_style_hidden_dim)

        s2: a post-activated style vector, to be used directly to modulate the filters.
        
        """

        h2 = self.GRU(w2, h1)
        s2 = self.affine(h2)




        return h2, s2

class GRUModulatedConv2d(nn.Module):
    def __init__(
        self,
        in_channel, # number of input channels  
        out_channel, # number of output channels
        kernel_size, # size of each filter. In stylegan, this is set to 3 in style conv, and 1 in ToRGB layer
        prev_style_hidden_dim, # this is the dimensions of the hidden state used in the GRU cells
        w_style_dim, # dimension of the style, (?): expected 32
        demodulate=True,
        upsample=False,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
        fused=True,
        use_external_affine = False, # if set to be true, w_style_dim should be the same as in_channel
        activation_for_style = 'sqrt_softmax'
    ):
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample
        self.downsample = downsample
        self.use_external_affine = use_external_affine
        self.activation_for_style = activation_for_style

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
        #self.modulation = EqualLinear(style_dim, in_channel, bias_init=1) #lr_mul is set to default (1), no activation, bias_init changed to 1 (from default zero)

        # this is a function to replace the affine map, which takes one more input:
        # it takes two input instead of one.
        # the first one is used to changed the affine map used.
        if use_external_affine:
            assert in_channel == w_style_dim, 'if use_external_affine is used, w_style_dim should be the same as in_channel'
            self.modulation = None
        else:
            self.modulation = GRU_map_with_prev_style(prev_style_hidden_dim= prev_style_hidden_dim, current_w_in_dim = w_style_dim,current_out_style_dim = in_channel, final_activation= activation_for_style)
            
# (prev_transformed_style_dim = prev_transformed_style_dim, current_w_in_dim= w_style_dim, current_out_style_dim = in_channel, activation = activation_pre_affine)

        self.demodulate = demodulate
        print('New stylemodulatedcov layer created, demulate is ', demodulate)
        self.fused = fused

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, "
            f"upsample={self.upsample}, downsample={self.downsample})"
        )

    def forward(self,h1 , input, current_w_style):
        batch, in_channel, height, width = input.shape

        if not self.fused:  # not the default
            # first, scale the weights and 
            weight = self.scale * self.weight.squeeze(0)
            if self.use_external_affine:
                style = current_w_style
                if self.activation_for_style=="sqrt_softmax":
                    style = F.softmax(style, dim = 2)
                    style = torch.sqrt(style)
            else:
                h2, style = self.modulation(h1,current_w_style) # output shape:(batch, prev_style_hidden_dim)(batch, in_channel)

            if self.demodulate:
                w = weight.unsqueeze(0) * style.view(batch, 1, in_channel, 1, 1)
                new_dcoefs = (w.square().sum((3, 4)) + 1e-8).rsqrt()

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

            return h2, out, style.view(batch, self.in_channel)
        # when fused = True (the default): 

        # transform the input style, of size (batch, style_dim),
        # to (batch, in_channel) via the FC layer (self.modulation)
        # finally, get a view of it with a different shape
        if self.use_external_affine:
            style = current_w_style.view(batch, 1, in_channel, 1, 1)
            h2 = None
            if self.activation_for_style=="sqrt_softmax":
                style = F.softmax(style, dim = 2)
                style = torch.sqrt(style)
        else:# default usepre_style
            h2, style = self.modulation(h1, current_w_style) # shape ( batch, 1, in_channel, 1, 1)
            style = style.view(batch, 1, in_channel, 1, 1)
        # adjust the weight. each channel is adjusted by      
        
        if self.demodulate:
            # adjust the weight, each channel by requiring the weight at each layer is normalized to one:
            weight_normalizer = torch.rsqrt(self.weight.pow(2).sum([3,4]) + 1e-8) # 
            weight = self.weight*weight_normalizer.view(1, self.out_channel, in_channel,1,1) # result is of shape (batch, out_channel, in_channel, kernel, kernel)
        else:
            weight = self.weight

        # finally, adjust the weight (across the in_channel) by multiplying with style
        if self.activation_for_style:
            weight = weight*style # adjust the weight with the style !, shape(batch, out_channel, in_channel, kernel, kernel)
        else:
            weight = self.scale*weight*style


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
        
        current_transformed_style = style.view(batch, self.in_channel)

        return h2, out, current_transformed_style 


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
        prev_style_hidden_dim,
        w_style_dim,
        upsample=False,
        blur_kernel=[1, 3, 3, 1],
        demodulate=True,
        use_external_affine = False,
        activation_for_style = 'sqrt_softmax'
    ):
        super().__init__()
        print('initialized with activation: ', activation_for_style)
        self.conv = GRUModulatedConv2d(
            in_channel,
            out_channel,
            kernel_size,
            prev_style_hidden_dim, # this is the prev_transformed_style dimension. Should be the same as the in_channel of the previous StyleConv Layer
            w_style_dim, # current w style dimension
            upsample=upsample,
            blur_kernel=blur_kernel,
            demodulate=demodulate,
            use_external_affine = use_external_affine,
            activation_for_style = activation_for_style
        )

        self.noise = NoiseInjection()
        # self.bias = nn.Parameter(torch.zeros(1, out_channel, 1, 1))
        # self.activate = ScaledLeakyReLU(0.2)
        self.activate = FusedLeakyReLU(out_channel)

    def forward(self, h1, input, current_w_style, noise=None):
        """
        takes three input:

        h1: initial hidden state, should be a tensor that has shape (batch, transformed_style_hidden_dim)
        input: 
        current_w_style: should be a tensor that has shape (batch, w_style_dim        
        
        """


        h2, out, current_transformed_style = self.conv(h1, input, current_w_style)
        out = self.noise(out, noise=noise)
        # out = out + self.bias
        out = self.activate(out)

        return h2, out, current_transformed_style


class ToRGB(nn.Module):
    def __init__(self, in_channel, prev_style_hidden_dim , w_style_dim, upsample=True, blur_kernel=[1, 3, 3, 1], activation = None):
        super().__init__()

        if upsample:
            self.upsample = Upsample(blur_kernel)

        self.conv = GRUModulatedConv2d(in_channel, 3, 1, prev_style_hidden_dim, w_style_dim,demodulate=False, activation_for_style = activation)# , activation_for_style= None)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))
        print(f'ToRGB initialized, with in_channel: {in_channel}, prev_transformed_style_dim {prev_style_hidden_dim}, w_style_dim {w_style_dim}, ')

    def forward(self, h1, input, current_w_style, skip=None):
        #print(f'\n inside ToRGB forward, input shape {input.shape}, prev_transformed_style shape {prev_transformed_style.shape}, current_w_style shape {current_w_style.shape}')
        h2, out, current_transformed_style = self.conv(h1,input, current_w_style)
        out = out + self.bias

        if skip is not None:
            skip = self.upsample(skip)

            out = out + skip

        return h2, out, current_transformed_style





class MyNewGenerator(nn.Module):
    def __init__(
        self,
        size,
        # note that the style_dim below is not the dimension of the latent space.
        prev_style_hidden_dim,
        w_style_dim, # dimension of each layer of the w space. This should be set to 32 in this new model
        n_neurons_middle_layer, # number of neurons in the middle layer in the FC layers fom w to the correction of the next affine map
        channel_multiplier=2,
        blur_kernel=[1, 3, 3, 1],
        lr_mlp=0.01,
        activation_for_style = 'sqrt_softmax'
    ):
        super().__init__()
        self.size = size

        # the dimension of 'style' at each style layer.
        # this is a global hyperparameter for the model
        # i.e. each layer has the same style dimension
        # they will be mapped to a suitable dimension by an affine map:
        self.w_style_dim = w_style_dim
        self.prev_style_hidden_dim = prev_style_hidden_dim



        # # Setup the mapping network from z to w:
        # layers = [PixelNorm()]

        # for i in range(n_mlp):
        #     layers.append(
        #         EqualLinear(
        #             style_dim, style_dim, lr_mul=lr_mlp, activation="fused_lrelu"
        #         )out_channel
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
            self.channels[4], self.channels[4], 
            3, # kernel_size 
            prev_style_hidden_dim, # prev_style_hidden_dim
            self.channels[4], #for this first layer, this needs to be the same as in_channel, self.channels[4] = 512
            blur_kernel=blur_kernel,  # need to change!!
            use_external_affine = True,
            activation_for_style = activation_for_style # check! 
        ) 
        self.to_rgb1 = ToRGB(self.channels[4], # in_channel_dimensions.
         prev_style_hidden_dim, # prev_transformed_style_dim 
         w_style_dim, upsample=False)

        self.log_size = int(math.log(size, 2))
        self.num_layers = (self.log_size - 2) * 2 + 1

        # (added by Alvin) initialize a ModuleList, with submodules to be added below:
        # only the first 32 dimension of w will go through FC layers:
        self.correct_for_first_w =  ThreeLayerFC(in_dim= w_style_dim, out_dim= self.channels[4], 
        hidden_dim_1st=  n_neurons_middle_layer, hidden_dim_2nd = n_neurons_middle_layer, bias_init=0, 
        hidden_layer_activation = "fused_lrelu",
        final_activation = None )

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




        # setting up the main, StyleConv
        for i in range(3, self.log_size + 1):
            
            #pick the number of channels from the self.channels dictionary above:
            out_channel = self.channels[2 ** i]

            if i == 3:
                prev_in_channel = self.channels[4]

            # first a StyleConv layer with upsampling:
            self.convs.append(
                StyledConv(
                    in_channel,
                    out_channel,
                    3, # kernel size
                    prev_style_hidden_dim, 
                   w_style_dim,
                    upsample=True,
                    blur_kernel=blur_kernel,
                    activation_for_style= activation_for_style
                )
            )
            # then, a StyleConv layer without upsampling:
            self.convs.append(
                StyledConv(
                    out_channel, out_channel, 3, 
                    prev_style_hidden_dim, # prev_transformed_style_dim 
                    w_style_dim, blur_kernel=blur_kernel,
                    activation_for_style = activation_for_style
                )
            )

            self.to_rgbs.append(ToRGB(out_channel, prev_style_hidden_dim, w_style_dim))


            in_channel = out_channel
            prev_in_channel = in_channel
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

    def get_transformed_latent(self, latent): # originally, 'input' is used in place of 'latent'
        """
        changed:
        from the concatenated w, output a transformed_latent.
        latent is assumed to be from the w space (the first latent space.)

        input:
        latent (w): a single tensor of shape (batch, self.style_dim*self.num_layers)
        
        output:
        transformed_latent,
        for each batch, only the first 32 elements was transformed

        ___________________________
        original:

        get the latent in w spacet
        input is assumed to be from the z space.
        
        
        """

        #check that latent is in the correct shape:
        assert latent.shape[1] == self.w_style_dim*(self.num_layers+1), "latent dimension disagrees with number of layers"       
        assert latent.requires_grad == True, 'latent do not require'

        transformed_latent = torch.cat((self.correct_for_first_w(latent[:, :self.w_style_dim]), latent[:, self.w_style_dim:]), dim=1)
       
        return transformed_latent



        # return self.style(input)

    def forward(
        self,
        latent, # New: this is now the new w = (w1, w2, .... w(self.num_layers)) shape (batch, 32*self.num_layers)
        #styles, # to simplify notation, we use 'latent' instead
        return_transformed_latent=False,
        return_style_space_list = False,
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
        # transformed_styleent having shape (batch ,32)
        
        #print('starting forward pass of generator \n')

       #print('shape of input latent:', latent.shape)
        transformed_latent = self.get_transformed_latent(latent)


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
        #             truncation_latent + truncation * (stransformed_style
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
        #print('shape of latent[0]:', latent.shape)
        out = self.input(transformed_latent)# result has nothing to do with the content of latent!
     
        zero_h_style = torch.zeros([transformed_latent.shape[0], self.channels[4]]).to('cuda:1') # need to edit! 
        # print('shape of zero_style:', zero_h_style.shape )
        # print('shape of first inter_transformed_latent:', inter_transformed_latent[0].shape)
        # print('shape of second inter_transformed_latent:', inter_transformed_latent[1].shape)
        # print('shape of out after constant layer', out.shape)
        # print('\n computing first conv: \n')
        # print('dimensions of first input latent in first conv: ', inter_transformed_latent[0].shape)
        h1, out, transformed_style_0 = self.conv1(zero_h_style, out, transformed_latent[:, :self.channels[4]], noise=noise[0])
        # print('type of the first h1:', type(h1))
        # print(f'\n first conv computed! Output shape is {out.shape}, transformed_style_shape is {transformed_style_0.shape}')
        # print('\n computing first rgb layer \n')
        h_skip_1, skip, transformed_style_skip1 = self.to_rgb1(h1, out, transformed_latent[:, self.channels[4]:(self.channels[4]+self.w_style_dim)])


        variance_of_hs= []

        prev_transformed_style = transformed_style_0
        transformed_style_space_list = [transformed_latent[:, :self.channels[4]], prev_transformed_style]
        i = 1
        for conv1, conv2, noise1, noise2, to_rgb in zip(
            self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_rgbs
        ):
            # print(f'\n computing {i+1}-th conv layer: \n')
            h2, out, transformed_style2 = conv1(h1, out, transformed_latent[:, self.channels[4] + (i-1)*self.w_style_dim:self.channels[4] + i*self.w_style_dim], noise=noise1)
            # print(f'\n computing {i}-th conv layer: \n')
            h3, out, transformed_style3 = conv2(h2, out, transformed_latent[:, self.channels[4] + i*self.w_style_dim:self.channels[4] + (i+1)*self.w_style_dim], noise=noise2)
            # print(f'\n computing {i}-th rgb layer \n ')

            h_skip, skip, transformed_style_skip = to_rgb(h_skip_1, out, transformed_latent[:,  self.channels[4] + (i+1)*self.w_style_dim:self.channels[4] + (i+2)*self.w_style_dim], skip)

            # add the transformed style into the style_space_list:
            transformed_style_space_list.append(transformed_style2)
            transformed_style_space_list.append(transformed_style3)

            if h1 is None:
                print('skipping None')
            else:
                print('h1 is not None')
                variance_of_hs.append(torch.var(h1, dim =1, unbiased = False, keepdim=False).mean())
                
            variance_of_hs.append(torch.var(h2, dim =1, unbiased = False, keepdim=False).mean())
            variance_of_hs.append(torch.var(h3, dim =1, unbiased = False, keepdim=False).mean())

            # set the style for next iteration:
            h1 = h3 # save current hidden state output, as input to the next conv layer
            h_skip_1 = h_skip # save current hidden state, as input to the next to_rgb 
            i += 2

        image = skip

        print('variances of hs on all layers', variance_of_hs)
        if return_style_space_list:

            return image, style_space_list
        
        if return_transformed_latent:
            return image, transformed_latent

        else:

            return image, None

# used only in the Discriminator network

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
