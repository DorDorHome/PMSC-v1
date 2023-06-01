# this is updated from Alvin_model_experiment, with the following changes made:


# 4. a bias term is added back to the affine map for style space 
# 5. when applying StyleEqualLinear in NewModulatedConv2d, bias is init to one instead of zero
# 6. 


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



class StyleEqualLinear(nn.Module):
    """
    this is adopted from EqualLinear class
    The difference is that the output of StyleEqualLinear is a sqrt of softmax output
    
    in_dim: the input style dimension. In the paper, it was set to be 512
    out_dim: output of the styleequallinear, but the input dimension of the corresponding StyleLayer
             This changes between layers.

    """
    def __init__(
        self, in_dim, out_dim, bias=True, style_mul_factor = 1, bias_init=0, lr_mul=1, activation= None
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul)) #weights are scaled by the inverse of lr_mul.

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        else:
            self.bias = None

        # initialize the activation function. Default is None
        self.activation = activation
        self.scale = (1 / math.sqrt(in_dim)) * lr_mul # shape (out_dim)
        self.lr_mul = lr_mul
        self.style_mul_factor = style_mul_factor

    def forward(self, input):
        """
        input: a style tensor with shape: (batch, style_dim), where style_dime is expected to be 512.
        
        
        """
        if self.activation == 'sqrt_softmax': 
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

class EqualLinear(nn.Module):
    def __init__(
        self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        else:
            self.bias = None

        self.activation = activation

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)

        else:
                        out = F.linear(
                input, self.weight * self.scale, bias=self.bias * self.lr_mul
            )

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})"
        )

class NewModulatedConv2d(nn.Module):
    def __init__(
        self,
        in_channel, # number of input channels  
        out_channel, # number of output channels
        kernel_size, # size of each filter. In stylegan, this is set to 3 in style conv, and 1 in ToRGB layer
        style_dim, # dimension of the style, (?): expected 512
        demodulate=True,
        upsample=False,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
        fused=True,
        activation_for_style = 'sqrt_softmax'
    ):
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample
        self.downsample = downsample
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
        # This is changed from the original file:
        self.modulation = StyleEqualLinear(style_dim, in_channel, style_mul_factor=1, bias_init=0, activation= activation_for_style
    )

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

        if not self.fused:  # not the default
            # first, scale the weights and 
            weight = self.scale * self.weight.squeeze(0)
            style = self.modulation(style) # output shape: (batch, in_channel)

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
 

            return out


        # when fused = True (the default): 

        # transform the input style, of size (batch, style_dim),
        # to (batch, in_channel) via the FC layer (self.modulation)
        # finally, get a view of it with a different shape
        style = self.modulation(style).view(batch, 1, in_channel, 1, 1) #activation might be applied.
        
        
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


        # 1. self.scale, which tries to preserve variance of the output
        # 2. style, which multiples filers at each channel by a scalar. 
        # the following do not work, as it will normalize all in/out pairs, thus cancalling the effect of the style! 
  
        # weight = self.scale * self.weight * style
        # if self.demodulate: # default is True
        #     new_demod = torch.rsqrt(weight.pow(2).sum([3,4])  + 1e-8)

        #     # the following do not work, as it will normalize all in/out pairs, thus cancalling the effect of the style! 
        #     weight = weight * new_demod.view(batch, self.out_channel, self.in_channel, 1, 1)*(1/math.sqrt(in_channel))



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


class NewStyledConv(nn.Module):
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

        self.conv = NewModulatedConv2d(
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

        self.conv = NewModulatedConv2d(in_channel, 3, 1, style_dim, demodulate=False, activation_for_style= None)
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
        style_dim, # dimension of the w space.
        n_mlp, # number of layers in the FC layers, from z to w
        channel_multiplier=2,
        blur_kernel=[1, 3, 3, 1],
        lr_mlp=0.01,
    ):
        super().__init__()
        self.size = size

        # the dimension of 'style' at each style layer.
        # this is a global hyperparameter for the model
        # i.e. each layer has the same style dimension
        # they will be mapped to a suitable dimension by an affine map:
        self.style_dim = style_dim 



        # Setup the mapping network from z to w:
        layers = [PixelNorm()]

        for i in range(n_mlp):
            layers.append(
                EqualLinear(
                    style_dim, style_dim, lr_mul=lr_mlp, activation="fused_lrelu"
                )
            )

        # this is the FC layers(with n_mlp =8 in the paper)
        # used to map z space into w space: 
        self.style = nn.Sequential(*layers)

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
        self.conv1 = NewStyledConv(
            self.channels[4], self.channels[4], 3, style_dim, blur_kernel=blur_kernel
        )
        self.to_rgb1 = ToRGB(self.channels[4], style_dim, upsample=False)

        self.log_size = int(math.log(size, 2))
        self.num_layers = (self.log_size - 2) * 2 + 1

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

            # first a StyleConv layer with upsampling:
            self.convs.append(
                NewStyledConv(
                    in_channel,
                    out_channel,
                    3, # kernel size
                    style_dim,
                    upsample=True,
                    blur_kernel=blur_kernel,
                )
            )
            # then, a StyleConv layer without upsampling:
            self.convs.append(
                NewStyledConv(
                    out_channel, out_channel, 3, style_dim, blur_kernel=blur_kernel
                )
            )

            self.to_rgbs.append(ToRGB(out_channel, style_dim))

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

    def mean_latent(self, n_latent):
        """
        added by Alvin:

        to get the mean_latent in the w space.
        n_latent: the number of latent to be used to calculate the mean.
        
        
        
        """


        latent_in = torch.randn(
            n_latent, self.style_dim, device=self.input.input.device
        )
        latent = self.style(latent_in).mean(0, keepdim=True)

        return latent

    def get_latent(self, input):
        # get the latent in w space
        # input is assumed to be from the z space.
        return self.style(input)

    def forward(
        self,
        styles,
        return_latents=False,
        inject_index=None,
        truncation=1, # if other than one, truncation trick is used
        truncation_latent=None, # if truncation is used, the truncation_latent to be used as starting vector
        input_is_latent=False, #If True, styles is taken to be a list from w space. Otherwise, from z space
        noise=None,
        randomize_noise=True,
    ):
        if not input_is_latent:
            # if 'styles' is not from w space, first map them from z space into w space 
            styles = [self.style(s) for s in styles]

        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers
            else: # if randomize_noise is set to False, take the noises saved in the buffer
                    # during initialization of self.noises.
                noise = [
                    getattr(self.noises, f"noise_{i}") for i in range(self.num_layers)
                ]

         # apply truncation trick when truncation is not 1.
        if truncation < 1:
            style_t = []

            for style in styles:
                style_t.append(
                    truncation_latent + truncation * (style - truncation_latent)
                )

            styles = style_t

         # if a single latent w/style is given
        if len(styles) < 2:
            inject_index = self.n_latent

            if styles[0].ndim < 3:
                latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)

            else:
                latent = styles[0]

        # if more than one style is given: 
        else:
            if inject_index is None:
                # randomly choose an integer, for crossover two styles:
                inject_index = random.randint(1, self.n_latent - 1)

            latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            latent2 = styles[1].unsqueeze(1).repeat(1, self.n_latent - inject_index, 1)

            latent = torch.cat([latent, latent2], 1)

        # go through the self.input. This doesn't do anything but the copy the fixed input n_sample times!
        out = self.input(latent)# result has nothing to do with the content of latent!


        out = self.conv1(out, latent[:, 0], noise=noise[0])

        skip = self.to_rgb1(out, latent[:, 1])

        i = 1
        for conv1, conv2, noise1, noise2, to_rgb in zip(
            self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_rgbs
        ):
            out = conv1(out, latent[:, i], noise=noise1)
            out = conv2(out, latent[:, i + 1], noise=noise2)
            skip = to_rgb(out, latent[:, i + 2], skip)

            i += 2

        image = skip

        if return_latents:
            return image, latent

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

