from collections import abc
import os

import torch
from torch.nn import functional as F
from torch.autograd import Function
from torch.utils.cpp_extension import load


module_path = os.path.dirname(__file__)
upfirdn2d_op = load(
    "upfirdn2d",
    sources=[
        os.path.join(module_path, "upfirdn2d.cpp"),
        os.path.join(module_path, "upfirdn2d_kernel.cu"),
    ],
)


class UpFirDn2dBackward(Function):
    @staticmethod
    def forward(
        ctx, grad_output, kernel, grad_kernel, up, down, pad, g_pad, in_size, out_size
    ):

        up_x, up_y = up
        down_x, down_y = down
        g_pad_x0, g_pad_x1, g_pad_y0, g_pad_y1 = g_pad

        grad_output = grad_output.reshape(-1, out_size[0], out_size[1], 1)

        grad_input = upfirdn2d_op.upfirdn2d(
            grad_output,
            grad_kernel,
            down_x,
            down_y,
            up_x,
            up_y,
            g_pad_x0,
            g_pad_x1,
            g_pad_y0,
            g_pad_y1,
        )
        grad_input = grad_input.view(in_size[0], in_size[1], in_size[2], in_size[3])

        ctx.save_for_backward(kernel)

        pad_x0, pad_x1, pad_y0, pad_y1 = pad

        ctx.up_x = up_x
        ctx.up_y = up_y
        ctx.down_x = down_x
        ctx.down_y = down_y
        ctx.pad_x0 = pad_x0
        ctx.pad_x1 = pad_x1
        ctx.pad_y0 = pad_y0
        ctx.pad_y1 = pad_y1
        ctx.in_size = in_size
        ctx.out_size = out_size

        return grad_input

    @staticmethod
    def backward(ctx, gradgrad_input):
        kernel, = ctx.saved_tensors

        gradgrad_input = gradgrad_input.reshape(-1, ctx.in_size[2], ctx.in_size[3], 1)

        gradgrad_out = upfirdn2d_op.upfirdn2d(
            gradgrad_input,
            kernel,
            ctx.up_x,
            ctx.up_y,
            ctx.down_x,
            ctx.down_y,
            ctx.pad_x0,
            ctx.pad_x1,
            ctx.pad_y0,
            ctx.pad_y1,
        )
        # gradgrad_out = gradgrad_out.view(ctx.in_size[0], ctx.out_size[0], ctx.out_size[1], ctx.in_size[3])
        gradgrad_out = gradgrad_out.view(
            ctx.in_size[0], ctx.in_size[1], ctx.out_size[0], ctx.out_size[1]
        )

        return gradgrad_out, None, None, None, None, None, None, None, None


class UpFirDn2d(Function):
    @staticmethod
    def forward(ctx, input, kernel, up, down, pad):
        up_x, up_y = up
        down_x, down_y = down
        pad_x0, pad_x1, pad_y0, pad_y1 = pad

        kernel_h, kernel_w = kernel.shape
        batch, channel, in_h, in_w = input.shape
        ctx.in_size = input.shape

        input = input.reshape(-1, in_h, in_w, 1)

        ctx.save_for_backward(kernel, torch.flip(kernel, [0, 1]))

        out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h + down_y) // down_y
        out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w + down_x) // down_x
        ctx.out_size = (out_h, out_w)

        ctx.up = (up_x, up_y)
        ctx.down = (down_x, down_y)
        ctx.pad = (pad_x0, pad_x1, pad_y0, pad_y1)

        g_pad_x0 = kernel_w - pad_x0 - 1
        g_pad_y0 = kernel_h - pad_y0 - 1
        g_pad_x1 = in_w * up_x - out_w * down_x + pad_x0 - up_x + 1
        g_pad_y1 = in_h * up_y - out_h * down_y + pad_y0 - up_y + 1

        ctx.g_pad = (g_pad_x0, g_pad_x1, g_pad_y0, g_pad_y1)

        out = upfirdn2d_op.upfirdn2d(
            input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1
        )
        # out = out.view(major, out_h, out_w, minor)
        out = out.view(-1, channel, out_h, out_w)

        return out

    @staticmethod
    def backward(ctx, grad_output):
        kernel, grad_kernel = ctx.saved_tensors

        grad_input = None

        if ctx.needs_input_grad[0]:
            grad_input = UpFirDn2dBackward.apply(
                grad_output,
                kernel,
                grad_kernel,
                ctx.up,
                ctx.down,
                ctx.pad,
                ctx.g_pad,
                ctx.in_size,
                ctx.out_size,
            )

        return grad_input, None, None, None, None


def upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0)):
    # added by Alvin: abc.Iterable means 'Abstract Base Classes'
    # The following checks whether 'up' is a class that are registered as Iterable 
    # or that has an __iter__() method 
    if not isinstance(up, abc.Iterable):
        up = (up, up)

    if not isinstance(down, abc.Iterable):
        down = (down, down)

    if len(pad) == 2:
        pad = (pad[0], pad[1], pad[0], pad[1])

    if input.device.type == "cpu":
        out = upfirdn2d_native(input, kernel, *up, *down, *pad)

    else:
        out = UpFirDn2d.apply(input, kernel, up, down, pad)

    return out


def upfirdn2d_native(
    input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1
):
    """Slow reference implementation of `upfirdn2d()` using standard PyTorch ops.

    added by Alvin, referenced from :
    https://github.com/NVlabs/stylegan2-ada-pytorch/blob/6f160b3d22b8b178ebe533a50d4d5e63aedba21d/torch_utils/ops/upfirdn2d.py#L120

    Pad, upsample, filter, and downsample a batch of 2D images.
    Performs the following sequence of operations for each channel:
    1. Upsample the image by inserting N-1 zeros after each pixel (`up`).
    2. Pad the image with the specified number of zeros on each side (`padding`).
       Negative padding corresponds to cropping the image.
    3. Convolve the image with the specified 2D FIR filter (`f`), shrinking it
       so that the footprint of all output pixels lies within the input image.
    4. Downsample the image by keeping every Nth pixel (`down`).
    This sequence of operations bears close resemblance to scipy.signal.upfirdn().
    The fused op is considerably more efficient than performing the same calculation
    using standard PyTorch ops. It supports gradients of arbitrary order.    
    
    
    
    
    """
    _, channel, in_h, in_w = input.shape
     # change the shape of input into (n_sample x channel, in_h, in_w, 1): 
    input = input.reshape(-1, in_h, in_w, 1) 

    _, in_h, in_w, minor = input.shape # ?? shouldn't minor be always 1?
    kernel_h, kernel_w = kernel.shape

    out = input.view(-1, in_h, 1, in_w, 1, minor) # shape (n_sample x channel, in_h, 1, in_w, 1, 1)



    # step 1: Upsample by inserting zeros.
    # pad the output
    # note that the pads are specified from the last dimension to the fist:
    # so the output has shape (n_sample x channel, in_h , up_y, in_w , up_x , 1) padded in h and w dimensions.
    out = F.pad(out, [0, 0, 0, up_x - 1, 0, 0, 0, up_y - 1])

    # view the output as a different shape,
    out = out.view(-1, in_h * up_y, in_w * up_x, minor)  
    ##// End of Step 1



    # step 2: Pad or crop
    # pad:
    out = F.pad(
        out, [0, 0, max(pad_x0, 0), max(pad_x1, 0), max(pad_y0, 0), max(pad_y1, 0)]
    )
    # Crop:
    out = out[
        :,
        max(-pad_y0, 0) : out.shape[1] - max(-pad_y1, 0),
        max(-pad_x0, 0) : out.shape[2] - max(-pad_x1, 0),
        :,
    ]
    ## // end of step 2


    # reshape the output, by putting the height and width as the last two dims
    out = out.permute(0, 3, 1, 2)
    out = out.reshape(
        [-1, 1, in_h * up_y + pad_y0 + pad_y1, in_w * up_x + pad_x0 + pad_x1]
    )

    # Setup filter.
    w = torch.flip(kernel, [0, 1]).view(1, 1, kernel_h, kernel_w) 

    # convolve with the filter:
    out = F.conv2d(out, w)
    out = out.reshape(
        -1,
        minor,
        in_h * up_y + pad_y0 + pad_y1 - kernel_h + 1,
        in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1,
    )
    out = out.permute(0, 2, 3, 1)

    # Downsample by throwing away pixels:
    out = out[:, ::down_y, ::down_x, :]

    # compute the new height and width for resizing:
    out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h + down_y) // down_y
    out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w + down_x) // down_x

    return out.view(-1, channel, out_h, out_w)
