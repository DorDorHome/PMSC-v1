# Load a latent (found either by backpropagation or by a separate trained encoder)
# then display the distribution of its components 

# for loading and manipulating the latent:
import numpy as np
import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
import torch.distributed as dist
from torchvision import transforms, utils



# for plotting:
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import norm

import argparse

# helper function for import and loading the model:

from load_model import load_g_ema



# load the latent:


if __name__ == "__main__":

    parser= argparse.ArgumentParser(description = 'examine the distribution of a latent across all its components.')
    
    parser.add_argument('--device',
                        type=str,
                        default= 'cuda:0')

    parser.add_argument('--arch',
                         type= str,
                         default = None)
    
    parser.add_argument("--size", type=int, default=256, help="image sizes for the model"
    )

    parser.add_argument(
    "--channel_multiplier",
    type=int,
    default=2,
    help="channel multiplier of the generator. config-f = 2, else = 1",
    )

    parser.add_argument("--ckpt",
    type=str,
    default=None,
    help="path to the checkpoints of the model",
    )

    parser.add_argument('--path_to_latent',
                        type=str,
                        default=None)

    
    args = parser.parse_args()

    
    
    # load the latent
    latent = torch.load(args.path_to_latent)

    # load the model:

    g_ema = load_g_ema(args)

    # load the checkpoint
    print('loading checkpoint from: ', args.ckpt)
    checkpoint = torch.load(args.ckpt, map_location=args.device)

    # g_ema.load_state_dict(checkpoint["g_ema"], strict = False)

    # if args.truncation < 1:
    #     with torch.no_grad():
    #         mean_latent = g_ema.mean_latent(args.truncation_mean)
    # else:
    #     mean_latent = None

    # examine the latent:
    sns.kdeplot(data = latent[0].detach().clone().to('cpu'))
    x_axis = np.arange(-13, 13, 0.01)
    plt.plot(x_axis, norm.pdf(x_axis, 0, 1))
    plt.savefig(args.path_to_latent[:-5]+'_distribution.png')







    

    
