import argparse
import math
from pyexpat import model
import random
import os
from turtle import Turtle

import numpy as np
import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
import torch.distributed as dist
from torchvision import transforms, utils
from tqdm import tqdm
torch.cuda.empty_cache()

import argparse

import torch
from torchvision import utils
from tqdm import tqdm

print('pytorch version in used is:', torch.__version__)


if __name__ == "__main__":
    device = "cuda:0"
    #device = 'cpu'
    print('empyting cache')
    torch.cuda.empty_cache()
    print('done empyting cache.')


    parser = argparse.ArgumentParser(description="Generate samples from the generator")

    # these arguments are added by Alvin Chan:
    parser.add_argument('--arch', type=str, default='stylegan2', help='model architectures (stylegan2 | swagan)')



    # the following are the original arguments
    parser.add_argument(
        "--size", type=int, default=1024, help="output image size of the generator"
    )

    # parser.add_argument(
    #     "--sample",
    #     type=int,
    #     default=1,
    #     help="number of samples to be generated for each image",
    # )

    # parser.add_argument(
    #     "--pics", type=int, default=20, help="number of images to be generated"
    # )

    parser.add_argument("--truncation", type=float, default=1, help="truncation ratio")
    parser.add_argument(
        "--truncation_mean",
        type=int,
        default=4096,
        help="number of vectors to calculate mean for the truncation",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="stylegan2-ffhq-config-f.pt",
        help="path to the model checkpoint",
    )

    # the following is the same as train.py
    parser.add_argument(
        "--channel_multiplier",
        type=int,
        default=2,
        help="channel multiplier of the generator. config-f = 2, else = 1",
    )

    args = parser.parse_args()

    # load the correct model:
    if args.arch=='hypernetwork_offset_wth_FC_on_3_FC_middle_dim_512_32':
        from Alvin_model_experiment_hypernetwork_offset_with_FC_on_3_FC import MyNewGenerator, Discriminator

    if args.arch=='changed_mapping_network_with_3FC_reduced_RGB_style_dim_path_regularized':
        print('loading model from ', args.arch)
        from Alvin_model_experiment_changed_mapping_network_3FC_with_reduced_RGB_style_dim import MyNewGenerator, Discriminator

    # default model:
    if args.arch == 'stylegan2':
        print('loading model from ', args.arch)
        from model import Generator, Discriminator


    # define specific hyperparameters for corresponding models:
    if args.arch =='hypernetwork_with_2_layer_FC_offset_middle_dim_512_32' or 'hypernetwork_offset_wth_FC_on_3_FC_middle_dim_512_32':
        print(f'setting parameters for {args.arch} ')
        args.style_dim_for_changed_network = 32
        args.n_neurons_middle_layer = 512
        args.n_neurons_for_middle_layer_hypernet = 32
        args.latent = args.style_dim_for_changed_network*14
        args.model_input_tensor = True

    if args.arch== 'stylegan2':
        print('defining the hyperparameters for original stylegan2')
        args.latent = 512
        args.n_mlp = 8
        args.model_input_tensor = False

        print(args.n_mlp)


    if args.arch =='changed_mapping_network_with_3FC_reduced_RGB_style_dim_path_regularized' or args.arch=='hypernetwork_with_FC':
        g_ema = MyNewGenerator(args.size, w_style_dim = args.style_dim_for_changed_network,
        channel_multiplier=args.channel_multiplier,
        n_neurons_middle_layer = args.n_neurons_middle_layer
        ).to(device)

        # generator = MyNewGenerator(args.size, w_style_dim = args.style_dim_for_changed_network,
        # channel_multiplier=args.channel_multiplier,
        # n_neurons_middle_layer = args.n_neurons_middle_layer
        # ).to(device)

    elif args.arch=='hypernetwork_with_2_layer_FC_offset_middle_dim_512_32' or args.arch=='hypernetwork_offset_wth_FC_on_3_FC_middle_dim_512_32':
        g_ema = MyNewGenerator(args.size, w_style_dim = args.style_dim_for_changed_network,
        channel_multiplier=args.channel_multiplier,
        n_neurons_middle_layer = args.n_neurons_middle_layer,
        n_neurons_for_middle_layer_hypernet=args.n_neurons_for_middle_layer_hypernet
        ).to(device)

    # default stylegan2:
    else:
        print('args.n_mlp:', args.n_mlp)
        g_ema = Generator(
            args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
        ).to(device)
        generator = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)

        # generator = MyNewGenerator(args.size, w_style_dim = args.style_dim_for_changed_network,
        # channel_multiplier=args.channel_multiplier,
        # n_neurons_middle_layer = args.n_neurons_middle_layer,
        # n_neurons_for_middle_layer_hypernet=args.n_neurons_for_middle_layer_hypernet
        # ).to(device)

    print('loading checkpoint from ', args.ckpt)
    checkpoint = torch.load(args.ckpt, map_location=device)
    print('type of checkpoint: ', type(checkpoint))
    print(f'keys of the checkpoint from {args.ckpt} is {checkpoint.keys()}')
    g_ema.load_state_dict(checkpoint["g_ema"], strict = False)

    print('examining parameters of g_ema:', type(g_ema.parameters()))
    for param in g_ema.parameters():
        print(type(param), param.size())

    print(' \n \n examing parameters of the style network \n')

    for param in g_ema.FC_from_w_to_correction_for_next.parameters():
        print(type(param), param.size())

    # trying to extract only the parameters in the g:
    