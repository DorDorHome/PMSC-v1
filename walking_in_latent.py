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

# for plotting:
import matplotlib.pyplot as plt
import seaborn as sns
from load_model import load_g_ema

import argparse


print('empyting cache')
torch.cuda.empty_cache()
print('done empyting cache.')

print('pytorch version in used is:', torch.__version__)

def generate(args, g_ema, device, mean_latent):

    with torch.no_grad():
        g_ema.eval()
        for i in tqdm(range(args.pics)):
            sample_z = torch.randn(args.sample, args.latent, device=device)
            #sample_z.requires_grad = True

            sample, _ = g_ema(
                sample_z, truncation=args.truncation, truncation_correlated_latent=mean_latent
            )

            utils.save_image(
                sample,
                f"sample/{str(i).zfill(6)}.png",
                nrow=1,
                normalize=True,
                range=(-1, 1),
            )
    
def walk_in_latent(args, g_ema, device):
    with torch.no_grad():
        g_ema.eval()

        # set initial image:
        torch.manual_seed(args.random_seed)
        if not args.use_specified_latent_and_noise: 
            initial_z = torch.randn(1, args.latent, device = device )
            noise = None
        elif args.use_specified_latent_and_noise:
            print('loading latent')
            initial_z = torch.load(args.z_latent_path).to(device)

            noise = torch.load(args.trained_noise_path)

        # plot the histogram of the initial_z
        sns.kdeplot(data = initial_z[0].detach().clone().to('cpu'))
        plt.show()
        
        print('size of initial_z', initial_z.size())
        # loop through the layers to be worked on:
        # layers is of type list:

        for layer in tqdm(args.layers_to_walk):
            
            # for each layer, loop through all components of initial_z that corresponds to that layer:  
            print(f'walking on the {layer}-th layer' )
            if args.walk_by_single_argument:
                z_walk =[]
                # loop through each argument in this layer:
                for i in range(args.style_dim_for_changed_network):
                    z_walk.append(initial_z)
                    for step in range(args.number_of_steps_to_take):
                        # print(f'step {step+ 1}')
                        displacement = torch.zeros(args.latent, device = device)
                        displacement[(layer-1)*14+i:(layer-1)*14+i+1] += args.walk_size_in_latent*(step +1)
                        next_z = initial_z + displacement
                        z_walk.append(next_z)


            else:
                z_walk = [initial_z]
                next_z = initial_z
                for i in range(args.style_dim_for_changed_network):
                    displacement = torch.zeros(args.latent, device = device)
                    print('type of layer: ', type(layer))
            
                    displacement[(layer-1)*14:(layer-1)*14+i+1 ] += args.walk_size_in_latent
                    next_z = next_z + displacement
                    z_walk.append(next_z)

            z_walk = torch.cat(z_walk, dim=0)
            print('z_walk =', z_walk)
            # generate an image with g_ema:
            print('check shape of z_walk:', z_walk.shape)

            sample, _ = g_ema(z_walk, truncation=args.truncation, truncation_correlated_latent=mean_latent, noise = noise
            )

            if not os.path.exists(f"sample/walk_in_latent/{args.arch}_with_random_seed_{args.random_seed}"):
                os.makedirs(f"sample/walk_in_latent/{args.arch}_with_random_seed_{args.random_seed}")
            
            if not os.path.exists(f"sample/walk_in_latent/{args.arch}_using_specified_latent_and_noise"):
                os.makedirs(f"sample/walk_in_latent/{args.arch}_using_specified_latent_and_noise")

            if not args.use_specified_latent_and_noise: 
                utils.save_image(
                sample,
                f"sample/walk_in_latent/{args.arch}_with_random_seed_{args.random_seed}/layer_{layer}_walk_size_{args.walk_size_in_latent}_walk_by_single_argument_{args.walk_by_single_argument}.png",
                nrow=(args.number_of_steps_to_take +1),
                normalize=True,
                range=(-1, 1),
                )

            else:
                utils.save_image(
                sample,
                f"sample/walk_in_latent/{args.arch}_using_specified_latent_and_noise/layer_{layer}_walk_size_{args.walk_size_in_latent}_walk_by_single_argument_{args.walk_by_single_argument}.png",
                nrow=(args.number_of_steps_to_take +1),
                normalize=True,
                range=(-1, 1),
                )



    #return None



if __name__ == "__main__":

    # device = 'cpu'



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

    parser.add_argument(
        "--walk_size_in_latent",
        type = float,
        default=0.1
    )

    parser.add_argument(
        "--number_of_steps_to_take",
        type = int,
        default= 10
    )        # # plot the histogram of the initial_z
        # sns.kdeplot(data = initial_z[0].detach().clone().to('cpu'))

    parser.add_argument(
        "--layers_to_walk",
        nargs="*",  # 0 or more values expected => creates a list
        type=int,
        default = [1,2,3,4,5]
    )

    parser.add_argument(
        "--walk_by_single_argument",
        type= bool,
        default = True
    )

    parser.add_argument(
        "--random_seed",
        type= int,
        default = 0
    )

    parser.add_argument(
        "--use_specified_latent_and_noise",
        type= bool,
        default = False)

    parser.add_argument(
        '--z_latent_path',
        type = str,
        default = None
    )

    parser.add_argument(
    '--trained_noise_path',
    type = str,
    default = None
    )
    
    parser.add_argument(
    '--device',
    type=str,
    default = 'cuda:0'            
    )



    # parser.add_argument(

    # )

    # parser.add_argument(

    # )


    args = parser.parse_args()


    # load the correct model:
    # if args.arch=='hypernetwork_offset_wth_FC_on_3_FC_middle_dim_512_32':
    #     from Alvin_model_experiment_hypernetwork_offset_with_FC_on_3_FC import MyNewGenerator, Discriminator

    # if args.arch=='changed_mapping_network_with_3FC_reduced_RGB_style_dim_path_regularized':
    #     print('loading model from ', args.arch)
    #     from Alvin_model_experiment_changed_mapping_network_3FC_with_reduced_RGB_style_dim import MyNewGenerator, Discriminator

    # if args.arch =='hypernetwork_with_2_layer_FC_offset_middle_dim_512_32' or 'hypernetwork_offset_wth_FC_on_3_FC_middle_dim_512_32':
    #     print(f'setting parameters for {args.arch} ')
    #     args.style_dim_for_changed_network = 32
    #     args.n_neurons_middle_layer = 512
    #     args.n_neurons_for_middle_layer_hypernet = 32
    #     args.latent = args.style_dim_for_changed_network*14
    #     args.model_input_tensor = True

    # if args.arch =='changed_mapping_network_with_3FC_reduced_RGB_style_dim_path_regularized' or args.arch=='hypernetwork_with_FC':
    #     g_ema = MyNewGenerator(args.size, w_style_dim = args.style_dim_for_changed_network,
    #     channel_multiplier=args.channel_multiplier,
    #     n_neurons_middle_layer = args.n_neurons_middle_layer
    #     ).to(device)

    #     # generator = MyNewGenerator(args.size, w_style_dim = args.style_dim_for_changed_network,
    #     # channel_multiplier=args.channel_multiplier,
    #     # n_neurons_middle_layer = args.n_neurons_middle_layer
    #     # ).to(device)

    # elif args.arch=='hypernetwork_with_2_layer_FC_offset_middle_dim_512_32' or args.arch=='hypernetwork_offset_wth_FC_on_3_FC_middle_dim_512_32':
    #     g_ema = MyNewGenerator(args.size, w_style_dim = args.style_dim_for_changed_network,
    #     channel_multiplier=args.channel_multiplier,
    #     n_neurons_middle_layer = args.n_neurons_middle_layer,
    #     n_neurons_for_middle_layer_hypernet=args.n_neurons_for_middle_layer_hypernet
    #     ).to(device)

    #     # generator = MyNewGenerator(args.size, w_style_dim = args.style_dim_for_changed_network,
    #     # channel_multiplier=args.channel_multiplier,
    #     # n_neurons_middle_layer = args.n_neurons_middle_layer,
    #     # n_neurons_for_middle_layer_hypernet=args.n_neurons_for_middle_layer_hypernet
    #     # ).to(device)

    g_ema = load_g_ema(args)

    print('loading checkpoint')
    checkpoint = torch.load(args.ckpt, map_location=args.device)

    g_ema.load_state_dict(checkpoint["g_ema"], strict = False)

    # if args.truncation < 1:
    #     with torch.no_grad():
    #         mean_latent = g_ema.mean_latent(args.truncation_mean)
    # else:
         
    mean_latent = None
    # generate(args, g_ema, device, mean_latent)
    walk_in_latent(args, g_ema, args.device)