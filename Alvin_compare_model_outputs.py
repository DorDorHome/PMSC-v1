# the purpose of this file is to compare the quality of the outputs using the two different models and their corresponding checkpoints

# please adjust save frequency(both checkpoing sample) before running 


import argparse
import math
import random
import os

import numpy as np
import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
import torch.distributed as dist
from torchvision import transforms, utils
from tqdm import tqdm

# this is copied from generate.py, for the purpose of generating images from models:

def generate(args, g_ema, device, mean_latent):
    """
    generate args.sample images separately
    
    """

    with torch.no_grad():
        g_ema.eval()
        for i in tqdm(range(args.pics)):
            sample_z = torch.randn(args.sample, args.latent, device=device)

            sample, _ = g_ema(
                [sample_z], truncation=args.truncation, truncation_latent=mean_latent
            )

            utils.save_image(
                sample,
                f"sample/{str(i).zfill(6)}.png",
                nrow=1,
                normalize=True,
                range=(-1, 1),
            )


def generate_grid(args, g_ema, device, mean_latent):
    """
    generate images from g_ema and saved as a grid
    
    """





# load model L:

if __name__ =='__main__':
    # set up the device to be used 
    device = 'cuda:1'
    torch.cuda.empty_cache()

    # create a parser:
    parser = argparse.ArgumentParser(description='generate a series of output from two different models for comparison')
    parser.add_argument('--L_model_checkpoint_path', type = str)
    parser.add_argument('--R_model_checkpoint_path', type = str)    

