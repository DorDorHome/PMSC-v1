# please adjust save frequency(both checkpoing sample) before running 


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


try:
    import wandb
    print('finished importing wandb')

except ImportError:
    wandb = None
    print('setting wandb = None')


from dataset import MultiResolutionDataset
from distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)
print('imported from distributed.')
from op import conv2d_gradfix
print('imported conv2d_gradfix')
from non_leaking import augment, AdaptiveAugment

print('defininng functions.')

def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    """
    added by Alvin
    Input:
    two model, model1 and model2,
    The two models are supposed to have the same architecture

    Output:
    Update the weight of model1 towards that of model2 using
    exponentially weighted moving average.

_______________________________
    Used in the model, this function is used to make model1 having the 
    exponentially weighted moving averaged weights of model2 (the stylegan2 
    used successively training steps)
    
    
    """


    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def d_logistic_loss(real_pred, fake_pred):
    """
    loss function for the discriminator
    basically a smoothed version of the logistic loss function

    input:
    shape? 
    
    
    """    


    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img):
    with conv2d_gradfix.no_weight_gradients():


        # the following computed the gradient of output (a scalar for each sample)
        # w.r.t . the real image.
        grad_real, = autograd.grad(
            outputs=real_pred.sum(), inputs=real_img, create_graph=True
        )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty


def g_nonsaturating_loss(fake_pred):
    """
    adversarial loss for g.
    
    input:
    a score of the fake image being real
    The value is from -infinity to infinity.
    
    
    """

    loss = F.softplus(-fake_pred).mean()

    return loss


def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    assert fake_img.requires_grad==True, 'fake_img does not require grad'
    assert latents[0].requires_grad == True, 'latents do not require grad'

    # if latents is a list instead of a tensor, concatenate it:
    if type(latents)==list:
        print('latent is of type list.')
        latents = torch.cat(latents, dim = 1)
        print('aftering concatenating latents, it is of type', type(latents))
        print('Whether concatenated latent requires grad:', latents.requires_grad )

    noise = torch.randn_like(fake_img) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3]
    )
    # print('whether noise requires grad:', noise.requires_grad)
    grad, = autograd.grad(
        outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True
    )
    # print('shape of latents within g_path_regularize: ', latents.shape)
    # print('shape of grad within g_path_regularize:', grad.shape)
    # print('grad.sum(2)', grad.sum(2))

    # original, which doesn't for adapted model:
    # path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    # adapted:
    if grad.dim() == 3:
        path_lengths = torch.sqrt(grad.pow(2).sum( -1).mean(1))
    if grad.dim() ==2:
        path_lengths = torch.sqrt(grad.pow(2).sum( -1))

    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_mean.detach(), path_lengths


def generate_gaussian_uniform_mixture(args, proportion_of_latent_with_uniform, device):
    num_layers_to_use_uniform = int(args.num_input_layers_for_latent*proportion_of_latent_with_uniform)
    #print('num_layers_to_use_uniform', num_layers_to_use_uniform)
    layers_to_use_uniform = np.random.choice(np.arange(args.num_input_layers_for_latent), num_layers_to_use_uniform)

    noise_list = []
    for layer in np.arange(args.num_input_layers_for_latent):
        if layer in layers_to_use_uniform:
            # make a batch of uniform noise on [-2, 2]:
            noise = torch.rand(args.batch, args.style_dim_for_changed_network, device = device)*4 -2 
            
        else:

            noise = torch.randn(args.batch, args.style_dim_for_changed_network, device =device)
        noise_list.append(noise)
    
    noise = torch.cat(noise_list, dim =1)

    return noise

def make_noise(batch, latent_dim, n_noise, device , model_input_tensor = False, use_gauss_with_uniform_mix = False, uniform_mixing_ratio = 0.4):
    """
    make noise tensor(s) of shape (latent_dim).
    If n_noise (number of noise tensor) is larger than one,
    make a tuple of n_noise tensors, each of shape latent_dim
    
    
    
    """
    
    
    if model_input_tensor == False:
        if n_noise == 1:
            return torch.randn(batch, latent_dim, device=device)

        noises = torch.randn(n_noise, batch, latent_dim, device=device).unbind(0)

    else:
        noises = torch.randn(batch, latent_dim, device=device)

    return noises


def mixing_noise(batch, latent_dim, prob, device, model_input_tensor = False):
    """
    with probability prob, make a tuple of two noise tensors having dimensions = latent_dim
    with probability 1- prob, use just a list of one noise tensor
    
    """
    if model_input_tensor==True:
        return make_noise(batch, latent_dim, 1, device, model_input_tensor=True)

    else:
        if prob > 0 and random.random() < prob:
            return make_noise(batch, latent_dim, 2, device, model_input_tensor=False)

        else:
            return [make_noise(batch, latent_dim, 1, device, model_input_tensor = False)]


def set_grad_none(model, targets):
    for n, p in model.named_parameters():
        if n in targets:
            p.grad = None


def train(args, loader, generator, discriminator, g_optim, d_optim, g_ema, device):
    
    # 
    loader = sample_data(loader)

    # for monitoring progress only:
    pbar = range(args.iter)

    # check if distributed training is available:
    if get_rank() == 0:
        # if it is not available:
        pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=True, smoothing=0.01)

    # initialize mean_path_length
    mean_path_length = 0

    d_loss_val = 0
    r1_loss = torch.tensor(0.0, device=device)
    g_loss_val = 0
    path_loss = torch.tensor(0.0, device=device)
    path_lengths = torch.tensor(0.0, device=device)
    mean_path_length_avg = 0
    
    # initialize the loss_dict
    # this loss dict will have keys
    #  "d", "real_score", 'fake_score', 'r1', 'g', 'path_length',

    loss_dict = {}

    if args.distributed:
        g_module = generator.module
        d_module = discriminator.module

    else:
        # if distributed learning is set to False, set g_module to be the generator network
        # and d_module to be the discriminator network
        g_module = generator
        d_module = discriminator

    accum = 0.5 ** (32 / (10 * 1000))
    ada_aug_p = args.augment_p if args.augment_p > 0 else 0.0
    r_t_stat = 0

    if args.augment and args.augment_p == 0:
        ada_augment = AdaptiveAugment(args.ada_target, args.ada_length, 8, device)

    print('generating a fixed latent to monitor progress:')
    sample_z = torch.randn(args.n_sample, args.latent, device=device)
    sample_z.requires_grad=True

    # start the traininng iteration:
    # for each iteration:
    for idx in pbar:

        # shift forward, starting from 'start_iter'
        i = idx + args.start_iter

        # check if specified iteration is reached. If so, break the loop.
        if i > args.iter:
            print("Done!")

            break

        # take the next batch of images, in the form of a tensor (batch_size,......)
        real_img = next(loader)
        real_img = real_img.to(device)

        requires_grad(generator, False)
        requires_grad(discriminator, True)

        # probability = args.mixing,
        # either make (a tuple of) two separate noise tensors of shape (batch, latent) for mixing purpose
        # or simply just use (a list of) one noise tensor of shape (batch, latent)
        # print( "model_input_tensor=", args.model_input_tensor)
        if args.mix_gaussian_and_uniform_for_training == True:
            print('using uniform in some layers....')
            noise = generate_gaussian_uniform_mixture(args,proportion_of_latent_with_uniform = 0.4, device = device)
        else:
            # print('not using uniform as noise')
            noise = mixing_noise(args.batch, args.latent, args.mixing, device, model_input_tensor=args.model_input_tensor)
        noise.requires_grad=True
        # generate a fake image with forward pass of generator
        # in the case of stylegan2, the output is expected to be image, None
        # please refer to Generator class in model.py 

        fake_img, _ = generator(noise)
        
        if args.augment:
            real_img_aug, _ = augment(real_img, ada_aug_p)
            fake_img, _ = augment(fake_img, ada_aug_p)

        else:
            real_img_aug = real_img

        # get the score for the fake and real images, respectively,
        # using the discriminator 
        fake_pred = discriminator(fake_img)
        real_pred = discriminator(real_img_aug)

        # get the loss for the discriminator:
        d_loss = d_logistic_loss(real_pred, fake_pred)
        # print('Current d loss: ', d_loss)

        # update the loss dict using the losses in this batch:
        loss_dict["d"] = d_loss
        loss_dict["real_score"] = real_pred.mean()
        loss_dict["fake_score"] = fake_pred.mean()

        # update the weight of the discriminator using d_loss:
        discriminator.zero_grad()
        d_loss.backward()
        d_optim.step()


        if args.augment and args.augment_p == 0:
            ada_aug_p = ada_augment.tune(real_pred)
            r_t_stat = ada_augment.r_t_stat

        # for every d_reg_every, set d_regularize to be 'True'
        d_regularize = i % args.d_reg_every == 0

        # lazy regularization for discriminator 
        if d_regularize:
            real_img.requires_grad = True

            if args.augment:
                real_img_aug, _ = augment(real_img, ada_aug_p)

            else:
                real_img_aug = real_img

            real_pred = discriminator(real_img_aug)
            r1_loss = d_r1_loss(real_pred, real_img)

            discriminator.zero_grad()
            (args.r1 / 2 * r1_loss * args.d_reg_every + 0 * real_pred[0]).backward()

            d_optim.step()

        loss_dict["r1"] = r1_loss

        #prepare to update the weights of the generator:
        requires_grad(generator, True)
        requires_grad(discriminator, False)

        # create a noise for the generator
        # with probability = args.mixing, this can be a tuple of two noise tensors
        if args.mix_gaussian_and_uniform_for_training == True:
            print('using uniform in some layers....')
            noise = generate_gaussian_uniform_mixture(args,proportion_of_latent_with_uniform = 0.4, device = device)
        else:
            #print('not using uniform as noise')
            noise = mixing_noise(args.batch, args.latent, args.mixing,  device, model_input_tensor=args.model_input_tensor)
        noise.requires_grad = True
        fake_img, _ = generator(noise)

        if args.augment:
            fake_img, _ = augment(fake_img, ada_aug_p)

        # get the probability of the fake_img being real:
        fake_pred = discriminator(fake_img)

        # get the adversarial loss for g
        g_loss = g_nonsaturating_loss(fake_pred)


        loss_dict["g"] = g_loss

        # update the weights of generator
        generator.zero_grad()
        g_loss.backward()
        g_optim.step()

        if args.clamp_parameter_values:
            for p in generator.parameters():
                p.data.clamp_(100000,100000)

        # lazy regularizations:
        # (added by Alvin): This turns on every few iteration:
        # print('use g path regularization: ', args.use_g_regularization)
        if args.use_g_regularization == True:
            g_regularize = i % args.g_reg_every == 0

            if g_regularize:
                path_batch_size = max(1, args.batch // args.path_batch_shrink)
                if args.mix_gaussian_and_uniform_for_training == True:
                    print('using uniform in some layers....')
                    noise = generate_gaussian_uniform_mixture(args,proportion_of_latent_with_uniform = 0.4, device = device)
                else:
                    # print('not using uniform as noise')
                    noise = mixing_noise(path_batch_size, args.latent, args.mixing, device, model_input_tensor=args.model_input_tensor)
                noise.requires_grad= True
                if args.arch=="changed_mapping_network_style_proj_to_sphere" \
                or args.arch=='multi_stage_GRU_Gauss_root_softmax' \
                    or args.arch =='changed_mapping_network_with_3FC_reduced_RGB_style_dim_path_regularized' \
                    or args.arch=='hypernetwork_with_FC' \
                    or args.arch=='hypernetwork_with_2_layer_FC_offset_middle_dim_512_32' \
                    or args.arch=='hypernetwork_offset_wth_FC_on_3_FC_middle_dim_512_32':
                    fake_img, latents = generator(noise, return_correlated_latent=True)
                elif args.arch=='multi_stage_GRU_Gauss_root_softmax_path_regularized' \
                    or args.arch=='GRU_multistage_mapping_network_with_path_regularization' \
                    or args.arch=='multi_stage_GRU_skip_connection_root_softmax_with_path_regularization':
                    print('using return_transformed_latent.')
                    fake_img, latents = generator(noise, return_transformed_latent = True)
                else:
                    fake_img, latents = generator(noise, return_latents=True)

                #for getting path loss    
                
                if args.g_path_regularize_from_z:
                    path_loss, mean_path_length, path_lengths = g_path_regularize(
                        fake_img, noise, mean_path_length
                        )


                elif not args.g_path_regularize_from_z:
                    path_loss, mean_path_length, path_lengths = g_path_regularize(
                        fake_img, latents, mean_path_length
                    )

                # preparing the update the weights of generator:
                generator.zero_grad()
                weighted_path_loss = args.path_regularize * args.g_reg_every * path_loss

                if args.path_batch_shrink:
                    weighted_path_loss += 0 * fake_img[0, 0, 0, 0]

                weighted_path_loss.backward()
                
                # updating the weight of generator:
                g_optim.step()

                mean_path_length_avg = (
                    reduce_sum(mean_path_length).item() / get_world_size()
                )
        
        # updating the latest path_loss
        loss_dict["path"] = path_loss
        # updating the latest path_length:
        loss_dict["path_length"] = path_lengths.mean()

        # with the accum hyperparameter, update the weight of g_
        # (exponetially weighted moving average of weights of successive g 
        accumulate(g_ema, g_module, accum)

        loss_reduced = reduce_loss_dict(loss_dict)

        d_loss_val = loss_reduced["d"].mean().item()
        g_loss_val = loss_reduced["g"].mean().item()
        r1_val = loss_reduced["r1"].mean().item()
        path_loss_val = loss_reduced["path"].mean().item()
        real_score_val = loss_reduced["real_score"].mean().item()
        fake_score_val = loss_reduced["fake_score"].mean().item()
        path_length_val = loss_reduced["path_length"].mean().item()

        # create folders to contain the sample and checkpoints:

        if not os.path.exists(f"{args.arch}_{args.activation_pre_affine}_trained_on_{args.path}_use_spectral_{args.use_spectral_norm_in_disc}_train_with_unif_{args.mix_gaussian_and_uniform_for_training}_path_reg_from_z_{args.g_path_regularize_from_z}_{args.size}_sample"):
            os.makedirs(f"{args.arch}_{args.activation_pre_affine}_trained_on_{args.path}_use_spectral_{args.use_spectral_norm_in_disc}_train_with_unif_{args.mix_gaussian_and_uniform_for_training}_path_reg_from_z_{args.g_path_regularize_from_z}_{args.size}_sample")
        
        if not os.path.exists(f"{args.arch}_{args.activation_pre_affine}_trained_on_{args.path}_use_spectral_{args.use_spectral_norm_in_disc}_train_with_unif_{args.mix_gaussian_and_uniform_for_training}_path_reg_from_z_{args.g_path_regularize_from_z}_{args.size}_checkpoint"):
            os.makedirs(f"{args.arch}_{args.activation_pre_affine}_trained_on_{args.path}_use_spectral_{args.use_spectral_norm_in_disc}_train_with_unif_{args.mix_gaussian_and_uniform_for_training}_path_reg_from_z_{args.g_path_regularize_from_z}_{args.size}_checkpoint")


        if get_rank() == 0:
            pbar.set_description(
                (
                    f"d: {d_loss_val:.4f}; g: {g_loss_val:.4f}; r1: {r1_val:.4f}; "
                    f"path: {path_loss_val:.4f}; mean path: {mean_path_length_avg:.4f}; "
                    f"augment: {ada_aug_p:.4f}"
                )
            )

            if wandb and args.wandb:
                wandb.log(
                    {
                        "Generator": g_loss_val,
                        "Discriminator": d_loss_val,
                        "Augment": ada_aug_p,
                        "Rt": r_t_stat,
                        "R1": r1_val,
                        "Path Length Regularization": path_loss_val,
                        "Mean Path Length": mean_path_length,
                        "Real Score": real_score_val,
                        "Fake Score": fake_score_val,
                        "Path Length": path_length_val,
                    }
                )

            if i< 1000:
                if i % 100 == 0:
                # for every 100 iterations, generate n_samples number of samples 
                # and save, from the same noise vector.
                    with torch.no_grad():
                        g_ema.eval()
                        if args.arch =='multi_stage_Gaussian_affine_with_activation' or \
                            args.arch=="changed_mapping_network_style_proj_to_sphere" or \
                                args.arch=='changed_mapping_network_with_reduced_RGB_style_dim' or \
                                args.arch=='hypernetwork_offset_wth_FC_on_3_FC_middle_dim_512_32' or \
                                    args.arch=="multi_stage_Gaussian_affine":
                            sample, _ = g_ema(sample_z)
                        else:
                            try:
                                sample, _ = g_ema([sample_z])
                            except:
                                sample, _ = g_ema(sample_z)
                        utils.save_image(
                            sample,
                            f"{args.arch}_{args.activation_pre_affine}_trained_on_{args.path}_use_spectral_{args.use_spectral_norm_in_disc}_train_with_unif_{args.mix_gaussian_and_uniform_for_training}_path_reg_from_z_{args.g_path_regularize_from_z}_{args.size}_sample/{str(i).zfill(7)}.png",
                            nrow=int(args.n_sample ** 0.5),
                            normalize=True,
                            range=(-1, 1),
                        )
            elif i>=1000 and i< 10000:
                if i % 1000 ==0:
                    with torch.no_grad():
                        g_ema.eval()
                        if args.arch =='multi_stage_Gaussian_affine_with_activation' or args.arch=="changed_mapping_network_style_proj_to_sphere" or args.arch=='changed_mapping_network_with_reduced_RGB_style_dim' or args.arch=="multi_stage_Gaussian_affine":
                            sample, _ = g_ema(sample_z)
                        else:
                            try:
                                sample, _ = g_ema([sample_z])
                            except:
                                sample, _ = g_ema(sample_z)
                        utils.save_image(
                            sample,
                            f"{args.arch}_{args.activation_pre_affine}_trained_on_{args.path}_use_spectral_{args.use_spectral_norm_in_disc}_train_with_unif_{args.mix_gaussian_and_uniform_for_training}_path_reg_from_z_{args.g_path_regularize_from_z}_{args.size}_sample/{str(i).zfill(7)}.png",
                            nrow=int(args.n_sample ** 0.5),
                            normalize=True,
                            range=(-1, 1),
                        )
            elif i>=10000 and i<40000:
                if i%5000==0:
                    with torch.no_grad():
                        g_ema.eval()
                        if args.arch =='multi_stage_Gaussian_affine_with_activation'  or args.arch=="changed_mapping_network_style_proj_to_sphere" or args.arch=='changed_mapping_network_with_reduced_RGB_style_dim' or args.arch=="multi_stage_Gaussian_affine":
                            sample, _ = g_ema(sample_z)
                        else:
                            try:
                                sample, _ = g_ema([sample_z])
                            except:
                                sample, _ = g_ema(sample_z)
                        utils.save_image(
                            sample,
                            f"{args.arch}_{args.activation_pre_affine}_trained_on_{args.path}_use_spectral_{args.use_spectral_norm_in_disc}_train_with_unif_{args.mix_gaussian_and_uniform_for_training}_path_reg_from_z_{args.g_path_regularize_from_z}_{args.size}_sample/{str(i).zfill(7)}.png",
                            nrow=int(args.n_sample ** 0.5),
                            normalize=True,
                            range=(-1, 1),
                        )
            else:
                if i%10000 == 0:
                    with torch.no_grad():
                        g_ema.eval()
                        if args.arch =='multi_stage_Gaussian_affine_with_activation'  or args.arch=="changed_mapping_network_style_proj_to_sphere" or args.arch=='changed_mapping_network_with_reduced_RGB_style_dim' or args.arch=="multi_stage_Gaussian_affine":
                            sample, _ = g_ema(sample_z)
                        else:
                            try:
                                sample, _ = g_ema([sample_z])
                            except:
                                sample, _ = g_ema(sample_z)
                        utils.save_image(
                            sample,
                            f"{args.arch}_{args.activation_pre_affine}_trained_on_{args.path}_use_spectral_{args.use_spectral_norm_in_disc}_train_with_unif_{args.mix_gaussian_and_uniform_for_training}_path_reg_from_z_{args.g_path_regularize_from_z}_{args.size}_sample/{str(i).zfill(7)}.png",
                            nrow=int(args.n_sample ** 0.5),
                            normalize=True,
                            range=(-1, 1),
                        )


            if i % 10000 == 0:
                torch.save(
                    {
                        "g": g_module.state_dict(),
                        "d": d_module.state_dict(),
                        "g_ema": g_ema.state_dict(),
                        "g_optim": g_optim.state_dict(),
                        "d_optim": d_optim.state_dict(),
                        "args": args,
                        "ada_aug_p": ada_aug_p,
                    },
                    f"{args.arch}_{args.activation_pre_affine}_trained_on_{args.path}_use_spectral_{args.use_spectral_norm_in_disc}_train_with_unif_{args.mix_gaussian_and_uniform_for_training}_path_reg_from_z_{args.g_path_regularize_from_z}_{args.size}_checkpoint/{str(i).zfill(7)}.pt",
                )

if __name__ == "__main__":
    print('starting device.')
    device = "cuda:1"
    print('empyting cache')
    torch.cuda.empty_cache()
    print('done empyting cache.')

    parser = argparse.ArgumentParser(description="StyleGAN2 trainer")

    parser.add_argument("path", type=str, help="path to the lmdb dataset")
    parser.add_argument('--arch', type=str, default='stylegan2', help='model architectures (stylegan2 | swagan)')
    parser.add_argument(
        "--iter", type=int, default=800000, help="total training iterations"
    )
    parser.add_argument(
        "--batch", type=int, default=16, help="batch sizes for each gpus"
    )
    parser.add_argument("--n_sample",
        type=int,
        default=64,
        help="number of the samples generated during training"
    )

    parser.add_argument("--size", type=int, default=256, help="image sizes for the model"
    )

    parser.add_argument(
        "--r1", type=float, default=10, help="weight of the r1 regularization"
    )

    parser.add_argument(
        "--use_g_regularization", type = bool, default = False, help= 'whether to use g path regularization.'
    )

    parser.add_argument(
        "--path_regularize",
        type=float,
        default=2,
        help="weight of the path length regularization",
    )
    parser.add_argument(
        "--path_batch_shrink",
        type=int,
        default=2,
        help="batch size reducing factor for the path length regularization (reduce memory consumption)",
    )
    parser.add_argument(
        "--d_reg_every",
        type=int,
        default=16,
        help="interval of the applying r1 regularization",
    )
    parser.add_argument(
        "--g_reg_every",
        type=int,
        default=4,
        help="interval of the applying path length regularization",
    )
    parser.add_argument( "--mixing", type=float, default=0.9, help="probability of latent code mixing"
    )
    parser.add_argument("--ckpt",
        type=str,
        default=None,
        help="path to the checkpoints to resume training",
    )
    parser.add_argument("--lr", type=float, default=0.002, help="learning rate")
    parser.add_argument('--discriminator_lr_reduction_factor', type= float, default = 1, help = 'reduce the learning rate of the discriminator by this factor. Refer to StyleGAN1 paper for details.' )
    parser.add_argument(
        "--channel_multiplier",
        type=int,
        default=2,
        help="channel multiplier factor for the model. config-f = 2, else = 1",
    )
    parser.add_argument(
        "--wandb", action="store_true", help="use weights and biases logging"
    )
    parser.add_argument(
        "--local_rank", type=int, default=0, help="local rank for distributed training"
    )
    parser.add_argument(
        "--augment", action="store_true", help="apply non leaking augmentation"
    )
    parser.add_argument(
        "--augment_p",
        type=float,
        default=0,
        help="probability of applying augmentation. 0 = use adapaccelerate envtive augmentation",
    )
    parser.add_argument(
        "--ada_target",
        type=float,
        default=0.6,
        help="target augmentation probability for adaptive augmentation",
    )

    parser.add_argument(
        "--g_path_regularize_from_z",
        type=bool,
        default = False,
        help ='whether the path regularize should regularize path lengths from z to images, instead of from w to images.\
            for PMSC-GAN, this should be set to True.'
    )

    parser.add_argument(
        "--ada_length",
        type=int,
        default=500 * 1000,
        help="target duraing to reach augmentation probability for adaptive augmentation",
    )
    parser.add_argument(
        "--ada_every",
        type=int,
        default=256,
        help="probability update interval of the adaptive augmentation",
    )


    parser.add_argument(
        "--activation_pre_affine",
        type = str,
        default = '',
        help = 'activation map used to make affine map from the previous style vector.'
    )

    parser.add_argument(
        "--activation_for_style",
        type= str,
        default = '',
        help = 'activation map used on the style output before applying style modulation.'

    )
    parser.add_argument(
        "--intermediate_activation",
        type= str,
        default = '',
        help = 'for those network with more complicated modulated Cov (e.g. with GRU), whether to use intermediate activation function before bilinear map.'

    )

    parser.add_argument(
        "--prev_style_hidden_dim",
        type = int,
        default = '256',
        help ='for memory based models, this specifies the dimensions of the hidden state'
    )

    parser.add_argument(
        "--clamp_parameter_values",
        type=bool,
        default = False,
        help = 'whether to restrict the range of values of parameters of model' )

    parser.add_argument(
        "--use_accelerate",
        type = bool,
        default =False,
        help= 'whether to use the accelerate package in order to use multiple GPUs.'
    )

    parser.add_argument(
        '--mix_gaussian_and_uniform_for_training',
        type = bool,
        default = False,
        help ='whether to set some layers as uniform distribution to generate the latent vector during training.'
    )

    parser.add_argument(
        '--use_spectral_norm_in_disc',
        type = bool,
        default= False,
        help = 'whether to use spectral normalisation in the discriminator'
    )

    args = parser.parse_args()
    print('whether to use g_regularization:', args.use_g_regularization)

    n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = False # n_gpu > 1

    print('args.distributed: ', args.distributed)
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    print('Setting model hyperparameters.')

    print('args.arch = ', args.arch)
    if args.arch =='hypernetwork_with_2_layer_FC_offset_middle_dim_512_32' or args.arch =='hypernetwork_offset_wth_FC_on_3_FC_middle_dim_512_32':
        print(f'setting parameters for {args.arch} ')
        args.style_dim_for_changed_network = 32
        args.n_neurons_middle_layer = 512
        args.n_neurons_for_middle_layer_hypernet = 32

        args.num_input_layers_for_latent = (int(math.log(args.size, 2)) -2)*2 +2 # 14  # to calculate how many upsampling stages is needed
        
        print('num_input_layers_for_latent', args.num_input_layers_for_latent )
        args.latent = args.style_dim_for_changed_network*args.num_input_layers_for_latent
        args.model_input_tensor = True

    elif args.arch=='hypernetwork_with_FC' or \
        args.arch=='multi_stage_GRU_Gauss_root_softmax_path_regularized' or \
        args.arch=='multi_stage_GRU_Gauss_root_softmax' or args.arch=='mutli_stage_GRU_with_root_softmax' or \
        args.arch=='multi_stage_Gaussian_affine_with_double_activation' or args.arch=="multi_stage_Gaussian_affine" or \
        args.arch == 'changed_mapping_network_with_2FC_reduced_RGB_style_dim' or args.arch=="changed_mapping_network_style_proj_to_sphere" or args.arch=='changed_mapping_network_with_reduced_RGB_style_dim' or \
        args.arch =='changed_mapping_network_with_3FC_reduced_RGB_style_dim_path_regularized':
        # args.use_g_regularization=False
        print('setting first set of parameters for training.')
        args.style_dim_for_changed_network = 32
        args.n_neurons_middle_layer = 512

        args.latent = args.style_dim_for_changed_network*14
        args.model_input_tensor = True
    elif args.arch=='GRU_multistage_mapping_network_with_path_regularization' or args.arch=="multi_stage_GRU_skip_connection_root_softmax_with_path_regularization" or args.arch =='multi_stage_Gaussian_affine_with_activation' or args.arch=="multi_stage_GRU_skip_connection_root_softmax": # same as above:
        print('setting parameters for training.')
        #args.use_g_regularization=True
        args.style_dim_for_changed_network = 32
        args.n_neurons_middle_layer =256
        args.latent = args.style_dim_for_changed_network*14
        args.model_input_tensor = True

    else:
        print('using stylegan2 ')
        args.latent = 512
        args.n_mlp = 8
        args.model_input_tensor = False

    args.start_iter = 0

    # load the model, according to args.arch:

    if args.arch=='hypernetwork_offset_wth_FC_on_3_FC_middle_dim_512_32':
        from Alvin_model_experiment_hypernetwork_offset_with_FC_on_3_FC import MyNewGenerator, Discriminator

    if args.arch =='hypernetwork_with_2_layer_FC_offset_middle_dim_512_32':
        from Alvin_model_experiment_hypernetwork_offset_with_FC import MyNewGenerator, Discriminator

    if args.arch=='hypernetwork_with_FC':
        from Alvin_model_experiment_hypernetwork_with_FC import MyNewGenerator, Discriminator

    if args.arch=='GRU_multistage_mapping_network_with_path_regularization':
        from Alvin_experiment_GRU_multistage_mapping_network_with_path_regularization import MyNewGenerator, Discriminator
    if args.arch=='multi_stage_GRU_skip_connection_root_softmax_with_path_regularization':
        print('load from multi_stage_GRU_skip_connection_root_softmax_with_path_regularization ')
        from multi_stage_GRU_skip_connection_root_softmax_with_path_regularization import MyNewGenerator, Discriminator

    if args.arch=="multi_stage_GRU_skip_connection_root_softmax":
        from Experiment_GRU_skip_connection_multistage_mapping_network import MyNewGenerator, Discriminator

    if args.arch=='multi_stage_GRU_with_root_softmax':
        from Alvin_experiment_GRU_multistage_mapping_network import MyNewGenerator, Discriminator

    if args.arch=='multi_stage_GRU_Gauss_root_softmax':
        from Alvin_experiment_GRU_plus_Gaussian_multistage_mapping_network import MyNewGenerator, Discriminator

    if args.arch=='multi_stage_GRU_Gauss_root_softmax_path_regularized':
        from Alvin_experiment_GRU_plus_Gaussian_multistage_mapping_network_path_regularized import MyNewGenerator, Discriminator

    if args.arch=="multi_stage_Gaussian_affine":
        from Alvin_model_experiment_changed_mapping_network_Multi_Gaussian_with_reduced_RGB_style_dim import MyNewGenerator, Discriminator

    if args.arch =='multi_stage_Gaussian_affine_with_activation':
        from Alvin_model_experiment_changed_mapping_network_multi_stage_Gaussian_with_activations import MyNewGenerator, Discriminator

    if args.arch=='multi_stage_Gaussian_affine_with_double_activation':
        from Alvin_model_experiment_changed_mapping_network_multi_stage_Gaussian_with_double_activations import MyNewGenerator, Discriminator


    if args.arch=="changed_mapping_network_style_proj_to_sphere":
        from Alvin_model_experiment_changed_mapping_network_with_style_projection_to_sphere_scaled import MyNewGenerator, Discriminator

    if args.arch=='changed_mapping_network_with_reduced_RGB_style_dim':
        from Alvin_model_experiment_changed_mapping_network_with_reduced_RGB_style_dim import MyNewGenerator, Discriminator

    if args.arch == 'changed_mapping_network_with_2FC_reduced_RGB_style_dim':
        from Alvin_model_experiment_changed_mapping_network_2FC_with_reduced_RGB_style_dim import MyNewGenerator, Discriminator

    if args.arch=='changed_mapping_network_with_3FC_reduced_RGB_style_dim_path_regularized':
        print('loading model from ', args.arch)
        from Alvin_model_experiment_changed_mapping_network_3FC_with_reduced_RGB_style_dim import MyNewGenerator, Discriminator


    if args.arch =="NewStylegan2_with_bias_spherical_projection_scaled_2nd_attempt":
        from Alvin_model_experiment_with_style_projection_to_sphere_scaled import MyNewGenerator, Discriminator

    if args.arch=="NewStylegan2_with_bias_spherical_projection":
        print('loading model from ', args.arch)
        from Alvin_model_experiment_with_style_projection_to_sphere import MyNewGenerator, Discriminator

    if args.arch == 'NewStylegan2':
        print('loading model from ', args.arch)
        from Alvin_model_experiment import MyNewGenerator, Discriminator

    if args.arch == 'NewStylegan2.2':

        print('loading model from ', args.arch)
        from model_small_changes import Generator, Discriminator

    if args.arch == 'stylegan2':
        print('loading model from ', args.arch)
        from model import Generator, Discriminator

    elif args.arch == 'swagan':
        print('loading model from ', args.arch)
        from swagan import Generator, Discriminator

    # initialize the model: 
    if args.arch == 'NewStylegan2' or args.arch == "NewStylegan2_with_bias_spherical_projection_scaled_2nd_attempt" or args.arch == 'NewStylegan2_softmax_sqrt_with_bias':
        g_ema = MyNewGenerator(
            args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
        ).to(device)

        generator = MyNewGenerator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
        ).to(device)

    elif  args.arch =='multi_stage_Gaussian_affine_with_activation' or  args.arch=='multi_stage_Gaussian_affine_with_double_activation':
        g_ema = MyNewGenerator(args.size, w_style_dim = args.style_dim_for_changed_network,
        channel_multiplier=args.channel_multiplier,
        n_neurons_middle_layer = args.n_neurons_middle_layer,
        activation_pre_affine = args.activation_pre_affine 
        ).to(device)

        generator = MyNewGenerator(args.size, w_style_dim = args.style_dim_for_changed_network,
        channel_multiplier=args.channel_multiplier,
        n_neurons_middle_layer = args.n_neurons_middle_layer,
        activation_pre_affine = args.activation_pre_affine 
        ).to(device)
    elif args.arch=='GRU_multistage_mapping_network_with_path_regularization' or args.arch=="multi_stage_GRU_skip_connection_root_softmax_with_path_regularization" or args.arch=='mutli_stage_GRU_with_root_softmax' or args.arch=="multi_stage_GRU_skip_connection_root_softmax":
        g_ema = MyNewGenerator(args.size, prev_style_hidden_dim = args.prev_style_hidden_dim , w_style_dim = args.style_dim_for_changed_network,
        channel_multiplier=args.channel_multiplier,
        n_neurons_middle_layer = args.n_neurons_middle_layer,
        activation_for_style = args.activation_for_style,
        ).to(device)

        generator =  MyNewGenerator(args.size, prev_style_hidden_dim = args.prev_style_hidden_dim , w_style_dim = args.style_dim_for_changed_network,
        channel_multiplier=args.channel_multiplier,
        n_neurons_middle_layer = args.n_neurons_middle_layer,
        activation_for_style = args.activation_for_style,
        ).to(device)
    elif args.arch=='multi_stage_GRU_Gauss_root_softmax' or args.arch=='multi_stage_GRU_Gauss_root_softmax_path_regularized':
        g_ema = MyNewGenerator(args.size, prev_style_hidden_dim = args.prev_style_hidden_dim , w_style_dim = args.style_dim_for_changed_network,
        channel_multiplier=args.channel_multiplier,
        n_neurons_middle_layer = args.n_neurons_middle_layer,
        activation_for_style = args.activation_for_style,
        intermediate_activation = args.intermediate_activation
        ).to(device)

        generator = MyNewGenerator(args.size, prev_style_hidden_dim = args.prev_style_hidden_dim , w_style_dim = args.style_dim_for_changed_network,
        channel_multiplier=args.channel_multiplier,
        n_neurons_middle_layer = args.n_neurons_middle_layer,
        activation_for_style = args.activation_for_style
        ).to(device)
    elif args.arch=="multi_stage_Gaussian_affine":
        g_ema = MyNewGenerator(args.size, w_style_dim = args.style_dim_for_changed_network,
        channel_multiplier=args.channel_multiplier,
        n_neurons_middle_layer = args.n_neurons_middle_layer
        ).to(device)

        generator = MyNewGenerator(args.size, w_style_dim = args.style_dim_for_changed_network,
        channel_multiplier=args.channel_multiplier,
        n_neurons_middle_layer = args.n_neurons_middle_layer
        ).to(device)

    elif args.arch =='changed_mapping_network_with_3FC_reduced_RGB_style_dim_path_regularized' or args.arch=='hypernetwork_with_FC':
        g_ema = MyNewGenerator(args.size, w_style_dim = args.style_dim_for_changed_network,
        channel_multiplier=args.channel_multiplier,
        n_neurons_middle_layer = args.n_neurons_middle_layer
        ).to(device)

        generator = MyNewGenerator(args.size, w_style_dim = args.style_dim_for_changed_network,
        channel_multiplier=args.channel_multiplier,
        n_neurons_middle_layer = args.n_neurons_middle_layer
        ).to(device)

    elif args.arch=='hypernetwork_with_2_layer_FC_offset_middle_dim_512_32' or args.arch=='hypernetwork_offset_wth_FC_on_3_FC_middle_dim_512_32':
        g_ema = MyNewGenerator(args.size, w_style_dim = args.style_dim_for_changed_network,
        channel_multiplier=args.channel_multiplier,
        n_neurons_middle_layer = args.n_neurons_middle_layer,
        n_neurons_for_middle_layer_hypernet=args.n_neurons_for_middle_layer_hypernet
        ).to(device)

        generator = MyNewGenerator(args.size, w_style_dim = args.style_dim_for_changed_network,
        channel_multiplier=args.channel_multiplier,
        n_neurons_middle_layer = args.n_neurons_middle_layer,
        n_neurons_for_middle_layer_hypernet=args.n_neurons_for_middle_layer_hypernet
        ).to(device)



    else:
        g_ema = Generator(
            args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
        ).to(device)
        generator = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
       

    if args.use_spectral_norm_in_disc:
        print('Using spectral normalisation in the discriminator.')
        del Discriminator
        from discriminator_with_spectral_normalisation import spectral_normed_Discriminator
        discriminator = spectral_normed_Discriminator(
            args.size, channel_multiplier=args.channel_multiplier
        ).to(device)

    elif not args.use_spectral_norm_in_disc:

            # no change to discriminator whether training NewStylegan2 or not:
        discriminator = Discriminator(
            args.size, channel_multiplier=args.channel_multiplier
        ).to(device)

    g_ema.eval()

    
    accumulate(g_ema, generator, 0)

    g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
    d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)

    g_optim = optim.Adam(
        generator.parameters(),
        lr=args.lr * g_reg_ratio,
        betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
    )
    
    d_optim = optim.Adam(
        discriminator.parameters(),
        lr=args.lr * d_reg_ratio*args.discriminator_lr_reduction_factor,
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    )
    if args.ckpt is not None:
        print("load model:", args.ckpt)

        ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)

        try:
            ckpt_name = os.path.basename(args.ckpt)
            args.start_iter = int(os.path.splitext(ckpt_name)[0])

        except ValueError:
            pass

        generator.load_state_dict(ckpt["g"])
        discriminator.load_state_dict(ckpt["d"])
        g_ema.load_state_dict(ckpt["g_ema"])

        g_optim.load_state_dict(ckpt["g_optim"])
        d_optim.load_state_dict(ckpt["d_optim"])

    # if args.distributed:
    #     generator = nn.parallel.DistributedDataParallel(
    #         generator,
    #         device_ids=[args.local_rank],
    #         output_device=args.local_rank,
    #         broadcast_buffers=False,
    #     )

    #     discriminator = nn.parallel.DistributedDataParallel(
    #         discriminator,
    #         device_ids=[args.local_rank],
    #         output_device=args.local_rank,
    #         broadcast_buffers=False,
    #     )

    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )

    dataset = MultiResolutionDataset(args.path, transform, args.size)
    
    loader = data.DataLoader(
        dataset,
        batch_size=args.batch,
        sampler=data_sampler(dataset, shuffle=True, distributed= args.distributed),
        drop_last=True,
    )

    if get_rank() == 0 and wandb is not None and args.wandb:
        wandb.init(project="stylegan 2")

    train(args, loader, generator, discriminator, g_optim, d_optim, g_ema, device)