import argparse

import torch
from torch.nn import functional as F
import numpy as np
from tqdm import tqdm

import lpips
# from model import Generator


def normalize(x):
    return x / torch.sqrt(x.pow(2).sum(-1, keepdim=True))


def slerp(a, b, t):
    a = normalize(a)
    b = normalize(b)
    d = (a * b).sum(-1, keepdim=True)
    p = t * torch.acos(d)
    c = normalize(b - d * a)
    d = a * torch.cos(p) + c * torch.sin(p)

    return normalize(d)


def lerp(a, b, t):
    return a + (b - a) * t


if __name__ == "__main__":
    device = "cuda:1"

    parser = argparse.ArgumentParser(description="Perceptual Path Length calculator")
    parser.add_argument('--arch', type=str, default='stylegan2', help='model architectures (stylegan2 | swagan)')
    parser.add_argument(
        "--space", choices=["z", "w"], help="space that PPL calculated with"
    )
    parser.add_argument(
        "--batch", type=int, default=64, help="batch size for the models"
    )
    parser.add_argument(
        "--n_sample",
        type=int,
        default=5000,
        help="number of the samples for calculating PPL",
    )
    parser.add_argument(
        "--size", type=int, default=256, help="output image sizes of the generator"
    )
    parser.add_argument(
        "--eps", type=float, default=1e-4, help="epsilon for numerical stability"
    )
    parser.add_argument(
        "--crop", action="store_true", help="apply center crop to the images"
    )
    parser.add_argument(
        "--sampling",
        default="end",
        choices=["end", "full"],
        help="set endpoint sampling method",
    )
    parser.add_argument(
        "--ckpt", metavar="CHECKPOINT", help="path to the model checkpoints"
    )

    args = parser.parse_args()

    latent_dim = 512

    ckpt = torch.load(args.ckpt, map_location= device)

    print('Setting model hyperparameters.')

    if args.arch == 'changed_mapping_network_with_2FC_reduced_RGB_style_dim' or args.arch=="changed_mapping_network_style_proj_to_sphere" or args.arch=='changed_mapping_network_with_reduced_RGB_style_dim':
        args.use_g_regularization=False
        args.style_dim_for_changed_network = 32
        args.n_neurons_middle_layer =128
        args.latent = args.style_dim_for_changed_network*14
        args.model_input_tensor = True
        args.channel_multiplier = 2
    else:
        print('loading default network hyperparameters.')
        args.latent = 512
        args.n_mlp = 8
        args.model_input_tensor = False
        args.channel_multiplier = 2

    args.start_iter = 0
    if args.arch=="changed_mapping_network_style_proj_to_sphere":
        from Alvin_model_experiment_changed_mapping_network_with_style_projection_to_sphere_scaled import MyNewGenerator, Discriminator

    if args.arch=='changed_mapping_network_with_reduced_RGB_style_dim':
        from Alvin_model_experiment_changed_mapping_network_with_reduced_RGB_style_dim import MyNewGenerator, Discriminator

    if args.arch == 'changed_mapping_network_with_2FC_reduced_RGB_style_dim':
        from Alvin_model_experiment_changed_mapping_network_2FC_with_reduced_RGB_style_dim import MyNewGenerator, Discriminator

    if args.arch =="NewStylegan2_with_bias_spherical_projection_scaled_2nd_attempt" or args.arch =="NewStylegan2_with_bias_spherical_projection":
        from Alvin_model_experiment_with_style_projection_to_sphere_scaled import MyNewGenerator, Discriminator

    # if args.arch=="NewStylegan2_with_bias_spherical_projection":
    #     print('loading model from ', args.arch)
    #     from Alvin_model_experiment_with_style_projection_to_sphere import MyNewGenerator, Discriminator

    if args.arch == 'NewStylegan2':
        print('loading model from ', args.arch)
        from Alvin_model_experiment import MyNewGenerator, Discriminator

    if args.arch == 'NewStylegan2.2':
        print('loading model from ', args.arch)
        from Alvin_model_experiment2 import MyNewGenerator, Discriminator

    if args.arch == 'NewStylegan2_softmax_with_bias':
        print('loading model from ', args.arch)
        from Alvin_model_experiment_softmax_with_bias import MyNewGenerator, Discriminator

    if args.arch == "NewStylegan2_softmax_sqrt_with_bias":
        from Alvin_model_experiment_softmax_sqrt_with_bias import MyNewGenerator, Discriminator


    if args.arch == 'stylegan2_small_changes':
        print('loading model from ', args.arch)
        from model_small_changes import Generator, Discriminator

    if args.arch == 'stylegan2':
        print('loading model from ', args.arch)
        from model import Generator, Discriminator

    elif args.arch == 'swagan':
        print('loading model from ', args.arch)
        from swagan import Generator, Discriminator

    try:
        # args.latent = 512
        # args.n_mlp = 8
        # args.model_input_tensor = False
        g = Generator(
            args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
        ).to(device)
    except:
        try:
            # args.latent = 512
            # args.n_mlp = 8
            # args.model_input_tensor = False
            # args.channel_multiplier = 2
            g = MyNewGenerator(args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
            ).to(device)
        except:
            g= MyNewGenerator(args.size, style_dim = args.style_dim_for_changed_network,
                channel_multiplier=args.channel_multiplier,
                n_neurons_middle_layer = args.n_neurons_middle_layer
                ).to(device)


    # g = Generator(args.size, latent_dim, 8).to(device)
    g.load_state_dict(ckpt["g_ema"])
    g.eval()

    # percept = lpips.PerceptualLoss(
    # model="net-lin", net="vgg", use_gpu=device.startswith("cuda")
    # )

    percept = lpips.PerceptualLoss(
    model="net-lin", net="vgg", use_gpu=device.startswith("cuda")
    ).to(device)

   # percept.to(device)

    distances = []

    n_batch = args.n_sample // args.batch
    resid = args.n_sample - (n_batch * args.batch)
    batch_sizes = [args.batch] * n_batch + [resid]

    with torch.no_grad():
        for batch in tqdm(batch_sizes):
            noise = g.make_noise()

            inputs = torch.randn([batch * 2, latent_dim], device=device)
            if args.sampling == "full":
                lerp_t = torch.rand(batch, device=device)
            else:
                lerp_t = torch.zeros(batch, device=device)

            if args.space == "w":
                latent = g.get_latent(inputs)
                latent_t0, latent_t1 = latent[::2], latent[1::2]
                latent_e0 = lerp(latent_t0, latent_t1, lerp_t[:, None])
                latent_e1 = lerp(latent_t0, latent_t1, lerp_t[:, None] + args.eps)
                latent_e = torch.stack([latent_e0, latent_e1], 1).view(*latent.shape)

            image, _ = g([latent_e], input_is_latent=True, noise=noise)

            if args.crop:
                c = image.shape[2] // 8
                image = image[:, :, c * 3 : c * 7, c * 2 : c * 6]

            factor = image.shape[2] // 256

            if factor > 1:
                image = F.interpolate(
                    image, size=(256, 256), mode="bilinear", align_corners=False
                )

        
            dist = percept(image[::2], image[1::2]).view(image.shape[0] // 2) / (
                args.eps ** 2
            )
            distances.append(dist.to("cpu").numpy())

    distances = np.concatenate(distances, 0)

    lo = np.percentile(distances, 1, interpolation="lower")
    hi = np.percentile(distances, 99, interpolation="higher")
    filtered_dist = np.extract(
        np.logical_and(lo <= distances, distances <= hi), distances
    )

    print("ppl:", filtered_dist.mean())
