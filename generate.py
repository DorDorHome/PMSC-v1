import argparse

import torch
from torchvision import utils
from tqdm import tqdm
import os

print('pytorch version in used is:', torch.__version__)

def generate(args, g_ema, device, mean_latent):

    with torch.no_grad():
        g_ema.eval()
        if args.from_loaded_latent:
            print('Only one image will be produced from the given latent.')
            args.pics = 1

        for i in tqdm(range(args.pics)):

            
            if args.from_loaded_latent:
                sample_z = torch.load(args.loaded_latent_path).to(device)
            else:
                sample_z = torch.randn(args.sample, args.latent, device=device)

            try:
                sample, _ = g_ema(
                    [sample_z], truncation=args.truncation, truncation_latent=mean_latent
                )
            except:
                sample, _ = g_ema(sample_z, truncation=args.truncation, truncation_correlated_latent=mean_latent
                )


            if not os.path.exists(f"sample/samples_from_{args.arch}_size_{args.size}_from_ckpt_{args.ckpt}"):
                os.makedirs(f"sample/samples_from_{args.arch}_size_{args.size}_from_ckpt_{args.ckpt}")
            utils.save_image(
                sample,
                f"sample/samples_from_{args.arch}_size_{args.size}_from_ckpt_{args.ckpt}/{str(i).zfill(6)}.png",
                nrow=4,
                normalize=True,
                range=(-1, 1),
            )
    


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

    parser.add_argument(
        "--sample",
        type=int,
        default=1,
        help="number of samples to be generated for each image",
    )
    parser.add_argument(
        "--pics", type=int, default=20, help="number of images to be generated"
    )
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
        "--from_loaded_latent",
        type=bool,
        default=False,
        help='whether to use user loaded latent instead of a randomly generated one.'
    )

    parser.add_argument(
        '--loaded_latent_path',
        type=str,
        default=None,
    )

    args = parser.parse_args()

    # load the correct model:

    if args.arch =="NewStylegan2_with_bias_spherical_projection_scaled":
        from Alvin_model_experiment_with_style_projection_to_sphere_scaled import MyNewGenerator, Discriminator

    if args.arch=="NewStylegan2_with_bias_spherical_projection":
        print('loading model from ', args.arch)
        from Alvin_model_experiment_with_style_projection_to_sphere import MyNewGenerator, Discriminator

    if args.arch == 'NewStylegan2':
        print('loading model from ', args.arch)
        from Alvin_model_experiment import MyNewGenerator, Discriminator

    if args.arch == 'NewStylegan2.2':
        print('loading model from ', args.arch)
        from Alvin_model_experiment2 import MyNewGenerator, Discriminator

    if args.arch == "NewStylegan2_softmax_sqrt_with_bias":
        from Alvin_model_experiment_softmax_sqrt_with_bias import MyNewGenerator, Discriminator

    if args.arch == 'NewStylegan2_softmax_with_bias':
        print('loading model from ', args.arch)
        from Alvin_model_experiment_softmax_with_bias import MyNewGenerator, Discriminator

    if args.arch == 'stylegan2_small_changes':
        print('loading model from ', args.arch)
        from model_small_changes import Generator, Discriminator

    if args.arch == 'stylegan2':
        print('loading model from ', args.arch)
        from model import Generator, Discriminator

    elif args.arch == 'swagan':
        print('loading model from ', args.arch)
        from swagan import Generator, Discriminator



    args.latent = 512
    args.n_mlp = 8



    print('args.arch = ', args.arch)

    if args.arch=='hypernetwork_offset_wth_FC_on_3_FC_middle_dim_512_32':
        from Alvin_model_experiment_hypernetwork_offset_with_FC_on_3_FC import MyNewGenerator, Discriminator

    if args.arch =='hypernetwork_with_2_layer_FC_offset_middle_dim_512_32' or args.arch =='hypernetwork_offset_wth_FC_on_3_FC_middle_dim_512_32':
        print(f'setting parameters for {args.arch} ')
        args.style_dim_for_changed_network = 32
        args.n_neurons_middle_layer = 512
        args.n_neurons_for_middle_layer_hypernet = 32

        args.num_input_layers_for_latent = 14 # (int(math.log(args.size, 2)) -2)*2 # to calculate how many upsampling stages is needed
        
        print('num_input_layers_for_latent', args.num_input_layers_for_latent )
        args.latent = args.style_dim_for_changed_network*args.num_input_layers_for_latent
        args.model_input_tensor = True

    if args.arch == "NewStylegan2_softmax_sqrt_with_bias" or args.arch=="NewStylegan2_with_bias_spherical_projection_scaled" or args.arch=="NewStylegan2_with_bias_spherical_projection" or args.arch == 'NewStylegan2' or args.arch == 'NewStylegan2.2' or args.arch == 'NewStylegan2_softmax_with_bias' or args.arch == 'stylegan2_small_changes':
        g_ema = MyNewGenerator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
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

    else:
        g_ema = Generator(
            args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
        ).to(device)
        
    print('loading checkpoint from: ', args.ckpt)
    checkpoint = torch.load(args.ckpt, map_location=device)

    g_ema.load_state_dict(checkpoint["g_ema"], strict = False)

    if args.truncation < 1:
        with torch.no_grad():
            mean_latent = g_ema.mean_latent(args.truncation_mean)
    else:
        mean_latent = None

    generate(args, g_ema, device, mean_latent)
