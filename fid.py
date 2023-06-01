import argparse
import pickle

import torch
from torch import nn
import numpy as np
from scipy import linalg
from tqdm import tqdm

# from model import Generator
from calc_inception import load_patched_inception_v3


@torch.no_grad()
def extract_feature_from_samples(
    generator, inception, truncation, truncation_latent, batch_size, n_sample, device
):
    n_batch = n_sample // batch_size
    resid = n_sample - (n_batch * batch_size)
    batch_sizes = [batch_size] * n_batch + [resid]
    features = []

    counter = 0
    for batch in tqdm(batch_sizes):
        #print('start batch number ', batch)
        latent = torch.randn(batch, 512, device=device)
        #print('random latent vector generated!')
        img, _ = g([latent], truncation=truncation, truncation_latent=truncation_latent)
        #print('forward pass of generator completed')
        #print('start Forward pass of inception.')
        feat = inception(img)[0].view(img.shape[0], -1)
        #print('Inception forward pass done! ')
        features.append(feat.to(device))
        counter += 1
        if counter%500 ==0:
            print(counter)

    print('type of features:',type(features))
    
    features = torch.cat(features, 0)
    print('featured extracted! ')

    return features


def calc_fid(sample_mean, sample_cov, real_mean, real_cov, eps=1e-6):
    cov_sqrt, _ = linalg.sqrtm(sample_cov @ real_cov, disp=False)

    if not np.isfinite(cov_sqrt).all():
        print("product of cov matrices is singular")
        offset = np.eye(sample_cov.shape[0]) * eps
        cov_sqrt = linalg.sqrtm((sample_cov + offset) @ (real_cov + offset))

    if np.iscomplexobj(cov_sqrt):
        if not np.allclose(np.diagonal(cov_sqrt).imag, 0, atol=1e-3):
            m = np.max(np.abs(cov_sqrt.imag))

            raise ValueError(f"Imaginary component {m}")

        cov_sqrt = cov_sqrt.real

    mean_diff = sample_mean - real_mean
    mean_norm = mean_diff @ mean_diff

    trace = np.trace(sample_cov) + np.trace(real_cov) - 2 * np.trace(cov_sqrt)

    fid = mean_norm + trace

    return fid


if __name__ == "__main__":
    #device = "cuda:1"
    device = "cpu"

    parser = argparse.ArgumentParser(description="Calculate FID scores")
    parser.add_argument('--arch', type=str, default='stylegan2', help='model architectures (stylegan2 | swagan)')
    parser.add_argument("--truncation", type=float, default=1, help="truncation factor")
    parser.add_argument(
        "--truncation_mean",
        type=int,
        default=4096,
        help="number of samples to calculate mean for truncation",
    )
    parser.add_argument(
        "--batch", type=int, default=64, help="batch size for the generator"
    )
    parser.add_argument(
        "--n_sample",
        type=int,
        default=50000,
        help="number of the samples for calculating FID",
    )
    parser.add_argument(
        "--size", type=int, default=256, help="image sizes for generator"
    )
    parser.add_argument(
        "--channel_multiplier",
        type=int,
        default=2,
        help="channel multiplier factor for the model. config-f = 2, else = 1",
    )
    parser.add_argument(
        "--inception",
        type=str,
        default=None,
        required=True,
        help="path to precomputed inception embedding",
    )
    parser.add_argument(
        "--ckpt", metavar="CHECKPOINT", help="path to generator checkpoint"
    )

    args = parser.parse_args()

    ckpt = torch.load(args.ckpt, map_location=device)

    print('Setting model hyperparameters.')

    if args.arch =='hypernetwork_with_2_layer_FC_offset_middle_dim_512_32' or 'hypernetwork_offset_wth_FC_on_3_FC_middle_dim_512_32':
        print(f'setting parameters for {args.arch} ')
        args.style_dim_for_changed_network = 32
        args.n_neurons_middle_layer = 512
        args.n_neurons_for_middle_layer_hypernet = 32

        args.num_input_layers_for_latent = 14 # (int(math.log(args.size, 2)) -2)*2 # to calculate how many upsampling stages is needed
        
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
        from swagan import Generator, DiscriminatorS

    # initialize the model: 
    if args.arch == 'NewStylegan2' or args.arch == "NewStylegan2_with_bias_spherical_projection_scaled_2nd_attempt" or args.arch == 'NewStylegan2_softmax_sqrt_with_bias':
        g = MyNewGenerator(
            args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
        ).to(device)

    elif  args.arch =='multi_stage_Gaussian_affine_with_activation' or  args.arch=='multi_stage_Gaussian_affine_with_double_activation':
        g = MyNewGenerator(args.size, w_style_dim = args.style_dim_for_changed_network,
        channel_multiplier=args.channel_multiplier,
        n_neurons_middle_layer = args.n_neurons_middle_layer,
        activation_pre_affine = args.activation_pre_affine 
        ).to(device)


    elif args.arch=='GRU_multistage_mapping_network_with_path_regularization' or args.arch=="multi_stage_GRU_skip_connection_root_softmax_with_path_regularization" or args.arch=='mutli_stage_GRU_with_root_softmax' or args.arch=="multi_stage_GRU_skip_connection_root_softmax":
        g = MyNewGenerator(args.size, prev_style_hidden_dim = args.prev_style_hidden_dim , w_style_dim = args.style_dim_for_changed_network,
        channel_multiplier=args.channel_multiplier,
        n_neurons_middle_layer = args.n_neurons_middle_layer,
        activation_for_style = args.activation_for_style,
        ).to(device)

    elif args.arch=='multi_stage_GRU_Gauss_root_softmax' or args.arch=='multi_stage_GRU_Gauss_root_softmax_path_regularized':
        g = MyNewGenerator(args.size, prev_style_hidden_dim = args.prev_style_hidden_dim , w_style_dim = args.style_dim_for_changed_network,
        channel_multiplier=args.channel_multiplier,
        n_neurons_middle_layer = args.n_neurons_middle_layer,
        activation_for_style = args.activation_for_style,
        intermediate_activation = args.intermediate_activation
        ).to(device)

    elif args.arch=="multi_stage_Gaussian_affine":
        g = MyNewGenerator(args.size, w_style_dim = args.style_dim_for_changed_network,
        channel_multiplier=args.channel_multiplier,
        n_neurons_middle_layer = args.n_neurons_middle_layer
        ).to(device)


    elif args.arch =='changed_mapping_network_with_3FC_reduced_RGB_style_dim_path_regularized' or args.arch=='hypernetwork_with_FC':
        g = MyNewGenerator(args.size, w_style_dim = args.style_dim_for_changed_network,
        channel_multiplier=args.channel_multiplier,
        n_neurons_middle_layer = args.n_neurons_middle_layer
        ).to(device)

    elif args.arch=='hypernetwork_with_2_layer_FC_offset_middle_dim_512_32' or args.arch=='hypernetwork_offset_wth_FC_on_3_FC_middle_dim_512_32':
        g = MyNewGenerator(args.size, w_style_dim = args.style_dim_for_changed_network,
        channel_multiplier=args.channel_multiplier,
        n_neurons_middle_layer = args.n_neurons_middle_layer,
        n_neurons_for_middle_layer_hypernet=args.n_neurons_for_middle_layer_hypernet
        ).to(device)



    else:
        g = Generator(
            args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
        ).to(device)


    g.load_state_dict(ckpt["g_ema"], strict = False)
    # g = nn.DataParallel(g)
    g.eval()

    print('model is setup successfully.')

    if args.truncation < 1:
        with torch.no_grad():
            mean_latent = g.mean_latent(args.truncation_mean)

    else:
        mean_latent = None

    # inception = nn.DataParallel(load_patched_inception_v3()).to(device)
    print('loading inception model.')
    inception = load_patched_inception_v3().to(device)

    inception.eval()
    print('inception loaded.')

    print('extracting feature from samples:')
    features = extract_feature_from_samples(
        g, inception, args.truncation, mean_latent, args.batch, args.n_sample, device
    ).to('cpu').numpy()
    print(f"extracted {features.shape[0]} features")

    sample_mean = np.mean(features, 0)
    sample_cov = np.cov(features, rowvar=False)

    print('feature extracted.')

    with open(args.inception, "rb") as f:
        embeds = pickle.load(f)
        real_mean = embeds["mean"]
        real_cov = embeds["cov"]

    print('calculating fid.')
    fid = calc_fid(sample_mean, sample_cov, real_mean, real_cov)

    print('finished calculating fid')
    print('\n')
    print("fid:", fid)
