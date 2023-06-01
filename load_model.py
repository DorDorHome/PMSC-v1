# given the model name and all the required hyperparameters,
# load and outout the required model

import math


def load_g_ema(args):


    print('args.arch = ', args.arch)
    if args.arch =='hypernetwork_with_2_layer_FC_offset_middle_dim_512_32' or args.arch =='hypernetwork_offset_wth_FC_on_3_FC_middle_dim_512_32':
        print(f'setting parameters for {args.arch} ')
        args.style_dim_for_changed_network = 32
        args.n_neurons_middle_layer = 512
        args.n_neurons_for_middle_layer_hypernet = 32

        # num_input_layers_for_latent should set to 14 for 256. General formula is:
        args.num_input_layers_for_latent =  (int(math.log(args.size, 2)) -1)*2 # to calculate how many upsampling stages is needed. 
        
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
        ).to(args.device)

        generator = MyNewGenerator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
        ).to(args.device)

    elif  args.arch =='multi_stage_Gaussian_affine_with_activation' or  args.arch=='multi_stage_Gaussian_affine_with_double_activation':
        g_ema = MyNewGenerator(args.size, w_style_dim = args.style_dim_for_changed_network,
        channel_multiplier=args.channel_multiplier,
        n_neurons_middle_layer = args.n_neurons_middle_layer,
        activation_pre_affine = args.activation_pre_affine 
        ).to(args.device)

        generator = MyNewGenerator(args.size, w_style_dim = args.style_dim_for_changed_network,
        channel_multiplier=args.channel_multiplier,
        n_neurons_middle_layer = args.n_neurons_middle_layer,
        activation_pre_affine = args.activation_pre_affine 
        ).to(args.device)
    elif args.arch=='GRU_multistage_mapping_network_with_path_regularization' or args.arch=="multi_stage_GRU_skip_connection_root_softmax_with_path_regularization" or args.arch=='mutli_stage_GRU_with_root_softmax' or args.arch=="multi_stage_GRU_skip_connection_root_softmax":
        g_ema = MyNewGenerator(args.size, prev_style_hidden_dim = args.prev_style_hidden_dim , w_style_dim = args.style_dim_for_changed_network,
        channel_multiplier=args.channel_multiplier,
        n_neurons_middle_layer = args.n_neurons_middle_layer,
        activation_for_style = args.activation_for_style,
        ).to(args.device)

        generator =  MyNewGenerator(args.size, prev_style_hidden_dim = args.prev_style_hidden_dim , w_style_dim = args.style_dim_for_changed_network,
        channel_multiplier=args.channel_multiplier,
        n_neurons_middle_layer = args.n_neurons_middle_layer,
        activation_for_style = args.activation_for_style,
        ).to(args.device)
    elif args.arch=='multi_stage_GRU_Gauss_root_softmax' or args.arch=='multi_stage_GRU_Gauss_root_softmax_path_regularized':
        g_ema = MyNewGenerator(args.size, prev_style_hidden_dim = args.prev_style_hidden_dim , w_style_dim = args.style_dim_for_changed_network,
        channel_multiplier=args.channel_multiplier,
        n_neurons_middle_layer = args.n_neurons_middle_layer,
        activation_for_style = args.activation_for_style,
        intermediate_activation = args.intermediate_activation
        ).to(args.device)

        generator = MyNewGenerator(args.size, prev_style_hidden_dim = args.prev_style_hidden_dim , w_style_dim = args.style_dim_for_changed_network,
        channel_multiplier=args.channel_multiplier,
        n_neurons_middle_layer = args.n_neurons_middle_layer,
        activation_for_style = args.activation_for_style
        ).to(args.device)
    elif args.arch=="multi_stage_Gaussian_affine":
        g_ema = MyNewGenerator(args.size, w_style_dim = args.style_dim_for_changed_network,
        channel_multiplier=args.channel_multiplier,
        n_neurons_middle_layer = args.n_neurons_middle_layer
        ).to(args.device)

        generator = MyNewGenerator(args.size, w_style_dim = args.style_dim_for_changed_network,
        channel_multiplier=args.channel_multiplier,
        n_neurons_middle_layer = args.n_neurons_middle_layer
        ).to(args.device)

    elif args.arch =='changed_mapping_network_with_3FC_reduced_RGB_style_dim_path_regularized' or args.arch=='hypernetwork_with_FC':
        g_ema = MyNewGenerator(args.size, w_style_dim = args.style_dim_for_changed_network,
        channel_multiplier=args.channel_multiplier,
        n_neurons_middle_layer = args.n_neurons_middle_layer
        ).to(args.device)

        generator = MyNewGenerator(args.size, w_style_dim = args.style_dim_for_changed_network,
        channel_multiplier=args.channel_multiplier,
        n_neurons_middle_layer = args.n_neurons_middle_layer
        ).to(args.device)

    elif args.arch=='hypernetwork_with_2_layer_FC_offset_middle_dim_512_32' or args.arch=='hypernetwork_offset_wth_FC_on_3_FC_middle_dim_512_32':
        g_ema = MyNewGenerator(args.size, w_style_dim = args.style_dim_for_changed_network,
        channel_multiplier=args.channel_multiplier,
        n_neurons_middle_layer = args.n_neurons_middle_layer,
        n_neurons_for_middle_layer_hypernet=args.n_neurons_for_middle_layer_hypernet
        ).to(args.device)

        generator = MyNewGenerator(args.size, w_style_dim = args.style_dim_for_changed_network,
        channel_multiplier=args.channel_multiplier,
        n_neurons_middle_layer = args.n_neurons_middle_layer,
        n_neurons_for_middle_layer_hypernet=args.n_neurons_for_middle_layer_hypernet
        ).to(args.device)



    else:
        g_ema = Generator(
            args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
        ).to(args.device)
        generator = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(args.device)
               
    return g_ema
