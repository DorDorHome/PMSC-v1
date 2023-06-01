import argparse

import torch
from torchvision import utils
from tqdm import tqdm
import os
from torchvision import transforms
import torch.nn.functional as F
import math

print('pytorch version in used is:', torch.__version__)

# the followings code are needed to transform the output images:
from PIL import Image

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

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
        "--langevin_correction_constant",
        type= float,
        default = 0.5
    )

    parser.add_argument(
        "--perturbation_constant",
        type = float,
        default = 0.2
    )

    parser.add_argument(
        "--variance_regularizer",
        type = float,
        default = 0.1

    )

    args = parser.parse_args()
    args.random_seed = 1
    # load the correct model:

    print('Setting model hyperparameters.')

    print('args.arch = ', args.arch)
    # load the correct model:
    if args.arch=='hypernetwork_offset_wth_FC_on_3_FC_middle_dim_512_32':
        from Alvin_model_experiment_hypernetwork_offset_with_FC_on_3_FC import MyNewGenerator, Discriminator

    elif args.arch=='changed_mapping_network_with_3FC_reduced_RGB_style_dim_path_regularized':
        print('loading model from ', args.arch)
        from Alvin_model_experiment_changed_mapping_network_3FC_with_reduced_RGB_style_dim import MyNewGenerator, Discriminator
    
    elif args.arch == 'stylegan2':
        print('loading model from ', args.arch)
        from model import Generator, Discriminator

    # setting hyperparameters for the models:
    if args.arch =='hypernetwork_with_2_layer_FC_offset_middle_dim_512_32' or args.arch =='hypernetwork_offset_wth_FC_on_3_FC_middle_dim_512_32':
        print(f'setting parameters for {args.arch} ')
        args.style_dim_for_changed_network = 32
        args.n_neurons_middle_layer = 512
        args.n_neurons_for_middle_layer_hypernet = 32
        args.latent = args.style_dim_for_changed_network*14
        args.model_input_tensor = True

    elif args.arch =='stylegan2':
        print('using stylegan2 ')
        args.latent = 512
        args.n_mlp = 8
        args.model_input_tensor = False

    # initialize the model:
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

    else:
        g_ema = Generator(
            args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
        ).to(device)



    # first load the targe real image:
    target_real_img_index = 2
    target_image_path = 'target_images/real{}.jpg'.format(target_real_img_index)






    # 
    img_original_target= Image.open(target_image_path)
    #img_target = transforms.ToTensor()(img_original_target)

    #img_target = img_original_target.to(device).float()

    # Define a transform to convert PIL 
    # image to a Torch tensor
    transform = transforms.Compose([
    
        transforms.Resize(size= args.size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )
    
    # img_target = img_target.unsqueeze(0)

    # img_target = transforms.ToTensor()(img_original_target)
    print('type of img_original_target', type(img_original_target))
    # print('with size', img_original_target.size())
    # print('type of img_target', type(img_target))
    # print('with size', img_target.size() )
    # img_target = img_target.to(device).float()
    # print('type of img_target', type(img_target))
    # print('with size', img_target.size())

    
    #print('type of img target after unsqueeze:', type(img_target))
    
    img_target = transform(img_original_target)
    print('size of img_target, after transform', img_target.size())

    img_target = img_target[:, : , :].unsqueeze(0).to(device)
    # img_target = img_target.unsqueeze(0).to(device)

    # img_target = img_target.unsqueeze(0)


    print('size of img target after transform: ',img_target.size())
    print('device of img_target: ', img_target.device)
    print('img_target requires grad:,', img_target.requires_grad)






    # if args.truncation < 1:
    #     with torch.no_grad():
    #         mean_latent = g_ema.mean_latent(args.truncation_mean)
    # else:
    #     mean_latent = None

    print('loading checkpoint from: ', args.ckpt)
    checkpoint = torch.load(args.ckpt, map_location=device)

    g_ema.load_state_dict(checkpoint["g_ema"], strict = False)
    g_ema.eval()

    torch.manual_seed(args.random_seed)
    step = int(math.log(args.size, 2)) - 2  

    # initiate a random latent that is trainable:
    z_trainable  = torch.randn(args.sample, args.latent,
                                requires_grad = True,
                                  device = device)
    z_original = z_trainable.clone().detach()
    z_original.requires_grad_(False)

    print('z_original requires_grad:', z_original.requires_grad)
    print('z_trainable requires grad:', z_trainable.requires_grad)

    # initiate a random noise component that is trainable:


    # noise_original = [getattr(g_ema.noises, f"noise_{i}") for i in range(g_ema.num_layers)
    #             ]

    # print('noise_original[0] size:', noise_original[0].size())
    

    # noise_trainable = []

    # for i in range(len(noise_original)):
    #     noise = noise_original[i].clone().detach().to(device)
    #     noise.requires_grad_(requires_grad=True)
    #     noise_trainable.append(noise)




    mean_latent = None

    if args.arch =='hypernetwork_with_2_layer_FC_offset_middle_dim_512_32' or args.arch =='hypernetwork_offset_wth_FC_on_3_FC_middle_dim_512_32':
        img_before_training, _ = g_ema(z_trainable, truncation=args.truncation, truncation_correlated_latent=mean_latent,
            noise = None #noise_original
            )
    else:
        img_before_training, _ = g_ema([z_trainable], truncation=args.truncation, truncation_latent=mean_latent,
            noise = None #noise_original
            )


    print('type of img_before_training:', type(img_before_training))
    print('size of img_before_training: ', img_before_training.size())
    print(type(img_before_training))
    if not os.path.exists('projection_to_latent_space_with_langevin_results/find_latent_for_real_images_results/{}_with_ckpt{}/langevin_correction_{}_var_reg_{}'.format(args.arch, args.ckpt, args.langevin_correction_constant, args.variance_regularizer)):
        os.makedirs('projection_to_latent_space_with_langevin_results/find_latent_for_real_images_results/{}_with_ckpt{}/langevin_correction_{}_var_reg_{}'.format(args.arch, args.ckpt, args.langevin_correction_constant, args.variance_regularizer))
    
    utils.save_image(img_before_training, 'projection_to_latent_space_with_langevin_results/find_latent_for_real_images_results/{}_with_ckpt{}/langevin_correction_{}_var_reg_{}/intial_before_training_targeting{}_seed{}.png'.format(args.arch, args.ckpt, args.langevin_correction_constant, args.variance_regularizer, target_real_img_index ,args.random_seed),
                                 normalize=True,
            range=(-1, 1),
            )
                     

    # needs to change the pbar variable to increase number of epochs:
    # start training:
    epochs = 1000000
    pbar = tqdm(range(epochs))


    # freezing the parameters of generator:
    requires_grad(g_ema, False)

    # defining the optimizer to be used:
    
    learning_rate = 0.005


    optimizer = torch.optim.SGD([z_trainable], lr = learning_rate, weight_decay= args.langevin_correction_constant*learning_rate)
    one = torch.tensor([1.0 ], device = device)
    for epoch in pbar:

            
        if args.arch =='hypernetwork_with_2_layer_FC_offset_middle_dim_512_32' or args.arch =='hypernetwork_offset_wth_FC_on_3_FC_middle_dim_512_32':
            img_generated, _ = g_ema(z_trainable, truncation=args.truncation, truncation_correlated_latent=mean_latent,
                noise = None# noise_trainable
                )
        else:
            img_generated, _ = g_ema([z_trainable], truncation=args.truncation, truncation_latent=mean_latent,
                noise = None #noise_trainable
                )

        loss = F.mse_loss(img_generated, img_target) + args.variance_regularizer*F.mse_loss(torch.linalg.norm(z_trainable, ord='fro'), one )

        # get gradients:
        loss.backward()

        # update weights in the image:
        optimizer.step()
        optimizer.zero_grad()

        # finally, correcting for the deviation from gaussian with Langevin dynamics:
        # z_trainable = z_trainable - args.langevin_correction_constant*learning_rate*z_trainable + \
        #     args.perturbation_constant*args.langevin_correction_constant*learning_rate*torch.randn(args.sample, args.latent,
        #                         requires_grad = True,
        #                           device = device)


        if epoch % 5000 ==0:
                  
            print('saving epoch {}'.format(epoch))
            if args.arch =='hypernetwork_with_2_layer_FC_offset_middle_dim_512_32' or args.arch =='hypernetwork_offset_wth_FC_on_3_FC_middle_dim_512_32':
                latest_img, _ = g_ema(z_trainable, truncation=args.truncation, truncation_correlated_latent=mean_latent,
                    noise = None # noise_trainable
                    )
            else:
                latest_img, _ = g_ema([z_trainable], truncation=args.truncation, truncation_latent=mean_latent,
                    noise =None # noise_trainable
                )
            # print(type(latest_img))
            # print(latest_img.size())
            utils.save_image(latest_img, 'projection_to_latent_space_with_langevin_results/find_latent_for_real_images_results/{}_with_ckpt{}/langevin_correction_{}_var_reg_{}/results_after_{}_epoch_targeting{}_seed{}.png'.format(args.arch, args.ckpt, args.langevin_correction_constant, args.variance_regularizer, epoch,target_real_img_index ,args.random_seed),
                            normalize=True,
                            range=(-1, 1),
            )  


        # for checking that the mdoel wasn't trained:
        # if epoch % 5000 ==0:
        #     if args.arch =='hypernetwork_with_2_layer_FC_offset_middle_dim_512_32' or args.arch =='hypernetwork_offset_wth_FC_on_3_FC_middle_dim_512_32':
        #         img_before_training, _ = g_ema(z_original, truncation=args.truncation, truncation_correlated_latent=mean_latent,
        #             noise = None# noise_original
        #             )
        #     else:
        #         img_before_training, _ = g_ema([z_original], truncation=args.truncation, truncation_latent=mean_latent,
        #             noise = None # noise_original
        #             )
            
        #     utils.save_image(img_before_training, \
        #             'projection_to_latent_space_with_langevin_results/find_latent_for_real_images_results/{}_with_ckpt{}/langevin_correction_{}_var_reg_{}/reality_check_after_{}_epoch_targeting{}_seed{}.png'.format(args.arch, args.ckpt, args.langevin_correction_constant, args.variance_regularizer, epoch,target_real_img_index ,args.random_seed),
        #         normalize=True,
        #         range=(-1, 1),
        #     )  


        
        if epoch % 25000 ==0 or epoch == epochs-1:
            torch.save(z_trainable,'projection_to_latent_space_with_langevin_results/find_latent_for_real_images_results/{}_with_ckpt{}/langevin_correction_{}_var_reg_{}/z_latent_after_{}_epoch_targeting{}_seed{}.npy'.format(args.arch, args.ckpt, args.langevin_correction_constant, args.variance_regularizer, epoch,target_real_img_index ,args.random_seed ))
            #torch.save(noise_trainable,'projection_to_latent_space_with_langevin_results/find_latent_for_real_images_results/{}_with_ckpt{}/trained_noise_after_{}_epoch_targeting{}_seed{}.npy'.format(args.arch, args.ckpt, epoch,target_real_img_index ,args.random_seed ))
                             
                             
                             
                #finally, save the trained latent vector:
            # noise_cpu_copy =  [noise.clone().cpu().detach().numpy() for noise in noise_trainable]
            
            #torch.save(noise_trainable,'projection_to_latent_space_results/find_latent_for_real_images_results/latent_after_{}_epoch_targeting{}_seed{}.npy'.format(epoch,target_real_img_index ,seed))

