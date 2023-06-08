environment:
use stylegan2, cloned from op.conv2d_gradfix import conv2d_gradfix
from stylegan3, added torchvision, tensorflow version 1.14 or 1.15 is required

created another one for better use of conv2d_gradfix:
stylegan2_torch_1_8
first cloned stylegan2
then, uninstall torch 1.9

removed:
  pytorch-1.9.1-py3.9_cuda11.1_cudnn8.0.5_0
  torchvision-0.2.2-py_3




__________________________



_____

folders created:

dataset

create lmdb datasets:

python prepare_data.py --out LMDB_PATH --n_worker N_WORKER --size SIZE1,SIZE2,SIZE3,... DATASET_PATH

edited:

create lmdb datasets:

python prepare_data.py --out LMDB_PATH --n_worker 2 dataset/ffhq

# create cat lmdb datasets:

python prepare_data.py --out LMDB_cat --n_worker 2 /home/sfchan/Desktop/Datasets/cat_single_folder

# create afhq cat datasets:

python prepare_data.py --out LMDB_AFHQ --n_work 2 /hdda/datasets/AFHQ/afhq/train

# create cow dataset:


python prepare_data.py --out LMDB_lsun_cow --n_work 2 /hdda/datasets/lsun_cow_single_folder


# codes for moving files from current folders and subfolders into another folder, without the subfolder structrue:

find -type f -exec mv "{}" /hdda/datasets/lsun_cow_single_folder \;
___________________
# training the original stylegan2:

python train.py \
--arch=stylegan2 \
--batch 8 LMDB_PATH   

# distributed:
python -m torch.distributed.launch --nproc_per_node=N_GPU --master_port=PORT train.py --batch 8 LMDB_PATH
_____________________________________________________________________

# creation of a alternative models

# experiments with new layer:
Alvin_model_experiment.py




# training the new model:
# please adjust 



# second training. 





# this is the initial training hyperparameters s   return True
--batch 8 LMDB_PATH   


python train.py \
--arch=NewStylegan2.2 \
--batch 8 LMDB_PATH   


python train.py \
--arch="NewStylegan2_with_bias_spherical_projection" \
--batch 8 LMDB_PATH   

python train.py \
--arch="NewStylegan2_with_bias_spherical_projection_scaled_2nd_attempt" \
--batch 32 LMDB_PATH   

# continue training:

python train.
py \
--arch="NewStylegan2_with_bias_spherical_projection_scaled_2nd_attempt" \
--ckpt=NewStylegan2_with_bias_spherical_projection_scaled_2nd_attempt_checkpoint/165000.pt \
--batch 32 LMDB_PATH   


python train.py \
--arch='changed_mapping_network_with_reduced_RGB_style_dim' \
--batch 8 LMDB_PATH   


python train.py \
--arch='changed_mapping_network_with_2FC_reduced_RGB_style_dim' \
--batch 32 LMDB_PATH   

python train.
py \
--arch='NewStylegan2_softmax_sqrt_with_bias' \
--batch 16 LMDB_PATH  


# train the multi Gaussian affine:
# 
python train.py \
--arch="multi_stage_Gaussian_affine" \
--batch 16 LMDB_PATH

python train.py \
--arch='multi_stage_Gaussian_affine_with_activation' \
--activation_pre_affine='tanh' \
--batch 16 LMDB_PATH

python train.py \
--arch='multi_stage_Gaussian_affine_with_double_activation' \
--activation_
pre_affine='tanh' \
--batch=16 LMDB_PATH

python train.py \
--arch='mutli_stage_GRU_with_root_softmax' \
--activation_for_style='sqrt_softmax' \
--prev_style_hidden_dim=256 \
--batch=16 LMDB_PATH 
#--ckpt=mutli_stage_GRU_with_root_softmax _checkpoint/045000.pt\

python train.py \
--arch='multi_stage_GRU_Gauss_root_softmax' \
--activation_for_style='sqrt_softmax' \
--intermediate_activation=None \
--prev_style_hidden_dim=256 \
--batch=16 LMDB_PATH


??:
python train.py \
--arch='multi_stage_GRU_Gauss_root_softmax' \
--use_g_regularization=False \
--activation_for_style='sqrt_softmax' \
--intermediate_activation='tanh' \
--prev_style_hidden_dim=256 \
--batch=16 LMDB_PATH \
--ckpt='multi_stage_GRU_Gauss_root_softmax _checkpoint/033000.pt'


python train.py \
--arch="multi_stage_GRU_skip_connection_root_softmax_with_path_regularization" \
--activation_for_style='sqrt_softmax' \
--prev_style_hidden_dim=256 \
--batch=16 LMDB_PATH \
--clamp_parameter_values=True 
# --ckpt='multi_stage_GRU_skip_connection_root_softmax_with_path_regularization _checkpoint/0
15000.pt'

python train.py \
--arch='GRU_multistage_mapping_network_with_path_regularization' \
--activation_for_style='sqrt_softmax' \
--prev_style_hidden_dim=256 \
--batch=16 LMDB_PATH \
--ckpt='GRU_multistage_mapping_network_with_path_regularization _checkpoint/006000.pt'





python train.py \
--arch='multi_stage_GRU_Gauss_root_softmax_path_regularized' \
--activation_for_style='sqrt_softmax' \
--intermediate_activation=None \
--prev_style_
hidden_dim=256 \
--batch=16 LMDB_PATH \
--use_g_regularization=True \
--ckpt='multi_stage_GRU_Gauss_root_softmax_path_regularized _checkpoint/045000.pt'


# initial trial with 3FC in initial layer
# no change in activation_for_style
# can be changed to decomposable style block for more flexibility:

# this is with path_regularization
python train.py \
--arch='changed_mapping_network_with_3FC_reduced_RGB_style_dim_path_regularized' \
--use_g_regularization=True \
--batch 16 LMDB_PATH \
--ckpt='changed_mapping_network_with_3FC_reduced_RGB_style_dim_path_regularized _checkpoint/498000.pt'  # batch size 32 is too large for 1 gpu


# hypernetwork, with path regularization:

python train.py \
--arch='hypernetwork_with_FC' \
--use_g_regularization=True \
--batch 16 LMDB_PATH

#hypernetwork, implemented with offset, with path regulariation:

python train.py \
--arch='hypernetwork_with_2_layer_FC_offset_middle_dim_512_32' \
--use_g_regularization=True \
--batch 16 LMDB_PATH \
--ckpt='hypernetwork_with_2_layer_FC_offset_middle_dim_512_32 _checkpoint/003000.pt'

# hypernetwork, on the 3-layered FC of each level:

python train.py \
--arch='hypernetwork_offset_wth_FC_on_3_FC_middle_dim_512_32' \
--use_g_regularization=True \
--batch 16 LMDB_PATH 

# hypernetwork, on the 3-layered FC of each level, g_path_regulaize from z
python train.py \
--arch='hypernetwork_offset_wth_FC_on_3_FC_middle_dim_512_32' \
--use_g_regularization=True \
--g_path_regularize_from_z=True \
--iter=2400000 \
--ckpt='hypernetwork_offset_wth_FC_on_3_FC_middle_dim_512_32__trained_on_LMDB_PATH_use_spectral_False_train_with_unif_False_path_reg_from_z_True_256_checkpoint/0220000.pt' \
--batch 16 LMDB_PATH 

# hypernetwork, on the 3-layered FC of each level, trained with cat:

python train.py \
--arch='hypernetwork_offset_wth_FC_on_3_FC_middle_dim_512_32' \
--use_g_regularization=True \
--g_path_regularize_from_z=True \
--iter=2400000 \
--batch 16 /home/sfchan/Desktop/Datasets/lsun/cat


# hypernetwork, on the 3-layered FC of each level, trained with AFHQ:


python train.py \
--arch='hypernetwork_offset_wth_FC_on_3_FC_middle_dim_512_32' \
--use_g_regularization=True \
--g_path_regularize_from_z=True \
--iter=800000 \
--ckpt='hypernetwork_offset_wth_FC_on_3_FC_middle_dim_512_32__trained_on_LMDB_AFHQ_use_spectral_False_train_with_unif_False_path_reg_from_z_True_256_checkpoint/0130000.pt' \
--batch 16 LMDB_AFHQ

# hypernetwork, on the 3-layered FC of each level, trained with lsun bedroom:

python train.py \
--arch='hypernetwork_offset_wth_FC_on_3_FC_middle_dim_512_32' \
--use_g_regularization=True \
--g_path_regularize_from_z=True \
--iter=800000 \
--batch 16 LMDB_bedroom_train

# hypernetwork, on the 3-layered FC of each level, trained with lsun bedroom train set:
python train.py \
--arch='hypernetwork_offset_wth_FC_on_3_FC_middle_dim_512_32' \
--use_g_regularization=True \
--iter=2400000 \
--batch 16 LMDB_bedroom_train

# hypernetwork, on the 3-layered FC of each level, trained to high res directly from python train.py, path_regularization from z space \
python train.py \
--arch='hypernetwork_offset_wth_FC_on_3_FC_middle_dim_512_32' \
--use_g_regularization=True \
--g_path_regularize_from_z=True \
--batch 4 LMDB_PATH \
--n_sample=25 \
--size=1024 \
--iter=2400000 \
#--discriminator_lr_reduction_factor=0.05

# hypernetwork, on the 3-layered FC of each level, trained to high res directly from python train.py, with spectral normalisation in the discriminator
python train.py \
--use_spectral_norm_in_disc=True \
--arch='hypernetwork_offset_wth_FC_on_3_FC_middle_dim_512_32' \
--use_g_regularization=True \
--g_path_regularize_from_z=True \
--batch 4 LMDB_PATH \
--n_sample=25 \
--size=1024 \
--iter=2400000 \
--discriminator_lr_reduction_factor=0.1

# training from the original stylegan2 ckpt:
# we only use the g part of the original
# training and saving only the 
python train_new_mapping_from_trained_stylegan.py \
--arch_of_original_model='stylegan2' \
--ckpt_of_original_model='checkpoint_original_stylegan2/stylegan2-ffhq-config-f.pt' \
--arch='hypernetwork_offset_wth_FC_on_3_FC_middle_dim_512_32_high_res' \
--n_sample=25 \
--use_g_regularization=True \
--size=1024 \
--batch 4 LMDB_PATH \
--iter=2400000 \
--ckpt='from_pretrained_high_res_hypernetwork_offset_wth_FC_on_3_FC_middle_dim_512_32_high_res _checkpoint/030000.pt' \

 

accelerate launch train_new_mapping_from_trained_stylegan.py \
--arch_of_original_model='stylegan2' \
--ckpt_of_original_model='checkpoint_original_stylegan2/stylegan2-ffhq-config-f.pt' \
--arch='hypernetwork_offset_wth_FC_on_3_FC_middle_dim_512_32_high_res' \
--n_sample=25 \
--use_g_regularization=True \
--size=1024 \
--batch 4 LMDB_PATH \
--iter=2400000 \
--ckpt='from_pretrained_high_res_hypernetwork_offset_wth_FC_on_3_FC_middle_dim_512_32_high_res _checkpoint/030000.pt'

# using the original 3 FC offset model for high res:
python \
train_new_mapping_from_trained_stylegan.py \
--arch_of_original_model='stylegan2' \
--ckpt_of_original_model='checkpoint_original_stylegan2/stylegan2-ffhq-config-f.pt' \
--arch='hypernetwork_offset_wth_FC_on_3_FC_middle_dim_512_32' \
--n_sample=25 \
--use_g_regularization=True \
--size=1024 \
--batch 4 LMDB_PATH \
--iter=2400000

accelerate launch train_new_mapping_from_trained_stylegan.py \
--arch_of_original_model='stylegan2' \
--ckpt_of_original_model='checkpoint_original_stylegan2/stylegan2-ffhq-config-f.pt' \
--arch='hypernetwork_offset_wth_FC_on_3_FC_middle_dim_512_32' \
--n_sample=25 \
--use_g_regularization=True \
--size=1024 \
--batch 4 LMDB_PATH \
--iter=2400000


# for debugging, check whether we can run on the original:

python train_new_mapping_from_trained_stylegan.py \
--arch_of_original_model='stylegan2' \
--ckpt_of_original_model='checkpoint_original_stylegan2/stylegan2-ffhq-config-f.pt' \
--arch='stylegan2' \
--use_g_regularization=True \
--size=1024 \
--batch 2 LMDB_PATH \
--use_accelerate=True \
--cpkt = 








________________________________________________________



# train model in distributed settings

# python -m torch.distributed.launch --nproc_per_node=N_GPU --master_port=PORT train.py --batch BATCH_SIZE LMDB_PATH

# after changing arguments:

# python -m torch.distributed.launch --nproc_per_node=2 --master_port=1 train.py --batch 8 LMDB_PATH

# try using torch.distributed.run instead (doesn't work either)

# python -m torch.distributed.run --nproc_per_node=2 --master_port=1 train.py --batch 8 LMDB_PATH OMP_NUM_THREADS=2

Try not using distributed trainin at all (work! )

python train.py --batch 8 LMDB_PATH   -ckpt='hypernetwork_with_2_layer_FC_offset_middle_dim_512_32 _checkpoint/003000.pt'
________________________________________________________
Training: 

# train model in distributed settings

# python -m torch.distributed.launch --nproc_per_node=N_GPU --master_port=PORT train.py --batch BATCH_SIZE LMDB_PATH

# after changing arguments:

# python -m torch.distributed.launch --nproc_per_node=2 --master_port=1 train.py --batch 8 LMDB_PATH

# try using torch.distributed.run instead (doesn't work either)

# python -m torch.distributed.run --nproc_per_node=2 --master_port=1 train.py --batch 8 LMDB_PATH OMP_NUM_THREADS=2

Try not using distributed trainin at all (work! )

python train.py --batch 8 LMDB_PATH   



--batch 16 LMDB_PATH 
________________________________________________
train the changed_mapping_network

the following needs to be adjusted:
style_dim (changed within train.py)
args.model_input_tensor = True (changed within train.py)

python train.py --arch=changed_mapping_network_style_proj_to_sphere --batch 8 LMDB_PATH
________________________________________________________

# to do continue training:

python train.py 





Convert weight from official checkpoints:

python convert_weight.py --repo ~/stylegan2 stylegan2-ffhq-config-f.pkl


________________________
# temp: for debug only 
python train.py \
--arch="NewStylegan2_with_bias_spherical_projection_scaled_2nd_attempt" \
--ckpt=NewStylegan2_with_bias_spherical_projection_scaled_2nd_attempt_checkpoint/165000.pt \
--batch 32 LMDB_PATH   



______________________________________________
generate samples:

# New Stylegan2 with bias, spherical projection scaled:
python generate.py --arch NewStylegan2_with_bias_spherical_projection_scaled --size 256 --sample 10 --pics 4 --ckpt NewStylegan2_with_bias_spherical_projection_scaled_checkpoint/570000.pt

# original stylegan2, low res;
python generate.py --size 256 --sample 4 --pics 4 --ckpt checkpoint_original_stylegan2/550000.pt

# original stylegan2, low res, trained by Alvin:
python generate.py --size 256 --sample 4 --pics 4 --ckpt checkpoint_original_stylegan2/470000.pt

# the original, high res one:

python generate.py --sample 4 --pics 4 --ckpt checkpoint_original_stylegan2/stylegan2-ffhq-config-f.pt
python generate.py --sample 4 --pics 4 --ckpt checkpoint_original_stylegan2/550000.pt

python generate.py --arch stylegan2 --sample 4 --pics 4 --ckpt checkpoint_original_stylegan2/470000.pt

python generate.py --sample 4 --pics 4 --ckpt checkpoint/380000.pt


# didn't work initially. Changed the generate.py file, line 76 to g_ema.load_state_dict(checkpoint["g_ema"], strict = False)

python generate.py  --arch NewStylegan2_with_bias_spherical_projection_scaled --size 256 --sample 12 --pics 4 --ckpt NewStylegan2_with_bias_spherical_projection_scaled_checkpoint/450000.pt

python train.py --arch=changed_mapping_network_style_proj_to_sphere --batch 8 LMDB_PATH
________________________________________________________

# to do continue training:

python train.py 





Convert weight from official checkpoints:

python convert_weight.py --repo ~/stylegan2 stylegan2-ffhq-config-f.pkl


________________________
# temp: for debug only 
python train.py \
--arch="NewStylegan2_with_bias_spherical_projection_scaled_2nd_attempt" \
--ckpt=NewStylegan2_with_bias_spherical_projection_scaled_2nd_attempt_checkpoint/165000.pt \
--batch 32 LMDB_PATH   



______________________________________________
generate samples:

# New Stylegan2 with bias, spherical projection scaled:
python generate.py --arch NewStylegan2_with_bias_spherical_projection_scaled --size 256 --sample 10 --pics 4 --ckpt NewStylegan2_with_bias_spherical_projection_scaled_checkpoint/570000.pt

# original stylegan2, low res;
python generate.py --size 256 --sample 4 --pics 4 --ckpt checkpoint_original_stylegan2/550000.pt

# original stylegan2, low res, trained by Alvin:
python generate.py --size 256 --sample 4 --pics 4 --ckpt checkpoint_original_stylegan2/470000.pt

# the original, high res one:
python generate.py --sample 4 --pics 4 --ckpt checkpoint_original_stylegan2/stylegan2-ffhq-config-f.pt

python generate.py --sample 4 --pics 4 --ckpt checkpoint_original_stylegan2/550000.pt

python generate.py --arch stylegan2 --sample 4 --pics 4 --ckpt checkpoint_original_stylegan2/470000.pt

python generate.py --sample 4 --pics 4 --ckpt checkpoint/380000.pt


# didn't work initially. Changed the generate.py file, line 76 to g_ema.load_state_dict(checkpoint["g_ema"], strict = False)

python generate.py  --arch NewStylegan2_with_bias_spherical_projection_scaled --size 256 --sample 12 --pics 4 --ckpt NewStylegan2_with_bias_spherical_projection_scaled_checkpoint/450000.pt


python generate.py --size 256 --sample 12 --pics 4 --ckpt stylegan2_checkpoint/450000.pt

python generate.py  --arch NewStylegan2_softmax_sqrt_with_bias --size 256 --sample 12 --pics 4 --ckpt NewStylegan2_softmax_sqrt_with_bias_checkpoint/390000.pt


______________________________________________________
# calculating FID:

# step 1: calculate mean and covariance of real images:

calc_inception.py

# this create a pickled file containing the mean, cov 

python calc_inception.py --batch 8 LMDB_PATH

# step 2: calculate the FID, referencing the real image 

# for my new model:(change the cpkt argument)
python fid.py --batch 8 --size 256--inception LMDB_PATH --ckpt NewStylegan2_with_bias_spherical_projection_scaled_checkpoint/705000.pt

python fid.py --arch NewStylegan2_with_bias_spherical_projection --batch 16 --size 256 --inception inception_ffhq.pkl --ckpt NewStylegan2_with_bias_spherical_projection_scaled_checkpoint/555000.pt --n_sample 53000

result = 6.844

python fid.py --arch NewStylegan2_with_bias_spherical_projection --batch 16 --size 256 --inception inception_ffhq.pkl --ckpt NewStylegan2_with_bias_spherical_projection_scaled_checkpoint/705000.pt --n_sample 53000

result = 6.55

python fid.py --arch NewStylegan2_with_bias_spherical_projection --batch 16 --size 256 --inception inception_ffhq.pkl --ckpt NewStylegan2_with_bias_spherical_projection_scaled_checkpoint/690000.pt --n_sample 51500

result = 6.52

python fid.py --arch NewStylegan2_with_bias_spherical_projection --batch 16 --size 256 --inception inception_ffhq.pkl --ckpt NewStylegan2_with_bias_spherical_projection_scaled_checkpoint/450000.pt --n_sample 51500

FID result = 7.14

python fid.py --arch NewStylegan2_with_bias_spherical_projection --batch 8 --size 256 --inception inception_ffhq.pkl --ckpt NewStylegan2_with_bias_spherical_projection_scaled_checkpoint/450000.pt --n_sample 51500

FID result = 7.24

python ppl.py --arch NewStylegan2_with_bias_spherical_projection --space w --batch 32 --size 256 --ckpt NewStylegan2_with_bias_spherical_projection_scaled_checkpoint/450000.pt --n_sample 5000

PPL result = 224.5


# for sqrt of softmax: 

python fid.py --arch NewStylegan2_softmax_sqrt_with_bias --batch 8 --size 256 --inception inception_ffhq.pkl --ckpt NewStylegan2_softmax_sqrt_with_bias_checkpoint/375000.pt --n_sample 51500

FID result (at 375000) = 6.22

python ppl.py --arch NewStylegan2_softmax_sqrt_with_bias --space w --batch 32 --size 256 --ckpt NewStylegan2_softmax_sqrt_with_bias_checkpoint/375000.pt --n_sample 5000

PPL result (at 375000) = 352.4

python fid.py --arch NewStylegan2_softmax_sqrt_with_bias --batch 8 --size 256 --inception inception_ffhq.pkl --ckpt NewStylegan2_softmax_sqrt_with_bias_checkpoint/390000.pt --n_sample 51500

FID result (at 390000) =

# 2nd attempt of spherical projection:

python ppl.py --arch NewStylegan2_with_bias_spherical_projection_scaled_2nd_attempt --space w --batch 32 --size 256 --ckpt NewStylegan2_with_bias_spherical_projection_scaled_2nd_attempt/380000.pt --n_sample 5000





# for the original model trained by the rosinality:

python fid.py --arch stylegan2 --batch 8 --size 256 --inception inception_LMDB_PATH.pkl --ckpt checkpoint_original_stylegan2/470000.pt


python fid.py --arch stylegan2 --batch 8 --size 256 --inception inception_LMDB_PATH.pkl --ckpt checkpoint_original_stylegan2/470000.pt --n_sample 9

python fid.py --arch stylegan2 --batch 8 --size 256 --inception inception_ffhq.pkl --ckpt checkpoint_original_stylegan2/550000.pt --n_sample 51500

result = 4.50

python fid.py --arch stylegan2 --batch 8 --size 256 --inception inception_ffhq.pkl --ckpt stylegan2_checkpoint/550000.pt --n_sample 51500

result = 4.50

# for the orignal model retrained by Alvin:

python fid.py --arch stylegan2 --batch 8 --size 256 --inception inception_ffhq.pkl --ckpt stylegan2_checkpoint/255000.pt --n_sample 51500

FID = 8.69

python fid.py --arch stylegan2 --batch 8 --size 256 --inception inception_ffhq.pkl --ckpt stylegan2_checkpoint/450000.pt --n_sample 51500

FID = 6.86

PPL:

python ppl.py --arch stylegan2 --space w --batch 32 --size 256 --ckpt stylegan2_checkpoint/450000.pt --n_sample 5000

PPL = 337.2 (lower is better)



# for the softmax model:

python fid.py --arch NewStylegan2_softmax_with_bias --batch 16 --size 256 --inception inception_ffhq.pkl --ckpt NewStylegan2_softmax_with_bias_checkpoint/555000.pt --n_sample 51500

python load_and_examine_models.py \
--arch='hypernetwork_offset_wth_FC_on_3_FC_middle_dim_512_32' \
--size 256 \
--ckpt 'hypernetwork_with_2_layer_FC_offset_middle_dim_512_32 _checkpoint/795000.pt'



# for the softmax with square root:

# for hypernetwork with 3 FC implementation of PMSC-GAN :

python fid_new_model.py --arch=hypernetwork_offset_wth_FC_on_3_FC_middle_dim_512_32 \
--batch=8 \
--size=256 \
--ckpt=hypernetwork_with_2_layer_FC_offset_middle_dim_512_32_checkpoint/795000.pt \
--inception=inception_ffhq.pkl \
--n_sample=51500

FID = 1.374 (fid: 1.3742205923191652e+23)

# for FC implementation of PMSC-GAN:

python fid_new_model.py --arch=changed_mapping_network_with_3FC_reduced_RGB_style_dim_path_regularized \
--batch=8 \
--size=256 \
--ckpt=changed_mapping_network_with_3FC_reduced_RGB_style_dim_path_regularized _checkpoint/798000.pt \
--inception=inception_ffhq.pkl \
--n_sample=50000


______________________
#load and examine models:

python load_and_examine_models.py \
--arch 'stylegan2' \
--size 1024 \
--ckpt checkpoint_original_stylegan2/stylegan2-ffhq-config-f.pt




python load_and_examine_models.py \
--arch='hypernetwork_offset_wth_FC_on_3_FC_middle_dim_512_32' \
--size 256 \
--ckpt 'hypernetwork_with_2_layer_FC_offset_middle_dim_512_32 _checkpoint/795000.pt'

_______________


# walking in latent:
# with loaded z_latent and noise:
python walking_in_latent.py \
--arch=hypernetwork_offset_wth_FC_on_3_FC_middle_dim_512_32 \
--size=256 \
--number_of_steps_to_take=5 \
--ckpt='hypernetwork_offset_wth_FC_on_3_FC_middle_dim_512_32 _checkpoint/795000.pt' \
--use_specified_latent_and_noise=True \
--z_latent_path=projection_to_latent_space_results/find_latent_for_real_images_results/hypernetwork_offset_wth_FC_on_3_FC_middle_dim_512_32/z_latent_after_90000_epoch_targeting2_seed2.npy \
--trained_noise_path=projection_to_latent_space_results/find_latent_for_real_images_results/hypernetwork_offset_wth_FC_on_3_FC_middle_dim_512_32/trained_noise_after_90000_epoch_targeting2_seed2.npy


python walking_in_latent.py \
--arch=hypernetwork_offset_wth_FC_on_3_FC_middle_dim_512_32 \
--size=256 \
--number_of_steps_to_take=4 \
--random_seed=7 \
--walk_size_in_latent=0.2 \
--ckpt='hypernetwork_offset_wth_FC_on_3_FC_middle_dim_512_32 _checkpoint/795000.pt'


python walking_in_latent.py \
--arch=stylegan2 \
--size=256 \
--number_of_steps_to_take=4 \
--random_seed=1 \
--walk_size_in_latent=0.3 \
--walk_by_single_argument=False \
--ckpt='checkpoint_original_stylegan2/550000.pt'

________________________
# check latent distribution:


python check_latent_distribution.py \
--device='cuda:0' \
--arch=hypernetwork_offset_wth_FC_on_3_FC_middle_dim_512_32 \
--size=256 \
--ckpt='hypernetwork_offset_wth_FC_on_3_FC_middle_dim_512_32 _checkpoint/795000.pt' \
--path_to_latent='projection_to_latent_space_with_langevin_results/find_latent_for_real_images_results/hypernetwork_offset_wth_FC_on_3_FC_middle_dim_512_32_with_ckpthypernetwork_offset_wth_FC_on_3_FC_middle_dim_512_32 _checkpoint/795000.pt/langevin_correction_0.05_var_reg_0.025/z_latent_after_0_epoch_targeting2_seed1.npy'



________________________________________
project into latent space:


python projector.py \
--arch=hypernetwork_offset_wth_FC_on_3_FC_middle_dim_512_32 \
--ckpt='hypernetwork_offset_wth_FC_on_3_FC_middle_dim_512_32 _checkpoint/795000.pt' \
--use_z_instead_of_w=True \
 --size 256 \
target_images/real2.jpg

python projector.py \
--arch=hypernetwork_offset_wth_FC_on_3_FC_middle_dim_512_32 \
--ckpt=hypernetwork_with_2_layer_FC_offset_middle_dim_512_32_checkpoint/795000.pt \
--use_z_instead_of_w=True \
--size=256 \
--noise_regularize='1e5' \
--mse=1 \
--lr_rampup=0.05 \
--lr_rampdown=0.25 \
--step=1000 \
target_images/real2.png


# original model (for reference):

python projector.py \
--arch=stylegan2 \
--step=1000 \
--ckpt='stylegan2_checkpoint/550000.pt' \
--w_plus \
--size 256 \
--noise_regularize='1e5' \
--mse=3 \
--lr_rampup=0.05 \
--lr_rampdown=0.25 \
target_images/real1.png



