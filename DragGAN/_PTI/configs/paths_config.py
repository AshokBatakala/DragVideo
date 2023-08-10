
# ## Pretrained models paths
# e4e = './pretrained_models/e4e_ffhq_encode.pt'
# stylegan2_ada_ffhq = '../pretrained_models/ffhq.pkl'
# style_clip_pretrained_mappers = ''
# ir_se50 = './pretrained_models/model_ir_se50.pth'
# dlib = './pretrained_models/align.dat'

# ## Dirs for output files
# checkpoints_dir = './checkpoints'
# embedding_base_dir = './embeddings'
# styleclip_output_dir = './StyleCLIP_results'
# experiments_output_dir = './output'

# turn above paths to absolute paths
# ----------------------------------------------------------------
main_root_path = '/home/bean/DragVideo/DragGAN/_PTI'
# ----------------------------------------------------------------
## Pretrained models paths
e4e = f'{main_root_path}/pretrained_models/e4e_ffhq_encode.pt'
stylegan2_ada_ffhq = f'{main_root_path}/pretrained_models/stylegan3-r-ffhqu-256x256.pkl' #stylegan3
#stylegan2_ada_ffhq = f'{main_root_path}/pretrained_models/ffhq.pkl'

# stylegan2_ada_ffhq = f'{main_root_path}/pretrained_models/stylegan2-ffhq-512x512.pkl'

style_clip_pretrained_mappers = ''
ir_se50 = f'{main_root_path}/pretrained_models/model_ir_se50.pth'
dlib = f'{main_root_path}/pretrained_models/align.dat'

## Dirs for output files
# checkpoints_dir = f'{main_root_path}/checkpoints'
# embedding_base_dir = f'{main_root_path}/embeddings'


# changed to inside the draggan folder ; so that draggan can useit
DragGan_base_path = "/home/bean/DragVideo/"
checkpoints_dir = DragGan_base_path + '/DragGAN/PTI_results/checkpoints' # tuned_stylegan_weights
embedding_base_dir = DragGan_base_path +'/DragGAN/PTI_results/embeddings' # latents


styleclip_output_dir = f'{main_root_path}/StyleCLIP_results'
experiments_output_dir = f'{main_root_path}/output'




#---------------------------------------------------------------

## Input info
### Input dir, where the images reside

# input_data_path = f'{main_root_path}/data/processed_images' #'./input_data' '/home/bean/DragVideo/Data_store/data/processed_images'
input_data_path = '/home/bean/DragVideo/Data_store/data/processed_images_sg3'
### Inversion identifier, used to keeping track of the inversion results. Both the latent code and the generator
input_data_id = 'barcelona'

## Keywords
pti_results_keyword = 'PTI'
e4e_results_keyword = 'e4e'
sg2_results_keyword = 'SG2'
sg2_plus_results_keyword = 'SG2_plus'
multi_id_model_type = 'multi_id'

## Edit directions
interfacegan_age = 'editings/interfacegan_directions/age.pt'
interfacegan_smile = 'editings/interfacegan_directions/smile.pt'
interfacegan_rotation = 'editings/interfacegan_directions/rotation.pt'
ffhq_pca = 'editings/ganspace_pca/ffhq_pca.pt'


# any changes to the path_configs should be done there 
# can be automated using a script
from configs.dummy.paths_config import *