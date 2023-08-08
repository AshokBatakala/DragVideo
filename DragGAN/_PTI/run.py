import os 
import sys
# root_dir = '/home/bean/DragVideo'
# os.chdir(f'{root_dir}/PTI')

import os
import sys
import pickle
import numpy as np
from PIL import Image
import torch
from configs import paths_config, hyperparameters, global_config
from utils.align_data import pre_process_images
from scripts.run_pti import run_PTI
from IPython.display import display
import matplotlib.pyplot as plt
from scripts.latent_editor_wrapper import LatentEditorWrapper

# paths_config.DragGan_base_path = '/home/bean/DragVideo/'

# # encoder path
# paths_config.e4e = paths_config.DragGan_base_path + 'Data_store/model_weights/restyle_e4e_ffhq.pt'

# #stylegan checkpoints
# paths_config.stylegan2_ada_ffhq = paths_config.DragGan_base_path + 'PTI/pretrained_models/stylegan3-r-ffhqu-256x256.pkl'

# #paths for saving chkpts and latents
# paths_config.checkpoints_dir = paths_config.DragGan_base_path + 'Data_store/data/PTI_results/checkpoints' # tuned_stylegan_weights
# paths_config.embedding_base_dir = paths_config.DragGan_base_path +'Data_store/data/PTI_results/embeddings' # latents

# print(os.environ['PATH'])

# use_multi_id_training = True

# from scripts.run_pti import run_PTI
# model_id = run_PTI(use_wandb=False, use_multi_id_training=use_multi_id_training)

def load_generators(model_id, image_name):
  with open(paths_config.stylegan2_ada_ffhq, 'rb') as f:
    old_G = pickle.load(f)['G_ema'].cuda()

  with open(f'{paths_config.checkpoints_dir}/model_{model_id}_{image_name}.pt', 'rb') as f_new:
    new_G = torch.load(f_new).cuda()

  return old_G, new_G

# generator_type = paths_config.multi_id_model_type if use_multi_id_training else image_name
# old_G, new_G = load_generators(model_id, generator_type)

#code from : https://github.com/danielroich/PTI/issues/26 , plus little bit modification

def export_updated_pickle(new_G,model_id):
  print("Exporting large updated pickle based off new generator and ffhq.pkl")
  with open(paths_config.stylegan2_ada_ffhq, 'rb') as f:
    d = pickle.load(f)
    old_G = d['G_ema'].cuda()
    old_D = d['D'].eval().requires_grad_(False).cpu()

  tmp = {}
  tmp['G'] = old_G.eval().requires_grad_(False).cpu()
  tmp['G_ema'] = new_G.eval().requires_grad_(False).cpu()
  tmp['D'] = old_D
  tmp['training_set_kwargs'] = None
  tmp['augment_pipe'] = None

  with open(f'{paths_config.checkpoints_dir}/stylegan2_{model_id}.pkl', 'wb') as f:
      pickle.dump(tmp, f)




# export_updated_pickle(new_G,model_id)

# print('--------------------------------')
# print('model .pkl file name:', model_id)

#-------------------------------------------------------------------------------
#paths_config.DragGan_base_path = '/home/bean/DragVideo/'

# # encoder path
# e4e_path ='/home/bean/DragVideo/Data_store/model_weights/restyle_e4e_ffhq.pt'

# #stylegan checkpoints
# stylegan2_ada_ffhq_path = '/home/bean/DragVideo/DragGAN/checkpoints/stylegan3-r-ffhq-1024x1024.pkl'

# #paths for saving chkpts and latents
# SG_checkpoints_dir = paths_config.DragGan_base_path + 'Data_store/data/PTI_results/checkpoints' # tuned_stylegan_weights
# embedding_base_dir = paths_config.DragGan_base_path +'Data_store/data/PTI_results/embeddings' # latents

def run(encoder_path=None,
        stylegan_path = None,
        save_tuned_SG_path = None,
        save_latents_path = None,
        use_multi_id_training=True,
        verbose=False):

  use_multi_id_training = use_multi_id_training

  if encoder_path!=None:
    paths_config.e4e = encoder_path
  if stylegan_path!=None:
    paths_config.stylegan2_ada_ffhq = stylegan_path
  if save_tuned_SG_path!=None:
    paths_config.checkpoints_dir = save_tuned_SG_path
  if save_latents_path!=None:
    paths_config.embedding_base_dir = save_latents_path

  if verbose==True:

    print('encoder_path:',paths_config.e4e)
    print('stylegan_path:',paths_config.stylegan2_ada_ffhq)
    print('save_tuned_SG_path:',paths_config.checkpoints_dir)
    print('save_latents_path:',paths_config.embedding_base_dir)

  from scripts.run_pti import run_PTI
  model_id = run_PTI(use_wandb=False, use_multi_id_training=use_multi_id_training)

  generator_type = paths_config.multi_id_model_type if use_multi_id_training else image_name
  old_G, new_G = load_generators(model_id, generator_type)

  export_updated_pickle(new_G,model_id)

  print('--------------------------------')
  print('model .pkl file name:', model_id)

if __name__=="__main__":
  # run(e4e_path, 
  #     stylegan2_ada_ffhq_path,
  #     SG_checkpoints_dir,
  #     embedding_base_dir, verbose=True)

# e4e_path =None
  #     stylegan2_ada_ffhq_path =
  #     SG_checkpoints_dir,
  #     embedding_base_dir, verbose=True)




  # run(encoder_path=e4e_path,
  #         stylegan_path = stylegan2_ada_ffhq_path,
  #         save_tuned_SG_path = SG_checkpoints_dir,
  #         save_latents_path = embedding_base_dir,
  #         use_multi_id_training=True,
  #         verbose=False)
    run()