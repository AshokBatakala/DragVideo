# this is used from Run_dragvideo.ipynb


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


def load_generators(model_id, image_name):
  with open(paths_config.stylegan2_ada_ffhq, 'rb') as f:
    old_G = pickle.load(f)['G_ema'].cuda()

  with open(f'{paths_config.checkpoints_dir}/model_{model_id}_{image_name}.pt', 'rb') as f_new:
    new_G = torch.load(f_new).cuda()

  return old_G, new_G








#code from : https://github.com/danielroich/PTI/issues/26 , plus little bit modification

def export_updated_pickle(new_G,model_id,name = 'stylegan2'):
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

  with open(f'{paths_config.checkpoints_dir}/{name}_{model_id}.pkl', 'wb') as f:
      pickle.dump(tmp, f)




# model_id = '/home/bean/DragVideo/Data_store/model_weights/sg3_3rdtime_weights/sg3-r-ffhq-1024_module.pt'
# use_multi_id_training = True

# generator_type = paths_config.multi_id_model_type if use_multi_id_training else image_name
# old_G, new_G = load_generators(model_id, generator_type)



# export_updated_pickle(new_G,model_id)


# model_id # 'AXOCWYGCAEDY'for 10 images