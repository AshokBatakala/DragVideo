

# =========================================================================================================

import os
import sys
import pickle
import numpy as np
from PIL import Image
import torch
from configs import paths_config, hyperparameters, global_config
from utils.align_data import pre_process_images
from scripts.run_pti import run_PTI
# from IPython.display import display
import matplotlib.pyplot as plt
from scripts.latent_editor_wrapper import LatentEditorWrapper





def load_generators(model_id, image_name):
  # return old_G from SG given to pti
  # new_G from pti_result
  with open(paths_config.stylegan2_ada_ffhq, 'rb') as f:
    old_G = pickle.load(f)['G_ema'].cuda()

  with open(f'{paths_config.checkpoints_dir}/model_{model_id}_{image_name}.pt', 'rb') as f_new:
    new_G = torch.load(f_new).cuda()

  return old_G, new_G


#code from : https://github.com/danielroich/PTI/issues/26 , plus little bit modification

def export_updated_pickle(new_G,model_id,name):
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

    
  pkl_path = f'{paths_config.checkpoints_dir}/{name}_{model_id}.pkl'

  with open(pkl_path, 'wb') as f:
      pickle.dump(tmp, f)

  return pkl_path