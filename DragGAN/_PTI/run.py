# ==================================================================================================
#                   instructions to run 
# ==================================================================================================

# INSTRUCTIONS TO RUN :

# Use “bold ganguly” container  then use conda env pti_env  for all the code.
# Tested; working with sg2 and sg3 also. 

# conda activate /home/bean/DragVideo/env/pti_env


# ==================================================================================================
# run on gpu:1
# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ==== IMP ==================================================================
#  this import is must to avoid the error
# RuntimeError: GET was unable to find an engine to execute this computation
import torch 
# ============================================================================

# # torch.cuda.is_available()
# print(torch.cuda.device_count())
# print(torch.__version__)


# why this is not working
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
# --------------------------------  configs   ------------------------------------------------

# DRAGVIDEO_ROOT_PATH = "/Ext_4T_SSD/ASHOK/"
DRAGVIDEO_ROOT_PATH = "/home/bean/"
EXPERIMENT_NAME_POSTFIX = "with_sg3_&_old_e4e_editing_100_steps"

# keep temp paths here
#---------------------------------
sg3_path = f"{DRAGVIDEO_ROOT_PATH}DragVideo/Data_store/OLD/model_weights/stylegan3_3rdtime_weights/stylegan3-r-ffhqu-1024_module.pkl"
#defualt path ::  "stylegan2_ada_ffhq":  f"{DRAGVIDEO_ROOT_PATH}DragVideo/DragGAN/_PTI/pretrained_models/ffhq.pkl",

# sg3_path = f"{DRAGVIDEO_ROOT_PATH}DragVideo/DragGAN/_PTI/pretrained_models/ffhq.pkl"



# --------------------------------  

import datetime
lazy_config = {
    "EXP_NAME": str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))+EXPERIMENT_NAME_POSTFIX,
    "e4e": f"{DRAGVIDEO_ROOT_PATH}DragVideo/DragGAN/_PTI/pretrained_models/e4e_ffhq_encode.pt",
    "stylegan2_ada_ffhq":  sg3_path,
    "video_path":  f"{DRAGVIDEO_ROOT_PATH}DragVideo/Data_store/OLD/original_videos/person_speaking_original.mp4",
    "model_name" : "stylegan3", # stylegan2 or stylegan3
    "n_frames" : 1,
    "IMAGE_SIZE": 1024,
    "N_STEPS_in_draggan": "100",
    
}

env_config = {
    "DragGAN_dir": f"{DRAGVIDEO_ROOT_PATH}DragVideo/DragGAN",
    "Experiment_base_path":f"{DRAGVIDEO_ROOT_PATH}DragVideo/Data_store/experiments/" ,
    "init_exp_dir_shell_path": f"{DRAGVIDEO_ROOT_PATH}DragVideo/DragGAN/_PTI/init_datadirs.sh",
    "dummy_config_path" : f"{DRAGVIDEO_ROOT_PATH}DragVideo/DragGAN/_PTI/configs/dummy",
    
    
}

#hyper parameters in PTI
hyper_config = {
    "max_pti_steps": 450,
    "first_inv_steps": 200,
    "max_images_to_invert": 200,
}


# ----------------------------------------------------------------------------------------------

from run_utils import *

# create experiment data folder structure
Experiment_name = lazy_config["EXP_NAME"]
Experiment_base_path = env_config["Experiment_base_path"]
Experiment_path = os.path.join(Experiment_base_path, Experiment_name)

init_experiment_dir(Experiment_name,Experiment_base_path,shell_script_path=env_config["init_exp_dir_shell_path"])



# dummy paths_config overwrites the paths_config.py
# dummy_config_path = f"{DRAGVIDEO_ROOT_PATH}DragVideo/DragGAN/_PTI/configs/dummy'
add_dummy_config_from_dict("hyperparameters.py", hyper_config,ROOT_PATH=env_config["dummy_config_path"])

# change path configs , hyperparameters 
paths_config_dict = {
    #pretrained models
    "e4e": lazy_config["e4e"],
    "stylegan2_ada_ffhq": lazy_config["stylegan2_ada_ffhq"],
    
    # to store tuned stylegan weights
    "checkpoints_dir": os.path.join(Experiment_path,'tuned_SG'),
    # to store latents
    "embedding_base_dir": os.path.join(Experiment_path,'latents'),
    # aligned / processed images
    "input_data_path": os.path.join(Experiment_path,'aligned'),
}

add_dummy_config_from_dict("paths_config.py", paths_config_dict,ROOT_PATH=env_config["dummy_config_path"])



# add all these configs to log.txt
# --------------------------------  
with open(os.path.join(Experiment_path,'log.txt'), 'a') as f:
    import json
    f.write(f"lazy_config: {json.dumps(lazy_config, indent=4)}\n")
    f.write(f"env_config: {json.dumps(env_config, indent=4)}\n")
    f.write(f"hyper_config: {json.dumps(hyper_config, indent=4)}\n")
    
    



from importlib.machinery import SourceFileLoader
# imports the module from the given path
video_utils = SourceFileLoader("video_utils","../utils_draggan/video_utils.py").load_module()
raw_path = os.path.join(Experiment_path, "raw")
from utils.align_data import pre_process_images

video_utils.extract_frames(lazy_config['video_path'], raw_path,n_frames=lazy_config['n_frames'])
pre_process_images(raw_path, IMAGE_SIZE=lazy_config['IMAGE_SIZE']) # o/p: config.input_data_path 




from utils.align_data import pre_process_images
from scripts.run_pti import run_PTI

from run_utils_2 import load_generators,export_updated_pickle
from configs import paths_config

use_multi_id_training = True
model_id = run_PTI(use_wandb=False, use_multi_id_training=use_multi_id_training)


generator_type =paths_config.multi_id_model_type if use_multi_id_training else "__"
old_G, new_G = load_generators(model_id, generator_type)
sg_tuned_pkl = export_updated_pickle(new_G,model_id,name = lazy_config["model_name"])

print(sg_tuned_pkl) # 'QBUXQCXZGWET'



#get landmarks
landmark_path = f"{DRAGVIDEO_ROOT_PATH}DragVideo/DragGAN/_facial-landmarks-recognition"
os.chdir(landmark_path)
print("pwd",os.getcwd())
import sys
sys.path.append(landmark_path)

from main import landmarks, dict_landmarks,show_landmarks,get_landmarks_dir

# to store landmarks
landmarks_dir =  os.path.join(Experiment_path,'landmarks')
processed_images_dir =  os.path.join(Experiment_path,'aligned')

# generate landmarks for all images in processed_images_dir
get_landmarks_dir(processed_images_dir,landmarks_dir)


# clean all data from gpu
import torch
torch.cuda.empty_cache()



os.chdir(env_config["DragGAN_dir"])
import subprocess

print(f"sg_tuned_pkl: {sg_tuned_pkl}")

subprocess.call(['python', 'run_dragvideo.py', '--Experiment_path', Experiment_path, '--N_STEPS', lazy_config["N_STEPS_in_draggan"], '--CHECKPOINT_PATH', sg_tuned_pkl])

# run above command with conda env "sg2ada" using subprocess
# import subprocess
# subprocess.call(['conda', 'activate', 'sg2ada'])
# subprocess.call(['python', 'run_dragvideo.py', '--Experiment_path', Experiment_path, '--N_STEPS', lazy_config["N_STEPS_in_draggan"], '--CHECKPOINT_PATH', sg_tuned_pkl])




#conda env to be used (/home/bean/DragVideo/env/Dragvideo)
# import subprocess
# subprocess.run('conda run -n <env_name> python <file_name>.py', shell=True)
# command : # subprocess.call(['python', 'run_dragvideo.py', '--Experiment_path', Experiment_path, '--N_STEPS', lazy_config["N_STEPS_in_draggan"], '--CHECKPOINT_PATH', sg_tuned_pkl])

# subprocess.run(f'conda run -p /home/bean/DragVideo/env/Dragvideo  python run_dragvideo.py --Experiment_path {Experiment_path} --N_STEPS {lazy_config["N_STEPS_in_draggan"]} --CHECKPOINT_PATH {sg_tuned_pkl}', shell=True)