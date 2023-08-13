# INSTRUCTIONS TO RUN :
# (/Ext_4T_SSD/ASHOK/DragVideo/env/sg3) cds2@vcl-winterfell:/Ext_4T_SSD/ASHOK/DragVideo/DragGAN/_PTI$ CUDA_VISIBLE_DEVICES=0  python run.py


# run on gpu:1
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# --------------------------------  configs   ------------------------------------------------

DRAGVIDEO_ROOT_PATH = "/Ext_4T_SSD/ASHOK/"

lazy_config = {
    "EXP_NAME": "exp_01_pti_1_inv_1",
    "e4e": f"{DRAGVIDEO_ROOT_PATH}DragVideo/DragGAN/_PTI/pretrained_models/e4e_ffhq_encode.pt",
    "stylegan2_ada_ffhq": f"{DRAGVIDEO_ROOT_PATH}DragVideo/DragGAN/_PTI/pretrained_models/ffhq.pkl",
    "video_path":  f"{DRAGVIDEO_ROOT_PATH}DragVideo/Data_store/OLD/original_videos/person_speaking_original.mp4",
    "model_name" : "stylegan2",
    "n_frames" : 1,
    "IMAGE_SIZE": 1024,
    "N_STEPS_in_draggan": "1",
    

}

env_config = {
    "DragGAN_dir": f"{DRAGVIDEO_ROOT_PATH}DragVideo/DragGAN",
    "Experiment_base_path":f"{DRAGVIDEO_ROOT_PATH}DragVideo/Data_store/experiments/" ,
    "init_exp_dir_shell_path": f"{DRAGVIDEO_ROOT_PATH}DragVideo/DragGAN/_PTI/init_datadirs.sh",
    "dummy_config_path" : f"{DRAGVIDEO_ROOT_PATH}DragVideo/DragGAN/_PTI/configs/dummy",
    
    
}

#hyper parameters in PTI
hyper_config = {
    "max_pti_steps": 1,#500,
    "first_inv_steps": 1,#00,
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



use_multi_id_training = True

generator_type =paths_config.multi_id_model_type if use_multi_id_training else image_name
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



os.chdir(env_config["DragGAN_dir"])
import subprocess
subprocess.call(['python', 'run_dragvideo.py', '--Experiment_path', Experiment_path, '--N_STEPS', lazy_config["N_STEPS_in_draggan"], '--CHECKPOINT_PATH', sg_tuned_pkl])





