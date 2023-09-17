# to run it from dragvideo folder 
#cd DragGAN/_PTI/  && python _02_run_PTI.py input_dir
#cd DragGAN/_PTI/  && python _02_run_PTI.py /media/Ext_4T_SSD/ASHOK/DragVideo/Data_store/experiments/2023-09-18_00-06-53_woman_waiting/inputs/part_0




from utils.align_data import pre_process_images
from scripts.run_pti import run_PTI
from run_utils_2 import load_generators,export_updated_pickle
batch_id = 0
from configs import paths_config
import os

paths_config.e4e = '/media/Ext_4T_SSD/ASHOK/DragVideo/DragGAN/_PTI/pretrained_models/e4e_ffhq_encode.pt'
paths_config.stylegan2_ada_ffhq = '/media/Ext_4T_SSD/ASHOK/DragVideo/DragGAN/_PTI/pretrained_models/ffhq.pkl'
model_name = "stylegan2"
paths_config.dlib = '/media/Ext_4T_SSD/ASHOK/DragVideo/DragGAN/_PTI/pretrained_models/align.dat'


# input_dir = "/media/Ext_4T_SSD/ASHOK/DragVideo/Data_store/experiments/2023-09-17_17-29-22_woman_waiting/"
#get input_dir from the command line
import sys
input_dir = sys.argv[1]
print(f"{input_dir=}")

IMAGE_SIZE = 1024


#=========================================================
# requirements: input_dir/raw => input_dir/aligned,qaud_values, latents, tuned_SG
# it should only have raw images
#=========================================================

for dir_name in ["tuned_SG","latents","aligned","quad_values","landmarks"]:
    os.makedirs( os.path.join(input_dir,dir_name),exist_ok=True)

raw_path = os.path.join(input_dir,"raw")

# paths_config.checkpoints_dir= f'{input_dir}tuned_SG'
# paths_config.embedding_base_dir= f'{input_dir}latents'
# paths_config.input_data_path= f'{input_dir}aligned'
# paths_config.quad_values_path= f'{input_dir}quad_values'

paths_config.checkpoints_dir= os.path.join(input_dir,"tuned_SG")
paths_config.embedding_base_dir= os.path.join(input_dir,"latents")
paths_config.input_data_path= os.path.join(input_dir,"aligned")
paths_config.quad_values_path= os.path.join(input_dir,"quad_values")




#=========================================================
#preprocess images : align-data.pre_process_images => aligned,quad_values
#=========================================================
from utils.align_data import pre_process_images

pre_process_images(raw_path, IMAGE_SIZE=IMAGE_SIZE,
                   save_output_path=os.path.join(input_dir,"aligned"),
                   save_quad_values_path=os.path.join(input_dir,"quad_values")
)

#=========================================================
#get landmaks => landmarks
#=========================================================
DRAGVIDEO_ROOT_PATH = "/media/Ext_4T_SSD/ASHOK/"

landmark_path = f"{DRAGVIDEO_ROOT_PATH}DragVideo/DragGAN/_facial-landmarks-recognition"
os.chdir(landmark_path)
# print("pwd:",os.getcwd())
import sys
sys.path.append(landmark_path)

from main import get_landmarks_dir

# to store landmarks
landmarks_dir =  os.path.join(input_dir,'landmarks')
processed_images_dir =  os.path.join(input_dir,'aligned')

# generate landmarks for all images in processed_images_dir
get_landmarks_dir(processed_images_dir,landmarks_dir)

#=========================================================
# get latents => latents
#=========================================================

use_multi_id_training = True
model_id = run_PTI(use_wandb=False, use_multi_id_training=use_multi_id_training)

generator_type =paths_config.multi_id_model_type if use_multi_id_training else "__"
old_G, new_G = load_generators(model_id, generator_type)
sg_tuned_pkl = export_updated_pickle(new_G,model_id,name = model_name)

print( f"{sg_tuned_pkl=}") # 'QBUXQCXZGWET'
