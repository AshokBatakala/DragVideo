# Preprocessing - dlib:

#pti :

PTI_DIR = "/home/bean/DragVideo/PTI" #/home/bean/DragVideo/PTI

import os
import sys 
os.chdir(PTI_DIR)
sys.path.append(PTI_DIR)


from utils.align_data import pre_process_images
from utils.align_data import paths_config


path_raw_img = "/home/bean/DragVideo/Data_store/data/raw_images" # Path to raw images
paths_config.input_data_path = '/home/bean/DragVideo/Data_store/data/processed_images_sg3' # save path for pre-processed images
# print(sys.argv)
# preprocessing
pre_process_images(path_raw_img, IMAGE_SIZE=256)