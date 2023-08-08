import os 
import pickle

draggan_path = "/home/bean/DragVideo/DragGAN" # dir_path
latents_dir = draggan_path+"/PTI_results/embeddings/barcelona/PTI"

#  run draggan on based on latents availability

temp = os.listdir(latents_dir)
temp.sort()
names = [i.split('.')[0] for i in temp]
# names 

# run dragan for each image in names
TUNED_STYLEGAN2_WEIGHTS_PATH = "/home/bean/DragVideo/DragGAN/PTI_results/checkpoints/stylegan3_IZRDVTQHLVHZ.pkl"

for name in names:
    args = get_arguments(name,CHECKPOINT_PATH =TUNED_STYLEGAN2_WEIGHTS_PATH,N_STEPS=1)
    do_drag(**args)
    