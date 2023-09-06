import os
import torch
os.chdir("/home/bean/DragVideo/DragGAN")
# from _do_drag import DragVideo
from _dragvideo_modules._do_drag import DragVideo

# w_load_path = "/home/bean/DragVideo/Data_store/experiments/_SAVE_actress_smile_sg2_smile_value_20_with_mask/latents/barcelona/PTI/000/0.pt"
# w_load = torch.load(w_load_path)

sg_path = "/home/bean/DragVideo/Data_store/experiments/_SAVE_actress_smile_sg2_smile_value_20_with_mask/tuned_SG/stylegan2_LNPIPBZYBTDD.pkl"
exp_dir = "/home/bean/DragVideo/Data_store/experiments/_SAVE_actress_smile_sg2_smile_value_20_with_mask"

output_dir = "/home/bean/DragVideo/Data_store/experiments/new/recursive"

dragvideo = DragVideo(#w_load = w_load,
                      stylegan2_wieghts_path=sg_path,
                        inputs_dir = exp_dir,
                        outputs_dir = output_dir,
                        # editing_function_name = "smile",
                        image_size = 1024,
                        # N_STEPS=50,
                        device = "cuda",
                        verbose=False,)

dragvideo.run(N_STEPS=50,edit_mode="smile")