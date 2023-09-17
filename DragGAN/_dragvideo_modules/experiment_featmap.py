from ._Run_SG import lazy_Run 
import os



# sg_tuned_pkl ="/home/bean/DragVideo/Data_store/experiments/_SAVE_vsauce_ffmpeg_inconsistent/tuned_SG/stylegan3_EZXZEVVUYSPX.pkl"
sg_tuned_pkl="/home/bean/DragVideo/Data_store/OLD/model_weights/stylegan3_3rdtime_weights/stylegan3-r-ffhq-1024_module.pkl"

ws_path = '/home/bean/DragVideo/Data_store/experiments/_SAVE_vsauce_ffmpeg_inconsistent/latents/barcelona/PTI/000/0.pt'
landmarks_path = "/home/bean/DragVideo/Data_store/experiments/2023-09-06_00-38-36_actress_smile/landmarks/000.pkl"

EXP_NAME = "untuned_SG"



img,feat = lazy_Run(sg_tuned_pkl,ws_path,want_plot=True,
                    set_feature = None,
                    # input_is_w =False,
                    )
# set_feature = set_feature)
print("got the features...")

EXP_dir = f"/home/bean/DragVideo/Data_store/featuremap_edit/{EXP_NAME}"
os.makedirs(EXP_dir,exist_ok=True)

import matplotlib.pyplot as plt
plt.imsave(f"{EXP_dir}/original.png",img)

from _dragvideo_modules.warp_image import *

edited_feat = [lazy_deform(feature,landmarks_path,img_size_during_landmark_calculation=1024) for feature in feat]


for i in range(1,16):
    n =  i
    print(f"===== {n} =====")
    set_feature = {
                "layer_num": n,
                # "feature":store
                "feature": edited_feat[n-1] 
            }

    img,feat = lazy_Run(sg_tuned_pkl,ws_path,want_plot=False,
                        # set_feature = None)
    set_feature = set_feature)
    
    print(img.shape) #(1024, 1024, 3)
    import matplotlib.pyplot as plt
    name = str(n).zfill(2)
    plt.imsave(f"{EXP_dir}/feat_{name}.png",img)
    

#create 4x4 grid of images
#---------------------------------------
import matplotlib.pyplot as plt
import numpy as np
import os

images_paths = [f"{EXP_dir}/{path}" for path in os.listdir(EXP_dir) if path.endswith(".png")]
images_paths.sort()


#16 images
#plot using subplots
fig, axs = plt.subplots(4, 4,figsize=(20,20))
for i in range(4):
    for j in range(4):
        img = plt.imread(images_paths[i*4+j])
        axs[i,j].imshow(img)
        axs[i,j].axis('off')
        # file name as title
        axs[i,j].set_title(images_paths[i*4+j].split("/")[-1].split(".")[0])

plt.savefig(f"{EXP_dir}/summery.png")

