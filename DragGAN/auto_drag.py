# Ashok added this file

import torch
from visualizer_auto import DragVideo

def do_drag(w_load_path=None,
            stylegan2_wieghts_path=None,
            points = dict(),
            N_STEPS=50,
            save_path=None
):
    
    w_load = torch.load(w_load_path)
    # from visualizer_auto import DragVideo
    drag_video = DragVideo(w_load=w_load,
                        stylegan2_wieghts_path=stylegan2_wieghts_path)
    
    feat = drag_video.run(N_STEPS=N_STEPS,points=points)
    image = drag_video.global_state['images']['image_raw']

    if save_path is not None:
        image.save(save_path)
    
    return drag_video.global_state['images']#['image_raw']


#----------------------------------------------
import pickle
import numpy as np
from utils.list2dict import list2dict

# get points used in argument
def modify_landmarks(landmarks_path):
    """ 
    1.modify landmarks
    2.create points,targets
    3. return dict in format of {'points':points,'targets':targets}
    """
    with open(landmarks_path,'rb') as f:
        list_lms = pickle.load(f)
    if len(list_lms) == 0:
        return dict()
    else:
        points = list_lms[0]
    
    # ----------------------------------------------
    #       modifications here
    # ----------------------------------------------
    # # get targets 
    # #  off set by 100 from middle line ie. x = 512 
    # targets = np.array(points) - np.array([512,0])
    # targets = None

    targets = np.array(points).copy()
    # up the nose by 100 in y direction 
    # targets[range(27, 36)] -= np.array([0, 50])

    # # lift eyebrows by 50 in y direction
    # targets[dict_landmarks['left_eyebrow']] -= np.array([0, 50])
    # targets[dict_landmarks['right_eyebrow']] -= np.array([0, 50])

    #collapse nose width 
    targets[ range(31,36)] = targets[33]

    # # make eyes smaller by 50 in x direction
    # targets[dict_landmarks['left_eye']] += np.array([50, 0])
    # targets[dict_landmarks['right_eye']] -= np.array([50, 0])

    # make jaw wider by 50 in x direction
    targets[range(0,6)] -= np.array([50, 0]) 
    targets[range(11,17)] += np.array([50, 0])

    # bottom jaw up by 50 in y direction
    targets[range(6,11)] -= np.array([0, 50])

    # ----------------------------------------------
    return list2dict(points,targets)



#================================================================
# TUNED_STYLEGAN2_WEIGHTS_PATH =  "/home/bean/DragVideo/DragGAN/PTI_results/checkpoints/stylegan2_ZEWLQSQSSQWA.pkl" #"/workspace/src/PTI_results/checkpoints/stylegan2_MRILHLYXXEXU.pkl" # MAN2

# from auto_drag import modify_landmarks

# BASE_DRAGGAN_PATH = "/home/bean/DragVideo/DragGAN/"

# def get_arguments(name,
#                   N_STEPS=100,
#                   CHECKPOINT_PATH=None):
    
#     assert CHECKPOINT_PATH is not None

#     landmarks_path = BASE_DRAGGAN_PATH+f"PTI_results/landmarks/{name}.pkl"
#     return {
#         'w_load_path':f"PTI_results/embeddings/barcelona/PTI/{name}/0.pt",
#         'stylegan2_wieghts_path' :CHECKPOINT_PATH,
        
#         'points' : modify_landmarks(landmarks_path),
#         'N_STEPS': N_STEPS,
#         'save_path':f"PTI_results/after_drag/{name}.png",
#     }

