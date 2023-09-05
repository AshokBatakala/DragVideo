import numpy as np
import pickle
from utils_draggan.draggan_utils import list2dict
from .main import get_border_points,clip_points_targets


# get points used in argument

# change the function later
def smile(landmarks_path,MAX_SIZE=1024):
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
        landmarks = list_lms[0]
    
    # ----------------------------------------------
    points = np.array([])
    targets = np.array([])
    
    # make eyes larger by 50 in y direction
    # pick one points from bottom, top of the eye then move it up or down
    
    # points = get_border_points(MAX_SIZE=MAX_SIZE, padding=20, num_points=7, sides=['bottom'])
    # targets = get_border_points(MAX_SIZE=MAX_SIZE, padding=20, num_points=7, sides=['bottom'])
    
    # nose_tip = (landmarks[34] +landmarks[31])/2
    # nose_tip = nose_tip.astype(int)
    
    # points = np.vstack([points, nose_tip])
    # targets = np.vstack([targets, landmarks[63]])
    
    
    # move point 61 up by 50 left by 50
    # points = np.vstack([points, landmarks[49]])
    # targets = np.vstack([targets, landmarks[49] + np.array([-50,-50])])

    points = np.array([landmarks[49]])
    smile_values = 20
    # targets = np.array([landmarks[49] + np.array([-50,-50])])
    targets = np.array([landmarks[49] + np.array([-smile_values,-smile_values])])
    
    # ----------------------------------------------
    #clip points and targets
    points,targets = clip_points_targets(points,targets,MAX_SIZE=MAX_SIZE)
    return list2dict(points,targets)
    
    
    
    
    
    
    
    
    
    