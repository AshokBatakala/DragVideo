import numpy as np
import pickle
from utils_draggan.draggan_utils import list2dict
from .main import get_border_points,clip_points_targets


# get points used in argument

# change the function later
def mouth_wide(landmarks_path,MAX_SIZE=1024):
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
    
    points = get_border_points(MAX_SIZE=MAX_SIZE, padding=20, num_points=5, sides=['bottom'])
    targets = get_border_points(MAX_SIZE=MAX_SIZE, padding=20, num_points=5, sides=['bottom'])
    
    # mouth wide 
    points = np.vstack([points,landmarks[48]])
    points = np.vstack([points,landmarks[54]])
    # points = np.vstack([points,landmarks[51]])
    
    targets = np.vstack([targets,landmarks[48]-np.array([50,0])])
    targets = np.vstack([targets,landmarks[54]+np.array([50,0])])
    
    # targets = np.vstack([targets,landmarks[51]+np.array([50,0])])
    
    
    # right eye
    # points = np.vstack([points,landmarks[37]])
    # points = np.vstack([points,landmarks[41]])
    
    # targets = np.vstack([targets,landmarks[37]-np.array([0,50])])
    # targets = np.vstack([targets,landmarks[41]+np.array([0,50])])
    
    # # left eye
    # points = np.vstack([points,landmarks[44]])
    # points = np.vstack([points,landmarks[46]])
    
    # targets = np.vstack([targets,landmarks[44]-np.array([0,50])])
    # targets = np.vstack([targets,landmarks[46]+np.array([0,50])])
    
    # ----------------------------------------------
    #clip points and targets
    points,targets = clip_points_targets(points,targets,MAX_SIZE=MAX_SIZE)
    return list2dict(points,targets)
    
    
    
    
    
    
    
    
    
    