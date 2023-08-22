import numpy as np
import pickle
from utils_draggan.draggan_utils import list2dict

from .main import get_border_points,clip_points_targets

# get points used in argument
def make_jaw_wider(landmarks_path,MAX_SIZE=1024):
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
    
    points = get_border_points(MAX_SIZE=MAX_SIZE, padding=20, num_points=10, points=points,sides=['bottom','top','left','right'])
    
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

    points,targets = clip_points_targets(points,targets,MAX_SIZE=MAX_SIZE)
    
    # ----------------------------------------------
    return list2dict(points,targets)
