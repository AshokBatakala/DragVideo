import numpy as np
import pickle


landmarks_dictionary = {'jaw': range(0, 17),
                    'right_eyebrow': range(17, 22),
                    'left_eyebrow': range(22, 27),
                    'nose': range(27, 36),
                    'right_eye': range(36, 42),
                    'left_eye': range(42, 48),
                    'mouth': range(48, 68)}




# function to give points around the border of image
def get_border_points(MAX_SIZE=1024, padding=20, num_points=10, points=None,sides=['bottom','top','left','right']):
    """
    returns points around the border of image
    num_points: number of points on each side of the image
    """
    if points is None:
        points = np.array([])
        
    bag = []
        
    # #bottom,top,left,right
        
    start = padding 
    end = MAX_SIZE-padding
    gap = (end-start)//(num_points)
    i_values = range(start,end,gap)
    if "bottom" in sides:
        bag.append(np.array([[i,MAX_SIZE-padding] for i in i_values]))
    if "top" in sides:
        bag.append(np.array([[i,padding] for i in i_values]))
    if "left" in sides:
        bag.append(np.array([[padding,i] for i in i_values]))
    if "right" in sides:
        bag.append(np.array([[MAX_SIZE-padding,i] for i in i_values]))
        
        
    if points.size == 0:
        return np.vstack(bag)
    else:
        return np.vstack([points,np.vstack(bag)])
    



#function to clip the points and targets to be within the image
def clip_points_targets(points,targets,MAX_SIZE=1024):
    """
    clip the points and targets to be within the image
    """
    targets = np.clip(targets,0,MAX_SIZE-1)
    points = np.clip(points,0,MAX_SIZE-1)
    return points,targets

