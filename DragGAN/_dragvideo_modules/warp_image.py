from PIL import Image
from molesq import ImageTransformer
import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch


#=== get control points ===
def get_controls_points(landmarks):
    offset = 50

    points = landmarks[0:17].copy()
    wide_jaw = False
    round_chin = True
    
    if wide_jaw:
        targets = landmarks[0:17].copy()
        targets[range(0,6)] -= np.array([offset, 0])
        targets[range(11,17)] += np.array([offset, 0])
        targets[range(6,11)] -= np.array([0, offset])
    
    #round chin
    if round_chin:
        centre = np.mean(points, axis=0)
        radius = np.linalg.norm(points[0] - centre)
        
        targets = np.zeros_like(points)
        for i in range(0, 17):
            vec = points[i] - centre
            vec /= np.linalg.norm(vec)
            targets[i] = centre + vec * radius
        
        
    return points, targets

#change coordinate system
def change_axis(points,max_y=1024):
    points = np.array(points)
    x = points[:,0]
    y = points[:,1]
    
    nex_x = max_y - y
    nex_y = x
    return np.array([nex_x,nex_y]).T

def load_and_resize_landmarks(landmarks_path, H,img_size_during_landmark_calculation=1024):
    #=== load landmarks ===
    with open(landmarks_path, 'rb') as f:
        landmarks = pickle.load(f)
        landmarks = np.array(landmarks)[0]
        
    #=== resize landmarks ===
    if H != img_size_during_landmark_calculation:
        landmarks = landmarks * H/img_size_during_landmark_calculation
        print(f"landmarks.shape: {landmarks.shape} \n resizing the landmarks to {img_size_during_landmark_calculation}")
        print("note: assuming SQUARE image")
    landmarks = landmarks.astype(np.int32)
    return landmarks

def deform(img, landmarks_path,img_size_during_landmark_calculation=1024):
    """
    img: np.array of shape (H, W, C)
    """
    img = np.array(img) #(1024, 1024, 3)
    
    
    H, W, C = img.shape
    print(f"H: {H}, W: {W}, C: {C}")
    
    landmarks = load_and_resize_landmarks(landmarks_path, H,
                                          img_size_during_landmark_calculation=img_size_during_landmark_calculation)


    old_points, old_targets = get_controls_points(landmarks)
    points, targets = change_axis(old_points,max_y=H), change_axis(old_targets,max_y=H)

    # Create a transformer object
    transformer = ImageTransformer(img, points, targets,color_dim=2,
                                        interp_order=2,)
    out = transformer.deform_viewport()
    
    # shift the image to right by 1/3 of the image width
    out = np.roll(out, W//3, axis=1)
    
    
    
    return out

def lazy_deform(*args, **kwargs):
    """
    accepts either a tensor, numpy array, file path or PIL image as input for img
    
    for tensor input, returns a tensor of the same shape and dtype
    """
    #check type of img
    if isinstance(args[0],torch.Tensor):
        tensor_shape, tensor_dtype, tensor_device = args[0].shape, args[0].dtype, args[0].device
        feature = args[0]
        
        #convert to numpy array
        img = feature.detach().cpu().numpy()
        

        
        #usually, the tensor is of shape (1,c,h,w)
        #we want to convert it to (h,w,c)
        # if last two dims equal then transpose
        if tensor_shape[-1] == tensor_shape[-2]:
            print("assuming tensor is of shape (1,c,h,w)")
            img = img[0].transpose(1,2,0)
            transpose_flag = True
        else:
            transpose_flag = False        
        
    elif isinstance(args[0],np.ndarray):
        img = args[0]
    elif isinstance(args[0],str):
        img = np.array(Image.open(args[0]))
    elif isinstance(args[0],Image.Image):
        img = np.array(args[0])
    else:
        print('Error: img should be either a tensor, numpy array, file path or PIL image')
        return None
    
    #check shape of img
    if len(img.shape) == 4:
        if img.shape[0] == 1:
            img = img[0]
        else:
            print('Batch size should be 1')
            return None
    elif len(img.shape) == 3:
        pass
    elif len(img.shape) == 2:
        img = np.expand_dims(img, axis=2)
        
    # check if square
    if img.shape[0] != img.shape[1]:
        print('Error: img should be square')
        return None
    
    #change dtype to float32
    if img.dtype != np.float32:
        img = img.astype(np.float32)
        
    print("giving img to deform... of shape: ", img.shape,"type: ", img.dtype,type(img))
    #pass to deform
    output =  deform(img, *args[1:], **kwargs)

    print("deform done...")
    if isinstance(args[0],torch.Tensor):
        if transpose_flag:
            output = output.transpose(2,0,1)

        output = torch.tensor(output).unsqueeze(0)

        output = output.to(tensor_device)
        output = output.to(tensor_dtype)
        output = output.view(tensor_shape)
        
    return output





if __name__ == '__main__':
    img_path = "/home/bean/DragVideo/Data_store/experiments/2023-09-06_00-38-36_actress_smile/aligned/000.jpg"
    landmarks_path = "/home/bean/DragVideo/Data_store/experiments/2023-09-06_00-38-36_actress_smile/landmarks/000.pkl"
    img = Image.open(img_path)

    # img = set_feature['feature']
    out = deform(img_path, landmarks_path)

    # out = transformer.deform_whole()
    plt.imshow(out)
    plt.axis('off')
    plt.show()