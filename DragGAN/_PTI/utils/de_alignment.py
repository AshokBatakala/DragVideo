#Author: B Ashok
#Date created: Aug 18, 2023.
#---------------------------------------------

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle
from skimage.transform import warp, AffineTransform


def put_back_the_edited_image(raw_image_path,
                              aligned_image_path,
                              quad_pkl_path,
                              save_path=None,
                              show=False):
    """ 
    needs quad values before cropping.
    """
    with open(quad_pkl_path,'rb') as f:
        quad = pickle.load(f)
    quad = np.array(quad)
    quad = quad+0.5 # preprocessing is done in this way
    # Load the image
    aligned_image = cv2.imread(aligned_image_path,cv2.IMREAD_UNCHANGED) # it reads alpha channel too
    aligned_image = cv2.cvtColor(aligned_image,cv2.COLOR_BGR2RGB)

    raw_image = cv2.imread(raw_image_path, cv2.IMREAD_UNCHANGED)
    raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)

    H,W,C = aligned_image.shape #(1024, 1024, 3)
    raw_H,raw_W,_ = raw_image.shape

    # quad # order of points: top left, bottom left,bottom right, top right 
    input_pts = np.float32([[0,0],[0,H],[W,H],[W,0]])
    output_pts = np.float32(quad)
    #-- using cv2 --
    # Compute the perspective transform M
    # M = cv2.getPerspectiveTransform(input_pts,output_pts)
    # edited_part = cv2.warpPerspective(src=aligned_image, M=M, dsize=(raw_W,raw_H), flags=cv2.INTER_LINEAR)
        
    tform = AffineTransform()
    tform.estimate(input_pts, output_pts)
    edited_part = warp(aligned_image, tform.inverse, output_shape=(raw_H,raw_W),mode = 'reflect')
    edited_part = (edited_part*255).astype(np.uint8)
         
    mask = create_mask_from_quad(quad,shape=edited_part.shape)
    final_image = combine_with_mask(edited_part,raw_image,mask)
    
    if show:
        # plt.imshow(cv2.cvtColor(final_image,cv2.COLOR_BGR2RGB))
        plt.imshow(final_image)
        return final_image
    
    if save_path is not None:
        # change the color space to BGR
        final_image = cv2.cvtColor(final_image,cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path,final_image)
        
# create a mask with region inside a 4 points polygon
def create_mask_from_quad(quad_points,shape,blur_kernel_size=51):
    """
    quad_points: numpy array of shape (4,2)
    shape: (H,W)
    return: binary mask of shape
            with dtype=np.uint8
            
    #usage:
        result = cv2.bitwise_and(image, mask)
    #to get inverse mask
        mask_inv = cv2.bitwise_not(mask)

    """    
    img = np.zeros(shape,dtype=np.uint8)
    # print("dtype of img:",img.dtype,"dtype of quad_points:",quad_points.dtype)
    quad_points = quad_points.astype(np.int32)
    cv2.fillPoly(img, pts =[quad_points], color =(255,255,255))
    # cv2.polylines(img,pts =[quad_points],isClosed=True,color=(255,255,255),thickness=2)
    # img = cv2.GaussianBlur(img, (blur_kernel_size, blur_kernel_size), 0)


    return img#[:,:,np.newaxis].astype(np.uint8)
    
def combine_with_mask(img1,img2,mask):
    """
    returns img1*mask + img2*(1-mask)
    args: numpy arrays
    #usage:
        combine_with_mask(edited,raw,mask)
    """
    inv_mask = cv2.bitwise_not(mask)
    return cv2.bitwise_and(img1,mask) + cv2.bitwise_and(img2, inv_mask)

