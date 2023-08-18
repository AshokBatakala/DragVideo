#Author: B Ashok
#Date created: Aug 18, 2023.
#---------------------------------------------

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle

def put_back_the_edited_image(raw_image_path,aligned_image_path,quad_pkl_path,save_path=None,show=False):
    """ 
    needs quad values before cropping.
    """
    with open(quad_pkl_path,'rb') as f:
        quad = pickle.load(f)
    quad = np.array(quad)
    quad = quad+0.5 # preprocessing is done in this way
    # Load the image
    aligned_image = cv2.imread(aligned_image_path,cv2.IMREAD_UNCHANGED) # it reads alpha channel too
    aligned_image = cv2.cvtColor(aligned_image,cv2.COLOR_BGR2RGBA)

    raw_image = cv2.imread(raw_image_path, cv2.IMREAD_UNCHANGED)
    raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGBA)

    H,W,C = aligned_image.shape #(1024, 1024, 3)
    raw_H,raw_W,_ = raw_image.shape

    # quad # order of points: top left, bottom left,bottom right, top right 


    input_pts = np.float32([[0,0],[0,H],[W,H],[W,0]])
    output_pts = np.float32(quad)
    # Compute the perspective transform M
    M = cv2.getPerspectiveTransform(input_pts,output_pts)

    # Apply the perspective transformation to the image
    # using both source and destination images
    # make alpha of img=0

    # out = cv2.warpPerspective(src=aligned_image,dst=raw_image, M=M, dsize=(raw_W,raw_H), flags=cv2.INTER_LINEAR)
    out = cv2.warpPerspective(src=aligned_image, M=M, dsize=(raw_W,raw_H), flags=cv2.INTER_LINEAR)
    
    mask = out[:,:,3][:,:,np.newaxis] / 225.0
    final_image = (mask*out[:,:,:3] + (1-mask)*raw_image[:,:,:3])/255.0
    final_image = (final_image*255).astype(np.uint8)

    if show:
        # plt.imshow(cv2.cvtColor(final_image,cv2.COLOR_BGR2RGB))
        plt.imshow(final_image)
        return final_image
    
    if save_path is not None:
        # change the color space to BGR
        final_image = cv2.cvtColor(final_image,cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path,final_image)
    