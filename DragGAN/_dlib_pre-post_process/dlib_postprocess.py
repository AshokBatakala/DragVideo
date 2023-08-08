import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
import pickle

def reverse_transform(full_image_path, crop_image_path, crop_pickle_path,
                      output_image_path="output.png"):


    crop_img = Image.open(crop_image_path)
    full_image = cv2.imread(full_image_path)
    with open(crop_pickle_path,'rb') as f:
        crop = pickle.load(f)

    org_size = full_image.shape # PIL IMage
    crop_size = crop_img.size # CV2 numpy array

    #pil_image = PIL.Image.open(crop_size).convert('RGB') 
    crop_img = np.array(crop_img) 
    # Convert RGB to BGR 
    crop_img = crop_img[:, :, ::-1].copy() 

    crop_img = cv2.cvtColor(crop_img, cv2.COLOR_RGB2RGBA)


    pt_A = [0, 0]
    pt_B = [0, crop_size[0]]
    pt_C = [crop_size[0], crop_size[0]]
    pt_D = [crop_size[0], 0]

    input_pts = np.float32([pt_A, pt_B, pt_C, pt_D])

    output_pts = np.float32(([crop[0], crop[1]],
                 [crop[0], crop[3]], 
                 [crop[2], crop[3]], 
                 [crop[2], crop[1]]))

    M = cv2.getPerspectiveTransform(input_pts,output_pts)
    #print(M.shape)
    #the canvas size is (width, height) format
    out = cv2.warpPerspective(crop_img,M,(org_size[1], org_size[0]),flags=cv2.INTER_LINEAR)

    plt.imshow(full_image)
    plt.imshow(out)
    plt.axis("off")
    plt.savefig(output_image_path)
    plt.show()