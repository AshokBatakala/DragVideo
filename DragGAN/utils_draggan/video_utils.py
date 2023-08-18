# imports 
import numpy as np
import PIL 
from PIL import Image
import os
import cv2
import matplotlib.pyplot as plt

    
#===============================================
# make GIF 
#===============================================
def make_gif(images, fname, duration=5, is_pil=True):
    import moviepy.editor as mpy
    
    def make_frame(t):
        try:
            x = images[int(len(images)/duration*t)]
        except:
            x = images[-1]
        
        # if true_image:
        #     return x.astype(np.uint8)
        # else:
        #     return ((x+1)/2*255).astype(np.uint8)
        
        # pil image to numpy array
        x = np.asarray(x)
        return x.astype(np.uint8)
        
        

    
    clip = mpy.VideoClip(make_frame, duration=duration)
    clip.write_gif(fname, fps = len(images) / duration)
    

def gif_from_folder(folder_path,gif_path,
                    duration=2, is_pil=True):
    image_dir = folder_path
    image_list = os.listdir(image_dir)
    image_list.sort()
    image_list = [os.path.join(image_dir, image) for image in image_list]
    images = [Image.open(image) for image in image_list]
    make_gif(images, gif_path, duration=duration, is_pil=is_pil)
    
#===============================================
# make video (avi)
#===============================================

def make_video(path,
                video_name='test_video',
                fps=24,
                ext=None,
                avi=False): 
    """Make a video from a folder of images.
    stores the video (.avi) in same folder as the images

    Args:
        path (str): Path to folder of images.
        fps (int): Frames per second of video. default=24
    """
    if ext is None:
        images = [img for img in os.listdir(path) ]
    else:
        images = [img for img in os.listdir(path) if img.endswith("." + ext)]
        
    images = sorted(images, key=lambda x: int(x.split(".")[0]))
    frame = cv2.imread(os.path.join(path, images[0]))
    height, width, layers = frame.shape

    avi_path = os.path.join(path, f"{video_name}.avi")
    video = cv2.VideoWriter(
        avi_path,
        cv2.VideoWriter_fourcc(*"DIVX"),
        fps,
        (width, height),
    )

    for image in images:
        video.write(cv2.imread(os.path.join(path, image)))

    cv2.destroyAllWindows()
    video.release()

    if not avi:
        # avi2mp4(avi_path=None)
        avi2mp4(avi_path=avi_path)
        # delete avi file
        os.remove(avi_path)
        print("avi file deleted")


# ==============================================
# Extract frames from video
# ==============================================
# video_path = "alien_girl.mp4"
# output_path = "frames/"

def extract_frames(video_path, output_path,n_digits_in_name=3,n_frames=None):
    """
    number of digits in name to make it easier to sort
    creates .jpg files
    """
    
    import cv2
    import os
    
    # Read the video from specified path
    cam = cv2.VideoCapture(video_path)
    # extract frames
    try:
        # creating a folder named data
        if not os.path.exists(output_path):
            os.makedirs(output_path)
    # if not created then raise error
    except OSError:
        print ('Error: Creating directory of data')
    # frame
    currentframe = 0
    n_frames = int(cam.get(cv2.CAP_PROP_FRAME_COUNT)) if n_frames is None else n_frames
    #log info 
    print(f" {int(cam.get(cv2.CAP_PROP_FRAME_COUNT))= } ,{n_frames=} ")
    while(currentframe < n_frames):
        # reading from frame
        ret,frame = cam.read()
        if ret:
            # if video is still left continue creating images
            name = os.path.join(output_path, str(currentframe).zfill(n_digits_in_name) + '.jpg')
            print ('Creating...' + name)
            # writing the extracted images
            cv2.imwrite(name, frame)
            # increasing counter so that it will
            # show how many frames are created
            currentframe += 1
        else:
            break


# ==============================================
# video side by side
# ==============================================

def video_side_by_side(video1_path, video2_path, output_path="./temp/",only_video=True):
    # imports 
    import cv2
    import os

    # Read the video from specified path
    cam1 = cv2.VideoCapture(video1_path)
    cam2 = cv2.VideoCapture(video2_path)

    # extract frames
    try:
        # creating a folder named data
        if not os.path.exists(output_path):
            os.makedirs(output_path)
    # if not created then raise error
    except OSError:
        print ('Error: Creating directory of data')
    # frame
    currentframe = 0
    while(True):
        # reading from frame
        ret1,frame1 = cam1.read()
        ret2,frame2 = cam2.read()
        if ret1 and ret2:
            # if video is still left continue creating images
            name = os.path.join(output_path, str(currentframe).zfill(3) + '.jpg')
            print ('Creating...' + name)
            # writing the extracted images
            frame = np.concatenate((frame1, frame2), axis=1)
            cv2.imwrite(name, frame)
            # increasing counter so that it will
            # show how many frames are created
            currentframe += 1
        else:
            break

    video1_name = os.path.basename(video1_path).split('.')[0]
    video2_name = os.path.basename(video2_path).split('.')[0]
    if only_video:
        make_video(output_path,video_name=f"{video1_name}_{video2_name}_s2s",fps=24,ext='jpg',avi=False)
        # delete frames
        list_of_images = [os.path.join(output_path, image) for image in os.listdir(output_path) if image.endswith(".jpg")]
        for image in list_of_images:
            os.remove(image)
        print("frames deleted")
    


# ==============================================
# avi to mp4
# ==============================================

def avi2mp4(avi_path=None):
    """ 
    ffmped must be installed
    sudo apt-get install ffmpeg

    """
    assert avi_path is not None, "avi_path is None"


    import os
    import subprocess
   
    
    # get video name
    video_name = os.path.basename(avi_path).split('.')[0]
    # get output path
    output_path = os.path.dirname(avi_path)
    # get output name
    output_name = video_name + '.mp4'
    # get output path
    output_path = os.path.join(output_path, output_name)
    # convert
    subprocess.call(['ffmpeg', '-i', avi_path, output_path])
    # delete avi file
    # os.remove(avi_path)
    print("avi file deleted")
    return output_path
