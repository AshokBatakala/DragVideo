import matplotlib.pyplot as plt

dict_landmarks = {'jaw': range(0, 17),
                    'right_eyebrow': range(17, 22),
                    'left_eyebrow': range(22, 27),
                    'nose': range(27, 36),
                    'right_eye': range(36, 42),
                    'left_eye': range(42, 48),
                    'mouth': range(48, 68)}



def show_landmarks(image_path, shape):
    import cv2

    # raed image 
    image = cv2.imread(image_path)

    for (x, y) in shape:
        cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

    # %matplotlib inline
    # show image 
    # cv2.imshow("Output", image)
    # import matplotlib.pyplot as plt
    # show using plt 
    # plt.imshow(image)# channels are messed up
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()

# for use after loading the landmarks in draggan
def list2dict(points,targets):
    assert len(points) == len(targets)
    dict = {}
    for i in range(len(points)):
        dict[i] = {'start': points[i], 'target': targets[i]}
    return dict

#===================================================================================================
#                   run styleggan on a latent vector
#===================================================================================================

def run_SG(SG_path,w_path,ws_size = None,points = None):
    # for 1024 w shape: [1, 18, 512]
    import torch
    from visualizer_auto import DragVideo
    # from visualizer_auto import DragVideo
    w_load = torch.load(w_path)
    if ws_size is not None:
        w_load = w_load[:,:ws_size,:] # ws shape is 16 for sg3

    drag_video = DragVideo(w_load=w_load,
                        stylegan2_wieghts_path=SG_path)
    # drag_video = DragVideo()
    if points is not None:
        feat = drag_video.run(N_STEPS=1,points=points)
    return drag_video.global_state['images']['image_show']

#===================================================================================================
#                   Encoder : img to w
#===================================================================================================
