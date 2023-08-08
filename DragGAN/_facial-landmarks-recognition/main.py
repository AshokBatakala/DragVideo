## code for imagefiles
# import the necessary packages
from imutils import face_utils
import imutils
import dlib
import cv2
import os
import pickle
import matplotlib.pyplot as plt



def landmarks(image_path,p= "shape_predictor_68_face_landmarks.dat"):
    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(p)

    # load the input image, resize it, and convert it to grayscale
    image = cv2.imread(image_path)
    # image = imutils.resize(image, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale image
    rects = detector(gray, 1) 
    # rect per face

    # only taking the first face in the image for now
    # rect = rects[0]

    # shape = predictor(gray, rect)
    # shape = face_utils.shape_to_np(shape)
    shapes = []
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        shapes.append(shape)

    return shapes



def get_landmarks_dir(image_dir,landmark_dir):
    for image_name in os.listdir(image_dir):
        image_path = os.path.join(image_dir,image_name)
        shapes = landmarks(image_path)
        name = image_name.split(".")[0]
        landmark_path = os.path.join(landmark_dir,name+".pkl")

        # pickle.dump(shapes,open(landmark_path,"wb"))
        # print("landmark_path: ",landmark_path)
        with open(landmark_path, 'wb') as f:
            pickle.dump(shapes, f)
        print("landmark_path: ",landmark_path)
        

"""
facial landmarks indexes
jaw: 0-16
right eyebrow: 17-21
left eyebrow: 22-26
nose: 27-35
right eye: 36-41
left eye: 42-47
mouth: 48-67
"""

dict_landmarks = {'jaw': range(0, 17),
                    'right_eyebrow': range(17, 22),
                    'left_eyebrow': range(22, 27),
                    'nose': range(27, 36),
                    'right_eye': range(36, 42),
                    'left_eye': range(42, 48),
                    'mouth': range(48, 68)}



def show_landmarks(image_path, shape):
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