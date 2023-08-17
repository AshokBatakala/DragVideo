
from configs import paths_config
import dlib
import glob
import os
from tqdm import tqdm
from utils.alignment import align_face
import pickle



def pre_process_images(raw_images_path,
                       save_output_path=paths_config.input_data_path,
                       save_quad_values_path=paths_config.quad_values_path,
                        IMAGE_SIZE=1024,):
    
    current_directory = os.getcwd()
    predictor = dlib.shape_predictor(paths_config.dlib)
    os.chdir(raw_images_path)
    images_names = glob.glob(f'*')

    aligned_images = []
    quad_values = []

    for image_name in tqdm(images_names):
        try:
            aligned_image,quad_value = align_face(filepath=f'{raw_images_path}/{image_name}',
                                       predictor=predictor, output_size=IMAGE_SIZE)
            
            aligned_images.append(aligned_image)
            quad_values.append(quad_value)
        except Exception as e:
            print(e)

    os.makedirs(save_output_path, exist_ok=True)
    for image, name in zip(aligned_images, images_names):
        real_name = name.split('.')[0]
        image.save(f'{save_output_path}/{real_name}.jpg')

    os.makedirs(save_quad_values_path, exist_ok=True)
    for quad_value, name in zip(quad_values, images_names):
        real_name = name.split('.')[0]
        with open(f'{save_quad_values_path}/{real_name}.pkl', 'wb') as f:
            pickle.dump(quad_value, f)


    os.chdir(current_directory)


if __name__ == "__main__":
    
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_images_path', type=str, default='/home/bean/DragVideo/PTI/data/raw_images')
    parser.add_argument('--IMAGE_SIZE', type=int, default=1024)
    parser.add_argument('--save_output_path', type=str, default='/home/bean/DragVideo/PTI/data/input_data')
    args = parser.parse_args()
    
    pre_process_images(args.raw_images_path,
                       args.save_output_path,
                          args.IMAGE_SIZE,
                            )
    

    
    # sample command ( well formatted in multiple lines )
    # python utils/align_data.py \
    # --raw_images_path /home/bean/DragVideo/PTI/data/raw_images \
    # --IMAGE_SIZE 1024 \
    # --save_output_path /home/bean/DragVideo/PTI/data/input_data
    
    
    
