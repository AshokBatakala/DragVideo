# implemented on .182 

1. build docker container, run it 
2. create conda env with env.yml 
3. install torch # conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
4. install requirements.
5. install pti_requirements

to commit 
docker container commit -m "pyrallis" 161884d4f309 dragvideo:v5
docker run --gpus all --ipc=host --net=host  -it -v $PWD:/home/bean/DragVideo   dragvideo:v4  /bin/bash


docker images 
dragvideo:v1 = just conda is installed 
dragvideo:v2 = complete draggan env is installed in conda: stylegan3 ( yml + requirements )
dragvideo:v3 = pti_requirements
dragvideo:v4 = cv2 (RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y)
dragvideo:v5 = pyrallis


example run 
python prepare_data/preparing_faces_parallel.py --mode align --root_path /home/bean/DragVideo/Data_store/delete/raw


# To run gradio  use draggan:latest ( both sg2/sg3 are tested)
----------------------------------------
docker run --gpus all --ipc=host --net=host  -it -v $PWD:/home/bean/DragVideo   draggan:latest  /bin/bash
cd /home/bean/DragVideo/DragGAN
python visualizer_drag_gradio.py --listen

#tested checkpoints 
1.stylegan3-r-ffhq-1024_module
2.stylegan3-r-ffhq-1024x1024

