# implemented on .182 

1. build docker container, run it 
2. create conda env with env.yml 
3. install torch # conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
4. install requirements.
5. install pti_requirements

to commit 
docker container commit -m "cv2" 18449b9d8dcb dragvideo:v4_
docker run --gpus all --ipc=host --net=host  -it -v $PWD:/home/bean/DragVideo   dragvideo:v4_  /bin/bash



docker images 
dragvideo:v1 = just conda is installed 
dragvideo:v2 = complete draggan env is installed in conda: stylegan3 ( yml + requirements )
dragvideo:v3 = pti_requirements
dragvideo:v4 = cv2 (RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y)
