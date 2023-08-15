#===================================================================================================
#                                          dockerfile for DragVideo
#===================================================================================================
#creates user  "bean" installs conda and adds conda to path.
#user bean can use sudo without password

# build using 
# docker build --network=host -t dragvideo:latest  .

# install Draggan, PTI, landmarks dependencies

# run it
#docker run --gpus all --ipc=host --net=host  -it -v $PWD:/home/bean/DragVideo   dragvideo:latest  /bin/bash

# run with gpu:1 ; for A5000 gpu on .7 mechine
#docker run --gpus '"device=1"' --ipc=host --net=host  -it -v $PWD:/home/bean/DragVideo   dragvideo:latest  /bin/bash

# --runtime=nvidia  ( i am not sure about this. but if not set as default. include it.)

#===================================================================================================

FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04

# FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04 



# Install base utilities
RUN apt-get update \
    && apt-get install -y build-essential \
    && apt-get install -y wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# install sudo 
RUN apt-get update && apt-get install -y sudo


# install basic utils like ls, vim, etc
RUN apt-get update && sudo apt-get install -y \
    apt-utils\
    git \
    curl \
    vim \
    wget \
    unzip \
    bzip2 \
    ca-certificates \
    libx11-6 \
    && sudo rm -rf /var/lib/apt/lists/*




# Install miniconda
ENV CONDA_DIR /opt/conda
RUN  wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda

# # Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH


# create a user named "bean"
# RUN useradd -ms /bin/bash bean
RUN useradd -m bean && echo "bean:bean" | chpasswd && adduser bean sudo

# passwordless sudo
RUN echo "bean ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
# add user to sudo group
RUN adduser bean sudo
USER bean
WORKDIR /home/bean/DragVideo


# # Create conda environment
# COPY e4e_env.yaml .
# RUN conda env create -f e4e_env.yaml

# # final setup
# RUN echo "source activate e4e_env" >> ~/.bashrc
# ENV PATH /opt/conda/envs/e4e_env/bin:$PATH

# add conda path to bashrc
RUN echo "export PATH=/opt/conda/bin:$PATH" >> ~/.bashrc

# copy Dockervideo.yml then create conda env
# COPY Dragvideo.yml .
# RUN conda env create -f Dragvideo.yml


#start with bash
CMD [ "/bin/bash" ]


#===================================================================================================
#                                          instructions
#===================================================================================================
# it has conda installed


# build using 
# docker build -t dragvideo:latest  .

# install Draggan, PTI, landmarks dependencies

# run it
#docker run --gpus all --ipc=host --net=host  -it -v $PWD:/home/bean/DragVideo   dragvideo:latest  /bin/bash