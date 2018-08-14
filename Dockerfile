FROM nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04

RUN rm /bin/sh && ln -s /bin/bash /bin/sh

# Install pip
RUN apt-get update
RUN apt-get -y install python3 python3-pip python3-dev python3-tk
RUN apt-get -y install libglu1-mesa libxi-dev libxmu-dev libglu1-mesa-dev

# Install basic libraries
RUN pip3 install --upgrade pip
RUN pip3 install numpy tensorflow-gpu==1.9 matplotlib scipy scikit-learn future

# Install MuJoCo + OpenAI gym
RUN pip3 install gym==0.9.4
RUN apt-get update
RUN apt-get -y install unzip unetbootin wget
RUN mkdir -p /.mujoco && cd /.mujoco && wget https://www.roboti.us/download/mjpro131_linux.zip && unzip mjpro131_linux.zip
ENV MUJOCO_PY_MJKEY_PATH="/root/.mujoco/mjkey.txt"
ENV MUJOCO_PY_MJPRO_PATH="/root/.mujoco/mjpro131"
RUN pip3 install mujoco-py==0.5.7

# Install additional requirements
RUN pip3 install datetime gitpython h5py tqdm dotmap cython

# GPFlow
RUN apt-get -y install git
RUN git clone https://github.com/GPflow/GPflow.git
RUN pip3 install pandas multipledispatch pytest
RUN cd GPflow/ && pip install . --no-deps

# Create copy of Deep MBRL repo and place in ~/handful-of-trials
RUN cd ~ && git clone https://github.com/kchua/handful-of-trials.git

# Environment setup
RUN echo 'LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu' >> /root/.bashrc
RUN echo 'alias python=python3' >> /root/.bashrc

CMD /bin/bash
