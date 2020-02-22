# FROM nvidia/cuda:9.0-cudnn7-devel
FROM nvidia/cuda:10.0-cudnn7-devel

# apt install python3.6
RUN apt update
RUN apt install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN add-apt-repository ppa:jonathonf/vim
RUN apt update && apt install -y python3.6 python-dev python3.6-dev python3-pip
RUN apt install -y vim libsm6 libxext6 libxrender-dev build-essential libssl-dev libffi-dev libxml2-dev libxslt1-dev zlib1g-dev graphviz curl
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 10
RUN pip3 install --upgrade pip
RUN apt install -y sudo
ENV DEBIAN_FRONTEND=noninteractive
RUN apt install -y wget git locales
RUN locale-gen "en_US.UTF-8"
RUN update-locale LC_ALL="en_US.UTF-8"

RUN pip3 install keras pillow matplotlib scikit-learn scikit-image opencv-python pydot GraphViz
RUN pip3 install PyGLM PySide2 pyopengl
# RUN pip3 install tensorflow-gpu==1.15
RUN pip3 install tensorflow-gpu==1.13.2

#setting USER group number
ENV USER docker
RUN echo "${USER} ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers.d/${USER}
RUN chmod u+s /usr/sbin/useradd \
   && chmod u+s /usr/sbin/groupadd
ENV HOME /home/${USER}
ENV SHELL /bin/bash
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8
ENV TERM xterm-256color
RUN echo 'Defaults visiblepw' >> /etc/sudoers
RUN echo 'USER ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER ${USER}

# fix keras to save models
# https://github.com/keras-team/keras/issues/9342
COPY saving.py /usr/local/lib/python3.6/dist-packages/keras/engine/saving.py

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["bash"]

WORKDIR /workspace