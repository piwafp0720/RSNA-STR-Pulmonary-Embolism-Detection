FROM nvidia/cuda:10.2-devel-ubuntu18.04
ENV CUDNN_VERSION 7.6.5.32
RUN apt-get update && apt-get install -y --no-install-recommends \
    libcudnn7=$CUDNN_VERSION-1+cuda10.2 \
libcudnn7-dev=$CUDNN_VERSION-1+cuda10.2 \
&& \
    apt-mark hold libcudnn7 && \
    rm -rf /var/lib/apt/lists/*

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y tzdata
RUN apt-get update && apt-get install -y  \
    wget \
    curl \
    make \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    llvm \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libffi-dev \
    liblzma-dev \
    vim \
    git \
    byobu \
    htop \
    zip \
    unzip
RUN apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /usr/local/src/*

RUN wget https://www.python.org/ftp/python/3.7.4/Python-3.7.4.tar.xz \
    && tar xvf Python-3.7.4.tar.xz \
    && cd Python-3.7.4 \
    && ./configure --enable-optimizations \
    && make altinstall
RUN rm Python-3.7.4.tar.xz

RUN ln -s /usr/local/bin/python3.7 /usr/local/bin/python
RUN ln -s /usr/local/bin/pip3.7 /usr/local/bin/pip

RUN pip install flake8 yapf mypy autopep8 isort
RUN pip install numpy==1.16.4
RUN pip install scipy pandas
RUN pip install pillow==6.2.1
RUN pip install opencv-python
RUN pip install scikit-learn==0.22.2
RUN pip install tqdm==4.41
RUN pip install omegaconf
RUN pip install torch torchvision
RUN pip install pytorch-lightning==0.8.5
RUN pip install test-tube
RUN pip install albumentations pretrainedmodels timm
RUN pip install jupyter
RUN pip install git+https://github.com/NVIDIA/apex -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext"
RUN pip install wandb
RUN pip install pydicom
RUN pip install vtk

# install gdcm for pydicom.
# https://github.com/HealthplusAI/python3-gdcm
RUN git clone --branch master https://github.com/HealthplusAI/python3-gdcm.git && cd python3-gdcm && dpkg -i build_1-1_amd64.deb && apt-get install -f
RUN cp /usr/local/lib/gdcm.py /usr/local/lib/python3.7/site-packages/.
RUN cp /usr/local/lib/gdcmswig.py /usr/local/lib/python3.7/site-packages/.
RUN cp /usr/local/lib/_gdcmswig.so /usr/local/lib/python3.7/site-packages/.
RUN cp /usr/local/lib/libgdcm* /usr/local/lib/python3.7/site-packages/.
RUN ldconfig

RUN apt-get update && apt-get install -y libturbojpeg
RUN pip install -U git+https://github.com/lilohuang/PyTurboJPEG.git

RUN pip uninstall dataclasses -y

ENV PYTHONPATH "${PYTHONPATH}:/usr/local/lib/python3.7/site-packages"
CMD ["/bin/bash"]
