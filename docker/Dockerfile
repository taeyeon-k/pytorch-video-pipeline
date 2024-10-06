FROM nvcr.io/nvidia/deepstream:6.3-gc-triton-devel

RUN apt-get update && apt install --no-install-recommends -y \
    ca-certificates \
    python3-gst-1.0 \
    wget

# allow GObject to find typelibs
ENV GI_TYPELIB_PATH /usr/lib/x86_64-linux-gnu/girepository-1.0/

# use conda to simplify some dependency managemeny
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh

ENV PATH /opt/conda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/cuda-11.8/compat:/opt/conda/lib:/opt/nvidia/deepstream/deepstream-6.3/lib:${LD_LIBRARY_PATH}

RUN conda install -y -c pytorch -c nvidia\
    cudatoolkit==11.8 \
    pytorch==2.4.1 \
    torchvision==0.19.1 \
    pytorch-cuda=11.8

RUN conda install -y -c conda-forge \
    pygobject \
    scikit-image \
    ncurses

# Check the current version
# RUN nvcc --version
# RUN ls /usr/local/cuda-11.8
# RUN python -c "import torch; print(torch.__version__) ; print(torch.cuda.is_available())"
# RUN python -c "import torchvision; print(torchvision.__version__)"

# Nvidia Apex for mixed-precision inference
RUN git clone https://github.com/NVIDIA/apex.git /build/apex
WORKDIR /build/apex
RUN python3 setup.py install

RUN pip install --upgrade cython
RUN pip install --upgrade gil_load

# Gstreamer debug output location
env GST_DEBUG_DUMP_DOT_DIR=/app/logs

RUN python -c "import torch; torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd', model_math='fp32')" 2>/dev/null | :
RUN python -c "import torch; torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd', model_math='fp16')" 2>/dev/null | :

RUN rm -rf /var/lib/apt/lists/* && \
    conda clean -afy

WORKDIR /app
