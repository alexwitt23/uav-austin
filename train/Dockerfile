FROM pytorch/pytorch:nightly-devel-cuda10.0-cudnn7

WORKDIR /tmp 

RUN apt-get update \
    && apt-get install -y \
	    python3 \
        python3-pip \
    && rm -rf /var/lib/apt/lists/* \ 
    && pip3 install --upgrade pip 

RUN git clone https://github.com/NVIDIA/apex && \
    cd apex && \
    pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

