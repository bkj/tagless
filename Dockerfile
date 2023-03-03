FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-devel

# --
# system dependencies

RUN apt clean
RUN apt update ; exit 0
RUN apt install -y git

# --
# python dependencies

RUN pip install pandas
RUN pip install ftfy
RUN pip install regex
RUN pip install tqdm
RUN pip install arrow
RUN pip install rich
RUN pip install flask
RUN pip install git+https://github.com/openai/CLIP.git
RUN pip install scikit-learn

# --
# model dependencies

RUN apt install -y wget
RUN mkdir -p /root/.cache/clip
RUN wget https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt \
    -O /root/.cache/clip/ViT-L-14-336px.pt

# --
# add code

RUN mkdir /tagless
ADD ./ /tagless
WORKDIR /tagless
RUN pip install -e .

# --
# run

RUN /bin/bash