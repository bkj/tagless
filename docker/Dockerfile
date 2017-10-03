FROM nvidia/cuda:8.0-cudnn5-devel

# Install dependencies
RUN apt-get -y update
RUN apt-get install -y git wget

# Install anaconda
RUN wget https://repo.continuum.io/archive/Anaconda2-4.4.0-Linux-x86_64.sh
RUN bash Anaconda2-4.4.0-Linux-x86_64.sh -b -p /root/.anaconda
RUN rm Anaconda2-4.4.0-Linux-x86_64.sh
RUN echo 'export PATH="/root/.anaconda/bin:$PATH"' >> ~/.bashrc 
ENV PATH="/root/.anaconda/bin:${PATH}"

# Download qnas + datasets
RUN mkdir -p /root/projects/
WORKDIR /root/projects
RUN git clone https://github.com/bkj/tagless -b simple
WORKDIR /root/projects/tagless/
RUN pip install -r requirements.txt
RUN python setup.py install