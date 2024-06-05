
FROM nvcr.io/nvidia/pytorch:22.09-py3

WORKDIR /workspace/llumnix

COPY . /workspace/llumnix

RUN apt-get update

RUN conda create -n artifact python=3.8

RUN conda run -n artifact pip install -e .

RUN echo "source activate artifact" > ~/.bashrc
ENV PATH /opt/conda/envs/artifact/bin:$PATH

RUN apt-get install bc
