FROM tensorflow/tensorflow:nightly-gpu-py3

RUN pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip3 install librosa matplotlib -y
RUN pip3 install mir_eval -y
RUN pip3 install pystoi pysoundfile -y
RUN apt-get update
RUN apt-get install -y vim
RUN apt-get install -y screen
