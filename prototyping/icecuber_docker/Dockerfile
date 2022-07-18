FROM ubuntu:bionic

RUN apt-get update && \
    apt-get install -y build-essential python3 python3-pip vim unzip git && \
    pip3 install psutil

ENV LC_ALL=C.UTF-8

ENV SRC_DIR=/root/arc
RUN mkdir -p $SRC_DIR && \
    cd $SRC_DIR && \
    git clone https://github.com/fchollet/ARC.git && \
    git clone https://github.com/top-quarks/ARC-solution

WORKDIR $SRC_DIR/ARC-solution
RUN make -j && \
    make -j count_tasks

# Figure out which riddles of the public dataset were correctly solved by the icecuber solution:
# - rename test directory to test_old
# - copy training and evaluation riddles into test directory
RUN mv /root/arc/ARC-solution/dataset/test /root/arc/ARC-solution/dataset/test_old && \
    mkdir /root/arc/ARC-solution/dataset/test && \
    cp /root/arc/ARC/data/evaluation/*.json /root/arc/ARC-solution/dataset/test && \
    cp /root/arc/ARC/data/training/*.json /root/arc/ARC-solution/dataset/test

ENTRYPOINT /bin/bash