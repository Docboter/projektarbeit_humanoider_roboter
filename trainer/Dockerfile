FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3 python3-pip git ffmpeg libgl1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

RUN pip3 install --upgrade pip

COPY requirements.txt .
RUN pip3 install -r requirements.txt

RUN mkdir -p /workspace/data /workspace/output

COPY train.sh /workspace/train.sh
COPY train_policy.py /workspace/train_policy.py

RUN chmod +x /workspace/train.sh

ENTRYPOINT ["/workspace/train.sh"]