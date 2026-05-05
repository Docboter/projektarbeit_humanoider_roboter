FROM nvcr.io/nvidia/pytorch:24.02-py3

WORKDIR /workspace

COPY . .

RUN pip install --no-cache-dir \
    numpy \
    pyyaml

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

CMD ["bash"]