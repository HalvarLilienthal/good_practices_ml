FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu20.04

RUN apt-get update && apt-get install -y \
    git \
    python3.8 \
    python3.8-dev \
    python3.8-venv \
    python3-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

#RUN git clone https://github.com/HalvarKelm/good_practices_ml.git . --depth 1
COPY . .
RUN pip install --no-cache-dir -r requirement.txt
