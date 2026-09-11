# Офіційний образ NVIDIA з підтримкою CUDA та TensorRT
FROM nvcr.io/nvidia/tensorrt:24.01-py3

# Встановлюємо системні залежності, C++20 (GCC 11+), CMake та Python
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    g++-11 \
    git \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Налаштування C++20 за замовчуванням
ENV CXX=g++-11

# Оновлення pip та встановлення базових Python-пакетів
WORKDIR /workspace
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt