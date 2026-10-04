# Офіційний образ NVIDIA з підтримкою CUDA та TensorRT
FROM nvcr.io/nvidia/tensorrt:26.09-py3

# Встановлюємо системні залежності, CMake, Python та графічний стек Qt6/PySide6
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    curl \
    libgl1 \
    libglib2.0-0 \
    python3-pip \
    libfontconfig1 \
    libfreetype6 \
    libx11-xcb1 \
    libxrender1 \
    libxext6 \
    libegl1 \
    libdbus-1-3 \
    libxkbcommon-x11-0 \
    libxkbcommon0 \
    libxcb-cursor0 \
    libxcb-shape0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-xinerama0 \
    libxcb-xfixes0 \
    && rm -rf /var/lib/apt/lists/*

# Встановлення стандартного CXX (системний g++ підтримує C++20 за замовчуванням)
ENV CXX=g++

# Оновлення pip та встановлення базових Python-пакетів
WORKDIR /workspace
COPY requirements.txt .
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt