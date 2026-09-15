# Про проєкт
Головною метою проєкту є створення високопродуктивного, відмовостійкого та апаратно-оптимізованого програмного рішення для автономного виявлення малорозмірних об'єктів на аерофотознімках (VisDrone, DOTA) в умовах суворих обмежень обчислювальних ресурсів бортових та edge-пристроїв.
# Key Architectural PrinciplesHardware-Aware Dynamic Tiling
1. Відмова від статичної сітки (SAHI) на користь розрахунку оптимального розміру плитки ($T_{calc}$) та перекриття ($O_{lap}$) залежно від телеметрії БПЛА (висоти польоту $H$), роздільної здатності кадру ($4K/8K$) та поточного обсягу вільної відеопам'яті ($\text{VRAM}_{\text{avail}}$).  
2. In-Memory Zero-Copy Slicing: Повний відмова від створення тимчасових файлів (.jpg/.png) на диску. Фрагментація здійснюється виключно у VRAM за допомогою GPU-вказівників та CUDA-зрізів (Sub-tensors).
3. Metadata-Driven UI (Non-Destructive Overlay): Обчислювальне ядро повертає на Host лише легкий вектор координат у форматі JSON. Нативний GUI (PySide6 QGraphicsView) рендерить обмежувальні рамки векторним шаром поверх чистого оригінального $8K$ кадру.
4. Принцип DRY через libtiling_core: Єдина низькорівнева C++/CUDA бібліотека забезпечує ідентичність алгоритмів нарізки як у високошвидкісному C++ Inference Engine, так і в Python Training Pipeline (через pybind11).
5. Dual Inference Engine Support: Пряме виконання через TensorRT C++ API на Linux/Jetson з автоматичним фолбеком на DirectML (ONNX Runtime) під Windows 10/11.
# Стек технологій
 - Мови розробки: C++20 (Обчислювальне ядро, математика, пам'ять), Python 3.11 (GUI, конвеєр навчання, пакування).
 - C++ Core Engine: C++20 / CUDA C++ (libtiling_core).
 - Inference Runtime: NVIDIA TensorRT C++ API (FP16/INT8) / DirectML (ONNX Runtime).
 - Python Binding: pybind11 (pytiling_core).
 - GUI Framework: PySide6 (Qt6 for Python) + QGraphicsView (Hardware-Accelerated Canvas Viewer).
 - Training Pipeline: Python / PyTorch / Ultralytics YOLO26 (NMS-Free & STAL) / YOLO11s.  
 - Post-Processing: C++ Cluster-DIoU-NMS + Host-Device Offset Mapping.
 - Build System & Environment: CMake 3.20+, GCC 11+, CUDA Toolkit 13.2+ / 12.8+, Docker (Dev Containers).
 - Distribution: Autonomous Standalone Bundles (Nuitka / PyInstaller — .exe для Windows, .AppImage для Linux).
# Структура проєкту
```text
├── .devcontainer/              # Налаштування відтворюваного середовища Dev Container
│   ├── devcontainer.json       # Конфігурація розширення VS Code
│   └── Dockerfile              # Docker-образ (Ubuntu 22.04, CUDA 12.8+, GCC-11, TensorRT)
├── include/                    # C++ Header файли
│   ├── tiling_math.hpp         # Динамічний розрахунок сітки T_calc та O_lap
│   ├── cuda_slicer.cuh         # CUDA-ядра для Zero-Copy Slicing у VRAM
│   ├── trt_detector.hpp        # C++ Wrapper над TensorRT API (enqueueV3)
│   └── postprocess.hpp         # Offset Mapping та Cluster-DIoU-NMS
├── src/                        # C++ та CUDA реалізація модулів ядра
│   ├── tiling_math.cpp
│   ├── cuda_slicer.cu
│   ├── trt_detector.cpp
│   └── postprocess.cpp
├── bindings/                   # Python bindings через pybind11
│   └── pybind_wrapper.cpp      # Модуль pytiling_core
├── gui/                        # PySide6 Графічний інтерфейс користувача
│   ├── main_window.py          # Головна форма додатка
│   ├── canvas_viewer.py        # QGraphicsView для апаратно-прискореного 8K зумінгу
│   └── async_worker.py         # QThread асинхронний конвеєр обробки
├── tests/                      # Модульні та системні тести (CTest)
│   ├── test_tiling_math.cpp
│   ├── test_cuda_slicer.cpp
│   └── test_trt_detector.cpp
├── .aider.conventions.md       # Конвенції розробки для AI-агента (aider)
├── .aider.conf.yml             # Автоматизація авто-тестування для aider
├── CMakeLists.txt              # Головний конфігураційний файл збірки CMake
└── requirements.txt            # Залежності Python
```
# Математична модель та геометрія
## 1. Динамічний розрахунок сітки ($T_{calc}$ та $O_{lap}$)
Планувальник обчислює математичний розмір плитки $T_{calc}$ та відсоток перекриття $O_{lap}$ на основі телеметрії висоти $H$, доступної пам'яті $\text{VRAM}_{\text{avail}}$ та роздільної здатності кадру:
$$T_{calc} = f(H, \text{VRAM}_{\text{avail}}, \text{Res}_{img})$$
Система здійснює дискретний вибір цільової конфігурації з наявного пулу експортованих TensorRT моделей $M \in \{320, 416, 512, 640\}$:
  $$M_{selected} = \max \{ m \in M \mid m \le T_{calc} \}$$
## 2. Host-Device Offset Mapping
Перерахунок локальних координат об'єкта $(x_{local}, y_{local}, w, h)$ з плитки $Tile_i(X_0, Y_0, W_t, H_t)$ у глобальні координати оригінального кадру $8K$:
  $$x_{global} = X_0 + x_{local} \cdot \frac{W_t}{M_{selected}}, \quad y_{global} = Y_0 + y_{local} \cdot \frac{H_t}{M_{selected}}$$
# Швидкий старт для розробників
## 1. Передумови (Host OS)
 - Docker Engine / Docker Desktop
 - NVIDIA Container Toolkit (для прокидання GPU у Docker)
 - VS Code + розширення Dev Containers (ms-vscode-remote.remote-containers)
## 2. Розгортання ізольованого середовища розробки
1. Клонуйте репозиторій та відкрийте папку проєкту у VS Code.
2. У правому нижньому кутку натисніть "Reopen in Container" (або викликайте через Ctrl+Shift+P -> Dev Containers: Reopen in Container).
3. VS Code автоматично розгорне контейнер із підготовленим середовищем (CUDA 12.8+, C++20, TensorRT, PySide6).
## 3. Ручна збірка та запуск тестів (усередині Dev Container)
В терміналі контейнера виконайте:
```bash
# 1. Очистити старі файли збірки
rm -rf build

# 2. Згенерувати build-файли з використанням Shared CUDA Runtime
cmake -B build -DCMAKE_CUDA_RUNTIME_LIBRARY=Shared

# 3. Збираємо проєкт
cmake --build build

# 4. Запустити модульні тести CTest
ctest --test-dir build --output-on-failure
# 2. Згенерувати build-файли з використанням Shared CUDA Runtime
cmake -B build -DCMAKE_CUDA_RUNTIME_LIBRARY=Shared

# 3. Збираємо проєкт
cmake --build build

# 4. Запустити модульні тести CTest
ctest --test-dir build --output-on-failure
```