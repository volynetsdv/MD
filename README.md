# Про проєкт

Головною метою проєкту є створення високопродуктивного, відмовостійкого та апаратно-оптимізованого програмного рішення для автономного виявлення малорозмірних об'єктів на аерофотознімках (VisDrone, DOTA) в умовах суворих обмежень обчислювальних ресурсів бортових та edge-пристроїв.

# Ключові архітектурні принципи

1. Hardware-Aware Dynamic Tiling: Відмова від статичної сітки (SAHI) на користь розрахунку оптимального розміру плитки (`T_calc`) та перекриття (`O_lap`) залежно від телеметрії БПЛА (висоти польоту $H$), роздільної здатності кадру (4K/8K) та поточного обсягу вільної відеопам'яті ($\text{VRAM}_{\text{avail}}$).
2. In-Memory Zero-Copy Slicing: Повна відмова від створення тимчасових файлів (.jpg/.png) на диску. Фрагментація здійснюється виключно у VRAM за допомогою GPU-вказівників та CUDA-зрізів (sub-tensors).
3. Metadata-Driven UI (Non-Destructive Overlay): Обчислювальне ядро повертає на Host лише легкий вектор координат у форматі JSON. Нативний GUI (PySide6 QGraphicsView) рендерить обмежувальні рамки векторним шаром поверх чистого оригінального 8K кадру.
4. Принцип DRY через libtiling_core: Єдина низькорівнева C++/CUDA бібліотека забезпечує ідентичність алгоритмів нарізки як у високошвидкісному C++ Inference Engine, так і в Python Training Pipeline (через pybind11).
5. Dual Inference Engine Support: Пряме виконання через TensorRT C++ API на Linux/Jetson з автоматичним фолбеком на DirectML (ONNX Runtime) під Windows 10/11.

# Стек технологій

- Мови розробки: C++20 (обчислювальне ядро, математика, пам'ять), Python 3.11 (GUI, конвеєр навчання, пакування).
- C++ Core Engine: C++20 / CUDA C++ (tiling_core: `tiling_math`, `cuda_slicer`, `trt_detector`).
- Inference Runtime: NVIDIA TensorRT C++ API (FP16/INT8) / DirectML (ONNX Runtime - заплановано).
- Python Binding: pybind11 (`pytiling_core` - заплановано).
- GUI Framework: PySide6 (Qt6 for Python) + QGraphicsView (Hardware-Accelerated Canvas Viewer - заплановано).
- Training Pipeline: Python / PyTorch / Ultralytics YOLO26 (NMS-Free & STAL) / YOLO11s (заплановано).
- Post-Processing: C++ Cluster-DIoU-NMS + Host-Device Offset Mapping (заплановано).
- Build System & Environment: CMake 3.20+, GCC 11+, CUDA Toolkit 12.x+ / 12.8+, Docker (Dev Containers).
- Distribution: Autonomous Standalone Bundles (Nuitka / PyInstaller - .exe для Windows, .AppImage для Linux).

# Структура проєкту та статус реалізації

```text
├── .devcontainer/              # Налаштування відтворюваного середовища Dev Container [реалізовано]
│   ├── devcontainer.json       # Конфігурація розширення VS Code
│   ├── Dockerfile              # Docker-образ (Ubuntu 22.04, CUDA 12.x, GCC-11, TensorRT)
│   └── requirements.txt        # Залежності Python
├── include/                    # C++ Header файли
│   ├── tiling_math.hpp         # Динамічний розрахунок сітки T_calc та O_lap [реалізовано]
│   ├── cuda_slicer.cuh         # CUDA-ядра для Zero-Copy Slicing у VRAM [реалізовано]
│   ├── trt_detector.hpp        # C++ Wrapper над TensorRT API (enqueueV3) [реалізовано]
│   └── postprocess.hpp         # Offset Mapping та Cluster-DIoU-NMS [заплановано]
├── src/                        # C++ та CUDA реалізація модулів ядра
│   ├── tiling_math.cpp         # Реалізація розрахунку параметрів сітки [реалізовано]
│   ├── cuda_slicer.cu          # CUDA-ядро білінійної екстракції плиток у VRAM [реалізовано]
│   ├── trt_detector.cpp        # Реалізація детектора на базі TensorRT [реалізовано]
│   └── postprocess.cpp         # Логіка NMS та проектування координат [заплановано]
├── bindings/                   # Python bindings через pybind11 [заплановано]
│   └── pybind_wrapper.cpp      # Модуль pytiling_core
├── gui/                        # PySide6 Графічний інтерфейс користувача [заплановано]
│   ├── main_window.py          # Головна форма додатка
│   ├── canvas_viewer.py        # QGraphicsView для апаратно-прискореного 8K зумінгу
│   └── async_worker.py         # QThread асинхронний конвеєр обробки
├── tests/                      # Модульні тести CTest [реалізовано]
│   ├── test_tiling_math.cpp    # Тестування математики сітки та граничних умов
│   ├── test_cuda_slicer.cpp    # Тестування екстракції плиток у пам'яті GPU
│   └── test_trt_detector.cpp   # Тестування завантаження .engine та інференсу
├── .aider.conventions.md       # Конвенції розробки для AI-агента [реалізовано]
├── .aider.conf.yml             # Конфігурація aider [реалізовано]
└── CMakeLists.txt              # Скрипт збірки CMake (C++20, CUDA, TensorRT) [реалізовано]
```

# Математична модель та геометрія

## 1. Динамічний розрахунок сітки (T_calc та O_lap)

Планувальник обчислює розрахунковий розмір плитки `T_calc` на основі поточної телеметрії висоти $H$ (у метрах) та доступної відеопам'яті $\text{VRAM}_{\text{avail}}$ (у мегабайтах):

$$
T_{\text{calc}} = \frac{\text{VRAM}_{\text{avail}}}{4} + 10 \cdot H
$$

На основі отриманого значення здійснюється дискретний вибір цільового розміру $M_{\text{selected}}$ із пулу конфігурацій експортованих моделей $M \in \{320, 416, 512, 640\}$ як максимальний розмір, що не перевищує розрахунковий:

$$
M_{\text{selected}} = \max \{ m \in M \mid m \le \lfloor T_{\text{calc}} \rfloor \}
$$

Якщо $T_{\text{calc}} < 320$, як базовий розмір обирається мінімальна плитка 320.

Коефіцієнт перекриття плиток `O_lap` динамічно зростає зі збільшенням висоти польоту та обмежується діапазоном $[0.1,\, 0.4]$:

$$
O_{\text{lap}} = \text{clamp}\left(0.1 + \frac{H}{500},\, 0.1,\, 0.4\right)
$$

Крок сітки між сусідніми плитками обчислюється як:

$$
\text{step} = \text{round}\left(M_{\text{selected}} \cdot (1 - O_{\text{lap}})\right)
$$

Кількість колонок ($N_{\text{cols}}$) та рядків ($N_{\text{rows}}$) сітки для вхідного кадру розміром $W_{\text{img}} \times H_{\text{img}}$ визначається через округлення вгору:

$$
N_{\text{cols}} = \left\lceil \frac{W_{\text{img}}}{\text{step}} \right\rceil, \quad N_{\text{rows}} = \left\lceil \frac{H_{\text{img}}}{\text{step}} \right\rceil
$$

## 2. Host-Device Offset Mapping

Для кожної плитки $Tile_i(X_0, Y_0, W_t, H_t)$ (де $W_t \le M_{\text{selected}}$, $H_t \le M_{\text{selected}}$ з урахуванням обрізки на межах зображення) вхідний фрагмент масштабується до розміру моделі $M_{\text{selected}} \times M_{\text{selected}}$.

Локальні координати знайденого об'єкта $(x_{\text{local}}, y_{\text{local}}, w_{\text{local}}, h_{\text{local}})$ перераховуються в глобальні координати оригінального кадру за формулами:

$$
x_{\text{global}} = X_0 + x_{\text{local}} \cdot \frac{W_t}{M_{\text{selected}}}
$$

$$
y_{\text{global}} = Y_0 + y_{\text{local}} \cdot \frac{H_t}{M_{\text{selected}}}
$$

$$
w_{\text{global}} = w_{\text{local}} \cdot \frac{W_t}{M_{\text{selected}}}
$$

$$
h_{\text{global}} = h_{\text{local}} \cdot \frac{H_t}{M_{\text{selected}}}
$$

# Швидкий старт для розробників

## 1. Передумови (Host OS)

- Docker Engine / Docker Desktop
- NVIDIA Container Toolkit (для прокидання GPU у Docker)
- VS Code + розширення Dev Containers (ms-vscode-remote.remote-containers)

## 2. Розгортання ізольованого середовища розробки

1. Клонуйте репозиторій та відкрийте папку проєкту у VS Code.
2. У правому нижньому кутку натисніть "Reopen in Container" (або викличте через Ctrl+Shift+P -> Dev Containers: Reopen in Container).
3. VS Code автоматично розгорне контейнер із підготовленим середовищем (CUDA 12.x, C++20, TensorRT, PySide6).

## 3. Ручна збірка та запуск тестів (усередині Dev Container)
В терміналі контейнера виконайте:
```bash
# 1. Очистити старі файли збірки (за потреби)
rm -rf build

# 2. Згенерувати build-файли з використанням Shared CUDA Runtime
cmake -B build -DCMAKE_CUDA_RUNTIME_LIBRARY=Shared

# 3. Зібрати проєкт
cmake --build build

# 4. Запустити всі модульні тести CTest
ctest --test-dir build --output-on-failure
```

### Модульні тести

- `test_tiling_math` - перевірка розрахунку розміру плиток, перекриття, кроку та меж для кадру 8K.
- `test_cuda_slicer` - перевірка виділення пам'яті VRAM, білінійної екстракції плитки в тензор `[1, 3, target, target]` та діапазону значень $[0.0, 1.0]$.
- `test_trt_detector` - перевірка десеріалізації моделі TensorRT, виконання інференсу та контролю пам'яті VRAM. Може приймати шлях до моделі: `./build/test_trt_detector <path/to/model.engine>` (за відсутності аргументу тест коректно пропускається зі статусом SKIP).