# Інструкція із запуску модуля системного бенчмаркінгу

Модуль [`scripts/benchmark_system.py`](file:///workspaces/MD/scripts/benchmark_system.py) призначений для автоматизованого збору емпіричних даних швидкодії, декомпозиції латентності, неперервної апаратної телеметрії (RAM, CPU, GPU Compute Core, VRAM), аудиту відсутності витоків пам'яті, порівняння режимів Standard (B=1) та Turbo (Dynamic B), а також оцінки точності конвеєра.

---

## 1. Передумови та підготовка середовища

Перед запуском бенчмарку переконайся, що проєкт зібрано та встановлено необхідні бібліотеки:

```bash
# 1. Перехід у робочу директорію проєкту
cd /workspaces/MD

# 2. Надання прав на виконання скрипту
chmod +x scripts/benchmark_system.py

# 3. Перевірка доступності прискорювача NVIDIA GPU та версії драйвера
nvidia-smi
```

> [!NOTE]
> Скрипт автоматично виявляє:
> - Наявність скомпільованого C++20 модуля `pytiling_core` (шукає в `build/` або `build/bindings/`). У разі його відсутності прозоро активується резервний Python/NumPy бекенд.
> - Доступні провайдери ONNX Runtime (`CUDAExecutionProvider`, `TensorrtExecutionProvider`, `CPUExecutionProvider`).
> - Динамічні класи детекції з моделі з безпечним fallback на 16 класів DOTA 1.5.

---

## 2. Основні режими роботи модуля

Модуль підтримує гнучкі режими конфігурації:

### Режим 1: Подвійний компаративний запуск (--mode both)
**Основний рекомендований режим** для порівняльного аналізу Standard Mode ($B=1$, Safe Edge) та Turbo Mode (Dynamic Max Batch):
- Виконує Pass 1 у Standard Mode ($B=1$), фіксуючи базову швидкодію та телеметрію.
- Виконує скидання пулів пам'яті CUDA та паузу охолодження.
- Виконує Pass 2 у Turbo Mode із динамічним розрахунком безпечного розміру батчу $B \in [1, B_{\text{cap}}]$.
- Формує порівняльну таблицю прискорення FPS, скорочення $T_{\text{infer}}$, споживання VRAM та збереження точності.

```bash
python3 scripts/benchmark_system.py \
    --input data/dota_v1.5/images/val \
    --mode both \
    --turbo-max-cap 4 \
    --output-detected-dir data/dota_v1.5/images/val_detected \
    --output-dir reports/benchmark \
    --altitude 150.0 \
    --vram-mb 2048
```

### Режим 2: Наскрізний пакетний тріаж директорії (Batch Triage & Continuous Telemetry)
Для оцінки конкретного режиму швидкодії (`--mode standard` або `--mode turbo`):
- Проходить усі зображення в заданій папці (`--input <директорія>`).
- Для кожного кадру заміряє точну тривалість стадій: $T_{\text{read}}$, $T_{\text{tiling}}$, $T_{\text{slice}}$, $T_{\text{infer}}$, $T_{\text{remap}}$, $T_{\text{nms}}$, $T_{\text{io\_write}}$, $T_{\text{total}}$.
- **Дискова реплікація (Batch Triage):** зображення з виявленими цілями ($\ge 1$) копіюються прямо з диска на диск (`shutil.copy2`) у директорію `--output-detected-dir` без зворотного завантаження растру через шину PCIe.
- Поряд із кожним збереженим кадром записується легковаговий JSON-кеш координат детекцій (`<stem>.json`).
- Фоновий телеметричний демон фіксує використання Host RAM, CPU %, GPU Compute Util % та VRAM Allocated/Reserved кожні 100 мс.

### Режим 3: Мультимасштабний синтетичний стрес-тест (Synthetic Presets: 4K, 8K, 13K)
Використовується за відсутності вибірки реальних зображень:
- Автоматично активується, якщо параметр `--input` не вказано.
- Генерує синтетичні панорами надвисокої роздільної здатності: **4K** ($3840 \times 2160$), **8K** ($7680 \times 4320$), **13K** ($13000 \times 6500$).
- Виконує серію повторних прогонів (`--samples 10`) для мінімізації статистичної дисперсії.
- Порівнює кількість плиток сітки 640 px проти адаптивної 736 px.
- Виконує тривалий стрес-тест на витоки пам'яті (`--leak-check-iterations 50`).

```bash
python3 scripts/benchmark_system.py \
    --mode both \
    --samples 10 \
    --leak-check-iterations 50 \
    --output-dir reports/benchmark
```

### Режим 4: Профілювання одиничного кадру (Single-File Deep Profiling)
Використовується для детального вивчення конкретного знімка:
- Передається шлях до одного файлу (`--input <шлях_до_файлу.jpg>`).
- Виконується вказана кількість повторів (`--samples N`) для розрахунку медіани ($p50$), $p90$, $p99$ та розкиду $\text{Mean} \pm \text{Std}$.

```bash
python3 scripts/benchmark_system.py \
    --input data/dota_v1.5/images/val/P0003.jpg \
    --samples 15 \
    --mode both \
    --output-dir reports/benchmark_single
```

---

## 3. Сценарії використання та приклади команд

### Сценарій А: Комплексний експеримент на датасеті DOTA з оцінкою точності
Обробка валідаційних кадрів DOTA v1.5 у режимах Standard та Turbo із розрахунком mAP50 та неперервним моніторингом:
```bash
QT_QPA_PLATFORM=offscreen python3 scripts/benchmark_system.py \
    --input data/dota_v1.5/images/val \
    --mode both \
    --turbo-max-cap 4 \
    --gt-labels-dir data/dota_v1.5/labels/val \
    --eval-accuracy \
    --output-detected-dir data/dota_v1.5/images/val_detected \
    --output-dir reports/benchmark \
    --monitor-interval 0.1 \
    --altitude 150.0 \
    --vram-mb 2048
```

### Сценарій Б: Швидкий експрес-тест пакетного режиму (перші 10 кадрів)
Для швидкої перевірки працездатності конвеєра:
```bash
python3 scripts/benchmark_system.py \
    --input data/dota_v1.5/images/val \
    --mode both \
    --max-images 10 \
    --output-dir reports/benchmark_fast
```

### Сценарій В: Швидкий синтетичний тест (без перевірки витоків пам'яті)
Для швидкої перевірки генератора звітів:
```bash
python3 scripts/benchmark_system.py \
    --mode standard \
    --samples 2 \
    --leak-check-iterations 0 \
    --output-dir reports/benchmark_quick
```

---

## 4. Повний перелік параметрів командного рядка

| Прапорець / Опція | Тип | За замовчуванням | Опис |
| :--- | :---: | :---: | :--- |
| `--input`, `--image-dir` | `str` | `None` | Шлях до зображення або папки з кадрами (`data/dota_v1.5/images/val`). Якщо не вказано — активується синтетичний режим |
| `--mode` | `str` | `both` | Режим продуктивності: `standard` ($B=1$), `turbo` (Dynamic Max Batch), або `both` (Double-Pass запуск) |
| `--turbo-max-cap` | `int` | `4` | Максимальний розмір батчу для Turbo режиму |
| `--output-detected-dir` | `str` | `<input>_detected` | Каталог для збереження позитивних кадрів ($\ge 1$ ціль) та JSON-кешів координат (Batch Triage) |
| `--output-dir` | `str` | `reports/benchmark` | Каталог збереження фінальних звітів (`benchmark_results.json` та `benchmark_summary.md`) |
| `--monitor-interval` | `float` | `0.1` | Інтервал опитування фонового телеметричного демона (в секундах, за замовчуванням 100 мс) |
| `--max-images` | `int` | `None` | Обмеження максимальної кількості оброблюваних зображень у пакетному режимі |
| `--samples` | `int` | `10` | Кількість повторних ітерацій для кожного кадру (для синтетичного/одиничного режиму) |
| `--altitude` | `float` | `150.0` | Висота польоту БПЛА (в метрах) для розрахунку перекриття $O_{\text{lap}}$ |
| `--vram-mb` | `int` | `2048` | Бюджет доступної відеопам'яті (MB) для вибору моделі у формулі $T_{\text{calc}}$ |
| `--conf-thresh` | `float` | `0.20` | Поріг впевненості детектора для відсікання фонових об'єктів |
| `--diou-thresh` | `float` | `0.50` | Поріг DIoU для кластеризації меж (Cluster-DIoU-NMS) |
| `--leak-check-iterations`| `int` | `50` | Кількість циклів стрес-тесту пам'яті в синтетичному режимі (`0` — вимкнути) |
| `--warmup` | `int` | `2` | Кількість попередніх холостих прогонів для стабілізації кешу CUDA та ONNX Runtime |
| `--resolutions` | `str` | `4K,8K,13K` | Список роздільних здатностей для синтетичного тесту |
| `--eval-accuracy` | `flag` | `False` | Вмикає зіставлення детекцій із Ground Truth та розрахунок метрик точності (P, R, mAP@50, mAP@50-95) |
| `--gt-labels-dir` | `str` | `None` | Шлях до папки з еталонними текстовими анотаціями (за замовчуванням автоматично шукає `labels/val`) |
| `--iou-eval-threshold` | `float` | `0.50` | Базовий поріг IoU для зіставлення детекцій із Ground Truth (та сітка 0.50..0.95 для mAP50-95) |

Для перегляду вбудованої довідки:
```bash
python3 scripts/benchmark_system.py --help
```

---

## 5. Вихідні артефакти та інтерпретація результатів

Після виконання бенчмарку формуються наступні матеріали:

### 1. Звіт `reports/benchmark/benchmark_summary.md`
Містить формалізовані таблиці у форматі Markdown:
- **Порівняння режимів (Standard vs Turbo):** швидкість (FPS), медіанна затримка ($p50$), час $T_{\text{infer}}$, споживання VRAM, статус витоків та метрики збереження точності (mAP@50, Recall).
- **Таблиця 1: Апаратне середовище:** тип процесора, доступна RAM, модель GPU, VRAM, версія CUDA, провайдери інференсу.
- **Таблиця 2: Апаратна телеметрія ресурсів:** порівняння Baseline, Median, Mean, Peak, Delta $\Delta$ для Host RSS RAM та GPU VRAM (Allocated / Reserved) зі статусом витоків (`PASS [OK]` за умови $\Delta\text{RAM} < 50\text{ MB}$).
- **Таблиця 3: Утилізація ядер:** процентне завантаження GPU Compute Core % та System CPU % (Median, Mean, Peak, $p90, p99$).
- **Таблиця 4: Показники пакетного тріажу:** кількість позитивних/негативних кадрів, загальне число виявлених об'єктів, фізичний час сесії та наскрізна швидкість конвеєра (**FPS**).
- **Таблиця 5: Декомпозиція латентності за стадіями:** таймінги $T_{\text{read}}$, $T_{\text{tiling}}$, $T_{\text{slice}}$, $T_{\text{infer}}$, $T_{\text{remap}}$, $T_{\text{nms}}$, $T_{\text{io\_write}}$, $T_{\text{total}}$ (Mean $\pm$ Std, Median, $p90, p99$).
- **Розділ точності (якщо передано `--eval-accuracy`):** наскрізні показники точності конвеєра (Precision, Recall, mAP@50, mAP@50-95) та покласова таблиця AP50/AP50-95.
- **Таблиця PCIe:** підтвердження інваріанту **Zero PCIe Readback** (заощадження пропускної здатності шини на рівні $> 99.96\%$).
- **Підсумковий аналіз ефективності системи:** узагальнені висновки щодо прискорення, збереження точності та OOM-Safety.

### 2. Структуровані дані `reports/benchmark/benchmark_results.json`
Повний набір сирих замірів, параметрів конфігурації, квантилів та часових рядів телеметрії для обох режимів.

### 3. Каталог позитивних кадрів `data/dota_v1.5/images/val_detected/` (Batch Triage)
Створюється під час обробки директорії та містить:
- Оригінальні збережені растри зображень (скопійовані без модифікації через `shutil.copy2`);
- Парні файли `<stem>.json` із координатами, класами та впевненістю виявлених цілей.

---

## 6. Верифікація та модульні тести

Для перевірки коректності всіх механізмів бенчмаркінгу, математичних розрахунків, телеметричного демона та інваріантів системи:

```bash
# 1. Запуск спеціалізованих тестів бенчмарку та конвеєра
QT_QPA_PLATFORM=offscreen pytest -v tests/test_benchmark_system.py tests/test_pipeline.py

# 2. Повна перевірка всієї тестової бази проєкту
QT_QPA_PLATFORM=offscreen pytest -v tests/

# 3. Наскрізний системний аудит відповідності архітектурним вимогам
python3 validate_system.py
```