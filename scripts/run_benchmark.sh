#!/usr/bin/env bash
set -euo pipefail

# 1. Автоматичне визначення кореневої директорії проєкту (на один рівень вище папки scripts)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$WORKSPACE"

# 2. Шляхи до даних та звітів
INPUT_DIR="data/dota_v1.5/images/val"
DETECTED_DIR="data/dota_v1.5/images/val_detected"
REPORTS_DIR="reports/benchmark"

echo "=========================================================================="
echo " ЗАПУСК НАУКОВОГО БЕНЧМАРКУ ТА ПАКЕТНОГО ТРІАЖУ DOTA v1.5"
echo "=========================================================================="
echo " Корінь проєкту:     $WORKSPACE"
echo " Вхідна папка:       $INPUT_DIR"
echo " Папка результатів:  $DETECTED_DIR"
echo " Звіти дисертації:   $REPORTS_DIR"
echo "=========================================================================="

# 3. Перевірка наявності вхідних зображень
if [ ! -d "$INPUT_DIR" ]; then
    echo "[ПОМИЛКА] Вхідна директорія $INPUT_DIR не знайдена у $WORKSPACE!"
    exit 1
fi

mkdir -p "$DETECTED_DIR"
mkdir -p "$REPORTS_DIR"

# 4. Перевірка доступності GPU
if command -v nvidia-smi &> /dev/null; then
    echo "[INFO] Апаратний прискорювач виявлено:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
fi

# 5. Запуск модуля бенчмаркінгу
echo ""
echo "[СТАРТ] Обробка зображень та неперервний моніторинг ресурсів..."

QT_QPA_PLATFORM=offscreen python3 scripts/benchmark_system.py \
    --input "$INPUT_DIR" \
    --gt-labels-dir "data/dota_v1.5/labels/val" \
    --eval-accuracy \
    --output-detected-dir "$DETECTED_DIR" \
    --output-dir "$REPORTS_DIR" \
    --monitor-interval 0.1 \
    --altitude 150.0 \
    --vram-mb 2048 \
    --conf-thresh 0.20 \
    --diou-thresh 0.50

echo ""
echo "=========================================================================="
echo " ЕКСПЕРИМЕНТ УСПІШНО ЗАВЕРШЕНО"
echo "=========================================================================="
echo "1. Збережені зображення та JSON-кеші: $DETECTED_DIR"
echo "2. Науковий звіт для 3-го розділу:    $REPORTS_DIR/benchmark_summary.md"
echo "3. Сирі заміри та часові ряди:        $REPORTS_DIR/benchmark_results.json"
echo "=========================================================================="