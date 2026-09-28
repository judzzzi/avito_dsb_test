#!/usr/bin/env bash
# Полный воспроизводимый прогон: от исходных parquet в ./data до answer.csv.
set -euo pipefail
cd "$(dirname "$0")"

python prep.py                    # 1. стемминг текстов, единая таблица объявлений
python split.py                   # 2. локальная валидация по образцу бенчмарка
python embed.py val               # 2b. two-tower эмбеддинги (обучение на fit-части валидации)
python build_val_cands.py         # 3. кандидаты + признаки для валидации
python train_ranker.py            # 4. обучение ранкера и оценка Recall@50 на eval
python train_ranker.py --final    # 5. финальный ранкер на всех валидационных запросах
python embed.py full              # 5b. two-tower эмбеддинги (обучение на всём train)
python predict.py                 # 6. кандидаты для бенчмарка -> answer.csv
