"""
Шаг 1. Подготовка данных.

* Собираем единую таблицу объявлений `items_all` = корпус бенчмарка
  + уникальные объявления из train (признаки объявления одинаковые, берём первое вхождение).
* Считаем стеммированные версии текстовых полей (заголовок, параметры, описание),
  чтобы все дальнейшие шаги (BM25, статистики, признаки) работали с готовыми токенами.
* Train сохраняем без тяжёлых колонок объявления (они есть в items_all).

Работаем потоково (батчами), т.к. train с описаниями после конвертации в pandas
не помещается в 7 ГБ памяти.

Запуск: python prep.py   (читает data/*.parquet, пишет work/*.parquet)
"""
import os
import re
from multiprocessing import Pool

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from common import stem_text

DATA = "data"
WORK = "work"
N_PROC = 2
os.makedirs(WORK, exist_ok=True)

ITEM_COLS = [
    "item_id", "item_title_raw", "item_description_raw", "item_infm_params_text",
    "item_category_id", "item_microcat_id", "item_price", "item_rating",
    "item_rating_reviews_count", "item_location_id", "item_latitude", "item_longitude",
    "item_is_phone_hidden", "item_is_message_forbidden",
]
SEARCH_COLS = [
    "search_query", "search_location_id", "search_is_delivery_search",
    "search_infm_params_text", "search_category",
]

# В параметрах объявления есть адрес («Место оказания услуг ...») и график работы —
# для текстового поиска это шум. Вырезаем адрес и куски графика.
_ADDR_RE = re.compile(
    r"Место оказания услуг .*?(?= Тип стоимости| Тип услуги| Работаете| График| Время| Опыт| Гарантия| Чем вы| Дни |$)"
)
_SCHED_RE = re.compile(r"(График работы|Время работы|Время для связи)[^А-ЯЁ]*?(?=[А-ЯЁ]|$)")


def clean_params(s: str) -> str:
    if not isinstance(s, str):
        return ""
    return _SCHED_RE.sub(" ", _ADDR_RE.sub(" ", s))


def _stem_chunk(args):
    titles, params, descs = args
    return (
        [stem_text(x) for x in titles],
        [stem_text(clean_params(x)) for x in params],
        # Описание обрезаем: суть услуги почти всегда в начале текста,
        # а хвост (контакты, «звоните») только раздувает индекс и память.
        [stem_text(x, max_chars=1000) for x in descs],
    )


def process_batch(d: pd.DataFrame, pool: Pool) -> pd.DataFrame:
    """Стемминг текстов + приведение числовых полей. Возвращает компактный датафрейм."""
    n = len(d)
    step = (n + N_PROC - 1) // N_PROC
    jobs = []
    for i in range(0, n, step):
        sl = d.iloc[i : i + step]
        jobs.append((sl.item_title_raw.tolist(), sl.item_infm_params_text.tolist(),
                     sl.item_description_raw.str.slice(0, 1000).tolist()))
    res = pool.map(_stem_chunk, jobs)
    out = d.drop(columns=["item_description_raw"]).copy()
    out["t_title"] = [x for r in res for x in r[0]]
    out["t_params"] = [x for r in res for x in r[1]]
    out["t_desc"] = [x for r in res for x in r[2]]
    for c in ["item_price", "item_latitude", "item_longitude"]:
        out[c] = out[c].astype(float)
    return out


def main():
    pool = Pool(N_PROC)
    writer = None
    seen = set()

    def write(df):
        nonlocal writer
        tbl = pa.Table.from_pandas(df, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(f"{WORK}/items_all.parquet", tbl.schema)
        writer.write_table(tbl.cast(writer.schema))

    # 1) корпус бенчмарка
    pf = pq.ParquetFile(f"{DATA}/benchmark_items.parquet")
    for b in pf.iter_batches(batch_size=40000, columns=ITEM_COLS):
        d = b.to_pandas()
        d["in_bench"] = True
        seen.update(d.item_id)
        write(process_batch(d, pool))
    print("bench items done:", len(seen), flush=True)

    # 2) объявления из train, которых нет в корпусе
    pf = pq.ParquetFile(f"{DATA}/train.parquet")
    for b in pf.iter_batches(batch_size=40000, columns=ITEM_COLS):
        d = b.to_pandas()
        d = d[~d.item_id.isin(seen)].drop_duplicates("item_id")
        if len(d) == 0:
            continue
        d["in_bench"] = False
        seen.update(d.item_id)
        write(process_batch(d, pool))
    writer.close()
    print("items_all:", len(seen), flush=True)

    # 3) пары train: только признаки запроса + item_id
    tr = pd.read_parquet(f"{DATA}/train.parquet", columns=SEARCH_COLS + ["item_id"])
    tr["q_stem"] = [stem_text(x) for x in tr.search_query]
    tr.to_parquet(f"{WORK}/train_pairs.parquet", index=False)

    bq = pd.read_parquet(f"{DATA}/benchmark_queries.parquet")
    bq["q_stem"] = [stem_text(x) for x in bq.search_query]
    bq.to_parquet(f"{WORK}/bench_queries.parquet", index=False)
    pool.close()


if __name__ == "__main__":
    main()
