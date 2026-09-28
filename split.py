"""
Шаг 2. Локальная валидация, имитирующая бенчмарк.

Устройство бенчмарка (по разведочному анализу):
  * 2452 запроса, все тексты уникальны; ~37% текстов встречаются в train;
  * корпус 189k объявлений, из них лишь ~10% встречаются в train;
  * у запроса обычно 1–2 релевантных объявления.

Воспроизводим это на train:
  1. Делим объявления train по хэшу на две группы: A (~55%, «новые») и B («старые»).
  2. Корпус валидации = 170k объявлений из A + ~19k из B (≈10% с историей), итого ≈189k.
  3. Событие поиска = (текст, локация, фильтры, категория, доставка); его релевантные
     объявления = выбранные в нём item_id. Берём события, у которых ВСЕ объявления
     лежат в корпусе, и сэмплируем по одному событию на уникальный текст.
  4. fit-выборка (на ней считаются статистики и обучаются модели) = строки train,
     чьи объявления НЕ из группы A и которые не принадлежат валидационным событиям.
     Так объявления-кандидаты из A не имеют истории — как 90% корпуса бенчмарка.

Валидационные события делим на две части: «rank» (обучение ранкера) и «eval»
(честная оценка Recall@50).
"""
import hashlib
import sys

import numpy as np
import pandas as pd

WORK = "work"
CORPUS_SIZE = 189_212
SHARE_B_IN_CORPUS = 0.10
N_VAL_TEXTS = 9000       # всего валидационных запросов
N_EVAL = 3000            # из них — на итоговую оценку, остальные для ранкера
EVENT_KEY = ["search_query", "search_location_id", "search_infm_params_text",
             "search_category", "search_is_delivery_search"]


def h01(s: str, salt: str = "") -> float:
    """Детерминированный хэш строки в [0, 1)."""
    return int(hashlib.md5((salt + s).encode()).hexdigest()[:8], 16) / 2**32


def main(split_id: int = 0):
    """split_id — номер разбиения: разные разбиения дают разные корпуса и запросы,
    что увеличивает обучающую выборку ранкера и делает оценку стабильнее."""
    SEED = 42 + split_id
    salt = str(split_id) if split_id else ""
    sfx = f"_{split_id}" if split_id else ""
    tr = pd.read_parquet(f"{WORK}/train_pairs.parquet")
    items = pd.Series(tr.item_id.unique())
    grp_a = items[items.map(lambda x: h01(x, salt)) < 0.55]
    grp_b = items[~items.isin(set(grp_a))]
    n_b = int(CORPUS_SIZE * SHARE_B_IN_CORPUS)
    corpus = pd.concat([
        grp_a.sample(n=min(len(grp_a), CORPUS_SIZE - n_b), random_state=SEED),
        grp_b.sample(n=n_b, random_state=SEED),
    ])
    corpus_set = set(corpus)
    set_a = set(grp_a)
    print("items:", len(items), "A:", len(grp_a), "corpus:", len(corpus_set))

    # события и их релевантные объявления
    tr["ev"] = tr.groupby(EVENT_KEY, sort=False).ngroup()
    ev_items = tr.groupby("ev").item_id.agg(lambda s: list(dict.fromkeys(s)))
    ev_ok = ev_items[ev_items.map(lambda lst: all(i in corpus_set for i in lst))]
    ev_first = tr.drop_duplicates("ev").set_index("ev")
    cand = ev_first.loc[ev_ok.index]
    # одно случайное событие на уникальный текст, затем случайные N_VAL_TEXTS текстов
    cand = cand.sample(frac=1.0, random_state=SEED)
    cand = cand[~cand.search_query.duplicated()]
    cand = cand.sample(n=N_VAL_TEXTS, random_state=SEED)

    val = cand[EVENT_KEY + ["q_stem"]].copy()
    val["relevant"] = ev_items.loc[val.index].values
    val["query_id"] = [f"val{i:06d}" for i in range(len(val))]
    val["part"] = np.where(np.arange(len(val)) < N_EVAL, "eval", "rank")
    val = val.reset_index(drop=True)

    val_ev = set(cand.index)
    fit = tr[~tr.item_id.isin(set_a) & ~tr.ev.isin(val_ev)].drop(columns=["ev"])

    # --- диагностика: насколько похоже на бенчмарк ---
    fit_texts = set(fit.search_query)
    fit_items = set(fit.item_id)
    print("val queries:", len(val), "fit rows:", len(fit))
    print("share of val texts seen in fit: %.3f (bench: 0.37)" % val.search_query.isin(fit_texts).mean())
    print("share of corpus items seen in fit: %.3f (bench: 0.096)" % np.mean([i in fit_items for i in corpus_set]))
    print("relevant per query:", val.relevant.map(len).describe().to_dict())

    val["split"] = split_id
    pd.DataFrame({"item_id": sorted(corpus_set)}).to_parquet(f"{WORK}/val_corpus{sfx}.parquet", index=False)
    val.to_parquet(f"{WORK}/val_queries{sfx}.parquet", index=False)
    fit.to_parquet(f"{WORK}/val_fit_pairs{sfx}.parquet", index=False)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
