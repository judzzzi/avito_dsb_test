"""
Шаг 3. Кандидаты и признаки для валидационных запросов.

Контекст строится по fit-части train и валидационному корпусу (см. split.py),
то есть валидационные объявления не «видны» в логах — как и в бенчмарке.
Результат: work/val_cands{sfx}.parquet (признаки + метка релевантности).
"""
import sys
import time

import numpy as np
import pandas as pd

from candidates import add_query_relative_features, generate
from context import Context, load_inputs

WORK = "work"


def recall_at(df, nrel: pd.Series, score_col, k=50):
    """Recall@k: для каждого запроса (индекс nrel) берём top-k кандидатов по score_col.
    nrel — число релевантных объявлений запроса; запросы без кандидатов дают 0."""
    top = df.sort_values(["qi", score_col], ascending=[True, False]).groupby("qi").head(k)
    hit = top.groupby("qi").label.sum().reindex(nrel.index, fill_value=0)
    return (hit / nrel).mean()


def main(split_id: int = 0):
    t0 = time.time()
    sfx = f"_{split_id}" if split_id else ""
    corpus_ids = pd.read_parquet(f"{WORK}/val_corpus{sfx}.parquet").item_id
    corpus, light = load_inputs(f"{WORK}/items_all.parquet", corpus_ids)
    fit = pd.read_parquet(f"{WORK}/val_fit_pairs{sfx}.parquet")
    val = pd.read_parquet(f"{WORK}/val_queries{sfx}.parquet")

    ctx = Context(corpus, light, fit)
    del corpus, light
    # эмбеддинги two-tower модели, обученной на fit-части (python embed.py val)
    ctx.set_embeddings(np.load(f"{WORK}/emb_items_val.npy"),
                       pd.read_parquet(f"{WORK}/emb_items_val_ids.parquet").item_id.values)
    q_emb = np.load(f"{WORK}/emb_queries_val.npy")
    print("context: %.0fs" % (time.time() - t0), flush=True)

    df = generate(ctx, val, q_emb)
    df = add_query_relative_features(df)
    # метки
    id2c = pd.Series(np.arange(ctx.N), index=ctx.item_ids)
    pos = set()
    for qi, rel in enumerate(val.relevant):
        for it in rel:
            pos.add((qi, int(id2c[it])))
    df["label"] = [int((a, b) in pos) for a, b in zip(df.qi.values, df.ci.values)]
    df.to_parquet(f"{WORK}/val_cands{sfx}.parquet", index=False)

    nrel = val.relevant.map(len).values
    hit = df.groupby("qi").label.sum().reindex(range(len(val)), fill_value=0).values
    print("candidates per query: %.0f" % df.groupby("qi").size().mean())
    print("candidate recall (ceiling): %.4f" % np.mean(hit / nrel))
    for name in ["all", "title", "prof", "tr", "hist", "mc", "emb"]:
        h = df[df["src_" + name] == 1].groupby("qi").label.sum().reindex(range(len(val)), fill_value=0).values
        print(f"  source {name}: recall {np.mean(h / nrel):.4f}")
    nrel_s = val.relevant.map(len)
    print("baseline recall@50 (bm_all + 2*loc): %.4f" % recall_at(df, nrel_s, "base"))
    nrel_eval = nrel_s[val.part == "eval"]
    print("  on eval part: %.4f" % recall_at(df[df.qi.isin(nrel_eval.index)], nrel_eval, "base"))
    print("total: %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
