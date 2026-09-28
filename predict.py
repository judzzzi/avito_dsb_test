"""
Шаг 5. Предсказание для бенчмарка -> answer.csv.

Контекст строится по ВСЕМУ train (логи) и корпусу бенчмарка; ранкер — из
train_ranker.py --final. Для каждого запроса берём top-50 кандидатов по скору ранкера.
"""
import lightgbm as lgb
import numpy as np
import pandas as pd

from candidates import add_query_relative_features, generate
from context import Context, load_inputs

WORK = "work"
DATA = "data"


def main():
    bench_ids = pd.read_parquet(f"{DATA}/benchmark_items.parquet", columns=["item_id"]).item_id
    corpus, light = load_inputs(f"{WORK}/items_all.parquet", bench_ids)
    fit = pd.read_parquet(f"{WORK}/train_pairs.parquet")
    q = pd.read_parquet(f"{WORK}/bench_queries.parquet").reset_index(drop=True)

    ctx = Context(corpus, light, fit)
    del corpus, light

    df = generate(ctx, q)
    df = add_query_relative_features(df)
    model = lgb.Booster(model_file=f"{WORK}/ranker.txt")
    df["score"] = model.predict(df[model.feature_name()])

    top = df.sort_values(["qi", "score"], ascending=[True, False]).groupby("qi").head(50)
    top["item_id"] = ctx.item_ids[top.ci.values]
    ans = top.groupby("qi").item_id.agg(" ".join).reindex(range(len(q)), fill_value="")
    out = pd.DataFrame({"query_id": q.query_id.values, "answer": ans.values})
    out.to_csv("answer.csv", index=False)

    # --- проверка формата ---
    chk = pd.read_csv("answer.csv", dtype=str, keep_default_na=False)
    valid = set(bench_ids)
    assert list(chk.columns) == ["query_id", "answer"]
    assert len(chk) == len(q) and chk.query_id.is_unique
    assert set(chk.query_id) == set(q.query_id)
    for a in chk.answer:
        ids = a.split()
        assert len(ids) <= 50 and len(ids) == len(set(ids)) and all(i in valid for i in ids)
    print("answer.csv OK:", len(chk), "rows; mean items per row %.1f"
          % chk.answer.str.split().map(len).mean())


if __name__ == "__main__":
    main()
