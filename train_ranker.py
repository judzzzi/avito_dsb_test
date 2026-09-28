"""
Шаг 4. Обучение ранкера (LightGBM, lambdarank) на кандидатах валидации.

* Обучаем на запросах части «rank», качество меряем на «eval» (Recall@50).
* Для бенчмарка итоговую модель переобучаем на всех валидационных запросах
  с тем же числом деревьев (флаг --final).

Запуск:  python train_ranker.py          — обучение + оценка
         python train_ranker.py --final  — финальная модель на всех запросах
"""
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

WORK = "work"
NON_FEATURES = {"qi", "ci", "label"}
PARAMS = dict(
    objective="lambdarank",
    metric="map",
    eval_at=[50],
    learning_rate=0.05,
    num_leaves=63,
    min_data_in_leaf=50,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambdarank_truncation_level=60,
    num_threads=2,
    verbose=-1,
)
N_ROUNDS = 400


def recall50(df, score, nrel: pd.Series):
    d = df[["qi", "label"]].copy()
    d["s"] = score
    top = d.sort_values(["qi", "s"], ascending=[True, False]).groupby("qi").head(50)
    hit = top.groupby("qi").label.sum().reindex(nrel.index, fill_value=0)
    return (hit / nrel).mean()


def to_dataset(df, feats):
    # запросы без единого релевантного кандидата ничему не учат ранкер — убираем
    has_pos = df.groupby("qi").label.transform("max") > 0
    d = df[has_pos].sort_values("qi")
    groups = d.groupby("qi", sort=False).size().values
    return lgb.Dataset(d[feats], d.label, group=groups, free_raw_data=True)


def main():
    final = "--final" in sys.argv
    df = pd.read_parquet(f"{WORK}/val_cands.parquet")
    val = pd.read_parquet(f"{WORK}/val_queries.parquet")
    feats = [c for c in df.columns if c not in NON_FEATURES]
    nrel = val.relevant.map(len)

    if final:
        ds = to_dataset(df, feats)
        model = lgb.train(PARAMS, ds, num_boost_round=N_ROUNDS)
        model.save_model(f"{WORK}/ranker.txt")
        print("final model saved")
        return

    is_eval = val.part.values == "eval"
    q_eval = np.where(is_eval)[0]
    tr = df[~df.qi.isin(q_eval)]
    te = df[df.qi.isin(q_eval)]
    dtr, dte = to_dataset(tr, feats), to_dataset(te, feats)
    model = lgb.train(PARAMS, dtr, num_boost_round=N_ROUNDS, valid_sets=[dte],
                      callbacks=[lgb.log_evaluation(100)])
    nrel_e = nrel[is_eval]
    print("eval Recall@50  baseline: %.4f" % recall50(te, te.base.values, nrel_e))
    for n in [200, 400, 600]:
        s = model.predict(te[feats], num_iteration=n)
        print(f"eval Recall@50  ranker ({n} trees): %.4f" % recall50(te, s, nrel_e))
    imp = pd.Series(model.feature_importance("gain"), index=feats).sort_values(ascending=False)
    print((imp / imp.sum()).head(25).round(4).to_string())


if __name__ == "__main__":
    main()
