"""
Шаг 2b. Плотные эмбеддинги: двухбашенная модель (two-tower), обученная с нуля на логах.

Зачем: BM25 не находит объявления, у которых нет общих слов с запросом
(«чистка пяток» -> «Подолог», «подписка нетфликс» -> «настройка Смарт ТВ»).
Плотные векторы, обученные на парах «запрос -> выбранное объявление», переносят
такие смысловые связи на новые объявления.

Почему своя модель, а не предобученная (e5 / bge): решение должно воспроизводиться
локально без скачивания весов и без GPU. Модель в духе fastText/StarSpace
обучается за несколько минут на CPU и при этом заточена под задачу.

Архитектура:
  * признаки текста = стемы слов + символьные 3-граммы слов (ловят опечатки,
    словоформы, составные слова вроде «автоподбор» / «подбор авто»);
    каждый признак хэшируется (crc32) в одну из 2^19 корзин;
  * общая таблица эмбеддингов EmbeddingBag(mean) для запросов и объявлений
    (одно слово — один вектор в обеих башнях), поверх — отдельная линейная
    «голова» для каждой башни, затем L2-нормировка;
  * башня объявления видит: заголовок (x2), символьные 3-граммы заголовка,
    уникальные стемы параметров и начала описания, токен микрокатегории;
  * функция потерь — softmax по батчу (in-batch negatives) с температурой 0.05;
    объявления из пар с тем же текстом запроса в батче маскируются, чтобы не
    считать их «негативами» (у «маникюр» тысячи правильных объявлений).

Запуск: python embed.py val   — обучение на fit-части валидации, эмбеддинги val-корпуса
        python embed.py full  — обучение на всём train, эмбеддинги корпуса бенчмарка
"""
import sys
import time
import zlib

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import normalize, stem_text, tokenize

WORK = "work"
DATA = "data"
N_BUCKETS = 1 << 19
DIM = 128
EPOCHS = 5
BATCH = 1024
TEMP = 0.05
SEED = 0
MAX_ITEM_TOKENS = 200

torch.manual_seed(SEED)
np.random.seed(SEED)
torch.set_num_threads(2)


# ---------------------------------------------------------------------------
# Токенизация в хэш-индексы
# ---------------------------------------------------------------------------

def _h(tok: str) -> int:
    return zlib.crc32(tok.encode()) % N_BUCKETS


def _char3(words):
    """Символьные 3-граммы слов с метками границ: «маникюр» -> <ма, ман, ани, ..., юр>."""
    out = []
    for w in words:
        w = f"<{w}>"
        out.extend("#" + w[i:i + 3] for i in range(len(w) - 2))
    return out


def query_tokens(raw_query: str) -> list:
    words = [w for w in normalize(raw_query).split() if w]
    stems = tokenize(raw_query)
    return [_h(t) for t in stems + _char3(words)] or [0]


def item_tokens(title_raw, t_title, t_params, t_desc, microcat) -> list:
    title_stems = t_title.split()
    words = normalize(title_raw).replace(",", " ").replace(".", " ").split()
    params = list(dict.fromkeys(t_params.split()))[:40]
    desc = list(dict.fromkeys(t_desc.split()))[:40]
    toks = title_stems * 2 + _char3(words) + params + desc + [f"@mc{microcat}"]
    return [_h(t) for t in toks[:MAX_ITEM_TOKENS]] or [0]


def ragged(lists):
    """Список списков -> (плоский массив, смещения) для EmbeddingBag."""
    lens = np.fromiter((len(x) for x in lists), np.int64, len(lists))
    offsets = np.zeros(len(lists) + 1, np.int64)
    np.cumsum(lens, out=offsets[1:])
    flat = np.fromiter((t for x in lists for t in x), np.int64, offsets[-1])
    return flat, offsets


def load_item_tokens(item_ids) -> tuple:
    """Токены объявлений из work/items_all.parquet (потоково, экономя память).
    Возвращает (ids, flat, offsets) в порядке ids."""
    need = set(item_ids)
    ids, toks = [], []
    pf = pq.ParquetFile(f"{WORK}/items_all.parquet")
    cols = ["item_id", "item_title_raw", "t_title", "t_params", "t_desc", "item_microcat_id"]
    for b in pf.iter_batches(batch_size=50000, columns=cols):
        d = b.to_pandas()
        d = d[d.item_id.isin(need)]
        for r in d.itertuples(index=False):
            ids.append(r.item_id)
            toks.append(item_tokens(r.item_title_raw or "", r.t_title or "", r.t_params or "",
                                    r.t_desc or "", r.item_microcat_id))
    flat, offs = ragged(toks)
    return np.array(ids), flat, offs


# ---------------------------------------------------------------------------
# Модель
# ---------------------------------------------------------------------------

class TwoTower(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.EmbeddingBag(N_BUCKETS, DIM, mode="mean", sparse=True)
        nn.init.normal_(self.emb.weight, std=0.1)
        self.q_head = nn.Linear(DIM, DIM)
        self.i_head = nn.Linear(DIM, DIM)

    def encode(self, flat, offs, head):
        x = self.emb(flat, offs)
        return F.normalize(head(x) + x, dim=-1)  # остаточная связь: голова лишь «доворачивает»

    def q(self, flat, offs):
        return self.encode(flat, offs, self.q_head)

    def i(self, flat, offs):
        return self.encode(flat, offs, self.i_head)


def _batch(flat, offs, idx):
    """Собирает батч из рваного массива по индексам строк."""
    starts, ends = offs[idx], offs[idx + 1]
    lens = ends - starts
    b_offs = np.zeros(len(idx), np.int64)
    np.cumsum(lens[:-1], out=b_offs[1:])
    b_flat = np.concatenate([flat[s:e] for s, e in zip(starts, ends)])
    return torch.from_numpy(b_flat), torch.from_numpy(b_offs)


def train(pairs: pd.DataFrame, it_ids, it_flat, it_offs):
    """pairs: колонки search_query, item_id (пары из логов)."""
    pairs = pairs.drop_duplicates(["search_query", "item_id"])
    row_of = pd.Series(np.arange(len(it_ids)), index=it_ids)
    pairs = pairs[pairs.item_id.isin(row_of.index)]
    uq, q_code = np.unique(pairs.search_query.values, return_inverse=True)
    q_flat, q_offs = ragged([query_tokens(q) for q in uq])
    item_row = row_of.loc[pairs.item_id].values
    print(f"[embed] pairs {len(pairs)}, queries {len(uq)}, items {len(it_ids)}", flush=True)

    model = TwoTower()
    opt_s = torch.optim.SparseAdam(list(model.emb.parameters()), lr=0.01)
    opt_d = torch.optim.Adam(list(model.q_head.parameters()) + list(model.i_head.parameters()), lr=1e-3)
    n = len(pairs)
    for ep in range(EPOCHS):
        t0, tot = time.time(), 0.0
        perm = np.random.permutation(n)
        for s in range(0, n - BATCH + 1, BATCH):
            b = perm[s:s + BATCH]
            qc = q_code[b]
            qv = model.q(*_batch(q_flat, q_offs, qc))
            iv = model.i(*_batch(it_flat, it_offs, item_row[b]))
            logits = qv @ iv.T / TEMP
            # одинаковый текст запроса -> «чужое» объявление тоже релевантно, не штрафуем
            qt = torch.from_numpy(qc)
            same = (qt[:, None] == qt[None, :]) & ~torch.eye(len(b), dtype=torch.bool)
            logits = logits.masked_fill(same, -1e4)
            target = torch.arange(len(b))
            loss = F.cross_entropy(logits, target)
            opt_s.zero_grad()
            opt_d.zero_grad()
            loss.backward()
            opt_s.step()
            opt_d.step()
            tot += loss.item()
        print(f"[embed] epoch {ep + 1}: loss {tot / (n // BATCH):.4f}  {time.time() - t0:.0f}s", flush=True)
    return model


@torch.no_grad()
def encode_items(model, flat, offs, bs=8192):
    model.eval()
    out = []
    n = len(offs) - 1
    for s in range(0, n, bs):
        idx = np.arange(s, min(s + bs, n))
        out.append(model.i(*_batch(flat, offs, idx)).numpy())
    return np.vstack(out).astype(np.float32)


@torch.no_grad()
def encode_queries(model, queries, bs=8192):
    model.eval()
    flat, offs = ragged([query_tokens(q) for q in queries])
    out = []
    n = len(queries)
    for s in range(0, n, bs):
        idx = np.arange(s, min(s + bs, n))
        out.append(model.q(*_batch(flat, offs, idx)).numpy())
    return np.vstack(out).astype(np.float32)


def main(mode: str):
    if mode == "val":
        pairs = pd.read_parquet(f"{WORK}/val_fit_pairs.parquet", columns=["search_query", "item_id"])
        corpus_ids = pd.read_parquet(f"{WORK}/val_corpus.parquet").item_id.values
        queries = pd.read_parquet(f"{WORK}/val_queries.parquet").search_query.values
    else:
        pairs = pd.read_parquet(f"{WORK}/train_pairs.parquet", columns=["search_query", "item_id"])
        corpus_ids = pd.read_parquet(f"{DATA}/benchmark_items.parquet", columns=["item_id"]).item_id.values
        queries = pd.read_parquet(f"{WORK}/bench_queries.parquet").search_query.values

    t0 = time.time()
    need = set(pairs.item_id) | set(corpus_ids)
    ids, flat, offs = load_item_tokens(need)
    print(f"[embed] tokenized {len(ids)} items: {time.time() - t0:.0f}s", flush=True)

    model = train(pairs, ids, flat, offs)

    # эмбеддинги корпуса (в порядке corpus_ids) и запросов
    row_of = pd.Series(np.arange(len(ids)), index=ids)
    rows = row_of.loc[corpus_ids].values
    sub_flat_offs = [(offs[r], offs[r + 1]) for r in rows]
    c_flat, c_offs = ragged([flat[a:b].tolist() for a, b in sub_flat_offs])
    E = encode_items(model, c_flat, c_offs)
    Q = encode_queries(model, queries)
    np.save(f"{WORK}/emb_items_{mode}.npy", E)
    pd.DataFrame({"item_id": corpus_ids}).to_parquet(f"{WORK}/emb_items_{mode}_ids.parquet", index=False)
    np.save(f"{WORK}/emb_queries_{mode}.npy", Q)
    torch.save(model.state_dict(), f"{WORK}/two_tower_{mode}.pt")
    print(f"[embed] done: {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "val")
