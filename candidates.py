"""
Генерация кандидатов и признаков для ранкера.

Для каждого запроса объединяем несколько источников кандидатов (каждый — свой top-K
по «быстрому» скору), а затем для всех кандидатов объединения считаем единый набор
признаков. Дальше LightGBM-ранкер выбирает из объединения итоговые 50.

Источники (у всех к текстовому скору добавляется лог-prior локации, λ=2 подобрано
на валидации — без него recall@50 BM25 падает с ~0.81 до ~0.39):
  all   — BM25 по заголовку+параметрам+описанию                         top 200
  title — BM25 только по заголовку                                      top 100
  prof  — BM25 заголовков по «профилю» запроса из логов (расширение)    top 100
  tr    — BM25 заголовков по «переводу» терминов запроса (словарь из логов) top 100
  hist  — объявления, которые выбирали по этому/похожим запросам в логах top 50
  mc    — популярные объявления наиболее вероятной микрокатегории        top 50
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp

from common import normalize

LAMBDA_LOC = 2.0
K_SRC = {"all": 200, "title": 100, "prof": 100, "tr": 100, "hist": 50, "mc": 50}
PROF_TERMS = 30


def _topk(idx, score, k):
    if len(score) > k:
        sel = np.argpartition(-score, k)[:k]
        return idx[sel]
    return idx


def _row(S, i):
    a, b = S.indptr[i], S.indptr[i + 1]
    return S.indices[a:b], S.data[a:b]


def _lookup(S, i, cand):
    """Значения строки i разреженной матрицы S в столбцах cand (0, если нет)."""
    idx, d = _row(S, i)
    out = np.zeros(len(cand), np.float32)
    if len(idx) == 0:
        return out
    order = np.argsort(idx)
    idx, d = idx[order], d[order]
    pos = np.searchsorted(idx, cand)
    pos = np.minimum(pos, len(idx) - 1)
    hit = idx[pos] == cand
    out[hit] = d[pos[hit]]
    return out


def _topk_rows(M, k):
    """Оставляет в каждой строке разреженной матрицы top-k значений."""
    M = M.tocsr()
    rows, cols, vals = [], [], []
    for i in range(M.shape[0]):
        idx, d = _row(M, i)
        if len(d) > k:
            sel = np.argpartition(-d, k)[:k]
            idx, d = idx[sel], d[sel]
        rows.append(np.full(len(idx), i))
        cols.append(idx)
        vals.append(d)
    return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                         shape=M.shape)


def _row_normalize(M):
    s = np.asarray(M.sum(axis=1)).ravel()
    return sp.diags(1.0 / np.maximum(s, 1e-9)).dot(M).tocsr()


def generate(ctx, queries: pd.DataFrame, batch_size: int = 250, verbose=True):
    """queries: DataFrame с колонками search_query, q_stem, search_location_id,
    search_infm_params_text. Возвращает DataFrame кандидатов с признаками."""
    out = []
    n = len(queries)
    for start in range(0, n, batch_size):
        qb = queries.iloc[start : start + batch_size]
        out.append(_generate_batch(ctx, qb, start))
        if verbose:
            print(f"[cands] {min(start + batch_size, n)}/{n}", flush=True)
    return pd.concat(out, ignore_index=True)


def _generate_batch(ctx, qb, offset):
    B = len(qb)
    stems = qb.q_stem.fillna("").values
    qnorm = [normalize(x) for x in qb.search_query.values]

    # --- текстовые скоры по всем документам (разреженные B x N) ---
    Qt = ctx.bm_title.query_matrix(stems)
    Qa = ctx.bm_all.query_matrix(stems)
    S_title = ctx.bm_title.scores(Qt)
    S_par = ctx.bm_params.scores(ctx.bm_params.query_matrix(stems))
    S_desc = ctx.bm_desc.scores(ctx.bm_desc.query_matrix(stems))
    S_all = ctx.bm_all.scores(Qa)
    C_title = (Qt @ ctx.cov_title).tocsr()   # сумма idf совпавших терминов (заголовок)
    C_all = (Qa @ ctx.cov_all).tocsr()       # сумма idf совпавших терминов (весь текст)
    q_idf_all = np.asarray(Qa @ ctx.bm_all.idf).ravel()  # сумма idf терминов запроса
    q_nterms = np.array([len(s.split()) for s in stems], np.float32)
    q_nknown = np.asarray(Qa.sum(axis=1)).ravel()

    # --- логи: соседние запросы, профиль, микрокатегории, история ---
    NB = ctx.neighbors(qnorm)
    NBw = NB.multiply(NB).tocsr()            # вес соседа = sim^2
    nb_max = np.asarray(NB.max(axis=1).todense()).ravel()
    nb_cnt = np.diff(NB.indptr).astype(np.float32)
    prof = _topk_rows(_row_normalize(NBw @ ctx.fq_prof), PROF_TERMS)
    S_prof = (prof @ ctx.bm_title.WT).tocsr()
    # «перевод» терминов запроса в термины заголовков (см. Context.tr_mat)
    Qtr = ctx.tr_vec.transform(stems)
    trans = _topk_rows(_row_normalize(Qtr @ ctx.tr_mat), PROF_TERMS)
    S_tr = (trans @ ctx.bm_title.WT).tocsr()
    P_mc = _row_normalize(NBw @ ctx.fq_mc).toarray()      # B x n_mc
    H = (NBw @ ctx.fq_item).tocsr()                          # история по похожим запросам
    # точные совпадения текста (sim ~ 1) — отдельная матрица истории
    NBe = NB.copy()
    NBe.data = (NBe.data > 0.99).astype(np.float32)
    NBe.eliminate_zeros()
    H_exact = (NBe @ ctx.fq_item).tocsr()

    for M in (S_title, S_par, S_desc, S_all, C_title, C_all, S_prof, S_tr, H, H_exact):
        M.sort_indices()

    rows = []
    for i in range(B):
        s_loc = int(qb.search_location_id.values[i])
        p_cnt_u, same_u, dist_u, logp_u = ctx.loc_features(s_loc)
        inv = ctx.loc_inv
        lp = logp_u[inv]

        # бонус за совпадение с фильтрами «Вид/Тип услуги» (доля выполненных условий)
        fcodes = ctx.query_filter_codes(qb.search_infm_params_text.values[i])
        n_cons = sum(len(v) for v in fcodes.values())
        fb = np.zeros(ctx.N, np.float32)
        for k, vs in fcodes.items():
            fb += np.isin(ctx.item_fcode[k], vs).astype(np.float32)
        if n_cons:
            fb /= n_cons

        # вероятность микрокатегории объявления по логам похожих запросов
        pmc_row = P_mc[i]
        pmc = np.where(ctx.item_mc_code >= 0, pmc_row[np.maximum(ctx.item_mc_code, 0)], 0.0).astype(np.float32)

        # ---- источники кандидатов ----
        src = {}
        for name, S in (("all", S_all), ("title", S_title), ("prof", S_prof), ("tr", S_tr)):
            idx, d = _row(S, i)
            if len(idx):
                sc = d + LAMBDA_LOC * lp[idx] + 0.5 * fb[idx]
                src[name] = _topk(idx, sc, K_SRC[name])
        idx, d = _row(H, i)
        if len(idx):
            src["hist"] = _topk(idx, np.log(d + 1e-3) + lp[idx], K_SRC["hist"])
        if pmc_row.sum() > 0:
            sc = np.log(pmc + 1e-3) + LAMBDA_LOC * lp + 0.1 * ctx.reviews + 0.5 * fb
            src["mc"] = _topk(np.arange(ctx.N), sc, K_SRC["mc"])
        if not src:
            # совсем пустой запрос: берём локально популярные объявления
            sc = LAMBDA_LOC * lp + 0.1 * ctx.reviews
            src["mc"] = _topk(np.arange(ctx.N), sc, K_SRC["mc"])

        cand = np.unique(np.concatenate(list(src.values())))
        m = len(cand)
        f = {
            "qi": np.full(m, offset + i, np.int32),
            "ci": cand.astype(np.int32),
        }
        for name in K_SRC:
            f["src_" + name] = np.isin(cand, src[name]).astype(np.int8) if name in src else np.zeros(m, np.int8)
        f["bm_title"] = _lookup(S_title, i, cand)
        f["bm_params"] = _lookup(S_par, i, cand)
        f["bm_desc"] = _lookup(S_desc, i, cand)
        f["bm_all"] = _lookup(S_all, i, cand)
        f["bm_prof"] = _lookup(S_prof, i, cand)
        f["bm_tr"] = _lookup(S_tr, i, cand)
        qi_idf = max(q_idf_all[i], 1e-6)
        f["cov_title"] = _lookup(C_title, i, cand) / qi_idf
        f["cov_all"] = _lookup(C_all, i, cand) / qi_idf
        f["hist_nb"] = _lookup(H, i, cand)
        f["hist_exact"] = _lookup(H_exact, i, cand)
        f["hist_cnt"] = np.log1p(ctx.item_hist_cnt[cand])
        li = inv[cand]
        f["loc_p"] = p_cnt_u[li]
        f["loc_same"] = same_u[li]
        f["loc_dist"] = np.log1p(dist_u[li])
        f["loc_logp"] = logp_u[li]
        f["filt_frac"] = fb[cand] if n_cons else np.full(m, -1, np.float32)
        f["pmc"] = pmc[cand]
        f["pmc_max"] = np.full(m, pmc_row.max() if len(pmc_row) else 0, np.float32)
        f["rating"] = ctx.rating[cand]
        f["reviews"] = ctx.reviews[cand]
        f["log_price"] = ctx.log_price[cand]
        f["phone_hidden"] = ctx.phone_hidden[cand]
        f["msg_forbidden"] = ctx.msg_forbidden[cand]
        f["title_len"] = ctx.title_len[cand]
        f["desc_len"] = ctx.desc_len[cand]
        f["cat114"] = (ctx.cat[cand] == 114).astype(np.int8)
        # признаки запроса
        f["q_nterms"] = np.full(m, q_nterms[i], np.float32)
        f["q_nknown"] = np.full(m, q_nknown[i], np.float32)
        f["q_idf"] = np.full(m, q_idf_all[i], np.float32)
        f["q_nb_max"] = np.full(m, nb_max[i], np.float32)
        f["q_nb_cnt"] = np.full(m, nb_cnt[i], np.float32)
        f["q_region"] = np.full(m, float(s_loc not in ctx.item_loc_set), np.float32)
        f["q_loc_n"] = np.full(m, np.log1p(ctx.loc_n.get(s_loc, 0)), np.float32)
        f["q_ncons"] = np.full(m, n_cons, np.float32)
        f["q_ncand"] = np.full(m, m, np.float32)
        rows.append(pd.DataFrame(f))
    df = pd.concat(rows, ignore_index=True)
    return df


def add_query_relative_features(df: pd.DataFrame) -> pd.DataFrame:
    """Признаки относительно других кандидатов того же запроса: ранги и доли от максимума."""
    g = df.groupby("qi", sort=False)
    for c in ["bm_title", "bm_all", "bm_prof", "bm_tr", "bm_desc", "pmc", "loc_logp", "reviews"]:
        mx = g[c].transform("max")
        df[c + "_rel"] = (df[c] / mx.where(mx > 0, 1)).astype(np.float32)
        df[c + "_rank"] = g[c].rank(ascending=False, method="min").astype(np.float32)
    # «быстрый» скор источника all и его ранг — хороший ориентир для ранкера
    df["base"] = (df.bm_all + 2.0 * df.loc_logp).astype(np.float32)
    df["base_rank"] = df.groupby("qi", sort=False)["base"].rank(ascending=False, method="min").astype(np.float32)
    return df
