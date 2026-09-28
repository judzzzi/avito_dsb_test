"""
Контекст поиска: всё, что строится один раз по корпусу и по логам (train),
а затем используется для генерации кандидатов и признаков.

Состав:
  1. BM25-индексы корпуса по полям: заголовок, параметры, описание и «всё вместе».
  2. Модель локации: P(локация объявления | локация поиска) по логам + геодистанция.
  3. Коды «Вид услуги» / «Тип услуги» объявлений для проверки фильтров запроса.
  4. Модель логов (query log): ближайшие по тексту запросы из train →
       * «профиль» запроса — какие слова стоят в заголовках выбранных объявлений
         (расширение запроса: «автоподбор» -> «осмотр», «эндоскопия», «толщиномер» ...);
       * распределение микрокатегорий выбранных объявлений;
       * история самих объявлений (сколько раз выбирали и по каким запросам).

Один и тот же код используется и на валидации (логи = fit-часть train,
корпус = валидационный корпус), и на бенчмарке (логи = весь train, корпус = бенчмарк).
"""
import re

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from bm25 import BM25
from common import normalize, parse_filters

# ---------------------------------------------------------------------------
# Вспомогательное
# ---------------------------------------------------------------------------

def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _extract_value(text: str, key: str, values_re) -> str | None:
    """Находит значение ключа (из известного списка) в строке параметров объявления."""
    if not isinstance(text, str):
        return None
    m = values_re.search(text)
    return m.group(1) if m else None


def _topk_rows(M, k):
    """Оставляет в каждой строке разреженной матрицы top-k значений."""
    M = M.tocsr()
    indptr, indices, data = [0], [], []
    for i in range(M.shape[0]):
        a, b = M.indptr[i], M.indptr[i + 1]
        idx, d = M.indices[a:b], M.data[a:b]
        if len(d) > k:
            sel = np.argpartition(-d, k)[:k]
            idx, d = idx[sel], d[sel]
        indices.append(idx)
        data.append(d)
        indptr.append(indptr[-1] + len(idx))
    return sp.csr_matrix((np.concatenate(data), np.concatenate(indices), np.array(indptr)),
                         shape=M.shape)


FKEYS = ("Вид услуги", "Тип услуги", "Тип услуги автосервиса")


LIGHT_COLS = ["item_id", "t_title", "item_microcat_id", "item_location_id",
              "item_latitude", "item_longitude"]


def load_inputs(items_path: str, corpus_ids):
    """
    Загружает объявления экономно по памяти (все тексты сразу не помещаются в 7 ГБ):
      corpus    — только объявления корпуса, со всеми полями;
      light     — все объявления, но только лёгкие поля (заголовок, микрокатегория, гео).
    """
    corpus_ids = list(set(corpus_ids))
    corpus = pd.read_parquet(items_path, filters=[("item_id", "in", corpus_ids)])
    light = pd.read_parquet(items_path, columns=LIGHT_COLS)
    return corpus, light


class Context:
    def __init__(self, corpus: pd.DataFrame, light: pd.DataFrame, fit: pd.DataFrame, verbose=True):
        """
        corpus — объявления, среди которых ищем (все поля, см. prep.py);
        light  — лёгкие поля всех известных объявлений (нужны для статистик по логам);
        fit    — логи: пары «запрос — выбранное объявление».
        """
        self.verbose = verbose
        C = corpus.reset_index(drop=True)
        self.C = C
        self.N = len(C)
        self.item_ids = C.item_id.values
        self._log(f"corpus: {self.N} items, fit rows: {len(fit)}")

        self._build_bm25()
        self._build_item_arrays()
        self._build_filters(fit)
        # тексты корпуса больше не нужны — освобождаем память
        self.C = C = C[["item_id"]]
        self._build_location(light, fit)
        self._build_querylog(light, fit)

    def _log(self, msg):
        if self.verbose:
            import resource
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
            print(f"[context] {msg} (peak RSS {rss} MB)", flush=True)

    # ------------------------------------------------------------------ BM25
    def _build_bm25(self):
        C = self.C
        # заголовок — самое информативное поле, поэтому в «общем» индексе дублируем его
        all_text = (C.t_title + " " + C.t_title + " " + C.t_params + " " + C.t_desc).values
        self.bm_title = BM25(k1=1.2, b=0.5).fit(C.t_title.values)
        self.bm_params = BM25(k1=1.2, b=0.75).fit(C.t_params.values)
        self.bm_desc = BM25(k1=1.2, b=0.75).fit(C.t_desc.values)
        self.bm_all = BM25(k1=1.2, b=0.75).fit(all_text)
        # «доля покрытых терминов запроса» (idf-взвешенная): индекс с весом idf,
        # если термин есть в документе, и 0 иначе. Для заголовка и для всего текста.
        self.cov_title = self._binary_idf(self.bm_title)
        self.cov_all = self._binary_idf(self.bm_all)
        for bm in (self.bm_title, self.bm_params, self.bm_desc, self.bm_all):
            bm.W = None  # хватает транспонированной матрицы, экономим память
        self._log("bm25 built")

    @staticmethod
    def _binary_idf(bm):
        WT = bm.WT.copy()
        rows = np.repeat(np.arange(WT.shape[0]), np.diff(WT.indptr))
        WT.data = bm.idf[rows].astype(np.float32)
        return WT

    # ------------------------------------------------------------------ items
    def _build_item_arrays(self):
        C = self.C
        self.loc = C.item_location_id.values.astype(np.int64)
        self.lat = C.item_latitude.fillna(0).values.astype(np.float64)
        self.lon = C.item_longitude.fillna(0).values.astype(np.float64)
        self.mc = C.item_microcat_id.values.astype(np.int64)
        self.cat = C.item_category_id.values.astype(np.int64)
        price = C.item_price.values.astype(np.float64)
        self.log_price = np.log1p(np.clip(price, 0, 1e7))
        self.rating = C.item_rating.fillna(-1).values.astype(np.float32)
        self.reviews = np.log1p(C.item_rating_reviews_count.fillna(0).values).astype(np.float32)
        self.phone_hidden = C.item_is_phone_hidden.values.astype(np.float32)
        self.msg_forbidden = C.item_is_message_forbidden.values.astype(np.float32)
        self.title_len = C.t_title.str.count(" ").values.astype(np.float32) + 1
        self.desc_len = C.t_desc.str.count(" ").values.astype(np.float32)

    # ------------------------------------------------------------------ фильтры
    def _build_filters(self, fit):
        """Коды значений «Вид услуги»/«Тип услуги»/«Тип услуги автосервиса» у объявлений."""
        values = {k: set() for k in FKEYS}
        for s in fit.search_infm_params_text.unique():
            for k, vs in parse_filters(s).items():
                if k in values:
                    values[k].update(vs)
        self.fvals = {k: {v: i + 1 for i, v in enumerate(sorted(vs))} for k, vs in values.items()}
        self.item_fcode = {}
        params = self.C.item_infm_params_text.values
        for k in FKEYS:
            vals = sorted(self.fvals[k], key=len, reverse=True)
            if not vals:
                self.item_fcode[k] = np.zeros(self.N, np.int32)
                continue
            # «Тип услуги X» не должен совпадать с «Тип услуги автосервиса X»
            neg = r"(?!автосервиса )" if k == "Тип услуги" else ""
            rx = re.compile(re.escape(k) + " " + neg + "(" + "|".join(map(re.escape, vals)) + r")(?= |$)")
            codes = np.zeros(self.N, np.int32)
            for i, p in enumerate(params):
                v = _extract_value(p, k, rx)
                if v is not None:
                    codes[i] = self.fvals[k][v]
            self.item_fcode[k] = codes
        self._log("filters built")

    def query_filter_codes(self, s: str) -> dict:
        f = parse_filters(s)
        out = {}
        for k in FKEYS:
            vs = [self.fvals[k].get(v, -1) for v in f.get(k, [])]
            if vs:
                out[k] = vs
        return out

    # ------------------------------------------------------------------ локация
    def _build_location(self, items_all, fit):
        """
        P(loc_item | loc_search) считаем по логам: в каких локациях находятся
        объявления, выбранные при поиске в данной локации. Для «региональных»
        локаций поиска (Москва+МО, вся Россия и т.п.) это единственный способ
        понять, какие объявления им подходят.
        Центроид локации поиска: медиана координат объявлений этой локации,
        а если это не локация объявлений — медиана координат выбранных объявлений.
        """
        il = items_all[["item_id", "item_location_id", "item_latitude", "item_longitude"]]
        cent = il.groupby("item_location_id")[["item_latitude", "item_longitude"]].median()
        tr = fit[["search_location_id", "item_id"]].merge(il, on="item_id")
        cent2 = tr.groupby("search_location_id")[["item_latitude", "item_longitude"]].median()
        cent2 = cent2[~cent2.index.isin(cent.index)]
        self.centroid = pd.concat([cent, cent2])
        self.item_loc_set = set(cent.index)

        cnt = tr.groupby(["search_location_id", "item_location_id"]).size()
        tot = cnt.groupby(level=0).transform("sum")
        self.loc_p = {}
        for (s, l), p in (cnt / tot).items():
            self.loc_p.setdefault(s, {})[l] = p
        self.loc_n = tr.groupby("search_location_id").size().to_dict()
        # уникальные локации корпуса -> индекс, чтобы считать prior на уровне локаций
        self.uloc, self.loc_inv = np.unique(self.loc, return_inverse=True)
        cl = self.centroid.reindex(self.uloc)
        self.uloc_lat = cl.item_latitude.values
        self.uloc_lon = cl.item_longitude.values
        self._log("location model built")

    def loc_features(self, s_loc: int):
        """
        Для локации поиска возвращает по каждой уникальной локации корпуса:
          p_cnt  — P(loc_item | loc_search) по логам,
          same   — совпадение локаций,
          dist   — расстояние (км) от центроида поиска до центроида локации,
          logp   — итоговый лог-prior для скоринга кандидатов.
        """
        U = len(self.uloc)
        same = (self.uloc == s_loc).astype(np.float32)
        d = self.loc_p.get(s_loc)
        p_cnt = np.zeros(U, np.float32)
        if d:
            idx = np.searchsorted(self.uloc, np.fromiter(d.keys(), np.int64))
            ok = idx < U
            keys = np.fromiter(d.keys(), np.int64)
            ok &= self.uloc[np.minimum(idx, U - 1)] == keys
            p_cnt[idx[ok]] = np.fromiter(d.values(), np.float32)[ok]
        if s_loc in self.centroid.index:
            c = self.centroid.loc[s_loc]
            dist = haversine_km(c.item_latitude, c.item_longitude, self.uloc_lat, self.uloc_lon)
            dist = np.nan_to_num(dist, nan=5000.0)
        else:
            dist = np.full(U, 5000.0)
        # Сглаживание: совпадение локаций само по себе даёт высокий prior (важно для
        # локаций, которых мало в логах), а соседние по расстоянию — небольшой.
        p_geo = 0.05 * np.exp(-dist / 30.0)
        p = np.maximum.reduce([p_cnt, 0.5 * same, p_geo])
        logp = np.log(p + 1e-4).astype(np.float32)
        return p_cnt, same, dist.astype(np.float32), logp

    # ------------------------------------------------------------------ логи запросов
    def _build_querylog(self, items_all, fit):
        # Уникальные тексты запросов из логов
        fit = fit[["search_query", "q_stem", "item_id"]].copy()
        fit["qn"] = fit.search_query.map(normalize)
        qtexts = fit.qn.unique()
        self.fq_text = qtexts
        q2i = {q: i for i, q in enumerate(qtexts)}
        fit["qi"] = fit.qn.map(q2i).values
        self.fq_count = np.bincount(fit.qi.values, minlength=len(qtexts)).astype(np.float32)

        # TF-IDF для поиска ближайших запросов: слова + символьные n-граммы
        # (символьные n-граммы ловят опечатки и словоформы: «шиномонтаж»/«шиномантаж»).
        self.q_vec_char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                                          sublinear_tf=True, dtype=np.float32)
        self.fq_mat = self.q_vec_char.fit_transform(qtexts).tocsr()
        self.fq_mat_T = self.fq_mat.T.tocsr()

        # Объявления из логов: заголовок (стемы) и микрокатегория
        fi = items_all[items_all.item_id.isin(set(fit.item_id))][["item_id", "t_title", "item_microcat_id"]]
        fi = fi.set_index("item_id")
        fit = fit[fit.item_id.isin(fi.index)]

        # Профиль запроса: доля выбранных объявлений, в заголовке которых есть термин.
        # Словарь — словарь заголовков корпуса, чтобы профиль сразу применять к BM25 заголовков.
        vec = self.bm_title.vec
        B = vec.transform(fi.t_title.values).tocsr()
        B.data[:] = 1.0
        item_row = pd.Series(np.arange(len(fi)), index=fi.index)
        rows = fit.qi.values
        cols = item_row.loc[fit.item_id].values
        A = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(len(qtexts), len(fi)))
        A = sp.diags(1.0 / np.maximum(np.asarray(A.sum(1)).ravel(), 1)).dot(A)
        self.fq_prof = (A @ B).tocsr().astype(np.float32)

        # «Словарь перевода» термин запроса -> термины заголовков (обучаем по логам).
        # Для термина t запроса: P(w|t) = средняя доля заголовков выбранных объявлений
        # с термином w среди запросов, содержащих t. Вес = P(w|t) * log(lift), где
        # lift = P(w|t) / P(w) — так остаются специфичные «переводы», а не общие слова:
        # «пяток» -> «подолог», «педикюр»; «автоподбор» -> «осмотр», «толщиномер».
        q_stems = (fit.drop_duplicates("qi").set_index("qi").q_stem
                   .reindex(range(len(qtexts))).fillna("").values)
        self.tr_vec = CountVectorizer(token_pattern=r"\S+", min_df=2, binary=True,
                                      lowercase=False, dtype=np.float32)
        Qtb = self.tr_vec.fit_transform(q_stems).tocsr()
        cnt_t = np.asarray(Qtb.sum(axis=0)).ravel()
        X = (Qtb.T @ self.fq_prof).tocsr()                       # Vq x Vt
        p_w = np.asarray(self.fq_prof.mean(axis=0)).ravel() + 1e-6
        X = sp.diags(1.0 / np.maximum(cnt_t, 1)).dot(X).tocsr()  # P(w|t)
        rows_x = np.repeat(np.arange(X.shape[0]), np.diff(X.indptr))
        lift = X.data / p_w[X.indices]
        X.data = (X.data * np.log(np.maximum(lift, 1.0))).astype(np.float32)
        # термины с поддержкой < 2 запросов ненадёжны
        X.data[cnt_t[rows_x] < 2] = 0
        X.eliminate_zeros()
        self.tr_mat = _topk_rows(X, 50)

        # Распределение микрокатегорий выбранных объявлений для каждого запроса
        mcs, mc_codes = np.unique(fi.item_microcat_id.values, return_inverse=True)
        self.mc_list = mcs
        M = sp.csr_matrix((np.ones(len(fi), np.float32), (np.arange(len(fi)), mc_codes)),
                          shape=(len(fi), len(mcs)))
        self.fq_mc = (A @ M).tocsr().astype(np.float32)
        corpus_mc_idx = np.searchsorted(mcs, self.mc)
        corpus_mc_idx[corpus_mc_idx >= len(mcs)] = 0
        self.item_mc_code = np.where(mcs[corpus_mc_idx] == self.mc, corpus_mc_idx, -1)

        # История объявлений корпуса в логах: популярность и «какие запросы к ним вели»
        id2c = pd.Series(np.arange(self.N), index=self.item_ids)
        fc = fit[fit.item_id.isin(id2c.index)]
        self.item_hist_cnt = np.zeros(self.N, np.float32)
        np.add.at(self.item_hist_cnt, id2c.loc[fc.item_id].values, 1)
        # матрица «запрос логов x объявление корпуса» (для истории по соседним запросам)
        self.fq_item = sp.csr_matrix(
            (np.ones(len(fc), np.float32), (fc.qi.values, id2c.loc[fc.item_id].values)),
            shape=(len(qtexts), self.N)).tocsr()
        self._log(f"query log built: {len(qtexts)} texts, corpus items with history: "
                  f"{(self.item_hist_cnt > 0).sum()}")

    def set_embeddings(self, E: np.ndarray, ids):
        """Плотные эмбеддинги корпуса (из embed.py), выравниваем по порядку ctx.item_ids."""
        row = pd.Series(np.arange(len(ids)), index=ids)
        self.emb = np.ascontiguousarray(E[row.loc[self.item_ids].values])

    def neighbors(self, queries_norm, k=30, min_sim=0.25):
        """Разреженная матрица (запросы x тексты логов) с косинусной близостью top-k соседей."""
        Q = self.q_vec_char.transform(queries_norm).tocsr()
        S = (Q @ self.fq_mat_T).tocsr()
        rows, cols, vals = [], [], []
        for i in range(S.shape[0]):
            a, b = S.indptr[i], S.indptr[i + 1]
            idx, d = S.indices[a:b], S.data[a:b]
            m = d >= min_sim
            idx, d = idx[m], d[m]
            if len(d) > k:
                sel = np.argpartition(-d, k)[:k]
                idx, d = idx[sel], d[sel]
            rows.append(np.full(len(idx), i))
            cols.append(idx)
            vals.append(d)
        rows = np.concatenate(rows) if rows else np.array([], int)
        return sp.csr_matrix((np.concatenate(vals), (rows, np.concatenate(cols))),
                             shape=(S.shape[0], len(self.fq_text)))
