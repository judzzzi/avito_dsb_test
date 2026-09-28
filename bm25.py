"""
Разреженный BM25 на scipy.sparse.

Индекс = матрица W (документы x термины) с уже посчитанными BM25-весами
    w(d,t) = idf(t) * tf*(k1+1) / (tf + k1*(1 - b + b*|d|/avgdl)).
Тогда скор запроса q по всем документам = W @ q_bin, где q_bin — бинарный
вектор терминов запроса. Для батча запросов это одно разреженное умножение.
"""
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer


class BM25:
    def __init__(self, k1: float = 1.2, b: float = 0.75, ngram_range=(1, 1), min_df=1,
                 vocabulary=None):
        self.k1, self.b = k1, b
        self.vec = CountVectorizer(token_pattern=r"\S+", ngram_range=ngram_range,
                                   min_df=min_df, lowercase=False, dtype=np.float32,
                                   vocabulary=vocabulary)

    def fit(self, docs):
        tf = self.vec.fit_transform(docs).tocsr().astype(np.float32)
        n_docs = tf.shape[0]
        df = np.bincount(tf.indices, minlength=tf.shape[1]).astype(np.float32)
        self.idf = np.log(1 + (n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)
        dl = np.asarray(tf.sum(axis=1)).ravel()
        avgdl = dl.mean() if dl.mean() > 0 else 1.0
        denom_norm = self.k1 * (1 - self.b + self.b * dl / avgdl)
        # tf -> BM25 вес, построчно
        rows = np.repeat(np.arange(n_docs), np.diff(tf.indptr))
        data = tf.data
        data = data * (self.k1 + 1) / (data + denom_norm[rows])
        data = data * self.idf[tf.indices]
        self.W = sp.csr_matrix((data.astype(np.float32), tf.indices, tf.indptr), shape=tf.shape)
        self.WT = self.W.T.tocsr()  # термины x документы: быстрый доступ к постингам
        return self

    def query_matrix(self, queries):
        """Бинарная матрица запросов (запросы x термины)."""
        q = self.vec.transform(queries).tocsr()
        q.data[:] = 1.0
        return q

    def scores(self, qmat):
        """Разреженная матрица скоров (запросы x документы)."""
        return (qmat @ self.WT).tocsr()
