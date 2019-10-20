import numpy as np
import pandas as pd
from typing import Tuple, Iterable, Callable

from big_roc.metrics import cosine_similarity_01


def split(df: pd.DataFrame, batch_size: int) -> Iterable[pd.DataFrame]:
    is_remainder = (df.shape[0] % batch_size != 0)
    split_indices = range(batch_size, (df.shape[0] // batch_size + is_remainder) * batch_size, batch_size)
    return np.split(df, split_indices)


def calc_gen_imp_sim(s1: pd.DataFrame, s2: pd.DataFrame, similarity_measure: Callable) -> Tuple[np.ndarray, np.ndarray]:
    similarities = similarity_measure(s1.values, s2.values)
    is_genuine = (s1.index.values.reshape(-1, 1) == s2.index.values.reshape(1, -1))
    return similarities[is_genuine], similarities[~is_genuine]


def update_hist(hist: np.ndarray, bin_edges: np.ndarray, similarities: np.ndarray) -> None:
    indices = np.searchsorted(bin_edges, similarities)
    np.add.at(hist, indices, 1)


def gen_imp_histogram(s1: pd.DataFrame, s2: pd.DataFrame, bin_edges: np.ndarray, batch_size: int = 1000,
                      similarity_measure: Callable = cosine_similarity_01) -> Tuple[np.ndarray, np.ndarray]:
    assert all(np.diff(bin_edges) > 0)

    gen_hist = np.zeros(len(bin_edges) + 1, dtype=np.int64)
    imp_hist = np.zeros(len(bin_edges) + 1, dtype=np.int64)

    for s1_batch in split(s1, batch_size):
        for s2_batch in split(s2, batch_size):
            gen_sim, imp_sim = calc_gen_imp_sim(s1_batch, s2_batch, similarity_measure)
            update_hist(gen_hist, bin_edges, gen_sim)
            update_hist(imp_hist, bin_edges, imp_sim)
    return gen_hist, imp_hist


def convert_to_numpy_style_histogram(gen_hist, imp_hist, bin_edges):
    assert gen_hist[0] == gen_hist[-1] == imp_hist[0] == imp_hist[-1] == 0
    return gen_hist[1:-1], imp_hist[1:-1], bin_edges
