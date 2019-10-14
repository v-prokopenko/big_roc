import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from typing import Tuple, Iterable, Callable, List
import multiprocessing
from functools import partial


def split(df: pd.DataFrame, chunk_size: int) -> Iterable[pd.DataFrame]:
    is_remainder = (df.shape[0] % chunk_size != 0)
    split_indices = range(chunk_size, (df.shape[0] // chunk_size + is_remainder) * chunk_size, chunk_size)
    return np.split(df, split_indices)


def calc_gen_imp_sim(s1: pd.DataFrame, s2: pd.DataFrame, similarity_measure: Callable) -> Tuple[np.array, np.array]:
    similarities = similarity_measure(s1.values, s2.values)
    is_genuine = (s1.index.values.reshape(-1, 1) == s2.index.values.reshape(1, -1))
    genuine = similarities[is_genuine]
    impostor = similarities[~is_genuine]
    return genuine, impostor


def update_bins(hist: np.array, bin_edges: np.array, similarities: np.array) -> None:
    indices = np.searchsorted(bin_edges, similarities)
    np.add.at(hist, indices, 1)


# def update_bins(hist: np.array, bin_edges: np.array, similarities: np.array) -> None:
#     np.sum(hist, np.histogram(similarities, bins=bin_edges)[0], out=hist)


def _calc_gen_imp_hist(s2_batches: Iterable[pd.DataFrame], bin_edges: np.array, similarity_measure: Callable,
                       s1_batch: pd.DataFrame) -> Tuple[np.array, np.array]:
    gen_hist = np.zeros(len(bin_edges) + 1, dtype=np.int64)
    imp_hist = np.zeros(len(bin_edges) + 1, dtype=np.int64)

    for s2_batch in s2_batches:
        gen_sim, imp_sim = calc_gen_imp_sim(s1_batch, s2_batch, similarity_measure)
        update_bins(gen_hist, bin_edges, gen_sim)
        update_bins(imp_hist, bin_edges, imp_sim)
    return gen_hist, imp_hist


def calc_gen_imp_hist(s1: pd.DataFrame, s2: pd.DataFrame, bin_edges: np.array, batch_size: int = 10000,
                      similarity_measure: Callable = cosine_similarity, n_workers: int = multiprocessing.cpu_count()) \
        -> Tuple[np.array, np.array]:
    gen_hist = (np.zeros(len(bin_edges) + 1, dtype=np.int64), bin_edges.copy())
    imp_hist = (np.zeros(len(bin_edges) + 1, dtype=np.int64), bin_edges.copy())

    func = partial(_calc_gen_imp_hist, split(s2, batch_size), bin_edges, similarity_measure)
    with multiprocessing.Pool(n_workers) as pool:
        results = pool.map(func, split(s1, batch_size))

    return results
    # return gen_hist, imp_hist

    # for s1_batch in split(s1, batch_size):
    #
    #     gen_hist, imp_hist = for s2_batch in split(s2, batch_size):
    #         gen_sim, imp_sim = calc_gen_imp_sim(s1_batch, s2_batch, similarity_measure)
    #         update_bins(gen_bins, gen_sim)
    #         update_bins(imp_bins, imp_sim)
    #         # b = np.histogram(imp_sim, imp_bins[0])[0]
    #         # imp_bins[1] = imp_bins[1] + b
    #         # indices = np.searchsorted(imp_bins[0], imp_sim)
    #         # np.add.at(imp_bins[1], indices, 1)
    # return gen_bins, imp_bins


def calc_confusion_matrices(gen_hist: np.array, imp_hist: np.array) -> pd.DataFrame:
    cum_gen, cum_imp = np.cumsum(gen_hist), np.cumsum(imp_hist)
    sum_gen, sum_imp = np.sum(gen_hist), np.sum(imp_hist)
    return pd.DataFrame([cum_gen, cum_imp, -cum_gen + sum_gen, -cum_imp + sum_imp], columns=['tp', 'fn', 'fp', 'tn'])


def fpr(cm):
    return cm['fp'] / (cm['fp'] + cm['tn'])


def fnr(cm):
    return cm['fn'] / (cm['fn'] + cm['tp'])



# np.random.seed(0)
# n_feat = 10
# size1, size2 = 10, 10
# s1 = pd.DataFrame(np.random.normal(size=(size1, n_feat)))
# s2 = pd.DataFrame(s1.values[:size2] + np.random.normal(size=(size2, n_feat)))
#
# sim_min = -1
# sim_max = 1
# eps = 1e-4
# n_intervals = 10
# intervals = np.linspace(sim_min - eps, sim_max + eps, n_intervals)
#
# calc_gen_imp_hist(s1, s2, intervals, batch_size=10000)
#
# print("Hello")
