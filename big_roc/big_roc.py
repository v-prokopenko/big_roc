import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from typing import Tuple, Iterable, Callable, List
import multiprocessing


def split(df: pd.DataFrame, batch_size: int) -> Iterable[pd.DataFrame]:
    is_remainder = (df.shape[0] % batch_size != 0)
    split_indices = range(batch_size, (df.shape[0] // batch_size + is_remainder) * batch_size, batch_size)
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


def calc_gen_imp_hist(s1: pd.DataFrame, s2: pd.DataFrame, bin_edges: np.array, batch_size: int = 10000,
                      similarity_measure: Callable = cosine_similarity) -> Tuple[np.array, np.array]:
    gen_hist = np.zeros(len(bin_edges) + 1, dtype=np.int64)
    imp_hist = np.zeros(len(bin_edges) + 1, dtype=np.int64)

    for s1_batch in split(s1, batch_size):
        for s2_batch in split(s2, batch_size):
            gen_sim, imp_sim = calc_gen_imp_sim(s1_batch, s2_batch, similarity_measure)
            update_bins(gen_hist, bin_edges, gen_sim)
            update_bins(imp_hist, bin_edges, imp_sim)
    return gen_hist, imp_hist


def calc_confusion_matrices(gen_hist: np.array, imp_hist: np.array) -> pd.DataFrame:
    cum_gen, cum_imp = np.cumsum(gen_hist), np.cumsum(imp_hist)
    sum_gen, sum_imp = np.sum(gen_hist), np.sum(imp_hist)
    return pd.DataFrame({'tp': sum_gen - cum_gen, 'fn': cum_gen, 'fp': sum_imp - cum_imp, 'tn': cum_imp})


def fpr(cm):
    return cm['fp'] / (cm['fp'] + cm['tn'])


def tpr(cm):
    return cm['tp'] / (cm['tp'] + cm['fn'])


def fnr(cm):
    return cm['fn'] / (cm['fn'] + cm['tp'])


def line_identity_point(x1, y1, x2, y2, eps=1e-6):
    if abs(x1 - x2) < eps or abs(y1 - y2) < eps:
        return x1 if abs(x1 - x2) < abs(y1 - y2) else y1
    a = (y2 - y1) / (x2 - x1)
    b = y1 - a * x1
    c = b / (1 - a)
    return c


def calc_eer(fpr, fnr, eps=1e-6):
    # fpr is non-increasing, fnr is non-decreasing (check accounting the floating point operations error)
    assert np.all(np.diff(fpr) < eps)
    assert np.all(np.diff(fnr) > -eps)
    diff = fpr - fnr
    i = np.argmin(diff[diff >= 0])
    j = np.argmax(diff[diff < 0])

    eer = line_identity_point(fpr[i], fnr[i], fpr[j], fnr[j], eps) * 100

    eer_min = max(fnr[i], fpr[j])
    eer_max = min(fpr[i], fnr[j])
    return eer, eer_min, eer_max

# if __name__ == "__main__":
    # np.random.seed(0)
    # n_feat = 3
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
    # calc_gen_imp_hist(s1, s2, intervals, batch_size=5)
    # #
    # print("Hello")
