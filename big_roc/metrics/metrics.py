import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

__all__ = ["rank1_ir", "cosine_similarity_01"]


def cosine_similarity_01(x, y):
    return (cosine_similarity(x, y) + 1) / 2


def rank1_ir(s1: pd.DataFrame, s2: pd.DataFrame, similarity_measure=cosine_similarity_01) -> float:
    correctly_identified = 0.
    for subj_id, row in s1.iterrows():
        sims = similarity_measure(row.values.reshape(1, -1), s2).flatten()
        ind_max = np.argwhere(sims == np.amax(sims)).flatten()
        if subj_id in s1.index[ind_max]:
            correctly_identified += 1 / len(ind_max)
    return correctly_identified / s1.shape[0]
