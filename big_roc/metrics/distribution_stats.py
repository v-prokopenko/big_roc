from collections import namedtuple
from typing import Tuple

import numpy as np
from scipy import stats

from big_roc.utils import HistogramSampler

__all__ = ["distribution_stats", "genuine_impostor_stats"]

DistributionStats = namedtuple("DistributionStats", ["Mdn", "Iqr", "Skew", "Kurt"])


def distribution_stats(sample: np.ndarray) -> DistributionStats:
    return DistributionStats(np.median(sample),
                             stats.iqr(sample),
                             stats.skew(sample),
                             stats.kurtosis(sample, fisher=False)
                             )


def genuine_impostor_stats(gen_hist: np.ndarray, imp_hist: np.ndarray, bin_edges: np.ndarray, sample_size: int = 10**6)\
        -> Tuple[DistributionStats, DistributionStats]:
    assert bin_edges.size == gen_hist.size + 1 == imp_hist.size + 1

    gen_hist_sampler = HistogramSampler(gen_hist, bin_edges)
    imp_hist_sampler = HistogramSampler(imp_hist, bin_edges)

    gen_stats = distribution_stats(gen_hist_sampler.probabilistic_sample(sample_size))
    imp_stats = distribution_stats(imp_hist_sampler.probabilistic_sample(sample_size))

    return gen_stats, imp_stats
