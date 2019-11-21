from collections import namedtuple

import numpy as np
import pandas as pd
from sklearn.metrics import auc

from big_roc.utils import find_indices_around_0, line_identity_point

__all__ = ["auc", "MetricEstimate", "confusion_matrix", "false_negative_rate", "false_positive_rate",
           "true_positive_rate", "equal_error_rate", "fnr_at_fpr"]

MetricEstimate = namedtuple("MetricEstimate", ["Name", "Value", "ValueMin", "ValueMax", "Error"])


def confusion_matrix(gen_hist: np.ndarray, imp_hist: np.ndarray) -> pd.DataFrame:
    # cumulative sum with added first element 0 (to have a confusion matrix where everything is classified as true)
    cum_gen, cum_imp = np.insert(np.cumsum(gen_hist), 0, [0]), np.insert(np.cumsum(imp_hist), 0, [0])
    sum_gen, sum_imp = np.sum(gen_hist), np.sum(imp_hist)
    return pd.DataFrame({'tp': sum_gen - cum_gen, 'fn': cum_gen, 'fp': sum_imp - cum_imp, 'tn': cum_imp})


def false_positive_rate(cm):
    return (cm['fp'] / (cm['fp'] + cm['tn'])).values


def true_positive_rate(cm):
    return (cm['tp'] / (cm['tp'] + cm['fn'])).values


def false_negative_rate(cm):
    return (cm['fn'] / (cm['fn'] + cm['tp'])).values


def equal_error_rate(fpr: np.ndarray, fnr: np.ndarray, eps: float = 1e-6) -> MetricEstimate:
    # fpr is non-increasing, fnr is non-decreasing (check accounting the floating point operations error)
    assert np.all(np.diff(fpr) < eps)
    assert np.all(np.diff(fnr) > -eps)

    diff = fpr - fnr
    i1, i2 = find_indices_around_0(diff)

    eer = line_identity_point(fpr[i1], fnr[i1], fpr[i2], fnr[i2], eps)

    eer_min = max(min(fnr[i1], fpr[i1]), min(fnr[i2], fpr[i2]))
    eer_max = min(max(fnr[i1], fpr[i1]), max(fnr[i2], fpr[i2]))
    error = max(eer - eer_min, eer_max - eer)

    return MetricEstimate("EER", eer, eer_min, eer_max, error)


def fnr_at_fpr(fnr: np.ndarray, fpr: np.ndarray, fpr_value: float, eps: float = 1e-6) -> MetricEstimate:
    assert min(fpr) < fpr_value < max(fpr)

    shifted_fpr = fpr - fpr_value
    i1, i2 = find_indices_around_0(shifted_fpr)

    _ = abs(shifted_fpr[i1]) + abs(shifted_fpr[i2])
    k = abs(shifted_fpr[i1] / _) if _ > eps else 0.5

    fnr_value = fnr[i1] * k + fnr[i2] * (1 - k)
    fnr_min = min(fnr[i1], fnr[i2])
    fnr_max = max(fnr[i1], fnr[i2])
    error = max(fnr_value - fnr_min, fnr_max - fnr_value)

    # Get number string in non-scientific mode, leave only a decimal part if it starts with "0."
    name = f"FNR_AT_FPR_{f'{fpr_value:.10f}'.replace('.', 'p').rstrip('0')}"
    return MetricEstimate(name, fnr_value, fnr_min, fnr_max, error)
