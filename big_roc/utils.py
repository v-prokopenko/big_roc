import numpy as np
import pandas as pd
from pathlib import Path
import re


# assumes array is monotonic and has (positive and negative numbers) or 0
# returns only 1 index if there is a 0
def find_indices_around_0(array: np.ndarray):
    i = np.where(array >= 0, array, np.inf).argmin()
    j = np.where(array <= 0, array, -np.inf).argmax()
    return min(i, j), max(i, j)


def line_identity_point(x1: float, y1: float, x2: float, y2: float, eps: float = 1e-6):
    if abs(x1 - x2) < eps or abs(y1 - y2) < eps:
        return x1 if abs(x1 - x2) < abs(y1 - y2) else y1
    a = (y2 - y1) / (x2 - x1)
    b = y1 - a * x1
    c = b / (1 - a)
    return c


def read_dataset(path: Path, session_column: str = "Sess", subject_column: str = "Sub",
                 feature_name_pattern: str = r"^f\d+$"):
    df = pd.read_csv(path)
    useful_columns = [name for name in df.columns if
                      name in [session_column, subject_column] or re.match(feature_name_pattern, name)]
    df = df[useful_columns]

    s1 = df.loc[df[session_column] == 1].copy()
    s2 = df.loc[df[session_column] == 2].copy()

    s1.drop(session_column, axis=1, inplace=True)
    s2.drop(session_column, axis=1, inplace=True)
    s1.set_index(subject_column, inplace=True)
    s2.set_index(subject_column, inplace=True)
    return s1, s2


class HistogramSampler:
    def __init__(self, hist: np.ndarray, bin_edges: np.ndarray):
        assert hist.size + 1 == bin_edges.size
        self.hist = np.copy(hist)
        self.bin_edges = np.copy(bin_edges)
        self.low2high = {low: high for low, high in zip(self.bin_edges[:-1], self.bin_edges[1:])}

    def randomize_interval(self, x: float):
        return np.random.uniform(x, self.low2high[x])

    def probabilistic_sample(self, sample_size: int):
        samples = np.random.choice(self.bin_edges[:-1], p=self.hist / self.hist.sum(), size=sample_size)
        samples = np.vectorize(self.randomize_interval)(samples)
        return samples

    def naive_sample(self):
        assert self.hist.sum() < 1e9
        middle_points = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2
        return np.repeat(middle_points, self.hist)


def uniformly_subsample(x: np.ndarray, subsample_size: int) -> np.ndarray:
    indices = []
    values_to_search = np.linspace(x.min(), x.max(), subsample_size)
    for value in values_to_search:
        indices.append(np.argmin(np.abs(x - value)))
    return np.array(indices)
