import datetime
import logging

import numpy as np
import pandas as pd
from pathlib import Path
import re
import random



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


def read_dataset(path: Path, session_column: str = "Session", subject_column: str = "Subject",
                 feature_name_pattern: str = r"^F\d+$"):
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


def subsample_sessions(session1, session2, n_subj, n_feat, copy=False):
    random.seed()
    if n_subj < len(session1.index):
        rows = random.sample(list(session1.index), n_subj)
    else:
        rows = list(session1.index)
    if n_feat < len(session1.columns):
        columns = random.sample(list(session1.columns), n_feat)
    else:
        columns = list(session1.columns)
    sub_session1, sub_session2 = session1.loc[rows, columns], session2.loc[rows, columns]

    if copy:
        return sub_session1.copy(), sub_session2.copy()
    else:
        return sub_session1, sub_session2


def setup_logging(level=logging.INFO):
    fmt = '%(asctime)s %(levelname)-8s %(message)s'
    date_fmt = '%Y-%m-%d %H:%M:%S'
    logging.basicConfig(format=fmt, datefmt=date_fmt, level=level)

    log_dir = Path('logs')
    if log_dir.exists():
        filename = str(log_dir / datetime.datetime.now().strftime('logfile_' + __name__ + '_%H_%M_%d_%m_%Y.log'))
        file_handler = logging.FileHandler(filename, mode='w')
        file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=date_fmt))
        logging.getLogger().addHandler(file_handler)


def construct_dir_name(dataset_name: str, n_subj: int, n_feat: int, repetition: int):
    return f"{dataset_name}_NSub{n_subj:06}_NFeat{n_feat:03}_Repeat{repetition:03}"
