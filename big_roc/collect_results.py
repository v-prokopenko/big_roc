import re
from pathlib import Path
from collections import namedtuple
import pandas as pd

from constants import METRICS_PREFIX
import logging

RunConfig = namedtuple("RunConfig", ["DatasetName", "NSubjects", "NFeatures", "Repetition"])


def collect_results(dir_path: Path, out_filename: str):
    logging.info("Collecting results")
    all_metrics = []
    for run_dir in dir_path.iterdir():
        if not run_dir.is_dir() or run_dir.name.startswith('.'):
            continue
        run_config = parse_dir_name(run_dir.name)
        logging.info(run_config)
        metrics = pd.read_csv(run_dir / f"{METRICS_PREFIX}{run_dir.name}.csv", index_col="Name")
        values_dict = metrics['Value'].to_dict()
        errors_dict = {key + "_ERROR": value for key, value in metrics['Error'].to_dict().items() if value is not None}
        min_dict = {key + "_MIN": value for key, value in metrics['ValueMin'].to_dict().items() if value is not None}
        max_dict = {key + "_MAX": value for key, value in metrics['ValueMax'].to_dict().items() if value is not None}
        all_metrics.append(pd.DataFrame([{**run_config._asdict(), **values_dict,
                                          **errors_dict, **min_dict, **max_dict}]))
    all_metrics = pd.concat(all_metrics)
    all_metrics.dropna(axis=1, how='all', inplace=True)
    all_metrics.to_csv(dir_path / out_filename, index=False)
    return all_metrics


def parse_dir_name(dir_name: str) -> RunConfig:
    name_parts = dir_name.split("_")
    dataset_name = "_".join(name_parts[:-3])
    n_subj = int(re.findall(r'\d+$', name_parts[-3])[0])
    n_feat = int(re.findall(r'\d+$', name_parts[-2])[0])
    repetition = int(re.findall(r'\d+$', name_parts[-1])[0])
    return RunConfig(dataset_name, n_subj, n_feat, repetition)