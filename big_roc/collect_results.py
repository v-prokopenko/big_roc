from pathlib import Path
from collections import namedtuple
import pandas as pd
import re
from constants import METRICS_PREFIX

RunConfig = namedtuple("RunConfig", ["DatasetName", "NSubjects", "NFeatures", "Repetition"])


def parse_dir_name(dir_name: str) -> RunConfig:
    name_parts = dir_name.split("_")
    print(name_parts)
    dataset_name = "_".join(name_parts[:-3])
    n_subj = int(re.findall(r'\d+$', name_parts[-3])[0])
    n_feat = int(re.findall(r'\d+$', name_parts[-2])[0])
    repetition = int(re.findall(r'\d+$', name_parts[-1])[0])
    return RunConfig(dataset_name, n_subj, n_feat, repetition)


def collect_results(dir_path: Path, out_filename: str):
    print("Collecting results")
    all_metrics = []
    for run_dir in dir_path.iterdir():
        if not run_dir.is_dir() or run_dir.name.startswith('.'):
            continue
        run_config = parse_dir_name(run_dir.name)
        metrics = pd.read_csv(run_dir / f"{METRICS_PREFIX}{run_dir.name}.csv", index_col="Name")
        all_metrics.append(pd.DataFrame([{**run_config._asdict(), **metrics['Value'].to_dict()}]))
    all_metrics = pd.concat(all_metrics)
    all_metrics.to_csv(dir_path / out_filename, index=False)
    return all_metrics
