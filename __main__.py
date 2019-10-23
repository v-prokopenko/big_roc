import click
from pathlib import Path
from glob import glob
import re
import pandas as pd
import json
from typing import List
from big_roc.utils import read_dataset, subsample_sessions
from big_roc.analysis import analyze_features, save_analysis_results
from big_roc.collect_results import collect_results
from constants import COLLECTED_METRICS_FILENAME


def expand_unix_path(path_list: List[str]) -> List[Path]:
    expanded_path_list = []
    for path in path_list:
        expanded_path = glob(str(Path(path)))
        if len(expanded_path) == 0:
            click.echo(f"Path '{path}' doesn't exist")
        expanded_path_list.extend(expanded_path)
    return [Path(path) for path in expanded_path_list]


def append_to_global_metrics(metrics: pd.Series, band: int, n_subj: int, n_feat: int, repetition: int,
                             filepath: Path) -> None:
    new_row = pd.DataFrame([{"Band": band, "NSubjects": n_subj, "NFeatures": n_feat, "Repetition": repetition,
                             **metrics.to_dict()}])
    if filepath.stat().st_size > 0:
        out_df: pd.DataFrame = pd.read_csv(filepath)
        out_df = out_df.append(new_row, ignore_index=True)
    else:
        out_df = new_row
    out_df.to_csv(filepath, index=False)


def construct_directory_name(dataset_name: str, n_subj: int, n_feat: int, repetition: int):
    return f"{dataset_name}_NSub{n_subj:06}_NFeat{n_feat:03}_Repeat{repetition:03}"


def cli_collect_results(ctx, param, dir_path: Path):
    collect_results(dir_path, COLLECTED_METRICS_FILENAME)
    ctx.exit()


def cli_use_config_file(ctx, param, config_file):
    config = json.load(config_file)
    config_file.close()
    config["data_files"] = expand_unix_path(config["data_files"])
    config["output_dir"] = Path(config["output_dir"])
    print(config)
    if not config["output_dir"].exists():
        config["output_dir"].mkdir()

    for data_file in config["data_files"]:
        s1, s2 = read_dataset(data_file)
        for n_subj in config["n_subjects"]:
            for n_feat in config["n_features"]:
                repetition = config.get("repetition_start", 1)
                for i in range(config["n_repetitions"]):
                    s1_part, s2_part = subsample_sessions(s1, s2, n_subj, n_feat)
                    results = analyze_features(s1_part, s2_part)

                    output_path = config["output_dir"] / construct_directory_name(data_file.stem,
                                                                                  n_subj, n_feat, repetition)
                    while output_path.exists():
                        repetition += 1
                        output_path = config["output_dir"] / construct_directory_name(data_file.stem,
                                                                                      n_subj, n_feat, repetition)

                    save_analysis_results(s1_part, s2_part, results, output_path)
        collect_results(config["output_dir"], COLLECTED_METRICS_FILENAME)
        ctx.exit()


@click.command()
@click.option("--collect", type=Path, callback=cli_collect_results, expose_value=False, is_eager=True)
@click.option("--use-config-file", type=click.File('rb'), callback=cli_use_config_file,
              expose_value=False, is_eager=True)
@click.argument("data_files", nargs=-1)
@click.argument("output_dir", type=Path)
def run_analysis(data_files: List[str], output_dir: Path):
    data_files = expand_unix_path(data_files)
    for data_file in data_files:
        s1, s2 = read_dataset(data_file)
        results = analyze_features(s1, s2)
        save_analysis_results(s1, s2, results, output_dir / data_file.name)


if __name__ == '__main__':
    run_analysis()
