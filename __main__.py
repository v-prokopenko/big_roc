import click
from pathlib import Path
from glob import glob
import pandas as pd
import json
from typing import List, Dict
import shutil
import logging
from big_roc.utils import read_dataset, subsample_sessions, setup_logging, construct_dir_name
from big_roc.analysis import analyze_features, save_analysis_results
from big_roc.collect_results import collect_results, RunConfig, parse_dir_name
from constants import *


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


def cli_collect_results(ctx, param, dir_path: Path):
    if not dir_path:
        return
    collect_results(dir_path, COLLECTED_METRICS_FILENAME)
    ctx.exit()


def run_from_config(config: Dict):
    config["data_files"] = expand_unix_path(config["data_files"])
    config["output_dir"] = Path(config["output_dir"])
    logging.info(config)

    if not config["output_dir"].exists():
        config["output_dir"].mkdir()

    for data_file in config["data_files"]:
        s1, s2 = read_dataset(data_file)
        for n_subj in config["n_subjects"]:
            for n_feat in config["n_features"]:
                repetition = config.get("repetition_start", 1)
                for i in range(config["n_repetitions"]):
                    logging.info(f"Running analysis for: "
                                 f"data_file={data_file}, n_subj={n_subj}, n_feat={n_feat}, repetition_index={i}"
                                 )
                    s1_part, s2_part = subsample_sessions(s1, s2, n_subj, n_feat)
                    results = analyze_features(s1_part, s2_part)

                    output_path = config["output_dir"] / construct_dir_name(
                        data_file.stem, n_subj, n_feat, repetition
                    )
                    while output_path.exists():
                        repetition += 1
                        output_path = config["output_dir"] / construct_dir_name(
                            data_file.stem, n_subj, n_feat, repetition
                        )

                    save_analysis_results(s1_part, s2_part, results, output_path)

    collect_results(config["output_dir"], COLLECTED_METRICS_FILENAME)


def cli_use_config_file(ctx, param, config_file):
    if not config_file:
        return
    config = json.load(config_file)
    config_file.close()
    if isinstance(config, list):
        for c in config:
            run_from_config(c)
    else:
        run_from_config(config)
    # config["data_files"] = expand_unix_path(config["data_files"])
    # config["output_dir"] = Path(config["output_dir"])
    # logging.info(config)
    #
    # if not config["output_dir"].exists():
    #     config["output_dir"].mkdir()
    #
    # for data_file in config["data_files"]:
    #     s1, s2 = read_dataset(data_file)
    #     for n_subj in config["n_subjects"]:
    #         for n_feat in config["n_features"]:
    #             repetition = config.get("repetition_start", 1)
    #             for i in range(config["n_repetitions"]):
    #                 logging.info(f"Running analysis for: "
    #                              f"data_file={data_file}, n_subj={n_subj}, n_feat={n_feat}, repetition_index={i}"
    #                              )
    #                 s1_part, s2_part = subsample_sessions(s1, s2, n_subj, n_feat)
    #                 results = analyze_features(s1_part, s2_part)
    #
    #                 output_path = config["output_dir"] / construct_dir_name(
    #                     data_file.stem, n_subj, n_feat, repetition
    #                 )
    #                 while output_path.exists():
    #                     repetition += 1
    #                     output_path = config["output_dir"] / construct_dir_name(
    #                         data_file.stem, n_subj, n_feat, repetition
    #                     )
    #
    #                 save_analysis_results(s1_part, s2_part, results, output_path)

    ctx.exit()


def merge_results(dir_paths: List[Path], out_path: Path):
    out_path.mkdir()
    for dir_path in dir_paths:
        for run_path in sorted(dir_path.iterdir()):
            if not run_path.is_dir():
                continue
            run_config = parse_dir_name(run_path.name)

            # find non-existing repetition
            repetition = 1
            while True:
                new_run_config = RunConfig(*run_config[:-1], repetition)
                new_run_path = out_path / construct_dir_name(*new_run_config)
                if not new_run_path.exists():
                    new_run_path.mkdir()
                    break
                repetition += 1

            gen_imp_filepath = run_path / f"{GEN_IMP_DISTRIBUTION_PREFIX}{run_path.name}.csv"
            new_gen_imp_filepath = new_run_path / f"{GEN_IMP_DISTRIBUTION_PREFIX}{new_run_path.name}.csv"
            shutil.copy(str(gen_imp_filepath), str(new_gen_imp_filepath))

            roc_filepath = run_path / f"{ROC_PREFIX}{run_path.name}.csv"
            new_roc_filepath = new_run_path / f"{ROC_PREFIX}{new_run_path.name}.csv"
            shutil.copy(str(roc_filepath), str(new_roc_filepath))

            metrics_filepath = run_path / f"{METRICS_PREFIX}{run_path.name}.csv"
            new_metrics_filepath = new_run_path / f"{METRICS_PREFIX}{new_run_path.name}.csv"
            shutil.copy(str(metrics_filepath), str(new_metrics_filepath))

            metadata_filepath = run_path / METADATA_FILENAME
            new_metadata_filepath = new_run_path / METADATA_FILENAME
            shutil.copy(str(metadata_filepath), str(new_metadata_filepath))

    collect_results(out_path, COLLECTED_METRICS_FILENAME)


def cli_expand_unix_path(ctx, param, path_list: List[str]) -> List[Path]:
    return expand_unix_path(path_list)


@click.command()
@click.option("--config-file", type=click.File('rb'), callback=cli_use_config_file,
              expose_value=False, is_eager=True,
              help="Specify the .json config file that specifies the analysis to run.")
@click.option("--collect", type=Path, callback=cli_collect_results, expose_value=False, is_eager=True,
              help="Collect the results produced using the .json config file")
@click.argument("data_files", nargs=-1, callback=cli_expand_unix_path)
@click.argument("output_dir", type=Path)
@click.option("--merge", is_flag=True,
              help="When you use this flag 'data_files' are treated as directories with analysis results. "
                   "The program will merge these results into 'output_dir'. "
                   "Directories will be processed in the order you entered them in. "
                   "Repetition index will be reset to 1. "
                   "Note: output dir should NOT exist."
              )
def run_analysis(data_files: List[Path], output_dir: Path, merge: bool):
    print(f"data_files: {data_files}, output_dir: {output_dir}")
    if merge:
        data_files = list(set([data_file.resolve() for data_file in data_files]))
        merge_results(data_files, output_dir)
    else:
        for data_file in data_files:
            logging.info(f"Running analysis for: data_file={data_file}")
            s1, s2 = read_dataset(data_file)
            results = analyze_features(s1, s2)
            save_analysis_results(s1, s2, results, output_dir / data_file.name)


if __name__ == '__main__':
    setup_logging()
    run_analysis()
