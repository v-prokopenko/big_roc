from pathlib import Path
import numpy as np
import pandas as pd
from collections import namedtuple
import json
import re
import logging

from big_roc import metrics
from big_roc.gen_imp_hist import gen_imp_histogram, convert_to_numpy_style_histogram
from big_roc.utils import uniformly_subsample
from constants import METRICS_PREFIX, ROC_PREFIX, GEN_IMP_DISTRIBUTION_PREFIX, METADATA_FILENAME

AnalysisResults = namedtuple("AnalysisResults", ['metrics', 'bin_edges', 'gen_hist', 'imp_hist', 'fpr', 'fnr'])


def analyze_features(s1: pd.DataFrame, s2: pd.DataFrame, n_bins: int = 10 ** 6) -> AnalysisResults:
    sim_min, sim_max = 0, 1
    eps = 1e-8
    bin_edges = np.linspace(sim_min - eps, sim_max + eps, n_bins + 1)

    gen_hist, imp_hist = gen_imp_histogram(s1, s2, bin_edges)
    gen_hist, imp_hist, bin_edges = convert_to_numpy_style_histogram(gen_hist, imp_hist, bin_edges)

    conf_mat = metrics.roc_metrics.confusion_matrix(gen_hist, imp_hist)
    fpr = metrics.roc_metrics.false_positive_rate(conf_mat)
    fnr = metrics.roc_metrics.false_negative_rate(conf_mat)

    all_metrics = pd.DataFrame([
        metrics.roc_metrics.equal_error_rate(fpr, fnr),
        metrics.roc_metrics.fnr_at_fpr(fpr, fnr, 0.001),
        metrics.roc_metrics.fnr_at_fpr(fpr, fnr, 0.0001),
        metrics.roc_metrics.fnr_at_fpr(fpr, fnr, 0.00001),
        metrics.roc_metrics.fnr_at_fpr(fpr, fnr, 0.000001)
    ])
    all_metrics.set_index("Name", inplace=True)

    tpr = 1 - fnr
    all_metrics.loc["AUC"] = pd.Series({"Value": metrics.auc(fpr, tpr)})
    rank1_ir = metrics.rank1_ir(s1, s2)
    all_metrics.loc["Rank1_IR"] = pd.Series({"Value": rank1_ir})

    gen_metrics, imp_metrics = metrics.genuine_impostor_stats(gen_hist, imp_hist, bin_edges)

    for key, value in gen_metrics._asdict().items():
        all_metrics.loc["Gen" + key] = pd.Series({"Value": value})
    for key, value in imp_metrics._asdict().items():
        all_metrics.loc["Imp" + key] = pd.Series({"Value": value})

    metrics_to_scale_pattern = "|".join(["^EER$", r"^FNR_AT_FPR_.*$", "^Rank1_IR$"])
    metrics_to_scale = [metric_name for metric_name in all_metrics.index
                        if re.match(metrics_to_scale_pattern, metric_name) is not None]
    all_metrics.loc[metrics_to_scale] = all_metrics.loc[metrics_to_scale] * 100

    # hang on there, not the best part of my code
    metrics_to_rename_pattern = "|".join([r"^FNR_AT_FPR_.*$"])
    metrics_to_rename = [metric_name for metric_name in all_metrics.index
                         if re.match(metrics_to_rename_pattern, metric_name) is not None]

    def convert_name_to_percentages_name(name: str):
        name_parts = name.split('_')
        static_part = name_parts[:-1]
        number_str = name_parts[-1]
        new_number_str = f"{float(number_str.replace('p', '.')) * 100:.10f}".replace('.', 'p').rstrip('0')
        return '_'.join(static_part + [new_number_str])

    all_metrics.rename(index={name: convert_name_to_percentages_name(name) for name in metrics_to_rename}, inplace=True)
    results = AnalysisResults(all_metrics, bin_edges, gen_hist, imp_hist, fpr, fnr)
    return results


def save_analysis_results(s1: pd.DataFrame, s2: pd.DataFrame, results: AnalysisResults, output_path: Path,
                          safe_output: bool = True) -> None:
    logging.info(f"Saving results at {output_path}")

    if output_path.exists() and safe_output:
        raise ValueError("Output path already exist")
    if not output_path.exists():
        output_path.mkdir()

    filename_gen_imp = output_path / f"{GEN_IMP_DISTRIBUTION_PREFIX}{output_path.name}.csv"
    n_gen_imp = 1000
    reduce_factor = results.gen_hist.size // n_gen_imp
    df_gen_imp = pd.DataFrame({"BinStart": results.bin_edges[:-1:reduce_factor],
                               "BinEnd": results.bin_edges[reduce_factor::reduce_factor],
                               "Genuine": results.gen_hist.reshape(-1, reduce_factor).sum(axis=1),
                               "Impostor": results.imp_hist.reshape(-1, reduce_factor).sum(axis=1)}
                              )
    df_gen_imp.to_csv(filename_gen_imp, index=False)

    filename_roc = output_path / f"{ROC_PREFIX}{output_path.name}.csv"
    n_roc = 1000
    indices = uniformly_subsample(results.fpr, n_roc)
    df_roc = pd.DataFrame({"Threshold": results.bin_edges[indices],
                           "FPR": results.fpr[indices],
                           "FNR": results.fnr[indices]}
                          )
    df_roc.to_csv(filename_roc, index=False)

    metrics_filename = output_path / f"{METRICS_PREFIX}{output_path.name}.csv"
    results.metrics.to_csv(metrics_filename)

    metadata_filename = output_path / METADATA_FILENAME
    metadata = {"s1_features": list(s1.columns), "s2_features": list(s2.columns),
                "s1_subjects": list(s1.index), "s2_subjects": list(s2.index)}
    with open(str(metadata_filename), 'w') as f:
        json.dump(metadata, f)
