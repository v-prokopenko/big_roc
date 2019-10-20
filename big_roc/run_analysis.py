from pathlib import Path
import numpy as np
import pandas as pd

from big_roc import metrics
import big_roc


# cut the far bins that should be 0 if sim_min and sim_max picked properly
from big_roc.gen_imp_hist import convert_to_numpy_style_histogram
from big_roc.utils import uniformly_subsample


def run_analysis(s1: pd.DataFrame, s2: pd.DataFrame, output_path: Path,
                 n_bins: int = 10 ** 6, safe_output: bool = True):
    if output_path.exists() and safe_output:
        raise ValueError("Output path already exist")

    sim_min, sim_max = 0, 1
    eps = 1e-8
    bin_edges = np.linspace(sim_min - eps, sim_max + eps, n_bins + 1)

    gen_hist, imp_hist = big_roc.gen_imp_histogram(s1, s2, bin_edges)
    gen_hist, imp_hist, bin_edges = convert_to_numpy_style_histogram(gen_hist, imp_hist, bin_edges)

    conf_mat = big_roc.metrics.roc_metrics.confusion_matrix(gen_hist, imp_hist)
    fpr = big_roc.metrics.roc_metrics.false_positive_rate(conf_mat)
    fnr = big_roc.metrics.roc_metrics.false_negative_rate(conf_mat)

    all_metrics = pd.DataFrame([
        big_roc.metrics.roc_metrics.equal_error_rate(fpr, fnr),
        big_roc.metrics.roc_metrics.fnr_at_fpr(fpr, fnr, 0.001),
        big_roc.metrics.roc_metrics.fnr_at_fpr(fpr, fnr, 0.0001),
        big_roc.metrics.roc_metrics.fnr_at_fpr(fpr, fnr, 0.00001),
        big_roc.metrics.roc_metrics.fnr_at_fpr(fpr, fnr, 0.000001)
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

    if not output_path.exists():
        output_path.mkdir()

    # filename1 = output_path / "gen_imp_hist_roc.csv"
    # df1 = pd.DataFrame({"bin_start": bin_edges[:-1], "bin_end": bin_edges[1:],
    #                     "gen_hist": gen_hist, "imp_hist": imp_hist, "fpr": fpr, "fnr": fnr})
    # df1.to_csv(filename1, index=False)

    filename_gen_imp = output_path / f"Gen_Imp_{output_path.name}.csv"
    n_gen_imp = 1000
    reduce_factor = n_bins // n_gen_imp
    df_gen_imp = pd.DataFrame({"BinStart": bin_edges[:-1:reduce_factor],
                               "BinEnd": bin_edges[reduce_factor::reduce_factor],
                               "Genuine": gen_hist.reshape(-1, reduce_factor).sum(axis=1),
                               "Impostor": imp_hist.reshape(-1, reduce_factor).sum(axis=1)}
                              )
    df_gen_imp.to_csv(filename_gen_imp, index=False)

    filename_roc = output_path / f"ROC_{output_path.name}.csv"
    n_roc = 1000
    indices = uniformly_subsample(fpr, n_roc)
    df_roc = pd.DataFrame({"Threshold": bin_edges[indices],
                           "FPR": fpr[indices],
                           "FNR": fnr[indices]}
                          )
    df_roc.to_csv(filename_roc, index=False)

    # tmp1 = output_path / "gen_stats.csv"
    # _df1 = pd.Series(gen_metrics._asdict())
    # _df1.to_csv(tmp1)
    # tmp2 = output_path / "imp_stats.csv"
    # _df2 = pd.Series(imp_metrics._asdict())
    # _df2.to_csv(tmp2)

    metrics_filename = output_path / f"Metrics_{output_path.name}.csv"
    df2 = pd.DataFrame(all_metrics)
    df2.to_csv(metrics_filename)

    assert bin_edges.size == gen_hist.size + 1 == imp_hist.size + 1 == fpr.size == fnr.size
    return df2, gen_metrics, imp_metrics
