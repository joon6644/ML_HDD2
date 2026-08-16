"""Build the paper's result tables from the experiment DB.

Body tables
  Table 1  Combined results: row-unit @ tau_row | disk-unit @ tau_row | disk-unit @ tau_op
           Reading left to right IS the argument: block 1->2 changes only the evaluation
           unit (RQ1); block 2->3 also changes the threshold criterion (RQ2).
  Table 2  Model-selection agreement, with the seed-only baseline (5.4, exploratory)

Appendix
  Table A1 Threshold selection evidence (validation FAR, constraint satisfied)
  Table A2 Five-outcome raw counts

Aggregation: median across seeds within each (dataset, model) cell. Seeds share the
same evaluation HDDs within a dataset, so they are re-randomizations rather than
independent replications; no test is run on the 360 runs directly. The only
inferential test is a paired Wilcoxon over the 12 cell medians, reported for RQ2.
"""
import argparse
import itertools
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "experiments"))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

from evaluator import read_results_table, get_db_path

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
OUT_DIR = os.path.join(RESULTS_DIR, "paper_tables")
DB = get_db_path(RESULTS_DIR)

DATASET_ORDER = ["HGST_20HUH721212ALN604", "ST12000NM0007", "TOSHIBA_20MG07ACA14TA"]
DATASET_SHORT = {"HGST_20HUH721212ALN604": "HGST", "ST12000NM0007": "Seagate",
                 "TOSHIBA_20MG07ACA14TA": "Toshiba"}
MODEL_ORDER = ["LGBM", "XGB", "LSTM", "GRU"]


def load():
    runs = read_results_table(DB, "runs")
    row = read_results_table(DB, "row_level")
    disk = read_results_table(DB, "disk_level")
    if runs.empty:
        raise RuntimeError(f"No runs recorded in {DB}.")
    return runs, row[row.split == "test"].copy(), disk[disk.split == "test"].copy()


def order(df):
    """Sort by dataset then model. Tables carry the short dataset labels."""
    df = df.copy()
    df["dataset"] = pd.Categorical(
        df["dataset"], [DATASET_SHORT[d] for d in DATASET_ORDER], ordered=True)
    df["model"] = pd.Categorical(df["model"], MODEL_ORDER, ordered=True)
    return df.sort_values(["dataset", "model"]).reset_index(drop=True)


def med_iqr(s, pct=False, digits=3):
    m, q1, q3 = s.median(), s.quantile(.25), s.quantile(.75)
    scale = 100 if pct else 1
    f = f"{{:.{digits}f}}"
    return f"{f.format(m*scale)} ({f.format(q1*scale)}-{f.format(q3*scale)})"


# ------------------------------------------------------------------------------
# Table 1
# ------------------------------------------------------------------------------
def table1(row, disk):
    r_at_row = row.query("threshold_kind == 'row_opt'")
    d_at_row = disk.query("threshold_kind == 'row_opt'")
    d_at_op = disk.query("threshold_kind == 'disk_opt'")

    recs, recs_raw = [], []
    for (ds, mo), g in d_at_op.groupby(["dataset", "model"], observed=True):
        rr = r_at_row[(r_at_row.dataset == ds) & (r_at_row.model == mo)]
        dr = d_at_row[(d_at_row.dataset == ds) & (d_at_row.model == mo)]
        recs.append({
            "dataset": DATASET_SHORT[ds], "model": mo, "n_seed": len(g),
            "row_Prec": med_iqr(rr.precision), "row_Rec": med_iqr(rr.recall),
            "row_FAR%": med_iqr(rr.far, pct=True, digits=2),
            "disk@row_Prec": med_iqr(dr.precision), "disk@row_Rec": med_iqr(dr.recall),
            "disk@row_FAR%": med_iqr(dr.far, pct=True, digits=2),
            "disk@row_MedLT": med_iqr(dr.median_lead_time, digits=1),
            "disk@op_Prec": med_iqr(g.precision), "disk@op_Rec": med_iqr(g.recall),
            "disk@op_FAR%": med_iqr(g.far, pct=True, digits=2),
            "disk@op_MedLT": med_iqr(g.median_lead_time, digits=1),
        })
        recs_raw.append({
            "dataset": DATASET_SHORT[ds], "model": mo,
            "row_Prec": rr.precision.median(), "row_Rec": rr.recall.median(),
            "row_FAR": rr.far.median(),
            "disk_row_Prec": dr.precision.median(), "disk_row_Rec": dr.recall.median(),
            "disk_row_FAR": dr.far.median(), "disk_row_MedLT": dr.median_lead_time.median(),
            "disk_op_Prec": g.precision.median(), "disk_op_Rec": g.recall.median(),
            "disk_op_FAR": g.far.median(), "disk_op_MedLT": g.median_lead_time.median(),
        })
    return order(pd.DataFrame(recs)), order(pd.DataFrame(recs_raw))


def direction_consistency(row, disk):
    """Per-run direction checks -- the claim that needs no independence assumption."""
    key = ["dataset", "model", "seed"]
    r = row.query("threshold_kind == 'row_opt'").set_index(key)
    dr = disk.query("threshold_kind == 'row_opt'").set_index(key)
    do = disk.query("threshold_kind == 'disk_opt'").set_index(key)
    idx = r.index.intersection(dr.index).intersection(do.index)
    r, dr, do = r.loc[idx], dr.loc[idx], do.loc[idx]

    checks = {
        "RQ1  disk FAR > row FAR  (@tau_row)": dr.far > r.far,
        "RQ2  disk FAR falls      (tau_op < tau_row)": do.far < dr.far,
        "RQ2  disk Precision rises": do.precision > dr.precision,
        "RQ2  Median LT falls": do.median_lead_time < dr.median_lead_time,
    }
    out = []
    for label, ok in checks.items():
        per_cell = ok.groupby(level=["dataset", "model"], observed=True).mean()
        out.append({
            "check": label,
            "runs": f"{int(ok.sum())}/{len(ok)}",
            "cells_all_consistent": f"{int((per_cell == 1).sum())}/{len(per_cell)}",
            "worst_cell": "-" if (per_cell == 1).all()
            else f"{per_cell.idxmin()} {per_cell.min():.0%}",
        })
    return pd.DataFrame(out)


def paired_summary(t1_raw):
    """Wilcoxon over the 12 cell medians -- the only inferential test reported."""
    rows = []
    for label, a, b, pct in (
        ("Disk Precision", "disk_row_Prec", "disk_op_Prec", False),
        ("Disk Recall", "disk_row_Rec", "disk_op_Rec", False),
        ("Disk FAR (%)", "disk_row_FAR", "disk_op_FAR", True),
        ("Median Lead Time (d)", "disk_row_MedLT", "disk_op_MedLT", False),
    ):
        x, y = t1_raw[a].values, t1_raw[b].values
        scale = 100 if pct else 1
        diff = (y - x) * scale
        ratio = np.median(y / np.where(x == 0, np.nan, x))
        stat, p = wilcoxon(x, y)
        rows.append({
            "metric": label,
            "tau_row (median of cells)": f"{np.median(x)*scale:.3f}",
            "tau_op  (median of cells)": f"{np.median(y)*scale:.3f}",
            "median diff": f"{np.median(diff):+.3f}",
            "ratio (op/row)": f"{ratio:.2f}x",
            "cells improved": f"{int((diff > 0).sum() if label != 'Disk FAR (%)' and label != 'Median Lead Time (d)' else (diff < 0).sum())}/12",
            "wilcoxon p (n=12)": f"{p:.4f}",
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------
# Table 2 -- model-selection agreement (exploratory)
# ------------------------------------------------------------------------------
def _ranks(df, value_col):
    """(dataset, seed) -> rank vector over models, higher value = rank 1."""
    out = {}
    for (ds, seed), g in df.groupby(["dataset", "seed"], observed=True):
        g = g.set_index("model").reindex(MODEL_ORDER)
        if g[value_col].isna().any():
            continue
        out[(ds, seed)] = (-g[value_col].values).argsort().argsort() + 1
    return out


def _top1(rank_vec):
    return MODEL_ORDER[int(np.argmin(rank_vec))]


def table2(row, disk):
    # Each paradigm evaluated as it would be done on its own terms.
    row_para = _ranks(row.query("threshold_kind == 'row_opt'"), "recall")
    op_para = _ranks(disk.query("threshold_kind == 'disk_opt'"), "recall")
    # Secondary: isolate the criterion only (unit held at disk level).
    disk_at_row = _ranks(disk.query("threshold_kind == 'row_opt'"), "recall")

    recs = []
    for ds in DATASET_ORDER:
        seeds = sorted({s for (d, s) in row_para if d == ds} & {s for (d, s) in op_para if d == ds})

        cross_top1 = [_top1(row_para[(ds, s)]) == _top1(op_para[(ds, s)]) for s in seeds]
        cross_rho = [spearmanr(row_para[(ds, s)], op_para[(ds, s)]).correlation for s in seeds]

        crit_top1 = [_top1(disk_at_row[(ds, s)]) == _top1(op_para[(ds, s)]) for s in seeds]
        crit_rho = [spearmanr(disk_at_row[(ds, s)], op_para[(ds, s)]).correlation for s in seeds]

        # Seed-only baseline: same paradigm, different seeds.
        base_top1, base_rho = [], []
        for a, b in itertools.combinations(seeds, 2):
            base_top1.append(_top1(op_para[(ds, a)]) == _top1(op_para[(ds, b)]))
            base_rho.append(spearmanr(op_para[(ds, a)], op_para[(ds, b)]).correlation)

        recs.append({
            "dataset": DATASET_SHORT[ds], "n_seed": len(seeds),
            "Top-1 agree: paradigms": f"{np.mean(cross_top1):.0%}",
            "Top-1 agree: criterion only": f"{np.mean(crit_top1):.0%}",
            "Top-1 agree: seed baseline": f"{np.mean(base_top1):.0%}",
            "Spearman: paradigms": f"{np.nanmean(cross_rho):+.3f}",
            "Spearman: criterion only": f"{np.nanmean(crit_rho):+.3f}",
            "Spearman: seed baseline": f"{np.nanmean(base_rho):+.3f}",
        })
    return pd.DataFrame(recs)


# ------------------------------------------------------------------------------
# Appendix
# ------------------------------------------------------------------------------
def table_a1(runs):
    recs = []
    for (ds, mo), g in runs.groupby(["dataset", "model"], observed=True):
        recs.append({
            "dataset": DATASET_SHORT[ds], "model": mo,
            "tau_row": med_iqr(g.threshold_row_opt),
            "tau_op": med_iqr(g.threshold_disk_opt),
            "val row FAR %": med_iqr(g.val_far_row_opt, pct=True, digits=2),
            "val disk FAR %": med_iqr(g.val_far_disk_opt, pct=True, digits=2),
            "row constraint met": f"{int(g.row_constraint_met.sum())}/{len(g)}",
            "disk constraint met": f"{int(g.disk_constraint_met.sum())}/{len(g)}",
        })
    return order(pd.DataFrame(recs))


def table_a2(disk):
    recs = []
    for (ds, mo, kind), g in disk.groupby(["dataset", "model", "threshold_kind"], observed=True):
        recs.append({
            "dataset": DATASET_SHORT[ds], "model": mo, "threshold": kind,
            "O": f"{g.N_on_time.median():.0f}", "E": f"{g.N_early.median():.0f}",
            "M": f"{g.N_missed.median():.0f}", "CE": f"{g.N_censored_early.median():.0f}",
            "CN": f"{g.N_censored_no_alarm.median():.0f}",
            "N_failed": f"{g.N_failed.median():.0f}", "N_censored": f"{g.N_censored.median():.0f}",
        })
    return order(pd.DataFrame(recs))


def main():
    ap = argparse.ArgumentParser(description="Build the paper's result tables.")
    ap.add_argument("--out", default=OUT_DIR)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    runs, row, disk = load()
    pd.set_option("display.width", 250, "display.max_columns", 50)

    t1, t1_raw = table1(row, disk)
    dc = direction_consistency(row, disk)
    ps = paired_summary(t1_raw)
    t2 = table2(row, disk)
    a1, a2 = table_a1(runs), table_a2(disk)

    seeds = sorted(runs.seed.unique())
    print("=" * 120)
    print(f"runs={len(runs)}  datasets={runs.dataset.nunique()}  models={runs.model.nunique()}  "
          f"seeds={len(seeds)} ({seeds[0]}-{seeds[-1]})")
    print("values are median (Q1-Q3) across seeds within each dataset-model cell")

    for name, df in (("TABLE 1  combined results (test split)", t1),
                     ("direction consistency (per run; no independence assumed)", dc),
                     ("paired summary over the 12 cell medians  [RQ2]", ps),
                     ("TABLE 2  model-selection agreement (exploratory)", t2),
                     ("TABLE A1 threshold selection evidence", a1),
                     ("TABLE A2 five-outcome counts (median)", a2)):
        print("\n" + "=" * 120)
        print(name)
        print("-" * 120)
        print(df.to_string(index=False))

    for fname, df in (("table1_combined.csv", t1), ("table1_raw_medians.csv", t1_raw),
                      ("direction_consistency.csv", dc), ("paired_summary.csv", ps),
                      ("table2_model_selection.csv", t2),
                      ("tableA1_thresholds.csv", a1), ("tableA2_counts.csv", a2)):
        df.to_csv(os.path.join(args.out, fname), index=False, encoding="utf-8-sig")
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
