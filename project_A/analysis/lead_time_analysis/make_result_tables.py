"""Emit the two result tables in markdown, straight from the swept curves.

Table 3 summarises each dataset's operating-characteristic curve rather than
listing arbitrary points on it: where operational recall peaks, what the first
alarm's timing is there, and how far the early alarm rate has to be squeezed
before the median first alarm lands inside H. The last block is the one that
matters most -- what doubling EAR actually buys.
"""
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(ROOT, "results", "lead_time_analysis")
SHORT = ["HGST", "Seagate", "Toshiba"]
H = 30

c = pd.read_parquet(os.path.join(RES, "burden_sweep_curves.parquet"))

print("표 3. Early Alarm Rate에 따른 운영 지표 (4모델 x 30 seed의 중앙값)\n")
print("| Dataset | EAR at peak recall | Recall | Precision | Median LT | EAR with median LT <= H | Recall there |")
print("| --- | --- | --- | --- | --- | --- | --- |")
for ds in SHORT:
    d = c[c.dataset == ds].sort_values("ear_grid")
    p = d.loc[d.recall_med.idxmax()]
    u = d[d.lt_p50_med <= H]
    if u.empty:
        cell, rec = "none", "—"
    else:
        top = u.loc[u.ear_grid.idxmax()]
        cell, rec = f"≤ {top.ear_grid*100:.2f}%", f"{top.recall_med:.3f}"
    print(f"| {ds} | {p.ear_grid*100:.2f}% | {p.recall_med:.3f} "
          f"({p.recall_q1:.3f}–{p.recall_q3:.3f}) | {p.precision_med:.3f} | "
          f"{p.lt_p50_med:.0f} d | {cell} | {rec} |")

print("\n\nEAR을 늘리면 무엇을 사는가 (본문 인용용)\n")
print("| Dataset | EAR | Recall | Precision | LT p25 | p50 | p75 | On-time | Early | Missed |")
print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
for ds in SHORT:
    d = c[c.dataset == ds].sort_values("ear_grid")
    for t in [0.005, 0.01, 0.025, 0.05, 0.10, 0.20]:
        r = d.loc[(d.ear_grid - t).abs().idxmin()]
        print(f"| {ds} | {r.ear_grid*100:.2f}% | {r.recall_med:.3f} | {r.precision_med:.3f} | "
              f"{r.lt_p25_med:.0f} | {r.lt_p50_med:.0f} | {r.lt_p75_med:.0f} | "
              f"{r.ot_med:.2f} | {r.early_med:.2f} | {r.missed_med:.2f} |")
