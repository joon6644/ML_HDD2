# 하이퍼파라미터 탐색 기록


## 2026-09-13 06:03 — TCN / `tcn_tos_win10_pauc5_mon`

- 시도 50회 → 완료 10 / 가지치기 40
- 최고 val pAUC@FAR<=5% **0.8965** (trial 4): width 32, levels 3, k 3, dropout 0.098, lr 5.46e-04, wd 1.92e-03, batch 1024
- 완료 시도 전체 폭 0.0051, 상위 5개 폭 0.0025
- 판정: 상위권이 구분되지 않는다 (상위 5개 폭 0.0025). 더 돌려도 같은 자리일 가능성이 높다.
- 이어 돌리려면: `python scripts/run_optuna_tcn.py --trials 90`
