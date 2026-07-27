# EasyEnsemble LightGBM Performance Summary (Seed 1234)

- **Model**: EasyEnsemble LightGBM (10 Base Estimators, Soft Voting)
- **Random Seed**: `1234`
- **Optimal Threshold (Val F1-max)**: `0.9643`

---

## 📊 Disk-Level Performance Table

| Model | Evaluation Type | Threshold | Disk-level Precision | Disk-level Recall | F1-score | PR-AUC | Disk-level FAR (%) | Median Lead Time | EDR@15 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **EasyEnsemble-LGBM** | General Classification | `0.9643` | `0.1247` | `0.2714` | `0.1709` | `0.0630` | `0.3579%` | N/A | N/A |
| **EasyEnsemble-LGBM** | Rolling Inference | `0.9643` | `0.5168` | `0.3348` | `0.4063` | N/A | `1.9694%` | `5.00 days` | `7.39%` |

---

## 📈 Key Rolling Inference Metrics

- **Disk-level Precision**: `0.5168` (경고한 HDD 중 실제 30일 이내 고장 적시 탐지 비율)
- **Disk-level Recall**: `0.3348` (실제 고장 HDD 중 30일 이내 사전 탐지한 비율)
- **Disk-level FAR**: `1.9694%` (정상 HDD 중 오경보 발생 비율)
- **Median Lead Time**: `5.00` days (탐지된 HDD의 조기 탐지 시점 중앙값)
- **EDR@15**: `7.39%` (15일 이상 충분히 일찍 탐지한 HDD 비율)
