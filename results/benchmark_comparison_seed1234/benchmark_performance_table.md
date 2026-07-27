# Model Benchmark Performance Table (Row-level vs Rolling Inference)

| Model | Row Precision | Row Recall | Row F1 | Row PR-AUC | Row FAR (%) | Rolling Thresh | Disk Precision | Disk Recall | Disk F1 | Disk FAR (%) | Median Lead Time | EDR@15 (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | 0.0236 | 0.5648 | 0.0452 | 0.0562 | 4.40% | 0.9466 | 0.5981 | 0.2783 | 0.3798 | 1.18% | 1.0d | 1.30% |
| **LightGBM** | 0.0185 | 0.6010 | 0.0359 | 0.0606 | 5.98% | 0.9652 | 0.5319 | 0.3261 | 0.4043 | 1.81% | 5.0d | 7.39% |
| **MLP** | 0.0110 | 0.5841 | 0.0217 | 0.0659 | 9.83% | 0.9863 | 0.5345 | 0.2696 | 0.3584 | 1.48% | 3.0d | 5.22% |
| **GRU** | 0.0187 | 0.5960 | 0.0363 | 0.0634 | 5.87% | 0.9673 | 0.5368 | 0.3174 | 0.3989 | 1.72% | 4.0d | 6.52% |
