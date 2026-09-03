"""불균형 처리 방식 비교 — notion.md 4장 표의 [+ Imbalance] 행을 채운다.

모델 셀렉에서 뽑힌 모델 하나를 고정하고, 불균형을 다루는 방식만 바꿔 가며
Recall@FAR 1% 를 잰다. 나머지(파생변수 없음, 분할, in_horizon 판정, val 에서
잡는 1% 운영점, 시드)는 baseline 과 완전히 같게 둔다.

    python scripts/run_imbalance.py --model randomforest
    python scripts/run_imbalance.py --model randomforest --arms none,rus_10,smote
    python scripts/run_imbalance.py --model randomforest --export-only

축이 넷이다.

  A 아무 처리 없음        전 표본 2,500만 행을 그대로 (= baseline)

  B 알고리즘 수준         손실/투표에 가중치를 준다. 표본은 안 건드린다.
                          부스팅은 scale_pos_weight, RF 는 class_weight.

  C 데이터 수준 · 감축     음성을 줄인다. DuckDB 단에서 처리해 메모리에 아예
                          안 올린다.
                            ratio     양성 전량 + 음성 (k x 양성) 무작위
                            stride    음성을 n일 간격으로 (시간축 유지)
                            match_val 학습 유병률을 val 창에 맞춘다

  D 데이터 수준 · 합성     소수 클래스를 늘린다. 이웃 탐색이 필요해 전 표본에는
                          걸 수 없으므로 ratio 1:50 으로 줄인 뒤에 적용한다.
                            ros / smote / borderline_smote / adasyn
                            smote_tomek / smote_enn (합성 후 경계 정리)

  E 앙상블                 표집을 모델 안으로 넣는다.
                            balanced_bagging  RUS 부분표본마다 같은 모델을 학습
                            balanced_rf       트리마다 RUS
                            easy_ensemble     RUS 부분표본마다 AdaBoost (정의 그대로)
                            rusboost          부스팅 매 회차에 RUS

주의 1. D 의 합성 표본은 원본 행에 대응하는 serial/record_date 가 없다. 학습에만
       쓰이고 평가 조각은 손대지 않으므로 val/test 의 유병률과 오탐률은 그대로다.

주의 2. E 는 표집 방식만이 아니라 분류기 자체가 바뀐다. balanced_bagging 만 기반
       모델을 셀렉된 것으로 둘 수 있고, 나머지 셋은 정의상 트리/AdaBoost 로
       고정이다. 표에 적을 때 그 사실을 함께 적는다.
"""

from __future__ import annotations

import argparse
import copy
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CONFIGS = ROOT / "configs"
EXP_DIR = CONFIGS / "experiments"
IMB_MODELS = CONFIGS / "models" / "imb"

# 합성 계열을 걸기 전에 DuckDB 에서 음성을 이만큼 줄인다.
# 양성 13,719 x 50 = 68만 음성 -> 70만 행. 이웃 탐색이 감당하는 규모다.
PRE_RATIO = 50

# 가중치 인자 이름이 모델마다 다르다.
COST_FLAG = {
    "lightgbm": "auto_scale_pos_weight",
    "xgboost": "auto_scale_pos_weight",
    "catboost": "auto_scale_pos_weight",
    "randomforest": "auto_class_weight",
    "extratrees": "auto_class_weight",
    "mlp": "auto_pos_weight",
    "tabresnet": "auto_pos_weight",
    "fttransformer": "auto_pos_weight",
}

ARMS: dict[str, dict] = {
    "none": {
        "desc": "처리 없음 (전 표본)",
        "sampling": {"strategy": "none"},
    },
    "cost": {
        "desc": "비용민감 가중 (전 표본)",
        "sampling": {"strategy": "none"},
        "cost_sensitive": True,
    },
    "rus_50": {
        "desc": "무작위 언더샘플링 1:50",
        "sampling": {"strategy": "ratio", "negative_ratio": 50, "seed": 42},
    },
    "rus_10": {
        "desc": "무작위 언더샘플링 1:10",
        "sampling": {"strategy": "ratio", "negative_ratio": 10, "seed": 42},
    },
    "rus_1": {
        "desc": "무작위 언더샘플링 1:1",
        "sampling": {"strategy": "ratio", "negative_ratio": 1, "seed": 42},
    },
    "stride_30": {
        "desc": "시간축 언더샘플링 (음성 30일 간격)",
        "sampling": {"strategy": "stride", "negative_stride": 30},
    },
    "match_val": {
        "desc": "학습 유병률을 val 창에 정합",
        "sampling": {"strategy": "match_val", "keep_recent_months": 1, "seed": 42},
    },
    "ros": {
        "desc": "RUS 1:50 + 무작위 오버샘플링 1:1",
        "sampling": {"strategy": "ratio", "negative_ratio": PRE_RATIO, "seed": 42},
        "resample": {"method": "random_over"},
    },
    "smote": {
        "desc": "RUS 1:50 + SMOTE 1:1",
        "sampling": {"strategy": "ratio", "negative_ratio": PRE_RATIO, "seed": 42},
        "resample": {"method": "smote", "params": {"k_neighbors": 5}},
    },
    "borderline_smote": {
        "desc": "RUS 1:50 + Borderline-SMOTE 1:1",
        "sampling": {"strategy": "ratio", "negative_ratio": PRE_RATIO, "seed": 42},
        "resample": {"method": "borderline_smote", "params": {"k_neighbors": 5}},
    },
    "adasyn": {
        "desc": "RUS 1:50 + ADASYN 1:1",
        "sampling": {"strategy": "ratio", "negative_ratio": PRE_RATIO, "seed": 42},
        "resample": {"method": "adasyn", "params": {"n_neighbors": 5}},
    },
    "smote_tomek": {
        "desc": "RUS 1:50 + SMOTE + Tomek link 정리",
        "sampling": {"strategy": "ratio", "negative_ratio": PRE_RATIO, "seed": 42},
        "resample": {"method": "smote_tomek"},
    },
    "smote_enn": {
        "desc": "RUS 1:50 + SMOTE + ENN 정리",
        "sampling": {"strategy": "ratio", "negative_ratio": PRE_RATIO, "seed": 42},
        "resample": {"method": "smote_enn"},
    },
    "balanced_bagging": {
        "desc": "Balanced Bagging (RUS 1:1 부분표본 10개 x 셀렉 모델)",
        "sampling": {"strategy": "none"},
        "ensemble": "balanced_bagging",
    },
    "balanced_rf": {
        "desc": "Balanced Random Forest (트리마다 RUS)",
        "sampling": {"strategy": "none"},
        "ensemble": "balanced_rf",
    },
    "easy_ensemble": {
        "desc": "EasyEnsemble (RUS 부분표본 10개 x AdaBoost)",
        "sampling": {"strategy": "none"},
        "ensemble": "easy_ensemble",
    },
    "rusboost": {
        "desc": "RUSBoost (부스팅 매 회차 RUS)",
        "sampling": {"strategy": "none"},
        "ensemble": "rusboost",
    },
}

# 앙상블 계열의 추정기 경로와 인자. 기반 모델을 바꿀 수 있는 건 bagging 뿐이다.
ENSEMBLES = {
    "balanced_rf": (
        "imblearn.ensemble.BalancedRandomForestClassifier",
        {
            "n_estimators": 300,
            "min_samples_leaf": 20,
            "max_features": "sqrt",
            "sampling_strategy": "auto",
            "replacement": True,
            "bootstrap": False,
            "n_jobs": 8,
        },
    ),
    "easy_ensemble": (
        "imblearn.ensemble.EasyEnsembleClassifier",
        {"n_estimators": 10, "sampling_strategy": "auto", "n_jobs": 4},
    ),
    "rusboost": (
        "imblearn.ensemble.RUSBoostClassifier",
        {"n_estimators": 200, "learning_rate": 0.05, "sampling_strategy": "auto"},
    ),
}

def write_ensemble_model(arm: str, model_name: str) -> Path:
    """앙상블 계열 arm 하나를 위한 모델 config 를 만든다."""
    name = model_name + "_" + arm

    if arm == "balanced_bagging":
        # 기반 모델을 셀렉된 것 그대로 쓴다. imblearn 의 BalancedBagging 은
        # sklearn 추정기만 받으므로 torch 모델(MLP 등)을 못 꽂는다. 같은
        # 알고리즘을 hddpred 모델 위에서 도는 형태로 따로 구현해 그걸 쓴다.
        cfg = {
            "name": name,
            "family": "tabular",
            "class": "hddpred.models.balanced_ensemble.BalancedEnsembleModel",
            "params": {
                "base": "configs/models/" + model_name + ".yaml",
                "n_estimators": 10,
                "negative_ratio": 1,
            },
            "training": {},
        }
        IMB_MODELS.mkdir(parents=True, exist_ok=True)
        path = IMB_MODELS / (name + ".yaml")
        path.write_text(
            yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )
        return path

    dotted, kwargs = ENSEMBLES[arm]
    kwargs = copy.deepcopy(kwargs)
    cfg = {
        "name": name,
        "family": "tabular",
        "class": "hddpred.models.sklearn_models.SklearnModel",
        "params": {"estimator": dotted, "kwargs": kwargs},
        # imblearn 앙상블은 결측을 그대로 받지 못한다. 트리는 스케일에 불변이라
        # 표준화 자체는 결과를 바꾸지 않고 결측을 평균(0)으로 채우는 역할만 한다.
        "training": {"scale_inputs": True},
    }
    IMB_MODELS.mkdir(parents=True, exist_ok=True)
    path = IMB_MODELS / (name + ".yaml")
    path.write_text(
        yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    return path


def write_experiment(arm: str, model_name: str) -> Path:
    spec = ARMS[arm]
    base = yaml.safe_load((EXP_DIR / "baseline.yaml").read_text(encoding="utf-8"))
    cfg = copy.deepcopy(base)
    cfg["experiment"] = "imb_" + arm

    if spec.get("ensemble"):
        model_path = write_ensemble_model(arm, model_name)
        cfg["models"] = ["configs/models/imb/" + model_path.name]
    else:
        cfg["models"] = ["configs/models/" + model_name + ".yaml"]

    ov = cfg["overrides"]
    # baseline 이 박아둔 표집 설정을 걷어내고 이 arm 것으로 다시 깐다.
    for key in list(ov):
        if key.startswith("features.train_sampling") or key.startswith(
            "features.resample"
        ):
            del ov[key]
    for key, value in spec["sampling"].items():
        ov["features.train_sampling." + key] = value
    for key, value in (spec.get("resample") or {}).items():
        ov["features.resample." + key] = value
    if spec.get("cost_sensitive"):
        flag = COST_FLAG.get(model_name)
        if flag is None:
            raise SystemExit(model_name + " 의 비용민감 인자 이름을 모른다.")
        ov["models." + model_name + ".training." + flag] = True
        if model_name in ("mlp", "tabresnet", "fttransformer"):
            ov["models." + model_name + ".training.loss"] = "bce"

    path = EXP_DIR / ("imb_" + arm + ".yaml")
    header = (
        "# 불균형 처리 비교 — arm: " + arm + "\n"
        "#   " + spec["desc"] + "\n"
        "#   기반 모델: " + model_name + " (모델 셀렉 결과)\n"
        "#   baseline.yaml 과 표집/가중 축 하나만 다르다.\n"
        "#   scripts/run_imbalance.py 가 생성한다. 직접 고치지 마라.\n\n"
    )
    path.write_text(
        header + yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return path


def run(cmd: list[str]) -> bool:
    """한 arm 이 죽어도 나머지를 계속 돌린다. 17개를 밤새 돌리는데 하나
    때문에 전부 멈추면 손해가 크다. 실패는 끝에 모아 보고한다."""
    print("\n$ " + " ".join(cmd), flush=True)
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        print("!!!! 실패 (exit " + str(result.returncode) + "): " + " ".join(cmd))
        return False
    return True


def merge(arms: list[str]) -> None:
    import pandas as pd

    frames = []
    for arm in arms:
        path = ROOT / "results" / ("imb_" + arm + "_long.csv")
        if not path.exists():
            print("  [merge] 건너뜀 (없음): " + path.name)
            continue
        frame = pd.read_csv(path)
        frame.insert(0, "arm", arm)
        frame.insert(1, "arm_desc", ARMS[arm]["desc"])
        frames.append(frame)
    if not frames:
        print("합칠 결과가 없다.")
        return

    long = pd.concat(frames, ignore_index=True)
    out = ROOT / "results" / "imbalance_long.csv"
    long.to_csv(out, index=False, encoding="utf-8-sig")
    print("\n[원자료] " + str(out) + "  (" + str(len(long)) + " 행)")

    # 세 달을 고장 사건 하나의 모집단으로 풀링한다.
    grouped = long.groupby(
        ["arm", "arm_desc", "model", "far_target"], as_index=False
    ).agg(
        n_failed=("n_failed", "sum"),
        n_healthy=("n_healthy", "sum"),
        tp=("tp", "sum"),
        fn=("fn", "sum"),
        fp=("fp", "sum"),
        tn=("tn", "sum"),
        roc_auc=("roc_auc", "mean"),
    )
    grouped["recall"] = grouped["tp"] / (grouped["tp"] + grouped["fn"])
    denom = (grouped["tp"] + grouped["fp"]).replace(0, np.nan)
    grouped["precision"] = grouped["tp"] / denom
    grouped["far"] = grouped["fp"] / (grouped["fp"] + grouped["tn"])
    grouped = grouped.sort_values(["far_target", "recall"], ascending=[True, False])
    out = ROOT / "results" / "imbalance_summary.csv"
    grouped.to_csv(out, index=False, encoding="utf-8-sig")
    print("[요약]   " + str(out) + "  (" + str(len(grouped)) + " 행)")

    one = grouped[np.isclose(grouped["far_target"], 0.01)]
    if len(one):
        print("\n--- Recall @ 디스크 FAR 1% ---")
        cols = ["arm", "arm_desc", "recall", "precision", "far", "roc_auc"]
        print(one[cols].to_string(index=False))


def main() -> int:
    ap = argparse.ArgumentParser(description="불균형 처리 방식 비교")
    ap.add_argument("--model", required=True, help="모델 셀렉에서 뽑힌 모델 이름")
    ap.add_argument("--arms", default=None, help="쉼표로 구분. 기본은 전부")
    ap.add_argument("--export-only", action="store_true", help="학습 없이 CSV 만")
    ap.add_argument("--configs-only", action="store_true", help="config 생성만")
    ap.add_argument("--merge-only", action="store_true", help="이미 나온 CSV 만 합친다")
    args = ap.parse_args()

    arms = [a.strip() for a in args.arms.split(",")] if args.arms else list(ARMS)
    unknown = [a for a in arms if a not in ARMS]
    if unknown:
        raise SystemExit("모르는 arm: " + str(unknown) + " (가능: " + str(list(ARMS)) + ")")

    if args.merge_only:
        merge(arms)
        return 0

    written = [write_experiment(a, args.model) for a in arms]
    print("[config] " + str(len(written)) + "개 생성")
    for arm in arms:
        print("  " + arm.ljust(18) + ARMS[arm]["desc"])
    if args.configs_only:
        return 0

    py = sys.executable
    failed = []
    for arm in arms:
        print("\n@@@@ arm " + arm + " — " + ARMS[arm]["desc"], flush=True)
        if not args.export_only:
            if not run([
                py, "-u", "scripts/run_experiment.py",
                "configs/experiments/imb_" + arm + ".yaml",
            ]):
                failed.append(arm)
                continue
        if not run([
            py, "-u", "scripts/export_results.py", "imb_" + arm,
            "--out", "results/imb_" + arm, "--threshold-source", "val_quantile",
        ]):
            failed.append(arm)

    merge([a for a in arms if a not in failed])
    if failed:
        print("\n!!!! 실패한 arm: " + ", ".join(failed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
