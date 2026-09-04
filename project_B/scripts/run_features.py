"""파생변수 비교 — notion.md 4장 표의 [+ Feature] 행을 채운다.

모델 셀렉(xgboost)과 불균형 처리(무처리)에서 확정된 값을 고정하고, 파생변수
구성만 바꿔 가며 Recall@FAR 1% 를 잰다. 나머지(분할, in_horizon 판정, val 에서
잡는 1% 운영점, 시드)는 baseline 과 완전히 같게 둔다.

    python scripts/run_features.py
    python scripts/run_features.py --arms age,rolling,roll14
    python scripts/run_features.py --export-only

파생변수는 일률적으로 생성한다 — 전 속성에 같은 규칙(N일 평균 등)을 적용해
설명 비용을 낮춘다. features/build.py 가 이미 설정만으로 다음을 지원하므로
새 코드는 없다: include_age, derived_on(critical|all), diff_lags,
rolling_windows x rolling_stats, include_since_start.

4단계로 나눈다.

  1단계 나이 격리        age_days / segment_age_days 만 켠다.

  2단계 변환 유형 분해    critical 8컬럼, 기본 창([7,30])에서 diff / rolling /
                        since_start 를 하나씩만 켜고, 마지막에 전부 합친다.
                        full_critical_age 는 features.yaml 원래 기본값(84피처,
                        zoo_3month 류가 쓰던 것)과 동일하다.

  3단계 창 길이 스윕      rolling 만, 7 / 14 / 30 을 각각 단독으로. 2단계의
                        rolling([7,30])과 합치면 네 지점이 나온다. 시퀀스
                        모델 lookback 스윕과 같은 축이라 4장에서 나란히 놓을
                        수 있다.

  4단계 범위 스윕         derived_on: all — critical 8개를 넘어 존재하는
                        SMART 18개 전부에 파생을 건다.
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

# 모델 셀렉 + 불균형 처리에서 확정된 값. 이 실험에서는 안 건드린다.
MODEL = "xgboost"

_ALL_OFF = {
    "include_age": False,
    "derived_on": "critical",
    "diff_lags": [],
    "rolling_windows": [],
    "rolling_stats": [],
    "include_since_start": False,
    "cid_windows": [],
    "asfd_windows": [],
}

ARMS: dict[str, dict] = {
    # --- 0단계: 대조군 (파생변수 없음 = baseline) ---
    "raw_only": {
        "desc": "파생 없음, SMART 원값 18개만 (baseline 대조군)",
        "base": dict(_ALL_OFF),
    },
    # --- 1단계: 나이 격리 ---
    "age": {
        "desc": "나이만 (age_days, segment_age_days)",
        "base": {**_ALL_OFF, "include_age": True},
    },
    # --- 2단계: 변환 유형 분해 (critical, 기본 창 [7,30]) ---
    "diff": {
        "desc": "차분만 (diff_lags 1/7/30)",
        "base": {**_ALL_OFF, "diff_lags": [1, 7, 30]},
    },
    "rolling": {
        "desc": "롤링 평균·표준편차만 (7/30일)",
        "base": {
            **_ALL_OFF,
            "rolling_windows": [7, 30],
            "rolling_stats": ["mean", "std"],
        },
    },
    "since_start": {
        "desc": "segment 시작 이후 누적 증가량만",
        "base": {**_ALL_OFF, "include_since_start": True},
    },
    "full_critical": {
        "desc": "차분+롤링+누적 전부 (critical, 나이 제외)",
        "base": {
            **_ALL_OFF,
            "diff_lags": [1, 7, 30],
            "rolling_windows": [7, 30],
            "rolling_stats": ["mean", "std"],
            "include_since_start": True,
        },
    },
    "full_critical_age": {
        "desc": "차분+롤링+누적+나이 전부 (features.yaml 원래 기본값과 동일)",
        "base": {
            "include_age": True,
            "derived_on": "critical",
            "diff_lags": [1, 7, 30],
            "rolling_windows": [7, 30],
            "rolling_stats": ["mean", "std"],
            "include_since_start": True,
        },
    },
    # --- 3단계: 창 길이 스윕 (rolling 만) ---
    "roll7": {
        "desc": "롤링 평균·표준편차, 7일만",
        "base": {**_ALL_OFF, "rolling_windows": [7], "rolling_stats": ["mean", "std"]},
    },
    "roll14": {
        "desc": "롤링 평균·표준편차, 14일만",
        "base": {**_ALL_OFF, "rolling_windows": [14], "rolling_stats": ["mean", "std"]},
    },
    "roll30": {
        "desc": "롤링 평균·표준편차, 30일만",
        "base": {**_ALL_OFF, "rolling_windows": [30], "rolling_stats": ["mean", "std"]},
    },
    # --- 4단계: 범위 스윕 ---
    "scope_all": {
        "desc": "차분+롤링+누적 전부, 존재하는 SMART 18개 전체",
        "base": {
            "include_age": False,
            "derived_on": "all",
            "diff_lags": [1, 7, 30],
            "rolling_windows": [7, 30],
            "rolling_stats": ["mean", "std"],
            "include_since_start": True,
        },
    },
    # --- 5단계: CID / ASFD 개별 창 스윕 ---
    # CID(Complexity-Invariant Distance 의 복잡도 추정치, Batista et al. 2011):
    #   sqrt(sum((x_i - x_{i-1})^2)) — 창 안에서 얼마나 요동쳤는지.
    # ASFD(Absolute Sum of First Differences): CID 의 L1 버전, 총변동량.
    "cid7": {
        "desc": "CID(복잡도 추정치), 7일만",
        "base": {**_ALL_OFF, "cid_windows": [7]},
    },
    "cid14": {
        "desc": "CID(복잡도 추정치), 14일만",
        "base": {**_ALL_OFF, "cid_windows": [14]},
    },
    "cid30": {
        "desc": "CID(복잡도 추정치), 30일만",
        "base": {**_ALL_OFF, "cid_windows": [30]},
    },
    "asfd7": {
        "desc": "ASFD(차분 절대합), 7일만",
        "base": {**_ALL_OFF, "asfd_windows": [7]},
    },
    "asfd14": {
        "desc": "ASFD(차분 절대합), 14일만",
        "base": {**_ALL_OFF, "asfd_windows": [14]},
    },
    "asfd30": {
        "desc": "ASFD(차분 절대합), 30일만",
        "base": {**_ALL_OFF, "asfd_windows": [30]},
    },
    # --- 6단계: 순차 전진 선택 (상위권끼리 짝 조합) ---
    # 5단계까지 단독 최고는 asfd7/roll30(29대), 근접 asfd30/cid30/rolling(28대).
    # 반대로 결합(full_critical=25대)은 단독 롤링(rolling=28대)보다도 낮았다.
    # "합치면 나빠진다"는 패턴이 새 최고 성능자에서도 유지되는지 짝으로 확인.
    "combo_roll30_asfd7": {
        "desc": "롤링30일 + ASFD7일 (공동 1위 결합)",
        "base": {**_ALL_OFF, "rolling_windows": [30], "rolling_stats": ["mean", "std"], "asfd_windows": [7]},
    },
    "combo_roll30_cid30": {
        "desc": "롤링30일 + CID30일",
        "base": {**_ALL_OFF, "rolling_windows": [30], "rolling_stats": ["mean", "std"], "cid_windows": [30]},
    },
    "combo_roll30_asfd30": {
        "desc": "롤링30일 + ASFD30일",
        "base": {**_ALL_OFF, "rolling_windows": [30], "rolling_stats": ["mean", "std"], "asfd_windows": [30]},
    },
    "combo_asfd7_cid30": {
        "desc": "ASFD7일 + CID30일",
        "base": {**_ALL_OFF, "asfd_windows": [7], "cid_windows": [30]},
    },
    "combo_roll30_diff": {
        "desc": "롤링30일 + 차분(1/7/30)",
        "base": {**_ALL_OFF, "rolling_windows": [30], "rolling_stats": ["mean", "std"], "diff_lags": [1, 7, 30]},
    },
    "combo_asfd7_diff": {
        "desc": "ASFD7일 + 차분(1/7/30)",
        "base": {**_ALL_OFF, "asfd_windows": [7], "diff_lags": [1, 7, 30]},
    },
    # --- 7단계: 순차 전진 선택 2회차 ---
    # 6단계 1위는 combo_asfd7_diff(32대). 여기에 세 번째 요소를 하나씩 더해
    # 더 개선되는지 확인한다. 개선이 없으면 2요소가 지역 최적이다.
    "triple_asfd7_diff_roll30": {
        "desc": "ASFD7일 + 차분 + 롤링30일",
        "base": {**_ALL_OFF, "asfd_windows": [7], "diff_lags": [1, 7, 30],
                 "rolling_windows": [30], "rolling_stats": ["mean", "std"]},
    },
    "triple_asfd7_diff_since_start": {
        "desc": "ASFD7일 + 차분 + 누적",
        "base": {**_ALL_OFF, "asfd_windows": [7], "diff_lags": [1, 7, 30],
                 "include_since_start": True},
    },
    "triple_asfd7_diff_cid30": {
        "desc": "ASFD7일 + 차분 + CID30일",
        "base": {**_ALL_OFF, "asfd_windows": [7], "diff_lags": [1, 7, 30], "cid_windows": [30]},
    },
    "triple_asfd7_diff_age": {
        "desc": "ASFD7일 + 차분 + 나이",
        "base": {**_ALL_OFF, "asfd_windows": [7], "diff_lags": [1, 7, 30], "include_age": True},
    },
    # --- 9단계: ASFD 창 x diff 격자 (주 단위 정렬) ---
    # 30일은 4주(28일)에 이틀이 붙어 요일 주기가 섞인다. 창을 7/14/28 로
    # 두면 전부 주의 배수라 그 문제가 없다. ASFD 창 3개 x diff 격자 2개를
    # 전부 채워 최적 조합을 찾는다.
    "asfd28": {
        "desc": "ASFD(차분 절대합), 28일만",
        "base": {**_ALL_OFF, "asfd_windows": [28]},
    },
    "combo_asfd14_diff": {
        "desc": "ASFD14일 + 차분(1/7/30)",
        "base": {**_ALL_OFF, "asfd_windows": [14], "diff_lags": [1, 7, 30]},
    },
    "combo_asfd28_diff": {
        "desc": "ASFD28일 + 차분(1/7/30)",
        "base": {**_ALL_OFF, "asfd_windows": [28], "diff_lags": [1, 7, 30]},
    },
    "combo_asfd7_diffw": {
        "desc": "ASFD7일 + 차분 주단위(1/7/28)",
        "base": {**_ALL_OFF, "asfd_windows": [7], "diff_lags": [1, 7, 28]},
    },
    "combo_asfd14_diffw": {
        "desc": "ASFD14일 + 차분 주단위(1/7/28)",
        "base": {**_ALL_OFF, "asfd_windows": [14], "diff_lags": [1, 7, 28]},
    },
    "combo_asfd28_diffw": {
        "desc": "ASFD28일 + 차분 주단위(1/7/28)",
        "base": {**_ALL_OFF, "asfd_windows": [28], "diff_lags": [1, 7, 28]},
    },
    # --- 10단계: 차분을 lag 1 만으로 ---
    # 설명 비용: "전일 대비 변화량" 한 문장이면 끝난다. lag 7/30 은 각각
    # 따로 정당화해야 한다.
    # 중복: ASFD_W 가 이미 창 안의 누적 변동(총변동)을 담는다. d7 은 7일
    # 순변화라 ASFD7 과 정보가 겹친다. d1(당일 변화) + ASFD_W(누적 변동)가
    # 서로 겹치지 않는 짝이다.
    "combo_asfd7_d1": {
        "desc": "ASFD7일 + 차분 1일만",
        "base": {**_ALL_OFF, "asfd_windows": [7], "diff_lags": [1]},
    },
    "combo_asfd14_d1": {
        "desc": "ASFD14일 + 차분 1일만",
        "base": {**_ALL_OFF, "asfd_windows": [14], "diff_lags": [1]},
    },
    "combo_asfd28_d1": {
        "desc": "ASFD28일 + 차분 1일만",
        "base": {**_ALL_OFF, "asfd_windows": [28], "diff_lags": [1]},
    },
    # --- 8단계: 승자 조합의 범위 스윕 ---
    # 7단계까지 최적은 critical 8컬럼에서 ASFD7일+차분(1/7/30) (32대). 이걸
    # critical 을 넘어 존재하는 SMART 18개 전체로 넓히면 더 나은지 확인.
    "scope_all_champion": {
        "desc": "승자 조합(ASFD7일+차분)을 SMART 18개 전체로 확장",
        "base": {
            "include_age": False, "derived_on": "all",
            "diff_lags": [1, 7, 30], "rolling_windows": [], "rolling_stats": [],
            "include_since_start": False, "cid_windows": [], "asfd_windows": [7],
        },
    },
}

STAGES = {
    1: ["age"],
    2: ["diff", "rolling", "since_start", "full_critical", "full_critical_age"],
    3: ["roll7", "roll14", "roll30"],
    4: ["scope_all"],
    5: ["cid7", "cid14", "cid30", "asfd7", "asfd14", "asfd30"],
}


def write_experiment(arm: str, seeds: list[int] | None = None) -> Path:
    spec = ARMS[arm]
    base = yaml.safe_load((EXP_DIR / "baseline.yaml").read_text(encoding="utf-8"))
    cfg = copy.deepcopy(base)
    cfg["experiment"] = "feat_" + arm
    cfg["models"] = ["configs/models/" + MODEL + ".yaml"]
    if seeds:
        cfg["seeds"] = list(seeds)

    ov = cfg["overrides"]
    # baseline 이 박아둔 파생변수 설정(전부 off)을 걷어내고 이 arm 것으로 다시 깐다.
    for key in list(ov):
        if key.startswith("features.base."):
            del ov[key]
    for key, value in spec["base"].items():
        ov["features.base." + key] = value

    path = EXP_DIR / ("feat_" + arm + ".yaml")
    header = (
        "# 파생변수 비교 — arm: " + arm + "\n"
        "#   " + spec["desc"] + "\n"
        "#   기반 모델: " + MODEL + " (모델 셀렉 결과) + 불균형 처리 없음 (확정값)\n"
        "#   baseline.yaml 과 features.base.* 축 하나만 다르다.\n"
        "#   scripts/run_features.py 가 생성한다. 직접 고치지 마라.\n\n"
    )
    path.write_text(
        header + yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return path


def run(cmd: list[str]) -> bool:
    """한 arm 이 죽어도 나머지를 계속 돌린다. 실패는 끝에 모아 보고한다."""
    print("\n$ " + " ".join(cmd), flush=True)
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        print("!!!! 실패 (exit " + str(result.returncode) + "): " + " ".join(cmd))
        return False
    return True


def merge(threshold_source: str, out_prefix: str) -> None:
    """arms 로 지정된 것만이 아니라, 이제까지 완료된 모든 arm 의 개별 CSV 를
    전부 훑어서 합친다. 1단계/2단계/3단계를 따로 실행하는 "차근차근" 흐름이라
    한 번의 호출이 아는 arm 목록으로만 합치면 이전 단계 결과가 지워진다."""
    import pandas as pd

    frames = []
    for arm in ARMS:  # arms 인자가 아니라 전체 정의된 arm 을 훑는다
        path = ROOT / "results" / (out_prefix + "_" + arm + "_long.csv")
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
    out = ROOT / "results" / (out_prefix + "_long.csv")
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
    out = ROOT / "results" / (out_prefix + "_summary.csv")
    grouped.to_csv(out, index=False, encoding="utf-8-sig")
    print("[요약]   " + str(out) + "  (" + str(len(grouped)) + " 행)")

    one = grouped[np.isclose(grouped["far_target"], 0.01)]
    if len(one):
        print("\n--- Recall @ 디스크 FAR 1% (" + threshold_source + ") ---")
        cols = ["arm", "arm_desc", "recall", "precision", "far", "roc_auc"]
        print(one[cols].to_string(index=False))


def main() -> int:
    ap = argparse.ArgumentParser(description="파생변수 비교")
    ap.add_argument("--arms", default=None, help="쉼표로 구분. 기본은 전부")
    ap.add_argument(
        "--stage", type=int, choices=[1, 2, 3, 4], default=None,
        help="단계별 실행 (1=나이, 2=변환유형, 3=창길이, 4=범위). --arms 와 함께 못 씀",
    )
    ap.add_argument("--export-only", action="store_true", help="학습 없이 CSV 만")
    ap.add_argument("--configs-only", action="store_true", help="config 생성만")
    ap.add_argument("--merge-only", action="store_true", help="이미 나온 CSV 만 합친다")
    # 시드를 여러 개 주면 학습 변동을 잰다. 평가 표본(고장 47대)은 그대로이므로
    # 시드가 잡는 것은 모델 학습의 변동뿐이고, 표본 자체의 변동은 못 잡는다.
    ap.add_argument("--seeds", default=None,
                    help="쉼표로 구분한 시드 목록. 기본은 baseline.yaml 의 값(42)")
    args = ap.parse_args()

    if args.stage and args.arms:
        raise SystemExit("--stage 와 --arms 는 함께 쓸 수 없다.")
    if args.stage:
        arms = STAGES[args.stage]
    elif args.arms:
        arms = [a.strip() for a in args.arms.split(",")]
    else:
        arms = list(ARMS)
    unknown = [a for a in arms if a not in ARMS]
    if unknown:
        raise SystemExit("모르는 arm: " + str(unknown) + " (가능: " + str(list(ARMS)) + ")")

    if args.merge_only:
        merge("validation", "features")
        return 0

    seeds = [int(x) for x in args.seeds.split(",")] if args.seeds else None
    written = [write_experiment(a, seeds) for a in arms]
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
                "configs/experiments/feat_" + arm + ".yaml",
            ]):
                failed.append(arm)
                continue
        ok_val = run([
            py, "-u", "scripts/export_results.py", "feat_" + arm,
            "--out", "results/features_" + arm, "--threshold-source", "val_quantile",
        ])
        ok_oracle = run([
            py, "-u", "scripts/export_results.py", "feat_" + arm,
            "--out", "results/features_oracle_" + arm,
            "--threshold-source", "month_quantile",
        ])
        if not (ok_val and ok_oracle):
            failed.append(arm)

    survivors = [a for a in arms if a not in failed]
    merge("validation", "features")
    merge("oracle", "features_oracle")
    if failed:
        print("\n!!!! 실패한 arm: " + ", ".join(failed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
