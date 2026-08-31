# project_B — 시간축 forward split 기반 HDD 고장 예측

## 연구 설정

> 과거 관측만으로 학습하고, **다음 한 달**을 예측한다.
> 관측 시점 `t` 마다 "이 HDD가 `(t, t + H]` 안에 고장나는가"를 판정하고,
> 그 예측들을 **디스크 하나의 혼동행렬 판정으로 접어서** 평가한다.
> `H` 는 설정값이고 기본 10일이다.

핵심 축 셋:

| | |
| --- | --- |
| 분할 | 시간축 forward chaining, test = 다음 1개월 |
| 라벨 | horizon 분류 `(t, t+H]`, 우측 중도절단은 마지막 `H` 일 절단 |
| 판정 | **구간 밖에서 울리면 FP, 구간 안에서만 울리면 TP** (아래 [평가 정의](#평가-정의--디스크-단위-혼동행렬)) |

### 설정의 선행연구 근거

`project_A/table1_hdd_unit_review.md` 의 전수 조사 결과를 그대로 따른다.

- **horizon 10일** — 다수파다. SMARTer(2020) `Interval 10 d`,
  LightGBM+CID(2021) `10 d`, Model update(2021) `10 d`,
  Disk replacement(2016) `10–15 d`.
- **디스크 단위 OR 집계** — 리뷰 문서의 **계열 B**. Murray et al.(2005) 이
  multiple-instance 로 정식화했고, LightGBM+CID 는 Algorithm 1 로,
  LSTM specificity(2020) 는 "If there are alarm records in the sequence" 로
  같은 규칙을 쓴다. 문서 결론은 *"굳이 대표를 꼽으면 OR 집계"* 다.
- **FAR 분모 = 미고장 디스크** — RODMAN(2019) 의 정의.

---

## 평가 정의 — 디스크 단위 혼동행렬

행 단위 예측을 HDD 하나의 판정으로 접는 규칙이 이 프로젝트의 핵심 정의다.
test 한 달 구간 안에서, horizon `H` 를 기준으로:

```text
out        = horizon 밖(y=0 행) 알람 >= 1
in         = horizon 안(y=1 행) 알람 >= 1
has_window = 창 안에 y=1 행이 있는가 (정답 구간의 존재)

out                       -> FP   구간 밖에서 울렸다
~out &  in                -> TP   구간 안에서만 울렸다
~out & ~in &  has_window  -> FN   구간이 있는데 안 울렸다
~out & ~in & ~has_window  -> TN   구간도 없고 안 울렸다
```

`~out & ~in` 이 곧 "알람 없음"이므로 **네 칸이 전체를 빈틈없이 나눈다.**
`y=1` 이 곧 "고장 `H` 일 이내"이므로 규칙은 이렇게도 쓸 수 있다.

> TP = (행 단위 TP ≥ 1) AND (행 단위 FP == 0)

우측 중도절단 디스크는 관측 종료 직전 `H` 일을 잘라내고 투입한다
(`labeling.require_horizon_observability`). 그 구간은 "고장이 없었다"가 아니라
"확인할 수 없다"이기 때문이다.

### 함께 기록하는 값

FP 는 두 종류가 섞인다. 분해해서 남긴다.

| 키 | 뜻 |
| --- | --- |
| `fp_early` | 정답 구간이 있는 디스크인데 구간 밖에서 울림 |
| `fp_healthy` | 정답 구간이 없는 디스크가 울림 |
| `recall` | `TP/(TP+FN)` — 혼동행렬 행 여백 |
| `detection_rate` | `TP/(TP+FN+fp_early)` — 정답 구간이 있는 디스크 중 제때 잡은 비율 |
| `far` | `FP/(FP+TN)` — 혼동행렬 열 여백 |
| `far_healthy` | `fp_healthy/미고장 디스크` — RODMAN 식 분모 |

`recall` 의 분모에는 `fp_early` 가 빠지므로 두 값을 같이 봐야 한다.

### 집계 창 `window_days`

기본값 `null` — **test 달 전체가 한 창**이다. `rule: on_time` 에서 `W <= H` 로
두면 창 안 모든 행이 고장 `H` 일 이내가 되어 horizon 밖 알람이 존재할 수 없고
규칙이 단순 OR 로 붕괴한다. 그 조합은 설정 단계에서 막는다.

### `rule: or` — 선행연구 계열 B 비교용

창 안에 알람이 하나라도 있으면 양성으로 보는 표준 집계다. Murray et al.(2005)
의 multiple-instance 정식화, LightGBM+CID(2021) Algorithm 1, LSTM
specificity(2020) 가 같은 규칙이다. 이때 정답은 `ground_truth` 로 정한다.

- `calendar` — 창의 달력 구간 안에 실제 고장일이 있는 디스크만 양성.
  RODMAN 정의(*"the total number of actual disk failures in one-month
  testing"*). 창 밖 고장을 미리 맞힌 알람은 FP 가 된다.
- `label_or` — 창 안 표본의 양성 라벨을 OR 한다.

### 운영점

RODMAN 과 StreamDFP 는 임곗값을 F1 으로 고르지 않고 **오탐률을 고정한다.**

| 연구 | 고정값 |
| --- | --- |
| RODMAN — Alibaba | FPR 0.1% (TPR 92.8%) |
| RODMAN — Backblaze | FPR 4.0% (TPR 82.4%) |
| StreamDFP | FPR 1% |

`threshold.policy: fixed_disk_far` 가 이 방식이다. 분모가 미고장 **디스크**라
행 단위 `fixed_fpr` 과 다르다. 이 정책에서는 "매일 알람" 같은 전략이
원천적으로 막힌다.

---

## 원칙

1. **원본은 절대 수정하지 않는다.** `../data/raw` 는 읽기 전용이다.
   모든 파생 데이터는 `project_B/data` 아래에만 쓴다.
2. **파생 데이터는 원본 + 코드 + 설정으로 재생성 가능하다.**
   `data/` 를 통째로 지워도 `scripts/prepare_data.py` 하나로 복원된다.
3. **설정이 바뀌면 새 hash 폴더가 생긴다.** 기존 결과를 덮어쓰지 않는다.
4. **단계는 독립 모듈이다.** 순서를 아는 것은 `experiments/runner.py` 뿐이다.
5. **test 는 모델 선정에 쓰지 않는다.** 임곗값도 validation 에서만 고른다.

---

## 데이터 흐름

```text
../data/raw/<drive>.parquet          원본. 읽기 전용. 전 컬럼 string.
        │
        │  canonicalize   전처리 로직은 project_A/preprocessing/common.py 이식
        │                 (수치 캐스팅 + 결측 90% 컬럼 제거 + 중복 제거
        │                  + segment 분리 + forward fill + 고장후 재기록 제거)
        ▼
data/canonical/<drive>/canon=<hash>/data/month=YYYY-MM/*.parquet
        │
        ├─ features   같은 디스크의 과거만 보는 인과적 변환 (diff / rolling)
        │             fold 와 무관하므로 전 구간에서 한 번만 계산한다
        │             ▼
        │       data/features/<drive>/canon=<h>/feat=<h>/
        │
        └─ labels     (t, t+H] 안의 고장 여부 + 검열 처리
                      ▼
                data/labels/<drive>/canon=<h>/label=<h>/
                      │
                      ▼  forward chaining
                data/splits/<drive>/split=<h>/split_manifest.json
                      │
                      ▼
                runs/<experiment>/<drive>/<model>/seed<N>/fold<NN>/
```

각 파생 폴더에는 `provenance.json`(부모 hash, config, git commit, 환경)과
`_SUCCESS` 마커가 있다. **`_SUCCESS` 가 없는 폴더는 재사용하지 않는다.**
중간에 죽은 parquet 을 다음 실행이 조용히 읽는 사고를 막기 위한 것이다.

---

## fold 구조

```text
[-------- train --------][embargo][ val ][embargo][ test ]
                                                    1개월
```

`embargo` 가 없으면 train 표본의 라벨 구간 `(t, t+H]` 가 val/test 를 덮는다.
즉 학습이 평가 구간의 고장 사건을 이미 본 상태가 된다.
`embargo_days >= horizon_days` 를 강제하며, 위반하면 `forward.build()` 가
예외를 던진다. `tests/test_no_leakage.py` 가 이 조건을 검사한다.

`configs/split.yaml` 의 `embargo_days: null` 은 `horizon_days` 를 그대로 쓴다는 뜻이다.

---

## 실행

```bash
# 0) 최초 1회. PyYAML 만 추가로 필요하다.
pip install pyyaml

# 1) 파생 레이어 생성 (raw -> canonical -> features/labels -> split manifest)
python scripts/prepare_data.py --drive HGST_20HUH721212ALN604

# 2) 배선 점검 (LightGBM 1개, fold 1개, 표본 축소)
python scripts/run_experiment.py configs/experiments/smoke.yaml

# 3) 본 실험 (6개 모델 x 3 seed x 6 fold)
python scripts/run_experiment.py configs/experiments/model_comparison.yaml

# 3b) RODMAN 평가 방식 재현 (rolling 3개월 -> 다음 1개월, 고정 FAR 4%)
python scripts/run_experiment.py configs/experiments/rodman.yaml

# 4) 집계와 모델 선정
python scripts/aggregate_results.py model_comparison

# 테스트
python -m pytest
```

`--force` 를 주지 않으면 이미 만들어진 파생 데이터와 완료된 run 을 재사용한다.
중간에 끊겨도 같은 명령으로 이어서 돌린다.

---

## 확장하기

**드라이브 추가** — `configs/data.yaml` 의 `enabled` 를 `true` 로 바꾸고
실험 설정의 `drives` 에 이름을 넣는다. 코드 수정은 없다.

```yaml
drives:
  - name: ST12000NM0007
    file: ST12000NM0007.parquet
    enabled: true
```

**horizon 변경** — `configs/labeling.yaml` 의 `horizon_days` 를 바꾼다.
새 label hash 폴더가 생기고 기존 결과는 남는다. embargo 는 자동으로 따라간다.

**모델 추가** — `BaseModel` 을 상속한 클래스 하나와 `configs/models/*.yaml`
하나면 된다. `evaluation/` 과 `runner.py` 는 건드리지 않는다.

```python
class MyModel(BaseModel):
    family = "tabular"          # 또는 "sequence"
    def fit(self, train, val) -> dict: ...
    def predict_proba(self, data) -> np.ndarray: ...
    def _save_weights(self, directory): ...
    def _load_weights(self, directory): ...
```

```yaml
name: mymodel
family: tabular
class: hddpred.models.mine.MyModel
params: {}
training: {}
```

---

## 구조

```text
project_B/
├─ configs/
│  ├─ data.yaml            원본 위치, 드라이브 목록
│  ├─ preprocessing.yaml   raw -> canonical
│  ├─ labeling.yaml        horizon, 검열 처리
│  ├─ split.yaml           forward chaining, embargo
│  ├─ features.yaml        인과적 파생 피처, 스케일링, 학습 표본 추출
│  ├─ evaluation.yaml      지표, 임곗값 정책, 부트스트랩, 모델 선정 규칙
│  ├─ models/              xgboost lightgbm lstm gru tcn transformer
│  └─ experiments/         smoke, model_comparison
│
├─ data/                   파생 데이터. 지워도 재생성된다.
├─ runs/                   실험 artifact
├─ reports/                집계 결과 (csv, selection.json)
│
├─ src/hddpred/
│  ├─ paths.py             파일시스템 레이아웃을 아는 유일한 모듈
│  ├─ config.py            설정 로딩 + config hash
│  ├─ data/canonicalize.py         raw -> canonical (DuckDB)
│  ├─ labeling/horizon_labels.py   canonical -> labels
│  ├─ features/build.py            canonical -> 인과적 파생 피처
│  ├─ features/fold.py             fold 조립 + 스케일러
│  ├─ splits/forward.py            split manifest
│  ├─ models/{base,registry,trees,sequence}.py
│  ├─ inference/threshold.py       validation 기반 임곗값
│  ├─ evaluation/{metrics,aggregate}.py
│  ├─ experiments/runner.py        유일하게 순서를 아는 모듈
│  └─ tracking/provenance.py       계보 기록 + 완료 마커
│
├─ scripts/                prepare_data / run_experiment / aggregate_results
└─ tests/
```

---

## HGST 기준 실제 규모 (기본 설정)

```text
원본            26,914,795행 / 108컬럼 (전 컬럼 string) / 2018-07-12 ~ 2026-03-31
canonical       26,863,127행 / 11,370 디스크 / 고장 1,608개
                SMART raw 52개 중 결측 90% 미만 18개 보존
features        84개 (raw 18 + 나이 2 + 파생 8컬럼 x 8종)
                critical 중 smart_187 / smart_188 은 HGST 에 없어 자동 제외
labels (H=10)   26,423,670 표본 / 양성 14,704 (0.0556%)
splits          fold 6개, test = 2025-09 ~ 2026-02
                2026-03 은 라벨이 잘려 자동 제외됨
```

준비 단계 소요는 canonical 약 5.5분, features 약 1.2분, labels 약 12초다.

## 알아둘 것

**메모리.** train fold 행렬은 `행수 x 피처수 x 4바이트` 다. HGST fold 05 는
25.5M행 x 84피처 = 약 8GB. `fold.py` 는 Arrow record batch 로 스트리밍해서
미리 잡아둔 행렬에 채우므로 사본이 두 벌 생기지 않고, train 구간에서는
`serial_number` 문자열을 아예 읽지 않는다. 그래도 부족하면
`configs/features.yaml` 의 `train_sampling.strategy` 를 `stride` 로 둔다.
음성 표본만 n일 간격으로 뽑고(양성은 전량 유지) **SQL 단계에서 걸러내므로**
메모리에 올라오지도 않는다. **평가에는 적용되지 않는다.**
논문 표에 쓸 때는 이 설정을 Scope 항목으로 명시해야 한다 (Full / Partial).

**hive partition 컬럼.** 파생 경로가 `canon=<hash>/feat=<hash>/` 라서
`read_parquet(..., hive_partitioning=true)` 는 `canon` 과 `feat` 를 컬럼으로
만들어 낸다. 그래서 피처 목록은 기록된 parquet 을 DESCRIBE 해서 얻지 않고
생성 시점의 별칭에서 직접 만든다. `tests/test_features.py` 가 이걸 검사한다.

**결측.** segment 시작부의 diff/rolling 은 NULL 이다. 트리 모델은 결측을
그대로 처리하고, 시퀀스 모델은 raw 컬럼만 쓰므로 이 NULL 을 보지 않는다.
임의의 0 으로 채우면 "변화가 없었다"는 잘못된 신호가 된다.

**project_A 와 다른 점 하나.** project_A 는 중복 제거에서 문자열 상태로
`MAX()` 를 걸어 사전식으로 비교했다(`"9" > "10"`). project_B 는 캐스팅을
먼저 하고 수치로 비교한다.

**부트스트랩 단위는 디스크다.** 같은 디스크의 행은 독립이 아니므로 행 단위로
재표집하면 신뢰구간이 실제보다 좁아진다.
