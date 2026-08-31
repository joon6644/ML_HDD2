# 표 1의 `unit = HDD` 논문 재검토

표에서 **`unit = HDD`인 논문들**을 실제 원문 기준으로 뜯어보면, 표의 `HDD`가 같은 의미의 "HDD-level classification"은 아니다. 이게 꽤 중요하다.

확인한 것은 **RODMAN (2019), SMARTer (2020), LightGBM+CID (2021), StreamDFP (2023)** 네 편이다. 이 네 편만 봐도 서로 집계 방식이 상당히 다르다.

---

## 1. RODMAN (2019)

표에서 `HDD / Full / FPR`로 되어 있는데, **실제로는 HDD를 평가 대상으로 삼지만 모델 자체는 시간별 sample을 분류하는 구조**에 가깝다.

논문은 SMART log를 **disk별 daily sample**로 만들고, 각 sample에 failure label을 부여한다. 그리고 failed/healthy disk를 positive/negative sample로 구분한다.

특히 RODMAN은 중요한 labeling 문제를 다룬다. 기존 방식은

> 실제 failure가 발생한 시점만 positive

로 두는 경우가 많았는데, RODMAN은 **failure 이전의 pre-failure samples도 positive로 backtracking**한다. 즉 단순히 "현재 고장인가?"가 아니라 "곧 고장할 disk의 관측치인가?"를 positive로 만든다.

평가에서는

> **false positive rate가 특정 수준 이하일 때 몇 %의 failed disks를 예측했는가**

를 본다. 대표 결과도 **FPR 0.1%에서 Alibaba 데이터의 최대 92.8%, Backblaze에서 82.4%의 disk failure prediction rate**라는 형태다.

### 따라서 RODMAN의 `HDD`는

```text
daily SMART sample
       ↓
sample-level label
       ↓
model prediction
       ↓
여러 prediction을 disk failure detection 관점에서 집계
       ↓
"실패 HDD 중 몇 개를 잡았는가?"
```

에 가깝다.

즉 **HDD 단위 metric은 존재하지만, "HDD 하나당 하나의 독립적인 binary prediction을 만들어 confusion matrix를 구성했다"라고 보면 안 된다.**

---

## 2. SMARTer (2020)

이건 더 명확하다.

SMARTer는 **380,000 HDD**를 대상으로 하고, 목표는

> **앞으로 10일 이내에 disk failure가 발생할 것인가?**

이다. 논문은 명시적으로 정확히 언제 고장나는지를 예측하는 것이 아니라 **next 10 days 안에 failure가 발생할지를 예측**한다고 설명한다.

그리고 각 disk에 여러 시점의 observation이 존재한다.

```text
HDD A
t1 → sample
t2 → sample
t3 → sample
...
```

각 observation을 이용해서 **10-day horizon failure classification**을 수행한다. 결과는 F-measure와 MCC 등으로 평가한다.

여기서 중요한 점은 **HDD-level이라고 해서 "한 HDD = 한 prediction"이 아니라는 것**이다. 실제로 논문의 입력 구성 자체가 한 disk에 여러 readings가 있고, 고정 길이 observation period를 하나의 sample로 사용하는 구조다.

따라서 표의 `HDD`는

> **평가의 관심 대상이 HDD**

라는 의미로 이해하는 게 맞고,

> **모델이 HDD 하나당 단 하나의 binary output을 낸다**

라는 의미로 해석하면 안 된다.

---

## 3. LightGBM + CID (2021)

논문 제목부터 **Hard Disk Failure Prediction Based on LightGBM with CID**이고, 약 80,000개 HDD를 사용한다.

결과는 TPR 0.28 → 0.96, failure prediction advance **+1.2 days**, AUC, F1, TPR 등이다.

그런데 여기서도 **HDD 하나를 최종적으로 TP/FP/TN/FN 하나로 만든다는 의미의 HDD-level classification은 아니다.** 기본 label은 결국 **sample/window에 대한 failure prediction**이다.

개념적으로

```text
HDD
 ├─ day 1 → X1 → y1
 ├─ day 2 → X2 → y2
 ├─ day 3 → X3 → y3
 ...
 └─ day n → Xn → yn
```

이고, 그 prediction들을 이용해서 failure detection 성능을 계산한다.

따라서 표의 `HDD`를 **"HDD 단위로 최종 판정한 평가"라고 강하게 해석하면 안 된다.**

---

## 4. StreamDFP (2023)

**본 연구와 비교하기에 가장 중요하다.**

StreamDFP는 아예 disk-level prediction을 명시한다. 매일 각 HDD의 SMART vector $x_t^i$를 입력으로 받고 $\hat y_t^i$를 예측한다. classification이면 **해당 disk가 가까운 미래에 failure할 것인지**를 예측한다.

### Labeling

failure가 발생한 HDD를 찾으면, failure 전 **20일을 positive**로 라벨링한다.

```text
failure
    ↑
20일
<---------------->
      positive
```

구체적으로 classification에서는 $t' \in [t-D_L,\,t]$인 sample들을 모두 positive로 만든다. 기본 $D_L = 20$일이다.

### 그런데 평가가 핵심이다

StreamDFP는 매일 prediction하고, **30-day future horizon**을 기준으로 평가한다. 논문에서 명시적으로 각 dataset에서 처음 30일을 warm-up하고, 이후 매일 다음 30일의 disk failure를 예측한다.

classification metric은 다음과 같다.

- **Precision** — predicted failed disks 중 실제 failed disks의 비율
- **Recall** — 실제 failed disks 중 prediction된 disks의 비율
- **F1**

여기서 **FPR도 특이하게 정의**한다.

> 하루의 FPR = **false predicted failed disks / 다음 30일 동안 healthy한 disks**

즉 단순 row count가 아니라 **disk population을 분모로 한다.**

### 그런데 이것도 본 연구의 HDD-level operational evaluation과는 다르다

이게 가장 중요한 부분이다.

StreamDFP에서 하루 $t$의 prediction은

```text
HDD A → P
HDD B → N
HDD C → P
HDD D → N
...
```

이고, 그날의 **다음 30일 failure 여부**와 비교한다. 따라서 하루 단위로 보면

$$\text{HDD}_i,\ t \rightarrow \text{30-day binary prediction}$$

이다.

하지만 HDD A가 10일 동안 계속

```text
Day 1  P
Day 2  P
Day 3  P
...
Day 10 P
```

였다고 해서 "A라는 HDD에 대해 하나의 operational alarm이 발생했다"로 집계하지 않는다. **각 daily prediction을 계속 평가한다.**

---

## 정리

| 연구 | Unit | 실제 평가 방식 |
| --- | --- | --- |
| RODMAN | HDD | daily/sample prediction → disk failure detection rate |
| SMARTer | HDD | observation/window → 10-day failure classification |
| LightGBM+CID | HDD | sample/window → failure classification + advance days |
| StreamDFP | HDD | **daily HDD prediction → next-30-day failure classification** |

이 네 논문의 공통점은 **"HDD를 관심 대상/분모로 사용한다"**는 것이지,

> **"한 HDD의 전체 prediction history를 하나의 operational decision으로 집계했다"**

는 것이 아니다.

---

## 본 연구와의 차이

```text
기존 연구

HDD
 ├─ t1 → P/N
 ├─ t2 → P/N
 ├─ t3 → P/N
 ├─ ...
 └─ tn → P/N

        ↓

각 시점의 classification metric
```

```text
본 연구

HDD
 ├─ t1 → N
 ├─ t2 → N
 ├─ t3 → P  ← FIRST ALARM
 ├─ t4 → P
 ├─ ...
 └─ failure

        ↓

HDD-level operational event

        ↓

Detected / Missed
        +
First Alarm
        +
Lead Time
```

**특히 StreamDFP가 상당히 좋은 비교 대상이다.** 이미 **disk-level prediction**, **30-day horizon**, **healthy disk를 분모로 한 FPR**, **daily online prediction**까지 상당히 운영환경에 가까이 갔다. 그런데도 평가의 기본 단위는 여전히 **매일의 HDD prediction**이고, 동일 HDD에 대한 여러 prediction을 **"첫 alarm이라는 하나의 운영 이벤트"로 collapse하지 않는다.**

따라서 본 연구의 차별점은 이렇게 잡는 것이 정확하다.

> **기존 연구가 HDD를 평가 대상으로 삼더라도 반복적으로 생성되는 시점별 예측을 독립적인 분류 결과로 평가하는 반면, 본 연구는 동일 HDD의 예측 이력을 하나의 운영 사건으로 통합하고 최초 경보 시점과 failure 시점 사이의 lead time을 평가한다.**

이 표현이면 **"기존 연구는 HDD-level 평가를 안 했다"라고 과장하는 실수도 피할 수 있다.**

---

## 표 수정 권고

지금 표의 `unit`에 **HDD / Row / Window**를 섞어 쓰면 독자가 "평가 단위"와 "prediction sample 단위"를 혼동할 수 있다. 차라리 열을 **Prediction / aggregation unit**으로 명확히 하고 다음처럼 쓰는 것도 방법이다.

- RODMAN → **Disk-oriented**
- SMARTer → **Window/sample**
- LightGBM+CID → **Window/sample**
- StreamDFP → **Daily HDD**
- This work → **HDD-level event**

특히 **StreamDFP를 단순히 `HDD`라고 쓰는 것은 맞기는 하지만, 주장하려는 "HDD-level operational aggregation"과 혼동될 가능성이 높다.** 원문 확인 결과 StreamDFP는 매일 각 HDD의 다음 30일 failure를 예측하고 그 일별 결과를 precision/recall/FPR로 평가한다.

이 차이가 표에서 상당히 중요한 포인트다.

---

# 기존 연구를 어떻게 비판할 것인가

처음 우려했던 것은 다음이었다.

> **"일반적인 이진분류 confusion matrix를 그대로 HDD 단위에 적용하면 평가가 모순되지 않는가?"**

그런데 확인한 HDD-level 연구들은 대체로 **그 함정에 그대로 빠진 것이 아니라, 평가 문제를 prediction horizon이나 disk-level event의 정의에 맞게 변형해서 사용하고 있다.** 다만 연구마다 변형 정도가 다르다.

| 연구 | 평가 방식 | 판단 |
| --- | --- | --- |
| **RODMAN (2019)** | row/sample 예측을 disk failure detection 관점으로 집계 | 문제를 HDD failure detection에 맞게 변형 |
| **SMARTer (2020)** | "향후 10일 내 failure"라는 명시적 horizon classification | 일반적인 현재상태 분류가 아니라 horizon classification |
| **LightGBM+CID (2021)** | failure 전 일정 기간을 positive로 정의 + advance days 별도 평가 | 시간 정보를 label/metric에 반영 |
| **StreamDFP (2023)** | 매일 각 HDD에 대해 향후 30일 failure 여부를 판단 | daily HDD-level horizon evaluation |
| **This work** | HDD별 최초 alarm을 operational event로 집계 + LT | **반복 prediction을 하나의 운영 사건으로 통합** |

그래서 **"기존 연구들이 잘못된 confusion matrix를 사용했다"라고 공격하면 안 된다.**

## 기존 연구들도 문제를 알고 평가 단위를 조정했다

예를 들어 StreamDFP는 그냥 "이 row가 고장인가?"를 묻지 않는다. 명확하게

> **"현재 시점에서 이 HDD가 향후 30일 안에 failure할 것인가?"**

라는 prediction problem으로 바꾼다. 그러면 현재 시점의 `N/P`가 의미를 갖는다. 그리고 healthy disk를 기준으로 FPR을 계산하기 때문에 negative population도 확보한다.

$$\text{현재 observation} \rightarrow P(\text{failure within 30 days})$$

즉 **시간적 문제를 binary classification으로 재정의한 것**이며, 이건 충분히 정당한 평가 방식이다.

## 그럼 본 연구는 무엇이 다른가

기존 연구의 질문:

> **"각 시점에서 이 HDD가 가까운 미래에 고장할 것인가?"**

본 연구의 질문:

> **"이 HDD가 실제 운영 과정에서 failure 전에 적절한 시점에 최초 alarm을 받았는가?"**

이 둘은 비슷해 보이지만 평가 질문이 다르다. 예를 들어 어떤 HDD가

```text
failure까지

30d  N
20d  P
15d  P
10d  P
 5d  P
 0d  FAILURE
```

라면 기존 horizon classification에서는 여러 시점의 prediction이 각각 평가될 수 있다. 그러나 본 연구가 묻는 것은 "그래서 실제로 이 HDD를 언제 처음 잡았는데?"이고, 답은 $LT = 20$일이다.

즉 **prediction accuracy → operational detection event**로 평가 관점을 한 단계 바꾸는 것이다.

## 논문의 논리를 이렇게 잡는 것이 가장 안전하다

**약한 주장 (피할 것)**

> 기존 연구들은 HDD 단위에서 confusion matrix를 잘못 적용하였다.

이건 틀릴 가능성이 높다.

**권장**

> 기존 연구들은 failure prediction을 일정 prediction horizon 내의 binary classification 문제로 정의하고, 이에 맞추어 sample 또는 HDD-level prediction을 평가하였다. 이러한 방식은 각 시점에서의 failure-risk prediction 성능을 평가하는 데 적절하다. 그러나 동일 HDD에 대해 반복적으로 생성되는 예측을 실제 운영상의 단일 alarm decision으로 통합하고, 최초 alarm과 실제 failure 사이의 시간적 관계를 평가하는 데에는 한계가 있다.

이렇게 쓰면 **기존 연구를 인정하면서 본 연구의 필요성을 확보할 수 있다.**

## 이게 오히려 논문에 유리하다

기존 연구가 허술했다면 심사자가 "그냥 기존 연구도 그렇게 하면 되잖아?"라고 할 수 있다. 그런데 기존 연구들이 이미 prediction horizon, disk-level aggregation, failure window, FPR, days in advance 등을 적절하게 도입했다는 것을 인정하면, 본 연구의 질문을 더 명확하게 만들 수 있다.

> **"기존의 horizon-based classification은 각 prediction 시점의 failure-risk discrimination을 평가한다. 본 연구는 이와 다른 질문, 즉 실제 운영에서 하나의 HDD에 대해 최초 alarm이 발생했는지와 그 alarm이 failure에 앞서 얼마나 유효한 시간을 제공했는지를 평가한다."**

이렇게 되면 본 연구는 **기존 평가가 틀렸다는 연구가 아니라, 기존 평가가 답하지 않는 operational question을 정의하는 연구**가 된다. 이쪽이 방어력이 훨씬 높다.

---

# 표 1 전수 확인 기록

## 왜 이 절이 필요한가

SMARTer를 원문에서 확인했더니 표의 `Scope = Full`이 흔들렸다. 이 논문은 70일치 데이터에서 HDD를 가져오지만, 평가 표본 자체는 failure 상태를 배제한 **고정 길이 healthy 관측 구간**이다. `데이터셋의 전체 기간을 썼다`와 `각 HDD의 관측 이력 전체를 썼다`는 다른 말인데, 표의 Full은 후자를 뜻한다.

같은 함정이 나머지 열에도 있다. `Interval N d`는 논문마다 **레이블링 구간·prediction horizon·채점 구간** 중 다른 것을 가리키는데 표에는 숫자 하나만 남는다. StreamDFP는 레이블 20일에 horizon 30일이라 둘이 아예 다르다.

그래서 열 라벨을 눈대중으로 붙이지 않도록, 아래에 **판정 기준**과 **논문당 고정 질문 7개**를 둔다. 원문을 볼 때마다 이 양식만 채우면 표 1이 자동으로 결정되고, 나중에 심사자가 물어도 근거 문장을 바로 꺼낼 수 있다.

---

## 열 판정 기준

### Eval. unit — 성능 지표가 집계되는 단위

기준 질문: **혼동행렬의 칸 하나에 들어가는 것이 무엇인가?**

| 라벨 | 조건 |
| --- | --- |
| `Row` | 관측 행 하나가 하나의 표본 |
| `Window` | 고정 길이 시퀀스 하나가 하나의 표본 |
| `HDD` | HDD 하나가 하나의 표본, 또는 HDD-일 하나가 하나의 표본 |

주의: `HDD`는 **"HDD 하나에 최종 판정 하나"를 뜻하지 않는다.** SMARTer는 HDD당 표본 하나이고 StreamDFP는 HDD-일당 표본 하나인데 둘 다 `HDD`다. 이 구분은 Q1과 Q5에 기록한다.

### Scope — 관측 이력 중 평가에 사용한 구간

기준 질문: **각 HDD의 정상 구간이 평가에 그대로 들어갔는가?**

| 라벨 | 조건 |
| --- | --- |
| `Full` | 각 HDD의 관측 이력을 구간 제한 없이 평가에 사용 |
| `Partial` | 고장 전 일부 구간이나 고정 길이 윈도우만 사용 (정상 구간의 대부분이 빠짐) |
| `Balanced` | 양성·음성 비율을 인위적으로 맞춘 표본을 사용 |

**두 가지를 반드시 분리할 것.**

- 데이터 수집 기간 전체를 썼는가 ≠ 각 HDD의 이력 전체를 썼는가. **후자만 본다.** (SMARTer가 걸린 지점)
- 학습에서 음성을 언더샘플링했는가 ≠ 평가에서 그랬는가. **평가 집합만 본다.** 학습 쪽 샘플링은 Scope에 반영하지 않고 비고에 적는다.

### Verdict — 무엇을 탐지 성공으로 보는가

`Interval N d`로 적되, **세 가지 중 무엇인지 반드시 Q3에 남긴다.**

| 종류 | 뜻 |
| --- | --- |
| 레이블링 구간 | 학습에서 양성으로 만든 고장 전 구간 |
| Prediction horizon | 모델이 "앞으로 며칠 안"을 예측한다고 선언한 기간 |
| 채점 구간 | 평가에서 정답으로 인정하는 구간 |

표 1에 넣는 값은 **채점 구간**이다. 셋이 다르면 그 사실 자체를 2.2-2 본문에 쓴다(StreamDFP는 이미 그렇게 처리했다).

### Oper. metrics — 운영 관련 지표

정확도 계열(Accuracy, AUC, F1, MCC)은 여기 넣지 않는다. **오탐 부담**(FAR, FPR)이나 **시간 여유**(DPF, Days in advance)를 보고한 경우만 적는다. 없으면 `—`.

---

## 논문당 고정 질문 7개

| # | 질문 | 무엇을 정하는가 |
| --- | --- | --- |
| Q1 | 평가 표본 하나가 무엇인가 (행 / 윈도우 / HDD / HDD-일) | Eval. unit |
| Q2 | 각 HDD의 어느 구간을 평가에 썼는가 | Scope |
| Q3 | 양성 기준 구간 — 레이블링 / horizon / 채점 각각 며칠인가 | Verdict |
| Q4 | 음성 모집단이 무엇인가 (음성 행 전체 / 미고장 HDD / 향후 N일 healthy HDD) | 2.2-3 논지 |
| Q5 | 최초 Alarm 개념이 있는가, 같은 HDD의 반복 예측을 하나로 합치는가 | 2.2-5 신규성 |
| Q6 | 보고한 지표는 무엇인가 | Oper. metrics |
| Q7 | 고장이 관측되지 않은 HDD를 어떻게 처리했는가 | 3.2-2, 3.2-4 |

각 답에는 **원문 문장과 위치(절 번호나 페이지)** 를 같이 남긴다. 요약만 적어두면 나중에 다시 찾아야 한다.

---

## 값이 바뀌면 따라 움직이는 곳

| 바뀌는 것 | notion2에서 고쳐야 할 문장 |
| --- | --- |
| Scope 라벨 하나 | 2.2-3 본문 끝 "표 1에서 평가 범위를 Full로 둔 연구는 11편 중 5편이다" |
| Eval. unit 라벨 하나 | 2.2-2 본문 "평가 단위는 11편 중 6편이 Row 또는 Window, 5편이 HDD이고" |
| Verdict 구간의 성격 | 2.2-2 본문 "다만 그 구간의 성격은 서로 달라, RODMAN[12]은…" 이하 예시 |
| Oper. metrics | 2.2-4 전체 (DPF·Days in advance 비판) |
| Q5 답이 하나라도 "있다" | **2.2-5의 신규성 주장이 무너진다.** 최우선 확인 항목 |

---

## 기록 양식

```
### [n] 이름 (연도)

- 표 1 현재: unit / scope / verdict / oper
- 상태: 미확인 | 부분 확인 | 원문 확인
- 출처:

| # | 답 | 근거 문장 (위치) |
| --- | --- | --- |
| Q1 |  |  |
| Q2 |  |  |
| Q3 |  |  |
| Q4 |  |  |
| Q5 |  |  |
| Q6 |  |  |
| Q7 |  |  |

- 판정: unit=__ / scope=__ / verdict=__ / oper=__
- 표 1 수정 필요: 없음 | (무엇을 무엇으로)
- 비고:
```

---

## 개별 기록

### [2] Disk replacement (2016) — Botezatu, Giurgiu, Bogojeska, Wiesmann (IBM Research)

- 표 1 현재: `Row / Partial / Interval 10–15 d / —`
- 상태: **원문 전문 확인 완료**
- 출처: KDD 2016, pp. 39–48
- 데이터: Backblaze 50,984대, 2013.4–2015.6 중 **2014.1 이후 17개월만 사용**(그 이전은 SMART 70% 이상 미수집). Seagate ST4000DM000(SgtA)과 Hitachi HDS722020ALA330(HitA)

| # | 답 | 근거 문장 |
| --- | --- | --- |
| Q0 | Backblaze SMART. HDD 맞음 | §3.1 |
| Q1 | **HDD 1대 = 표본 1개로 보인다.** 아래 참조 | Table 1, §3.4 |
| Q2 | **K-means 기반 informed downsampling으로 균형화한 뒤 80/20 분할.** 아래 참조 | §2.3, §3.4 |
| Q3 | **속성별로 창 너비가 다르다.** 아래 참조 | §3.3 |
| Q4 | 균형화된 healthy disk | §2.3 |
| Q5 | **없음** | — |
| Q6 | Precision, Recall, F-score (replaced/healthy 각각). **FAR·FPR은 지표로 보고하지 않는다** | Table 3 |
| Q7 | **언급 없음.** 기간 내 교체되지 않으면 healthy | — |

#### ✅ Q3 해결 — 10–15일이 범위인 이유

표 1에서 유일하게 범위인 이유가 §3.3에 있다. **창 너비를 SMART 속성마다 따로 정한다.** changepoint가 관측된 시점 분포의 중앙값을 그 속성의 창 너비로 쓴다.

| 속성 | 창 너비 |
| --- | --- |
| SMART 1 raw (read error rate) | **4일** |
| SMART 197 raw (pending sectors) | **10일** |
| SMART 5 raw (reallocated sectors) | **12일** |
| SMART 187 raw (uncorrectable errors) | **15일** |
| SMART 7 / 240 raw (seek·transfer error) | **25일** |

초록과 결론의 "10-15 days in advance"는 이 중 대표적인 값을 요약한 표현이다. 즉 **레이블링 구간이 아니라 속성별 집계 창 너비**다. 레이블은 "이 디스크가 교체되었는가"라는 디스크 단위 결과다.

**2.2-2의 "표 1의 구간 값들은 서로 같은 개념이 아니며"의 세 번째 사례다.** [3]은 한 논문 안에서 갈렸고, [11]은 lead time이었고, [2]는 속성마다 다르다.

#### ✅ 전문 재확인 (2차) — 두 칸 모두 확정. 더 이상 추론이 아니다

**Eval. unit `HDD` 확정.** post-aggregation 수치가 디스크라는 증거가 셋이고 서로 독립이다.

| 증거 | 계산 |
| --- | --- |
| 교체 비율 | SgtA 457/(17,769+457) = **2.51%**, HitA 115/(4,616+115) = **2.43%** → §3.4 "only **2.5 to 3% of the disks** for both SgtA and HitA models are replaced" |
| 전체 대수 | post-aggregation 4개 모델 합 = 17,769+2,188+4,616+4,662+457+227+115+73 = **30,107** → 초록 "**more than 30000 disks**", §1 "a large population of disks (**>30000**)" |
| 주 실험 대수 | SgtA + HitA만 = 17,769+457+4,616+115 = **22,957** → §5 "the number of disks we consider is significantly larger, with **over 23000 drives**" |

세 수치가 전부 맞는다. **"Original" 열이 디스크가 아닌 것**(4개 모델 합 429,189 > 데이터셋 50,984)이고 캡션의 "disks"가 두 열에 뭉뚱그려 붙은 것이다. 원문 표의 흠이지 우리 판단의 흠이 아니다.

**Scope `Balanced` 확정.** 균형화된 집합이 평가까지 간다는 근거가 용어 사슬로 이어진다.

- §2.3 "we balance the training dataset ... clustering ... K-means"
- Algorithm 1 3단계가 균형 집합을 만들고 **4단계가 그것을 쓴다**
- §3.8 "Once we have the **balanced training dataset** ... (this is obtained by running **steps 1 through 3** in Algorithm 1)"
- §3.4 "We generate 100 random splits of **the dataset** into training (80%) and test (20%)" — 여기서 "the dataset"이 그 균형 집합이다
- SgtA는 healthy 1,000(K-means 100군집 × 10점) 대 replaced 457 ≈ **2.2:1**. 자연 비율은 39:1

수치로도 뒷받침된다. Table 3에서 **Replaced 클래스와 Healthy 클래스의 Precision이 0.98 대 0.99로 거의 대칭**이다. 39:1 불균형 테스트셋이었다면 소수 클래스 Precision이 이렇게 나올 수 없다.

#### ~~⚠ Eval. unit: `Row` → `HDD` (권고, 다만 원문 표에 모순이 있다)~~ — 위에서 확정됨

Table 1의 post-aggregation 수치가 근거다.

| | Healthy | Replaced | 교체 비율 |
| --- | ---: | ---: | ---: |
| SgtA | 17,769 | 457 | **2.51%** |
| HitA | 4,616 | 115 | **2.43%** |

이 비율이 본문의 **"only about 2% of disks are replaced"**, **"only 2.5 to 3% of the disks for both SgtA and HitA models are replaced"** 와 정확히 일치한다. 즉 **post-aggregation 수치는 디스크 수**이고, 각 디스크가 집계된 관측 하나로 압축된다.

§2.4도 같은 방향이다 — "$x_i$ is a multivariate temporal observation **aggregating information between time points $t_{i-k}$ and $t_i$**". 그리고 §3.4에서 healthy를 1,000개(SgtA)로 줄여 replaced 457개와 맞추는데, 이 숫자들이 디스크 단위여야 말이 된다.

**⚠ 다만 Table 1의 "Original" 열은 이 해석과 맞지 않는다.** SgtA healthy가 247,524인데 데이터셋 전체가 50,984대다. Original 열은 레코드 수이고 post-aggregation 열은 디스크 수인 것으로 보이나, 캡션은 둘 다 "disks"라고 쓴다. **원문의 표가 자체 모순이다.**

권고: 비율 일치가 강한 증거이므로 `HDD`. 다만 이 판단은 우리 쪽 추론이므로 기록해 둘 것.

#### ⚠ Scope: `Partial` → `Balanced` (권고, 추론)

§2.3이 K-means로 healthy 클래스를 대표 표본만 남겨 줄인다. §3.4에서 **SgtA는 1,000개, HitA는 500개**로 줄여 replaced(457, 115)와 맞춘다. 자연 비율 약 40:1이 **약 2:1**이 된다.

그리고 평가가 그 위에서 이뤄진다.

> "We generate 100 random splits of **the dataset** into training (80%) and test (20%), and for each such split, we train the model on the training set and evaluate it on the test set"

Algorithm 1의 3단계가 균형 데이터셋을 만들고 4단계가 그것을 쓰므로, 여기서 "the dataset"은 균형화된 집합이다. **테스트셋도 균형화되어 있다.**

수치로도 뒷받침된다. Table 3의 Replaced 클래스 Precision이 0.98인데, 자연 비율 2.5%에서 Precision 0.98은 극히 어렵다. 균형 집합(약 31% 양성)에서는 자연스럽다. 논문이 "98% accuracy"를 반복해서 내세우는 것도 균형 집합에서만 의미 있는 표현이다.

**2.2-3의 Balanced 논지에 딱 맞는다.** 자연 2.5%가 약 31%로 바뀌니 자릿수가 달라진다.

⚠ 다만 §2.3은 "we balance the **training** dataset"이라고 쓴다. 평가셋까지 균형화했다고 명시하지는 않는다. **`Partial` 유지도 방어 가능하다.** 어느 쪽이든 Full 3편은 안 바뀐다.

#### Oper. metrics `—` 확정

Precision·Recall·F-score만 보고하고 FAR·FPR을 지표로 쓰지 않는다. §3.6이 "it has low false alarm rate"라고 말하지만 수치가 없다. **현행 `—`가 맞다.**

#### 패턴 표에 또 하나 — 탐지율 대 lead time

§3.7 "Early vs. late replacement detection"에서 교체 1·3·10·30일 전 스냅숏으로 평가한다.

> SgtA: 3일 전 97%, 10일 전 92%, **30일 전 73%**
> "an administrator can identify 73 to 75% of the disks to replace **a month in advance**"

**앞당길수록 탐지가 떨어진다는 관찰이 [11]에 이어 둘째다.** 그런데 [2]도 이것을 별도 분석으로 두고 판정에는 넣지 않는다. Oper. 열이 `—`인 이유이기도 하다.

- 판정: unit=**`HDD`(권고)** / scope=**`Balanced`(권고) 또는 `Partial` 유지** / verdict=`Interval 10–15 d` / oper=`—`
- 표 1 수정 필요: **두 열 권고, 둘 다 추론에 근거하므로 판단 필요**

### [12] RODMAN (2019)

- 표 1 현재: `HDD / Full / Interval varies / FPR`
- 상태: **원문 확인 완료** (arXiv 공개본, 직접 확인)
- 출처: S. Han, J. Wu, E. Xu, C. He, P. P. C. Lee, Y. Qiang, Q. Zheng, T. Huang, Z. Huang, R. Li, arXiv:1912.09722. **[10] StreamDFP와 같은 연구진**

| # | 답 | 근거 문장 |
| --- | --- | --- |
| Q1 | 예측은 일별 행 단위: "It converts SMART logs into time-series samples, in daily granularity, for different disks." **다만 지표의 분모가 disk이므로 집계 단위는 HDD** | — |
| Q2 | **시간 분할.** "we set the training phase from July 2017 to April 2018 (10 months) and the testing phase for May 2018 (one month)." 테스트 한 달 안에서는 고장 직전 구간으로 제한하거나 균형화하지 않고 그대로 사용 | §평가 설정 |
| Q3 | Bayesian change point detection으로 데이터셋마다 자동 결정. 고장 상관 SMART 속성별 일수의 75분위를 구해 그 최댓값을 pre-failure period로 삼음. **A1은 29일, B1은 27일** | — |
| Q4 | **미고장 disk 전체.** "False positive rate (FPR): The ratio of the number of falsely predicted failed disks (which are indeed healthy) to the total number of healthy disks in one-month testing." | — |
| Q5 | **없음** (언급 자체가 없다) | — |
| Q6 | **TPR과 FPR 둘뿐.** "we focus on two metrics: True positive rate (TPR)... False positive rate (FPR)." | — |
| Q7 | **언급 없음** | — |

- 판정: unit=`HDD` / scope=`Full` / verdict=`Interval varies` / oper=`FPR` — **네 열 전부 현행이 맞음**
- 표 1 수정 필요: **없음**

#### Scope가 `Full`인 이유 (SMARTer와 갈리는 지점)

2.2-2의 정의는 "Full은 선택된 기간의 관측 이력을 그대로 사용함"이다. RODMAN은 **달력 기간**을 한 달로 제한하지만 그 안에서는 모든 관측을 쓴다. 고장 직전 구간으로 좁히지도, 양성·음성을 맞추지도 않는다. SMARTer는 **각 HDD의 이력**을 고정 길이 healthy 구간으로 잘라낸다. 제한하는 축이 다르다.

2.2-3의 논지("정상 구간이 평가에서 빠져 오탐 기회가 줄어든다")는 RODMAN에 해당하지 않는다. 미고장 disk가 한 달치 관측을 온전히 내고 오탐이 실제로 발생할 수 있다.

**다만 한 달이라는 점은 기록해 둘 것.** 각 disk가 약 30일씩만 기여하므로 3.2-6이 말하는 관측 구간 의존성이 이 논문에도 걸린다. 표 1의 라벨을 바꿀 사안은 아니다.

#### 2.2-2의 RODMAN 서술은 정확하다

현재 본문: "RODMAN[12]은 데이터셋과 실행 조건에 따라 달라지는 역추적 구간으로 레이블을 만들고 테스트는 별도의 시간 분할 구간에서 수행하며" — **원문과 일치한다.** 필요하면 29일·27일이라는 실제 값을 댈 수 있다.

#### 또 하나의 "구성요소는 이미 있다" 사례

RODMAN의 FPR은 **미고장 disk를 분모로 하는 disk 단위 오탐률**이다. 구조가 notion2의 식 (3) 운영 FAR과 같다. 그런데 최초 Alarm 개념이 없어 disk의 여러 예측을 하나로 합치지 않는다. [5]·[8]·Aggarwal과 같은 패턴이 하나 더 늘었다.

---

## [12] 전문 확인 후 추가 (2차)

전문을 다시 받아 세부를 확정했다. **소속은 CUHK(Patrick P. C. Lee) + Alibaba로 [10] StreamDFP와 같은 연구진이다.** 데이터는 Alibaba(300만 대 이상, 2017.7–2018.6)와 Backblaze(H1 = Hitachi HDS722020ALA330 18개월, S1 = Seagate ST4000DM000 40개월) 둘이다.

### Scope `Full` 유지 — 음성 표본 축소는 학습에만 걸린다

§V-A에 이 문장이 있어 하마터면 `Partial`로 갈 뻔했다.

> "To mitigate data imbalance ..., **our training** chooses the positive samples over the entire training phase, while **choosing the negative samples only on the last day observed**"

**"our training"으로 한정된다.** 테스트 지표는 "in one-month testing"의 disk 전체를 분모로 한다. 양식에 적어 둔 구분(학습 샘플링 ≠ 평가 샘플링)이 또 한 번 갈라냈다. **`Full` 유지.**

⚠ 다만 테스트 시점에 disk 단위로 어떻게 합치는지는 원문에 명시가 없다. 일별 예측을 내는데 지표 분모는 disk이므로 어떤 집계가 있어야 하는데, 그 규칙이 안 적혀 있다. 기록해 둘 것.

### Verdict `Interval varies` — 근거가 더 강해졌다

A1은 29일, B1은 27일이라는 값 외에 **Backblaze에서는 실행마다 크게 달라진다.**

> "The automated backtracking days vary from **2 to 58 days for H1** and from **3 to 40 days for S1** across different runs."

한 데이터셋 안에서 2일에서 58일까지 움직인다. `Interval varies`가 이보다 잘 맞을 수 없다.

### ✅ 2.2-5의 "시간 순 온라인 추론" 절에 직접 근거가 생겼다

§IV에 이렇게 적혀 있다.

> "**RODMAN focuses on offline learning** and assumes that the whole training dataset is available in advance, while we address **online learning on real-time data in future work.**"

가장 가까운 계열의 연구가 **온라인 추론을 스스로 후속 과제로 남긴다.** 2.2-5가 "HDD의 전체 관측 이력에 시간 순 온라인 추론을 적용하고"라고 쓴 부분의 인용 가능한 근거다.

### ⚠⚠ 3.2-4와 같은 기법이 이미 있다 — Observation Window

**이게 이번 확인에서 가장 중요하다.** §IV-C의 observation window가 우측 중도절단 문제를 정확히 지목한다.

> "we regard the disks that are not reported in the trouble tickets as healthy in the training dataset, but **the disks may fail right after the training phase** and their samples now become 'mis-labeled' as negative."
> "for healthy disks, their samples in the backtracking window are actually **ambiguous samples**, as they may fail right after the training phase"
> "we propose an **observation window** to **drop the ambiguous samples**"

그리고 notion2 3.2-4가 하는 일이 이것이다.

> "우측 중도절단 HDD는 관측 종료 시점 이후의 고장 여부를 확인할 수 없어 ... 따라서 **이 마지막 $H$일 구간을 평가에서 제외하고** 그 이전의 최초 Alarm만을 평가 대상으로 둔다."

**마지막 불확실 구간을 잘라낸다는 기법이 같다.** 목적은 다르다 — RODMAN은 학습 레이블 오염을 막으려고, 본 연구는 판정이 정의되도록(CE가 On-time이 될 수 없음을 보장) 자른다.

**그리고 결정적 차이는 그 다음이다.** RODMAN은 자른 뒤 나머지 healthy disk를 그냥 음성으로 쓴다. 중도절단이 판정에 들어가지 않는다. 본 연구는 자른 뒤 **CE / CN 두 범주로 나눈다.**

**대응 방향:** 3.2-4에서 마지막 $H$일을 자르는 것을 새 기여로 내세우지 말 것. 그건 선행 사례가 있다. 기여는 **자른 다음에 남은 중도절단 HDD를 두 판정 범주로 보존한 것**이다. 지금 3.2-4 본문이 이미 그렇게 쓰여 있다 — "원 프레임워크는 이벤트가 관측되지 않은 대상의 Alarm을 단일 False로 처리하나, 본 연구는 이를 두 범주로 나누어 관측되지 않은 미래를 정상으로 가정하지 않는다." **문장은 그대로 두면 된다.** 다만 심사자가 RODMAN을 알면 물어올 자리이므로 알고 있을 것.

### 게재본 없음 — arXiv 인용이 맞다

검색 범위에서 학회·저널 게재본이 확인되지 않는다. 같은 연구진의 [10] StreamDFP는 ICDCS 2020 → IEEE TC로 갔지만, 이 논문은 arXiv 프리프린트로 남아 있는 것으로 보인다. **서지 교체 불필요. 현행 유지.**

### [13] SMARTer (2020)

- 표 1 현재: `HDD / Full / Interval 10 d / —`
- 상태: **원문 확인**
- 출처:

| # | 답 | 근거 문장 (위치) |
| --- | --- | --- |
| Q1 | HDD 하나가 하나의 표본. 여러 시점 관측 $\{a_1,\ldots,a_n\}$을 하나의 sample로 취급 | (위치 미기록) |
| Q2 | **고정 길이 healthy 관측 구간.** 입력 시퀀스에 failure 상태가 들어가지 않도록 구성 | (위치 미기록) |
| Q3 | prediction horizon 10일. "detect if a given disk will fail within the next 10 days" | (위치 미기록) |
| Q4 | healthy disk. TN = healthy disk를 healthy로 예측 | J-Index 설명 부분 |
| Q5 | **없음.** Alarm이라는 평가 사건 자체가 없다 | — |
| Q6 | Precision, Recall, F-measure, MCC, FPR, FNR | — |
| Q7 |  |  |

- 판정: unit=`HDD` / scope=**`Partial`** / verdict=`Interval 10 d` / oper=`FPR`
- 표 1 수정 필요: **`Full` → `Partial`.** 그리고 Oper.가 지금 `—`인데 FPR/FNR을 보고하므로 **`FPR` 추가 검토.**
- 비고: 이 재분류로 2.2-3 본문의 "11편 중 5편"이 **4편**이 된다. 논지에는 유리하다(Partial 사례가 하나 늘어난다).

### [3] LSTM specificity (2020) — Hu, Han, Xu, Jiang, Qi

- 표 1 현재: `Window / Partial / Interval 15 d / FAR`
- 상태: **원문 전문 확인 완료**
- 출처: Procedia Computer Science 176, pp. 791–799 (KES 2020). 오픈액세스
- 데이터: Backblaze **ST4000DM000**(정상 35,187 / 고장 1,064)과 **ST8000DM002**(9,977 / 93), 12개월, SMART 24종 중 10종 선택

| # | 답 | 근거 문장 |
| --- | --- | --- |
| Q0 | Backblaze SMART. HDD 맞음 | §4.1 |
| Q1 | **지표가 drive 단위로 정의된다.** 아래 참조 | §4.3 |
| Q2 | 양성은 disk당 360일, 음성은 **마지막 180일**만. 게다가 음성을 언더샘플링해 비율을 약 1:4로 맞춤 | §4.1 |
| Q3 | **논문 내부가 어긋난다.** 아래 참조 | 초록, §3, §4.4 |
| Q4 | **미고장 drive 전체.** "FAR ... the proportion of good disks that are falsely predicted as failed" | §4.3 |
| Q5 | 최초 Alarm은 없다. 다만 **윈도우 안에 Alarm이 하나라도 있으면 해당 disk를 고장으로 본다** — "If there are alarm records in the sequence given by the prediction model, it is considered that the disk may fail next day" | §3.2 |
| Q6 | FDR(=recall), FAR, Precision, F1 | §4.3 |
| Q7 | **언급 없음** | — |

#### ⚠ Eval. unit: `Window` → `HDD` (권고)

§4.3의 지표 정의가 결정적이다.

> **FDR**: "the fraction of failed **drives** that are predicted correctly as failed"
> **FAR**: "the fraction of good **drives** that are mis-classified as failed ... the proportion of good **disks** that are falsely predicted as failed"

입력은 15일 sliding window지만, **지표는 drive 단위로 집계된다.** 2.2-2의 정의는 "평가 단위는 성능지표가 집계되는 단위"이므로 `HDD`가 맞다.

§4.1도 같은 방향이다 — "the failed **disk** is generally called a positive sample, and the healthy **disk** is called a negative sample."

**[6]과 대조하면 분명해진다.** [6]의 FAR은 "the proportion of normal **samples** that are incorrectly classified as failures"로 샘플 단위다. [3]은 drive 단위다. 겉보기에는 둘 다 sliding window 입력이지만 집계 단위가 다르다.

#### ⚠ Scope: `Partial` 유지 vs `Balanced`로 변경 — 판단 필요

**이 논문은 둘 다 한다.**

> "we use the down sampling method for the negative samples, **so that the ratio of negative samples to positive samples is maintained at about 1 to 4**. ... we set the sampling space of the positive sample to 360 days ... **The last 180 days of negative samples are taken as the sampling space for negative samples.**"

- **Balanced 근거**: disk 단위 자연 비율이 35,187 : 1,064 ≈ 33:1인데 약 4:1로 맞췄다. 8배 조정이다. 학습·평가를 나눠 적용했다는 언급이 없어 평가 집합에도 걸린다.
- **Partial 근거**: 미고장 disk는 **마지막 180일만** 쓴다. 12개월 데이터에서 절반을 버리는 것이고, 버려지는 앞쪽이 오탐이 날 수 있는 구간이다.

Scope 열은 값을 하나만 담으므로 선택이 필요하다. **`Balanced` 권고** — 비율을 의도적으로 바꾼 것이 평가 모집단을 더 크게 왜곡하고, 2.2-3의 두 논지 중 Balanced 쪽 사례가 지금 [6] 하나뿐이라 얇다. 다만 180일 절단도 본문에서 언급할 가치가 있다.

`Partial`로 두는 선택도 방어된다. 그 경우 Balanced 사례는 [6] 하나로 남는다.

#### ⚠ Verdict: 논문 내부가 어긋난다

| 위치 | 값 |
| --- | --- |
| 초록 | "predict a disk will fail in **next fifteen days** with an average precision of 86.31" |
| §3 | "we constrain such period into **30 days** before the disk failure" |
| §4.4 | "we finally choose a **time span of 15 days as the input** of the model" |

15일이 예측 구간인지 입력 윈도우 길이인지가 논문 안에서 갈리고, §3은 따로 30일을 말한다. §3.2의 구조("sliding window of $t$ days as the input ... return a sequence of length $t$")를 보면 $t$=15는 입력 길이인데, 초록은 그것을 예측 구간처럼 쓴다.

표 1의 `Interval 15 d`는 **초록의 표현을 따른 것이라 방어된다.** 그리고 이 혼란 자체가 2.2-2의 "표 1의 구간 값들은 서로 같은 개념이 아니며"를 뒷받침하는 사례다 — 한 논문 안에서도 갈린다.

- 판정: unit=**`HDD`(변경)** / scope=**`Balanced` 권고(변경) 또는 `Partial` 유지** / verdict=`Interval 15 d` / oper=`FAR`
- 표 1 수정 필요: **Eval. unit 변경 확정, Scope 판단 필요**

#### 또 하나의 "HDD 단위 집계 + 구간 판정" 사례

§3.2의 "윈도우 안에 Alarm이 하나라도 있으면 그 disk를 고장으로 본다"가 [8]의 map function(threshold 1)과 같은 구조다. **최초 Alarm의 시점은 쓰지 않는다.** 이 칸을 채우는 선행 연구가 [8]에 이어 둘이 되었다.

그리고 FAR이 미고장 **disk**를 분모로 하므로 [12] RODMAN과 함께 식 (3)과 구조가 같은 사례가 둘이다.

### [8] LightGBM + CID (2021)

- 표 1 현재: `HDD / Partial / Interval 10 d / Days in adv.`
- 상태: **원문 확인**

| # | 답 | 근거 문장 (위치) |
| --- | --- | --- |
| Q1 | **HDD.** 일별 예측 $y_1,\ldots,y_N$을 map function으로 HDD 단위 이진값으로 변환. threshold가 1이라 "as long as one day's hard disk data is predicted to be about to failure, we think that hard disk is about to fail" | Algorithm 1 |
| Q2 | **10일 time window 하나가 평가 단위.** "take the data $\{x_1,\ldots,x_N\}$ of a hard disk in the time window $N$ as a unit $X$" | (위치 미기록) |
| Q3 | 10일. "predict whether the hard disk will fail within the next ten days". 별도로 §IV-C1에서 **30일 extended window** 분석 (Fig. 4) | 결론, §IV-C1 |
| Q4 | HDD 단위 혼동행렬의 TN. $FPR = FP/(FP+TN)$ | — |
| Q5 | **부분적으로 있음.** $t_{predict}$ = "the time when the hard disk is **first** predicted to be failed in the time window". 최초 예측 개념은 있으나 **판정에 쓰이지 않고 사후 요약값 $\Delta interval$에만 쓰인다** | — |
| Q6 | TPR, FPR, Precision, Recall, F1, AUC / $\Delta interval = t_{failure}-t_{predict}$, SMART 7.75일 → SMART+CID+MA 8.95일 | Table IV |
| Q7 | 미확인 | |

- 판정: unit=`HDD` / scope=`Partial` / verdict=`Interval 10 d` / oper=`Days in adv.` — **현행 표 그대로 맞음**
- 표 1 수정 필요: **없음**

#### 2.2-4에 미치는 결과 — [8]에 대해서는 현행 본문이 맞다

| 2.2-4의 주장 | [8]에 대해 |
| --- | --- |
| "고장 시점과 최초 예측 시점의 간격" | **성립.** $\Delta interval = t_{failure}-t_{predict}$, $t_{predict}$는 최초 예측 시점 |
| "10일의 Prediction Horizon 아래에서 7.75일에서 8.95일로" | **성립.** Table IV의 수치와 정확히 일치 |
| "탐지에 성공한 HDD의 평균값으로" | **거의 성립, 단어 하나 주의.** 원문은 "we only analyzed the results of the failed disks here" — 즉 **고장이 확인된** HDD다. 여기서 다시 탐지 성공분으로 좁혔는지는 미확인 |
| "정의상 Prediction Horizon을 넘을 수 없다" | **성립하고 인용도 가능.** $t_{predict}$가 "in the time window" 안에서만 찾아지므로 window 길이가 곧 상한이다 |

**따라서 DPF와 Days in Advance를 한 문장에 묶은 것이 문제이지, 문장 자체가 틀린 게 아니다.** [8]에는 전부 맞고 [14]에는 앞의 둘이 안 맞는다. 두 지표를 갈라 쓰면 된다.

**"탐지에 성공한" → "고장이 확인된"으로 바꾸면 [8]과 [14] 양쪽에 다 맞는다.** 논지도 그대로다 — 모집단이 사후에 정해진다는 것이 요점이므로.

#### 새로 얻은 재료 — 값이 window에 따라 달라진다

[8]은 **같은 논문 안에서 두 개의 window를 쓴다.** 예측 문제는 10일이고, §IV-C1에서는 30일로 늘려 실패 HDD의 예측확률이 얼마나 일찍 오르는지를 본다.

$t_{predict}$가 window 안에서만 찾아지므로, **Days in Advance는 분석 window를 넓히면 커진다.** 즉 이 값은 모델이 운영에서 언제 울리는지의 성질이 아니라 **분석자가 얼마나 넓게 보기로 했는지**에 함께 좌우된다. 이건 추론이 아니라 그 논문이 두 window를 쓴다는 사실에서 바로 나온다.

"정의상 Horizon을 넘을 수 없다"를 이 형태로 바꾸면 인용 가능한 근거 위에 서게 된다.

#### [8]의 map function은 내가 앞서 "선행 연구에 없다"고 결론 낸 그 칸이다

이전 판단을 정정한다. 집계 단위는 HDD인데 판정은 구간 소속인 칸 — window 안에 하루라도 양성이면 HDD를 양성으로 보는 방식 — 을 [8]이 정확히 채우고 있다. Algorithm 1의 threshold 1이 그것이다.

따라서 3.1이나 3.2에 "집계 단위를 HDD로 올려도 격차는 최초 Alarm 판정에서 온다"는 문장을 넣을 경우, **그건 만들어낸 비교군이 아니라 [8]이 실제로 쓴 방식**이 된다. seed 42 캐시로 계산해 둔 값(48개 실행 전부에서 HDD 단위 집계가 행 단위보다 Recall이 높았고 최소 증가폭 0.036, HGST 5%에서 0.793 → 0.872 → 운영 0.068)이 그대로 근거가 된다.

- 비고: Q7 미확인. 다만 이 논문은 판정 단위가 window라 중도절단 처리가 없을 가능성이 높다.

### [4] Model update (2021) — Züfle, Erhard, Kounev

- 표 1 현재: `Row / Full / Interval 10 d / FAR`
- 상태: **원문 전문 확인 완료**
- 출처: ICMLA 2021, pp. 1379–1386. DOI 10.1109/ICMLA52953.2021.00223
- 데이터: Backblaze **STM4000DM000** 단일 기종(원문 표기 그대로), 2017.2.1–2020.12.31, HDD 35,170대 중 고장 2,246대

| # | 답 | 근거 문장 |
| --- | --- | --- |
| Q0 | Backblaze SMART. HDD 맞음 | §V |
| Q1 | **관측 행.** 지표가 instance 단위로 계산된다 — "when the data set is analyzed by measurement instance rather than HDD serial number" | §V |
| Q2 | **고장 HDD의 이력을 마지막 10일로 자른다.** 아래 참조 | §V |
| Q3 | 레이블링 = 채점 = **10일**. "predicting whether a particular HDD will fail within the next ten days" | §III-B |
| Q4 | 음성 instance 전체. $FAR = FP/(TN+FP)$ | 식 (1) |
| Q5 | **없음.** HDD 단위로 합치는 절차가 전혀 없다 | — |
| Q6 | FDR, FAR, 그리고 저자들이 새로 정의한 $\zeta = 2 \cdot FDR(1-FAR)/(FDR + 1 - FAR)$ | 식 (1)(2)(8) |
| Q7 | **언급 없음** | — |

#### ⚠ Scope: `Full` → `Partial` (확정)

결정적 문장이다.

> "For model learning and evaluation, we labeled all instances of each HDD that failed within the next ten days as positive **while discarding all older instances of those HDDs.** All other instances were labeled as negative."

**고장 HDD는 마지막 10일만 남기고 그 이전 이력을 통째로 버린다.** 2.2-2의 정의상 명백히 `Partial`이다.

그리고 이건 **2.2-3의 논지에 딱 맞는 사례다.** 버려지는 구간은 고장 HDD가 열화되어 가는 장기 정상 구간, 즉 모델이 오탐을 낼 가능성이 가장 높은 자리다. 그 구간이 빠지면 FP가 될 뻔한 instance가 분모와 분자에서 함께 사라져 **FAR이 실제보다 낮게 측정된다.** 2.2-3의 "False Alarm이 발생할 기회 자체가 감소하여"가 이 논문에서 그대로 일어난다.

#### 균형화는 학습에만 — `Balanced`가 아니다

불균형 비율 상한 $\gamma$를 두고 "good" instance를 언더샘플링하는데, **"we limit the number of 'good' instances in the training set"** 으로 학습 집합에 한정된다. 테스트 집합은 원래 분포를 유지하며 prevalence가 **0.0639%** 라고 명시한다.

양식에 적어 둔 구분(학습 샘플링 ≠ 평가 샘플링)이 실제로 갈라낸 사례다. `Balanced`가 아니라 `Partial`이 맞다.

#### 시간 분할이라는 점 (RODMAN과 같은 축)

첫 6개월 학습, 나머지 41개월을 월 단위 배치로 평가한다. 달력 기간을 자르는 것은 RODMAN과 같다. **그러나 [4]는 거기에 더해 고장 HDD의 이력까지 자른다.** 그래서 RODMAN은 `Full`이고 [4]는 `Partial`이다. 두 논문을 나란히 놓으면 Scope 기준이 무엇을 보는지가 분명해진다.

- 판정: unit=`Row` / scope=**`Partial`** / verdict=`Interval 10 d` / oper=`FAR`
- 표 1 수정 필요: **Scope `Full` → `Partial`**

#### 그 밖에 기록해 둘 것

- **FDR은 여기서도 Failure Detection Rate**(= TP/(TP+FN) = Recall)다. [5]·[6]에 이어 **세 번째**다. 이 분야의 관행이다.
- 주요 발견 (III): **"The higher the FDR achieved, the higher the FAR."** 행 단위 Recall과 FAR이 같이 오른다는 것으로, notion2 5.1의 단조성 관찰과 같은 현상이다. 인용할 필요는 없지만 우리 관찰이 특이한 게 아님을 보여준다.
- **[4]는 무작위 분할을 명시적으로 비판한다**: "In real-world applications, however, a temporal split is required because the prediction model must be learned on data that is already known." notion2 4.1-4는 HDD 단위 무작위 그룹 분할을 쓴다. 아래 별도 항목 참조.
- **결론 6에 쓸 수 있는 인용이다.** "재학습을 포함한 동적 운영 환경으로의 확장"이 바로 이 논문의 주제다. 지금 결론 6은 인용이 없다.

### [10] StreamDFP (2023)

- 표 1 현재: `HDD / Full / Interval 20 d / Days in adv.`
- 상태: **원문 확인**
- 출처:

| # | 답 | 근거 문장 (위치) |
| --- | --- | --- |
| Q1 | HDD-일 하나가 하나의 표본. 매일 각 HDD의 $x_t^i \to \hat y_t^i$ | (위치 미기록) |
| Q2 | 전체 스트림. 첫 30일 warm-up 후 매일 예측 | (위치 미기록) |
| Q3 | **레이블링 20일**($t' \in [t-D_L, t]$, $D_L=20$) / **horizon 30일** / 채점 = 다음 30일 고장 여부 | (위치 미기록) |
| Q4 | **다음 30일 동안 healthy한 disk.** 하루의 FPR = false predicted failed disks / 그 모집단 | (위치 미기록) |
| Q5 | **없음.** 같은 HDD가 10일 연속 P여도 하나의 alarm으로 합치지 않고 daily prediction을 계속 평가 | — |
| Q6 | Precision, Recall, F1, FPR | — |
| Q7 |  |  |

- 판정: unit=`HDD` / scope=`Full` / verdict=`Interval 20 d` / oper=**`FPR` 권고(아래 참조)**

---

## ⚠ [10] 전문 확인 후 정정 — 앞선 "30일" 권고는 틀렸다

전문(IEEE TC 72(2), 2023)을 받아 확인한 결과 **Verdict를 30일로 바꾸라고 한 권고가 오류였다.**

### 30일은 채점 구간이 아니라 버퍼 창이다

§4.3 원문:

> "it configures a **sliding time window, denoted by $W$, to buffer** the samples of a sufficiently long recent period (**30 days in our case**), as well as the number of extra labeled days before the disk failure occurs, denoted by $D_L$. If a failed disk is found, STREAMDFP labels the samples within $D_L$ before the failure as positive ... **By default, we set $D_L = 20$ days.**"

즉 **$W = 30$일은 버퍼 크기**이고 **$D_L = 20$일이 양성 레이블 구간**이다. Algorithm 3의 "if $W$ is full then Set $\hat{y}_t = M(x_t)$"가 앞서 "30일 warm-up"으로 알려진 것의 정체다 — 버퍼가 차기를 기다리는 것이지 예측 구간이 아니다.

§2.1의 "(e.g., **within the next 30 days**)"는 문제 설정을 설명하는 예시 표현이고, 실제 판정을 정하는 것은 $D_L$이다.

**따라서 표 1의 `Interval 20 d`가 맞다. 바꾸지 말 것.**

그리고 **2.2-2 본문의 서술이 정확하다** — "StreamDFP[10]는 30일 이내 고장을 예측하면서 고장 전 20일을 레이블링에 사용한다". 두 숫자를 그대로 구분해 적고 있다. 손댈 필요 없다.

### Oper.는 `Days in adv.` → `FPR` 권고

§1과 초록이 보고하는 지표는 **precision, recall, F1**이다. 여기에 §7의 FPR(미고장 disk 분모)이 더해진다.

**시간 여유를 성능 지표로 보고하지 않는다.** 관련된 것은 §4.5의 이 문장뿐이다.

> "for regression, ... the product $(1 - \hat{y}_t^i)(D_L + 1)$ **can be viewed as the predicted residual lifetime** of the disk."

**"can be viewed as"** — 회귀 출력의 해석이지 보고되는 평가 지표가 아니다. 게다가 §4.3이 스스로 "the failure probability defined here **only approximates** the likelihood ... and we pose the analysis of the true failure probability of a disk as **future work**"라고 적는다.

**❌ 위 추론은 전부 틀렸다. §7 원문으로 뒤집힘.**

## ✅ §7 확인 — [10]의 두 칸 최종 확정

§7 전문을 받아 확정했다. **이 항목에서 두 번 오판했으므로 경위를 남긴다.**

### Oper. = `Days in adv.` (현행이 맞음)

Table 7의 제목이 문자 그대로다.

> **TABLE 7: Days in Advance When a Failed Disk is Predicted** for Different Incremental Learning Algorithms in Exp#1
> "Table 7 further evaluates the **average number of days ahead of a disk failure** when the prediction is made (i.e., the duration from the day when a disk is predicted as failed to the day when the disk failure occurs). Here, we **only consider the failed disks that are correctly predicted.**"

값은 알고리즘별 9.1~14.6일이다.

**`FPR`로 바꾸라는 권고는 오류였다.** FPR은 §7.1에서 **운영점 제약**으로 쓰인다 — "we **fix the threshold** of the average false positive rate (FPR) ... we set the default FPR threshold as 1%". 보고되는 결과가 아니라 고정하는 조건이다. (구조가 본 연구와 같다 — 오탐률을 고정하고 나머지를 읽는다.)

### Verdict = `Interval 30 d` (현행 20일에서 변경)

§7.1이 채점 구간을 확정한다.

> "we first warm-up the prediction model from scratch using the first 30 days of samples, same as the length of $W$. **We then predict disk failures in the next 30 days on a daily basis**, and evaluate the accuracy for each day of prediction."
> "on each day, we compute the FPR as the fraction of falsely predicted failed disks over the total number of **healthy disks in the next 30 days**"

산술로도 맞는다. 데이터셋당 **460일**을 뽑고, 30일 warm-up, 마지막 30일은 예측 대상 구간으로 남겨 → 평가 기간 **400일**. $460 - 30 - 30 = 400$이다.

따라서 **레이블링 $D_L = 20$일 / 채점 30일**로 갈린다. 양식의 규칙("표에 넣는 값은 채점 구간")에 따라 **30일**이다.

**2.2-2 본문은 원래부터 정확했다** — "StreamDFP[10]는 30일 이내 고장을 예측하면서 고장 전 20일을 레이블링에 사용한다". 두 숫자를 정확히 구분해 적고 있다.

### 오판 경위 (같은 실수 반복 방지)

| 시점 | 판단 | 근거 | 결과 |
| --- | --- | --- | --- |
| 1차 | 20 → **30** | 사용자의 §7 요약 "30-day future horizon 기준 평가" | **맞음** |
| 2차 | 30 → **20** | §1~§6만 보고 "$W$=30은 버퍼, $D_L$=20이 레이블링"이라 판단 | **틀림 — §7 없이 과잉 정정** |
| 3차 | 20 → **30** | §7.1의 채점 정의 + 460/30/400 산술 | **확정** |

**교훈: 평가 절의 원문 없이 방법론 절만으로 채점 구간을 판정하지 말 것.** §4의 $D_L$은 학습 레이블이고 채점 기준은 §7에만 있었다.

### ✅ 2.2-4에 네 번째 사례 — 그리고 가장 명시적이다

> "Here, we **only consider the failed disks that are correctly predicted.**"

2.2-4의 "**탐지에 성공한 HDD의 평균값으로 보고되며**"가 [10]에 대해 **문자 그대로 정확하다.** [8]의 "we only analyzed the results of the failed disks here"(고장 HDD)보다 강한 진술이다.

앞서 "탐지에 성공한 → 고장이 확인된"으로 바꾸라고 권했는데, **그건 [14]를 함께 담기 위한 약화였다.** [10]을 사례로 쓰면 원래 표현이 그대로 맞는다. 문장을 [8]+[10] 중심으로 다시 짜면 강한 표현을 유지할 수 있다.

**Exp#2에도 같은 조건이 있다.** 회귀 평가의 ARE(잔여 수명의 평균 상대오차, 일 단위)에 대해 "we **exclude the falsely predicted failed disks** for this metric"이라 적는다. **시간 지표 둘 다 탐지 성공 조건부다.**

2.2-4의 사례가 이제 넷이다 — [8] Days in Advance, [14] DPF, Xu(2016) average time in advance, **[10] Days in Advance + ARE**.

### 패턴 표에 여덟째

**잔여 수명을 산출할 수 있다고 명시하고, 그것을 평가하지 않는다.** 회귀 출력이 $(1-\hat{y})(D_L+1)$로 잔여 수명이 된다는 것을 스스로 적어 놓고 성능 지표로는 precision/recall/F1/FPR만 쓴다. 그리고 진짜 고장 확률 분석은 후속 과제로 남긴다.

[8]·[14]·Xu(2016)·[11]·[2]·Aggarwal·Amram에 이어 **여덟 번째**다.

### 그 밖에 확인된 것

- **Scope `Full` 유지.** §3.2의 healthy 언더샘플링과 §4.4의 Poisson 다운샘플링은 전부 **학습**용이다. Algorithm 3의 예측 단계는 매일 전체 disk에 대해 $\hat{y}_t = M(x_t)$를 낸다.
- **Q5 최초 Alarm 없음 확정.** 매일 disk마다 예측하고 매일 채점한다. 하나의 Alarm으로 합치는 절차가 없다.
- 저자 소속: Peking University + CUHK(Patrick P. C. Lee) + Xiamen University + Alibaba. [12] RODMAN과 연구진이 겹친다.
- 비고: **가장 가까운 선행 연구.** 2.2-5에서 명시적으로 지목한다. Q5가 "없음"이라는 것이 신규성 주장의 핵심 근거이므로 원문 문장을 반드시 확보할 것.

### [5] Survival deepnet (2024) — Ahmed & Green

- 표 1 현재: `Row / Full / Interval 15 d / FAR`
- 상태: **원문 확인**
- 출처: Neural Computing and Applications, https://link.springer.com/article/10.1007/s00521-024-10479-6
- 데이터: Backblaze ST4000DM000, 2013–2022, 37,037대

| # | 답 | 근거 문장 (위치) |
| --- | --- | --- |
| Q1 | 관측 행 하나. 평가식이 TP/FP/TN/FN으로 정의된 관측별 이진분류 | §3.5 |
| Q2 | 10년 전체 기간. 다만 KM 기반 informative sampling은 **학습 쪽**에 적용 | (위치 미기록) |
| Q3 | 레이블링 = 채점 = **고장 15일 이내 모든 관측을 failure로 labeling**. 셋이 일치 | (위치 미기록) |
| Q4 | 음성 관측 전체. FAR = FP/(FP+TN) | §3.5 |
| Q5 | **없음.** 최초 prediction을 쓴다거나 HDD별로 하나의 prediction으로 합치는 절차가 제시되지 않음 | — |
| Q6 | Precision, Recall, FAR, FDR, G-mean / 최종 비교는 **G-mean, FDR, FAR, ROC AUC** | §3.5, 실험 |
| Q7 | **우측 중도절단을 명시적으로 처리한다. 단 survival analysis 쪽에서만이다.** KM으로 생존분포 추정 → informative sampling, Cox로 SMART와 생존시간의 hazard ratio 분석. **CNN의 혼동행렬에는 들어가지 않는다.** | — |

- 판정: unit=`Row` / scope=`Full` / verdict=`Interval 15 d` / oper=`FAR` — **현행 표 그대로 맞음**
- 표 1 수정 필요: **없음**
- 비고:
  - 이름의 `survival`에 속으면 안 된다. **생존분석은 데이터 구성과 수명 분석에 쓰였고, 예측모델 평가는 별도의 이진분류다.** 모델 자체도 1D CNN에 sigmoid 출력.
  - `FDR`은 여기서 **Failure Detection Rate**(= TP/(TP+FN))이다. False Discovery Rate가 아니다. 나중에 헷갈리지 말 것.
  - **Q7의 답이 2.2-5에 쓸 수 있는 가장 좋은 재료다.** "중도절단을 다룬 연구가 없다"는 거짓이고 이 논문이 반례다. 참인 것은 더 좁다 — 중도절단을 **판정에** 넣은 연구가 없다는 것. 이 논문은 중도절단을 알고, 모델링하고, 표본 구성에까지 쓰면서도 평가 판정에서는 뺀다. 경계가 어디에 있는지를 보여주는 사례다.
  - 저자들이 기존 HDD 연구의 "misleading evaluation measures"를 문제로 지적하면서 정작 자신들의 CNN 평가는 전통적 이진분류 혼동행렬로 수행한다. 2.2의 논지에 쓸 만하다.
  - **남은 확인 하나**: informative sampling이 학습 집합에만 적용된 게 맞는지. 평가 집합까지 KM 가중으로 뽑았다면 `Full`이 무너진다.

**Q7에서 일반화되는 구분** (다른 논문에도 적용할 것): 중도절단을 *안다/모델링한다/표본 구성에 쓴다*와 중도절단이 *판정에 들어간다*는 서로 다른 층위다. 앞의 셋은 선행 연구에도 있으므로, Q7의 답은 항상 **어느 층위인지**까지 적는다.

### [14] DFPoLD (2025) — Wei et al.

- 표 1 현재: `Row / Partial / Interval varies / DPF`
- 상태: **원문 확인**
- 출처: MDPI Informatics 12(3):73, https://www.mdpi.com/2227-9709/12/3/73
- 데이터: Backblaze ST4000DM000

| # | 답 | 근거 문장 (위치) |
| --- | --- | --- |
| Q1 | 관측 행. "disk-level predictions"라는 표현을 쓰지만 DPF 실험이 560대에 87,995 관측이므로 대당 약 157행 | §6.4, Table 12 |
| Q2 | **고장 HDD는 마지막 $y$일, 정상 HDD는 최초 $y$일만 선택.** 나머지는 잘라냄 | (위치 미기록) |
| Q3 | $y = 17e^{0.63x}$ ($x$ = 결측률). 용도는 **데이터 truncation + label reset**이지 채점 구간이 아니다. 별도로 1~30일 window를 바꿔가며 TPR/FPR/AUC/F1을 비교하는 실험이 있음 | (위치 미기록) |
| Q4 | 정상 HDD의 최초 $y$일 관측 (label 0) | (위치 미기록) |
| Q5 | **없음** | — |
| Q6 | 일반 평가: TPR (FPR < 0.1%), FPR, AUC, F1. LightGBM objective=`binary`, metric=`binary_error`. **DPF는 별도 실험** | §6.4 |
| Q7 | **DPF 실험 모집단에 미고장 HDD가 없다.** 2023년 실제 고장 560대만 사후 추출 | §6.4 |

- 판정: unit=`Row` / scope=`Partial` / verdict=`Interval varies` / oper=`DPF` — **현행 표 그대로 맞음**
- 표 1 수정 필요: **없음**

#### DPF 실험의 실제 구조 (§6.4, Table 12)

2022년 데이터로 학습한 모델을, 2023년에 실제로 고장난 ST4000DM000 **560대의 전체 시간 데이터 87,995 관측**에 적용한다. 이 실험에서는 time window를 **10일로 통일**한다. 논문이 말하는 이상적 결과는 "all data are predicted as 1"이다.

| Data removal | Positive predictions | Positive prediction probability | DPF |
| ---: | ---: | ---: | ---: |
| 0% | 86,606 | 0.9842 | 9.84 |
| 50% | 86,736 | 0.9857 | 9.86 |
| 80% | 85,824 | 0.9753 | 9.75 |

**표에서 읽히는 것 (추론이지 원문 인용이 아님):** DPF = positive prediction probability × 10. 세 행 모두 소수점 둘째 자리까지 일치한다(0.9842×10=9.842, 0.9857×10=9.857, 0.9753×10=9.753). 즉 DPF는 **10일 window 안에서 몇 %의 시점이 양성으로 예측되었는지**를 일수로 환산한 값이지, 최초 예측과 고장 사이의 간격이 아니다.

#### 2.2-4에 미치는 결과

현재 2.2-4 본문의 네 주장 중 **둘이 [14]에 대해 성립하지 않는다.**

| 2.2-4의 주장 | [14]에 대해 | 비고 |
| --- | --- | --- |
| "고장 시점과 최초 예측 시점의 간격이라는 점에서 본 연구의 Lead Time과 형태가 같으나" | **틀림** | DPF는 간격이 아니라 window 내 양성 예측 밀도다 |
| "탐지에 성공한 HDD의 평균값으로 보고되며" | **틀림** | 모집단은 실제 고장 560대 전부이지 탐지 성공분이 아니다 |
| "사후적으로 탐지에 성공한 HDD에 대해서만 계산할 수 있으므로 Alarm 발생 시점에는 해당 Alarm이 이 집단에 속하는지를 알 수 없다" | **성립** | 다만 조건이 *탐지 성공*이 아니라 *실제 고장*이다. 이유를 바꿔야 함 |
| "탐지로 인정되는 구간 밖의 예측은 계산 대상에서 제외되므로 정의상 Prediction Horizon을 넘을 수 없다" | **성립, 오히려 더 강하게** | 구간 밖 데이터가 모델에 들어가기도 전에 잘려 나간다 |

**주의:** 위의 `DPF = 확률 × 10`은 Table 12를 역산한 결과다. 원문에 그 정의식이 인용 가능한 형태로 있는지 확인되지 않았다. **이 역산을 근거로 논문에서 비판을 세우면 안 된다.** 반박당했을 때 인용할 문장이 없다. 비판은 원문에 명시된 것 — 560대 사후 추출, 10일 truncation — 위에서만 세울 것.

- 비고: [8]의 Days in Advance는 아직 같은 수준으로 확인되지 않았다. **DPF와 Days in Advance를 한 문장에 묶어 같은 성질이라고 말하면 안 된다.** 지금 2.2-4가 그렇게 하고 있다.

### [11] Explainable TS (2025) — Li, Zhou, Radhakrishnan, Kamarthi

- 표 1 현재: `HDD / Partial / Interval 5 d / FAR`
- 상태: **원문 전문 확인 완료**
- 출처: Engineering Applications of Artificial Intelligence 152, 110674 (Northeastern University). 오픈액세스
- 데이터: Backblaze **ST4000DM000**, 2022 Q1–Q3. 18,802대 중 고장 471대. **Q1+Q2 학습 / Q3 테스트**

| # | 답 | 근거 문장 |
| --- | --- | --- |
| Q0 | Backblaze SMART. HDD 맞음 | §4.1 |
| Q1 | **HDD.** "we structured raw time series data in a tabular form, **each row representing a hard drive**" | §3.2 |
| Q2 | **각 HDD가 32일 구간 하나만 기여한다.** 아래 참조 | §4.2 |
| Q3 | signal length 32일 + lead time 5일. **5일은 horizon이 아니라 버퍼다.** 아래 참조 | §4.2, §5.1 |
| Q4 | **미고장 drive 전체.** "FAR is the ratio of the number of operational drives incorrectly identified as failures to the total number of actual operational drives" | 식 (5) |
| Q5 | **없음.** HDD당 예측이 하나뿐이라 Alarm 시점이라는 개념이 성립하지 않는다 | — |
| Q6 | FDR, FAR (74.7% / 0.73%) | §5 |
| Q7 | **언급 없음.** 벤치마크로 optimal survival tree를 쓰면서도 중도절단을 평가에 넣지 않는다 | §5.3 |

#### Scope `Partial` 확정 — 근거 문장 확보

이전에 `Segments`에서 `Partial`로 바꾼 것이 **맞다.** 근거는 이것이다.

> "In data centers, it is impractical to store the lifetime SMART data for every hard drive. Therefore, we designed our failure detection model to **utilize only a moving window of the latest data**."

signal length 32일, lead time 5일로 각 drive에서 구간 **하나**를 잘라낸다. 고장 drive는 고장 5일 전에서 끝나는 32일, 미고장 drive는 현재 시점 5일 전에서 끝나는 32일이다.

**2.2-3의 교과서적 사례다.** 고장 drive의 장기 정상 구간이 통째로 빠지고, 미고장 drive도 마지막 32일만 남는다. 6개월 전에 오탐을 냈을 drive가 평가에 한 번도 등장하지 않는다.

언더샘플링은 학습에만 걸린다 — "We randomly undersampled **training data** of the operational hard drives". 테스트셋은 고장 비율 2%(3,717대 중 약 75대)로 자연 분포를 유지하므로 `Balanced`가 아니다.

#### ⚠ Verdict의 `Interval 5 d`는 성격이 다르다

[11]의 5일은 **lead time**이며 "고장 N일 이내"라는 horizon이 아니다.

> "Lead time is the number of days that cover the buffer period between the end of the data window and the failure event for a failed drive."

즉 입력 구간이 고장 5일 전에서 끊긴다는 뜻이고, 레이블은 "이 drive가 연구 기간에 고장났는가"라는 drive 단위 결과다. 다른 논문의 `Interval N d`가 "고장 전 N일 안의 관측을 양성으로 본다"인 것과 구성이 다르다.

표 1에 `Interval 5 d`로 둔 것 자체는 방어된다(그 논문에서 판정에 쓰인 유일한 시간 값이다). 다만 **2.2-2가 이미 "표 1의 구간 값들은 서로 같은 개념이 아니며"라고 적어 둔 것의 또 하나의 사례**다. [3]에 이어 둘째다.

#### 또 하나의 "구성요소는 이미 있다" 사례 — 이번엔 Lead Time

§5.1에서 **lead time을 0~35일, signal length를 2~50일로 바꿔가며 1,764조합을 탐색한다.** 그리고 결론이 이렇다.

> "the FDR **decreases** as we predict more in advance"

> "the sensitivity analysis optimizes the signal length and the lead time to improve prediction accuracy and **inform predictive maintenance policies**"

**Lead Time을 정책 변수로 다루고, 앞당길수록 탐지가 나빠진다는 것까지 관찰한다.** notion2 5.1의 관찰과 방향이 같다.

그런데 그 lead time은 **데이터 준비 파라미터**이지 판정 기준이 아니다. 판정은 여전히 drive 단위 이진 분류이고, Oper. 열이 `FAR`이지 시간 지표가 아닌 이유도 그것이다. lead time은 성능 지표로 보고되지 않는다.

패턴 표에 한 줄 더 들어갈 사례다 — **Lead Time조차 선행 연구에 있으나 판정에 들어가지 않고 전처리 파라미터로만 쓰인다.**

- 판정: unit=`HDD` / scope=`Partial` / verdict=`Interval 5 d` / oper=`FAR` — **네 열 전부 현행이 맞음**
- 표 1 수정 필요: **없음**
- 비고: **FDR = Failure Detection Rate가 여기서도 그렇다. [5]·[6]·[4]에 이어 네 번째.**

### [6] TCN-LSTM-Attn (2026) — Li, Ma, Ren

- 표 1 현재: `Window / Balanced / Interval 7 d / FAR`
- 상태: **원문 전문 확인 완료 — 표 1 그대로가 맞다. 수정 없음**
- 출처: Li, Ma, Ren (CETC 32nd Research Institute), ISCTIS 2026, pp. 261–265. DOI `10.1109/ISCTIS70043.2026.11572580`
- 데이터: **Backblaze SMART, ST4000DM000 (Seagate) 단일 기종, 2015–2025, 약 8,600만 일별 레코드**

#### 결론 — 네 열 전부 현행이 맞다

| 열 | 현재 | 원문 근거 |
| --- | --- | --- |
| Eval. unit | `Window` | "sliding time windows with a length of 30 days are constructed to generate sequential samples", 샘플 단위 이진 분류 (§III-A, §IV-A) |
| Scope | `Balanced` | TimeGAN 증강 + majority random undersampling → **180,000 samples, 양성:음성 ≈ 1:8**. "All models are trained and **evaluated** on the dataset constructed using the preprocessing method described in Section III-C" (§IV-C) |
| Verdict | `Interval 7 d` | "predict whether a failure will occur within the next 7 days" (§III-A) |
| Oper. | `FAR` | "FAR is employed to measure the false alarm rate, indicating the proportion of normal samples that are incorrectly classified as failures" (§IV-B) |

**2.2-3의 근거로 쓰기에 이상적이다.** 1:8이면 양성 11.1%인데, 표 2의 실제 양성 레이블 비율은 0.083~0.181%다. **약 60~130배**다. "양성 비율이 실제 운영과 자릿수 단위로 달라진다"가 정확히 이 사례다.

#### ⚠ 다만 [6]의 수치는 인용하지 말 것

Table I이 내부적으로 정합하지 않는다. Precision·FDR·FAR을 표준 정의로 두고 각 행이 함의하는 테스트셋 양성:음성 비를 역산하면 이렇게 나온다.

| Method | Precision | FDR | FAR[%] | FAR이 함의하는 양성:음성 |
| --- | ---: | ---: | ---: | ---: |
| ORF | 0.90 | 0.82 | 0.06 | 1 : 152 |
| LSTM | 0.67 | 0.77 | 0.71 | 1 : 53 |
| TCN | 0.90 | 0.84 | 0.11 | 1 : 85 |
| Proposed | 0.93 | 0.91 | 0.21 | 1 : 33 |

논문이 명시한 구성은 **1 : 8**이다. 네 행이 서로도 안 맞고 명시된 구성과도 안 맞는다. (FAR을 백분율이 아니라 분수로 읽어도 해소되지 않는다.)

**따라서:** `Balanced` 라벨은 논문 본문의 서술에 근거하므로 유지한다. 그러나 **[6]의 성능 수치를 본문에 인용하면 안 된다.** 그리고 이 역산은 우리 쪽 계산이므로 논문에 쓰지 말 것.

#### 그 밖에 확인된 것

- **Q5 최초 Alarm: 없음.** HDD별로 시점을 순차 추론해 최초 Alarm을 찾거나 HDD 단위 판정을 내리는 절차가 전혀 없다. 30일 윈도우 샘플 각각이 분류 대상이다.
- **Q7 우측 중도절단: 언급 자체가 없다.** 고장 없이 관측이 끝난 HDD를 어떻게 다루는지 논문에 나오지 않는다.
- FDR = Recall이라고 논문이 직접 정의한다("FDR, equivalent to Recall"). False Discovery Rate가 아니다. [5]와 같은 함정.
- 합성 표본 처리는 서술이 엇갈린다. "TimeGAN-generated synthetic samples are incorporated into the **training data**"라고 두 번 말하지만, 1:8의 180,000 샘플을 만든 뒤 7:3으로 나눈다고도 한다. undersampling은 어느 쪽으로 읽어도 테스트셋에 걸린다.

#### 확인 A는 오류였다 (기록)

직전 확인에서 "중국 노천 구리광산 haul truck 센서 데이터, 4개 health state, SMOTE"라고 보고된 내용은 **이 논문과 무관하다.** 원문 전문에 그런 내용이 없다. 다른 논문의 정보가 섞여 들어온 것으로 보인다.

**교훈:** 2차 자료로 원문을 대신하지 말 것. [6]에서만 두 번 어긋났고(확인 A는 통째로, 확인 B는 8~10절이 [8]에서 유입), 둘 다 그럴듯한 형식을 갖추고 있었다. 남은 미확인 논문은 원문을 직접 볼 것.

#### 아래 「충돌 상태」 이하는 해소 전 기록이다 (무효)

두 차례 확인이 서로 배타적인 답을 냈다. 같은 논문일 수 없다.

| 항목 | 확인 A | 확인 B |
| --- | --- | --- |
| 데이터 | 중국 노천 구리광산 haul truck·hydraulic excavator 센서 | Backblaze HDD SMART |
| 규모 | 12개 장비, 18개월, 847 fault event | 180,000 samples |
| 문제 정의 | 4개 health state (Normal 68.3 / Minor 15.2 / Severe 11.8 / Imminent 4.7%), softmax | 이진 분류, next 7 days |
| 입력 | — | 30일 sliding window |
| 균형화 | Normal을 40%로 undersampling + SMOTE | TimeGAN + undersampling, 양성:음성 ≈ 1:8 |
| split | — | 7:3 |
| 지표 | accuracy, macro-F1, 4-state confusion matrix | Precision, F1, FAR, FDR(=Recall) |

**확인 B는 최소한 일부가 [8]에서 온 것으로 보인다.** 근거:

- `t_{predict}` 정의, $\Delta interval = t_{failure}-t_{predict}$, "we only analyzed the results of the failed disks here", "we have extended the time window to 30 days", III-C의 map function — 전부 직전 턴에 [8]로 확정한 내용과 일치
- 결정적으로 **"SMART + CID + MA: 8.95 days"** 의 CID는 [8]의 기법 이름이다
- 확인 B 자체가 내부 모순을 갖는다: 1절은 "next 7 days", 10절은 "next ten days"

따라서 확인 B의 8~10절은 [8]의 내용이고, 1~7절의 출처도 보장되지 않는다.

#### 가르는 질문

원문 실험 절에서 **데이터가 Backblaze SMART인가 광산 장비 센서인가** 하나면 결판난다. 이진 분류인지 4-state인지도 같이 확인된다. 그리고 **참고문헌 [6] 서지 항목이 이 DOI를 가리키는지**도 볼 것.

#### 확인 B가 맞을 경우 — 표 1은 그대로가 맞다

| 열 | 현재 | 확인 B 기준 |
| --- | --- | --- |
| Eval. unit | `Window` | ✓ 30일 sliding window sequence sample |
| Scope | `Balanced` | ✓ **오히려 근거가 강해진다.** "All models are trained and evaluated on the dataset constructed using the preprocessing method described in Section III-C" — 평가 집합까지 balancing을 거쳤다는 뜻 |
| Verdict | `Interval 7 d` | ✓ next 7 days |
| Oper. | `FAR` | ✓ |

그리고 2.2-3의 "양성·음성 비율을 맞춘 표본을 사용하면 양성 비율이 실제 운영과 자릿수 단위로 달라진다"가 **정확히 뒷받침된다.** 1:8이면 양성 11%인데 표 2의 실제 양성 레이블 비율은 0.083~0.181%다. 약 100배 차이다.

확인 B가 맞다면 미확인으로 남는 것은 따로 있다 — HDD 단위 split인지 sample 단위 random split인지, 시간 순인지, **test set에 TimeGAN 합성 표본이 남아 있는지.** 마지막 것은 사실이면 별개의 문제다.

#### ❌ 아래 「제외 사유」와 「따라 움직이는 곳」은 오류인 확인 A에 근거한 것이다 — 전부 무효

**[6]은 표 1에 남는다. `11편 → 10편`, `Balanced` 범주 삭제, 2.2-3 문장 삭제, 인용 세 곳 수정 — 어느 것도 하지 말 것.** 아래 내용은 삭제하지 않고 남겨 두되 실행 대상이 아니다.

| # | 답 (원문 전문 기준) | 근거 문장 (위치) |
| --- | --- | --- |
| Q0 | **Backblaze SMART, ST4000DM000, 2015–2025, 약 8,600만 레코드** | §IV-A |
| Q1 | 30일 sliding window로 만든 sequence sample 하나 | §III-A, §IV-A |
| Q2 | TimeGAN 증강 + majority undersampling으로 구성한 180,000 샘플(1:8)을 7:3으로 분할. 평가도 이 데이터셋 위에서 수행 | §IV-A, §IV-C |
| Q3 | 레이블링 = 채점 = **next 7 days**. 셋이 일치 | §III-A |
| Q4 | 균형화된 정상 샘플. FAR = 정상 샘플 중 고장으로 오분류된 비율 | §IV-B |
| Q5 | **없음** | — |
| Q6 | Precision, F1(주지표), FAR, FDR(=Recall) | §IV-B |
| Q7 | **언급 없음** | — |

- 판정: unit=`Window` / scope=`Balanced` / verdict=`Interval 7 d` / oper=`FAR`
- 표 1 수정 필요: **없음**

#### 제외 사유 (Balanced보다 큰 문제)

**이 논문은 SMART 기반 HDD 연구가 아니다.** 실험 데이터는 중국 노천 구리광산의 대형 장비(haul truck, hydraulic excavator) 센서 데이터로, 12개 장비에서 18개월간 수집한 847개 fault event다. 제목에 `Disk Failure Prediction`이 들어가지만 실험 대상이 HDD가 아니다.

그리고 고장 예측이 이진 분류도 아니다. 4개 health state(Normal 68.3%, Minor degradation 15.2%, Severe degradation 11.8%, Imminent failure 4.7%)를 softmax로 분류한다.

표 1의 캡션과 2.2-1이 "SMART 기반 HDD 고장 예측 연구"라고 못 박고 있으므로 이 논문은 들어갈 수 없다.

**교훈:** 제목만 보고 편입하면 안 된다. Q0으로 **"실험 데이터가 SMART/HDD인가"** 를 다른 미확인 논문에도 먼저 물을 것. 지금 미확인은 [2], [3], [4]이다.

#### [6]을 빼면 따라 움직이는 곳 (전부 확인함)

| 자리 | 현재 | 조치 |
| --- | --- | --- |
| 표 1 | `TCN-LSTM-Attn (2026)[6]` 행 | 삭제 |
| 서론 2 본문 | "탐지 성공 여부를 정한다[3]-[6]" | 범위에서 제외 |
| 2.1-1 본문 | "시계열 딥러닝 모델이 뒤따랐다[3][6]" | **[3]만 남음.** [3]이 LSTM 사례를 이미 담당하므로 대체 없이 성립 |
| 2.1-2 본문 | "성능을 산출하는 연구가 많다[3]-[6]" | 범위에서 제외 |
| 2.2-1 | "11편을 정리한 것이다" (틀·본문 양쪽) | **10편** |
| 2.2-2 본문 | "Balanced는 양성·음성 비율을 맞춘 표본을 사용함을 뜻한다" | **사례가 0이 되므로 정의 자체를 삭제** |
| 2.2-2 본문 | "11편 중 6편이 Row 또는 Window, 5편이 HDD" | **10편 중 5편이 Row 또는 Window, 5편이 HDD** |
| 2.2-2 본문 | "평가 범위도 Full, Partial, Balanced로 나뉘지만" | **Full, Partial** |
| 2.2-3 본문 | "양성·음성 비율을 맞춘 표본을 사용하면 양성 비율이 실제 운영과 자릿수 단위로 달라진다" | **근거를 잃으므로 삭제.** 남는 논지(일부 구간만 쓰면 정상 구간이 빠진다)는 그대로 성립하고, Partial이 과반이 되므로 오히려 강해진다 |
| 2.2-3 본문 | "평가 범위를 Full로 둔 연구는 11편 중 5편이다" | **[12]·[4] 확정 전까지 숫자를 쓰지 말 것.** 아래 참조 |

#### Full 개수가 아직 확정 불가

[6] 삭제 + [13] SMARTer의 `Full → Partial`을 반영하면 현재 값으로는 **10편 중 4편**이다([12] RODMAN, [4] Model update, [10] StreamDFP, [5] Survival deepnet).

그런데 [12]는 Q2 미확인이고 SMARTer와 같은 함정이 의심되며, [4]는 아예 미확인이다. **둘 다 Partial로 뒤집힐 수 있어 최종값은 2~4 사이다.** 2.2-3의 숫자는 이 둘이 끝난 뒤에 확정할 것.

#### 대체 논문이 필요한가 — 아니다

[6]이 하던 일을 따져 보면 대체 없이 삭제만으로 성립한다.

- 2.1-1의 시계열 딥러닝 사례 → [3]이 이미 담당
- 서론 2·2.1-2의 인용 범위 → 세 편이 남아 범위로 유지됨
- 표 1의 유일한 `Balanced` → 범주 자체를 없애면 됨

즉 **뺄셈만 하면 되고 새로 찾을 필요가 없다.** 최신성 때문에 2026년 인용을 원한다면 별도 과제로 두되 급하지 않다 — [11]과 [14]가 2025년이다.

- 비고: **서지 항목이 실제로 읽은 논문을 가리키는지 한 번만 확인할 것.** 제목이 `Disk Failure Prediction`인데 실험이 광산 장비인 것은 그 자체로 이상하다. 참고문헌의 [6]이 다른 논문을 가리키고 있을 가능성을 배제하고 나면, 어느 쪽이든 표 1에서는 빠진다.

---

## 확인 결과 드러난 패턴 — 2.2-5를 부정 주장에서 긍정 주장으로 바꿀 수 있다

원문 확인을 진행할수록 같은 모양이 반복된다. **본 연구의 구성요소가 선행 연구에 이미 있다. 다만 판정에 들어가지 않고 다른 용도로 쓰인다.**

| 구성요소 | 선행 연구에 있는가 | 어디에 쓰이는가 |
| --- | --- | --- |
| 우측 중도절단 | **있다** — [5]가 KM으로 모델링 | 표본 구성(informative sampling)과 수명 분석. **CNN 혼동행렬에는 없음** |
| 최초 예측 시점 | **있다** — [8]의 $t_{predict}$ | 사후 요약값 $\Delta interval$. **판정 기준이 아님** |
| HDD 단위 집계 | **있다** — [8]의 map function, [10]의 daily disk prediction | 구간 소속 판정을 HDD로 올린 것. **최초 Alarm으로 합치지 않음** |
| 전체 관측 이력 | **있다** — [10]의 스트림, [5]의 10년 | 시점별 분류. **하나의 운영 사건으로 통합하지 않음** |

**따라서 2.2-5의 신규성 주장은 "판정에"라는 단어 하나에 걸려 있고, 그 단어 덕분에 전부 참이다.** 다행이지만 위태롭기도 하다. 심사자가 "중도절단 다룬 연구 있다", "최초 예측 쓴 연구 있다"로 각각 반례를 들이밀 수 있는데, 지금 문장은 "확인되지 않았다"는 부정 존재 주장이라 방어가 수세적이다.

**바꿀 수 있는 방향:** "못 찾았다"가 아니라 **"구성요소는 이미 와 있고, 어느 연구도 그것을 판정에 넣지 않는다"** 로 쓰면 긍정 주장이 된다. 반례로 제시될 논문들이 오히려 근거가 된다. [5]와 [8]이 각각 한 사례씩 대므로 예시도 확보되어 있다.

이쪽이 훨씬 단단하고, 문장 수도 늘지 않는다.

---

# ⭐ HDD 단위 집계에는 대표 방식이 없다 — 그리고 차별점은 집계 단위가 아니다

**언제 꺼내는가:** 심사자가 "표 1을 보니 HDD 단위가 7편으로 더 많은데, 왜 행 단위를 대조군으로 쓰는가", "우리도 HDD 단위인데 뭐가 다른가", "HDD 단위로 집계하면 되는 것 아닌가"를 물을 때. 그리고 **3.2-7(제안 방식의 의의)을 쓸 때.**

## 세 갈래로 갈린다 — 표준이 없다

### 계열 A — 개체당 표본 하나로 접기 ([13] [11] [2])

각 HDD의 이력에서 구간 하나를 뽑거나 압축해서 **HDD 하나 = 표본 하나**로 만든다. **표본 안에 시점이라는 개념이 없다.**

| 논문 | 방식 |
| --- | --- |
| [13] SMARTer | failure 상태를 배제한 고정 길이 healthy 관측 구간을 하나의 sample로. "10일 내 고장?" → 0/1 |
| [11] Explainable TS | signal length 32일 + lead time 5일로 구간 하나를 잘라내 "**each row representing a hard drive**" |
| [2] Disk replacement | 속성별 창을 지수평활로 단일 값에 압축. post-aggregation 수치가 disk 수와 일치(2.51%/2.43%) |

### 계열 B — 시점별 예측 후 OR 집계 ([8] [3] [12], Murray 2005)

시점마다 예측하고 **창 안에 양성이 하나라도 있으면 그 HDD를 양성으로** 본다.

| 논문 | 원문 |
| --- | --- |
| [8] LightGBM+CID | Algorithm 1, threshold 1 — "**as long as one day's hard disk data is predicted to be about to failure, we think that hard disk is about to fail**" |
| [3] LSTM specificity | "**If there are alarm records in the sequence** given by the prediction model, it is considered that the disk may fail next day" |
| [12] RODMAN | FPR = "falsely predicted failed **disks** ... to the total number of healthy **disks**" — 분모가 disk |
| Murray et al. (2005) | MI 가정 — "**if any of the instances is labeled 1, then the bag label is 1**". "every vector of $n$ consecutive samples **in the history of the drive** is used ... **If any $x$ is classified as failed, then the drive is predicted to fail.**" |

### 어느 쪽도 아님 — [10] StreamDFP

**집계를 하지 않는다.** 매일 각 HDD에 대해 "다음 30일 내 고장?"을 예측하고 **일별 예측을 각각 채점**한다. HDD-일이 표본이다.

## 굳이 대표를 꼽으면 OR 집계

**Murray et al.(2005)이 multiple-instance learning으로 정식화**했고 JMLR에 실렸으며 [2][3][4][11] 네 편이 인용한다. [8]이 Algorithm 1로 명시 구현했고, [3]도 같은 규칙이며, [11]이 인용하는 Zhang et al.(2020)의 IMDA도 "마지막 14일 중 하나라도 고장으로 분류되면 disk를 고장으로 표시"다.

**정식 이름이 붙은 유일한 형태**라는 점에서 대표성이 있다. 다만 계열 A도 세 편이라 다수라고 할 수는 없다.

## 이 사실 자체가 2.2의 증거다

일곱 편이 HDD 단위를 하는데 **방식이 세 갈래로 갈리고 어느 것도 표준이 아니다.** 각자 자기 문제를 다시 정의했기 때문이다.

**"혼동행렬에 맞추려고 문제를 재정의한다"는 2.2의 논지가 이보다 직접적인 증거를 갖기 어렵다.** 그리고 3.1-1의 "재현할 대표 사례가 없어 통제군으로 쓴다"도 수사가 아니라 사실 진술이 된다.

---

## ⭐ 차별점은 "현실 반영"이 아니라 "해석 가능성"이다

### 왜 "현실을 더 잘 반영한다"로 쓰면 안 되는가

**정도의 문제**라 계열 B도 같은 주장을 할 수 있다. OR 집계도 "HDD 단위로 대응하니 HDD 단위로 재는 것이 현실적이다"라고 말할 수 있다. 반박할 근거가 없다.

### **"산출된 값이 운영 결정에 그대로 대입되는가"는 종류의 문제다**

세 갈래 전부 이것이 안 된다.

| 계열 | 산출값의 뜻 | 왜 운영 결정으로 안 옮겨지는가 |
| --- | --- | --- |
| A (개체당 표본) | 이 HDD가 고장 그룹인가를 맞힌 비율 | **언제 대응할지가 나오지 않는다.** 표본에 시점이 없다 |
| B (OR 집계) | 이 HDD를 한 번이라도 지목했는가 | **400일 전 지목과 20일 전 지목이 같은 성공이다.** 점검을 언제 시작할지가 나오지 않는다 |
| [10] (HDD-일) | 하루 단위 precision/recall | 하루 단위 값이라 "이 HDD를 어떻게 할 것인가"로 옮겨지지 않는다 |

**본 연구의 네 지표는 옮겨진다.**

| 지표 | 운영자가 읽는 값 |
| --- | --- |
| 운영 Recall | 고장 HDD 중 대응 가능한 시점에 Alarm이 온 비율 → **실제로 구제된 비율** |
| 운영 FAR | 미고장 HDD 중 점검 대상이 된 비율 → **점검 부담** |
| Median Lead Time | 확보된 대응 시간의 중앙값 → **일정 계획** |
| 운영 Precision | 발생한 Alarm 중 대응 가능했던 비율 → **알람 신뢰도** |

## 3.2-7에 쓸 방향

지금 본문은 "최초 Alarm을 기준으로 판정 결과를 결정하며 우측 중도절단 문제를 판정 체계 안에서 처리한다는 점에서 **기존 평가와 구분된다**"이다. **구분점만 말하고 값의 성격을 말하지 않는다.**

방향: 판정 절차가 다르다는 것이 아니라, **그 절차를 거쳐야만 운영자가 자기 결정에 대입할 수 있는 값이 나온다.** 그리고 **HDD 단위로 집계하는 것만으로는 그 값이 나오지 않는다** — 지목 여부만 남고 시점이 사라지기 때문이다.

이렇게 쓰면 **"우리도 HDD 단위인데?"가 원천적으로 막힌다.** 집계 단위의 문제가 아니라 값의 해석 가능성 문제가 되기 때문이다.

## 반문 대응 스크립트

**"HDD 단위가 7편으로 더 많은데 왜 행 단위를 대조군으로 쓰나"**
→ 그 일곱 편의 방식이 세 갈래로 갈리고 표준이 없다. 재현할 대표 사례가 없으므로 대표성이 아니라 통제를 목적으로 골랐다. 행 단위 평가는 학습에 쓴 레이블을 그대로 채점에 쓰므로 추가 설정이 없는 가장 기본형이다.

**"그럼 결과가 대조군 선택의 산물 아닌가"**
→ 한 HDD의 양성 행 중 하나라도 맞으면 HDD 단위 판정에서는 탐지로 집계되므로, 집계 단위를 올리면 탐지가 줄지 않는다. 더 흔한 쪽으로 바꾸면 격차가 **커진다.** (실측: seed 42, 48개 실행 전부, 최소 증가폭 +0.036. HGST 5%에서 행 0.793 → HDD 집계 0.872 → 운영 0.068)

**"HDD 단위로 집계하면 되는 것 아닌가"**
→ 집계 단위를 올려도 지목 여부만 남고 시점이 사라진다. 400일 전 Alarm과 20일 전 Alarm이 같은 성공이 되므로 점검을 언제 시작할지, 대응 시간이 얼마나 확보되었는지가 나오지 않는다.

**"HDD 단위 평가를 하나 구현해서 비교하지 그랬나"**
→ 미고장 HDD에는 고장 시점이 없어 구간 판정을 적용할 수 없으므로, HDD 단위 오탐률의 분모는 미고장 HDD가 되어 **식 (3)과 같은 구조**가 된다. 그러면 "행 단위로 건 제약이 운영에서 어떻게 나타나는가"라는 첫째 실험의 질문 자체가 성립하지 않는다.

---

## 진행 상황

| 논문 | 상태 | 표 1 수정 가능성 | 우선순위 |
| --- | --- | --- | --- |
| [13] SMARTer | 원문 확인 | **Full → Partial (확정적)** | 반영만 하면 됨 |
| [5] Survival deepnet | 원문 확인 | **없음 (현행 표가 맞음)** | 2.2-5 보강 재료로 활용 |
| [14] DFPoLD | 원문 확인 | 표는 없음. **2.2-4 본문의 주장 둘이 틀림** | 본문 수정 필요 |
| [6] TCN-LSTM-Attn | **원문 전문 확인** | **없음 (현행 표가 맞음)** | 2.2-3 근거로 활용 |
| [8] LightGBM+CID | 원문 확인 | **없음.** 2.2-4가 [8]에 대해서는 정확함 | 지표 분리만 하면 됨 |
| [12] RODMAN | **원문 확인** (arXiv) | **없음 (현행 표가 맞음)** | 완료 |
| [10] StreamDFP | 원문 확인 | Verdict 20d → 30d 검토 | 판단만 필요 |
| — Aggarwal (2018) | **원문 확인** | **표 1 편입 여부 판단 필요** | 판단만 필요 |
| [4] Model update | **원문 확인** | **Full → Partial (확정)** | 완료 |
| [16][17] Early Warning | **원문 확인** ([17]) | 표 밖. 3.2 서술 전부 일치 | 완료 |
| [3] LSTM specificity | **원문 확인** | **unit → HDD (확정), Scope 판단 필요** | 완료 |
| [11] Explainable TS | **원문 확인** | **없음 (현행 표가 맞음).** `Segments → Partial` 정정이 옳았음 | 완료 |
| [2] Disk replacement | **원문 확인** | **unit·Scope 두 열 권고 (추론)** | 완료 |
| [15] Tuttle (sepsis) | **원문 확인** | **서지 3곳 오류.** 수치는 정확 | 완료 |
| [16] Scully & Daluwatte | **원문 확인** | **없음.** 묶음 인용 정확 | 완료 |
| [12] arXiv 게재본 유무 | **확인** | **없음.** arXiv 인용이 맞음 | 완료 |

## ✅ 1차 조사 완료

표 1의 11편, 표 밖의 [15][16][17], Aggarwal, 그리고 아래 Amram까지 **전부 원문 확인을 마쳤다.**

---

# 읽은 논문들의 참고문헌에서 나온 추가 후보

각 논문의 참고문헌을 훑어 나온 것들이다. **하나는 확인을 마쳤고, 나머지는 미확인이다.**

## ✅ Amram et al. (2021) — 확인 완료. 2.2-5에 위협 없음

> M. Amram et al., "**Interpretable predictive maintenance for hard drives**," *Machine Learning with Applications*, 2021. arXiv:2102.06509

[11]이 벤치마크로 쓰는 논문이고([11] Table 4: FDR 54.68% / FAR 11.85%), [11]의 Table 1이 "**Survival trees for disk failure prediction with long lead time**"이라고 소개해서 **가장 위험한 후보였다.** 생존분석 + HDD + lead time이 한 줄에 다 있었다.

확인 결과 안전하다.

| 항목 | 결과 |
| --- | --- |
| 우측 중도절단 | **생존 모델의 우도에만.** "we can still use these data for learning by exploiting the fact that we have observed these machines without failure for some time, and using this to form lower bounds for remaining useful life" |
| 평가에서의 중도절단 | 평가 노드에 Kaplan-Meier 곡선을 보여줄 뿐, **중도절단을 보정한 지표를 따로 보고하지 않는다** |
| 최초 Alarm | **없음.** 일별 스냅숏을 각각 독립으로 평가 |
| **Lead Time** | **정의도 보고도 없다.** 예측 창(30·60·90일)은 지정하지만 실제 확보된 lead time은 다루지 않는다 |
| 평가 단위 | 일별 행 |
| 지표 | AUC, Accuracy, Sensitivity, false alarm rate |

**"long lead time"은 [11]이 붙인 표현이고, 원문은 lead time을 지표로 쓰지 않는다.** 창을 길게 잡았다는 뜻이다.

**패턴 표에 또 한 줄.** 생존분석을 HDD에 직접 적용하고 중도절단을 우도에 넣으면서도, 평가는 일별 행의 AUC/FAR로 돌아간다. [5]·Aggarwal·[12]·[16]에 이어 다섯째다.

### 덤 — B-1(무작위 분할 비판)에 반례가 생겼다

Amram의 분할은 이렇다.

> "separated the data into training and testing **based on hard drives' serial numbers** so that no drive had observations appearing in both sets"

**HDD 단위 무작위 분할이다. notion2 4.1-4와 같은 설계다.** [4] Züfle가 "a temporal split is required"라고 비판하지만, 동료심사를 거친 다른 HDD 논문이 개체 단위 분할을 쓴다. 심사자가 [4]를 들어 물어오면 이쪽을 댈 수 있다.

---

## 미확인 후보 — 우선순위 순

### 1순위 — 2.2-5를 흔들 수 있는 것

**(a) ✅ Xu et al. (2016) — 확인 완료. 2.2-5에 위협 없음**

> C. Xu, G. Wang, X. Liu, D. Guo, and T.-Y. Liu, "**Health Status Assessment and Failure Prediction for Hard Drives with Recurrent Neural Networks**," *IEEE Transactions on Computers*, Vol. 65, No. 11, pp. 3502–3508, Nov. 2016. DOI 10.1109/TC.2016.2538237

데이터는 Backblaze가 아니다. Zhu et al. (2013)이 공개한 실데이터센터 데이터 "W"(정상 22,790 / 고장 434), "S"(38,819 / 170), "M"(10,010 / 147) 셋이다.

| 항목 | 결과 |
| --- | --- |
| Health degree | **잔여 시간을 6단계로 이산화한 것.** "we quantify the health status of a hard drive as **the time before it breaks down**". Level 1은 잔여 72시간 미만의 "red alert" |
| 판정 규칙 | **N개 연속 표본에 대한 투표.** VAT2F/VAT2H — level 1–4의 합과 level 6의 수를 비교. $N = 3, 7, \ldots, 47$ |
| 최초 Alarm | **없음** |
| 우측 중도절단 | **언급 없음** |
| 지표 | FDR, FAR, 그리고 health degree 정확도 $H_{acc}$ |

**2.2-5의 연언 세 항 중 둘을 만족하지 않는다. 안전하다.**

#### ✅ 2.2-4에 세 번째 사례가 생겼다 — 그리고 가장 선명하다

> "Another important variable is **how long in advance** we can detect an impending drive failure. ... The **average time in advance** of our proposed RNN based failure prediction method is **241.6 hours** ... on dataset "W", **494 hours** ... on "S", and **369.4 hours** ... on "M", **which is sufficient for backing up data before the failure actually occurs.**"

494시간은 **20.6일**이다. 그리고 이 값이 크다는 것을 **좋은 결과로 제시한다.** 상한을 묻지 않는다.

2.2-4의 "값이 클수록 좋은 것으로 해석된다"가 [8] Days in Advance, [14] DPF에 이어 **세 번째 사례**를 얻었다. 그리고 이쪽이 수치가 훨씬 커서(20일) 상한 부재가 더 눈에 띈다.

**다만 지면을 생각하면 굳이 추가 인용을 하지 않아도 된다.** 2.2-4는 이미 [8]로 사례를 하나 대고 있다. 이 발견의 값은 "우리 비판이 특정 논문의 특수 사례가 아니라 관행"이라는 확신이지, 새 인용이 아니다.

#### ✅ 결론 6의 확장 방향에 선례가 있다

결론 6이 "최초 Alarm 하나 대신 **일정 기간 안의 반복 Alarm**을 요구하면"을 후속 과제로 든다. **Xu et al.이 정확히 그것을 한다** — N개 연속 표본에 대한 투표. 원조는 Zhu et al. (2013)의 voting-based failure detection algorithm이다.

결론 6은 신규성을 주장하지 않으므로 **틀리지 않았다.** 다만 알고 쓰면 확장 방향이 막연한 제안이 아니라 **"이미 확립된 집계 규칙을 운영 판정 안으로 가져오는 일"** 이 되어 구체성이 생긴다. 인용을 붙일지는 지면 판단.

#### 패턴 표에 가장 선명한 줄

**모델이 시간 정보를 직접 산출하는데 판정에서 그것을 버린다.**

health degree는 잔여 시간으로 정의된 6단계 레이블이다. 그런데 고장 예측 판정에서는 **level 1–4를 "고장", level 6을 "정상"으로 접어버린다**(level 5는 기권). 시간 정보를 만들어 놓고 판정 단계에서 이진으로 되돌린다.

동기도 운영적이다 — "it enables technicians to **schedule the recovery** of different hard drives according to the **level of urgency**". 그런데 평가는 FDR/FAR과 $H_{acc}$다.

지금까지 본 것 중 **"구성요소는 있으나 판정에 안 들어간다"가 가장 극명한 사례**다.

#### ⚠ 서지 충돌 하나 — Murray et al. (2005)의 제3저자

Xu et al.의 참고문헌 [4]는 이렇게 적는다.

> "J. F. Murray, G. F. Hughes, and **D. Schuurmans**, "Machine learning methods for predicting failures in hard drives: A multiple-instance application," J. Mach. Learn. Res., vol. 6, pp. 783–816, 2005."

그런데 [3] Hu et al.의 참고문헌은 **"Murray, J.F., Hughes, G.F., Kreutzdelgado, K."** 로 적는다. **제3저자가 서로 다르다.**

앞서 드린 인용에 K. Kreutz-Delgado를 넣었는데, **JMLR 페이지에서 직접 확인하고 넣으실 것.** 어느 쪽이 오기인지 여기서는 판정할 수 없다.

[4]가 "modeled the current health state of an HDD using a **health index** and predicted the **time-to-failure** using a Recurrent Neural Network"라고 소개한다. **HDD를 대상으로 time-to-failure를 직접 예측하는 연구**라 Lead Time과 가장 가깝다. Amram이 안전했으므로 여기가 남은 최대 위험이다.

확인할 것: time-to-failure가 **판정 기준**인가 별도 회귀 출력인가. 최초 Alarm 개념이 있는가. 중도절단을 평가에 넣는가.

**(b) ✅ Murray, Hughes, Kreutz-Delgado (2005) — 확인 완료. 2.2-5에 위협 없음**

> J. F. Murray, G. F. Hughes, and K. Kreutz-Delgado, "**Machine Learning Methods for Predicting Failures in Hard Drives: A Multiple-Instance Application**," *Journal of Machine Learning Research*, Vol. 6, pp. 783–816, 2005.

#### ✅ 저자 충돌 해결 — Kreutz-Delgado가 맞다

원문 표지가 **Joseph F. Murray, Gordon F. Hughes, Kenneth Kreutz-Delgado** (전원 UCSD)다. 그리고 같은 면에 **"Editor: Dale Schuurmans"** 가 있다.

**Xu et al. (2016)이 JMLR 편집자 이름을 제3저자로 잘못 넣은 것이다.** [3] Hu et al.의 표기가 맞다. 우리 인용도 맞다.

#### MI 구조 — 예상대로 bag = 드라이브, OR 집계

> "Each pattern $x$ (composed of $n$ samples) is an **instance**, and the set of all patterns for a drive $i$ is the **bag** $X_i$. The terms **bag label and drive label are interchangeable**"
> "**if any of the instances is labeled 1, then the bag label is 1.** This ... is known as the MI assumption."

판정 규칙은 이것이다.

> "**every vector of $n$ consecutive samples in the history of the drive is used** ... **If any $x$ is classified as failed, then the drive is predicted to fail.**"

**드라이브의 이력 전체에 걸친 OR 집계. 시점은 쓰지 않는다.**

| 항목 | 결과 |
| --- | --- |
| 평가 단위 | **HDD** (bag = drive) |
| 판정 | 이력 전체 패턴에 대한 OR. **구간(horizon)이 아예 없다** — 레이블은 드라이브의 최종 결과다 |
| 최초 Alarm | **없음** |
| 우측 중도절단 | **언급 없음** |
| 지표 | FDR, FAR |

**2.2-5의 연언 중 둘을 만족하지 않는다. 안전하다.**

#### ⚠ 2.2-2의 "판정은 모두 고장 전 일정 구간의 레이블에 근거한다"와의 관계

**Murray et al.에는 구간이 없다.** 레이블은 "이 드라이브가 고장으로 반품되었는가"이지 고장 전 N일이 아니다.

다만 2.2-2의 문장은 **"표 1의 연구들은"** 으로 범위가 한정되어 있고 Murray는 표 1에 없다. **문장은 그대로 유효하다.**

**표 1에 넣지 말 것을 권한다.** 넣으면 Verdict 열에 구간이 없는 행이 생겨 2.2-2의 문장을 손봐야 하고, good이 실험실 시험 / failed가 현장 반품이라는 특수한 구성이라 다른 행과 성격이 다르다.

#### ✅ C-2(집계 단위 문장)를 쓴다면 이것이 정전 인용이다

"HDD 단위 집계 + 구간 판정" 칸의 가장 순수한 형태다 — MI 가정 자체가 "bag은 instance 하나라도 양성이면 양성"이고, 그것을 **이력 전체**에 적용한다. [8]의 map function이나 [3]의 윈도우 규칙보다 앞서고 일반적이며, [2][3][4][11] 네 편이 인용하는 고전이다.

C-2를 넣기로 하면 [8] Algorithm 1에서 추론하는 대신 이쪽을 대는 것이 낫다.

#### ✅ 3.2-6의 기제를 2005년에 이미 지적했다

> "Since the classifier is applied **repeatedly to all $N$ vectors from the same drive**, each test must be **very resistant to false alarms.**"

반복 적용이 오탐 기회를 누적시킨다는 것을 명시한다. **notion2 3.2-6의 "관측이 길수록 각 HDD가 한 번이라도 Alarm을 낼 가능성만 높아지기 때문이다"와 같은 통찰이다.**

[15]의 임상 사례에 이어 **독립적으로 같은 기제를 지목한 두 번째 사례**이고, 이쪽은 HDD 분야 안에서 2005년에 나왔다. 3.2-6이 특이한 관찰이 아니라는 근거가 하나 더 늘었다.

#### 저자들이 스스로 인정하는 모집단 교란 — 2.2-3에 쓸 수 있다

> "Drives labeled as good were from **a reliability test, run in a controlled environment by the manufacturer.** Drives labeled as failed were **returned to the manufacturer from users after a failure.** ... Algorithms that attempt to learn the difference between the good and failed populations may in fact be **learning this difference and not the desired difference** between good and nearly-failing drive samples."

good 178대 / failed 191대로 거의 균형이고, 두 집단의 출처가 아예 다르다. **평가 모집단이 운영과 다르면 무엇을 학습·측정하는지가 달라진다는 것을 저자들이 직접 적는다.**

2.2-3의 논지와 같은 방향이되 결이 다르다(범위 제한이 아니라 모집단 불일치). 인용할지는 판단이지만, **"평가 모집단 문제는 이 분야가 20년 전부터 알고 있었다"는 근거**로는 강하다.

#### 참고 — 드라이브당 이력이 짧다

2시간 간격 표본이고 드라이브에 **최근 300개만 저장**된다(= 600시간 ≈ 25일). 설계 선택이 아니라 장치 제약이라고 명시한다. 따라서 "이력 전체"라 해도 실제로는 25일 창이다.

### 2순위 — 특정 주장을 보강할 수 있는 것

**(c) Zhang et al. (2020) IMDA** — [11]의 관련연구에서

> J. Zhang et al., "**Minority Disk Failure Prediction Based on Transfer Learning in Large Data Centers of Heterogeneous Disk Systems**," *IEEE Transactions on Parallel and Distributed Systems*, Vol. 31, No. 9, pp. 2155–2169, 2020.

⚠ **주의: IMDA 귀속이 미확인이다.** [11]이 "Zhang et al. (2020) proposed the **Instances Map to Disk Algorithm (IMDA)**, which marks the health state of a disk as a failure **if any of the last 14 days of records were classified as failures**"라고 적는데, 검색 범위에서 위 TPDS 논문에 IMDA라는 용어가 확인되지 않았다. **[11]의 참고문헌에서 Zhang et al. (2020)의 정확한 항목을 먼저 확인할 것.** 같은 해에 다른 Zhang 논문일 수 있다.

**"HDD 단위 집계 + 구간 판정"에 이름이 붙은 알고리즘이다.** 지금은 그 칸을 [8]의 Algorithm 1에서 추론하고 있는데, 이걸 쓰면 명명된 선행 기법을 직접 댈 수 있다. C-2(집계 단위 문장)를 넣기로 하면 값이 커진다. [11]의 참고문헌에서 정확한 서지를 뽑아야 한다.

**(d) Xiao et al. (2018)** — [4]·[6]·[11]이 인용

> J. Xiao, Z. Xiong, S. Wu, Y. Yi, H. Jin, K. Hu, "**Disk failure prediction in data centers via online learning**," *Proc. 47th International Conference on Parallel Processing (ICPP)*, 2018.

**온라인 학습**이다. 2.2-5의 "시간 순 온라인 추론" 절과 결론 6의 재학습 확장에 모두 닿는다. [12] RODMAN이 온라인 학습을 후속 과제로 남긴다고 했으므로, 이미 한 연구가 있다면 그 서술을 다듬어야 할 수 있다.

**(e) Heagerty & Zheng (2005)** — [15]의 참고문헌 [8]

> P. J. Heagerty, Y. Zheng, "**Survival model predictive accuracy and ROC curves**," *Biometrics*, vol. 61, no. 1, pp. 92–105, 2005.

[15]가 "prediction-level analyses, **favored for dynamic prediction in the statistical literature**"라고 할 때 가리키는 통계 문헌이다. 중도절단을 반영한 시간 의존 ROC의 표준 틀이다. 본 연구가 통계 문헌과의 관계를 명시할 생각이면 여기가 앵커다. **다만 지면을 고려하면 안 건드리는 쪽이 나을 수 있다.**

### 3순위 — 서론 4의 범위 선언과 관련

서론 4가 "교체·점검 비용과 인력 운용 등 실제 운영의 대응 정책은 포함하지 않으므로"라고 선언하는데, **비용 기반 운영 지표를 만든 선행 연구가 실제로 있다.** [11]의 관련연구에서 셋이 나온다.

- **Xu et al. (2018) CDEF** (Microsoft Research) — "developed a metric that estimates the **migration cost of virtual machines** in a data center, considering both the unnecessary migration cost and the cost of data loss"
- **Li et al. (2018)** — "proposed **migration rate and mis-migration rate** to measure preserved and lost data"
- **Mahdisoltani et al. (2017)** — "proposed to adjust the **hard drive scrubber speed** according to the predicted errors"

이들을 확인하면 서론 4의 범위 제외가 **"안 했다"가 아니라 "그쪽은 별도 계열이 이미 다룬다"** 가 되어 방어가 쉬워진다. 다만 참고문헌이 셋 늘어난다. **지면 대비 판단 필요.**

**(f) Aussel et al. (2017)** — [3]과 [4]가 모두 인용

> N. Aussel et al., "**Predictive models of hard drive failures based on operational data**," *Proc. 16th IEEE ICMLA*, 2017.

제목에 **"operational data"** 가 들어간다. 본 연구가 "운영 환경 기반 평가"라는 표현을 쓰므로, 이 논문의 "operational"이 무엇을 뜻하는지는 알아 둘 값이 있다. 현장 데이터 대 실험실 데이터의 구분일 가능성이 높지만, 같은 단어라 심사자가 연결 지을 수 있다.

### 확인 불필요로 판단한 것

- Burrello et al. (2020) TCN, Mohapatra & Sengupta (2023) TFBEST, Sun et al. (2019) TCN — 모델 구조 연구. 평가는 표준 horizon 분류로 보인다
- Ahmed & Green II (2024) cost-aware LSTM — [5]·[9]와 같은 저자. 이미 [5]를 확인했다
- Hamerly & Elkan (2001), Hughes et al. (2002) — 1990–2000년대 통계 기법. [2]의 관련연구가 "if any of its snapshots are identified as anomalous"로 소개하므로 OR 집계다
- Ganguly et al. (2016), Xu et al. (2018) CDEF의 모델 부분 — 시스템 신호를 SMART에 결합하는 연구. 판정 절차는 표준

## ✅ 표 1 전수 조사 완료 — 11편 전부 원문 확인

**Scope는 어느 판단을 택하든 Full 3편으로 확정.** 2.2-3에 그 숫자를 써도 된다.

확인 과정에서 표 1이 네 군데 틀렸다.

| 논문 | 열 | 기존 → 정정 | 근거 |
| --- | --- | --- | --- |
| [13] SMARTer | Scope | Full → **Partial** | failure 상태를 배제한 고정 길이 healthy 구간 |
| [4] Model update | Scope | Full → **Partial** | "discarding all older instances of those HDDs" |
| [3] LSTM specificity | Eval. unit | Window → **HDD** | FDR·FAR이 drive 단위로 정의됨 |
| [2] Disk replacement | Eval. unit | Row → **HDD**† | post-aggregation 비율이 본문의 디스크 교체율과 일치 |

그리고 두 군데가 판단 대기다 — [3]과 [2]의 Scope를 `Balanced`로 볼 것인가.

**남은 것은 표 1 밖의 세 건뿐이다.**

**남은 미확인 논문은 2차 자료를 거치지 말고 원문 전문을 볼 것.** [6]에서 두 번 어긋났고 둘 다 그럴듯한 형식이었다. Q0("실험 데이터가 SMART/HDD인가")도 계속 먼저 물을 것 — 결과적으로 [6]은 통과했지만 확인 비용은 거의 없다.

### Aggarwal et al. (2018) — 표 1에 없지만 가장 가까운 연구. 원문 확인 완료

> K. Aggarwal, O. Atan, A. K. Farahat, C. Zhang, K. Ristovski, C. Gupta, **"Two birds with one network: Unifying failure event prediction and time-to-failure modeling,"** IEEE Big Data 2018. arXiv:1812.07142. (Hitachi America R&D)

**Backblaze ST4000DM000, 2014.1–2015.6, 26개 SMART 특성.** C-MAPSS 터보팬과 함께 두 데이터셋을 쓴다. 즉 HDD SMART 연구가 맞다.

#### 본 연구의 구성요소를 얼마나 갖고 있나 — 거의 다 갖고 있다

| 구성요소 | Aggarwal et al. |
| --- | --- |
| 우측 중도절단 | **명시적으로 다룬다.** "The only form of censoring we observe is the right-censoring." 중도절단 변수 $c_p$, survival likelihood로 학습 |
| 미고장 장비 활용 | **핵심 기여다.** MTL의 제약 손실 $L^{nf}_{r,p} = I(t_{p,f} > c_p)\sum \max(c_p - \widehat{RUL}, 0)$ — 미고장 장비의 예측 RUL이 관측 기간보다 커야 한다는 제약 |
| 고장까지의 시간 | **연속량으로 모델링한다.** $RUL = E[T-t \mid T>t]$, Weibull 분포, hazard rate |
| 대응 시간의 하한 | **있다.** Filter window $\tau_f$ (HDD 4일) — "prediction just before the failure time $t_f$ does not give enough warning in the realistic settings" |

#### 그런데 판정에는 들어가지 않는다 — 결정적 근거 두 개

**하나.** 평가 모집단에서 중도절단 장비를 뺀다.

> **"We perform test results only on the non-censored devices."** (§V-A)

중도절단은 **학습 목적함수**에 들어가지, 평가에는 들어가지 않는다. 테스트는 고장이 관측된 디스크만으로 한다.

**둘.** 판정 단위가 윈도우이고 최초 Alarm 개념이 없다.

각 landmark window(HDD는 크기 4일)를 독립적으로 분류하고, evidence window에 걸리면 failure label을 준다. 한 HDD의 여러 윈도우 예측을 하나의 Alarm으로 합치는 절차가 없다. 평가는 윈도우 단위 **AUC-ROC / AUC-PR**이다(Backblaze MTL-RNN 기준 AUCPR 0.2612).

#### 판정: 2.2-5는 성립한다

현재 문장의 연언 세 항 중 (b) 최초 Alarm 시점을 판정에 포함, (c) 우측 중도절단을 판정에 포함 — **둘 다 만족하지 않는다.**

**그러나 지금까지 확인한 것 중 가장 가깝다.** [5]는 중도절단을 표본 구성에 썼고, 이쪽은 **학습 목적함수에 넣었다.** 그런데도 평가로 넘어오는 순간 윈도우 단위 이진 분류 AUC로 되돌아간다. 앞서 정리한 패턴의 가장 선명한 사례다.

#### 3.2-2에도 쓸 것이 있다

notion2의 3.2-2는 하한을 두지 않는 이유를 이렇게 쓴다.

> 판정에는 Prediction Horizon이라는 상한만을 사용하고 하한은 두지 않는데, 대응에 필요한 최소 시간은 운영 정책에 따라 달라져 이를 판정에 넣으면 평가 결과가 정책에 의존하게 되기 때문이다.

**Aggarwal et al.의 filter window가 정확히 그 하한이다.** 그리고 그들 스스로 "Size of $\tau_e$ is domain dependent and a critical parameter"라고 적으며, 도메인별로 값을 다르게 잡는다(C-MAPSS 20, HDD 12, filter는 5와 4). 즉 **지금 3.2-2가 근거 없이 주장하는 "정책 의존성"을 이 논문이 실제로 보여준다.**

지금 3.2-2의 이유는 논증뿐이고 사례가 없다. 사례가 하나 붙으면 설계 선택이 방어된다.

#### 조치 판단

**표 1에 넣을 것을 권한다.** 넣는다면:

| 열 | 값 | 근거 |
| --- | --- | --- |
| Eval. unit | `Window` | landmark window (HDD $w$=4일) 단위 분류 |
| Scope | `Balanced` | 고장/미고장 장비 수를 random sampling으로 맞춤. 게다가 **테스트는 미고장 장비를 제외** — 실제로는 그보다 더 좁다 |
| Verdict | `Interval 12 d` | evidence window $\tau_e$=12일, filter window $\tau_f$=4일. **표 1에서 유일하게 하한이 있는 판정** |
| Oper. metrics | `RUL (RMSE)` | 고장까지의 시간을 직접 회귀. Days in adv./DPF와 또 다른 계열 |

**넣지 않을 경우의 위험:** 이 논문을 아는 심사자가 "이미 생존분석으로 중도절단을 다룬 HDD 연구가 있다"고 하면, 답이 원고 안에 없다. 지금 2.2-5는 "확인되지 않았다"는 부정 주장이라 반례 하나에 흔들린 것처럼 보인다.

**넣을 경우의 이득:** 부정 주장을 긍정 주장으로 바꾸는 가장 좋은 재료다. 구성요소가 가장 많이 모인 연구조차 평가는 윈도우 단위 AUC로 되돌아간다는 것을 한 사례로 보일 수 있다. 그리고 3.2-2의 하한 논의에도 같은 논문이 쓰인다.

지면은 표 1에 한 행, 2.2-5에 한 문장이다.

---

# [16][17] Early Warning 프레임워크 — 원문 확인 완료

> C. Daluwatte, F. Yaghouby, C. Scully, "A framework to characterize the performance of early warning index alarm systems for patient monitoring," **MethodsX** 6, pp. 1660–1667, 2019. (US FDA)
> PMC 무료 전문: PMC6660561

**3장 판정 체계 전체가 이 프레임워크의 변형이라고 선언하고 있으므로, 논문에서 가장 하중이 큰 인용이다.** 확인 결과 notion2의 서술이 전부 맞다.

## 원 프레임워크의 다섯 범주

| 범주 | 정의 (원문) |
| --- | --- |
| **False** | Warnings from non-event records ($time_{EVENT}$ undefined) |
| **Early** | Warnings from event records where $T_{WARNING} \ge T_{MAX}$ |
| **On time** | Warnings from event records where $T_{MAX} > T_{WARNING} \ge T_{MIN}$ |
| **Late** | Warnings from event records where $T_{WARNING} < T_{MIN}$ |
| **Missed** | Event records where a warning was not generated |

시간 경계는 둘이다. $T_{MAX}$ = "maximum amount of time prior to the event when warning is expected meaningful", $T_{MIN}$ = "minimum amount of time prior to event that would allow meaningful action". 사례 연구의 기본값은 14분과 1분.

집계는 **Alarm 단위와 기록 단위 두 가지**로 한다("normalize with total number of warnings" / "normalize with respect to total number of records"). Alarm 부담도 포함한다 — "the warning burden of the system (occurrence of multiple warning per event/record)".

이벤트 상태는 **전부 알려져 있다고 전제한다.** "time-series data with an annotation of critical event time, or lack of event time"를 요구하며, 중도절단이나 결과를 모르는 대상에 대한 논의가 없다.

## notion2의 서술 대조 — 전부 일치

| notion2 | 원문 확인 |
| --- | --- |
| 2.2-5 "Alarm을 발생 시점에 따라 구분하여 Alarm 단위와 환자 기록 단위로 집계하고" | ✓ |
| 2.2-5 "한 이벤트에 여러 Alarm이 발생할 때의 Alarm 부담까지 평가에 포함하는" | ✓ warning burden |
| 3.2-2 "원 프레임워크는 이벤트 발생 여부가 확정된 대상을 전제하나" | ✓ 중도절단 논의 없음 |
| 3.2-2 "원 프레임워크의 Late에 해당하는 판정은 두지 않았다" | ✓ Late가 실재함 |
| 3.2-4 "원 프레임워크는 이벤트가 관측되지 않은 대상의 Alarm을 단일 False로 처리하나" | ✓ False를 더 쪼개지 않음 |

**다섯 범주의 대응이 깔끔하게 성립한다.**

| 원 프레임워크 | notion2 |
| --- | --- |
| On time | On-time |
| Early | Early |
| Missed | Missed |
| Late | **버림** (관측이 고장까지만이라 늦은 Alarm이 불가능) |
| False | **CE / CN으로 분할** (관측되지 않은 미래를 정상으로 가정하지 않기 위해) |

## 새로 나온 것 — 원 프레임워크에는 하한 $T_{MIN}$이 있다

3.2-2는 하한을 두지 않는 이유를 이렇게 쓴다.

> 판정에는 Prediction Horizon이라는 상한만을 사용하고 하한은 두지 않는데, 대응에 필요한 최소 시간은 운영 정책에 따라 달라져 이를 판정에 넣으면 평가 결과가 정책에 의존하게 되기 때문이다.

**원 프레임워크의 $T_{MIN}$이 바로 그 하한이다.** 즉 notion2는 상한 $T_{MAX}$만 가져오고 $T_{MIN}$을 의도적으로 버린 것인데, 지금 본문은 그 사실을 말하지 않는다.

지금 상태로는 [17]을 확인한 심사자가 "$T_{MIN}$이 있는데 왜 안 썼나, 놓친 것 아닌가"라고 물을 수 있다. **버렸다고 명시하면 의도적 설계가 되고, 이유까지 이미 적혀 있다.** 문장 하나 안에서 처리된다.

그리고 Aggarwal의 filter window와 합치면 **하한을 둔 선례가 둘**이다. 3.2-2의 논거가 사례 없는 논증에서 사례 둘을 가진 주장으로 바뀐다.

## [16] Scully & Daluwatte (JBI 2017) — 원문 확인 완료. 묶음 인용은 정확하다

> C. G. Scully, C. Daluwatte, "Evaluating performance of early warning indices to predict physiological instabilities," **Journal of Biomedical Informatics** 75, pp. 14–21, 2017. (US FDA)

**[16]이 원형이고 [17]이 정리판이다.** notion2가 `[16][17]`로 묶어 인용하는 세 가지가 **[16]에 전부 있다.**

| notion2의 서술 | [16] 원문 |
| --- | --- |
| Alarm을 발생 시점에 따라 구분 | 네 구간: "before Tmax (**early**), within [Tmax,Tmin] (**on time**), after Tmin (**late**, but still before tevent), and after tevent (**missed**)" |
| Alarm 단위와 기록 단위로 집계 | "we normalize the summary histogram two ways: **with respect to total number of warnings and total number of events**" (time profile of warning proportions / time profile of warnings per event) |
| 한 이벤트에 여러 Alarm이 발생할 때의 Alarm 부담 | "the **burden of additional warnings**", 사례 연구에서 "ratio of all warnings to all events of **2.06** for RESPONSIVE compared to **1.13** for STAY-ON" |

$T_{max}$·$T_{min}$도 [16]에 정의가 있고 사례값도 같다(14분/1분). **묶음 인용에 문제가 없다. 수정 불필요.**

차이는 하나다. [16]은 네 구간(early/on time/late/missed)이고 **[17]이 다섯째 False(non-event records)를 정식 범주로 올린다.** [16]은 그것을 본문에서 언급만 한다 — "False positives could be added to the histogram ... at the left end".

## ⚠ 3.2-2를 더 정확하게 만들 수 있다 — $T_{min}=0$은 프레임워크가 허용한다

[16]의 $T_{min}$ 정의에 이런 괄호가 붙어 있다.

> "$T_{min}$ – minimum amount of time prior to the event that would allow for meaningful action (**depending on the application $T_{min}$ may be 0, the time of the event, or negative, after the event**)"

**원 프레임워크가 $T_{min}=0$을 명시적으로 허용한다.** 따라서 본 연구가 하한을 두지 않은 것은 프레임워크에서 벗어난 것이 아니라 **$T_{min}$을 0으로 둔 것**이다.

이게 3.2-2에 두 가지 이득을 준다.

**하나.** "하한을 두지 않았다"보다 "$T_{min}$을 0으로 두었다"가 방어가 쉽다. 전자는 원본을 안 따랐다는 인상을 주고, 후자는 원본이 허용한 설정을 골랐다는 뜻이다. **앞서 A-8에 적은 "버렸다고 명시" 권고를 이쪽으로 바꿀 것.**

**둘.** $T_{min}=0$이면 Late 구간(after $T_{min}$, but still before $t_{event}$)이 **정의상 공집합**이 된다. 지금 3.2-2는 Late를 두지 않는 이유를 따로 대고 있는데(관측이 고장 시점까지만이라 늦은 Alarm이 불가능), **$T_{min}=0$ 하나로 두 결정이 동시에 설명된다.** 문장을 줄일 수 있다.

## [16]이 스스로 인정하는 공백 — 3.2-4의 근거가 된다

> "In this example we used an **enriched data set where the event of interest was known to occur for every experiment.** This allowed us to characterize the warning provided when an event is known to occur. **False positives should be assessed in a broader monitored population** that includes the expected prevalence of the condition."

원 프레임워크가 **사건이 일어난 대상만으로 사례 연구를 했고, 오탐은 더 넓은 모집단에서 따로 평가해야 한다고 스스로 적는다.**

3.2-2의 "원 프레임워크는 이벤트 발생 여부가 확정된 대상을 전제하나"가 이보다 정확할 수 없다. 그리고 3.2-4가 CE/CN으로 나눈 것이 **그 논문이 남겨 둔 자리를 채우는 일**이 된다. 인용 가능한 문장이 확보되었다.

## 그 밖에

[16]도 생존분석과의 유사성을 언급하지만 중도절단을 다루지는 않는다 — "Similar approaches are used in survival analysis where both occurrence of an event ... and the time of the event occurrence are important". [5]·Aggarwal·[11]에 이어 **생존분석이 언급되면서도 판정에는 안 들어가는 사례가 넷째다.**

---

# 개편 계획 — 한 번에 반영할 것

조사가 끝난 시점에 이 절만 보고 순서대로 고치면 된다. 각 항목에 **근거**와 **의존 관계**를 붙였다.

## A. 확정 — 지금 반영해도 되는 것

### A-1. 표 1 [13] SMARTer: Scope `Full` → `Partial`

평가 표본이 failure 상태를 배제한 고정 길이 healthy 관측 구간이다. 2.2-2의 정의상 Partial이다.

### A-2. 표 1 [13] SMARTer: Oper. `—` → `FPR` 검토

Precision·Recall·F-measure·MCC와 함께 **FPR/FNR을 보고한다.** 지금 `—`는 과소 기재다. 다만 FPR을 주지표로 내세우지는 않으므로 그대로 둘 여지도 있다. 판단 필요.

### A-3. 표 1 [10] StreamDFP: Verdict `Interval 20 d` → `30 d` 검토

20일은 **레이블링 구간**($D_L$)이고 채점은 **다음 30일 고장 여부**로 한다. 표 1에 넣는 값은 채점 구간이라는 규칙을 따르면 30일이다. 다만 2.2-2 본문이 이미 "StreamDFP[10]는 30일 이내 고장을 예측하면서 고장 전 20일을 레이블링에 사용한다"로 둘을 구분해 서술하고 있어, 표를 20일로 두고 본문에서 푸는 현재 방식도 성립한다. **판단 필요.**

### A-4. 2.2-4 본문: DPF와 Days in Advance를 분리

현재 첫 문장 "DPF(Days Prior to Failure)와 Days in Advance는 고장 시점과 최초 예측 시점의 간격이라는 점에서 본 연구의 Lead Time과 형태가 같으나"가 **[14]에 대해 거짓**이다.

- **[8] Days in Advance**: $\Delta interval = t_{failure}-t_{predict}$, $t_{predict}$는 최초 예측 시점. **간격이 맞다.**
- **[14] DPF**: Table 12를 보면 window 내 양성 예측 밀도를 일수로 환산한 값이다. **간격이 아니다.**

두 지표를 같은 성질로 묶지 말 것. 본문이 수치를 인용하는 것은 [8] 하나뿐이므로, [8]을 중심에 두고 [14]는 구성이 다르다는 점만 짚는 편이 짧다.

### A-5. 2.2-4 본문: "탐지에 성공한 HDD" → "고장이 확인된 HDD"

- [8]: "we only analyzed the results of the failed disks here"
- [14]: 2023년 실제 고장 560대 전부

둘의 공통점은 탐지 성공이 아니라 **고장이 확인되었다는 것**이다. 논지(모집단이 사후에 정해진다)는 그대로이고 양쪽에 다 맞게 된다.

### A-6. 2.2-4 본문: Horizon 상한 주장을 인용 가능한 형태로

현재: "탐지로 인정되는 구간 밖의 예측은 해당 지표의 계산 대상에서 제외되므로, 이 값은 정의상 Prediction Horizon을 넘을 수 없다."

**[8]이 같은 논문 안에서 window를 두 번 쓴다** — 예측 문제는 10일, §IV-C1의 lead-time 분석은 30일. $t_{predict}$가 "in the time window" 안에서만 찾아지므로 **window를 넓히면 값이 커진다.** 즉 모델의 성질이 아니라 분석자가 정한 폭에 좌우된다.

이 형태로 쓰면 추론이 아니라 원문 사실 위에 서게 된다.

### A-7. 2.2-5 본문: 부정 존재 주장 → 긍정 주장

현재 마지막 문장이 "…연구는 확인되지 않았으며"라는 부정 존재 주장이다. 반례가 하나 나올 때마다 방어해야 한다.

조사 결과 **구성요소는 전부 선행 연구에 있고, 어느 것도 판정에 들어가지 않는다**는 패턴이 확인되었다. 사례가 넷이다.

| 구성요소 | 있는 연구 | 실제 쓰이는 곳 |
| --- | --- | --- |
| 우측 중도절단 | [5] KM 기반 표본 구성 / **Aggarwal은 학습 목적함수** | 평가 판정에는 없음. Aggarwal은 "We perform test results only on the non-censored devices" |
| 최초 예측 시점 | [8] $t_{predict}$ | 사후 요약값 $\Delta interval$. 판정 기준 아님 |
| **미고장 HDD 분모의 오탐률** | **[12] RODMAN, [3], [11]** — 셋 다 미고장 drive 수를 분모로 하는 FAR/FPR | **식 (3)과 구조가 같다.** 그러나 최초 Alarm으로 합치지 않음 |
| HDD 단위 집계 | [8] map function(threshold 1), **[3] 윈도우 내 Alarm이 하나라도 있으면 고장**, [10] daily disk prediction | 구간 소속 판정을 HDD로 올린 것. 시점은 쓰지 않음 |
| 고장까지의 시간 | **Aggarwal** RUL 회귀, Weibull, hazard rate | 별도 회귀 출력. 판정은 윈도우 단위 AUC |
| **Lead Time을 정책 변수로** | **[11]** — lead time 0~35일 민감도 분석, "the FDR decreases as we predict more in advance", "inform predictive maintenance policies" | **전처리 파라미터**일 뿐 판정 기준이 아니다. 성능 지표로 보고되지도 않는다 |

**여섯 줄 전부가 같은 모양이다.** 식 (3)의 구조는 세 편에, 집계 방식은 세 편에, Lead Time은 두 편에 이미 있다. 없는 것은 **최초 Alarm의 시점을 판정 기준으로 쓰는 것** 하나뿐이다.

"못 찾았다"가 아니라 **"가장 많이 모인 연구조차 평가로 넘어오면 그것을 뺀다"** 로 쓰면 반례로 들어올 논문들이 근거가 된다. 문장 수는 늘지 않는다.

### A-8. 3.2-2 본문: 하한을 **"$T_{min}$을 0으로 두었다"** 로 다시 쓴다

**[16] 확인으로 이 항목이 바뀌었다.** 원 프레임워크는 $T_{min}$ 정의에 이렇게 적는다.

> "$T_{min}$ ... (depending on the application **$T_{min}$ may be 0**, the time of the event, or negative, after the event)"

즉 $T_{min}=0$은 **프레임워크가 명시적으로 허용하는 설정**이다. 본 연구는 하한을 버린 것이 아니라 허용된 값을 고른 것이다.

**이득 둘.**

1. "하한은 두지 않는데"는 원본을 안 따랐다는 인상을 준다. "$T_{min}$을 0으로 두었다"는 원본이 허용한 설정을 골랐다는 뜻이 되어 방어가 쉽다. 이유(정책 의존성)는 이미 그 자리에 적혀 있다.
2. **$T_{min}=0$이면 Late 구간(after $T_{min}$, but still before $t_{event}$)이 정의상 공집합이 된다.** 지금 3.2-2는 Late를 두지 않는 이유를 따로 대고 있는데, $T_{min}=0$ 하나로 두 결정이 함께 설명된다. **문장이 줄어든다.**

지면이 오히려 줄어드는 항목이다.

### A-9. 표 1 [4] Model update: Scope `Full` → `Partial`

"we labeled all instances of each HDD that failed within the next ten days as positive **while discarding all older instances of those HDDs**". 고장 HDD의 이력을 마지막 10일로 자른다. 정의상 `Partial`이다.

균형화는 학습에만 적용되고 테스트 prevalence는 0.0639%로 유지되므로 `Balanced`가 아니다.

### A-10. 2.2-3 본문 끝: "11편 중 5편" → **3편**

A-1(SMARTer)과 A-9(Model update) 두 정정을 반영한 결과다.

| 논문 | Scope | 상태 |
| --- | --- | --- |
| [12] RODMAN | **Full** | 원문 확인 |
| [10] StreamDFP | **Full** | 원문 확인 |
| [5] Survival deepnet | **Full** | 원문 확인 |
| [13] SMARTer | Partial (정정) | 원문 확인 |
| [4] Model update | Partial (정정) | 원문 확인 |
| [2] · [3] · [11] | Partial (현행) | **미확인** |
| [8] · [14] | Partial | 원문 확인 |
| [6] | Balanced | 원문 확인 |

**Full로 확인된 것은 3편이고, 나머지 셋([2]·[3]·[11])은 현재 Partial이다.** 그 셋이 Partial로 확인되면 3편으로 확정된다. Full로 뒤집힐 가능성은 낮다 — 지금까지 정정은 두 번 다 Full → Partial 방향이었다.

Aggarwal을 넣으면 분모가 12편이 되지만 Aggarwal 자신은 `Balanced`이므로 분자는 3편 그대로다.

**논지가 더 강해진다.** Partial이 과반이 되어 2.2-3이 지적하는 문제가 예외가 아니라 관행이 된다.

### A-11. 표 1 [3] LSTM specificity: Eval. unit `Window` → `HDD`

§4.3의 지표 정의가 drive 단위다 — FDR은 "the fraction of failed **drives**", FAR은 "the proportion of good **disks** that are falsely predicted as failed". 입력은 15일 윈도우지만 집계는 disk 단위다.

### A-12. 표 1 [2] Disk replacement: Eval. unit `Row` → `HDD` (권고)

Table 1의 post-aggregation 비율(SgtA 2.51%, HitA 2.43%)이 본문의 "only 2.5 to 3% of the disks ... are replaced"와 일치한다. 각 디스크가 집계된 관측 하나로 압축된다. 다만 원문 Table 1의 "Original" 열과는 맞지 않는다(원문 자체 모순). **우리 쪽 추론이라는 점을 기록해 둘 것.**

### A-13. 2.2-2 본문: "11편 중 6편이 Row 또는 Window, 5편이 HDD" → **4편 / 7편**

A-11([3]→HDD)과 A-12([2]→HDD)를 반영한 결과다.

| 단위 | 논문 | 수 |
| --- | --- | --- |
| Row | [4] [5] [14] | 3 |
| Window | [6] | **1** |
| HDD | [2] [3] [8] [10] [11] [12] [13] | **7** |

**Window가 [6] 하나만 남는다.** [6]은 원문 확인이 끝났고 명확히 sample 단위라 범주 자체는 근거가 있다. Aggarwal을 넣으면 Window가 2편이 된다.

A-12를 채택하지 않으면 **5편 / 6편**이다.

### A-14. 참고문헌 [15] 서지 정정 (세 군데)

| 항목 | 현재 | 정정 |
| --- | --- | --- |
| 플랫폼 | medRxiv | **Research Square** |
| 게시일 | May 2026 | **June 2, 2026** |
| DOI | 10.64898/2026.05.26.26354141 | **10.21203/rs.3.rs-9726164/v1** |

인용한 두 수치(환자 단위 0.86, 시점 단위 0.62)는 **원문과 정확히 일치한다.** 서지만 고치면 된다.

### A-15. 3.2-6의 관측 구간 의존성에 타 분야 근거를 붙일 수 있다

[15]가 편향 기제를 진단하면서 본 연구가 독립적으로 도달한 것과 같은 두 가지를 지목한다.

> "**false positive alerts that occur outside the pre-event window among cases are discarded**, and **shorter lengths of stay among controls further reduce opportunities for false positives.**"

앞은 2.2-3의 "평가 범위를 제한하면 ... False Alarm이 발생할 기회 자체가 감소하여"와, 뒤는 3.2-6의 "관측이 길수록 각 HDD가 한 번이라도 Alarm을 낼 가능성만 높아지기 때문이다"와 같다.

다른 분야의 임상 연구가 같은 기제를 독립적으로 지목했다는 것은 **3.2-6이 우리만의 특이한 관찰이 아니라는 근거**다. 이미 [15]를 인용하고 있으므로 참고문헌이 늘지 않는다.

---

## 최종 표 1 (권고안 반영)

| Study | Eval. unit | Scope | Verdict | Oper. metrics |
| --- | --- | --- | --- | --- |
| Disk replacement (2016)[2] | **HDD**† | **Balanced**† | Interval 10–15 d | — |
| RODMAN (2019)[12] | HDD | Full | Interval varies | FPR |
| SMARTer (2020)[13] | HDD | **Partial** | Interval 10 d | — (FPR 검토) |
| LSTM specificity (2020)[3] | **HDD** | **Balanced**† | Interval 15 d | FAR |
| LightGBM+CID (2021)[8] | HDD | Partial | Interval 10 d | Days in adv. |
| Model update (2021)[4] | Row | **Partial** | Interval 10 d | FAR |
| StreamDFP (2023)[10] | HDD | Full | Interval 20 d (30 d 검토) | Days in adv. |
| Survival deepnet (2024)[5] | Row | Full | Interval 15 d | FAR |
| DFPoLD (2025)[14] | Row | Partial | Interval varies | DPF |
| Explainable TS (2025)[11] | HDD | Partial | Interval 5 d | FAR |
| TCN-LSTM-Attn (2026)[6] | Window | Balanced | Interval 7 d | FAR |
| *(선택) Aggarwal (2018)* | *Window* | *Balanced* | *Interval 12 d* | *RUL (RMSE)* |
| **This work** | **Row·HDD** | **Full** | **First alarm 30 d** | **Precision, Recall, FAR, Median LT** |

굵게 표시한 것이 변경. † 표시는 우리 쪽 추론에 근거한 것이라 판단이 필요하다.

**Scope 집계 (권고안 기준):** Full 3 / Partial 5 / Balanced 3
**Scope 집계 († 미채택 시):** Full 3 / Partial 7 / Balanced 1

**어느 쪽이든 Full은 3편이다.** 이것이 2.2-3에 들어갈 숫자다.

## B. 판단이 필요한 새 항목

### B-0. 표 1 [3]의 Scope: `Partial` 유지인가 `Balanced`인가

**[3]은 둘 다 한다.** 음성을 언더샘플링해 disk 비율을 33:1에서 약 4:1로 맞추고(Balanced), 동시에 미고장 disk는 마지막 180일만 쓴다(Partial).

- **`Balanced` 권고**: 비율을 의도적으로 바꾼 것이 평가 모집단을 더 크게 왜곡한다. 그리고 2.2-3의 두 논지 중 Balanced 쪽 사례가 지금 [6] 하나뿐이라 얇다. 둘이 되면 "양성·음성 비율을 맞춘 표본" 문장이 단일 사례에 기대지 않게 된다.
- **`Partial` 유지**: 180일 절단도 실질적이고, Partial 과반이라는 A-10의 논지가 유지된다.

어느 쪽이든 A-10의 Full 3편은 바뀌지 않는다.

### B-1. [4]가 무작위 분할을 명시적으로 비판한다 — 대응할 것인가

[4] §I:

> "In real-world applications, however, **a temporal split is required** because the prediction model must be learned on data that is already known. In the case of a random split between training and testing, this is not guaranteed."

notion2 4.1-4는 HDD 단위 무작위 그룹 분할을 쓴다. **표 1에 있는 논문이 그 설계를 정면으로 비판하는 셈이다.**

다만 4.1-4에 이미 방어가 있다.

> 학습과 테스트가 같은 달력 구간을 공유하는 점은 두 평가에 동일하게 작용하므로 절차 간 비교에는 영향을 주지 않는다.

**이 방어는 옳다.** 본 연구는 예측 성능의 절대값을 주장하지 않고 같은 예측 결과에 두 판정 절차를 적용해 비교하므로, 분할 방식은 양쪽에 공통이다.

판단: **지금 문장으로 충분하다고 본다.** [4]를 명시적으로 언급하면 오히려 쟁점을 키운다. 다만 심사자가 물어올 자리라는 것은 알고 있을 것. 물어오면 위 문장이 답이다.

### B-2. [15]의 결론이 표면적으로 본 연구와 반대다 — 대응할 것인가

[15]는 두 평가가 다르다고만 하지 않고 **환자 단위 쪽이 낙관적으로 편향되었다**고 결론짓는다. 본 연구는 개체 단위 운영 평가가 필요하다고 주장하므로 표면적으로 반대다.

**대응 근거는 강하다.** [15]가 지목한 세 기제 중 둘을 본 연구가 구조적으로 피한다 — 최초 Alarm은 고장 시점을 모른 채 시간 순으로 정해지므로 outcome-dependent selection이 아니고, 창 밖의 오탐을 버리지 않고 Early로 판정한다. 셋째(대조군의 짧은 관측)는 3.2-6이 이미 인정하고 있다.

**선택지 셋.**

1. **아무것도 안 한다.** 현재 2.2-5는 "달라질 수 있다"로 중립 인용이라 틀리지 않는다. 지면 0.
2. **한 절만 덧붙인다.** [15]의 환자 단위가 사건 시점을 알고 고른 최댓값이라는 점을 짚으면, 본 연구의 최초 Alarm이 그와 다르다는 것이 드러난다. 지면 한 문장.
3. **적극적으로 쓴다.** [15]가 "neither patient-level nor prediction-level evaluation fully resolves the challenges"라고 스스로 닫으므로, 두 방식 다 부족하다고 그 분야가 인정한 자리에 본 연구가 세 번째를 놓는 구도로 만든다. 지면 두 문장.

**2번 권고.** 1번은 심사자가 원문을 보면 위태롭고, 3번은 좋지만 2.2-5가 이미 길다. 다만 3번의 인용문("neither ... fully resolves")은 값이 커서 지면이 허락하면 쓸 만하다.

### B-3. Alert muting 분석을 언급할 것인가

[15]는 경보 후 8시간 억제를 모사하면 AUC가 **0.53**으로 떨어진다고 보고한다. 서론 4가 "Alarm 억제, 최소 지속시간 및 재발생 대기"를 범위에서 제외한 것과 직결된다.

- **언급하면**: 억제 정책이 결과를 크게 바꾼다는 실증이 붙어, 정책을 판정에서 빼는 결정이 방어된다(3.2-2의 하한 논리와 같은 형태).
- **언급하지 않으면**: 아무 일도 없다. 다만 "가장 중요한 걸 뺐다"는 반론에 실증이 없다.

**판단 보류.** 서론 4는 범위를 선언하는 자리라 근거를 대기 시작하면 길어진다. 결론 6의 확장 방향에 얹는 편이 나을 수도 있다.

## C. 판단이 필요한 것

### C-1. Aggarwal et al. (2018)을 표 1에 넣을 것인가

위의 전용 절 참조. 넣으면 `Window / Balanced / Interval 12 d / RUL (RMSE)`이고, 표 1에서 **유일하게 하한(filter window 4일)이 있는 판정**이 된다.

- **넣을 때 이득**: A-7의 긍정 주장에 가장 강한 사례가 붙는다. 그리고 **3.2-2에도 재사용된다** — 하한을 두지 않는 이유로 든 "정책 의존성"을 이 논문이 실제로 보여준다(C-MAPSS는 evidence 20/filter 5, HDD는 12/4로 도메인마다 다르게 잡고, 본인들이 "domain dependent"라고 적는다). 지금 3.2-2의 이유는 논증뿐이고 사례가 없다.
- **안 넣을 때 위험**: 이 논문을 아는 심사자가 "이미 생존분석으로 중도절단을 다룬 HDD 연구가 있다"고 하면 답이 원고에 없다.
- **비용**: 표 1에 한 행, 2.2-5에 한 문장. 대신 **2.2-1·2.2-2·2.2-3의 편 수가 전부 11 → 12로 바뀐다.**

### C-2. 집계 단위 대비 실험 문장을 넣을 것인가 (3.1-1 또는 3.2-7)

[8]의 map function이 "HDD 단위 집계 + 구간 판정" 칸을 실제로 채우고 있다(Algorithm 1, threshold 1). 따라서 만들어낸 비교군이 아니다.

계산해 둔 값: seed 42 캐시 기준 **48개 실행 전부에서 HDD 단위 집계 Recall이 행 단위보다 높았다**(최소 증가폭 0.036). HGST 5%에서 행 0.793 → HDD 단위 0.872 → 운영 0.068.

즉 집계 단위를 올리는 것은 Recall을 **올리는** 방향이고, 떨어뜨리는 것은 최초 Alarm 판정뿐이다. 한 문장이면 "격차가 집계 단위에서 온 것 아니냐"는 반론이 닫힌다. **선택 사항.**

## D. 변경 없음으로 확정된 것

건드리지 말 것. 이미 한 번씩 흔들렸던 항목들이다.

| 대상 | 판정 |
| --- | --- |
| 표 1 [6] TCN-LSTM-Attn | `Window / Balanced / Interval 7 d / FAR` **전부 맞음.** 원문 전문 확인. 삭제 계획은 오류였음 |
| 표 1 [12] RODMAN | `HDD / Full / Interval varies / FPR` **전부 맞음** |
| 표 1 [8] LightGBM+CID | `HDD / Partial / Interval 10 d / Days in adv.` **전부 맞음** |
| 표 1 [5] Survival deepnet | `Row / Full / Interval 15 d / FAR` **전부 맞음** |
| 표 1 [14] DFPoLD | `Row / Partial / Interval varies / DPF` **전부 맞음** (본문만 고침) |
| 2.2-2의 RODMAN 서술 | 원문과 일치. 필요하면 29일·27일 실제 값 인용 가능 |
| 2.2-3의 Balanced 논지 | [6]이 뒷받침. 1:8은 양성 11.1%, 표 2의 실제 비율은 0.083~0.181% — **약 60~130배** |

## E. 남은 조사 요청 (넷)

전부 유료 접근이라 이쪽에서 못 받는다. 각 논문에 **꼭 필요한 질문만** 남긴다.

### ~~E-1. [4] Züfle, Erhard, Kounev, ICMLA 2021~~ — **완료**

원문 전문 확인. `Full` → `Partial`. 2.2-3의 숫자가 3편으로 정해졌다. 위 [4] 항목 참조.

### ~~E-2. [3] Hu, Han, Xu, Jiang, Qi, Procedia CS 176 (2020)~~ — **완료**

원문 전문 확인. Eval. unit `Window` → `HDD` 확정, Scope는 판단 필요(B-0). 위 [3] 항목 참조.

### ~~E-3. [11] Li, Zhou, Radhakrishnan, Kamarthi, EAAI 152 (2025)~~ — **완료**

원문 전문 확인. 네 열 전부 현행이 맞고 `Segments → Partial` 정정이 옳았다. 근거 문장 확보: "we designed our failure detection model to utilize only a moving window of the latest data" (§4.2). 위 [11] 항목 참조.

### E-4. [2] Botezatu, Giurgiu, Bogojeska, Wiesmann, KDD 2016

> DOI 10.1145/2939672.2939699

**Q3**만. 표 1에서 유일하게 구간이 범위(10–15일)로 적혀 있는데 왜 범위인지가 기록에 없다.

---

## 표 1 밖에서 추가로 확인이 필요한 것

참고문헌 전체를 다시 훑어 나온 것들이다. 표 1 항목은 아니지만 본문이 기대고 있다.

### E-5. [15] Tuttle et al. — **원문 확인 완료. 서지 세 군데 오류 + 논증상 중요 발견**

#### ⚠ 서지 정정 (세 군데)

| 항목 | 현재 참고문헌 | **원문** |
| --- | --- | --- |
| 플랫폼 | medRxiv | **Research Square** |
| 게시일 | May 2026 | **June 2nd, 2026** |
| DOI | 10.64898/2026.05.26.26354141 | **10.21203/rs.3.rs-9726164/v1** |

접두사 `10.21203`은 Springer Nature의 **Research Square**다. medRxiv(`10.1101`)가 아니다. 의심했던 접두사 문제가 맞았고, 플랫폼 자체가 다르다.

여전히 **동료심사를 거치지 않은 preprint**다. 경쟁이익 선언("MT reports royalty income from Genentech")과 Article 유형으로 보아 Springer Nature 계열 학술지 심사 중으로 보인다. 투고 전에 게재본이 나왔는지 한 번 더 확인할 것.

#### ✅ 인용 수치 두 개 확인

초록 원문: "Patient-level analyses suggested excellent discrimination **AUC 0.86**; [IQR 0.85, 0.87], whereas prediction-level analyses demonstrated lower performance **AUC 0.62**; [IQR 0.57, 0.65]."

2.2-5의 "환자 단위 0.86, 시점 단위 0.62"가 **정확하다.**

#### ⚠⚠ 논증상 위험 — [15]의 결론은 "개체 단위 쪽이 부풀려졌다"이다

이게 가장 중요하다. [15]는 두 평가가 다르다고만 말하지 않고, **환자 단위 쪽이 낙관적으로 편향되어 있으므로 시점 단위를 함께 쓰라**고 결론짓는다.

> "patient-level evaluation introduces **systematic optimistic bias**"
> "prediction-level analyses should more routinely complement traditional evaluations"

**표면적으로 본 연구와 반대 방향이다.** 본 연구는 행(시점) 단위로 부족하니 HDD(개체) 단위 운영 평가가 필요하다고 주장한다. 심사자가 [15]를 확인하면 "이 논문은 개체 단위가 부풀려졌다는데?"라고 물을 수 있다.

현재 2.2-5는 "시점 단위와 개체 단위의 평가 결과가 달라질 수 있다"로 중립적으로만 인용하므로 **틀린 것은 아니다.** 그러나 대응이 원고 안에 없다.

#### ✅ 그런데 대응이 아주 강하다 — 본 연구는 [15]가 지목한 기제를 구조적으로 피한다

[15]는 편향의 원인을 셋으로 진단한다.

> "patient-level approaches rely on **outcome-dependent prediction selection**—where the prediction used for evaluation is selected based on future knowledge of when the outcome occurs"
> "**false positive alerts that occur outside the pre-event window among cases are discarded**"
> "**shorter lengths of stay among controls further reduce opportunities for false positives**"

| [15]가 지목한 기제 | 본 연구의 처리 |
| --- | --- |
| 사건 시점을 알고 그 직전 창에서 **최댓값**을 고름 | **최초 Alarm을 시간 순으로 정한다.** 고장 시점을 모른 채 결정되고 최댓값이 아니다. outcome-dependent selection이 아예 성립하지 않는다 |
| 사건 전 창 밖의 오탐을 **버림** | **버리지 않고 Early로 판정한다.** 식 (1)의 분모에 들어간다 |
| 대조군의 짧은 관측이 오탐 기회를 줄임 | **3.2-6에서 명시적으로 인정**하고, 관측 조건이 다른 데이터셋 사이 값 비교를 하지 않는다 |

즉 [15]가 비판하는 것은 **개체 단위 평가 일반이 아니라 사후 선택에 기반한 특정 설계**다. 본 연구의 최초 Alarm 판정은 그 설계가 아니다.

**세 번째 기제는 HDD에서 방향이 반대일 가능성이 높다.** 병원에서는 패혈증이 없는 환자가 빨리 퇴원해 대조군 관측이 짧다. Backblaze에서는 고장난 HDD가 먼저 빠지고 **중도절단 HDD가 수집 종료까지 남으므로 관측이 더 길다.** 그렇다면 오탐 기회가 오히려 더 많아 운영 FAR이 과소가 아니라 과대 측정되는 쪽이다. ⚠ **미확인 — 데이터로 확인 가능하다.** 고장 HDD와 중도절단 HDD의 관측 일수 분포를 비교하면 된다.

#### ✅ [15]가 진단한 기제가 2.2-3·3.2-6과 같다

위 인용 중 둘째와 셋째가 본 연구가 독립적으로 도달한 것과 같다.

- "false positive alerts outside the pre-event window are discarded" ↔ **2.2-3** "평가 범위를 제한하면 ... False Alarm이 발생할 기회 자체가 감소하여"
- "shorter lengths of stay among controls reduce opportunities for false positives" ↔ **3.2-6** "관측이 길수록 각 HDD가 한 번이라도 Alarm을 낼 가능성만 높아지기 때문이다"

**다른 분야의 임상 연구가 같은 두 기제를 독립적으로 지목했다.** 3.2-6이 우리만의 특이한 관찰이 아니라는 근거다. 인용할 수 있다.

#### ✅ 인용 가능한 공백 선언

[15]가 스스로 이렇게 닫는다.

> "**neither patient-level nor prediction-level evaluation fully resolves** the challenges of assessing dynamic prediction models"

2.2-5의 공백 주장에 그대로 쓸 수 있는 문장이다. 두 방식 다 부족하다고 그 분야가 인정한 자리에 본 연구가 세 번째를 놓는 구도가 된다.

#### 그 밖에 쓸 만한 것

- **PPV 14.5%(환자 단위) 대 4%(시점 단위)**, 오탐 대 정탐이 **6:1 대 24:1**. AUC보다 운영 부담을 직접 보여주는 수치다.
- **Alert muting 분석**: 경보 후 8시간 억제를 모사했더니 AUC가 **0.53**으로 떨어져 무작위와 다를 바 없어진다. 서론 4가 "Alarm 억제, 최소 지속시간 및 재발생 대기"를 범위에서 제외한 결정과 직결된다. **억제 정책이 결과를 크게 바꾼다는 실증**이라 제외 결정을 방어해 주는 동시에, "가장 중요한 걸 뺐다"는 반론의 근거도 된다. 언급 여부는 판단 필요.
- ESMv2는 15분 간격으로 0–100 점수를 내고 임곗값 25 이상에서 경보한다. 임곗값을 기관이 정한다는 점도 본 연구의 임곗값 논의와 닿는다.

### ~~E-6. [12] Han et al. — arXiv 게재본 유무~~ — **완료. 교체 불필요**

검색 범위에서 학회·저널 게재본이 확인되지 않는다. 같은 연구진의 [10] StreamDFP는 ICDCS 2020 → IEEE TC로 갔지만 이 논문은 arXiv 프리프린트로 남아 있다. **현행 arXiv 인용이 맞다.**

### E-7. [16] Scully & Daluwatte — [17]과 묶여 인용되는데 미확인

> C. G. Scully, C. Daluwatte, "Evaluating performance of early warning indices to predict physiological instabilities," **Journal of Biomedical Informatics**, Vol. 75, pp. 14–21, Nov. 2017. DOI 10.1016/j.jbi.2017.09.008

[17]은 확인 완료다. 그런데 2.2-5와 3.2-2가 `[16][17]`로 **묶어서** 인용한다. [17]에만 있는 내용(다섯 범주, $T_{MAX}$/$T_{MIN}$, warning burden)을 [16]에도 있는 것처럼 돌리고 있지는 않은지 확인이 필요하다.

우선순위는 낮다. [17]이 정리판으로 보이고 두 논문의 저자가 겹친다. 다만 묶음 인용은 심사자가 짚기 쉬운 자리다.

---

## 조사 원칙 (두 번 데인 것)

**2차 자료로 원문을 대신하지 말 것.** [6] 하나에서 두 번 어긋났다. 한 번은 통째로 다른 논문(광산 장비)이었고, 한 번은 [8]의 내용이 절 번호와 인용문까지 갖춘 채 섞여 들어왔다. 둘 다 형식이 그럴듯했다.

**Q0("실험 데이터가 SMART/HDD인가")을 먼저 물을 것.** 확인 비용이 거의 없다.

**근거 문장은 원문 그대로, 위치와 함께 적을 것.** 지금 기록에서 "(위치 미기록)"이 붙은 칸들은 나중에 심사자가 물으면 다시 찾아야 한다.

---

# 2026-08-25 추가 — 원문 4편 재확인, 표 1 재설계, 프레이밍 전환

이 절은 하루치 조사와 측정을 한곳에 모은 것이다. **위의 기존 기록과 충돌하는 부분은 이 절이 최신이다.**

## A. 원문에서 새로 확정한 것

### [4] Model update (Züfle et al., ICMLA 2021) — 전문 확인

**평가 모집단을 고장 기준으로 자른다. 학습뿐 아니라 평가에도 적용된다.**

> "**For model learning and evaluation**, we labeled all instances of each HDD that failed within the next ten days as positive **while discarding all older instances of those HDDs**. All other instances were labeled as negative."

- 고장 HDD: 마지막 10일만 남기고 그 이전 관측을 전부 버린다. **고장 100일 전 관측은 평가에 들어가지도 않는다.**
- 미고장 HDD: 전체 유지
- 언더샘플링(γ, 최대 불균형비)은 **학습 전용**
- 시간 분할은 별도로 있다 — 첫 6개월 학습, 나머지 41개월을 월 단위 배치로 평가
- 데이터: ST4000DM000, 2017.2–2020.12, 35,170대 중 2,246대 고장. **테스트 prevalence 0.0639%**
- 지표: FDR = TP/(TP+FN), FAR = FP/(TN+FP), 관측 단위. ζ = FDR과 (1−FAR)의 조화평균 (자체 정의)

**표 1 네 칸 다 현행이 맞다** — `Row / Partial / Interval 10 d / FAR`

**중요한 함의:** 버려지는 것이 *가장 헷갈리기 쉬운 음성*이다. 곧 고장할 HDD의 아직 이른 관측이 음성 집단에서 빠지므로 FAR이 쉬운 모집단 위에서 측정된다. **Early가 데이터 단계에서 사라지는 사례**이고, 집계 방식이 아니라 표본 구성 단계의 소실이라 2.2에서 가장 직접적이다.

⚠ **주의:** 이 논문은 "a temporal split is required"를 주장하는 논문이다. 시간 분할의 필요성을 지적하면서 모집단 절단은 그대로 두는 구조인데, **이 대비를 지적하면 공격으로 읽힌다.** 절단 사실만 적고 평가는 붙이지 말 것.

**Scope 종류: Sample형** (고장 기준 구간 선택). Period형 아님.

---

### [8] LightGBM+CID (Wang, Yang, Yang, ISCC 2021) — 전문 확인

**Eval. unit = HDD 확정.** 2단계 구조다. 시점별 예측 → map function → disk-level 결과 → 그것으로 평가.

> "use the map function to act on the {y1...yN} to obtain Ypredict (**disk-level result**)"

> Algorithm 1, threshold = 1: "in the time window, **as long as one day's hard disk data is predicted to be about to failure, we think that hard disk is about to fail**"

OR 집계(계열 B) 확정.

**Scope = Partial, Sample형.** 전처리 1단계가 구간 절단이다.

> "**Intercept data**: Intercept the data of the same time window for each hard disk. In this paper, we take the time window size N as **ten days**."

- 데이터: Backblaze 2019년 전체, 4기종 77,615대 (Table I 합계는 69,926대 — 원문 불일치), 고장 1,606대
- **양성:음성 = 1:47**로 보고. 자연 비율보다 훨씬 낮다 → 모집단 축소의 방증
- 분할: **serial number 기준 무작위 80/20**
- 지표: AUC, f1, TPR(FPR<0.1%) — 즉 **FPR도 쓴다**

#### 시간 지표에 상한이 걸린다는 직접 인용을 확보했다

> "We set the time when the hard disk is **first predicted to be failed in the time window** as t_predict, and the actual hard disk failure time is t_failure. So the days that the failure can be predicted in advance Δinterval = t_failure − t_predict."

**"in the time window"가 원문에 있다.** t_predict가 창 안에서만 찾아지므로 Δinterval은 창 길이를 넘을 수 없다. 창이 10일이고, 이 분석에서만 30일로 늘려 7.75일 → 8.95일을 얻었다(Table IV).

**"판정 구간을 넘는 시간 지표는 정의상 나올 수 없다"는 주장의 인용 가능한 근거다.** 지금까지 추론이었던 것이 근거를 얻었다.

⚠ **모집단 서술 주의.** 원문은 "we only analyzed the results of the **failed disks** here", "the average Δinterval of **failed hard disks** in the testing set"이다. **"탐지에 성공한 HDD에 대해서만"이라고 쓰면 안 된다.** 탐지 실패분 처리는 미명시. 안전한 서술은 "고장이 확인된 HDD에 대해 사후적으로 계산된다"까지.

⚠ **창을 늘리면 값이 커지는 구조가 원문에 드러나지만 인용하지 말 것.** Horizon 민감도를 논문에서 빼기로 한 것과 같은 이유다(H 해킹 유도).

---

### [5] Survival deepnet (Ahmed & Green, NCA 2024) — 전문 확인. **표 1 수정 필요**

**Scope `Full` → `Partial`.** 미확인으로 남겨뒀던 항목의 답이 나왔다.

§4.1 Data preparation:

> "This paper **includes drives older than seven years** which includes **12,993 healthy drives and 4,889 failed drives**."

§6 Experiments:

> "First, **the HDD dataset** is divided into training (80%) and testing (20%) subsets."

**연령 필터가 80/20 분할보다 먼저 적용되므로 테스트 집합에도 걸린다.** KM 기반 informative sampling이 학습 전용이 아니었다.

- 분할은 **무작위** 80/20 (시간 분할 아님)
- 효과: 고장 드라이브 비율 4,889/17,882 = **27.3%**. Backblaze 자연 연간 고장률 1.83%의 약 15배
- 방법은 연령 필터지 비율 맞추기가 아니므로 `Balanced`는 아니고 `Partial`
- Verdict 15일 확인: "extends the failure horizon concept to a window of 15 days"
- unit `Row` 확인: FAR = FP/(FP+TN), 관측 단위
- 모델은 1D CNN + sigmoid. **생존분석은 표본 구성(KM)과 수명 분석(Cox)에만 쓰이고 CNN 평가는 전통적 혼동행렬**

**아이러니 하나:** 서두에서 선행 연구의 "misleading evaluation measures"를 비판하면서 자기 평가 모집단은 7년 초과로 제한한다. 2.2에서 쓸 수 있으나 공격이 되지 않게 사실만.

**Scope 종류: Sample형** (개체 선택).

---

### [12] RODMAN (Han et al., arXiv 2019) — 전문 재확인. 경계 처리 확정

#### observation window는 **학습 경계에만** 적용된다

기존 기록에 observation window가 3.2-4와 같은 기법이라고 적혀 있는데, **적용 위치를 명확히 해야 한다.**

T0가 학습/테스트 경계일 때, healthy 디스크의 Tn~T1(경계 직전 n일) 표본을 버린다. n = pre-failure period (A1 29일, B1 27일).

**테스트 창 안에는 이 처리가 없다.**

> "False positive rate (FPR): The ratio of the number of falsely predicted failed disks (which are indeed healthy) to the total number of healthy disks **in one-month testing**."

> "True positive rate (TPR): The ratio of the number of predicted failed disks to the total number of actual disk failures **in one-month testing**."

따라서 **5월 31일에 울리고 6월 1일에 고장하는 디스크는 5월 평가에서 healthy이고 그 알람은 FP다.** 경계 문제가 평가 쪽에 그대로 남는다.

**정리:** RODMAN은 개념을 알아채고 이름 붙이고(ambiguous samples) 학습에서 처리했으나, 평가에는 같은 처리를 하지 않았다. 본 연구는 판정에서 처리한다. **같은 개념, 다른 적용 위치.** 이 구분을 흐리면 안 된다.

#### Backblaze 평가는 슬라이딩 월 단위 평균이다

> "we use the first three months as the training phase and the fourth month as the testing phase. We slide one month for the next run... We have **15 and 37 runs** in total for H1 and S1, respectively. We present the **average results over all runs**."

Alibaba만 단일 달(10개월 학습 / 2018.5 테스트)이고 **Backblaze는 월 단위 반복이다.** 매 run 재학습한다.

#### 운영점은 고정 FPR이다

Alibaba **FPR 0.1%**, Backblaze **FPR 4.0%**. TPR 92.8% / 82.4%. Backblaze의 4%는 "to make the TPR comparable to that for the Alibaba dataset"라고 밝힌다.

#### Scope가 달력 구간 제한만인 것이 확정됐다

> "**our training** chooses the positive samples over the entire training phase, while choosing the negative samples only on the last day observed"

표본 축소는 학습 전용. 테스트는 그 달 전체. **다른 Partial들(Sample형)과 종류가 다르다.**

---

## B. 표 1 재설계 — 5열

**Oper. metrics를 없애고 Timing만 남겼다.** FAR과 FPR은 Scope·unit·Verdict가 정해지면 혼동행렬에서 자동으로 나오는 값이라 **선택이 아니라 결과**다. 다른 세 열과 급이 다르다. Timing은 별도로 재기로 한 선택이라 남는다.

| Study | Scope | Eval. unit | Verdict | Timing |
| --- | --- | --- | --- | --- |
| Disk replacement (2016)[2] | Balanced | HDD | Interval 10–15 d | — |
| RODMAN (2019)[12] | Partial | HDD | Interval varies | — |
| SMARTer (2020)[13] | Partial | HDD | Interval 10 d | — |
| LSTM specificity (2020)[3] | Balanced | HDD | Interval 15 d | — |
| LightGBM+CID (2021)[8] | Partial | HDD | Interval 10 d | Days in adv. |
| Model update (2021)[4] | Partial | Row | Interval 10 d | — |
| StreamDFP (2023)[10] | Full | HDD-day | Interval 30 d | Days in adv. |
| Survival deepnet (2024)[5] | Partial | Row | Interval 15 d | — |
| DFPoLD (2025)[14] | Partial | Row | Interval varies | DPF |
| Explainable TS (2025)[11] | Partial | HDD | Interval 5 d | — |
| TCN-LSTM-Attn (2026)[6] | Balanced | Window | Interval 7 d | — |
| **This work** | **Full** | **HDD** | **First alarm 30 d** | **Median LT** |

### 표에서 직접 읽히는 것 셋

**(Full, HDD) 교차가 비어 있다.** Full은 [10] 하나인데 unit이 HDD-day이고, HDD 여섯 편은 전부 Partial 아니면 Balanced다. This work만 그 칸에 들어간다.

| | Full | Partial | Balanced |
| --- | --- | --- | --- |
| **HDD** | **없음** | [12] [13] [8] [11] | [2] [3] |
| HDD-day | [10] | | |
| Row | | [4] [5] [14] | |
| Window | | | [6] |

**Timing이 세 칸만 채워진다.** 그리고 [8]의 정의가 "in the time window"라 Verdict 열의 구간이 상한이다. 두 열을 나란히 놓으면 상한의 출처가 보인다.

**Verdict만 갈리지 않는다.** 나머지 세 열이 제각각인데 판정 형태는 전부 Interval이다. 열한 편이 열한 조합으로 완전히 겹치는 행이 없다.

**This work은 Timing이 Verdict 안에 있다.** 다른 연구는 구간 판정으로 성공을 정한 뒤 시간을 별도로 재는데, 본 연구는 판정 자체가 시간 검사다.

### Scope 재설계는 하지 않기로 했다

RODMAN(달력 구간)과 나머지(고장 기준 구간·개체 선택)를 값으로 구분하려면 `Full / Period / Sample / Balanced` 네 값이 필요한데, **[4]가 Period형이면서 Sample형이라 배타적이지 않다.** 심각도 순 할당 규칙을 세우면 Period가 RODMAN 한 편짜리 값이 된다.

**결론: 세 값 유지하고 RODMAN이 다른 이유는 본문 한 문장으로 푼다** — 제한한 축이 표본이 아니라 달력 구간이며 그 안에서는 모집단을 재구성하지 않았다는 것.

### Eval. unit 열이 새 의미를 얻었다

**분모가 disk인가 아닌가**의 축이다. HDD 여섯 편 + HDD-day 한 편은 disk 수로 답하고, Row 셋 + Window 하나는 관측 행이 분모라 **점검 대상 대수로 옮겨지지 않는다.**

실측 근거: 행 단위 FAR 0.1%가 HGST에서 점검 부담 몇 퍼센트에 대응했다.

---

## C. 측정으로 확정한 것

### 평가 창이 H 이하이면 Early가 구조적으로 발생하지 않는다

`analysis/lead_time_analysis/run_window_length_effect.py` / 결과 `results/lead_time_analysis/window_length_effect.csv` (seed 42, 4모델 × 3기종)

각 고장 HDD의 마지막 W일만 보고 같은 최초 Alarm 판정을 적용했다.

```
창 길이           30d     45d     60d     90d    180d    365d    730d    전체
Recall 하락폭    0.0%   91.3%   90.8%   90.3%   98.2%  100.0%  100.0%  100.0%   HGST
                 0.0%   93.6%   93.4%   93.3%   93.2%   93.1%   93.0%   93.0%   Seagate
                 0.0%   96.0%   96.0%   96.0%   95.7%   95.6%   95.5%   95.5%   Toshiba
Early 비중       0.00    0.98    0.98    0.98    0.99    1.00    1.00    1.00   HGST
```

**30일 창에서 Early가 정확히 0이고 Recall이 완전히 단조다.** 다섯 범주가 세 범주로 무너진다. 45일부터 즉시 나타난다 — 점진적이 아니라 W>H가 되는 순간이다.

**RODMAN에 그대로 적용된다.** 창이 한 달이고 pre-failure 구간이 29일이니 W ≈ H. **그 설정에서 Early는 나올 수 없다.** 못 본 게 아니라 구조적으로 불가능하다.

우열 주장이 전혀 없는 형태라 논문에 쓰기 좋다. 3장 [평가 구간과 오탐 부담]에 붙이면 5분류의 필요조건이 명시된다.

### 왜 전역에서 Recall이 비단조인가 — 구조

행 단위에서는 한 행이 τ에서 양성이면 더 낮은 τ에서도 양성이라 **나가는 표본이 없다.** 그래서 단조.

전역에서는 각 고장 HDD가 **Missed → On-time → Early**를 지난다. On-time은 τ의 한 구간 안에서만 유지되는 **통과 상태**다. 위쪽 끝은 알람을 못 내는 지점, 아래쪽 끝은 첫 알람이 H보다 이르게 되는 지점. 구간들의 겹침이 Recall이므로 봉우리가 생긴다.

스윕 코드의 `cut_on` / `cut_off`가 이 구조다.

### 봉우리 위치는 불안정하다

EAR 축 기준.

```
모델별 봉우리   HGST 3.11~11.02% (3.5배)   Seagate 4.57~9.34% (2.0배)   Toshiba 4.32~5.69% (1.3배)
run별 IQR       HGST 3.87~9.87%            Seagate 5.03~8.84%           Toshiba 4.57~6.01%
전체 범위       HGST 1.52~20.2%
```

⚠ **곡선의 봉우리와 run별 봉우리의 중앙값이 다르다.** HGST에서 중앙값 곡선은 3.66%에서 최대인데 개별 run 봉우리의 중앙값은 7.10%다. argmax는 중앙값 연산과 교환되지 않는다. **본문에서 봉우리 위치를 인용하면 개별 run과 안 맞는다.**

**결론: 봉우리를 "최적 운영점"으로 지목하지 말고 "상한"으로 쓸 것.** 봉우리 너머는 어느 지표도 개선되지 않으므로 운영점을 그보다 오른쪽에 둘 이유가 없다. 그 왼쪽 어디에 둘지는 Precision과 확보 시간을 보고 정한다. Recall만 최대화하는 게 최적이라는 건 규범적 주장이고 "최선의 기준"을 주장하지 않기로 한 것과 부딪힌다.

### 월 단위 평가와 전역 평가는 다른 운영점을 선호한다

`run_monthly_vs_global.py` / `monthly_vs_global.csv` (seed 42, 4모델). 각 (디스크, 달)을 표본으로 하고 Youden J로 최적점을 잡았다.

| | τ 월 단위 | τ 전역 | 배율 | 월 TPR | 월 FPR | 전역 Recall @월최적 | @전역봉우리 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| HGST | 0.0029 | 0.0838 | 29배 | 0.889 | 6.4% | 0.077 | 0.151 |
| Seagate | 0.0046 | 0.0099 | 2.2배 | 0.698 | 3.7% | 0.287 | 0.316 |
| Toshiba | 0.0011 | 0.0043 | 4.1배 | 0.691 | 3.7% | 0.136 | 0.213 |

**월 단위는 훨씬 낮은 임곗값을 선호하고 TPR이 0.69~0.89로 높게 나온다.**

⚠ **"과소평가"라고 쓰면 안 된다.** 월 단위는 자기 모집단(디스크-월)에서 정확히 잰다. 모집단이 다른 것이지 틀린 게 아니다. **디스크-월에서는 TPR이 단조이고 디스크에서는 Recall이 비단조다** — 같은 예측, 같은 임곗값, 모집단만 다르다. 이 형태로만 쓸 것.

⚠ 이 구현은 매 run 재학습하지 않는다(RODMAN Backblaze는 한다). 최적점을 J로 잡은 것도 임의 선택이며 **원문은 고정 FPR(0.1% / 4.0%)을 쓴다.**

⚠ 단일 달 재현은 불가능하다. 테스트 분할의 **월별 고장이 중앙값 1~3건**이다(HGST 93개월 162대, Seagate 103개월 225대, Toshiba 94개월 209대). RODMAN이 한 달로 끝낼 수 있었던 건 Alibaba 300만 대를 썼기 때문이다.

---

## D. 리버설 대비 자료 — 논문에는 넣지 않기로 한 것들

전부 `results/lead_time_analysis/` 아래에 있다.

| 파일 | 무엇을 답하나 | 왜 뺐나 |
| --- | --- | --- |
| `scope_effect.csv` | 평가 모집단 축소 시 보고되는 Precision. 행 단위 Recall 0.3 고정에서 Balanced 0.99 대 Full 0.11 | 선행 연구 공격이 됨. "왜 값이 이렇게 낮냐"에 답할 때만 사용 |
| `window_optimism.csv` | 수집 기간을 자르면 측정되는 부담이 실제의 몇 %인가. 1년이면 4~19% | 관측 길이가 값을 바꾼다는 건 산수라 말로 넘김 |
| `horizon_sensitivity.csv` | H를 5~30일로 바꿀 때 Recall 1.9~3.6배 | **H 해킹 레시피가 됨. 절대 싣지 말 것** |
| `monthly_vs_global.csv` | 월 단위 최적점과 전역 봉우리의 차이 | 비교 프레임. 넣으려면 "선호하는 운영점이 다르다"까지만 |

---

## E. 프레이밍 전환 기록 (2026-08-25 확정)

**논문 이름이 바뀐다.** 「운영 환경 기반 평가」 → **「최초 Alarm 기반 평가」**. RODMAN·StreamDFP가 운영 현장에 더 가깝고, 본 연구는 우열이 아니라 다른 질문을 재기 때문이다. "운영 환경 기반"은 우열을 주장하는 이름이다.

**문제의식 세 줄**

1. HDD 고장 예측은 산업 요구로 발전한 분야이며 실무 투입이 최종 목적. 운영에서의 관심사는 alarm이 얼마나 울리는지, 얼마나 유효한지, 얼마나 일찍 울리는지.
2. 선행 연구의 평가 방식은 각자 다르지만 혼동행렬 기반 평가를 위해 시점 정보를 보존할 수 없는 형태. 그중 HDD 집계 일부는 앞의 둘을 효과적으로 알 수 있으나 얼마나 일찍 울리는지에는 한계.
3. 본 연구는 alarm 리드타임과 우측 중도절단 처리를 기반으로 5분류하여 평가하는 최초 alarm 기반 평가를 제시. 공개된 데이터 전부로 이 모델이 이 데이터셋에서 어느 정도의 퍼포먼스를 보였는지 평가. 같은 종류 다른 HDD에 대한 일반화 성능. 운영에서 의미를 가지는 지표도 산출.

**핵심 전환 셋**

**"전체 기간을 쓰면 중도절단이 유입된다"는 틀렸다.** 중도절단은 어느 평가에나 있다. 한 달만 봐도 그 달에 안 죽은 디스크가 나중에 죽을지는 모르고, RODMAN은 그걸 healthy로 놓고 FPR 분모에 쓴다. **실제 강제는 판정 방식에서 온다** — 구간 소속으로 판정하면 음성으로 두고 산술이 돌아가지만, Lead Time으로 판정하면 고장 시점이 없는 HDD의 LT를 정의할 수 없어 같은 처리가 불가능하다.

**행 단위 평가는 운영에서 아무것도 정해주지 않는다.** 분모가 관측 행이라 점검 대상 대수로 옮겨지지 않는다. "앞의 둘에 답해 왔다"는 것은 **HDD를 단위로 삼은 연구에만** 해당한다.

**본 연구의 위치는 배치 판단 시점이다.** 월간 운영 계획은 RODMAN 형태가 맞고, 전체 이력 평가는 "이 모델을 켜면 알람 스트림이 어떤 모양이 될 것인가"에 답한다. 둘은 보완 관계이며 경쟁하지 않는다.

**주장하지 않는 것** (사용자 확정): 선행 방법이 틀림(문제 정의가 다른 것), 지표의 대체 불가능성(있는 개념들을 모은 것), 최선의 기준(한계 명확), 실무 유용성(완벽한 실무 환경이 아님), 본 평가의 채택 권고(영업 아님).

**추가로 하지 말 것**: 모델 순위·배치 지침, H 민감도, 데이터셋 간 우열, 봉우리를 최적 운영점으로 지목.

---

## F. 지표 변경 — FAR → EAR

식 (3)이 바뀌었다. EAR = (N_E + N_CE) / N_all

**근거:** 3장이 "중도절단은 정상을 뜻하지 않는다"고 명시해놓고 식 (3)이 그 집합을 음성 모집단으로 쓰는 것은 자기 주장과 충돌한다. 음성은 시간에 따라 정해지고 안 죽는 HDD는 없으므로, 양성·음성 분할 자체가 관측 창이 만든 인공물이다. EAR은 아무도 분류하지 않고 전체 자산 중 대응할 수 없는 시점에 울린 비율만 센다.

**이중 계상 문제가 사라진다.** Early를 오탐 분자에 못 넣었던 이유가 "같은 HDD가 양성 분모와 음성 분모에 동시에"였는데, EAR에는 음성 분모가 없다.

**EAR은 임곗값에 대해 단조다** — Early는 흡수 상태이고 CE는 증가만 한다. 360개 run 전부 확인. 그래서 축으로 쓸 수 있다.

**Precision은 EAR 축 위에서 Recall과 함께 결정되는 파생값이다.** Precision = (Recall·f) / (Recall·f + EAR), f는 고장 관측 HDD 비율. 표에 남겨도 되나 독립된 정보는 아니다.

⚠ **`METRIC_DESIGN.md`가 낡았다.** "Early를 FAR에 넣으면 같은 HDD가 양성·음성 분모에 동시에 들어간다"는 설계 근거가 EAR에서는 성립하지 않는다.

**지표 이름에서 `op` 첨자를 뺐다.** 행 단위 평가가 논문에서 빠져 구분할 상대가 없고, EAR은 첨자가 없어 넷 중 둘만 첨자가 되면 어수선하다. 대신 표 캡션에 "평가 범위를 제한한 연구의 동명 지표와 직접 비교되지 않는다"를 넣는다 — 캡션은 표와 함께 이동하므로 첨자보다 낫다.

---

## G. 분석 스크립트 목록

전부 `analysis/lead_time_analysis/` 아래. 캐시(`results/prediction_cache/`)만 있으면 재학습 없이 돌아간다.

| 스크립트 | 무엇을 하나 | 제약 |
| --- | --- | --- |
| `run_burden_sweep.py` | EAR 축 전 구간 스윕. 판정 구성·네 지표·LT 백분위 | 4모델 × 30 seed 전부 |
| `plot_burden_outcome.py` | 그림 — EAR에 따른 판정 구성 | |
| `plot_burden_leadtime.py` | 그림 — EAR에 따른 LT 백분위 밴드 | |
| `make_result_tables.py` | 표 3 생성 | |
| `run_window_length_effect.py` | 창 길이가 Early·단조성에 미치는 영향 | **seed 42만** (창 안 최댓값이 필요해 step 캐시로 불가) |
| `run_monthly_vs_global.py` | 월 단위 최적점 대 전역 봉우리 | **seed 42만** |
| `run_horizon_sensitivity.py` | H 민감도 | 논문 제외 |
| `run_scope_effect.py` | 모집단 축소 효과 | 논문 제외 |
| `run_window_optimism.py` | 수집 기간 절단 효과 | 논문 제외 |

**step 캐시(`disk_steps`)는 누적 최댓값 증가점만 담는다.** 창 안의 최댓값이 필요한 분석(월 단위, 창 길이)은 `full_preds`가 있어야 하고 그건 seed 42 / test만 있다.
