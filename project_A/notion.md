한국정보기술학회 게재, KCI 등재 목표

- 규칙
    
    <aside>
    
    기존에 발표된 논문을 참조할 경우, 적절한 위치에 대괄호(예: [1]-[4])를 사용하여 순차적으로 번호를 매기고, 원고 말미에 번호순으로 정리된 참고문헌 목록을 제공해야 합니다.
    
    </aside>
    
    <aside>
    
    그림: 모든 그림은 아라비아 숫자로 순차적으로 표시해야 합니다. 그림의 내용을 나타내는 명확한 제목을 포함해야 합니다. 그림의 크기는 균일하게 유지해야 합니다. KIIT는 JPG/TIFF 파일 형식의 흑백 및 컬러 그림 모두 권장합니다. 그림의 선명도를 높이기 위해 600dpi/1200dpi 해상도를 사용하십시오. 그림이 여러 개의 그래프로 구성된 경우, 각 그래프 바로 아래에 (a), (b), (c) 등으로 표시하고 본문에서는 Fig. 4(a), Fig. 4(b) 등으로 참조하십시오. 모든 그림과 글자는 원본의 50% 축소 후에도 읽을 수 있는 크기여야 합니다. 그림 설명에서는 각 문장의 첫 단어의 첫 글자만 대문자로 표기하십시오. 사진은 논문에 필수적인 경우에만 허용되며, 추가 요금이 부과됩니다.
    
    </aside>
    
    <aside>
    
    표: 모든 표는 아라비아 숫자로 순차적으로 번호를 매겨야 합니다. 표의 내용을 나타내는 명확한 제목을 포함해야 합니다. 표 제목의 첫 단어 첫 글자는 대문자로 작성해야 합니다. 필요한 추가 정보는 표 각주에 포함할 수 있습니다. 권장 글꼴은 9pt의 중세 고딕체입니다. 표에는 세로선이 없어야 하며, 표 제목은 본문을 참조하지 않고도 이해할 수 있어야 합니다.
    
    </aside>
    
    <aside>
    
    방정식 및 기호: 모든 방정식과 수식은 별도의 줄에 입력하고 오른쪽 괄호 안에 아라비아 숫자로 순차적으로 번호를 매겨야 합니다. 방정식이나 수식을 참조할 때는 괄호 안에 인용 부호를 붙여야 합니다.
    
    </aside>
    
    <aside>
    
    참고
    
    (1) Journal Article
    
    D. W. Ryoo and CS Bae, "Design of The Wearable Gadgets for Life-Log Services based on UTC", IEEE Transactions on Consumer Electronics, Vol. 53, No. 4, pp. 1477-1482, Nov. 2007. https://doi.org/10.1109/TCE.2007.4429240.
    
    (2)
    
    HK Hartline, AB Smith, 및 F. Ratlliff 저서, "망막의 억제적 상호작용", 감각생리학 핸드북, Springer-Verlag, pp. 381-390, 1972.
    
    (3)
    
    Y. Yamamoto, S. Machida, 및 K. Igeta 저서, "향상된 자발적 방출을 갖는 마이크로 공동 반도체", 제16회 유럽 광통신 학회 회의록, 암스테르담, 네덜란드, pp. 3-13, 1990년 9월.
    
    (4)
    
    C. E. Larsen, R. Trip, 및 CR Johnson 저서, 심장의 전기생리학과 관련된 절차 방법. 미국 특허 5,529,067, 1995년 6월 25일.
    
    (5) 웹 URL
    
    INRIA 개인 데이터 세트: http://pascal.inrialpes.fr/data/human/. [접속일: 2015년 11월 4일]
    
    </aside>
    

orcid 가입

구상

kci 평가기준 

선행연구 정리

하드 디스크 드라이브 모델 선정 기준

HDD 모델 선택

- 용어 정리
    
    
    | 용어 | 뜻 |
    | --- | --- |
    | 행 단위 평가 (Row-level evaluation) | 각 관측 행을 독립적인 분류 대상으로 평가하는 방식 |
    | 운영 환경 기반 평가 (Operational evaluation) | HDD의 시간 순차적 관측 과정을 유지한 채, 최초 Alarm과 실제 고장 또는 관측 종료 시점의 관계를 기준으로 HDD 단위 성능을 산출하는 평가 방식 |
    | 온라인 추론 (Online Inference) | 각 HDD의 관측값을 시간 순으로 입력하여 각 시점에서 고장 예측을 수행하는 과정 |
    | 최초 Alarm (First Alarm) | 한 HDD에서 의사결정 임곗값을 최초로 초과하여 발생한 Alarm |
    | Lead Time | 최초 Alarm 발생 시점부터 실제 고장 발생 시점까지의 시간 |
    | 예측 기간 (Prediction Horizon) | 최초 Alarm이 On-time으로 인정되는 고장 이전의 시간 범위 |
    | 우측 중도절단 (Right-censoring) | 관측 종료 시점까지 고장이 발생하지 않아 실제 고장 시점을 관측할 수 없는 상태 |
    | On-time Alarm | 예측 기간 이내에 발생한 최초 Alarm |
    | Early Alarm | 고장보다 예측 기간 이상 앞서 발생한 최초 Alarm |
    | Censored Early Alarm | 실제 고장 시점이 관측되지 않은 우측 중도절단 HDD에서 평가 구간 내 발생한 최초 Alarm |
    | Missed Failure | 고장이 관측된 HDD에서 최초 Alarm이 발생하지 않은 경우 |
    | Censored No Alarm | 우측 중도절단 HDD에서 평가 구간 내 최초 Alarm이 발생하지 않은 경우 |
    | HDD-level Precision | 최초 Alarm이 발생한 전체 HDD 중 해당 Alarm이 On-time으로 분류된 HDD의 비율 |
    | HDD-level Recall | 실제 고장 HDD 중 On-time Alarm으로 탐지된 HDD의 비율 |
    | HDD-level FAR | 우측 중도절단 HDD 중 평가 구간 내 최초 Alarm이 발생한 HDD의 비율 |
    | Median Lead Time | 최초 Alarm이 발생한 고장 관측 HDD의 Lead Time 중앙값 |
- 주장과 근거
    
    **주장**
    **행 단위 분류 성능은 운영 성능의 필요조건이지만 그 값에서 운영 성능이 정해지지 않는다. 따라서 운영에 배치할 모델은 행 단위 지표만으로 선택할 수 없으며, 행 단위 평가에 더해 우리의 제안 방식인 운영 환경 기반 평가도 측정해야 한다.**
    
    기여
    
    운영 환경 기반 평가 방법론을 제안함.
    
    - 점검과 교체는 HDD 단위로 일어나며, 무엇이 유효한 Alarm인지 모르기 때문에 발생 시점에는 유효성을 구별할 수 없어 발생 순서 외에 선별 기준이 없다
        - Alarm의 경향성을 전지적인 관점에서 평가하는게 아닌, 실제 운영자의 입장(가장 단순한 최초 Alarm 기준)에서 평가하는 방법론.
    1. **집계 단위** — 성능을 무엇 하나당으로 세는가 (행 / 윈도우 / HDD)
    2. **평가 범위** — 관측 이력 중 어디를 평가에 넣는가 (전체 / 고장 직전 구간 / 균형 표본)
    3. **판정 규칙** — 무엇을 맞혔다고 보는가 (구간 안에 양성이 하나라도 있었나 / 처음 언제 울렸나)
    - 행 단위 평가 → 집계 단위도 다름 / 판정 규칙에 시간 고려 안함
    - 윈도우 단위 평가 → 집계 단위도 다름 / 판정 규칙에 시간 고려 안함
    - HDD 단위 평가 → 집계 단위는 맞음 / 판정 규칙이 실제 알람 부담을 고려하지 않음
    
    행 단위 값에서 제안 방식의 값이 정해지지 않음을 실증함.
    
    - 두 평가의 성능이 서로 다르게 측정되며, 행 단위 값에서 운영 값이 정해지지 않는다.
    - 두 평가에서 최고 성능을 기록한 모델이 달라진다

# HDD 고장 예측 모델의 운영 환경 기반 평가

# Operational Evaluation of HDD Failure Prediction Models

## 요약

본 연구에서는 HDD별 관측 이력을 시간 순으로 처리해 최초 Alarm과 우측 중도절단을 반영한 다섯 가지 판정 범주를 도출하고, 이를 바탕으로 HDD 단위 지표를 산출하는 운영 환경 기반 평가 방법을 제안한다. 유지보수는 HDD 단위로 이루어지지만, 기존 연구의 상당수는 각 관측 행이 고장에 근접했는지에 대한 분류 성능만을 측정하므로 Alarm의 적시성과 운영 부담을 반영하지 못한다. Backblaze 데이터셋의 HDD 3종을 4가지 모델로 예측하여 두 평가에 각각 적용하는 과정을 반복한 결과, 행 단위 평가를 기준으로 선정한 임곗값은 제안된 평가에서 의도한 오탐률 제약을 크게 초과했다. 또한 평가 지표의 값이 달라졌을 뿐 아니라 예측 모델 간 순위도 보존되지 않았다. 이는 행 단위 분류 성능만으로는 운영 성능을 충분히 대변할 수 없음을 보여준다. 따라서 운영 배치를 고려하는 모델은 행 단위 평가 외에 운영 환경 기반 평가를 추가로 적용해야 한다.

We propose an operational evaluation that processes each drive’s observation history chronologically, derives five outcome categories reflecting first alarms and right-censoring, and computes drive-level metrics. Although maintenance is performed at the drive level, much prior work evaluates only row-level proximity-to-failure classification, capturing neither alarm timeliness nor operational burden. Repeating the procedure across three Backblaze drive models and four prediction models, with both evaluations applied to each, showed that the row-level-selected threshold far exceeded the proposed evaluation’s intended false-alarm constraint; metric values differed and model rankings were not preserved. Thus, row-level performance alone cannot represent operational performance, and deployment candidates should be assessed using operational evaluation in addition to row-level evaluation.

Keywords: HDD failure prediction, SMART, operational evaluation, first alarm, right-censoring

## 1. 서론

데이터 저장 수요의 증가와 함께 하드 디스크 드라이브(Hard Disk Drive, HDD)는 대규모 저장 시스템과 데이터센터에서 널리 활용되고 있다. HDD 고장은 데이터 손실과 서비스 가용성 저하로 이어지므로 고장을 사전에 탐지하여 유지보수 의사결정에 활용하려는 연구가 지속되어 왔으며[1], HDD의 상태와 오류 관련 속성을 기록하는 SMART(Self-Monitoring, Analysis and Reporting Technology) 데이터가 그 주요 입력으로 사용되어 왔다[2].

*→ **HDD 고장은 중요한 운영 문제이고, SMART 기반 고장 예측은 이를 해결하기 위한 주요 연구 분야이다.***

HDD의 SMART 데이터는 HDD별로 시간에 따라 반복적으로 수집된다. 각 HDD는 도입된 날부터 하루 한 번 기록되어 수백에서 수천 개의 관측 행을 남기며, 그 기록은 고장으로 교체될 때 끝나기도 하지만 고장 없이 운용에서 빠지거나 수집 기간이 종료되어 끊기기도 한다. 실제 운영에서 유지보수와 교체의 대상이 되는 것은 이 개별 관측 행이 아니라 HDD 자체이다. 그럼에도 기존 HDD 고장 예측 연구는 개별 행 또는 특정 시점 윈도우의 예측 결과를 중심으로 Precision, Recall, F1-score, AUC 및 FAR 등을 산출하는 경우가 많아[3]-[6], 한 HDD에서 반복적으로 발생한 양성 예측이 여러 개의 개별 결과로 집계된다.

→ **점검과 교체의 대상은 개별 관측 행이 아니라 HDD이지만, 기존 평가는 각 행을 독립적인 결과로 집계하고 고장 전 일정 구간을 맞혔는지로 판정해 왔다.**

운영에서는 고장을 한 번이라도 탐지했는지뿐 아니라 최초 Alarm이 언제 발생했는지, 고장 전 충분한 대응 시간이 확보되었는지, 고장이 관측되지 않은 HDD에서 불필요한 Alarm이 발생했는지가 중요하다[7]. 고장보다 지나치게 이른 Alarm은 정비 의사결정에 유효한 Alarm으로 해석되지 않을 수 있으며, 수집이 끝날 때까지 고장이 확인되지 않은 HDD는 이후의 고장 가능성을 확인할 수 없어 별도의 처리가 필요하다. 따라서 HDD 고장 예측을 평가하려면 개별 관측 행의 분류 결과뿐만 아니라, 각 HDD의 관측 이력을 시간 순으로 처리하고 Alarm 발생과 고장 또는 관측 종료 시점의 관계를 반영해야 한다.

→ **운영자는 Alarm이 발생한 시점에 그 유효성을, 관측이 끝나는 시점에 이후의 고장 여부를 알 수 없으므로, 평가도 같은 정보 조건에서 이루어져야 실제 점검 부담과 대응 시간이 산출된다.**

본 연구의 목적은 HDD의 시간적 관측 과정을 반영한 운영 환경 기반 평가 방법을 제안하고, 기존 행 단위 평가와 비교하여 평가 절차에 따른 성능 차이를 분석하는 데 있다. 여기서 '운영 환경 기반 평가'는 ① 관측 이력의 시간 순 온라인 추론, ② 예측확률의 Alarm 변환, ③ HDD 단위 최초 Alarm 판정, ④ 고장이 확인되지 않은 채 관측이 끝나는 우측 중도절단 구간의 처리를 평가 절차에 포함하는 것을 뜻한다. 반면 Alarm 억제, 최소 지속시간 및 재발생 대기, Alarm 단계화, 교체·점검 비용과 인력 운용 등 실제 운영의 대응 정책은 포함하지 않으므로, 본 연구의 평가 대상은 모델의 예측이 HDD 단위 Alarm으로 전환되기까지의 과정이다. 이하에서 점검 부담은 그 비용이 아니라 최초 Alarm으로 점검 대상이 되는 HDD의 대수를 뜻하며, 이 값은 대응 정책과 무관하게 정해진다.

*→ **본 연구의 운영 환경 기반 평가는 ‘예측 → Alarm → HDD 판정’까지를 대상으로 하며, 이후의 운영 대응 정책은 범위에서 제외한다.***

본 논문은 행 단위 결과로부터 운영 성능이 정해지지 않으며 그 차이가 지표 값에 그치지 않고 모델 선택에까지 이른다는 것을 보이고, 운영에 배치할 모델은 행 단위 지표만으로 선택할 수 없으므로 제안한 평가를 함께 사용해야 함을 주장한다. 이를 위해 동일한 예측 결과에 두 평가 절차와 두 임곗값 기준을 적용하여, 관찰된 차이를 평가 절차에 귀속시킬 수 있는 조건에서 비교한다. 2장에서 선행 연구의 평가 프로토콜을 정리하고, 3장에서 비교 기준선인 행 단위 평가와 제안하는 운영 환경 기반 평가를 정의하며, 4장에서 실험 구성과 임곗값 선정 절차를, 5장에서 지표 수준과 모델 선택 수준의 결과를 제시하고, 6장에서 결과와 한계를 정리한다.

*→*  **행 단위 결과로부터 운영 성능이 정해지지 않으므로 운영에 배치할 모델은 행 단위 지표만으로 선택할 수 없으며, 본 연구는 이를 동일한 예측 결과의 비교로 검증한다.**

---

## 2. 관련 연구

### 2.1 HDD SMART 기반 고장 예측 연구

HDD 고장 예측 연구는 고장 기록과 SMART 속성의 관계를 분석하는 통계적·신뢰성 분석에서 시작되었다. 대규모 관측에서 상당수 고장은 SMART 속성에 뚜렷한 이상이 나타나지 않은 채 발생하여, 단일 속성의 임곗값 규칙만으로는 예측이 어렵다는 점이 확인되었다[1]. 이후 SMART 데이터로 고장이나 교체 여부를 직접 예측하는 머신러닝 이진 분류로 확장되어 Random Forest, SVM, XGBoost, LightGBM 등이 적용되었고[2][8], Sliding Window, LSTM, CNN-LSTM과 같이 시간적 특성을 반영하는 모델이 뒤따랐다[3][6]. 최근에는 클래스 불균형[9], Concept Drift와 온라인 학습[10], 설명 가능한 예측[11]으로 범위가 넓어졌으나, 성능을 어떤 단위로 집계할 것인지는 선행 연구의 관행을 따른다. 평가 대상은 여전히 개별 행 또는 윈도우이며[3]-[6], 최초 Alarm의 시점이나 HDD 단위의 오탐 부담은 측정되지 않는다.

→ **HDD 고장 예측은 통계적·신뢰성 분석에서 머신러닝 이진 분류를 거쳐 시간적 특성을 반영하는 모델로 발전해 왔다.**

두 계열은 입력 구성에서 갈린다. 트리 기반을 포함한 일반 분류기는 각 시점의 SMART 속성 벡터를 독립적인 표본으로 다루고, 순환신경망 계열은 직전 일정 기간의 관측을 하나의 시퀀스로 받는다. 그러나 학습 목표는 두 계열 모두 해당 시점의 이진 레이블이고 산출물도 관측 시점별 고장 예측확률이므로, 이 확률을 언제 Alarm으로 볼 것인지와 그 Alarm으로 HDD를 어떻게 판정할 것인지는 어느 계열에서도 모델 바깥의 절차로 남는다. 본 연구는 모델 구조의 개선이 아니라 이 절차를 다룬다.

*→ **모델 구조와 입력 방식이 달라도 예측확률을 Alarm 및 HDD 단위 판정으로 전환하는 평가 절차는 공통적으로 필요하다.***

### 2.2 HDD 고장 예측의 평가와 운영 관점 연구

표 1은 SMART 기반 HDD 고장 예측 연구 가운데 평가 단위와 판정 기준을 본문에서 확인할 수 있는 11편을 정리한 것이다.

표 1. 선행 연구의 평가 프로토콜
Table 1. Evaluation protocols in prior work

| Study | Eval. unit | Scope | Verdict | Oper. metrics |
| --- | --- | --- | --- | --- |
| Disk replacement (2016)[2] | Row | Partial | Interval 10–15 d | — |
| RODMAN (2019)[12] | HDD | Full | Interval varies | FPR |
| SMARTer (2020)[13] | HDD | Full | Interval 10 d | — |
| LSTM specificity (2020)[3] | Window | n/s | Interval 15 d | FAR |
| LightGBM+CID (2021)[8] | HDD | Partial | Interval 10 d | Days in adv. |
| Model update (2021)[4] | Row | Full | Interval 10 d | FAR |
| StreamDFP (2023)[10] | HDD | Full | Interval 20 d | Days in adv. |
| Survival deepnet (2024)[5] | Row | Full | Interval 15 d | FAR |
| DFPoLD (2025)[14] | Row | Partial | Interval varies | DPF |
| Explainable TS (2025)[11] | HDD | Segments | Interval 5 d | FAR |
| TCN-LSTM-Attn (2026)[6] | Window | Balanced | Interval 7 d | FAR |
| This work | Row·HDD | Full | **First alarm 30 d** | FAR, Median LT |

평가 단위는 성능지표가 집계되는 단위, 평가 범위는 관측 이력 중 평가에 사용한 구간, 판정은 무엇을 탐지 성공으로 볼 것인지를 정하는 기준이다. 평가 범위의 Full은 선택된 기간의 관측 이력을 그대로 사용함을, Partial과 Balanced는 각각 고장 전 일부 구간과 양성·음성 비율을 맞춘 표본을 사용함을 뜻하며, n/s는 본문에 명시가 없는 경우이다. 평가 단위는 11편 중 6편이 Row 또는 Window, 5편이 HDD이고 평가 범위도 넷으로 갈리지만, 판정은 모두 고장 전 일정 구간의 레이블에 근거한다. 다만 그 구간의 성격은 서로 달라, RODMAN[12]은 데이터셋과 실행 조건에 따라 달라지는 역추적 구간으로 레이블을 만들고 테스트는 별도의 시간 분할 구간에서 수행하며, DFPoLD[14]는 결측률에 따라 구간 길이를 정하고, StreamDFP[10]는 30일 이내 고장을 예측하면서 고장 전 20일을 레이블링에 사용한다. 따라서 표 1의 구간 값들은 서로 같은 개념이 아니며, 본 연구의 30일도 그중 어느 것과 직접 대응하지 않는다.

→ **표 1의 연구들은 평가 단위와 평가 범위를 저마다 다르게 정해 명시하지만, 판정은 모두 고장 전 일정 구간의 레이블에 근거하며 그 구간의 정의가 서로 달라 값을 같은 개념으로 볼 수 없다.**

평가 범위도 성능 값을 바꾼다. 고장 직전 일부 구간만 사용하면 모델이 운영에서 마주하는 시간의 대부분을 차지하는 정상 구간이 평가에서 빠지고, 양성·음성 비율을 맞춘 표본을 사용하면 양성 비율이 실제 운영과 자릿수 단위로 달라진다. Precision과 FAR은 모두 음성 표본의 구성에 직접 의존하므로, 이렇게 산출된 값은 운영과 다른 모집단에서 나온 값이다. 표 1에서 평가 범위를 Full로 둔 연구는 11편 중 5편이다.

→ **평가 범위를 고장 직전 구간이나 균형 표본으로 한정하면 운영에서 모델이 마주하는 음성 구간의 대부분이 빠지므로, 그 위에서 산출된 Precision과 FAR은 운영과 다른 모집단의 값이다.**

DPF(Days Prior to Failure)와 Days in Advance는 고장 시점과 최초 예측 시점의 간격이라는 점에서 본 연구의 Lead Time과 형태가 같으나, 판정을 가르는 기준이 아니라 탐지에 성공한 HDD의 평균값으로 보고되며 값이 클수록 좋은 것으로 해석된다. 예컨대 [8]은 10일의 Prediction Horizon 아래에서 이 값이 7.75일에서 8.95일로 늘어난 것을 개선으로 제시한다. 본 연구에서는 같은 값이 Horizon을 넘으면 Early로 판정되므로 이 해석이 성립하지 않는다.

**→ *DPF와 Days in Advance는 Lead Time과 유사한 값을 사용하지만 탐지의 적시성을 판정하는 기준이 아니라 평균적인 탐지 선행시간으로 해석되므로 본 연구의 Lead Time과 기능이 다르다.***

표 1에서 HDD를 평가 단위로 삼은 연구는 다섯 편이나, 이들도 고장 시점을 기준으로 정한 구간의 이진 분류 결과를 HDD별로 집계하므로 본 연구와의 차이는 집계 단위가 아니라 판정을 만드는 방식에 있다. 이 가운데 StreamDFP[10]는 전체 관측 이력을 시간 순으로 흘려보내며 HDD 단위로 일별 집계한다는 점에서 본 연구와 가장 가까우나, 판정 자체는 고장 전 20일 구간의 이진 분류로 이루어진다. 이러한 구성에서는 탐지 성공 여부가 그 구간 안에서 정해지므로 구간 밖의 예측은 적중한 탐지로 인정되지 않고, 구간 안에서 고장 직전에 발생한 예측과 시작점에서 발생한 예측이 모두 탐지 성공으로 집계되어 대응 시간의 차이가 Precision이나 Recall에 반영되지 않으며, 관측 종료 이후의 잠재적 고장이 별도 범주로 보존되지 않아 해당 HDD를 음성으로 처리하거나 불확실한 구간을 평가에서 제외하게 된다. 시간창의 길이나 오탐 제약 수준은 데이터 이전에 필요한 시간이나 점검 부담과 같은 운영상의 이유로 정해지지만, 실제로 확보된 대응 시간과 점검 대상이 되는 HDD 대수는 측정 대상에 포함되지 않는다.

**→ *HDD를 평가 단위로 사용하는 것만으로는 운영 성능을 평가할 수 없으며, 기존 HDD 단위 연구도 고장 전 특정 구간의 분류 결과를 집계하여 최초 Alarm의 시점·실제 대응시간·관측 종료 이후의 불확실성을 평가에 반영하지 못한다.***

개체를 연속적으로 감시하는 다른 분야에서도 개별 예측의 정확성과 개체 단위 성능은 갈린다. 패혈증 예측에서 동일한 모델의 AUC는 환자 단위 0.86, 시점 단위 0.62였다[15]. 환자 모니터링의 Early Warning 시스템에서는 Alarm을 발생 시점에 따라 구분하여 Alarm 단위와 환자 기록 단위로 집계하고, 한 이벤트에 여러 Alarm이 발생할 때의 Alarm 부담까지 평가에 포함하는 프레임워크가 제안되었다[16][17]. 저자가 확인한 범위에서, HDD의 전체 관측 이력에 시간 순 온라인 추론을 적용하고 최초 Alarm의 시점과 우측 중도절단을 함께 판정에 포함한 연구는 확인되지 않았다.

**→ *다른 연속 감시 분야에서는 개체 단위 판정과 Alarm 시점을 평가에 포함하는 관점이 활용되고 있으나, HDD 고장 예측에서는 전체 관측 이력·최초 Alarm·우측 중도절단을 함께 반영한 평가가 확인되지 않는다.***

---

## 3. 평가 방법

### 3.1 행 단위 평가

행 단위 평가는 각 HDD의 개별 관측 행을 독립적인 분류 대상으로 취급하는 방식으로, 표 1에서 복수의 사례로 나타난다. 본 연구는 HDD 전체 관측 이력에 이를 적용한 결과를 비교 기준선으로 삼되, 표 1의 절차가 서로 다르므로 추가 설정이 필요 없는 가장 단순한 구성을 택하였다. Sliding Window나 부분 구간을 쓰는 절차를 기준선으로 삼으면 윈도우 길이나 구간 선택이 평가 절차와 함께 달라져 관찰된 차이를 어느 쪽에 귀속할지 알 수 없기 때문이다. 다만 우측 중도절단 HDD의 마지막 예측 기간을 제외한 것은 두 평가의 대상 행 집합을 일치시키기 위한 수정이므로, 이 기준선은 선행 연구의 절차와 완전히 같지는 않다.

→ **행 단위 평가는 개별 관측 행을 독립적인 분류 대상으로 평가하며, 본 연구에서는 이를 전체 관측 이력에 적용한 가장 단순한 구성을 비교 기준선으로 삼아 평가 범위를 두 평가에서 일치시킨다.**

하나의 행은 특정 HDD의 특정 날짜 상태에 대응하므로, 이하에서 관측 시점 $t$와 Prediction Horizon $H$는 모두 일 단위이다. 고장 시점이 관측된 HDD에서 고장까지 남은 기간이 $H$ 이내인 행($t_{\mathrm{failure},d}-t \le H$)을 양성으로, 그 외의 행을 음성으로 정의한다. 마지막 기록일까지 고장이 확인되지 않은 우측 중도절단 HDD는 그 직전 $H$일 구간의 레이블을 확정할 수 없으므로 평가에서 제외한다(3.2절의 평가 구간과 동일). 평가 대상은 테스트 분할 HDD의 전체 관측 이력 중 이 기준으로 제외되지 않은 모든 행이다.

**→ 행 단위 평가에서는 고장까지의 잔여 기간이 Prediction Horizon 이내인 행을 양성으로, 우측 중도절단 HDD의 확정할 수 없는 마지막 구간을 제외한 나머지 행을 음성으로 정의한다.**

HDD $d$의 관측 시점 $t$에 대한 예측확률 $p_{d,t}$가 의사결정 임곗값 $\tau$ 이상이면 해당 행을 양성으로 예측한 것으로 보고, 각 행의 예측 결과와 레이블을 대조하여 True Positive, False Positive, False Negative 및 True Negative를 집계한다. Precision은 양성으로 예측한 행 중 실제 양성인 행의 비율, Recall은 실제 양성인 행 중 양성으로 예측한 행의 비율, FAR(False Alarm Rate)은 실제 음성인 행 중 양성으로 예측한 행의 비율이다. 이때 각 행은 독립적인 분류 결과로 취급되므로 한 HDD에서 처음 발생한 양성 예측도 이후의 양성 예측과 구분되지 않고 합산되며, 그것이 고장보다 얼마나 앞섰는지를 나타내는 양은 이 절차에서 산출되지 않는다.

**→ 각 행의 예측 결과를 독립적으로 집계하므로 동일 HDD에서 반복되는 양성 예측과 최초 Alarm의 시점 및 Lead Time을 구분할 수 없다.**

### 3.2 운영 환경 기반 평가

운영 환경 기반 평가는 각 HDD의 관측 이력을 시간 순으로 처리하며, 각 시점의 예측확률을 임곗값에 따라 Alarm으로 변환한 뒤 그 결과를 HDD 단위로 집계한다. 여기서 온라인 추론은 학습된 모델을 시간 순으로 적용하는 것을 뜻하며 모델 갱신을 포함하지 않는다. 고장 없이 기록이 끊긴 HDD의 마지막 기록일이 그 HDD의 관측 종료 시점이고, 그 이후의 고장 여부와 고장 시점은 확인할 수 없다. 이러한 HDD를 우측 중도절단(right-censoring) HDD라 하며, 이는 해당 HDD가 정상임을 뜻하지 않는다. 따라서 평가 대상은 고장 시점이 확인된 HDD와 우측 중도절단 HDD로 나뉜다.

→ **운영 환경 기반 평가는 HDD별 전체 관측 이력을 시간 순으로 처리하고, 고장 관측 여부에 따라 평가 대상을 구분하여 HDD 단위로 예측 결과를 평가한다.**

각 모델은 시간 순서에 따라 관측 시점별 고장 예측확률 $p_{d,t}$를 산출하며, $p_{d,t}\ge\tau$인 시점에 Alarm이 발생한다. 한 HDD에서 Alarm은 여러 번 발생할 수 있으므로 평가 대상 구간의 최초 Alarm 시점 $t_{\mathrm{alarm},d}$만을 판정에 사용한다. 고장 관측 HDD의 Lead Time은 $\mathrm{LT}_d=t_{\mathrm{failure},d}-t_{\mathrm{alarm},d}$이며, 관측이 고장 시점까지 이루어지므로 음수가 될 수 없다. Prediction Horizon을 $H=30$일로 두고, $\mathrm{LT}_d\le H$이면 On-time, $\mathrm{LT}_d>H$이면 Early, 최초 Alarm이 없으면 Missed로 분류한다.

→ **대응이 시작될 수 있는 가장 이른 시점인 최초 Alarm만을 판정에 사용하며, 고장 관측 HDD는 Lead Time과 Prediction Horizon의 관계로 On-time·Early·Missed로 판정한다.**

우측 중도절단 HDD는 관측 종료 시점 $t_{\mathrm{end},d}$ 이후의 고장 여부를 확인할 수 없어, 관측 종료 직전 $H$일에 발생한 Alarm은 On-time 여부를 판단할 수 없다. 따라서 이 마지막 $H$일 구간을 평가에서 제외하고 그 이전의 최초 Alarm만을 평가 대상으로 둔다. 평가 대상 구간에서 최초 Alarm이 발생한 HDD는 고장 시점을 알 수 없어 Lead Time을 산출할 수 없으나, Alarm이 관측 종료보다 최소 $H$일 앞서므로 이후 고장이 발생하더라도 그 Lead Time은 $H$를 초과한다. 즉 On-time으로 분류될 수 없음이 보장되며, 이를 Censored Early로, 평가 대상 구간에서 최초 Alarm이 발생하지 않은 우측 중도절단 HDD를 Censored No Alarm으로 분류한다. 이로써 각 HDD는 다섯 판정 결과 중 하나로 판정된다(그림 1).

**→ 우측 중도절단 HDD는 관측 종료 직전 \(H\)일을 평가에서 제외함으로써 관측되지 않은 미래의 고장 여부와 관계없이 해당 Alarm이 On-time이 될 수 없음을 보장하며, 이에 따라 Censored Early와 Censored No Alarm으로 나뉘어 각 HDD는 다섯 판정 결과 중 하나로 분류된다.**

!그림 1. 고장 시점 관측 여부에 따른 HDD별 평가 구간 및 판정 체계
Fig. 1. Evaluation intervals and outcome categories by HDD observation status

그림 1. 고장 시점 관측 여부에 따른 HDD별 평가 구간 및 판정 체계
Fig. 1. Evaluation intervals and outcome categories by HDD observation status

본 판정 체계는 환자 모니터링의 Early Warning 시스템을 평가하기 위해 Alarm을 발생 시점에 따라 False, Early, On Time, Late, Missed로 구분한 프레임워크[16][17]를 HDD 고장 예측에 맞게 변형한 것이다. 원 프레임워크는 이벤트 발생 여부가 확정된 대상을 전제하여 이벤트가 관측되지 않은 대상의 Alarm을 단일 False로 처리하나, 관측 종료 이후를 확인할 수 없는 HDD에는 적용할 수 없으므로 본 연구는 이를 Censored Early와 Censored No Alarm으로 나누었다. 관측이 고장 시점까지만 이루어져 Alarm이 고장보다 늦게 발생할 수 없으므로 Late에 해당하는 판정은 두지 않았다. 다만 Lead Time이 0에 가까운 Alarm도 On-time으로 집계되며, 이를 분리하는 최소 대응시간의 하한과 원 프레임워크가 함께 다루는 반복 Alarm의 부담은 모두 운영 정책에서 정해지므로 판정에 포함하지 않았다.

**→ 본 판정 체계는 기존 Early Warning 평가 프레임워크의 Alarm 시점 기반 판정을 HDD의 우측 중도절단 특성에 맞게 변형한 것이며, 판정이 운영 정책에 의존하지 않도록 하한 없이 상한 조건만을 사용하였다.**

이상의 판정 결과로부터 HDD-level Precision, Recall, FAR 및 Median Lead Time을 산출한다. 다섯 판정 결과를 $\mathrm{OT}$(On-time), $\mathrm{E}$(Early), $\mathrm{M}$(Missed), $\mathrm{CE}$(Censored Early), $\mathrm{CN}$(Censored No Alarm)으로 줄여 쓰고, 이들 각각을 $X$로 나타내어 해당 HDD의 집합을 $D_X$, 그 수를 $N_X$로 표기하면 네 지표는 식 (1)~(4)와 같다.

$$
\begin{aligned}
\mathrm{Precision}_{\mathrm{HDD}} &= \frac{N_{\mathrm{OT}}}{N_{\mathrm{OT}}+N_{\mathrm{E}}+N_{\mathrm{CE}}} &&(1)\\[4pt]
\mathrm{Recall}_{\mathrm{HDD}} &= \frac{N_{\mathrm{OT}}}{N_{\mathrm{OT}}+N_{\mathrm{E}}+N_{\mathrm{M}}} &&(2)\\[4pt]
\mathrm{FAR}_{\mathrm{HDD}} &= \frac{N_{\mathrm{CE}}}{N_{\mathrm{CE}}+N_{\mathrm{CN}}} &&(3)\\[4pt]
\mathrm{Median\ LT} &= \operatorname{median}\{\mathrm{LT}_d \mid d\in D_{\mathrm{OT}}\cup D_{\mathrm{E}}\} &&(4)
\end{aligned}
$$

네 지표는 모집단이 서로 다르며, 이들을 합치면 다섯 판정 결과가 모두 포함되어 각 HDD는 적어도 하나의 지표에 집계된다. 식 (1)의 Precision은 Early와 Censored Early를 유효한 사전 Alarm으로 인정하지 않으므로, Alarm의 발생 빈도가 아니라 발생한 Alarm이 대응 가능한 시점에 있었는지를 평가한다. 식 (2)의 Recall은 고장 시점이 관측된 HDD 전체를 분모로 두어 Prediction Horizon 이내의 탐지만을 센다. 식 (4)의 Median Lead Time은 On-time과 Early를 함께 포함하며, 지나치게 긴 값은 Early의 증가를 뜻하므로 클수록 좋은 성능 지표가 아니라 Alarm이 고장으로부터 어느 정도의 여유를 두고 발생하는지를 나타내는 운영 특성 지표이다. 네 지표는 모두 다섯 판정 결과를 거쳐 산출되고 그 판정은 각 HDD의 최초 Alarm 시점으로 정해지므로, 각 행을 독립으로 집계하는 3.1의 절차로부터는 어느 것도 얻을 수 없다.

**→ HDD-level Precision·Recall·FAR과 Median Lead Time은 모두 최초 Alarm 시점으로 정해지는 다섯 판정 결과를 거쳐 산출되므로, 각 행을 독립으로 집계하는 절차에서는 어느 것도 얻을 수 없다.**

HDD-level FAR은 행 단위 FAR과 같은 구조를 유지하기 위해 고장이 관측되지 않은 HDD만을 분모로 둔다. 두 지표가 같은 꼴이어야 동일한 수치 제약을 두 평가에 함께 걸 수 있기 때문이다. 다만 이 분모는 고장하지 않는 HDD가 아니라 관측 범위 안에서 고장이 확인되지 않은 HDD이다. 고장이 관측된 HDD에서 지나치게 이른 Alarm이 만드는 점검 부담은 식 (1)의 분모에 Early$(N_{\mathrm E})$가 포함되므로 Precision이 대신 받는다. 이 지표는 평가 구간 전체에서 최초 Alarm의 발생 여부를 집계하므로 다른 조건이 같다면 관측 기간이 길수록 값이 커지며, 따라서 관측 기간이 다른 데이터셋 사이의 우열 판단에는 사용하지 않는다.

→ **HDD-level FAR은 고장이 확인되지 않은 HDD에서 최초 Alarm이 발생한 비율로 정의하여 행 단위 FAR과 동일한 형태의 오탐 제약을 적용하되, 최초 Alarm은 취소되지 않으므로 같은 HDD 집합에서 평가 구간이 길어지면 이 값은 줄어들 수 없고 따라서 관측 기간이 다른 데이터셋 사이의 비교에는 사용하지 않는다.**

두 평가는 무엇을 세는지와 모델에 무엇을 요구하는지가 모두 다르다. 점검과 교체는 HDD 단위로 이루어지므로 운영에서 부담이 되는 것은 오분류된 행의 수가 아니라 그 오분류가 발생한 HDD의 대수이나, 각 행을 독립으로 세는 절차에서는 한 HDD에서 오래 반복된 오분류와 여러 HDD에 하루씩 분산된 오분류가 같은 값으로 집계된다. 요구하는 것도 같지 않다. 행 단위 분류 성능은 운영 환경 기반 평가에서도 요구되나 그것은 Missed를 줄일 뿐 On-time을 보장하지 않으며, Early는 다른 요인에서 발생한다. 3.1의 정의에 따라 고장보다 $H$일 넘게 앞선 행은 음성이므로 그 구간에서 발생한 Alarm은 행 단위 평가에서 하나의 False Positive로 집계될 뿐이고, 음성 행이 전체의 대부분을 차지하므로 행 단위 FAR에 거의 반영되지 않는다. 그러나 운영 환경 기반 평가는 최초 Alarm만을 판정에 사용하므로 해당 HDD는 그 시점에 Early로 확정된다. 따라서 이 평가는 고장을 탐지하는 능력과 함께, 고장과 무관한 신호에 반응하지 않고 대응 가능한 시점에 최초 Alarm이 발생하도록 하는 적시성을 요구한다.

→ **운영 환경 기반 평가는 점검과 교체가 일어나는 HDD를 집계 단위로 두며, 행 단위 분류 성능만으로는 충분하지 않고 대응 가능한 시점에 최초 Alarm이 발생하는 적시성을 함께 요구한다.** 

---

## 4. 실험 설계

### 4.1 데이터 및 실험 설정

본 연구는 Backblaze가 공개하는 데이터센터 HDD의 SMART 기록을 사용하였다[18]. 고장은 해당 HDD가 교체되어 기록이 끝나는 날에 표시되며, 본 연구는 그 날짜를 고장 시점 $t_{\mathrm{failure},d}$로, 고장 표시 없이 기록이 끝나는 HDD의 마지막 날짜를 관측 종료 시점 $t_{\mathrm{end},d}$로 둔다. 제조사가 다른 HDD 기종 3종(HGST HUH721212ALN604, Seagate ST12000NM0007, Toshiba MG07ACA14TA)을 선정하였으며, 세 기종은 HDD 수·고장률·관측 기간이 서로 달라 데이터 특성이 다른 조건에서 두 평가의 차이가 각각 재현되는지 확인할 수 있다(표 2). 이하에서는 각 데이터셋을 제조사명으로 지칭한다.

**→ 본 연구는 서로 다른 특성을 가진 세 제조사 HDD 데이터셋을 사용하여 두 평가 방식의 차이가 데이터 조건에 따라 재현되는지를 확인한다.**

표 2. 데이터셋 특성
Table 2. Dataset characteristics

|  | HGST | Seagate | Toshiba |
| --- | --- | --- | --- |
| HDDs | 11,370 | 38,790 | 39,351 |
| Failed HDDs (%) | 1,608 (14.1%) | 2,236 (5.8%) | 2,076 (5.3%) |
| Rows | 26,863,127 | 37,331,086 | 75,086,961 |
| Positive label rate | 0.176% | 0.181% | 0.083% |
| Obs. days, Med. (IQR) | 2,512 (2,423–2,587) | 962 (772–1,141) | 1,933 (1,772–2,065) |

세 데이터셋에 동일한 전처리를 적용하였다. Backblaze는 각 SMART 속성을 raw 값과 제조사 정규화 값으로 제공하나 정규화 척도가 제조사마다 달라 raw 값만 사용하였고, 메타데이터 컬럼과 결측치 비율 90% 초과 컬럼 및 중복 행을 제거하였다. 관측일 사이의 공백은 3일 이하이면 직전 값으로 보간하고, 4일 이상이면 불연속 구간이 하나의 입력 윈도우에 섞이지 않도록 별도의 시계열 세그먼트로 분리하였다(HGST 813대, 7.1%). 이후 결측치가 남은 행과 고장 기록 후 다시 정상으로 기록된 HDD를 제거하였으며, 전처리로 제거된 HDD는 데이터셋별 0.1~0.6%, 사용된 특성은 18~26개이다. 입력 윈도우는 각 세그먼트 안에서 구성하되 평가에서는 세그먼트가 분리된 HDD도 하나의 HDD로 취급하고 Lead Time은 달력일 기준으로 산출한다. 입력 구간을 온전히 확보할 수 없는 관측 초반부는 첫 행을 반복하여 채웠으므로, 네 모델 모두 동일한 행에서 예측확률이 산출된다.

**→ 세 데이터셋에는 동일한 전처리를 적용하여 제조사별 SMART 척도 차이와 시계열 불연속성을 통제하고, 모든 모델이 동일한 관측 행에서 예측확률을 산출하도록 구성한다.**

고장 예측은 행 단위 이진 분류로 정의하고, 고장까지 남은 기간이 $H=30$일 이내인 행을 양성으로 두었으며, 모든 비교 조건에 동일하게 적용하였다. Early는 $\mathrm{LT}_d>H$인 HDD이므로 $H$가 클수록 Early로 분류되는 HDD는 줄어들며, 30일은 이 판정에서 On-time 구간을 넓게 잡은 값이다. 우측 중도절단 HDD는 관측 종료 이후를 확인할 수 없으므로 $t_{\mathrm{end},d}-t<H$인 행을 학습·검증·테스트에서 모두 제외하였다(3.2절).

→ **고장 예측은 행 단위 이진 분류로 정의하고, On-time 판정을 넓게 잡아 Early가 과다 계상되지 않도록 Prediction Horizon을 30일로 두었으며, 우측 중도절단 HDD의 평가 불가능한 구간을 모든 실험에서 동일하게 제외한다.**

동일 HDD가 서로 다른 분할에 걸치지 않도록, 고장 관측 여부로 층화한 HDD 단위 그룹 분할을 8:1:1로 적용하였다. 특정 시점 이후를 예측하는 설정이 아니라 처음 보는 HDD에 대한 일반화 성능을 평가하기 때문이며, 각 테스트 HDD는 관측 이력 전체를 시간 순으로 하루씩 진행하며 추론하므로 HDD 내부의 시간 순서는 보존된다. 분할은 고정된 난수로 한 번만 수행하여 평가 대상 HDD 집합을 모든 실험에서 동일하게 두었고, 30개 random seed는 학습의 무작위성만 바꾼다. 학습과 테스트가 같은 달력 구간을 공유하는 점은 두 평가에 동일하게 작용하므로 절차 간 비교에는 영향을 주지 않는다.

**→ HDD 단위로 데이터를 8:1:1 분할하여 처음 보는 HDD에 대한 일반화 성능을 평가하면서, 테스트 HDD 내부의 시간 순서와 모든 실험의 평가 대상 집합은 동일하게 유지한다.**

2.1에서 정리한 두 계열을 대표하도록 트리 기반(LightGBM, XGBoost)과 순환신경망(LSTM, GRU) 모델을 선정하였다. 트리 기반 모델은 학습률 0.05와 표본·피처 추출 비율 0.9로 학습하였고, 순환신경망 모델은 은닉 64차원 2층에 14일 입력 윈도우를 사용하여 Adam으로 학습하였다. 학습 손실은 네 모델 모두 이진 교차 엔트로피이며 검증 성능을 기준으로 Early Stopping을 적용하였고, 순환신경망 모델에 한해 입력 피처를 학습 데이터에서만 적합한 StandardScaler로 표준화하였다. 네 모델은 입력 구성과 학습 절차가 서로 다르지만 모두 관측 시점별 예측확률을 산출하므로(2.1절), 이후의 평가 절차 비교는 특정 모델 구조에 의존하지 않는다.

→ **2.1의 두 계열을 대표하도록 네 모델을 선정하고 동일한 학습·검증 체계를 적용하여, 특정 모델 구조에 의존하지 않는 조건에서 평가 절차를 비교한다.**

### 4.2 실험 구성 및 임곗값 선정

본 연구의 실험 절차는 그림 2와 같다. 전처리와 분할, 학습, 추론까지는 두 평가가 하나의 경로를 공유하며, 평가 방식과 임곗값 기준이라는 두 축에서만 갈라진다. 각 평가는 자신의 오탐 지표로 의사결정 임곗값을 검증 데이터에서 선정하며, 행 단위 FAR을 기준으로 정한 값을 $\tau_{\mathrm{row}}$, HDD-level FAR을 기준으로 정한 값을 $\tau_{\mathrm{op}}$로 표기한다.

**→ 두 평가 방식은 동일한 전처리·분할·학습·추론 경로를 공유하고, 평가 절차와 임곗값 선정 기준이라는 두 축에서만 구분된다.**

!그림 2. 실험 절차와 비교 구성
Fig. 2. Experimental procedure and comparison design

그림 2. 실험 절차와 비교 구성
Fig. 2. Experimental procedure and comparison design

그림 2에서 조건 사이의 점선 양방향 화살표는 본 연구가 수행하는 세 가지 대조로, 각각 평가 절차만, 임곗값 기준만, 그리고 둘 다 다른 경우에 해당한다. 회색으로 표시한 우측 하단 칸은 행 단위 평가에 $\tau_{\mathrm{op}}$를 적용하는 조합으로, 대응하는 사용 방식이 없어 분석에서 제외하였다.

→ **그림 2의 세 조건은 평가 절차만, 임곗값 기준만, 둘 다 다른 대조에 각각 해당하며, 대응하는 사용 방식이 없는 한 조합은 분석에서 제외한다.**

평가 대상 행 집합과 예측확률은 두 평가에서 동일하다. 운영 환경 기반 평가는 각 HDD의 전체 관측 이력을 시간 순으로 처리하므로, 행 단위 평가를 다른 행 집합에서 산출하면 관찰된 차이를 평가 절차와 평가 대상 중 어느 쪽에 귀속할지 알 수 없다. 학습 목적함수도 고정하여 최초 Alarm 판정이나 HDD 단위 지표를 학습 목표로 삼지 않았다. 따라서 비교 대상은 학습 방식이 아니라 동일한 예측 결과에 적용한 평가 절차이다. 다만 양성 레이블 비율이 0.083~0.181%로 낮아 본 연구의 행 단위 Precision은 고장 전 일부 구간이나 균형 표본으로 평가한 선행 연구의 보고값보다 낮으므로, 절대 수준은 선행 연구와 비교하지 않는다.

→ **학습 목적함수까지 고정하였으므로 비교 대상은 학습 방식이 아니라 평가 절차이며, 이 구성에서 행 단위 Precision의 절대 수준은 선행 연구와 직접 비교하지 않는다.**

대규모 스토리지 시스템에서는 개별 HDD의 불필요한 Alarm도 시스템 규모에서는 상당한 운영 부담으로 누적되므로, 두 평가 모두 오탐 수준을 제한하는 제약을 부여하였다. 오탐 수준을 동일한 수치적 기준에서 비교하기 위해 행 단위 FAR과 HDD-level FAR에 모두 1%를 적용하였으나, 두 지표는 적용 모집단과 산출 방식이 다르므로 동일한 1%가 동일한 운영적 엄격도를 뜻하지는 않는다. 임곗값은 0.001 간격으로 탐색하여 각 오탐 지표가 1% 이하가 되는 최소값으로 정하였으며, 두 값은 독립적으로 선정한 뒤 동일한 테스트 데이터에 적용하였다. 제약을 만족하는 임곗값이 탐색 범위 내에 없는 경우에는 제약 위반이 가장 작은 임곗값을 사용하고 해당 seed도 집계에 포함하였다(5.1절).

**→ 대규모 스토리지 환경의 오탐 부담을 고려하여 두 평가 방식에 동일한 1% 오탐 제약을 적용하되, 각 평가의 오탐 지표를 기준으로 임곗값을 독립적으로 선정한다.**

세 조건을 30개 random seed(42–71)에 대해 반복하고, 이 대조를 지표 수준(5.1절)과 모델 순위 수준(5.2절)에서 수행한다. 이하에서 하나의 실험은 데이터셋·모델·seed를 각각 하나씩 정한 것을, 조합은 데이터셋과 모델의 짝 12개를 가리킨다. 모든 결과는 테스트 분할에서 산출하였으며, 조합별 30개 seed 결과는 중앙값으로 요약하고 두 조건의 비교는 항상 동일한 실험 내에서 짝지어 수행한다. 30개 seed는 동일한 테스트 HDD 집합을 공유하는 재무작위화이므로 통계적 검정 대신 360개(30 seed × 12 조합) 비교의 방향 일관성을 제시한다. 다만 그림 3과 판정 대수를 제시하는 서술은 seed 42를 사례로 사용하므로 30개 seed 중앙값과 다르다.

**→ 행 단위 평가와 운영 환경 기반 평가에 각각 고유한 임곗값을 적용한 세 조건을 30개 random seed에 반복하여, 평가 절차와 임곗값 기준의 차이를 지표 및 모델 순위 수준에서 비교한다.**

---

## 5. 실험 결과

### 5.1 평가 조건에 따른 성능 변화

본 절은 동일한 예측 결과에 두 평가를 적용하고 이어서 각 평가의 기준으로 임곗값을 선정하여, 행 단위 지표가 운영에서 필요한 것을 나타내지 못하며 그 값에서 운영 성능이 정해지지 않음을 지표 수준에서 보인다. 결과는 표 3과 같다. Row-level과 Operational은 각각 두 평가를 적용한 조건이고 괄호 안은 사용한 임곗값이며, Operational 열의 Prec/Rec/FAR은 식 (1)~(3)의 HDD 단위 지표이므로 이하 HDD-level Precision/Recall/FAR로 부른다. Median Lead Time은 식 (4)로 산출하며 행 단위 평가에서는 정의되지 않는다. 두 임곗값은 검증 데이터에서 선정하였으므로 테스트 FAR은 제약선 1%를 상회할 수 있고, $\tau_{\mathrm{op}}$에서는 12개 조합 중 8개가 1.04~1.74%였다. 탐색 상한에서도 검증 제약을 만족하지 못한 것은 예측확률이 1에 가깝게 포화한 HGST의 LightGBM 한 조합의 7개 seed뿐이다.

→ **본 절은 동일한 예측 결과에 서로 다른 평가 절차와 임곗값 기준을 적용하여, 행 단위 지표가 무엇을 나타내지 못하며 그 값에서 운영 성능이 정해지지 않음을 지표 수준에서 보인다.**

표 3. 평가 방식·임곗값 기준별 성능
Table 3. Performance by evaluation method and threshold criterion

| Dataset | Model | Row-level$(\tau_{\mathrm{row}})$ | Operational$(\tau_{\mathrm{row}})$ | Operational$(\tau_{\mathrm{op}})$ |
| --- | --- | --- | --- | --- |
|  |  | Prec / Rec / FAR | Prec / Rec / FAR / Med. LT | Prec / Rec / FAR / Med. LT |
| HGST | LightGBM | 0.075 / 0.440 / 0.96% | 0.089 / 0.105 / 7.47% / 221 | 0.302 / 0.086 / 1.13% / 56 |
|  | XGBoost | 0.081 / 0.469 / 0.95% | 0.071 / 0.086 / 7.93% / 229 | 0.263 / 0.068 / 1.13% / 64 |
|  | LSTM | 0.070 / 0.453 / 1.09% | 0.066 / 0.068 / 6.14% / 223 | 0.266 / 0.117 / 1.54% / 80 |
|  | GRU | 0.071 / 0.459 / 1.07% | 0.057 / 0.062 / 6.96% / 223 | 0.283 / 0.136 / 1.74% / 72 |
| Seagate | LightGBM | 0.067 / 0.370 / 0.97% | 0.252 / 0.302 / 3.53% / 32 | 0.439 / 0.236 / 1.14% / 11 |
|  | XGBoost | 0.065 / 0.391 / 1.05% | 0.209 / 0.300 / 4.76% / 38 | 0.419 / 0.249 / 1.15% / 14 |
|  | LSTM | 0.065 / 0.379 / 1.02% | 0.167 / 0.273 / 6.02% / 47 | 0.354 / 0.182 / 1.08% / 24 |
|  | GRU | 0.070 / 0.386 / 0.97% | 0.171 / 0.284 / 6.18% / 45 | 0.412 / 0.224 / 1.04% / 21 |
| Toshiba | LightGBM | 0.084 / 0.392 / 0.36% | 0.193 / 0.196 / 2.33% / 68 | 0.306 / 0.148 / 0.55% / 45 |
|  | XGBoost | 0.044 / 0.462 / 0.88% | 0.160 / 0.220 / 4.04% / 76 | 0.282 / 0.163 / 0.89% / 46 |
|  | LSTM | 0.039 / 0.478 / 1.03% | 0.082 / 0.177 / 8.25% / 112 | 0.275 / 0.134 / 0.79% / 52 |
|  | GRU | 0.038 / 0.478 / 1.03% | 0.077 / 0.177 / 8.77% / 113 | 0.287 / 0.148 / 0.86% / 48 |

임곗값을 $\tau_{\mathrm{row}}$로 고정한 채 평가 절차만 바꾸면, 행 단위에서 제한한 오탐 수준이 운영 수준의 점검 부담을 반영하지 못한다는 것이 드러난다. 행 단위 FAR은 테스트에서도 0.36~1.09%로 제약선 1%를 크게 벗어나지 않았으나, 이 값은 그 오분류가 몇 대의 HDD에서 발생하였는지를 구분하지 않는다. 같은 임곗값을 유지한 채 각 HDD의 이력을 시간 순으로 처리하는 것은 행 단위 기준으로 임곗값을 정한 모델을 HDD 단위 운영에 투입한 상황에 해당한다. 이때 한 HDD는 962~2,512일에 이르는 관측 이력(표 2) 중 한 시점만 임곗값을 넘어도 점검 대상이 되므로, 고장이 확인되지 않은 HDD의 2.33~8.77%(중앙값 6.16%)에서 Alarm이 발생하였다. 이 값은 예외 없이 행 단위 FAR을 웃돌았으나 그 수준은 12개 조합에 동일한 1% 제약을 걸었음에도 조합마다 달라, HDD 단위 점검 부담은 행 단위 오탐 수준에서 정해지지 않았다. 점검 대상 대수로 환산하면 HGST의 테스트 중도절단 HDD 977대 중 약 60~78대에 해당한다.

→ **동일한 임곗값에서 평가 절차만 변경하면 행 단위 FAR의 제한 수준이 HDD 단위의 점검 부담을 통제하지 못한다.**

Row-level($\tau_{\mathrm{row}}$) 열과 Operational($\tau_{\mathrm{row}}$) 열을 대조하면 Recall과 Precision에서도 값이 달라지며, 그 차이는 일정한 관계를 이루지 않는다. HDD-level Recall은 행 단위 Recall보다 낮았고(HGST의 LightGBM 0.440 → 0.105), Precision은 변화 방향이 일정하지 않아 HGST에서는 네 모델 중 셋에서 오히려 낮아졌다. FAR 상승과 Recall 하락은 360개 짝지은 비교 전부에서 예외가 없었다. 조합 사이에서는 값의 순서도 보존되지 않아, 행 단위 Recall이 12개 조합 중 가장 낮은 Seagate의 LightGBM(0.370)이 HDD-level Recall은 가장 높다(0.302).

→ **Recall과 Precision의 값도 달라지며, 조합 사이의 순서도 보존되지 않아 행 단위 Recall이 가장 낮은 조합이 HDD 단위에서는 가장 높다.**

!그림 3. Lead Time의 누적 분포
Fig. 3. Cumulative distribution of Lead Time

그림 3. Lead Time의 누적 분포
Fig. 3. Cumulative distribution of Lead Time

행 단위 평가가 탐지로 집계하는 것과 운영에서 유효한 탐지는 일치하지 않는다. 그림 3은 $\tau_{\mathrm{row}}$에서 최초 Alarm이 고장보다 얼마나 앞서 발생하였는지를, 최초 Alarm이 발생한 고장 관측 HDD를 분모로 하여 누적 비율로 나타낸 것이다(XGBoost, seed 42). 세 데이터셋 모두 Prediction Horizon 시점의 누적 비율이 0.5에 미치지 못하며, 30개 seed 중앙값으로도 12개 조합 전부에서 최초 Alarm의 Lead Time 중앙값이 Horizon 밖에 있었다(32~229일). 이러한 HDD는 행 단위 평가에서 고장 직전 구간의 행을 정확히 분류한 만큼 탐지로 집계되지만, 운영에서는 그보다 앞선 시점에 이미 점검 대상이 되어 있다. 이 분모에 Alarm이 발생하지 않은 고장 HDD까지 포함한 값이 식 (2)의 HDD-level Recall이다.

→ **행 단위에서 탐지로 집계된 최초 Alarm 중 상당수는 고장보다 지나치게 이른 시점에 발생하여 운영 환경에서는 유효한 탐지로 볼 수 없다.**

임곗값까지 운영 기준으로 선정하면, 즉 $\tau_{\mathrm{row}}$ 대신 $\tau_{\mathrm{op}}$를 적용하면 HDD-level FAR의 12개 조합 중앙값은 1.10%로 제약 수준에 들었고, Precision은 0.295로 상승하였으며 Median Lead Time은 94일에서 47일로 감소하여 Prediction Horizon에 근접하였다. 세 방향 모두 360개 비교 전부에서 예외가 없었다. 이 가운데 FAR의 감소만이 분모가 중도절단 HDD 수로 고정된 채 분자가 줄어들어 정의상 보장된다. 반면 HDD-level Recall의 중앙값은 0.187에서 0.148로 낮아졌으나 그 변화는 단조적이지 않아, HGST의 LSTM(0.068→0.117)과 GRU(0.062→0.136)는 30개 seed 전부에서 상승하였다. HDD-level Recall의 분자는 최초 Alarm의 위치로 정해지므로(식 (2)), 임곗값이 오르면 최초 Alarm이 지연되어 Early였던 HDD가 Horizon 안에 들어올 수 있다. 임곗값이 오르면 양성 예측이 줄어들 뿐인 행 단위 Recall에서는 이러한 상승이 발생할 수 없으며, 두 Recall이 같은 이름 아래 서로 다른 사건을 센다는 근거가 여기에 있다.

→ **임곗값까지 운영 기준으로 선정하면 오탐이 제약 수준으로 돌아오고 Precision과 최초 Alarm의 시점도 개선되나, HDD-level Recall은 단조적이지 않아 Early가 On-time으로 전환되는 조합에서는 오히려 상승한다.**

같은 크기의 Recall 손실이라도 그것이 Early인지 Missed인지에 따라 필요한 조치는 반대가 된다. Early는 고장에서 먼 구간의 Alarm이므로 임곗값을 올리면 감소하지만 Missed는 증가하기 때문이다. $\tau_{\mathrm{row}}$에서 손실의 47~70%가 Early였고 $\tau_{\mathrm{op}}$에서는 8~31%로 낮아졌으나(12개 조합, seed 42), 감소한 Early가 모두 On-time으로 전환되지는 않았다. HGST의 테스트 고장 관측 HDD 162대 가운데 GRU는 Early가 103대에서 40대로 줄면서 On-time이 10대에서 23대로 늘었으나, LightGBM은 Early가 95대에서 27대로 줄어드는 동안 대부분이 Missed로 이동하여 On-time이 18대에서 10대로 감소하였다. 같은 데이터셋의 두 모델이 같은 조치에 상반되게 반응한 것이며, 행 단위 Recall은 손실의 크기만 알려줄 뿐 그것이 Alarm이 지나치게 이르기 때문인지 발생하지 않았기 때문인지를 구분하지 않는다. 이 구분은 다섯 판정 결과에서 얻어진다.

→ **동일한 Recall 손실이라도 Early와 Missed의 구성에 따라 필요한 조치가 반대가 되며, 이 구분은 다섯 판정 결과를 통해서만 얻어진다.**

이상의 결과에서 본 실험 조건의 행 단위 지표는 운영 성능의 대리 지표로 쓸 수 없음을 보였다. 점검 대상이 몇 대인지, Alarm이 대응 가능한 시점에 발생하는지, 탐지 손실이 Early인지 Missed인지는 어느 것도 행 단위 지표에서 복원되지 않으며, 조합 사이의 순서도 보존되지 않으므로 그 값의 크기로 우열을 대신할 수도 없다. 이상은 HDD 수·고장률·관측 기간이 서로 다른 세 데이터셋에서 모두 확인되었다.

→ **본 실험 조건에서 행 단위 지표는 점검 대상 규모, Alarm의 적시성, Early와 Missed의 구성을 나타내지 못하고 조합 사이의 순서도 보존하지 않으므로 운영 성능의 대리 지표로 사용할 수 없으며, 이는 데이터 특성이 다른 세 데이터셋에서 모두 확인되었다.**

### 5.2 평가 조건에 따른 모델 선택의 변화

앞 절의 차이가 모델 선택에서 어떻게 나타나는지를 seed 단위로 확인한다. 각 seed에서 Recall이 가장 높은 모델을 1위로 판정하고, 동점인 경우 공동 1위 모델의 집합이 완전히 일치할 때만 유지로 집계하였다. 행 단위 평가$(\tau_{\mathrm{row}})$와 운영 환경 기반 평가$(\tau_{\mathrm{op}})$는 각각 자신의 오탐 지표를 1% 이하로 제약한 상태이므로, 두 조건의 Recall 순위는 같은 오탐 수준에서 더 많은 고장을 탐지하는 모델의 순위에 해당한다. 반면 운영 환경 기반 평가$(\tau_{\mathrm{row}})$는 HDD-level FAR이 조합마다 2.33~8.77%로 달라(표 3) 오탐 부담을 통제하지 않은 순위이므로, 행 단위 기준으로 임곗값을 정한 모델을 그대로 운영에 투입했을 때 어느 모델이 선택되는지를 보여주는 참고 조건으로 두고 본 절의 결론은 두 평가를 각각 자신의 기준으로 사용하는 대조에 둔다. 표 4는 학습 변동만으로 1위가 흔들리는 정도와, 표시한 조건만 바꾸었을 때 30개 seed에서 1위 모델이 달라진 비율을 함께 적은 것이다.

→ **앞 절의 차이가 모델 선택에서 어떻게 나타나는지를 seed 단위로 확인하며, 두 평가를 각각 자신의 오탐 기준으로 사용한 대조를 결론 근거로 두고 임곗값을 통제하지 않은 조건은 참고로만 사용한다.**

표 4. 1위 모델의 변경 비율
Table 4. Change rate of the top-ranked model

| Changed factor | HGST | Seagate | Toshiba |
| --- | --- | --- | --- |
| *Seed only, per condition (435 pairs)* |  |  |  |
| ① Row-level ($\tau_{\mathrm{row}}$) | 63% | 51% | 65% |
| ② Operational ($\tau_{\mathrm{row}}$) | 45% | 58% | 35% |
| ③ Operational ($\tau_{\mathrm{op}}$) | 44% | 54% | 59% |
| *Evaluation Condition (30 seeds)* |  |  |  |
| Evaluation (① vs ②) | 97% | 63% | 87% |
| Threshold (② vs ③) | 100% | 37% | 47% |
| Both (① vs ③) | 73% | 60% | 87% |

표 4의 상단 세 행은 조건을 하나로 고정한 채 30개 seed의 모든 쌍에서 1위가 달라진 비율로, 학습 변동만으로 1위가 얼마나 흔들리는지를 나타내는 대조군이다. 하단 세 행은 같은 seed 안에서 두 조건의 1위가 달라진 비율이며, 괄호의 번호는 상단의 조건을 가리킨다. 하단의 각 값은 해당 비교에 쓰인 두 조건의 대조군 값과 견주어 읽는다.

→ **동일한 평가 조건에서도 학습의 무작위성만으로 1위 모델이 상당한 비율로 변하므로, 평가 조건에 따른 순위 변화를 이와 구분하여 해석해야 한다.**

평가 방식만 바꾸면 1위 모델은 대부분의 seed에서 달라졌다(표 4의 Evaluation, 63~97%). 다만 seed만 바꾸어도 1위는 35~65%에서 흔들리므로, 이 변경률이 대조군을 넘는 정도는 데이터셋에 따라 다르다. HGST(97%, 대조군 63%·45%)와 Toshiba(87%, 대조군 65%·35%)에서는 차이가 뚜렷하나 Seagate(63%, 대조군 51%·58%)는 대조군과 큰 차이가 없었다. 더 직접적인 근거는 1위 판정에 앞서는 모델 쌍 단위 비교에서 나온다. 데이터셋별 6쌍씩 18개 쌍 중 11개에서 우열의 다수 방향이 뒤바뀌었으며, 특히 Toshiba의 LightGBM은 LSTM 및 GRU에 대해 행 단위 평가에서 30개 seed 전부 열세였으나 운영 환경 기반 평가에서는 30개 seed 전부 우세로 나타나, 이 역전은 학습 변동으로 설명될 여지가 없다. 이 쌍에서 LightGBM은 HDD-level Recall과 FAR 양쪽에서 우세하므로(2.33% 대 8.25·8.77%), 순위 판정이 Recall 단독이라는 점도 이 결과에 영향을 주지 않는다.

→ **평가 방식만 변경해도 1위 모델이 대부분의 seed에서 달라졌으며, 일부 모델 쌍에서는 행 단위와 운영 환경 기반 평가 사이에 일관된 순위 역전이 나타났다.**

임곗값 기준만 바꾼 경우(표 4의 Threshold)의 영향은 데이터셋에 따라 갈려, HGST에서만 30개 seed 전부에서 1위가 바뀌어 트리 기반에서 순환신경망 기반으로 전환되었고 Seagate와 Toshiba에서는 대조군과 구분되지 않았다. 반면 두 변경을 함께 적용한 경우, 즉 각 평가를 자신의 기준으로 사용하는 방식에서는 변경률이 60~87%로 세 데이터셋 모두 대조군을 넘어, 관찰된 역전이 학습 변동만으로 설명되지 않았다. 다만 비교 대상 모델이 4개이고 30개 seed가 동일한 테스트 HDD 집합을 공유하므로 변경률의 크기 자체는 본 실험 조건에 한정하여 해석한다. 이상의 결과에서 행 단위 평가로 선정한 1위 모델이 운영 환경 기반 평가에서도 유지된다고 볼 근거는 얻지 못하였다.

→ **임곗값 기준 변경의 영향은 데이터셋에 따라 달랐으나, 평가 방식과 임곗값을 함께 변경하면 세 데이터셋 모두에서 1위 모델이 대조군보다 높은 비율로 변경되었으며, 행 단위 평가로 선정한 1위 모델이 운영 환경 기반 평가에서도 유지된다고 볼 근거는 확인되지 않았다.**

---

## 6. 결론

본 연구는 개별 관측 행의 분류 정확도만으로는 HDD 단위 Alarm 과정의 성능을 반영하기 어렵다는 문제의식에서 출발하여, 각 HDD의 SMART 이력을 시간 순으로 처리하고 최초 Alarm과 고장 또는 관측 종료 시점의 관계로 각 HDD를 다섯 판정 결과 중 하나로 구분하는 운영 환경 기반 평가를 구성하였다. 3개 데이터셋과 4개 모델, 30개 random seed에 동일한 예측 결과를 두고 두 평가 절차와 두 임곗값 기준을 적용하였다.

→ **본 연구는 개별 행의 분류 정확도만으로는 HDD 단위 Alarm 과정의 성능을 반영하기 어렵다는 문제에서 출발하여, 시간 순 온라인 추론과 최초 Alarm을 중심으로 한 운영 환경 기반 평가를 구성하였다.**

같은 예측 결과에서도 두 평가는 다른 성능을 나타냈다. 행 단위 FAR을 약 1%로 제한한 조건에서 점검 대상이 되는 HDD는 2.33~8.77%로 행 단위 값의 3.7~8.6배였고, 최초 Alarm의 Lead Time 중앙값은 12개 조합 전부에서 Prediction Horizon 밖에 있었으며, 탐지 손실의 47~70%는 Missed가 아니라 Early였다. 임곗값을 운영 기준으로 재선정하면 오탐은 제약 수준으로 돌아왔으나 HDD-level Recall은 단조적으로 반응하지 않았다. 모델 선택도 달라져, 평가 방식만 바꾸었을 때 18개 모델 쌍 중 11개에서 우열의 다수 방향이 뒤바뀌었고 각 평가를 자신의 기준으로 적용했을 때 1위가 달라진 비율은 60~87%로 세 데이터셋 모두 학습 변동만으로 바뀌는 수준을 넘었다.

→ **동일한 예측 결과에서도 두 평가는 점검 부담·탐지 시점·손실 구성에서 다르게 측정되었고, 1위 모델도 학습 변동으로 설명되지 않는 비율로 달라졌다.**

평가 대상을 개별 행의 분류 정확성에서 HDD의 시간적 Alarm 성능으로 옮기고, 우측 중도절단을 포함한 다섯 판정 결과로 각 지표의 모집단을 명시한 것이 본 연구의 의의다. 행 단위 분류 성능은 이 평가에서도 요구되며 Missed를 줄이는 것은 여기에 달려 있으나, 몇 대가 점검 대상이 되는지, 언제 위험을 감지하여 얼마의 대응 시간이 확보되는지, 남은 실패가 어느 쪽인지는 각 행을 독립으로 세는 절차에서 산출되지 않는다. 따라서 운영에 배치할 모델의 성능은 행 단위 결과만으로 판단할 수 없으며, 이 세 가지를 함께 제시하는 운영 환경 기반 평가를 더해 측정해야 한다.

→ **점검 대상의 규모, Alarm의 적시성, 탐지 손실의 구성은 행 단위 집계에서 산출되지 않으므로, 운영에 배치할 모델은 행 단위 지표에 운영 환경 기반 평가를 더해 선택해야 한다.**

이 결과는 Backblaze 데이터셋 3종과 예측 모델 4개라는 제한된 조건에서 얻은 것이므로 다른 설비로 일반화하기 어렵고, 기준선으로 삼은 행 단위 평가도 Sliding Window 등 다른 절차와의 차이를 검증할 필요가 있다. 비교 기준선이 행 단위 평가 하나이므로 제안한 절차가 표 1에서 HDD를 평가 단위로 삼은 프로토콜보다 나은지는 확인하지 못하였고, 학습을 행 단위 이진 분류로 고정하였으므로 최초 Alarm 판정을 직접 학습 목표로 삼았을 때의 성능도 확인하지 못하였다. 그룹 분할로 학습과 테스트가 같은 달력 구간을 공유한다는 점, FAR 1%와 Prediction Horizon 30일이 비교를 위한 설정이라는 점도 한계이며, 시간 분할과 다양한 $H$·FAR 제약에서의 검증이 뒤따라야 한다.

→ **본 연구는 제한된 데이터셋·모델과 비교 조건으로 수행되어 일반화 가능성, 기준선의 범위, 학습 방식, 데이터 분할 및 평가 설정 측면에서 추가 검증이 필요하다**

제안한 평가 방법 자체에도 남은 문제가 있다. 이 절차는 고장이 관측된 HDD와 충분히 긴 중도절단 이력을 함께 요구하므로 관측 규모나 기간이 작은 환경에는 적용하기 어렵고, HDD-level FAR이 소수 사건으로 추정되는 것도 같은 제약에서 비롯되어 임곗값 선정의 안정성에까지 영향을 준다. 이 지표는 우측 중도절단이 고장 위험과 독립이라는 가정에도 의존하며, 생존 분석과 같이 중도절단을 확률 모형에 직접 반영하면 가정을 완화할 수 있으나 그 경우 다섯 판정 결과가 갖는 직접적인 운영 해석은 일부 포기하게 된다. 또한 본 평가는 예측이 HDD 단위 Alarm으로 전환되기까지를 대상으로 하므로, Alarm 정책과 재학습을 포함한 동적 운영 환경으로의 확장이 향후 과제로 남는다.

→ **제안한 평가 방법은 데이터 규모와 중도절단 가정, 소수 사건에 따른 FAR 추정의 안정성, 그리고 Alarm 정책과 재학습을 포함한 동적 운영 환경으로의 확장이라는 한계를 가진다.**

---

## 참고문헌

[1] E. Pinheiro, W.-D. Weber, and L. A. Barroso, "Failure trends in a large disk drive population", Proc. 5th USENIX Conf. on File and Storage Technologies (FAST), San Jose, CA, USA, pp. 17-28, Feb. 2007.

[2] M. M. Botezatu, I. Giurgiu, J. Bogojeska, and D. Wiesmann, "Predicting disk replacement towards reliable data centers", Proc. 22nd ACM SIGKDD Int. Conf. on Knowledge Discovery and Data Mining (KDD), San Francisco, CA, USA, pp. 39-48, Aug. 2016. https://doi.org/10.1145/2939672.2939699

[3] L. Hu, L. Han, Z. Xu, T. Jiang, and H. Qi, "A disk failure prediction method based on LSTM network due to its individual specificity", Procedia Computer Science, Vol. 176, pp. 791-799, Sep. 2020. https://doi.org/10.1016/j.procs.2020.09.074

[4] M. Züfle, F. Erhard, and S. Kounev, "Machine Learning Model Update Strategies for Hard Disk Drive Failure Prediction", Proc. 20th IEEE Int. Conf. on Machine Learning and Applications (ICMLA), Pasadena, CA, USA, pp. 1379-1386, Dec. 2021. https://doi.org/10.1109/ICMLA52953.2021.00223

[5] J. Ahmed and R. C. Green II, "Leveraging survival analysis in cost-aware deepnet for efficient hard drive failure prediction", Neural Computing and Applications, Vol. 37, pp. 1089-1104, Jan. 2025. https://doi.org/10.1007/s00521-024-10479-6

[6] W. Li, P. Ma, and M. Ren, "A hybrid TCN-LSTM-attention network for disk failure prediction", Proc. 6th Int. Symposium on Computer Technology and Information Science (ISCTIS), pp. 261-265, 2026. https://doi.org/10.1109/ISCTIS70043.2026.11572580

[7] S. A. Fahrenkrog-Petersen, N. Tax, I. Teinemaa, and M. Dumas, "Fire now, fire later: alarm-based systems for prescriptive process monitoring", Knowledge and Information Systems, Vol. 64, pp. 559-587, Feb. 2022. https://doi.org/10.1007/s10115-021-01633-w

[8] H. Wang, Y. Yang, and H. Yang, "Hard disk failure prediction based on LightGBM with CID", Proc. IEEE Symposium on Computers and Communications (ISCC), pp. 1-7, Sep. 2021. https://doi.org/10.1109/ISCC53001.2021.9631504

[9] J. Ahmed and R. C. Green II, "Predicting severely imbalanced data disk drive failures with machine learning models", Machine Learning with Applications, Vol. 9, p. 100361, Sept. 2022. https://doi.org/10.1016/j.mlwa.2022.100361

[10] S. Han, P. P. C. Lee, Z. Shen, C. He, Y. Liu, and T. Huang, "StreamDFP: A general stream mining framework for adaptive disk failure prediction", IEEE Transactions on Computers, Vol. 72, No. 2, pp. 520-534, Feb. 2023. https://doi.org/10.1109/TC.2022.3160365

[11] W. Li, H. Zhou, S. Radhakrishnan, and S. V. Kamarthi, "Explainable time series features for hard disk drive failure prediction", Engineering Applications of Artificial Intelligence, Vol. 152, p. 110674, Jul. 2025. https://doi.org/10.1016/j.engappai.2025.110674

[12] S. Han, J. Wu, E. Xu, C. He, P. P. C. Lee, Y. Qiang, Q. Zheng, T. Huang, Z. Huang, and R. Li, "Robust data preprocessing for machine-learning-based disk failure prediction in cloud production environments", arXiv preprint arXiv:1912.09722, 2019. https://arxiv.org/abs/1912.09722

[13] S. Lu, B. Luo, T. Patel, Y. Yao, D. Tiwari, and W. Shi, "Making disk failure predictions SMARTer!", Proc. 18th USENIX Conf. on File and Storage Technologies (FAST), Santa Clara, CA, USA, pp. 151-167, Feb. 2020.

[14] S. Wei, X. Lu, H. Yang, C. Tu, J. Guo, H. Sun, and Y. Feng, "DFPoLD: A hard disk failure prediction on low-quality datasets", Informatics, Vol. 12, No. 3, p. 73, Jul. 2025. https://doi.org/10.3390/informatics12030073

[15] M. Tuttle, C. C. H. M. Maas, J. An, B. S. Wessler, W. F. Harvey, H. P. Selker, D. van Klaveren, and D. M. Kent, "Patient Versus Prediction-Level Evaluation of a Dynamic Clinical Prediction Model of Sepsis", medRxiv, May 2026. https://doi.org/10.64898/2026.05.26.26354141

[16] C. G. Scully and C. Daluwatte, "Evaluating performance of early warning indices to predict physiological instabilities", Journal of Biomedical Informatics, Vol. 75, pp. 14-21, Nov. 2017. https://doi.org/10.1016/j.jbi.2017.09.008

[17] C. Daluwatte, F. Yaghouby, and C. Scully, "A framework to characterize the performance of early warning index alarm systems for patient monitoring", MethodsX, Vol. 6, pp. 1660-1667, Jul. 2019. https://doi.org/10.1016/j.mex.2019.07.003

[18] Backblaze, "Hard Drive Data and Stats." https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data. [accessed: Aug. 18, 2026]