26.8.11


정보기술학회 게재, KCI 등재 목표

- 데이터셋
    
    https://drive.google.com/drive/folders/18TOKVO7WfFPBD3UTUz4EEedY6UjSC2ct?usp=sharing
    

[[테스트] 롤링추론 결과 및 비교](https://app.notion.com/p/3a980c45869980a09384d4c27cbcd6ba?pvs=21)

kci 평가기준 

구버전

# 제목: SMART 기반 HDD 고장 예측 모델의 운영 환경 기반 성능 평가

> 
> 
> 
> #### 운영 환경을 반영한 성능평가 방법을 제안하고, 기존 평가와 비교하여 그 필요성을 보이는 것
> 

| 용어 | 뜻 |
| --- | --- |
| 행 단위 평가 (Row-level evaluation) | 각 관측 행을 독립적인 분류 대상으로 평가하는 방식 |
| 운영 환경 기반 평가 (Operational evaluation) | HDD의 시간 순차적 관측 과정을 유지하고 실제 운영 상황을 반영하여 평가하는 방식 |
| 온라인 추론 (Online Inference) | 각 HDD의 관측값을 시간 순으로 입력하여 각 시점에서 고장 예측을 수행하는 과정 |
| 최초 Alarm (First Alarm) | 한 HDD에서 의사결정 임곗값을 최초로 초과하여 발생한 Alarm |
| Lead Time | 최초 Alarm 발생 시점부터 실제 고장 발생 시점까지의 시간 |
| 예측 기간 (Prediction Horizon) | 최초 Alarm이 On-time으로 인정되는 고장 이전의 시간 범위 |
| On-time Alarm | 예측 기간 이내에 발생한 최초 Alarm |
| Early Alarm | 예측 기간을 초과하여 이른 시점에 발생한 최초 Alarm |
| Censored Early Alarm | 실제 고장 시점이 관측되지 않은 우측 검열 HDD에서 평가 구간 내 발생한 최초 Alarm |
| Missed Failure | 고장이 관측된 HDD에서 최초 Alarm이 발생하지 않은 경우 |
| Censored No Alarm | 우측 검열 HDD에서 평가 구간 내 최초 Alarm이 발생하지 않은 경우 |
| OAP | 고장 관측 HDD 중 최초 Alarm이 발생한 HDD에서 On-time으로 분류된 HDD의 비율 |
| ODR | 실제 고장 HDD 중 On-time Alarm으로 탐지된 HDD의 비율 |
| EAP | 전체 평가 HDD 중 Early 또는 Censored Early로 분류된 HDD의 비율 |
| Median Lead Time | 최초 Alarm이 발생한 고장 관측 HDD의 Lead Time 중앙값 |

## 연구 질문

- RQ1. 동일한 모델의 성능을 행 단위 평가와 운영 환경 기반 평가로 측정할 경우, 성능 평가 결과에 어떤 차이가 나타나는가?
- RQ2. 운영 환경을 기준으로 임곗값을 최적화하면 기존 Row 기반 최적화 결과와 어떤 차이가 나타나는가?
- RQ3. 운영 환경 기반 성능평가는 기존 평가에서 확인하기 어려운 어떤 운영 정보를 제공하는가?

## 요약

- 다양한 제조사의 Backblaze HDD 데이터를 이용하여 머신러닝 및 딥러닝 기반 HDD 고장 예측 모델을 대상으로 기존 Row 기반 성능평가와 운영 환경 기반 성능평가를 비교하였다.
- 실제 HDD 운영 절차를 모사하기 위해 HDD의 전체 관측 기간을 대상으로 시간 순의 온라인 추론을 수행하고, 각 HDD에서 발생한 최초 알람을 기준으로 운영 환경의 성능을 평가하였다.
- 동일한 모델과 동일한 예측 결과를 사용하더라도 평가 단위와 임곗값 최적화 기준에 따라 Precision, Recall, FAR 및 Lead Time이 달라졌으며, 기존 평가에서 우수하게 평가된 모델이 운영 환경에서는 상대적으로 낮은 성능을 보이는 등 모델 간 상대적 성능 순위가 달라지는 현상을 확인하였다.
- 또한 기존 Row-level 평가와 운영 환경 기반 평가 사이에는 지표별로 상관 정도가 다르게 나타났으며, 일부 모델에서는 평가 기준에 따라 상대적 순위가 변화하였다. Row 단위 분류 성능만으로 실제 운영 환경에서의 모델 성능을 판단하는 데 한계가 있음을 확인하였다.
- 운영 환경 기반 평가는 기존 분류 성능 지표뿐만 아니라 최초 Alarm 발생 시점, Lead Time 및 False Alarm 발생 특성과 같은 시간적·운영적 정보를 함께 제공하여, 모델의 운영 특성을 추가적으로 평가할 수 있음을 확인하였다.
- 따라서 HDD 예지보전 모델의 성능평가는 단순한 Row 단위의 분류 정확도에 한정하기보다, 실제 운영 과정에서의 알람 발생과 고장 개체 단위의 성능 및 충분한 Lead Time을 함께 고려하는 방식으로 수행될 필요가 있다.

## 1. 서론

<aside>

*1장.* 

*데이터 저장 수요의 증가와 함께 HDD는 대규모 저장 인프라에서 널리 활용되고 있다. HDD 고장은 데이터 손실과 서비스 운영에 영향을 줄 수 있으므로, 고장을 사전에 예측하여 유지보수 전략을 수립하는 것이 중요하다. 그러나 HDD 고장 예측 모델의 성능은 평가 단위와 방법에 따라 실제 운영 관점에서 서로 다르게 해석될 수 있으며, 개별 관측치 또는 일정한 예측 구간을 중심으로 한 평가만으로는 HDD가 시간에 따라 운영되는 과정에서 발생하는 경보의 특성을 충분히 반영하기 어려울 수 있다.*

</aside>

### 1.1 연구 배경 및 문제 제기

- HDD 고장 예측은 대규모 스토리지 시스템의 안정적인 운영과 예방적 유지보수를 위해 중요하며, 머신러닝과 딥러닝을 활용한 연구가 지속적으로 발전해 왔다.
- HDD의 SMART 데이터는 HDD별로 시간에 따라 반복적으로 수집되며, 실제 운영에서는 개별 관측치가 아닌 HDD 자체가 유지보수 및 관리의 대상이 된다.
- 그러나 기존 성능평가는 주로 개별 Row 또는 특정 시점·구간의 예측 결과를 기준으로 이루어져, 하나의 HDD에서 시간에 따라 반복적으로 발생하는 예측과 Alarm의 운영적 의미를 충분히 반영하기 어렵다.
- 실제 운영에서는 최초 Alarm 시점, Lead Time, 반복 Alarm 및 오탐 발생 시점 등 **HDD 전체 생애주기에서 나타나는 경보 특성**을 함께 고려할 필요가 있다.
- 따라서 개별 예측의 분류 성능뿐만 아니라, **HDD를 운영 단위로 하여 시간에 따른 예측과 Alarm을 평가하는 관점이 필요하다.**

### 1.2 연구 목적 및 기여

- HDD 전체 생애주기의 시간적 운영 과정을 반영한 운영 환경 기반 성능평가 방법을 제안한다.
- HDD별 시간 온라인 추론 결과를 Alarm으로 변환하고, 최초 Alarm과 실제 고장 시점의 관계를 이용하여 운영 환경 기반 성능과 Lead Time을 평가한다.
- 기존 Row-level 평가와 운영 환경 기반 평가를 동일한 예측 결과에 적용하여, 평가 단위에 따른 성능 및 모델 순위의 차이를 분석한다.
- 각 평가 기준에서 임곗값을 최적화하여, 평가 기준에 따른 임곗값 선택과 모델 성능의 관계를 비교한다.
- Lead Time, First Alarm, 오탐 발생 시점 및 HDD 생애주기상의 Alarm 변화 등을 분석하여, 기존 Row-level 평가만으로 확인하기 어려운 운영 특성을 분석할 수 있음을 보인다.

---

## 2. 관련 연구

<aside>

*2장. 
HDD 고장 예측 연구에서는 다양한 예측 모델과 평가 방법이 제안되어 왔다. 선행 연구에서는 개별 Row, 일정 Window, 특정 예측 기간 및 HDD 단위 등 다양한 평가 단위가 사용되고 있으며, 일부 연구에서는 FAR이나 Lead Time과 같은 운영 관련 지표도 활용하고 있다. 그러나 운영 관련 지표를 사용하더라도 개별 Row의 예측 결과를 독립적인 분류 대상으로 취급하여 지표를 산출하는 경우, 동일 HDD에서 반복적으로 발생하는 예측과 Alarm이 하나의 운영 단위에서 갖는 의미를 충분히 반영하기 어렵다. 실제 운영에서는 여러 시점의 예측이 하나의 HDD에 대한 반복적인 Alarm으로 나타나므로, Row 단위에서 산출한 성능이 HDD 단위의 운영 성능으로 그대로 이어진다고 보기 어렵다.*

*한편, 예지보전 및 의료 AI 등 시계열 예측을 활용하는 분야에서는 개별 예측의 정확성뿐만 아니라, 반복적인 예측 결과와 실제 사건의 관계를 개체 단위에서 평가하려는 방법론이 제안되고 있다. 이러한 연구 흐름은 개별 관측치의 분류 성능과 실제 운영 단위에서의 성능이 서로 다를 수 있으며, 반복적인 예측을 실제 사건 및 운영 의사결정과 연결하여 평가할 필요가 있음을 보여준다. 이에 본 연구에서는 HDD의 시간적 운영 과정을 반영한 운영 환경 기반 평가 방법을 제안한다.*

</aside>

- HDD SMART 기반 고장 예측 연구 정리
    
    
    | 논문 | 유형 |  | 연구 목적 | 데이터셋 | 모델 | 입력 방식 | 레이블 정의 | 평가 절차 | 유형 정리 | 주요 성능평가 | 비고 | 링크 |
    | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
    | Failure Trends in a Large Disk Drive Population (USENIX FAST 2007 / Pinheiro et al., Google Inc.) |  |  | 대규모 인터넷 서비스 인프라(Google 데이터센터)에서 운영 중인 수십만 대의 상용 하드디스크 드라이브(HDD) 모집단을 대상으로 실제 운영 환경에서의 고장 통계를 수집하고, 온도, 가동률(활동 수준), SMART(Self-Monitoring Analysis and Reporting Technology) 지표 등 고장 수명에 영향을 미친다고 알려진 주요 요인들과의 상관관계를 통계적으로 분석 | Google의 대규모 프로덕션 환경에서 수집된 **10만 대 이상**의 소비자용 ATA 하드디스크 드라이브(5400~7200 rpm, 80~400 GB)로부터 9개월간(2005년 12월~2006년 8월) 수집된 주기적 SMART 파라미터, 환경 지표, 가동률 로그 및 5년간의 수리/교체 이력 데이터 | 통계적 상관관계 분석 모델 (분할 표 분석, 생존 분석/Survival Analysis, 카이제곱 및 통계적 유의성 검정 기반 데이터 마이닝) | 각 서버에서 수분 주기로 수집되는 실시간 센서 환경 변수(온도), 읽기/쓰기 대역폭 기반 가동률(Utilization), 디스크 자가 진단 지표(SMART 파라미터: 스캔 에러, 재할당 섹터 수, 오프라인 재할당, 보류 중인 프로비저널 카운트 등) | 관리자 수리 절차에 따라 장애 또는 성능 저하로 인해 디스크 교체(Drive Replacement)가 수행된 사건을 고장(Failure)으로 정의 | 대규모 분산 데이터 분석 프레임워크(Bigtable, MapReduce, Sawzall)와 통계 패키지(R)를 활용해 수집된 대규모 시계열 데이터를 정제·필터링하고, 연간 고장률(AFR, Annualized Failure Rates), 연령대별/가동률별/온도별 고장 분포, 그리고 특정 SMART 지표 발동 전후의 생존 확률 분석 수행 |  | 스캔 에러(Scan Error), 재할당 섹터 수(Reallocation Count), 오프라인 재할당, 프로비저널 카운트 등의 특정 SMART 신호는 고장 확률을 수십 배 높이는 높은 상관관계를 보임. 그러나 전체 고장 난 디스크의 **56% 이상**이 이러한 주요 SMART 에러 신호가 전혀 감지되지 않은 상태에서 고장 나므로, SMART 데이터 단독으로는 개별 부품의 고장 예측 모델을 정확히 구축하기 어렵다는 결론 도출. | 하드디스크 고장 예측 및 신뢰성 연구 분야에서 **최초로 초대규모 프로덕션 환경의 실측 데이터를 기반으로 통계적 분석**을 수행하여 기존 제조사의 가속 수명 시험 결과와 통념을 정량적으로 반박한 마중물 같은 클래식 연구 논문 | https://static.googleusercontent.com/media/research.google.com/en//archive/disk_failures.pdf |
    | Predicting Disk Replacement towards Reliable Data Centers (2016) |  |  | 데이터센터에서 HDD 교체 시점을 예측하기 위한 디스크 교체(Replacement) 예측 프레임워크 제안 | 실제 데이터센터 운영 데이터 (Seagate·Hitachi, 30,000+ HDD, 17개월) | Regularized Greedy Forest (RGF) | Changepoint Detection으로 선택한 SMART 속성에 지수평활(Exponential Smoothing) 적용 | 유지보수를 위해 교체된 디스크를 Positive, 정상 운영 디스크를 Negative로 정의한 이진 분류 | 일정 길이의 타임 윈도우로 잘라 시계열 정보를 담은 하나의 행으로 압축. 각 행을 독립적으로 이진 분류. 현재 row의 상태 예측 (레이블링 규칙 미기재)
    행 단위 Precision, Recall, F1 계산
    추가적으로 조기 탐지 성능 평가(생애 시뮬레이션이 아닌, 여러 스냅샷에 넣어보는 방식) | 특정 기간만 사용, 독립적인 행 단위 예측, 행 단위 평가 | Seagate 기준 Precision/Recall 약 98% (10~15일 Prediction Horizon), Hitachi 81~84% | Disk 단위 평가를 수행하지만 HDD 생애 동안의 연속 추론이 아니라 단일 입력 기반 교체 여부를 예측함. | https://dl.acm.org/doi/epdf/10.1145/2939672.2939699 |
    | Making Disk Failure Predictions SMARTer! (2020) |  |  | SMART 정보와 성능 지표 및 위치 정보를 결합하여 실제 데이터센터 환경에서 HDD 고장 예측 성능 향상 | 실제 데이터센터 운영 데이터 (약 38만 HDD, 64개 사이트) | CNN-LSTM (비교: Bayes, RF, GBDT, LSTM) | SMART + 성능 지표 + 위치 정보(SPL)를 시계열 입력 | 향후 10일 이내 고장 여부를 예측하는 이진 분류, 레이블은 운영자의 실제 교체 결정(read/write 실패 + 재시작 후 미작동)을 기준으로 한 이진 라벨 | 레이블링을 변화하면서 실험. 전체 HDD 생애를 전부 사용. 각 행을 독립적으로 분류.
    행 단위 Precision, Recall, F1, MCC 계산 | 전체 생애 사용, 독립적인 행 단위 예측, 행 단위 평가 | Prediction Horizon 10일 기준 F-measure 0.95, MCC 0.95 | Window 단위 예측 성능 평가이며 물리적 HDD의 최초 알람 기반 평가는 아님. | https://www.usenix.org/system/files/fast20-lu.pdf |
    | Predicting severely imbalanced data disk drive failures with machine learning models (2022) |  |  | 극심한 클래스 불균형과 데이터 누수를 고려한 HDD 고장 예측 머신러닝 프레임워크 제안 | Backblaze 2014·2017 (5개 서브 데이터셋, ST4000DM000 등) | BRF, EasyEnsemble, WLR (비교: RF, DT) | MaxAbsScaler를 적용한 SMART 원시 속성(주요 5개 SMART 포함) | 특정 시점의 HDD 상태를 고장/정상으로 분류하는 이진 분류 | hdd고장 예측 분야에서 클래스 불균형을 제대로 다루기 위해 이지앙상블과 비용민감학습을 도입해 불균형 문제를 다루는 논문, 이진분류 목적을 가짐 
    명시적인 n일 이내 positive 규칙x- 기존 데이터셋 구성을 그대로 재사용(d1/d2는 1일, d3/d4/d5는 아마 원 논문 구성 그대로)
    독립행 정적 분류
    평가 단위는 명시적x, 아마 행단위로 추정
    평가지표는 G-mean(주지표), FDR, FAR, AUC | 독립적인 행 단위 예측, 행 단위 평가 | BRF: Gmean 0.86, FDR 81%, FAR 9% | 데이터 불균형 처리에 초점을 둔 Row 단위 분류 연구. | https://pdf.sciencedirectassets.com/777839/1-s2.0-S2666827022X00030/1-s2.0-S2666827022000585/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEPz%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJIMEYCIQDEGaHygqNKWa3uajVZpzYc9Nwc4WezemO1oiMAnGezngIhAKPGpMXo5PfsiGXsfiwtNDLD5WQGeHeuLvNV96cHF7XAKrwFCMT%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEQBRoMMDU5MDAzNTQ2ODY1IgzUbuzYNMmKQrUwdIQqkAWvrohlYGHUYcJ3o7hJksSsvWt2Nkn8fxuDPrS9h8WfP7kB%2FxlJBN2JmQ1HqlXz7q1ur8kzUQuHGKlz247xCTYN1r1cEo5gR0VGqfBynFDZtobmluj1XeGW7LhdFaJ7yrGnNvNwu%2F%2BwGMUVQHTdDAJjS8rL9pkUVYN8tultwAzsXnhClFji9sd42zeb3H9yVBgyB1W0Fqr0hkuXCABSk2lt72Sz43%2FEawfOv2EED7ItF%2BPoioHGZD7PEDMlBajOLHpV7xcZ3xSHNjsiGbzTZv8W%2BRCNnbashDQdPjAXDaAKv1bXPmmBXKfHRVaX0QbDlYhN2JNfAIH2Smn0K2paMqTFlkUUIZsanwvJl%2FJAEWiwwBNUmPeaGf2tEE5dr%2Fd1mdVkTyGNpKMqQKRihPP%2FJH0u0KKOd4Ld8cBNLdDAY6%2FSKG9gI7ID7kFHbZHIHEM3nWgh9O3IBs08yOM9u9qzVJdxSaaooA32X9cE00EJdugi4cU%2F9zEigjzuGB0yQMZzQSq2Nc3TjFcqZG%2F5J1BxylvAy557AAbeMpwEgCNvkYbmGFQqPW4Hvoaz%2F9Fz77RwgwNAGG3fUM7JHOOO76qSQXymk8v7BimmCrz%2BGkX%2BcV5JTxqXOum4rzPuzk9SUrSshB8aAdi60gF09NMjulxv7FDLoAk1WU8%2Fjs1W4v4tkXzSK4U%2FiFBlEZ%2BIPXJPuWIlDaUAZsrnnsupoAau%2F6KjRNnWJnj7QI8mdYAu%2BXeY3XKCK0aEUeYfgT9kyOmrjDfkBACVzPELeMNFlI9gwalLqwrshXR4Y3TdI8B3WPVrUv9zU7R%2BKxigQ8PHjhibMl%2Fzp4dk2ZPbuVBLbGxp73q%2BgJCjaYojKowFZ81uf758pkRjxDD2q7fTBjqwAe3tSoTne6kO7k9ftwswwWhGqR0UG5jLCwIFGfIIEvclo1bOhkVpmbMW6lujaA4VThuf4pSIqo1DBrpHVTvdA45%2BBdgCHrkkY7Uw5oxeY%2FExIaHwqx5OoO%2F3b49Rsil8dBVhk5u8Rc3%2BDJ31o7E9ifUO%2B8njW4DuS%2B1hpA4fGjnlvfbd3yMefg9rsnCw0Daob1hrJM1V%2Bw9Qy1u0j%2FhBFpM%2FwEOJyuKP3cYJWoKoeNvF&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20260801T120634Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTY766VAV5F%2F20260801%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=86ce07c517af3d8806d851681fbd82705c6b3c6f66de248afe7fb49414245e25&hash=5e4eaa1678bf3ffe2b554a92b57e42cdba2ce7f03955d3295bcba61c26282860&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S2666827022000585&tid=spdf-7b0a56c4-4d1e-4558-969b-0d592af8190e&sid=c83de2f23354814dd54b87474eb7826a4d3agxrqa&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&rh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=0d12045555550750015f&rr=a244b7b18b0cea27&cc=kr |
    | TFBEST: Dual-Aspect Transformer with Learnable Positional Encoding for Failure Prediction (2023) |  |  | 학습 가능한 위치 인코딩과 듀얼 인코더를 적용한 Transformer 기반 HDD 잔여 수명(RUL) 예측 모델 제안 | Backblaze 2013~2021 (Seagate ST4000DM000) | TFBEST (Dual-Encoder Transformer) | 고장 전 60일 데이터를 이용한 길이 30의 Sliding Window 시퀀스 | 잔여 수명(RUL = 고장일 − 관측일) 회귀 | RUL 회귀 모델을 학습한 후 RMSE 기준으로 기존 LSTM·Transformer와 성능 비교, 90% 신뢰구간 산출 |  | RMSE 9.54 (DAST 13.10, LSTM 15.25 대비 개선), 90% Confidence Interval 제공 | 잔여수명(RUL) 회귀 연구로 운영 기반 분류 평가와 목적이 다름. | https://arxiv.org/pdf/2309.02641 |
    | A Hybrid TCN-LSTM-Attention Network for Disk Failure Prediction (2026) |  |  | 기존 모델들이 단일 차원의 정보(시계열 또는 다중 변수 상관관계)만 다루어 글로벌 시간 동역학과 복잡한 피처 간 상호작용을 동시에 고려하지 못하는 한계를 극복하기 위해, TCN, LSTM, Attention 메커니즘을 결합한 하이브리드 딥러닝 고장 예측 프레임워크 제안 | Backblaze 공개 SMART 데이터 중 Seagate ST4000DM000 모델 (2015~2025년 데이터, 약 8,600만 건 중 슬라이딩 윈도우 및 균형화 전처리를 거친 180,000개 샘플 사용) | TCN-LSTM-Attention 하이브리드 네트워크 (비교군: Online Random Forest, LSTM, TCN) | 결측치 처리 및 이상치 스무딩, 누락률 50% 이상 피처 제거, XGBoost 기반 피처 중요도 선택을 거친 후, **TimeGAN**을 활용한 소수 클래스(고장 샘플) 데이터 증강 및 언더샘플링을 결합하여 정규화된 30일 길이의 슬라이딩 시계열 윈도우를 입력 | 향후 7일 이내 고장 발생 여부 (Binary Classification: 고장 1, 정상 0) | 향후 7일 이내 고장 발생 여부를 예측 목표로 이진분류함. 
    30일 고정 길이 슬라이딩 윈도우시퀀스 세그먼트, 독립 샘플 취급. 30일치 데이터를 보고 그 시점 기준 7일 내 고장여부를 예측.
    윈도우단위의 Precision, F1-score(주지표), FAR, FDR 계산. | 전체 생애 사용, 독립적인 행 단위 예측, 행 단위 평가 | F1-score **0.92**, FDR(결함 탐지율/Recall) **0.91**, Precision **0.93**을 달성하여 기존 단일 모델(TCN, LSTM 등) 대비 우수한 예측 성능과 신뢰성 입증 | 30일 Sliding Window 기반 분류 성능만 평가하며, HDD 전체 수명 동안의 연속 추론이나 최초 알람 기반 운영 평가는 수행하지 않음. | https://ieeexplore-ieee-org-ssl.openlink.mju.ac.kr/stamp/stamp.jsp?tp=&arnumber=11572580 |
    | Large-scale End-of-Life Prediction of Hard Disks in Distributed Datacenters (2023) |  |  | 대규모 장기 데이터를 활용한 HDD 잔여 수명(RUL) 예측 및 모델 일반화 성능 검증 | Backblaze 2013–2022 (주로 Seagate ST4000DM000) | Encoder–Decoder LSTM (Seq2Seq LSTM) | XGBoost 기반 특징 선택 후 25일 Sliding Window 시계열 입력 | 잔여 수명(RUL = 고장일까지 남은 일수) 회귀 | 시간순(Train: 2013–2019 / Validation: 2020 / Test: 2021–2022)으로 학습·평가 후, 다른 Seagate HDD 모델에 전이하여 일반화 성능 검증 |  | RMSE 0.86, R² 0.98 (최적 설정), 타 Seagate 모델에서도 일반화 성능 확인 | RUL 회귀 연구이며 Disk 단위 유지보수 의사결정 평가는 수행하지 않음. | https://arxiv.org/pdf/2303.08955 |
    | A disk failure prediction method based on LSTM network due to its individual specificity (2020) |  |  | HDD의 개별 특이성을 고려한 LSTM 기반 고장 예측 모델 제안 | Backblaze (ST4000DM000, ST8000DM002) | Stacked LSTM (비교: RF, DT, SVM) | Pearson 검정으로 선택한 10개 SMART 속성, Z-score 정규화 후 **15일 Sliding Window** 입력 | 10개 SMART 속성, Z-score 정규화 후 15일 Sliding Window 입력향후 15일 이내 고장 여부를 예측하는 이진 분류 | 고장전 15일 구간을 positive로 레이블링. 현재 row의 레이블을 맞추는 이진분류.
    그 구간 내 행 단위 배분 규칙(모든 행에 1을 붙이는지, 윈도우 라벨 1개인지)은 논문에 명시X
    고정길이(15)슬라이딩 윈도우 시퀀스를 입력단위로 씀. 디스크 전체 생애를 통째로 넣는 것도 아니고 하루짜리 독립 행을 넣는 것도 아님 
    윈도우 단위 FDR, FAR, Preicision, F1score 계산. | 독립적인 행 단위 예측, 행 단위 평가 | Precision 86.31%, FAR 1.3%, FDR ≈80% | Sliding Window 기반 Row 분류 평가로 HDD 전체 운영 과정은 고려하지 않음. | https://pdf.sciencedirectassets.com/280203/1-s2.0-S1877050920X00147/1-s2.0-S1877050920319700/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEPz%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJHMEUCIE%2Fe40yCoe3%2FyxKIX0dYja4n0po3itAq1remU0oHuE1CAiEA4lp2fqVybYiqRUKSRDdKCWtaTYKp%2FLymDw0Y2hblKaQqvAUIxP%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARAFGgwwNTkwMDM1NDY4NjUiDEAnow5xyEky9IkcUCqQBQ6dHnDyQW9YhVKdtzyYjzxRPrz4sH8neFUSzYTttfaquFQ46sz2FiVFOCsIIVGdpE%2FHa0%2FAvlhab%2F0UEQmuzlo5XWzg7Kb3TyJcsDISsK%2FPk8xNor1diTGANsTZfc751PNh2j8ot1X5pLT3gVr6fvt%2BVqW9%2B8fQSYteji6xamEVaOhauO5GXm4SuEWapjtcTpTqdD4tRFgX3GQKOBb%2BH%2BzWedmDpNXG7Fy6faiavXRLlhislGIs982ZjgtIeSImLkPqz1M%2FiApBAxQ%2BKOzc70GMF25iSiMui56Lmh7WzluhjBnELmTAXmAA1FwG9hczBt3%2FoYn1bKWu8G5L7ybrkkTCFPJ3fLmqmiLbdhp%2BZMSmmjLjM4WPGnfBCNHsTIS0fCGV5jNkRKwVapegHTd4b7ub2dwV5n3RaaA88Y9c4sMlB9YWY5XvRMYjdk2oKPsGNWL92%2FuQuCaXjBQy2hhG%2BacBRPAogyOfFmwyZTiMSojU12yX2JH37R4gKsPSgBLvU1QfQGgaQDlK7gTi3WoctRCqpp2AgHVeoYTUNGW%2BnI2hI0Z0m9jLu%2F15yfHULC0wPFtg1XXUofr8sPHtAJ90mIDvrP4eMw51yyyQe%2FTKZZ8dDFASYyOEH5DXD5ejfJeenOJ8itpXL1D7U%2Bamio5HI3avLt9QD0x1D8yZMIRZNR6X3h%2BzbRhxeLk9PoaPtN9%2BlqngIvy0RMJf%2BHBF2VR2uaKy3uOr2sEUOdKz2%2FPA59HHOTtFoJlf7UHCoMxE4aCbxg4EoSv2lIDg49IzPgoF352O%2FLJ3%2Bj9YjSEmEpc4uDo10VMwMrBxd2BR%2FFrMe7gErK6yq%2BB5mwq3skSEGUahR%2BYrzhtfgA0llQ%2FL0CXQMeadMJ6ot9MGOrEB%2FHBow%2BtU1mtgngfTAsRBUFIZGEV8uG90oxIXWOFLp2Pp5Hhic1EFSOVvTMHEAsqtPHkHX4AIe7CshxcajtgbRACdJzVSJKlGlUqaCW%2F52fB0cGY3cptvsU%2FTDErYy6OdKJ6nA6oz26Eb6UpMFdFj6%2BUcIXkDVpj%2FL3VjrNPQUvniVZDDatFBxiNay3Hf%2Bw6aDCFL%2BYKip7sqpzuQf3UG1sq3AJgOUGDHOXsiImht0LWV&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20260801T120558Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYYGHBCMWI%2F20260801%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=c3f384287951b4ed93568c76cba5215f0a7bcb6436258ab85d43e02ea22de8b1&hash=57f53d66d1d4be73575814f2d2b5d9aa7bf5a71dfb1d0391503f9749044ea00c&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S1877050920319700&tid=spdf-0f9d770b-4aa4-4a75-a621-c581213e40ff&sid=c83de2f23354814dd54b87474eb7826a4d3agxrqa&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&rh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=0d12045555550751000a&rr=a244b6cdde81ea27&cc=kr |
    | Mechanisms for Integrated Feature Normalization and Remaining Useful Life Estimation Using LSTMs Applied to Hard-Disks (Basak, Sengupta, Dubey, 2019 IEEE SMARTCOMP) |  |  | 디바이스마다 고장을 일으키는 SMART 속성값의 범위(최소~최대)가 크게 다른 상황에서, 디바이스별 맞춤 정규화(device-specific normalization) 기법을 적용한 LSTM 기반 RUL(잔여수명) 온라인 예측 프레임워크를 제안. 특히 미래 정보를 전혀 쓰지 않는 실시간 온라인 시뮬레이션 환경에서도 동작하도록 설계하고, 학습된 모델이 동일 제조사의 다른 디스크 모델로도 일반화·전이 가능함을 입증하는 것이 목표 | Backblaze HDD 데이터셋. 2017년 1월~12월(1년치), 다양한 제조사의 91,243대 디바이스 중 **Seagate ST4000DM000** 모델을 선정(2017년 고장 통계상 Seagate가 가장 많이 고장났고, 그중에서도 ST4000DM000이 고장 기여도 1위였기 때문). 30개 SMART 지표(raw+normalized)를 24시간(1일) 간격으로 수집. 학습 데이터는 71,072개 샘플(71072 × 25 × 5 텐서) | 2-layer Stacked LSTM (레이어당 유닛 100개, Dropout 0.2). 비교군으로 Naive Bayes 분류기도 학습해봤으나 시뮬레이션(테스트) 단계에서 완전히 실패함을 보여 LSTM의 필요성을 뒷받침 | 24개 SMART raw 지표 중 상관계수와 Fisher Score 기반 필터링으로 **5개 핵심 SMART 변수 선정**: 이 5개 변수의 고장일 기준 과거 **150일치** 시계열을, LSTM에는 **25일(timesteps=25)** 단위 슬라이딩 윈도우(3차원 텐서: Samples × 25 × 5)로 잘라 입력. 정규화는 학습 시엔 실제 고장일 기준 Min-Max, 온라인 시뮬레이션(테스트) 시엔 미래 고장 시점을 모르므로 **과거 2개월치 이력 데이터의 75th 백분위수를 "가상 최댓값"으로 삼아 정규화**하는 독자적 기법("Prediction Strategy 2") 적용 | 회귀 문제로 설계: 각 시점(timestep)의 레이블은 **RUL = 고장일(Tf) − 현재 시점(j)**, 즉 "고장까지 남은 일수"를 직접 예측하는 연속값 회귀. 이후 평가 단계에서는 이 RUL 예측값을 "10일 이내 고장 여부"라는 고정 임계값으로 변환해 이진분류 형태의 Precision/Recall도 별도 산출 | RUL 연속값 회귀 (LSTM), 평가 시 이진분류로 변환. 10일 (평가용), 학습 라벨 범위는 0~125일.
    날짜별 잔존일수 연속 라벨, 윈도우당 라벨 1개(끝 시점 기준). 25일 고정 슬라이딩 윈도우 (실패 전 150일 구간에서 추출). 평가단위는 명시 안됨(device 단위와 day별 집계 혼재, 정확한 정의 불명). 	Precision / Recall / F1 (FDR/FAR 아님), 학습 loss는 MSE |  | 10일 이내 고장 예측 기준 **평균 Precision 0.8435(≈0.84), Recall 0.72, F1 0.77**. 7일 연속 측정에서도 곡선이 평탄(flat)해 시간에 따른 일관성·강건성을 확인. RUL 예측은 고장이 가까워질수록(EOL 근처) 예측 불확실성이 줄어듦을 시각적으로 확인. **일반화 검증**: ST4000DM000으로 학습한 모델을 동일 제조사의 다른 모델 **ST8000DM002**에 그대로 적용해도 RUL 예측이 양호하게 작동함을 확인(Prediction Strategy 2 적용 시) | (1) 미래 정보를 쓰지 않는 순수 온라인 시뮬레이션 조건에서 평가했다는 점이 핵심 차별점 — 저자들은 미래 정보를 허용했을 때 성능이 훨씬 좋아 보이지만 이는 "실제 상황을 대표하지 못하는 착시"라고 명시적으로 경계함. (2) Aussel et al.(RF 모델, Precision 0.93/Recall 0.6)과 비교해 본 연구는 Recall이 더 높은(0.72) 대신 Precision은 다소 낮음(0.84) — 저자는 이를 "미래 정보 없이도 실전에 준하는 성능"이라는 근거로 제시. (3) 동일 제조사 내 타 모델로의 전이·일반화를 명시적으로 입증한 최초 사례라고 저자들이 주장 | https://arxiv.org/abs/1810.08985 |
- HDD 고장 예측의 평가와 운영 관점 연구 정리
    
    
    | 논문 | 연구 목적 | 데이터셋 | 모델 | 입력 방식 | 평가 절차 | 유형 정리 | 주요 성능평가 | 비고 | 링크 |
    | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
    | Leveraging survival analysis in cost-aware deepnet for efficient hard
    drive failure prediction (2024) | 생존 분석 기반 샘플링과 비용 인지형 손실 함수를 적용한 HDD 고장 예측 프레임워크 제안 | Backblaze 2013–2022 (ST4000DM000) | 1D CNN | 고정 길이 시계열(Window) | 고장 전 15일 row를 Positive로 레이블링함.  HDD 전체 생애에 대해 각 행을 독립적인 분류 샘플로 취급하여 이진분류함. 현재 row가 고장 임박 상태인지 예측. Precision, Recall, FAR을 계산함.  | 전체 생애 사용, 독립적인 행 단위 예측, 행 단위 평가 | Gmean 0.692, FDR 69.8%, FAR 31.2% (WCE-TCNN-MP) | 고정된 15일 Horizon 기반 Row 분류 평가. 운영 절차나 최초 알람 기반 평가는 수행하지 않음. | https://link.springer.com/article/10.1007/s00521-024-10479-6?utm_source |
    | Machine Learning Model Update Strategies for Hard Disk Drive Failure Prediction (ICMLA 2021) | 모델 노후화(Model Aging)와 Concept Drift를 고려한 HDD 고장 예측 모델의 런타임 업데이트 전략 비교 및 평가 | Backblaze ST4000DM000 (2017.02~2020.12), 35,170 HDD, 고장 2,246대 | XGBoost, Random Forest, Logistic Regression, SVM | 선별된 SMART 속성(Xiao et al. 기준)을 Min-Max 정규화하여 일 단위 SMART 관측치를 입력 | 고장 전 10일 row를 Positive로 레이블링함. 시간 순서를 보존한 채 분할. 
    HDD 전체 생애에 대해 각 행을 독립적인 분류 샘플로 취급하여 이진분류함. 오늘의 SMART 정보를 보고 이 HDD가 앞으로 10일 이내에 고장날 것인가? 학습 후 Month1 예측 (학습 업데이트 판단) → Month2 예측 → … 이런 식으로 월 단위 온라인 평가를 진행함. 
    그러나 성능 평가는 행 단위로 진행했다는 점. FDR(Recall), FAR 등 계산 | 시간축 분할, 독립적인 행 단위 예측, 행 단위 평가 | Hoeffding Bound 기반 XGBoost가 ζ-value 64.86%, 13회 업데이트로 가장 우수한 Prediction Quality–Update Cost Trade-off를 달성 | 시간 순 데이터에서 모델 갱신 전략을 비교하지만, 운영 기간 동안의 최초 탐지·반복 알람·Disk 운영 성능 평가는 수행하지 않음. | https://ieeexplore-ieee-org-ssl.openlink.mju.ac.kr/stamp/stamp.jsp?tp=&arnumber=9680247 |
    | Robust Data Preprocessing for Machine-Learning-Based Disk Failure Prediction in Cloud Production Environments (2019) | 실제 클라우드 환경에서 활용 가능한 HDD 고장 예측을 위한 데이터 전처리 파이프라인(RODMAN) 제안 | Alibaba Cloud(300만+ HDD), Backblaze | LightGBM | SMART 원본 + 차분 + 7·14일 통계 특징 | 계산에 의해 사전 고장 기간을 계산하여 각각 레이블링. 시간 순서를 보존한 채 분할. 각 row를 독립적인 샘플로 예측. 이 row가 고장 직전 상태인가? 정상 상태인가(현재 row의 상태), TPR, FPR 계산 | 독립적인 행 단위 예측, 행 단위 평가 | Alibaba: FPR 0.1%에서 TPR(FDR) 92.8%, Backblaze: FPR 4.0%에서 TPR(FDR) 82.4% | Disk 단위 성능은 보고하지만 월 단위 오프라인 평가이며 시간 순 온라인 추론은 수행하지 않음. | https://arxiv.org/pdf/1912.09722 |
    | Hard Disk Failure Prediction Based on Lightgbm with CID (2021) | 기존 기계학습 기반 하드디스크 고장 예측 모델들이 시계열 데이터의 변화 과정을 반영하지 못해 높은 오경보율(False Alarm Rate)을 기록하는 한계를 극복하기 위해, 시계열 복잡도를 반영하는 통계적 특징(CID)과 LightGBM을 결합한 새로운 고장 예측 접근법 제안 | Backblaze 2019년 데이터 기반 2개 제조사(Seagate, HGST) 소속 총 77,615대의 하드디스크 운영 데이터 (정상 대비 고장 비율 약 1:47의 불균형 데이터) | LightGBM (비교군: Random Forest, Regularized Greedy Forest, GBDT, Logistic Regression) | SMART 속성 데이터의 결측치를 평균값으로 채우고 Min-Max 정규화를 수행한 뒤, 슬라이딩 윈도우를 활용하여 시계열 데이터의 변동 추세를 나타내는 이동평균(MA) 및 **복잡도 불변 거리(CID; Complexity Invariant Distance)** 통계적 특징을 추가하여 입력 | HDD를 10일 시간창 단위로 분할하여 생애 전체를 덮음. 각 row를 독립적으로 예측. 향후 10일 내 고장날 것인가? 시간창 내 row에서 하나라도 고장이라고 예측하면 HDD 전체를 Failure. 이 결과를 실제 레이블과 비교하여 평가. AUC, F1, Precision, Recall, TPR, FPR. 추가적으로 얼마나 일찍 고장을 예측했는지도 평가함.
    사실상 윈도우를 안쓰고 추론하는 것과 같음. 고장이 관측된 HDD에서는 리드타임이 1000일이더라도 분류 성공으로 집계됨.  (의아한 부분) | 독립적인 행 단위 예측, 행 단위 평가 | CID 피처를 추가함으로써 TPR(진짜 양성률)이 기존 0.28에서 **0.96**으로 크게 향상되었으며, 사전 예측 가능 기간(DPF)이 평균 1.2일 연장됨. AUC score 0.9998, F1-score 0.8768을 기록하여 비교 대상 모델들 대비 압도적인 성능 우수성을 입증함 | Window 기반 예측을 Map Function으로 Disk 결과에 집계하며, HDD 생애 전체에 대한 연속 온라인 추론 및 최초 탐지 기반 운영 평가는 수행하지 않음. DPF는 Window 내 최초 탐지 시점을 이용한 평균 조기 예측 일수임. | https://ieeexplore-ieee-org-ssl.openlink.mju.ac.kr/stamp/stamp.jsp?tp=&arnumber=9631504 |
    | DFPoLD: A Hard Disk Failure Prediction on Low-Quality Datasets (2025) | 데이터 수집 및 전송 과정에서 대량의 결측이 발생하는 실제 산업 현장의 한계를 극복하기 위해, 10~99%의 데이터가 소실된 저품질 데이터셋 환경에서도 높은 정확도를 유지하는 하드디스크 고장 예측 프레임워크(DFPoLD) 제안 | Backblaze 2022-2023 데이터(Seagate ST4000DM000)에서 인위적으로 10~99%의 데이터를 무작위 삭제하여 구축한 저품질 데이터셋 'Backblaze-' 및 실측 검증용 Nankai-Baidu 데이터셋 | LightGBM | SMART 원본 + ASFD + CID 시계열 특징 | 고장 HDD는 마지막 y일을 가져와 실패 레이블링, 정상 HDD는 초기 y일을 가져와 정상 레이블링. 이것을 섞어서 데이터를 구성. 평가도 같은 데이터셋으로 진행. TPR, FPR, AUC, F1 계산. 추가적으로 전체 생애에 대해 리드타임을 계산함.(사후분석이라는 한계 존재) | 이상함 | TPR 99.46%, FPR <0.04%, AUC 0.9971, F1 0.9871, 평균 DPF 9.75일 (80% 데이터 손실) | DPF를 통해 조기 예측 일수는 평가하지만 최초 탐지 기반 운영 절차는 고려하지 않음. | https://www.mdpi.com/2227-9709/12/3/73?utm_source |
    | Explainable time series features for hard disk drive failure prediction (2025) |  설명 가능한 HDD 고장 예측 프레임워크 제안 | Backblaze 2022 (Seagate ST4000DM000) | XGBoost | 마지막 32일 시계열 특징 추출 | Q1~Q3 2022”
    
    슬라이딩 윈도우 방식으로 Signal Length(모델 입력에 쓰는 데이터 구간 길이)와 Lead Time(입력 구간 끝~ 실제 고장 시점 사이의 버퍼)라는 두 파라미터로 시계열을 잘라 라벨링함.
    이진분류(고장/정상)을 목적으로함.
     Lead Time=5일로 최종 채택. signal length는 32일인데 이는 몇 일치 데이터를 볼 것인지에 관한 것 
    디스크 전체  생애가 아니라 디스크당 1개의 세크먼트를 독립 샘플로 취급
    평가는 디스크 단위 
    평가지표는 FDR, FAR | 디스크 당 한번 예측, 디스크 단위 평가 | FDR 74.70%, FAR 0.73% (XGBoost+Feature Extraction) | HDD당 마지막 32일만 입력하여 단일 추론 수행. HDD 전체 생애에 대한 연속 추론은 수행하지 않음. | https://pdf.sciencedirectassets.com/271095/1-s2.0-S0952197625X00117/1-s2.0-S0952197625006748/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEPv%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJIMEYCIQCJgh54BzqL7GuagNEveo2Zn2YbCMR7ZzQTD2Pxp3tkQwIhAOPzNaO%2FskhfaEYv%2FH5iLo%2Fl6wfR3z1kTZZsNiQPppnUKrsFCMT%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEQBRoMMDU5MDAzNTQ2ODY1Igzv6tsiMO%2B9Xi97SpUqjwVNZ41llpTpTpY86XvdzPg55sayyG5DtWugiMIWzCGDcXc31A6TBVDQJmOPc8JvMu9gsX5OOmxnhfzpTvK7d9gFi1%2BpWYRM5HO0hNb%2Bl5Voi7SRKqzCMswrswx6YORxG8JiJ5uq9H0vHexzLhgwibXJwEF42jERQhEm2X6MjDCWh88HYJqOIshl%2BMoGZS5YEVT2fSRu3jsvejvNEgxEajrqBlABN1m8hVO2MT2Fec34wzKTwC9ue6WK8z6iwLbQdyaY7brU%2BCw2dtleC0JK3YIcP76ycvBaJQlYtajfJmuq7iL93R2j%2B23Oc83BVSdZ0CoWWYY41FwGbBiDa1%2Bg7SpwN6JPIhviNan6%2BPAr97wI3cbhCXWAt3E2x2e8JEUeWFSV64Vufp6ec9ReBvRshlHspkcpAQJmb%2B8iWZty3YjtOrRBDyO0EwvthAizhjFkdO7%2FHwnddZr6aaX0iO1NO8e0xE5IBfS30c78b9KDcVOnGWaZ3emqklZgf9UVw5r6G48ziVpSaBnTWUBXHRv28LlbZytxSBw%2Bg2SrxWN0y0MCamYlafqMPzVLBsubFqNneO6kEVn5jjyNYb6oquQnesuGdp%2Fp2%2BprYkdWlSbMqyEAR3vrR5QWWMZPGPRkewXZjqpimm11uHBsBY4QTl7ByPSW%2Bm4ElO%2FEk1Y9hCHSq5otusXJ6gJugd77QQaqojorTpNZdtzKTQU2KSx8LWf7AWkbx2D7wp3001QVtf%2F5j4tWMROR3KyDbNQM1LKxMPa77zOcT0JsfVEFsiZ%2FomGFkL29LjuB01ySobyOxzrs78SchethlOFO4BcTBu8pube%2Flm%2FI1BvYFEQI6tum99y98iX47R5GbMMZl900z1qtb2cRMLKXt9MGOrABBZW5Z6Sh2KEP17yiSgOQ6AA6ET9sRfK1UF4Jk%2FpWZ13PJiV0PiuZLb%2FeeNc511BFy5ViA0AYFg0QFUEl%2Bqel86LgVjYUeqhCb3eTQ41QAZw1BTcaH%2FpMkzCCfFO8D2kuCr6m7789aaPqQvLZmf4%2B5PFxUJEAjRYMWQ6%2Bo9n0N7cPww2aZ9yhmRZOziZ9dfDW9FHdmvjuwy19tlEGxrFjsf6PZ2Ok0yXHe7ehXQavf7w%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20260801T110448Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYYCP7MMAB%2F20260801%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=ee31b4e78d82b054c0cb93e2e3f41637d7e010f9cc25b8adabc05d4ea7bf28ef&hash=137d8fc8ea672da6e8472c23408b5e3956cb92c92bcc894777fdfcbc0dd36408&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S0952197625006748&tid=spdf-95210ca9-4694-47d1-baf2-b35010e28b92&sid=c83de2f23354814dd54b87474eb7826a4d3agxrqa&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&rh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=0d12045555555003505a&rr=a2445d34cb7520d9&cc=kr |
    | DISK FAILURE PREDICTION BASED ON MULTILAYER DOMAIN ADAPTIVE LEARNING (2023) | 고장 데이터가 부족한 HDD 모델의 예측 성능 향상을 위한 다층 도메인 적응(MDA) 기반 HDD 고장 예측 프레임워크 제안 | Backblaze 2021 (다양한 Seagate 모델, 정상:고장 = 10:1) | Multi-layer Domain Adaptation (MDA) Network | 9개 핵심 SMART 속성(11차원) + Min-Max 정규화([-1,1]) | 시간 개념 자체가 이 논문의 입력 구조에 없움. 시퀀스 자체를 사용x 
    Positive의 정의가 기재되어 있지 않음.  특정시점의 hdd상태(고장/정상)자체를 분류하는 정적 문제, 독립적인 행(샘플)단위이며 시계열 예측 문제는 아님
    target 도메인(고장 데이터가 적은 모델)마다 여러 source도메인(고장 데이터가 많은 모델)을 바꿔가며 도메인 적응 성능을 비교함. 다른 디스크 모델의 지식을 빌려와서 학습한 것과 그렇지 않은 것을 비교
    평가지표는 G-mean 하나만 사용. | 독립적인 행 단위 예측, 행 단위 평가 | G-mean 0.89~0.92 (Double-layer Coral+MMD) | 도메인 적응을 통한 모델 일반화가 목적이며 운영 평가 절차는 다루지 않음. | https://arxiv.org/pdf/2310.06534 |
    | StreamDFP: A General Stream Mining Framework for Adaptive Disk Failure Prediction (IEEE TC 2023) | 오프라인 방식의 기존 디스크 고장 예측 연구들이 학습 데이터가 사전에 모두 확보되어 있다고 가정하는 한계를 지적하고, SMART 로그를 지속적으로 유입되는 스트림으로 간주하여 온라인으로 학습·예측하면서 **concept drift(시간에 따른 통계적 패턴 변화)**에 적응하는 범용 스트림 마이닝 프레임워크(STREAMDFP) 제안 | Backblaze 공개 HDD SMART 데이터 12종(D1~D4, D6~D12, 서로 다른 제조사·모델) + Alibaba 비공개 HDD 데이터(D5) + Alibaba SSD 데이터 3종(D13~D15), 총 15개 데이터셋. 평가 기간은 각 데이터셋에서 동일하게 460일(D5는 150일)을 잘라 사용 | Hoeffding Tree(HT) / Hoeffding Adaptive Tree(HAT), Oza's Bagging(Bag) / Bagging with ADWIN(BA), Oza's Boosting(Boost) / BOLE, Online Random Forest(RF) / Adaptive Random Forest(ARF), FIMT-DD(회귀 트리), MLP(역전파). 즉 concept-drift 적응 유무가 짝을 이루는 여러 증분학습(incremental learning) 알고리즘을 동일 프레임워크 위에서 비교 | 디스크 i의 SMART 속성(원시값+정규화값, 최대 29종)을 하루 단위 벡터 x_t로 구성. **(disk, day) 페어를 개별 샘플로 취급**하는 온라인 스트림 — 30일치를 하나의 시퀀스로 묶어 넣는 시계열 입력이 아니라, 매일 도착하는 단일 시점 벡터를 슬라이딩 윈도우(30일)에 버퍼링하며 순차 처리 | 고장 전 20일(기본값 DL) row를 Positive로 온라인 레이블링함. 시간 순서를 보존한 채 스트림으로 처리(사전 분할이 아니라 매일 도착하는 데이터를 그대로 흘려보냄).
    HDD 전체 생애가 아니라, 하루 단위 (disk, day) 행을 독립적인 스트림 샘플로 취급하여 이진분류함. 오늘의 SMART 정보를 보고 이 HDD가 향후 30일 이내에 고장날 것인가? 를 예측. + 잔여수명 예측 지원
    학습 후 Day1 예측(데이터로 모델 즉시 업데이트 후 예측) → Day2 예측 → … 이런 식으로 일 단위 온라인 학습·평가를 400일간 반복함(30일 워밍업 이후부터 시작).
    row 단위 Precision, Recall, F1-score(분류), ARE(회귀, 잔여수명 오차)와 디스크 단위 Days in advance 등 계산 | 실시간 운영 환경 모사.
    독립적인 행 단위 예측, 행 단위 평가 | F1-score 26.8~53.2% 향상, 37,000대 HDD 일일 데이터 13.5초 처리 | 온라인 학습(Online Learning)을 연구하지만 운영 성능 평가 절차를 제안하는 연구는 아님.
    온라인 추론, 온라인 학습 둘다 함  | https://ieeexplore-ieee-org-ssl.openlink.mju.ac.kr/stamp/stamp.jsp?tp=&arnumber=9737380 |
    
- 예지보전, 의료 AI 분야 연구 정리
    
    
    | 논문 | 연구 목적 | 요약 | 비고 | 링크 |
    | --- | --- | --- | --- | --- |
    | Decision Making in Predictive Maintenance: Literature Review and Research Agenda for Industry 4.0 (2019) | 인더스트리 4.0 및 스마트 제조 환경에서 센서 기반의 실시간 예측을 활용해 장비 고장을 방지하는 **동적 의사결정(Dynamic Decision Making)** 관련 기존 문헌을 체계적으로 고찰하고, 주요 연구 공백을 식별하여 향후 연구 방향 제시 | 2013년~2018년 출판된 문헌을 체계적 문헌 고찰(SLR) 방식으로 분석하여 정비 의사결정 영역을 5가지(정비 계획 및 스케줄링, 신뢰성/상태 저하 기반, 공동 최적화, 다중 상태/부품 시스템 최적화, 정비 비용/리스크 최적화)로 분류함. 기존 연구들이 대부분 일괄 처리 방식의 정적 모델이나 단편적 정비 비용 절감에 편중되어 있음을 지적하고, 향후 연구 과제로 **① 스트리밍 기반 실시간 의사결정 알고리즘 개발, ② 범용 규범적 의사결정 모델 구축, ③ 데이터 기반 자동화된 모델 생성, ④ 인간 피드백 루프 구축**의 필요성 역설 | 예지보전 분야의 의사결정 프로세스와 인더스트리 4.0 트렌드를 종합적으로 조망하는 핵심 서베이 논문 | https://www.researchgate.net/profile/Alexandros-Bousdekis/publication/338171890_Decision_Making_in_Predictive_Maintenance_Literature_Review_and_Research_Agenda_for_Industry_40/links/60336aff299bf1cc26e08b6d/Decision-Making-in-Predictive-Maintenance-Literature-Review-and-Research-Agenda-for-Industry-40.pdf |
    | A Survey of Predictive Maintenance: Systems, Purposes and Approaches (2019) | 예지보전 시스템의 구조, 최적화 목적, ML·DL 기반 접근을 종합적으로 검토 | 예지보전을 상태 모니터링, 고장 진단·예지, 유지보수 계획으로 이어지는 통합 시스템으로 정리한다. 주요 운영 목표를 **유지보수 비용 최소화**, **다운타임 감소**, **가용성·신뢰성 최대화**, 그리고 이들 간의 다목적 최적화로 제시한다. 온라인 설비 건강상태를 이용해 고장 후 수리와 불필요한 예방정비 사이의 균형을 맞추는 것이 PdM의 핵심이라고 설명한다. | 정상 Disk 경고는 불필요한 정비 부담으로, 고장 전 경고 시점은 사전 대응 가능성으로 이어진다는 근거가 된다. HDD_level Precision/FAR/Lead Time을 함께 제시할 논리적 배경으로 적합하다. 직접적인 Disk 평가 실증은 아니다. | https://arxiv.org/pdf/1912.07383v2 |
    | Comprehensive Evaluations of Condition Monitoring-Based Technologies in Industrial Maintenance: A Systematic Review (2025) | 상태감시·고장탐지·예측 기술을 실제 산업 정비에 적용할 때, 어떤 방식으로 공학적·재무적 효과를 평가했는지 체계적으로 검토 | 2001~2023년 논문 465편 중 조건을 충족한 42편을 분석했다. 단순 모델 정확도뿐 아니라 산업 적용, 정비 배치 방식, 평가기법, 성능지표, 경제성 분석을 함께 보며, 기존 연구의 평가가 서로 비교하기 어려울 정도로 이질적이라고 지적한다. 분석적 모델·시뮬레이션·민감도 분석의 결합을 권고한다. | 행 단위 정확도만으로는 정비기술의 가치를 충분히 평가하기 어렵고, 정비 적용·성과지표까지 연결해야 한다”는 당위성의 최상위 근거로 좋음. | https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=959739 |
    | Fire now, fire later: alarm-based systems for prescriptive process monitoring (*Knowledge and Information Systems*, 2022 / Fahrenkrog-Petersen et al.) | 예측 결과를 제공하는 기존 Predictive Process Monitoring을 넘어, 개입 비용과 효과를 고려하여 언제 알람을 발생시키는 것이 가장 경제적인지를 결정하는 Prescriptive Process Monitoring 프레임워크를 제안 | 개입 비용(Intervention Cost), 오탐 보상 비용(Compensation Cost), 미개입으로 인한 손실(Outcome Cost), 개입 효과(Mitigation Effectiveness)를 통합한 비용 모델을 제안하였다. 이를 기반으로 경험적 임계값(Empirical Thresholding) 을 이용해 알람 발생 시점을 최적화하고, 고정 임계값뿐 아니라 지연 발동(Delayed Firing), 구간별 임계값(Prefix-length-dependent Threshold), 다중 알람(Multi-alarm) 전략을 비교하였다. 실제 이벤트 로그 실험 결과, 단순히 높은 예측 확률에서 즉시 알람을 발생시키는 것보다 상황에 따라 알람 시점을 조정하는 것이 전체 운영 비용(Net Cost)을 효과적으로 감소시킴을 보였다. | 알람 발생 시점을 비용 관점에서 최적화한 대표 연구로, 예측 성능뿐 아니라 운영 환경에서 언제 알람을 발생시켜야 하는지에 대한 평가 필요성을 제시한 핵심 문헌 | https://link.springer.com/article/10.1007/s10115-021-01633-w |
    | Approaches and Applications of Early Classification of Time Series: A Review (Gupta et al., 2020) | 의료 진단, 산업 모니터링 등 시간 민감형(Time-critical) 응용에서 예측 정확도를 유지하면서 가능한 한 빠르게 의사결정을 수행하기 위한 시계열 조기 분류(Early Time Series Classification) 연구들을 체계적으로 분석하고 향후 연구 방향을 제시 | 의료 진단, 산업 프로세스 모니터링, 인간 활동 인식, 지능형 교통 등 다양한 응용 분야의 Early Classification 연구를 조사하였다. 기존 기법을 Prefix 기반, Shapelet 기반, Model 기반, 기타(Deep Learning·Reinforcement Learning 포함) 의 네 가지 범주로 분류하고, Accuracy–Earliness Trade-off, 예측 신뢰성(Reliability), 해석가능성(Interpretability), 다변량 시계열 상관관계 활용 등을 핵심 연구 과제로 정리하였다. 또한 산업 모니터링 분야에서는 조기 예측이 유지보수 비용 절감과 운영 의사결정 지원에 중요함을 강조하였다.` | 조기 예측 시점(Earliness)과 예측 정확도 간의 Trade-off를 체계적으로 정리한 대표적인 서베이 논문으로, 운영 환경에서 '언제 알람을 발생시킬 것인가'에 대한 평가 필요성을 뒷받침하는 근거로 활용 가능 | https://arxiv.org/pdf/2005.02595 |
    | The Impact of Evaluation Strategy on Sepsis Prediction Model Performance Metrics in Intensive Care Data: Retrospective Cohort Study (*Journal of Medical Internet Research*, 2026 / Do et al.) | 중환자실(ICU) 패혈증 조기 예측 머신러닝 모델의 성능 평가 시 활용되는 **평가 전략(고정 예측 시점, 피크 점수, 연속 평가) 및 대조군 매칭 방식**이 성능 측정 지표(AUROC 등)에 미치는 편향과 왜곡 영향을 규명하고, 실제 임상 현장의 연속적 모니터링 환경에 가장 부합하는 평가 방법론 제시 | MIMIC-IV 데이터셋으로 사전 학습된 TCN(시간 합성곱 네트워크) 모델을 독일 BerlinICU 데이터셋(40,132건)에 외부 검증한 결과, 동일한 모델과 데이터셋을 사용함에도 평가 전략에 따라 AUROC가 0.61(고정 시점)에서 0.67(연속 평가)까지 크게 변동함을 입증함. 피크 점수(Peak Score) 평가 방식은 재원 기간이 긴 대조군 환자에서 우연히 높은 예측 점수가 포착되어 성능이 저하되는 왜곡이 발생하므로, 대조군과 패혈증 환자 간 재원 기간 분포를 맞추는 발병 시점 매칭(Onset Matching)이 필수적임을 밝힘. 정 시점 및 피크 점수 방식과 달리, 재원 기간 전체의 시계열 데이터를 고려하는 **연속 평가(Continuous Evaluation)** 방식이 실제 ICU에서 환자를 지속 모니터링하는 임상 현장을 가장 잘 반영하는 평가 전략임을 역설함. | 예지보전 및 시계열 조기 예측 분야에서 모델 자체의 성능뿐만 아니라 **평가 프레임워크(Evaluation Framework)의 설계 방식이 성능 지표 착시를 유발할 수 있음**을 임상 빅데이터를 통해 증명한 핵심 방법론 논문 | https://www.jmir.org/2026/1/e72083/PDF |
    | Continuous Evaluation Frameworks for Retrospective Evaluation of Clinical Machine Learning Models (Critical Care, 2026 / Natarajan et al., Philips North America) | 병상에서 환자 상태를 지속적으로 모니터링하는 머신러닝(ML) 모델의 성능을 단일 시점 지표(AUROC, AUPRC 등)로만 평가할 때 발생하는 한계(알람 피로, 임상적 유용성 미반영 등)를 극복하기 위해, 회고적 데이터셋에서 연속 실행 모델의 성능과 임상적 유용성을 다각도로 검증할 수 있는 3가지 새로운 연속 평가 프레임워크(Continuous Evaluation Frameworks, CEFs)를 제안함. | 환자 재원 기간을 이벤트 발생 전 후의 임상적 중요도에 따라 여러 시간 영역(Zone A, B, C)으로 나누고, 영역별 예측 정답률(TP, TN, FP, FN)에 가중치를 부여하여 단일 평균 연속 점수(Mean Continuous Score, MCS)를 산출함. 경보 발생 시점과 이벤트 발생 사이의 선제 시간(Anticipation Time)과 오경보율(False Alarm Rate) 간의 상충관계를 시각화하여, 임상적으로 유효한 리드 타임을 보장하는 리스크 임계값을 선택하도록 지원함. 연속적으로 발생하는 중복 고위험 예측을 하나의 단일 알림(Notification)으로 통합·절단한 후, 연속 시간 지표(C-Recall, C-Precision, C-Specificity, C-NPV)를 계산하여 실제 알림 환경에서의 유용성을 평가함. 공개 데이터셋(Emory 및 MIMIC-IV)의 패혈증 예측 모델(LGBM, SIRS)을 대상으로 실험한 결과, 단일 시점 AUROC가 우수하게 측정된 모델이라도 연속 평가 시 대량의 허위 알림(낮은 C-Precision, 약 4~7%)이 발생하거나 초기/중기 발병 환자군에서 예측 성능이 저하되는 등의 실제 운영상 문제점을 명확히 포착해냄. | 단일 시점 정확도 중심의 ML 평가에서 벗어나, **임상 현장에서의 지속적 실행 형태 및 알람 부담(Alarm Burden)을 정량화**할 수 있도록 구현된 의료 AI 분야의 최신 후속 평가 프레임워크 논문 | https://link.springer.com/article/10.1186/s13054-025-05725-9?utm_source |
    | Patient Versus Prediction-Level Evaluation of a Dynamic Clinical Prediction Model of Sepsis (2026) | 동적 시계열 예측 모델에서 Patient-level 평가와 Prediction-level 평가의 차이를 비교하고, 평가 방식에 따라 성능이 어떻게 달라지는지 분석 | 패혈증 위험을 지속적으로 예측하는 모델(ESMv2)에 대해 **Patient-level**(환자당 최대 예측값만 사용)과 **Prediction-level**(모든 시점의 예측을 평가)을 각각 적용하였다. Patient-level에서는 AUC 0.86, PPV 14.5%로 우수한 성능을 보였으나, Prediction-level에서는 AUC 0.62, PPV 4%로 크게 감소하였다. 저자들은 **Patient-level 평가는 동적 예측 모델의 성능을 과대평가할 수 있으며**, 실제 운영 환경에서는 Prediction-level 평가를 함께 수행해야 한다고 주장하였다. | **Patient-level = 개체 단위 평가**, **Prediction-level = 시점(Row) 단위 평가**를 직접 비교한 논문. HDD의 **HDD_level vs Row-level 평가**와 매우 유사한 문제의식을 제시함. 다만 Patient-level은 환자당 최대 위험도를 사용하는 방식이며, 알람 정책을 적용한 운영 평가까지는 수행하지 않음. | https://assets-eu.researchsquare.com/files/rs-9726164/v1/525245fd-bf5d-4696-9f42-f68dc3147c04.pdf |

### 2.1 HDD SMART 기반 고장 예측 연구

- HDD 고장 예측 연구는 대규모 운영 데이터에서 고장 특성과 SMART 속성의 관계를 분석하는 통계적·신뢰성 분석 연구에서 시작되었다.
- 이후 SMART 데이터를 활용하여 고장 여부를 직접 예측하는 머신러닝 기반 이진 분류 연구가 등장하였으며, Random Forest, RGF, SVM, XGBoost, LightGBM 등 다양한 모델이 적용되었다.
- 이후 시계열 정보의 활용이 확대되면서 Sliding Window, LSTM, CNN-LSTM, Transformer 등 시간적 특성을 반영하는 모델이 제안되었고, 고장 전 일정 기간을 예측 대상으로 설정하는 예측 기간 기반 연구가 이루어졌다.
- 최근에는 클래스 불균형, Concept Drift, 온라인 학습, 도메인 적응, 데이터 품질 등 실제 HDD 데이터가 갖는 문제를 고려하는 연구로 확장되고 있다.
- 이와 함께 연구 목적도 단순한 고장 여부 분류에서 잔여수명(RUL) 예측 및 조기 고장 탐지 등으로 다양화되고 있다.
- 이러한 연구들은 서로 다른 데이터셋과 모델 구조를 사용하고 있으며, 이에 따라 예측 결과를 평가하는 단위와 절차 또한 다양하게 나타난다.
- 선행 연구에서는 Row 또는 Window 단위의 예측 결과를 대상으로 Precision, Recall, F1-score, AUC, FAR 등의 분류 성능을 평가하거나, Lead Time, Days in Advance, DPF 등을 통해 조기 탐지 성능을 추가적으로 평가하고 있다.
- 따라서 HDD 고장 예측 연구의 발전은 모델의 복잡화뿐만 아니라, 예측 결과를 어떤 단위와 기준으로 평가할 것인가의 문제로도 이어진다.

### 2.2 HDD 고장 예측의 평가와 운영 관점 연구

- HDD 고장 예측 연구에서는 개별 Row, Window, HDD 등 서로 다른 평가 단위가 사용되며, Prediction Horizon의 설정과 평가 절차 또한 연구마다 상이하다.
- 일부 연구에서는 FAR, Lead Time, Days in Advance 등 운영과 관련된 지표를 활용하고 있으며, 시간 순으로 데이터를 처리하거나 HDD 전체 생애의 데이터를 활용하는 연구도 존재한다.
- 그러나 운영 관련 지표를 사용한다는 사실과 운영 단위에서 모델을 평가한다는 것은 동일하지 않다.
- 예를 들어 각 시점의 예측을 독립적인 분류 대상으로 취급하여 FAR이나 Recall 등을 계산하면, 동일 HDD에서 수십 번 반복된 Positive 예측도 개별 예측 결과로 분산되어 평가된다.
- 실제 운영에서는 이러한 여러 예측이 독립적인 사건이라기보다 하나의 HDD에 대해 지속적으로 발생하는 하나의 경보 과정으로 나타날 수 있다.
- 따라서 Row-level 평가에서 높은 성능을 보인 모델이 실제 HDD 운영 환경에서도 동일한 수준의 성능을 보인다고 단정하기 어렵다.
- 특히 운영 관점에서는 단순히 한 번이라도 고장을 맞혔는지가 아니라, 최초 Alarm이 언제 발생했는지, 실제 고장까지 어느 정도의 대응 시간이 확보되었는지, 정상 HDD에서 불필요한 Alarm이 얼마나 발생했는지가 중요하다.
- 이러한 관점에서 HDD의 시간적 운영 과정을 하나의 평가 단위로 연결하여 성능을 평가할 필요가 있다.

### 2.3 타 분야의 평가 방법론

- 예지보전 분야에서는 예측 결과 자체의 정확성뿐만 아니라 유지보수 비용, 의사결정 시점, Alarm 발생 시점 등을 고려하여 예측 결과의 실제 활용 가능성을 평가하려는 연구가 이루어지고 있다.
- 시계열 기반 의료 AI 분야에서도 지속적으로 예측이 수행되는 환경에서 Prediction-level과 Patient-level 평가 결과가 서로 다를 수 있음이 보고되고 있으며, 연속적인 예측 과정에서 발생하는 중복 Alarm과 실제 사건의 관계를 고려한 평가 방법이 제안되고 있다.
- 이러한 연구들은 개별 시점의 예측 성능과 실제 운영 단위에서의 성능을 구분할 필요성을 보여준다.
- 이는 HDD 고장 예측에서도 개별 Row의 분류 결과를 HDD의 시간적 운영 과정과 연결하여 평가할 필요가 있음을 시사한다.

**연구 공백**

- HDD 고장 예측 분야에서는 모델 구조와 온라인 학습 기법이 지속적으로 발전하였으며, 일부 연구에서는 FAR, Lead Time 등 운영 관련 지표도 활용하고 있다.
- 그러나 HDD 전체 생애를 대상으로 시간 순의 온라인 추론을 수행하고, 최초 알람을 기준으로 Disk 단위의 운영 성능과 운영 특성을 함께 평가하는 성능평가 절차는 충분히 연구되지 않았다.
- 따라서 본 연구에서는 HDD의 시간적 운영 과정을 반영한 운영 환경 기반 성능평가 절차를 제안하고, 기존 Row 기반 평가와 비교하여 평가 기준에 따른 성능 차이와 운영 특성의 차이를 분석한다.

---

## 3. 운영 환경 기반 평가 방법

<aside>

*3장. 
본 연구에서는 HDD별로 시간 순서에 따라 온라인 추론을 수행하고, 예측값이 임계값을 초과한 시점을 Alarm으로 변환한다. 이후 동일 HDD에서 발생한 반복적인 Alarm을 하나의 운영 사건으로 통합하고, 최초 Alarm과 실제 고장 시점의 관계를 기준으로 운영 환경 기반 평가 성능과 Lead Time을 평가한다. 이를 통해 개별 Row의 예측 정확성을 넘어, 하나의 HDD가 운영 과정에서 실제로 경보 대상이 되었는지와 해당 경보가 고장 이전에 얼마나 일찍 발생했는지를 평가한다. 그 결과를 전체 HDD 수준으로 집계하여 모델의 운영 환경 기반 성능을 평가한다.*

</aside>

### 3.1 운영 환경 기반 성능평가 개요

<aside>

> 운영 환경 기반 평가는 Row-level 예측 결과를 단순 집계하는 것이 아니라, 실제 HDD 운영에서 발생하는 시간적 예측과 Alarm 의사결정 과정을 평가 단위에 반영한다.
> 
- 기존 HDD 고장 예측에서는 개별 Row 또는 특정 시점의 예측 결과를 기준으로 성능을 평가해 왔다는 점을 간단히 설명한다.
- 그러나 실제 HDD 운영에서는 하나의 HDD에 대해 시간이 흐르면서 여러 관측값이 발생하고, 모델도 여러 시점에서 반복적으로 예측하게 된다.
- 따라서 동일 HDD에서 발생하는 여러 예측을 독립적인 사건으로 취급하는 것과 실제 운영에서 HDD를 대상으로 경보를 발생시키는 것은 서로 다른 문제라는 점을 제시한다.
- 실제 운영에서는 모델의 개별 예측값 자체보다, 그것이 언제 Alarm으로 전환되었는지, 고장 전에 얼마나 일찍 발생했는지, 하나의 HDD에서 Alarm이 어떻게 반복되는지가 중요하다고 설명한다.
</aside>

**운영 환경의 기본 전제**

- HDD의 SMART 정보는 운영 기간 동안 시간 순으로 반복 수집된다.
- 실제 운영에서는 새로운 관측값이 수집될 때마다 HDD의 상태를 반복적으로 판단한다.
- 따라서 모델의 최종적인 유지보수 대상은 개별 Row가 아니라 HDD 개체이다.

**Row-level 평가와 운영 평가의 차이**

- Row-level 평가는 각 관측 시점의 예측을 독립적인 분류 결과로 취급한다.
- 그러나 실제 운영에서는 동일 HDD에서 여러 시점의 예측이 하나의 연속적인 경보 과정으로 나타난다.
- 따라서 개별 Row의 분류 성능이 높더라도 동일 HDD에서 반복적으로 발생하는 예측의 운영적 의미까지 직접 나타내지는 못한다.

**운영 단위에서 고려해야 할 요소**

- 실제 운영에서는 HDD에 대한 경보가 발생했는지뿐만 아니라 **최초 Alarm의 발생 시점**이 중요하다.
- 최초 Alarm과 실제 고장 시점 사이의 시간은 유지보수에 활용 가능한 사전 대응 시간을 나타내므로 **Lead Time**으로 평가할 필요가 있다.
- 또한 고장이 관측되지 않은 HDD에서 발생하는 Alarm은 불필요한 유지보수 부담으로 이어질 수 있으므로 HDD 단위의 오탐도 함께 고려할 필요가 있다.
- 따라서 반복적인 Row-level 예측을 HDD 단위의 Alarm 사건으로 연결하여 평가할 필요가 있다.

**본 연구의 평가 관점**

- 본 연구에서는 HDD별 관측값을 시간 순으로 따라가며 순차적으로 예측하고, 각 시점의 예측 결과를 Alarm으로 변환한다.
- 동일 HDD에서 발생한 여러 Alarm 중 **최초 Alarm만을 해당 HDD의 운영 사건으로 취급**한다.
- 최초 Alarm과 실제 고장 시점의 관계를 이용하여 HDD-level 탐지 성능과 Lead Time을 산출한다.
- 이를 통해 개별 예측의 분류 성능과 구분되는 **HDD 단위의 운영 성능**을 평가한다.

**운영 환경 모사의 근거**

- 온라인 추론
- 시점별 Alarm 발생
- HDD 단위의 최초 Alarm 판정
- 실제 고장과의 시간적 관계를 이용한 탐지 및 Lead Time 평가

→ **실제 운영에서 HDD를 반복적으로 상태 판단하고 경보하는 과정을 평가 절차에 반영한다.**

(전체 성능평가 절차 그림)

### 3.2 운영 환경 기반 평가 절차

- 선행연구
    
    
    | 선행연구 | 적용 맥락 | 방법론적 관련성 | 평가 방법 | 링크 |
    | --- | --- | --- | --- | --- |
    | Technical considerations for evaluating clinical prediction indices: a case study for predicting code blue events with MEWS (2021) | ICU 환자의 **Code Blue와 같은 급성 임상 악화 사건을 조기에 예측·경고하는 환자 모니터링 문제 | 반복적으로 발생하는 경보를 단순히 각 시점의 이진 예측으로 평가하지 않고, 실제 사건 발생 시점과 경보 발생 시점의 관계를 고려하여 사건 단위로 평가한다. 특히 예측 가능 시간 범위와 경보가 사건에 얼마나 앞서 발생했는지를 평가에 반영한다는 점에서 네 방법과 직접적으로 관련된다. | 각 경보를 사건 발생 시점과의 관계에 따라 Early, On-time, Late, Missed 등으로 구분하고, 이 분류를 바탕으로 민감도와 오경보율 등의 성능지표를 계산한다. 즉, 시계열상 반복적으로 발생하는 경보 → 사건과의 시간적 관계에 따른 경보 분류 → 분류 결과를 이용한 성능지표 계산이라는 평가 구조를 사용한다. | https://pmc.ncbi.nlm.nih.gov/articles/PMC8414372/pdf/pmea_42_5_055005.pdf |
    | Hypoglycemia Prediction with Subject-Specific Recursive Time-Series Models | 연속혈당측정 데이터를 이용한 저혈당 발생 조기예측 및 경보 | 실제 사건이 발생했다는 사실만으로 모든 사전 경보를 정답으로 인정하지 않고, 경보가 사건 발생에 비해 얼마나 이르게 발생했는지를 기준으로 정답 여부를 구분한다는 점에서 네 FP_early와 직접적으로 연결된다. | 저혈당 발생 45분 이내의 경보를 TP로 인정하고, 46분 이상 일찍 발생한 경보는 실제 저혈당이 발생하더라도 FP로 처리한다. 사건이 발생했으나 경보가 없으면 FN, 사건과 경보가 모두 없으면 TN으로 분류한다. 이후 이 분류 결과를 이용하여 Sensitivity, Specificity, False Alarm Rate, Time to Detection 등을 계산한다. 즉, 시계열 예측 → 사건과의 시간 관계에 따른 TP/FP/FN/TN 분류 → 분류 결과를 이용한 성능지표 계산의 구조다. | https://pmc.ncbi.nlm.nih.gov/articles/PMC2825621/pdf/dst-04-0025.pdf |
    | A framework to characterize the performance of early warning index alarm systems for patient monitoring | 환자 모니터링에서 임상적 이상 사건의 발생을 조기에 경고하는 경보 시스템 | 하나의 환자 기록에서 반복적으로 발생하는 경보를 독립적인 시점별 예측으로만 평가하지 않고, 사건 발생 여부와 경보의 시간적 위치를 함께 고려하여 경보의 결과를 분류한다는 점에서 네 평가 방법과 유사하다. | 경보를 False, Early, On-time, Late, Missed로 구분한다. 각 경보의 분류 결과를 이용하여 PPV, Sensitivity, FPR 등의 성능지표를 계산한다. 따라서 핵심적인 평가 구조는 시계열 경보 → 사건과의 시간적 관계에 따른 5가지 결과 분류 → 분류 결과를 이용한 성능지표 계산이다. | https://pmc.ncbi.nlm.nih.gov/articles/PMC6660561/pdf/main.pdf |
    | Moor et al. (2021), Early Prediction of Sepsis in the ICU Using Machine Learning: A Systematic Review | ICU 환자의 패혈증 발생을 조기에 예측하는 머신러닝 연구 | 시계열 예측에서는 동일한 사건을 대상으로 하더라도 어느 시점의 예측을 TP로 인정하고 어느 시점의 예측을 FP로 볼 것인지에 따라 성능평가 결과가 달라질 수 있음을 지적한다. 이는 Row 단위의 단순 이진분류만으로 미래 사건 예측의 성능을 평가하기 어려운 이유를 뒷받침한다. | 여러 연구에서 패혈증 발생 시점과 예측 시점의 관계에 따라 TP와 FP를 정의하는 방식이 서로 다름을 정리한다. 즉, 시계열 예측에서는 예측 결과 자체뿐 아니라 예측 시점과 실제 사건 발생 시점의 관계를 성능평가에 어떻게 반영할 것인지가 중요한 평가 설계 요소임을 보여준다. | https://www.frontiersin.org/journals/medicine/articles/10.3389/fmed.2021.607952/full?utm_source |
    | PhysioNet/Computing in Cardiology Challenge 2019 | ICU 환자의 패혈증 발생을 실시간으로 조기 예측하는 문제 | 실제 사건이 발생했는지 여부만으로 예측의 정답 여부를 결정하지 않고, 예측 시점과 사건 발생 시점의 관계에 따라 예측 결과의 평가값을 다르게 부여한다는 점에서 시계열 예측 평가의 직접적인 선례가 된다. | 패혈증 발생 전의 예측을 예측 시점에 따라 서로 다르게 평가하며, 너무 이른 예측이나 사건이 발생하지 않은 환자의 양성 예측에는 불이익을 부여한다. 다만 이를 네 방법처럼 5개 결과 범주로 분류하여 혼동행렬 기반 성능지표를 계산하는 방식과 동일하다고 볼 수는 없다. 따라서 이 연구는 FP_early의 직접적인 선례라기보다 예측 시점과 실제 사건 발생 시점의 관계를 평가에 반영해야 한다는 근거로 사용하는 것이 적절하다. | https://physionet.org/content/challenge-2019/1.0.0/?utm_source |
    | Scully & Daluwatte, “Evaluating performance of early warning indices to predict physiological instabilities,” Journal of Biomedical Informatics, 2017 |  |  |  |  |
    | A framework to characterize the performance of early warning index alarm systems for patient monitoring |  |  |  |  |
    |  |  |  |  | https://pmc.ncbi.nlm.nih.gov/articles/PMC6660561/pdf/main.pdf |

**평가 대상과 기본 단위**

- 평가 단위는 Row가 아닌 HDD이다.
- HDD를 d, 관측 시점을 t, HDD d의 관측 시점 집합을 Td로 정의한다.
- 각 HDD의 S.M.A.R.T 관측값을 시간 순으로 정렬하여 운영 이력으로 구성한다.
- 평가 대상 HDD는 **고장이 관측된 HDD**와 **우측 검열 HDD**로 구분한다.
- 두 집단의 차이는 HDD가 정상인지 여부가 아니라 **관측 기간 내 고장 시점이 확인되었는지 여부**에 있다.

**예측 기간 설정**

본 연구에서는 최초 Alarm 시점부터 실제 고장 발생 시점까지의 시간적 간격을 평가하기 위한 기준으로 Prediction Horizon H=30일을 설정한다. Lead Time이 H 이하인 최초 Alarm은 On-time, H를 초과하는 최초 Alarm은 Early로 분류한다.

$$
H = 30\ \mathrm{days}
$$

**Online Inference 및 Alarm 생성**

각 모델은 HDD의 시간 순서에 따라 관측 시점별 고장 예측 확률을 산출한다. 모델별 입력 구조와 내부 추론 방식은 각 모델의 학습 과정에서 정의되며, 본 평가에서는 모델이 산출한 예측 확률을 동일한 방식으로 Alarm 판정에 사용한다.

모델의 예측 확률 $p_{d, t}$가 의사결정 임곗값 $\tau$ 이상인 경우 Alarm을 발생시킨 것으로 정의한다.

$$
A_{d,t} =
\begin{cases}
1, & p_{d,t} \geq \tau,\\
0, & p_{d,t} < \tau.
\end{cases}
$$

**최초 Alarm 및 Lead Time**

동일 HDD에서 여러 시점에 Alarm이 발생할 수 있으므로, 성능평가에서는 동일 고장에 대한 반복 Alarm의 중복 집계를 방지하기 위해 **최초 Alarm**을 기준으로 HDD 단위 판정을 수행한다.

각 HDD에서 처음으로 $A_{d,t}=1$이 발생한 시점을 최초 Alarm 시점 $t_{alarm,d}$ 로 정의한다.

$$
t_{\mathrm{alarm},d}
=
\min\{t\in T_d:A_{d,t}=1\}
$$

해당 집합이 공집합인 경우 해당 HDD에는 최초 Alarm이 존재하지 않는 것으로 정의한다.

$$
LT_d
=
t_{\mathrm{failure},d}
-
t_{\mathrm{alarm},d}
$$

고장이 관측된 HDD에서 최초 Alarm이 존재하는 경우, Lead Time은 최초 Alarm 시점부터 실제 고장 발생 시점까지의 시간으로 정의한다.

**우측 검열 처리**

우측 검열 HDD는 관측 종료 시점 이후의 고장 여부와 고장 시점을 확인할 수 없으므로 Lead Time을 직접 계산할 수 없다. 이에 본 연구에서는 우측 검열 HDD의 관측 종료 시점으로부터 예측 기간 H에 해당하는 마지막 기간을 평가에서 제외한다.

$$
T_d^{\mathrm{eval}}
=
\{t\in T_d\mid t\leq t_{\mathrm{end},d}-H\}
$$

따라서 고장이 관측된 HDD에서는 $T_d$을 기준으로 최초 Alarm을 추출하고, 우측 검열 HDD에서는 $T_d^\text{eval}$을 기준으로 최초 Alarm을 추출한다.

우측 검열 HDD에서는 실제 고장 시점을 확인할 수 없으므로 정확한 Lead Time을 산출할 수 없다. 그러나 관측 종료 시점으로부터 H에 해당하는 마지막 기간을 평가에서 제외하였으므로, Tdeval 내에서 발생한 최초 Alarm은 관측 종료 시점을 기준으로 최소 H만큼 이른 시점에 발생한 Alarm으로 볼 수 있다. 따라서 해당 Alarm을 **Censored Early**로 분류한다.

최종 결과 분류

| HDD 상태 | 최초 Alarm | Lead Time | 판정 |
| --- | --- | --- | --- |
| 고장 관측 | 있음 | (LT_d\leq H) | **On-time** |
| 고장 관측 | 있음 | (LT_d>H) | **Early** |
| 고장 관측 | 없음 | — | **Missed** |
| 우측 검열 | 있음 | 미계산 | **Censored Early** |
| 우측 검열 | 없음 | — | **Censored No Alarm** |

Censored Early와 Censored No Alarm은 우측 검열 HDD를 관측 가능한 정보의 범위 내에서 판정할 수 있는 결과를 별도로 나타낸 것이다.

### 3.3 운영 성능평가 지표

3.2에서 정의한 HDD별 최초 Alarm의 판정 결과를 이용하여 모델의 운영 성능을 평가한다. 본 연구에서는 **On-time Alarm Proportion (OAP), On-time Detection Rate (ODR), Early Alarm Proportion (EAP)**을 주요 평가 지표로 사용하며, 실제 고장이 관측된 HDD에 대해서는 **Median Lead Time**을 추가로 산출한다.

각 지표는 서로 다른 운영 특성을 평가한다. OAP는 Alarm이 발생한 경우 해당 Alarm의 **시간적 적절성**을 평가하고, ODR은 실제 고장 중 **On-time으로 탐지된 비율**을 평가한다. EAP는 전체 운영 대상에서 **Early 성격의 Alarm이 발생하는 정도**를 평가하며, Median Lead Time은 실제 고장에 대한 최초 Alarm의 **시간적 여유**를 나타낸다.

**On-time Alarm Proportion (OAP)**

$$
\mathrm{OAP}
=
\frac{N_{\mathrm{On\text{-}time}}}
{N_{\mathrm{On\text{-}time}} + N_{\mathrm{Early}}}
$$

여기서 NOn-time과 NEarly는 각각 고장이 관측된 HDD 중 On-time 및 Early로 분류된 HDD의 수를 의미한다.

OAP는 **최초 Alarm이 발생한 HDD만을 대상으로 Alarm 시점의 적절성을 평가**한다. 따라서 Alarm이 전혀 발생하지 않은 Missed HDD는 분모에서 제외된다. 우측 검열 HDD 역시 실제 고장 시점을 확인할 수 없으므로 OAP 산출에서 제외한다.

**On-time Detection Rate (ODR)**

ODR은 관측 기간 내 실제 고장이 발생한 HDD 중 Prediction Horizon 내 최초 Alarm을 통해 고장을 탐지한 HDD의 비율을 나타낸다.

$$
\mathrm{ODR}
=
\frac{N_{\mathrm{On\text{-}time}}}
{N_{\mathrm{On\text{-}time}} + N_{\mathrm{Early}} + N_{\mathrm{Missed}}}
$$

ODR의 분모는 고장이 관측된 HDD 전체를 의미한다. 따라서 최초 Alarm이 발생했더라도 고장보다 30일을 초과하여 일찍 발생한 경우에는 Early로 분류되어 ODR의 성공적인 탐지에 포함되지 않는다.

따라서 ODR은 **실제 고장 중 운영상 유효한 시간 범위 내에서 탐지된 고장의 비율**을 나타내며, HDD 단위의 최초 Alarm과 예측 기간을 함께 반영한다.

**Early Alarm Proportion (EAP)**

EAP는 전체 평가 대상 HDD 중 Early 또는 Censored Early로 분류된 HDD의 비율을 나타낸다.

$$
\mathrm{EAP}
=
\frac{
N_{\mathrm{Early}} + N_{\mathrm{Censored\ Early}}
}{
N_{\mathrm{On\text{-}time}}
+
N_{\mathrm{Early}}
+
N_{\mathrm{Missed}}
+
N_{\mathrm{Censored\ Early}}
+
N_{\mathrm{Censored\ No\ Alarm}}
}
$$

분모는 전체 평가 대상 HDD를 의미하며, 우측 검열 HDD도 포함한다. 고장이 관측된 HDD에서는 실제 고장보다 30일을 초과하여 이르게 발생한 최초 Alarm을 Early로 분류하고, 우측 검열 HDD에서는 3.2절에서 정의한 평가 구간 내 최초 Alarm을 Censored Early로 분류한다.

따라서 EAP는 실제 고장 여부를 완전히 확인할 수 없는 우측 검열 HDD까지 포함하여 운영 환경에서 이른 Alarm이 발생하는 정도를 보수적으로 평가하기 위한 지표로 정의한다.

Censored No Alarm은 관측 종료 이후의 고장 여부를 확인할 수 없으므로 Early 또는 On-time의 어느 쪽에도 포함하지 않고 분모에만 포함한다.

**Median Lead Time**

$$
\mathrm{Median\ Lead\ Time}
=
\operatorname{median}
\left\{
LT_d
\mid
d \in D_{\mathrm{On\text{-}time}}
\cup
D_{\mathrm{Early}}
\right\}
$$

실제 고장이 관측된 HDD 중 최초 Alarm이 존재하는 경우의 Lead Time을 이용하여 중앙값을 산출한다. 우측 검열 HDD는 실제 고장 시점을 확인할 수 없으므로 제외한다.

| 지표 | 평가 대상 | 평가 의미 |
| --- | --- | --- |
| **OAP** | 고장 관측 + 최초 Alarm 존재 | 발생한 최초 Alarm의 시간적 적절성 |
| **ODR** | 고장 관측 HDD 전체 | 실제 고장 중 On-time으로 탐지된 비율 |
| **EAP** | 전체 평가 HDD | Early 성격의 최초 Alarm이 발생한 정도 |
| **Median Lead Time** | 고장 관측 + 최초 Alarm 존재 | 최초 Alarm부터 실제 고장까지의 시간적 여유 |

.

---

## 4. 실험 설계

<aside>

*4장.
Backblaze 데이터를 사용하여 평가 기준과 임곗값 선정 기준의 차이가 모델 성능 평가에 미치는 영향을 분석하기 위한 실험을 설계하였다. 먼저 Row-level 평가를 기준으로 임곗값을 최적화한 후, 동일한 임곗값을 HDD의 시간적 운영 과정을 반영한 평가에 적용하여 평가 단위의 차이에 따른 성능 변화를 비교하였다. 다음으로 Row-level 평가와 운영 환경 기반 평가를 각각 기준으로 임곗값을 최적화하고, 두 기준에서의 성능 및 모델 간 상대적 성능 변화를 비교하였다. 추가적으로 운영 환경 기반 평가에서 나타나는 Lead Time과 Alarm 발생 특성을 분석하여 평가 기준에 따라 나타나는 시간적 운영 특성의 차이를 확인하였다.*

</aside>

### 4.1 데이터 구축

#### 4.1.1 데이터 수집 및 HDD 선정

HDD 모델 선택

하드 디스크 드라이브 모델 선정 기준

- Backblaze SMART 데이터를 활용하였다.
- 서로 다른 제조사의 HDD 3종(Seagate, Toshiba, HGST)을 선정하여 특정 HDD 모델에 결과가 지나치게 의존하는지를 확인하고자 하였다.
- 각 모델은 HDD 수, 고장률, 관측기간 등 데이터 특성이 상이하므로, 서로 다른 데이터 특성을 갖는 HDD 모델에서 평가 결과를 비교할 수 있도록 구성하였다.

| 모델 | 전체 행 수 | 고유 시리얼 수 | 관측일 | 고장 드라이브 수 | 개체 고장률 | 개체 불균형 비율(정상 대비 고장) | 행 기준 고장률 | 드라이브 관측 일수(최솟값) | 드라이브 관측 일수(1사분위수) | 드라이브 관측 일수(중앙값) | 드라이브 관측 일수(3사분위수) | 드라이브 관측 일수(최댓값) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TOSHIBA_20MG07ACA14TA | 75016864 | 39387 | 2018.06.07-2026.03.31 | 2112 | 0.053621753 | 17.64914773 | 0.00282% | 1 | 1763 | 1930 | 2059 | 2726 |
| ST12000NM0007 | 37275453 | 38843 | 2017.09.06-2026.03.31 | 2289 | 0.058929537 | 15.96941896 | 0.00615% | 1 | 769 | 958 | 1137 | 3120 |
| HGST_20HUH721212ALN604 | 26905125 | 11438 | 2018.07.12-2026.03.31 | 1676 | 0.146529113 | 5.824582339 | 0.00624% | 2 | 2416 | 2503 | 2582 | 2805 |

#### 4.1.2 데이터 전처리

- 모델 간 공정한 비교를 위해 가능한 동일한 전처리 절차를 적용하였다.
1. Normalization 컬럼 제거 및 분석에 불필요한 컬럼 제거
2. 결측치 비율이 90% 이상인 컬럼 제거
3. 중복 행 제거
4. 시계열 공백 처리
    - 공백이 3일 이하인 경우 누락된 날짜의 행을 생성한 후 Forward Fill을 적용
    - 공백이 4일 이상인 경우 별도의 시계열 세그먼트 분리
5. 남은 결측치는 제거
6. 고장(1)로 기록되었다가 다시 정상(0)으로 기록되는 HDD 개체 제거
7. 모델별 중복 컬럼 정리 및 이상치 처리

### 4.2 실험 설정

**레이블 생성**

- HDD 고장 예측 문제를 이진 분류 문제로 정의하였다.
- 실제 고장일 이전 30일을 Positive로 정의하였다.
- 우측 검열 HDD의 경우 관측 종료 이후의 failure 여부를 확인할 수 없으므로, 30일 Prediction Horizon 전체를 관측할 수 없는 관측 시점은 학습 및 평가에서 제외하였다. 본 연구에서는 일별 관측 자료를 기준으로 관측 종료 시점으로부터 30일 미만의 잔여 관측 기간을 갖는 행을 해당 구간으로 정의하여 제거하였다.

**데이터 분할**

시계열 분할 결과 시뮬레이션

- HDD 단위 데이터 누수를 방지하기 위해 Group Stratified Split을 적용하였다.
- 동일 HDD가 서로 다른 데이터셋에 포함되지 않도록 구성하였다.
- 새로운 HDD에 대한 일반화 성능을 평가하도록 설계하였다.

**예측 모델** 

- Tree 기반 모델(LightGBM, XGBoost)과 시계열 모델(LSTM, GRU)을 비교 대상으로 선정하였다.

**모델 학습 환경**

- 모델 간 공정한 비교를 위해 동일한 학습 조건을 적용하였다.
- 데이터 누수를 방지하도록 표준화를 수행하였다.
- Early Stopping을 적용하여 과적합을 방지하였다.
- 세부 내용
    - **데이터 전처리 및 표준화**
    신경망 기반 모델(LSTM, GRU)에 한해 입력 피처를 표준화하였다. 데이터 누수를 방지하기 위해 학습 데이터셋에서만 StandardScaler를 학습(Fit)하고, 동일한 스케일러를 검증 및 테스트 데이터셋에 적용하였다. 또한 표준화 이후 발생할 수 있는 극단적인 이상치의 영향을 완화하기 위해 피처 값을 [-10.0, 10.0] 범위로 클리핑하였다. Tree 기반 모델(LightGBM, XGBoost)은 원본 피처를 그대로 사용하였다.
    - **하이퍼파라미터 설정**
    모델 간 공정한 비교를 위해 일반적으로 사용되는 기본 하이퍼파라미터를 적용하였다. GBDT 계열은 Learning Rate 0.05와 Number of Trees 300을 공통으로 사용하였으며, LightGBM은 Num Leaves 31, XGBoost는 Max Depth 6으로 설정하였다. 신경망 계열은 Adam Optimizer(Learning Rate 0.001, Weight Decay 1e-5)와 Batch Size 16384를 적용하였다.
    - **조기 종료(Early Stopping)**
    과적합을 방지하기 위해 검증 데이터셋의 손실을 기준으로 Early Stopping을 적용하였다. GBDT 계열은 20회(Rounds), 신경망 계열은 5회(Epochs) 연속 성능이 개선되지 않을 경우 학습을 종료하였으며, 최적의 검증 성능을 기록한 **모델 가중치**를 사용하였다.
    - **드롭아웃(Dropout)**
    신경망 모델(LSTM, GRU)의 과적합을 방지하기 위해 Dropout 0.2를 적용하였다.

**임곗값 선정**

- 각 임곗값 선정 시나리오에서는 Validation set에서 threshold를 0.001 간격으로 탐색하였다. 기존 행 단위 평가에서는 FAR ≤ 1%를, 운영 환경 기반 평가에서는 EAP ≤ 1%를 제약조건으로 설정하고, 각 제약조건을 만족하는 최소 임곗값을 선택하였다.
- 대규모 스토리지 시스템에서는 개별 HDD에서 발생하는 낮은 수준의 오탐이라도 전체 시스템 규모에서는 상당한 수의 불필요한 알람으로 누적될 수 있으며, 이는 운영자의 알람 피로(Alarm Fatigue)와 불필요한 유지보수 비용을 초래할 수 있다. 따라서 본 연구에서는 실제 운영 환경에서 허용 가능한 오탐 수준을 반영하기 위해 기존 행 단위 평가에서는 FAR(False Alarm Rate) 1% 이하를, 운영 환경 기반 평가에서는 EAP 1% 이하를 임곗값 최적화의 제약 조건으로 설정하였다.

### 4.3 실험 시나리오

제안한 운영 환경 기반 성능평가 방법의 유효성을 검증하기 위해 동일한 학습 모델에 대해 세 가지 성능평가 시나리오를 구성하였다. 또한 각 시나리오에서 산출된 운영 환경 기반 평가 결과를 이용하여 운영 특성을 추가적으로 분석하였다.

**(1) 기존 Row 기반 성능평가 재현**

- 검증 데이터에서 FAR 1% 이하를 만족하는 최소 임곗값을 선정하였다.
- 테스트 데이터의 모든 관측 시점을 독립적인 Row로 평가하였다.
- Row-level Precision, Recall, FAR를 산출하였다.

→ 기존 연구의 평가 절차 재현

**(2) 기존 임곗값의 운영 환경 적용**

- (1)에서 검증 데이터의 Row-level 평가를 통해 선정한 임곗값을 변경하지 않고 테스트 데이터에 적용하였다. 3장에서 정의한 운영 환경 기반 평가 **절차**에 따라 HDD별 최초 Alarm을 산출하고, 이를 기준으로 Precision, Recall, FAR 및 Median Lead Time을 평가하였다.

**(3) 운영 환경 기반 최적화**

- 검증 데이터에서 시간 순차적 최초 Alarm 기반 EAP 1% 이하를 만족하는 최소 임곗값을 별도로 선정하였다.
- 테스트 데이터에 동일한 임곗값과 운영 환경 기반 평가 절차를 적용하였다.
- HDD-level Precision, Recall, FAR 및 Median Lead Time을 산출하였다.
- 각 데이터셋과 random seed 조합을 하나의 비교 단위로 설정하고, 해당 조건에서 4개 모델의 Row-opt와 Op-opt 성능 순위를 각각 산출하였다.
- 두 평가 방식에서 최상위 모델이 동일한 비율(Top-1 agreement)을 비교하고, 4개 모델의 순위 일치 정도를 Spearman rank correlation을 이용하여 분석하였다.

**(4) 운영 특성 분석**

- 3장에서 정의한 운영 환경 기반 평가를 통해 산출된 HDD별 최초 Alarm과 Lead Time을 이용하여 모델의 시간적 예측 특성을 추가적으로 분석하였다. 고장이 관측된 HDD에 대해서는 Lead Time의 분포를 비교하고, 고장 시점과의 상대적 위치에 따른 Alarm 발생 양상을 분석하였다. 또한 우측 검열 HDD에서 발생한 False Alarm의 시점을 확인하여 오탐이 HDD 관측 기간의 어느 시점에서 발생하는지 비교하였다. 대표적인 HDD에 대해서는 시간에 따른 예측 확률과 Alarm 발생 시점을 시각화하여 개별 HDD의 운영 과정에서 나타나는 예측 양상을 확인하였다.

---

## 5. 실험 결과

<aside>

*5장. 
Row-level 기준으로 최적화한 임곗값을 운영 환경 기반 평가에 적용한 결과, FAR이 크게 증가하였으며 평가 단위에 따라 모델의 성능과 상대적 순위가 달라질 수 있음을 확인하였다. 또한 운영 환경 기반 평가를 기준으로 임곗값을 재최적화한 경우에도 Row-level 기준의 최적화 결과와 모델 간 순위가 일관되게 유지되지 않았다. 추가적으로 제안한 평가 방법을 통해 Lead Time, 오탐 발생 시점, HDD 생애주기상의 Alarm 변화 등 기존 평가에서 확인하기 어려운 운영 특성을 분석할 수 있음을 확인하였다.*

</aside>

### 5.1  Row-level 기준 임곗값의 평가 기준별 성능 비교

행 단위 분류 평가에서 최적화한 임곗값을 운영 환경 기반 평가에 대입하여 두 평가 방식에 따른 성능을 산출하고 비교함.

| Dataset | Model | Threshold | **Row Precision** | **Row Recall** | **Row FAR (%)** | **OAP** | **ODR** | **EAP (%)** | **Med LT** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HGST_20HUH721212ALN604 | LSTM | 0.043 | 7.36% | 45.70% | 1.03% | 10.48% | 6.79% | 12.55% | 219.0일 |
| HGST_20HUH721212ALN604 | GRU | 0.040 | 7.63% | 46.71% | 1.01% | 8.85% | 6.17% | 14.66% | 223.0일 |
| HGST_20HUH721212ALN604 | XGB | 0.025 | 7.13% | 44.80% | 1.04% | 11.86% | 8.64% | 16.15% | 239.5일 |
| HGST_20HUH721212ALN604 | LGBM | 0.018 | 7.56% | 46.21% | 1.01% | 13.91% | 9.88% | 13.87% | 212.0일 |
| ST12000NM0**0**07 | LSTM | 0.016 | 6.54% | 37.13% | 1.00% | 42.47% | 27.56% | 8.09% | 41.0일 |
| ST12000NM0007 | GRU | 0.016 | 6.90% | 38.80% | 0.98% | 43.33% | 28.89% | 8.68% | 45.5일 |
| ST12000NM0007 | XGB | 0.017 | 6.61% | 39.09% | 1.04% | 47.26% | 30.67% | 6.13% | 36.5일 |
| ST12000NM0007 | LGBM | 0.010 | 7.27% | 38.97% | 0.93% | 48.57% | 30.22% | 5.33% | 31.5일 |
| TOSHIBA_20MG07ACA14TA | LSTM | 0.004 | 3.84% | 47.96% | 1.03% | 27.14% | 18.18% | 10.03% | 114.0일 |
| TOSHIBA_20MG07ACA14TA | GRU | 0.004 | 4.39% | 48.10% | 0.90% | 26.06% | 17.70% | 10.54% | 108.0일 |
| TOSHIBA_20MG07ACA14TA | XGB | 0.002 | 6.57% | 43.36% | 0.53% | 34.04% | 22.97% | 5.54% | 56.0일 |
| TOSHIBA_20MG07ACA14TA | LGBM | 0.002 | 7.24% | 38.39% | 0.42% | 32.03% | 19.62% | 4.75% | 69.5일 |
- 동일한 모델을 행 단위 평가와 운영 환경 기반 평가로 측정한 결과, 성능 수준 자체가 서로 다르게 나타났다. 행 단위 평가에서 FAR은 모든 데이터셋과 모델에서 약 0.5~1.0% 수준으로 나타났으나, 운영 환경 기반 평가에서는 개별 HDD에서 최초 Alarm이 발생했는지를 기준으로 오탐을 집계하기 때문에 오탐 수준이 더 높게 나타났다. 또한 Recall 역시 평가 방식에 따라 차이를 보였다. 예를 들어 HGST에서 행 단위 Recall은 44.80~46.71%였으나, 운영 환경 기반 Recall(REAP)은 6.17~9.88%로 크게 낮아졌다. ST12000NM0007에서도 행 단위 Recall은 37.13~39.09%였던 반면, 운영 환경 기반 Recall은 27.56~30.67%로 나타났다. 이는 동일한 모델이라도 행 단위에서 개별 관측 시점의 예측 성능을 측정하는 경우와 실제 운영 상황에서 HDD 단위의 고장 식별 성능을 측정하는 경우 성능의 크기가 크게 달라질 수 있음을 보여준다.
- 모델 간 상대적 순위 역시 평가 방식에 따라 달라졌다. HGST에서 행 단위 Recall은 GRU > LightGBM > LSTM > XGBoost 순이었으나, 운영 환경 기반 Recall은 LightGBM > XGBoost > LSTM > GRU 순으로 나타났다. TOSHIBA에서도 행 단위 Recall은 GRU > LSTM > XGBoost > LightGBM이었지만, 운영 환경 기반 Recall은 XGBoost > LightGBM > LSTM > GRU로 역전되었다. 즉, 행 단위 평가에서 높은 성능을 보인 모델이 운영 환경 기반 평가에서도 동일한 상대적 성능을 보인다고 할 수 없었다. 이는 평가 방식의 차이가 단순히 성능 수치의 크기에만 영향을 미치는 것이 아니라 모델 간 성능 비교의 결과에도 영향을 줄 수 있음을 의미한다.
- 운영 환경 기반 평가에서는 행 단위 평가만으로는 확인하기 어려운 고장 탐지의 시간적 특성도 나타났다. 특히 HGST에서 모델별 Median Lead Time은 212.0~239.5일로 나타나, 동일하게 고장 HDD를 식별하더라도 최초 Alarm이 실제 고장보다 얼마나 앞서 발생하는지에는 모델별 차이가 존재하였다. 이는 운영 환경에서 모델의 성능을 비교할 때 고장 여부를 얼마나 정확하게 예측하는지뿐만 아니라, 실제 고장에 앞서 어느 시점에 최초 Alarm을 발생시키는지도 중요한 평가 요소임을 보여준다.
- 종합하면, 동일한 모델을 행 단위 평가와 운영 환경 기반 평가로 측정했을 때 ① 오탐 및 Recall의 성능 수준, ② 모델 간 상대적 순위, ③ 고장 탐지의 시간적 특성이 서로 다르게 나타났다. 행 단위 평가는 개별 관측 시점에서의 예측 성능을 측정하는 반면, 운영 환경 기반 평가는 시간 순차적인 관측 과정에서 최초 Alarm이 실제 고장 HDD의 식별로 이어지는지를 평가한다. 따라서 행 단위 평가만으로는 실제 운영 환경에서 나타나는 모델의 성능과 탐지 특성을 충분히 파악하기 어려우며, 운영 환경에서의 모델 성능을 평가하기 위해서는 운영 환경 기반 평가를 함께 고려할 필요가 있다.

### 5.2 운영 환경 기준 임곗값 최적화 결과

검증 데이터에서 Row-level 평가와 운영 환경 기반 평가에 대해 각각 FAR 1% 이하를 만족하는 최적 임곗값을 선정하고, 동일한 테스트 데이터에 적용하여 운영 환경 기반 평가 성능을 비교하였다.

| Dataset | Model | Threshold | **OAP** | **ODR** | **EAP (%)** | **Med LT** |
| --- | --- | --- | --- | --- | --- | --- |
| HGST_20HUH721212ALN604 | LSTM | 0.327 | 61.54% | 9.88% | 1.23% | 19.5일 |
| HGST_20HUH721212ALN604 | GRU | 0.485 | 66.67% | 8.64% | 0.79% | 11.0일 |
| HGST_20HUH721212ALN604 | XGB | 0.477 | 50.00% | 4.94% | 1.14% | 28.0일 |
| HGST_20HUH721212ALN604 | LGBM | 0.998 | 53.12% | 10.49% | 1.76% | 26.0일 |
| ST12000NM0**0**07 | LSTM | 0.354 | 57.45% | 12.00% | 1.03% | 21.0일 |
| ST12000NM0007 | GRU | 0.360 | 67.92% | 16.00% | 1.16% | 12.0일 |
| ST12000NM0007 | XGB | 0.251 | 75.93% | 18.22% | 0.70% | 4.0일 |
| ST12000NM0007 | LGBM | 0.173 | 64.10% | 11.11% | 1.24% | 22.0일 |
| TOSHIBA_20MG07ACA14TA | LSTM | 0.314 | 44.44% | 9.57% | 0.99% | 38.0일 |
| TOSHIBA_20MG07ACA14TA | GRU | 0.192 | 44.68% | 10.05% | 0.91% | 51.0일 |
| TOSHIBA_20MG07ACA14TA | XGB | 0.905 | 43.90% | 8.61% | 0.71% | 49.0일 |
| TOSHIBA_20MG07ACA14TA | LGBM | 0.526 | 50.00% | 6.22% | 0.51% | 33.5일 |
- 평가 기준에 따라 선택되는 임곗값이 서로 다르게 나타났다. 동일한 데이터셋과 모델이라도 행 단위 평가를 기준으로 최적화한 경우와 운영 환경 기반 평가를 기준으로 최적화한 경우에 서로 다른 임곗값이 선택되었다. 이는 동일한 예측 점수를 사용하더라도 어떤 평가 기준을 임곗값 최적화에 적용하는지에 따라 선택되는 운영점이 달라질 수 있음을 보여준다.
- 임곗값 최적화 기준의 차이는 동일한 테스트 데이터에서 운영 환경 기반 성능의 차이로 이어졌다. 운영 환경 기준으로 선정한 임곗값을 적용했을 때 OAP와 ODR, Median Lead Time은 데이터셋과 모델에 따라 서로 다른 수준을 보였다. 예를 들어 ST12000NM0007에서는 XGB가 ODR 18.22%로 가장 높았으며 Median Lead Time은 4.0일로 나타났다. 반면 HGST에서는 LGBM의 ODR이 10.49%로 가장 높았으며 Median Lead Time은 26.0일이었다. 이는 동일한 임곗값 최적화 기준을 적용하더라도 모델의 운영 성능과 고장 탐지 시점이 데이터셋에 따라 다르게 나타날 수 있음을 보여준다.
- 임곗값 최적화 기준에 따라 모델 간 상대적 성능도 달라질 수 있었다. 특히 TOSHIBA에서는 운영 환경 기준 임곗값을 적용했을 때 ODR이 GRU 10.05%, LSTM 9.57%, XGB 8.61%, LGBM 6.22%로 나타나 모델 간 성능 차이가 확인되었다. 이는 모델 자체의 예측 성능뿐만 아니라 어떤 기준으로 임곗값을 선정하는지가 최종적인 운영 환경에서의 모델 비교 결과에도 영향을 줄 수 있음을 보여준다.
- 종합하면, 운영 환경을 기준으로 임곗값을 최적화하는 것은 단순히 임곗값의 선택 기준을 변경하는 데 그치지 않고, 동일한 테스트 데이터에서 관찰되는 운영 성능과 모델 간 상대적 성능에도 영향을 미칠 수 있었다. 다만 그 차이의 방향과 크기는 데이터셋과 모델에 따라 달랐다. 따라서 RQ2에 대한 결과는 운영 환경 기준 임곗값 최적화가 모든 경우에 성능을 향상시킨다는 것이 아니라, 임곗값을 어떤 평가 기준에 따라 최적화하는지가 운영 환경에서의 성능 평가 결과 자체를 변화시킬 수 있다는 것으로 해석할 수 있다.

### 5.3 다중시드 기반 결과의 강건성 분석

방법론

단일 학습 seed에 따른 결과의 우연성을 배제하고 5.2에서 확인된 임곗값 최적화 기준의 차이가 반복 실험에서도 나타나는지를 검증하기 위해 다중시드 반복 실험(multi-seed repeated experiments)을 수행하였다. 총 13개의 random seed에 대해 5.2와 동일한 학습 및 평가 절차를 반복하였으며, 각 데이터셋·모델·seed 조합에서 행 단위 평가 기준으로 선정한 임곗값과 운영 환경 기반 평가 기준으로 선정한 임곗값을 각각 동일한 테스트 데이터에 적용하였다. 따라서 두 임곗값의 결과는 동일한 데이터셋, 동일한 모델, 동일한 seed에서 얻어진 대응 관측값(paired observations)으로 구성하여 비교하였다.

분석은 세 가지 관점에서 수행하였다. 첫째, 두 임곗값 적용 결과의 OAP, ODR 및 Median Lead Time을 비교하여 운영 환경 기반 성능의 변화 방향과 크기를 확인하였다. 둘째, 순위 안정성 분석(rank stability analysis)을 통해 각 Dataset–Seed에서 네 모델의 ODR 순위를 비교하고, 두 임곗값 최적화 기준 간 1위 모델의 일치 여부와 순위 변화를 확인하였다. 이때 행 단위 평가와 운영 환경 기반 평가에서 서로 대응되는 성능 지표를 기준으로 모델 순위를 비교하여, 기존 평가에서의 모델 우수성이 운영 환경 기반 평가에서도 유지되는지를 확인하였다. 또한 동일한 임곗값 최적화 기준에서 seed만 변경했을 때 나타나는 모델 순위의 변동을 함께 확인하여, 학습 seed에 따른 변동성과 임곗값 최적화 기준에 따른 변화를 구분하였다. 셋째, 두 기준에서의 모델 순위 전체가 얼마나 일치하는지를 확인하기 위해 각 Dataset–Seed별 Spearman 순위상관(Spearman's rank correlation)을 계산하였다. 이를 통해 단일 seed에서 관찰된 결과가 특정 학습 조건에 의존한 것인지, 또는 반복 학습에서도 임곗값 최적화 기준에 따른 운영 성능 및 모델 순위의 차이가 나타나는지를 검토하였다.

| Dataset | 행 단위 기준: seed에 따른 1위 변동 | 운영 환경 기준: seed에 따른 1위 변동 | 동일 seed에서 최적화 기준 변경 시 1위 변동 |
| --- | --- | --- | --- |
| HGST | 2/13 (15.4%) | 2/13 (15.4%) | **5/13 (38.5%)** |
| ST12000NM0007 | 5/13 (38.5%) | 4/13 (30.8%) | **8/13 (61.5%)** |
| TOSHIBA | 0/13 (0.0%) | 7/13 (53.8%) | **11/13 (84.6%)** |
- **다중시드 분석에서도 행 단위 기준 임곗값과 운영 환경 기준 임곗값의 적용에 따른 운영 환경 기반 성능 차이가 반복적으로 관찰되었다.** 특히 ODR의 변화는 데이터셋에 따라 뚜렷한 차이를 보였다. HGST에서는 모델에 따라 증가와 감소가 혼재한 반면, ST12000NM0007과 TOSHIBA에서는 모든 모델에서 운영 환경 기준 임곗값을 적용했을 때 ODR이 감소하는 경향이 나타났다. 이는 5.2에서 확인된 성능 차이가 특정 seed에서만 발생한 결과라기보다 **데이터셋과 모델에 따라 반복적으로 나타나는 현상**임을 보여준다. 동시에 모든 데이터셋에서 동일한 방향의 변화가 나타난 것은 아니므로, 운영 환경 기준 임곗값 최적화의 효과를 일률적인 성능 향상으로 해석하기보다는 **임곗값 최적화 기준에 따라 운영 환경에서 측정되는 성능의 수준과 분포가 달라지는 현상**으로 해석할 필요가 있다.
- **학습 seed 자체에 따라서도 모델 간 상대적 성능의 변동이 나타났다.** 동일한 임곗값 최적화 기준을 유지한 상태에서 13개 seed의 ODR 순위를 비교한 결과, 데이터셋에 따라 1위 모델이 seed에 따라 변경되었다. 특히 행 단위 기준에서는 HGST에서 2/13, ST12000NM0007에서 5/13의 seed에서 1위 모델이 변경되었으며, 운영 환경 기준에서는 HGST 2/13, ST12000NM0007 4/13, TOSHIBA 7/13의 seed에서 1위 모델이 변경되었다. 이는 **단일 seed에서 관찰되는 모델 순위가 학습 조건에 따라 달라질 수 있음**을 보여주며, 따라서 특정 seed의 결과만으로 모델 간 우열을 판단하는 데에는 한계가 있음을 의미한다.
- **그러나 동일한 seed에서 임곗값 최적화 기준만 변경한 경우에도 모델 순위의 변화가 확인되었다.** 각 Dataset–Seed에서 네 모델의 ODR 순위를 비교한 결과, 행 단위 기준 임곗값과 운영 환경 기준 임곗값을 적용했을 때 1위 모델이 변경된 경우는 HGST 5/13(38.5%), ST12000NM0007 8/13(61.5%), TOSHIBA 11/13(84.6%)로 나타났다. 즉, **동일한 학습 seed와 동일한 테스트 데이터를 사용하더라도 임곗값을 어떤 평가 기준에 따라 최적화하는지에 따라 1위 모델이 달라질 수 있음**을 확인하였다. 이는 5.2에서 관찰된 모델 간 성능 차이가 단순히 학습 seed에 따른 변동만으로 설명되지 않으며, **임곗값 최적화 기준 자체도 모델 선택 결과에 영향을 미칠 수 있음**을 보여준다.
- **모델의 전체 순위 구조를 비교한 결과에서도 두 최적화 기준 사이의 완전한 일치는 확인되지 않았다.** 각 Dataset–Seed에서 행 단위 기준 임곗값과 운영 환경 기준 임곗값을 적용했을 때의 ODR 순위에 대해 Spearman 순위상관을 계산한 결과, 전체 비교의 중앙값은 0.60으로 나타났다. 데이터셋별 중앙값은 HGST 0.40, ST12000NM0007 −0.20, TOSHIBA 0.80으로 나타나 데이터셋에 따라 순위 일치 정도에도 차이가 있었다. 특히 ST12000NM0007에서는 음의 순위상관이 나타나, **행 단위 기준에서 상대적으로 높은 성능을 보인 모델이 운영 환경 기준 임곗값을 적용했을 때에도 높은 성능을 보인다는 관계가 유지되지 않을 수 있음**을 확인하였다. 다만 모델 수가 4개로 제한되어 있으므로, 순위상관은 모델 순위의 안정성을 확인하기 위한 **보조적인 분석**으로 해석하였다.
- **종합하면, 13개 seed를 이용한 반복 실험에서 학습 seed에 따른 모델 순위의 변동성과 함께 임곗값 최적화 기준에 따른 추가적인 순위 변화가 확인되었다.** 특히 동일한 seed에서 임곗값 최적화 기준만 변경했을 때에도 데이터셋별로 38.5~84.6%의 비교에서 1위 모델이 변경되었다. 이는 **임곗값 최적화 기준에 따른 모델 순위의 변화가 특정 학습 seed의 결과에만 의존하지 않으며, 동일한 학습 조건에서도 평가 기준의 변화에 의해 발생할 수 있음**을 보여준다. 따라서 운영 환경에서 모델을 비교·선택할 때에는 단일 seed의 행 단위 성능만을 기준으로 판단하기보다, **반복 학습을 통한 결과의 변동성을 확인하고 운영 환경을 반영한 임곗값 최적화 및 평가를 함께 고려할 필요가 있다.**

### 5.4 운영 환경 기반 평가를 통한 운영 특성 분석

운영 환경 기반 평가는 HDD별 최초 Alarm을 기준으로 HDD-level 성능을 산출하는 동시에 HDD의 전체 시간 순차적 Alarm 정보를 보존한다. **따라서 행 단위 또는 집계형 성능지표만으로는 확인하기 어려운 Alarm의 발생 시점, 반복 및 지속 양상, False Alarm의 시간적 위치, 그리고 개별 HDD에서의 예측확률 변화를 분석할 수 있다.** 본 절에서는 최초 Alarm의 발생 시점, 반복 Alarm의 시간적 분포, False Alarm의 발생 시점, 그리고 개별 HDD에서의 예측확률 변화를 분석하여, 모델의 경보가 실제 운영 과정에서 언제 발생하고 어떻게 지속되는지를 살펴본다.

- 사진 보관소
    
    !image.png
    
    !image.png
    
    !image.png
    

#### **5.4.1 Lead Time 분석**

!**그림 2. 모델별 최초 알람의 Lead Time 분포(HGST 20HUH721212ALN604).** 운영 환경 기반 평가에서 고장이 발생한 HDD를 대상으로, 각 HDD에서 발생한 **최초 알람 시점부터 실제 고장까지의 Lead Time** 분포를 모델별로 나타낸다. 각 패널의 n은 최초 알람이 발생한 HDD 수를 나타내며, 적색 점선은 해당 모델의 **중앙 Lead Time(Median Lead Time)**을 표시한다. 모델별로 중앙값뿐 아니라 Lead Time의 분포 범위와 장기 Lead Time의 발생 양상이 상이하게 나타나며, 이를 통해 최초 알람이 실제 고장에 앞서 발생하는 시간적 특성을 모델별로 비교할 수 있다.

**그림 2. 모델별 최초 알람의 Lead Time 분포(HGST 20HUH721212ALN604).** 운영 환경 기반 평가에서 고장이 발생한 HDD를 대상으로, 각 HDD에서 발생한 **최초 알람 시점부터 실제 고장까지의 Lead Time** 분포를 모델별로 나타낸다. 각 패널의 n은 최초 알람이 발생한 HDD 수를 나타내며, 적색 점선은 해당 모델의 **중앙 Lead Time(Median Lead Time)**을 표시한다. 모델별로 중앙값뿐 아니라 Lead Time의 분포 범위와 장기 Lead Time의 발생 양상이 상이하게 나타나며, 이를 통해 최초 알람이 실제 고장에 앞서 발생하는 시간적 특성을 모델별로 비교할 수 있다.

- 최초 Alarm의 Lead Time은 모델별로 차이를 보였다. LightGBM과 XGBoost는 비교적 짧은 Lead Time 구간에 최초 Alarm이 집중된 반면, LSTM과 GRU는 상대적으로 넓은 범위의 Lead Time을 보였다. 중앙값은 LightGBM 26.0일, XGBoost 28.0일, LSTM 19.5일, GRU 11.0일이다.
- 이러한 차이는 모델별로 HDD를 처음 위험 대상으로 식별하는 시점이 서로 다를 수 있음을 보여준다. 또한 중앙값만으로는 분포의 폭이나 장기 Lead Time 사례를 설명하기 어려워, 최초 Alarm이 발생하는 시간적 범위를 함께 확인할 필요가 있다.
- 따라서 최초 Alarm Lead Time은 기존의 탐지 여부나 성능 비율만으로는 확인하기 어려운 “고장 이전 언제부터 해당 HDD를 위험 대상으로 식별하기 시작하는가”라는 운영 정보를 제공한다. 이는 모델이 실제 운영에서 어느 정도의 사전 대응 시간을 제공하는지를 파악하는 데 활용할 수 있다.

#### **5.4.2 알람 발생 특성 분석**

운영 환경 기반 평가는 HDD별 최초 Alarm을 이용한 성능지표뿐만 아니라, 각 관측 시점에서 발생한 Alarm의 시간적 위치를 보존한다. 이를 통해 고장 HDD에서 Alarm이 실제 고장과 어떤 시간적 관계를 갖는지와, False Positive로 판정된 HDD에서 최초 False Alarm이 운영 기간 중 언제 발생하는지를 분석할 수 있다.

!**그림 3. 모델별 고장 이전 알람의 Lead Time 분포(HGST 20HUH721212ALN604).** 운영 환경 기반 평가에서 고장이 발생한 HDD의 전체 알람을 대상으로, 각 알람 발생 시점에서 실제 고장까지 남은 기간(Lead Time)의 분포를 모델별로 나타낸다. 각 패널의 n은 해당 모델에서 발생한 전체 알람 수를 나타내며, 적색 점선은 Lead Time의 중앙값을 표시한다. 모델에 따라 알람이 고장에 앞서 발생하는 시점과 그 분포가 상이하게 나타나며, 이는 알람의 **발생 빈도뿐 아니라 고장까지의 시간적 여유 역시 모델별로 다를 수 있음**을 보여준다.

**그림 3. 모델별 고장 이전 알람의 Lead Time 분포(HGST 20HUH721212ALN604).** 운영 환경 기반 평가에서 고장이 발생한 HDD의 전체 알람을 대상으로, 각 알람 발생 시점에서 실제 고장까지 남은 기간(Lead Time)의 분포를 모델별로 나타낸다. 각 패널의 n은 해당 모델에서 발생한 전체 알람 수를 나타내며, 적색 점선은 Lead Time의 중앙값을 표시한다. 모델에 따라 알람이 고장에 앞서 발생하는 시점과 그 분포가 상이하게 나타나며, 이는 알람의 **발생 빈도뿐 아니라 고장까지의 시간적 여유 역시 모델별로 다를 수 있음**을 보여준다.

- 전체 Alarm의 Lead Time 분포는 모델별로 서로 다른 시간적 양상을 보였다. 모든 모델에서 고장에 가까운 구간에 Alarm이 상대적으로 많이 분포했지만, LightGBM과 XGBoost는 짧은 Lead Time 구간에 보다 집중된 반면 LSTM과 GRU는 고장 이전의 넓은 기간에 걸쳐 Alarm이 분포하였다.
- 이는 최초 Alarm만으로는 확인하기 어려운 Alarm의 반복 및 지속 범위를 보여준다. 즉, 모델이 처음 위험을 감지한 시점뿐 아니라 이후 고장까지의 기간 동안 어느 시점에 경보를 반복적으로 발생시키는지를 확인할 수 있다.
- 따라서 전체 Alarm Lead Time 분포는 “모델의 경보가 고장 이전 시간축에서 얼마나 넓게, 그리고 어떤 시점에 반복적으로 발생하는가”라는 운영 정보를 제공한다. 최초 Alarm Lead Time이 첫 경보의 발생 시점을 나타낸다면, 전체 Alarm 분포는 그 이후의 경보 발생 양상을 보완적으로 나타낸다.

!**그림 4. 모델별 운영 환경 오탐의 최초 발생 시점 분포(HGST 20HUH721212ALN604).** 운영 환경 기반 평가에서 오탐으로 분류된 HDD를 대상으로 관측 시작일 이후 최초 오탐이 발생한 시점을 모델별로 나타낸다. 각 점은 하나의 HDD에서 관측된 최초 오탐 발생 시점을 의미하며, 가로축은 관측 시작 이후 경과 일수이다. 이를 통해 모델별 오탐 발생 시점의 차이와 시간적 집중 양상을 확인할 수 있으며, 오탐의 빈도뿐 아니라 **발생 시점 역시 운영 환경에서 고려할 필요가 있음을 보여준다.**

**그림 4. 모델별 운영 환경 오탐의 최초 발생 시점 분포(HGST 20HUH721212ALN604).** 운영 환경 기반 평가에서 오탐으로 분류된 HDD를 대상으로 관측 시작일 이후 최초 오탐이 발생한 시점을 모델별로 나타낸다. 각 점은 하나의 HDD에서 관측된 최초 오탐 발생 시점을 의미하며, 가로축은 관측 시작 이후 경과 일수이다. 이를 통해 모델별 오탐 발생 시점의 차이와 시간적 집중 양상을 확인할 수 있으며, 오탐의 빈도뿐 아니라 **발생 시점 역시 운영 환경에서 고려할 필요가 있음을 보여준다.**

- False Alarm의 최초 발생 시점은 모델별로 서로 다른 시간적 분포를 보였다. LightGBM과 XGBoost에서는 관측 초기와 후반부 모두 False Alarm 사례가 나타났으며, 특히 후반부에 사례가 상대적으로 많이 분포하였다. LSTM과 GRU에서는 관측 중·후반부에 False Alarm이 주로 나타나는 양상이 확인되었다.
- 이는 FAR이 **오탐의 발생 비율**을 나타내는 것과 달리, 본 분석에서는 **오탐이 HDD의 운영 기간 중 언제 처음 발생하는지**를 확인할 수 있음을 보여준다.
- 따라서 False Alarm의 시간적 분포는 모델별 **오탐 발생 시점과 시간적 집중 양상**이라는 운영 정보를 제공한다. 이를 통해 오탐이 운영 초기부터 나타나는지, 특정 기간에 집중되는지 등을 구분할 수 있다.

#### **5.4.3 운영 타임라인 사례 분석**

!**그림 5. 운영 환경 기반 평가에서의 네 가지 알람 사례에 대한 시간적 예시 (HGST 20HUH721212ALN604, GRU).** 예측 확률과 의사결정 임곗값의 시간적 관계를 이용하여 On-time Alarm, Early Alarm, Censored Early Alarm 및 Missed Failure의 대표 사례를 제시한다. 각 사례에서 최초 임곗값 초과 시점과 실제 고장 시점의 관계를 통해, 운영 환경에서 알람의 발생 시점과 고장 발생 여부에 따른 판정 기준을 직관적으로 나타낸다.

**그림 5. 운영 환경 기반 평가에서의 네 가지 알람 사례에 대한 시간적 예시 (HGST 20HUH721212ALN604, GRU).** 예측 확률과 의사결정 임곗값의 시간적 관계를 이용하여 On-time Alarm, Early Alarm, Censored Early Alarm 및 Missed Failure의 대표 사례를 제시한다. 각 사례에서 최초 임곗값 초과 시점과 실제 고장 시점의 관계를 통해, 운영 환경에서 알람의 발생 시점과 고장 발생 여부에 따른 판정 기준을 직관적으로 나타낸다.

- On-time Alarm 사례에서는 고장에 가까워지는 시점에서 예측확률이 임곗값을 초과하여 최초 Alarm이 발생하고, 이후에도 임곗값을 상회하는 구간이 반복된다. 이를 통해 **최초 Alarm 이후 위험 신호가 어떻게 변화하고 반복되는지**를 확인할 수 있다.
- Early Alarm 사례에서는 고장보다 이른 시점에 최초 Alarm이 발생한 후 예측확률이 다시 하락하고 이후 재상승하는 양상이 나타난다. 이를 통해 **최초 Alarm과 실제 고장 사이의 시간적 관계뿐 아니라 Alarm 이후 예측확률의 변화**를 확인할 수 있다.
- Censored Early Alarm 사례에서는 실제 고장이 관측되지 않은 상태에서 예측확률이 임곗값 주변에서 반복적으로 변동하며 Alarm이 발생한다. 이를 통해 **관측 종료 전까지 위험 신호와 Alarm이 어떻게 나타나는지**를 확인할 수 있다.
- Missed Failure 사례에서는 고장에 가까워지면서 예측확률이 상승하지만 임곗값을 넘지 않는다. 따라서 최종적인 FN 여부뿐 아니라 **미탐지 과정에서 예측확률이 어느 수준까지 상승했는지**를 확인할 수 있다.
- 종합하면, 운영 타임라인은 동일한 TP·FP·FN 등의 최종 판정에 대해서도 **최초 임곗값 초과 시점, 실제 고장과의 시간적 관계, 이후 Alarm의 반복 여부, 그리고 미탐지 과정의 예측확률 변화**를 개별 HDD 수준에서 확인할 수 있게 한다.

이상의 분석을 통해 운영 환경 기반 성능평가는 기존의 집계형 성능지표만으로는 확인하기 어려운 **네 가지 운영 정보를 제공함을 확인하였다.** 첫째, **최초 Alarm Lead Time**을 통해 고장 이전 언제부터 HDD를 위험 대상으로 식별하기 시작하는지를 확인할 수 있다. 둘째, **전체 Alarm의 시간적 분포**를 통해 최초 Alarm 이후 경보가 고장 이전 어느 기간에 걸쳐 반복적으로 발생하는지를 확인할 수 있다. 셋째, **False Alarm의 발생 시점**을 통해 오탐이 HDD의 운영 기간 중 언제 발생하는지를 확인할 수 있다. 넷째, **개별 HDD의 운영 타임라인**을 통해 예측확률의 변화, 임곗값 초과 시점, Alarm의 반복 및 지속 양상을 시간축에서 확인할 수 있다. **즉, 운영 환경 기반 평가는 기존 평가가 제공하는 탐지 여부나 집계된 성능 수준을 넘어, 모델의 경보가 운영 과정에서 언제 발생하고, 고장까지 얼마나 앞서 나타나며, 이후 어떻게 반복·변화하는지를 확인할 수 있는 시간적·운영적 정보를 제공한다.**

---

## 6. 결론

<aside>

*6장. 
본 연구에서는 HDD 고장 예측 모델의 평가가 개별 관측 시점의 분류 성능에만 의존할 경우 실제 HDD 운영 과정에서의 성능을 충분히 나타내기 어려울 수 있다는 문제의식에서 출발하였다. 이를 위해 HDD 전체 관측 기간에 대해 시간 순서에 따른 연속 추론을 수행하고, 각 HDD에서 발생하는 최초 Alarm을 기준으로 성능을 평가하는 운영 환경 기반 평가 방법을 구성하였다. 이를 통해 동일한 예측 결과를 사용하더라도 평가 단위와 임곗값 최적화 기준에 따라 운영 환경에서 관찰되는 성능이 어떻게 달라지는지를 분석하였다.*

</aside>

**연구 요약**

- 첫째, Row-level 평가와 운영 환경 기반 HDD-level 평가에서는 동일한 모델과 동일한 임곗값을 사용하더라도 서로 다른 성능이 나타났다. Row-level 기준으로 최적화한 임곗값을 운영 환경 기반 평가에 적용한 결과, HDD-level FAR이 Row-level FAR보다 높게 나타났으며, Precision과 Recall 등 성능지표의 수준과 모델 간 상대적 성능도 평가 기준에 따라 달라졌다. 이는 개별 관측 Row의 분류 결과와 HDD 단위에서 최초 Alarm을 기준으로 산출한 운영 성능이 동일한 정보를 제공하지 않음을 보여준다.
- 둘째, 임곗값을 어떤 평가 기준으로 최적화하는지에 따라서도 운영 환경에서의 결과가 달라졌다. 운영 환경 기반 평가에서 FAR 1% 이하의 조건을 만족하도록 임곗값을 재최적화한 결과, Row-level 기준으로 선정한 임곗값과 상당한 차이가 나타났다. 운영 환경 기준 임곗값을 적용하면 HDD-level FAR은 전반적으로 감소하고 일부 모델에서는 Precision이 향상되는 반면, Recall과 Median Lead Time은 데이터셋과 모델에 따라 증가하거나 감소하였다. 또한 두 최적화 기준에서 모델의 상대적 성능 순위가 일관되게 유지되지 않았다. 따라서 임곗값 최적화의 목적과 평가 단위가 달라지면 동일한 모델의 운영 성능을 평가하는 결과뿐만 아니라 모델 선택 결과 자체도 달라질 수 있음을 확인하였다.
- 셋째, 이러한 차이가 특정 학습 seed에만 의존하는지를 확인하기 위해 다중시드 반복 실험을 수행하였다. 13개의 random seed에 대해 동일한 학습 및 평가 절차를 반복한 결과, 임곗값 최적화 기준에 따른 운영 성능 차이가 반복적으로 관찰되었다. 또한 39개의 Dataset–Seed 비교에서 두 임곗값 기준의 ODR 1위 모델이 동일하게 유지된 경우는 11개(28.2%)에 그쳤으며, 28개(71.8%)에서는 1위 모델이 변경되었다. 두 기준에서의 모델 순위 전체를 비교한 Spearman 순위상관의 중앙값 역시 0.60으로 나타났으며, 데이터셋별로 순위 일치 정도에 차이가 있었다. 이는 앞서 확인된 모델 순위 변화가 단일 학습 seed에서 우연히 발생한 현상으로만 보기 어렵고, 반복 학습 조건에서도 평가 기준에 따라 모델의 상대적 성능이 달라질 수 있음을 보여준다.
- 넷째, 운영 환경 기반 평가는 집계된 성능지표 외에도 기존 Row-level 평가에서 직접 확인하기 어려운 시간적·운영적 정보를 제공하였다. 최초 Alarm의 Lead Time을 통해 모델이 고장 이전 언제부터 HDD를 위험 대상으로 식별하기 시작하는지를 확인할 수 있었으며, 전체 Alarm의 시간적 분포를 통해 최초 Alarm 이후 경보가 고장까지 어떤 범위에서 반복적으로 발생하는지를 확인할 수 있었다. 또한 False Alarm의 최초 발생 시점을 통해 오탐이 HDD의 운영 기간 중 언제 발생하는지를 확인하였으며, 개별 HDD의 운영 타임라인을 통해 예측확률의 변화와 임곗값 초과, Alarm의 발생 및 반복 양상을 시간축에서 확인할 수 있었다.

**연구 의의 및 시사점**

- 본 연구의 주요 의의는 HDD 고장 예측의 성능평가 대상을 개별 관측 Row에서 HDD의 시간적 운영 과정으로 확장하였다는 점에 있다. 기존 Row-level 평가는 개별 시점의 분류 성능을 정량적으로 비교하는 데 유용하지만, 하나의 HDD에서 여러 시점에 걸쳐 발생하는 예측 결과와 최초 Alarm의 발생 시점, 고장과의 시간적 관계 등을 직접적으로 반영하기 어렵다. 본 연구에서는 HDD 전체 관측 기간의 시간 순차적 추론과 최초 Alarm 기반 HDD-level 평가를 통해 이러한 시간적 관계를 평가 과정에 포함하였다.
- 또한 본 연구는 평가 기준이 단순히 성능을 측정하는 단계뿐만 아니라 임곗값과 모델의 선택 결과에도 영향을 줄 수 있음을 실험적으로 확인하였다. Row-level 기준으로 최적화한 임곗값과 운영 환경 기준으로 최적화한 임곗값이 서로 다르게 선택되었고, 두 기준에서 모델의 상대적 성능 순위도 일관되게 유지되지 않았다. 따라서 실제 운영 환경에서 사용할 모델과 임곗값을 결정할 때에는 Row-level 평가 결과를 그대로 적용하기보다, 실제 운영 단위와 Alarm 발생 과정을 반영한 평가를 함께 고려할 필요가 있다.
- 마지막으로 운영 환경 기반 평가는 단일한 성능 수치로 모델의 우열을 판단하는 데 그치지 않고, 언제 위험을 감지하는지, 고장까지 어느 정도의 시간적 여유를 제공하는지, 오탐이 언제 발생하는지, 그리고 개별 HDD에서 예측과 Alarm이 시간에 따라 어떻게 변화하는지를 함께 분석할 수 있다는 점에서 운영 의사결정에 활용 가능한 추가 정보를 제공한다.

**연구의 한계 및 향후 연구**

- 데이터 범위의 한계: 본 연구는 Backblaze의 3개 HDD 모델과 4개 예측 모델을 대상으로 평가하였다. 따라서 본 연구에서 확인된 결과를 모든 HDD 또는 다른 설비의 예지보전 문제에 일반화하기에는 한계가 있다. 향후에는 더 다양한 HDD 모델과 SSD, 서버 및 산업 설비 등의 시계열 데이터를 대상으로 동일한 평가 절차를 적용하여 일반화 가능성을 검증할 필요가 있다.
- 운영 조건 설정의 한계: 본 연구에서는 FAR 1% 이하를 임곗값 최적화 조건으로 설정하고, 고장 이전 30일을 예측 기간으로 설정하였다. 이러한 조건은 본 연구의 비교를 위한 실험 설정이므로 실제 운영 환경의 요구사항을 모두 반영한다고 볼 수 없다. 향후에는 다양한 FAR 제약, 예측 기간 및 유지보수 비용을 반영하여 운영 정책에 따른 최적 임곗값과 모델 선택의 변화를 분석할 필요가 있다.
- Row-level 비교 기준의 한계: 본 연구에서는 HDD 전체 관측 기간의 각 관측 시점을 독립적인 Row로 평가하는 방식을 기존 평가의 비교 기준으로 사용하였다. 그러나 실제 Row-level 평가에는 Sliding Window 등 다양한 평가 절차가 존재할 수 있으므로, 이러한 평가 설정의 차이가 운영 환경 기반 평가와의 관계에 미치는 영향을 추가적으로 검증할 필요가 있다.
- 동적 운영 환경의 한계: 본 연구에서는 HDD 전체 관측 기간에 대해 시간 순차적 추론을 수행함으로써 실제 운영 과정에서의 연속적인 예측을 모사하였다. 그러나 실제 운영 환경에서 발생할 수 있는 지속적인 데이터 유입, 모델 재학습 및 모델 갱신까지는 고려하지 않았다. 향후에는 온라인 학습과 모델 갱신을 포함하는 동적 운영 환경으로 평가 범위를 확장할 필요가 있다.
- Alarm 정책의 한계: 본 연구에서는 HDD 단위의 최초 Alarm을 중심으로 운영 성능을 평가하였다. 따라서 반복 Alarm의 횟수와 지속 시간, 연속적인 Alarm 발생 여부 및 Alarm 이후의 예측확률 변화 등을 정량적인 평가 지표에 직접 반영하지 않았다. 향후에는 이러한 Alarm의 시간적 특성과 유지보수 비용 및 대응 정책을 함께 고려하여 보다 다양한 운영 시나리오를 반영하는 평가 체계로 확장할 필요가 있다.