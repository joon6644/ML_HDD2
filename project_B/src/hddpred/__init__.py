"""HDD 고장 예측 - 시간축 forward split 기반 다음 한 달 예측.

파이프라인 단계는 서로를 직접 호출하지 않는다. 순서를 아는 것은
hddpred.experiments.runner 하나뿐이다.

    raw -> canonical -> features -+
                             \\    +-> fold data -> model -> threshold -> metrics
                              labels -> splits -+
"""

__version__ = "0.1.0"
