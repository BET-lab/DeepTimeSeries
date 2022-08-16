DeepTimeSeries documentation
============================================

Time series forecasting을 위한 PyTorch 기반의 딥러닝 라이브러리입니다.

Why DeepTimeSeries?
-------------------
DeepTimeSeries는 ``Darts`` 나 ``Pytorch Forecasting`` 과 같은 라이브러리에서 많은
영감을 얻었습니다. 그렇다면 이러한 라이브러리가 있는 와중에 DeepTimeSeries가 개발된
이유는 무었일까요?

DeepTimeSeries의 설계 철학은 다음과 같습니다.

1. 다양한 시계열 예측 딥러닝 구조를 설계할 수 있는 논리적 가이드라인을 제시한다.
2. 라이브러리를 경량화 하여 불필요한 의존성을 최소화한다.

우리의 메인 타겟 사용자는 시계열 예측을 위하여 딥러닝 모델을 개발해야 하는 중급자
수준의 사용자입니다. 우리는 이러한 사용자가 타임시리즈라는 독특한 데이터를 사용함에
따라 맞딱드리게 되는 문제들에 대한 해결책을 제시합니다.

우리는 추가적으로 high-level API를 구현하여 비교적 초보자도 이미 구현된 모델을
사용할 수 있게 합니다.


.. toctree::
   :hidden:

   User Guide <user_guide/index>
   Tutorial <tutorials/index>
   API Reference <_autosummary/deep_time_series>
