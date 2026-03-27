"""
Forecast Service orchestration layer.

이 파일의 책임
--------------
- API 요청(payload)을 받아 예측 흐름을 조합한다.
- target(it_load / cooling_demand / both)에 따라 필요한 모델을 선택한다.
- 요청의 model_type(lgbm / lstm)에 따라 적절한 모델을 고른다.
- feature frame 생성 함수를 호출한다.
- 모델 추론 결과를 ForecastResponse 형태로 조립한다.
- cooling mode를 rule-based로 판정한다.
- 필요 시 prediction interval을 결과에 포함한다.
"""