"""
Forecast Service model loader.

이 파일의 책임
--------------
- FastAPI startup시기 호출되어, IT Load / Cooling Demand용 LGBM, LSTM 모델을 모두 로드한다.
- 외기 예보, feature 기본값, interval 설정 등 부가 리소스를 로드한다.
- services/forecast.py에서 바로 사용할 수 있는 model_bundle 형태로 반환한다.
"""