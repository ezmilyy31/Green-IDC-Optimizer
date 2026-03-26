from fastapi import FastAPI
from core.schemas.forecast import *
# from domain.forecasting.intervals import 
# from domain.forecasting.lgbm_model.py import 
# from domain.forecasting.lstm_model.py import 

app = FastAPI(
    title="Forecast Service",
    description="IT 부하 및 냉각 수요 예측 API",
)

"""
POST /api/v1/forecast

프로젝트 명세 '2. 최종 결과물' 내용
2. IT 부하/냉각 수요 예측 API
    - 입력: POST /api/v1/forecast 호출 시 예측 기간을 JSON으로 전달
    - 출력: 24~168시간 예측값과 90% 신뢰구간이 JSON으로 반환된다.

프로젝트 명세 '4. 기능요구사항' 내용
1. **IT 부하 예측 모듈 [필수]**
    - 최소 2가지 이상의 예측 모델을 구현하고 비교해야 한다.
    - 24시간 ahead 예측 MAPE가 5% 이내여야 한다.
    - 168시간 ahead 예측 MAPE가 8% 이내여야 한다.
    - 90% 예측 구간 Coverage가 85% 이상이어야 한다.
2. **냉각 수요 예측 모듈 [필수] ← 이거 할 때 6번 기반으로 해야.**
    - IT 부하와 외기 조건으로 냉각 전력을 예측해야 한다.
    - 예측 오차 nMAE가 10% 이내여야 한다.
    - 시간대별 최적 냉각 모드를 예측해야 한다.
    - 외기 15도 이하 시 자연공조 전환 로직을 구현해야 한다.
    - 냉각 방식별 에너지 소비량을 비교 분석해야 한다.
"""

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "forecast-service"}


