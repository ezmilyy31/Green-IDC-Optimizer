from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum
from core.config.enums import CoolingMode, PredictionTarget, ModelType

"""
Forecast Service에서 사용하는 Pydantic 스키마 정의 파일.
"""

# Request ##############################################

"""
`/api/v1/forecast`에서 예측 요청 전체를 받는 메인 request schema
"""
class ForecastRequest(BaseModel):
    prediction_target: PredictionTarget = PredictionTarget.BOTH
    model_type: ModelType = ModelType.LGBM
    forecast_horizon_hours: int = Field(..., ge=1, le=168)
    current_timestamp: datetime | None = None
    include_prediction_interval: bool = True


# Response #############################################

"""
한 시점의 예측 결과를 표현하는 schema
"""
class ForecastPoint(BaseModel):
    timestamp: datetime
    predicted_it_load_kw: float | None = None
    predicted_cooling_load_kw: float | None = None
    cooling_mode: CoolingMode | None = None
    lower_bound_it_load_kw: float | None = None
    upper_bound_it_load_kw: float | None = None
    lower_bound_cooling_load_kw: float | None = None
    upper_bound_cooling_load_kw: float | None = None

"""
`/api/v1/forecast`의 최종 response schema
"""
class ForecastResponse(BaseModel):
    prediction_target: PredictionTarget
    model_type_used: ModelType
    generated_at: datetime
    horizon_hours: int
    predictions: list[ForecastPoint]

"""
예외 처리용 공통 응답 schema
"""
class ErrorResponse(BaseModel):
    error_code: str
    message: str
    detail: str | None = None

class HealthResponse(BaseModel):
    status: str
    service: str
    model_ready: bool
    model_load_error: str | None = None