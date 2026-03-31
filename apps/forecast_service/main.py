from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request, status
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from core.schemas.forecast import ForecastRequest, ForecastResponse, ErrorResponse, HealthResponse
from .services.forecast import run_forecast
from .models.loader import load_model_bundle

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    앱 시작 시 1회 실행되는 lifecycle 함수.

    역할
    - 예측에 필요한 모델 번들을 미리 로드!!!!
    - app.state에 저장하여 각 요청에서 재사용 가능하게 함
    - 로드 실패 시 에러 정보를 저장
    """
    try:
        # 모델, 스케일러, 메타데이터 등을 한 번에 로드
        app.state.model_bundle = load_model_bundle()
        app.state.model_load_error = None
        print("[forecast-service] model bundle loaded successfully.")

    except Exception as exc:
        # 모델 로드 실패 시 None으로 두고 에러 메시지 저장
        app.state.model_bundle = None
        app.state.model_load_error = str(exc)
        print(f"[forecast-service] failed to load model bundle: {exc}")

    # 앱 실행
    yield

    # 앱 종료 시 정리
    app.state.model_bundle = None
    app.state.model_load_error = None
    print("[forecast-service] shutdown completed.")

app = FastAPI(
    title="Forecast Service",
    description="IT 부하 및 냉각 수요 예측 API",
    lifespan=lifespan,
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

@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health(request: Request) -> HealthResponse:
    ready = getattr(request.app.state, "model_bundle", None) is not None
    model_load_error = getattr(request.app.state, "model_load_error", None)

    return HealthResponse(
        status="ok",
        service="forecast-service",
        model_ready=ready,
        model_load_error=model_load_error,
    )

@app.post(
    "/api/v1/forecast",
    response_model=ForecastResponse, 
    responses={
        400: {"model": ErrorResponse}, # 잘못된 request인 경우
        500: {"model": ErrorResponse}, # 서버 내부 요류인 경우
        503: {"model": ErrorResponse}, # 모델 미준비 상태인 경우
    },
    tags=["Forecast"], # swagger 문서에서 Forecast 그룹으로 분류
)
def forecast(payload: ForecastRequest, request: Request) -> ForecastResponse:
    """
    예측 요청 엔드포인트.

    처리 흐름
    1. 모델 로드 여부 확인
    2. 요청 스키마 검증 완료된 payload 수신
    3. service 계층에 예측 처리 위임
    4. 예측 결과를 response schema로 반환
    """

    # FastAPI app.state에 저장되어 있는 모델 번들을 가져옴
    # 보통 startup 시점에 모델을 로드해서 app.state.model_bundle에 저장해 둠
    model_bundle = getattr(request.app.state, "model_bundle", None)
    model_load_error = getattr(request.app.state, "model_load_error", None)

    # 모델 번들이 없으면 아직 모델이 로드되지 않은 상태 
    if model_bundle is None:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=ErrorResponse(
                error_code="MODEL_NOT_READY",
                message="Forecast model is not loaded.",
                detail=model_load_error,
            ).model_dump(),
        )

    try:
        # 실제 예측 로직은 서비스 계층 함수(run_forecast)에 위임
        # payload: 사용자가 보낸 예측 요청 데이터
        # model_bundle: 미리 로드된 모델, 스케일러, 메타데이터 등
        result = run_forecast(model_bundle=model_bundle, request=payload)
        return result

    except ValueError as exc:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=ErrorResponse(
                error_code="INVALID_REQUEST",
                message="Invalid forecast request.",
                detail=str(exc),
            ).model_dump(),
        )

    except Exception as exc:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                error_code="FORECAST_FAILED",
                message="Unexpected error occurred during forecast.",
                detail=str(exc),
            ).model_dump(),
        )