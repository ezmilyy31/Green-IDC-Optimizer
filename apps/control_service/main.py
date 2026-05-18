import numpy as np
from fastapi import FastAPI, HTTPException

from core.config.constants import WET_BULB_FREE_THRESHOLD_C, WET_BULB_HYBRID_THRESHOLD_C
from core.config.enums import CoolingMode
from core.schemas.control import ControlRequest, ControlResponse
from domain.controllers.rule_based import decide_cooling_mode, run_rule_based
from domain.thermodynamics.chiller import calculate_wet_bulb_c

app = FastAPI(title="Control Service")

"""
POST /api/v1/control/optimize → 현재 최적 제어 결과 (rule_based, 추후 RL로 교체)
POST /control/rule-based      → Rule-based 제어 결과
POST /control/rl              → RL(SAC) 에이전트 결과
POST /control/mpc             → MPC 최적 설정값 (선택)
POST /control/scenario        → 위기 시나리오 주입
GET  /control/status          → 현재 제어 상태
GET  /esg/summary             → 탄소·WUE 요약
"""

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "control-service"}

@app.post("/api/v1/control/optimize") # 추후 가장 최적화한 방식이 들어갈 부분, 현재는 구현된 rule_based
def optimize(req: ControlRequest) -> ControlResponse:
    result = run_rule_based(
        outdoor_temp_c=req.outdoor_temp_c,
        outdoor_humidity =req.outdoor_humidity,
        it_power_kw=req.it_power_kw,
    )
    return ControlResponse(
        cooling_mode=result.cooling_mode.value,
        supply_air_temp_setpoint_c=result.supply_air_temp_setpoint_c,
        free_cooling_ratio=result.free_cooling_ratio,
    )

@app.post("/control/rule-based")
def rule_based(req: ControlRequest) -> ControlResponse:
    result = run_rule_based(
        outdoor_temp_c = req.outdoor_temp_c,
        outdoor_humidity = req.outdoor_humidity,
        it_power_kw= req.it_power_kw
    )
    return ControlResponse(
        cooling_mode = result.cooling_mode.value,
        supply_air_temp_setpoint_c = result.supply_air_temp_setpoint_c,
        free_cooling_ratio = result.free_cooling_ratio
    )


def _build_obs(req: ControlRequest) -> np.ndarray:
    """ControlRequest → IDCEnv 9-dim obs 변환. RL 필수 필드 누락 시 422 raise."""
    missing = [k for k, v in {
        "zone_temp_c": req.zone_temp_c,
        "supply_setpoint_c": req.supply_setpoint_c,
        "cpu_utilization": req.cpu_utilization,
    }.items() if v is None]
    if missing:
        raise HTTPException(status_code=422, detail=f"RL 추론에 필요한 필드 누락: {missing}")

    from domain.thermodynamics.chiller import calculate_wet_bulb_c
    hour = req.timestamp.hour if req.timestamp else 12
    wet_bulb = calculate_wet_bulb_c(req.outdoor_temp_c, req.outdoor_humidity)
    # obs 순서: IDCEnv._get_obs()와 동일
    # [hour, outdoor_temp, outdoor_trend, humidity, cpu_util, zone_temp, supply_temp, it_power, wet_bulb]
    return np.array([
        hour,
        req.outdoor_temp_c,
        req.outdoor_temp_trend_c_per_s,
        req.outdoor_humidity,
        req.cpu_utilization,
        req.zone_temp_c,
        req.supply_setpoint_c,
        req.it_power_kw,
        wet_bulb,
    ], dtype=np.float32)


def _derive_cooling_metadata(
    outdoor_temp_c: float,
    outdoor_humidity_pct: float,
) -> tuple[CoolingMode, float]:
    """외기 + 습도 기반 cooling_mode + free_cooling_ratio 계산 (wet-bulb 기준).

    rule_based / chiller / free_cooling 모듈과 환경 일관성 유지.
    """
    cooling_mode = decide_cooling_mode(outdoor_temp_c, outdoor_humidity_pct)
    if cooling_mode == CoolingMode.FREE_COOLING:
        ratio = 1.0
    elif cooling_mode == CoolingMode.HYBRID:
        wet_bulb = calculate_wet_bulb_c(outdoor_temp_c, outdoor_humidity_pct)
        ratio = 1.0 - (wet_bulb - WET_BULB_FREE_THRESHOLD_C) / (
            WET_BULB_HYBRID_THRESHOLD_C - WET_BULB_FREE_THRESHOLD_C
        )
    else:
        ratio = 0.0
    return cooling_mode, ratio


@app.post("/control/rl")
def rl_control(req: ControlRequest) -> ControlResponse:
    """효율 우선 best 모델로 supply setpoint 추론 (PUE 최우수).

    safe fallback 자동 적용 (zone > 26.5°C 시 T_SUPPLY_MIN 강제).
    """
    obs = _build_obs(req)
    try:
        from domain.controllers.rl_inference import predict_best
        setpoint = predict_best(obs)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=f"RL 모델 로드 실패: {e}")

    cooling_mode, ratio = _derive_cooling_metadata(req.outdoor_temp_c, req.outdoor_humidity)
    return ControlResponse(
        cooling_mode=cooling_mode.value,
        supply_air_temp_setpoint_c=setpoint,
        free_cooling_ratio=ratio,
    )


