import numpy as np
from fastapi import FastAPI, HTTPException

from core.config.constants import FREE_COOLING_THRESHOLD_C, HYBRID_THRESHOLD_C
from core.config.enums import CoolingMode
from core.schemas.control import ControlRequest, ControlResponse
from domain.controllers.rule_based import decide_cooling_mode, run_rule_based

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


@app.post("/control/rl")
def rl_control(req: ControlRequest) -> ControlResponse:
    """SAC 모델로 supply setpoint 추론.

    cooling_mode/free_cooling_ratio는 외기 온도 기반 derive (rule_based와 동일 기준).
    """
    missing = [k for k, v in {
        "zone_temp_c": req.zone_temp_c,
        "supply_setpoint_c": req.supply_setpoint_c,
        "cpu_utilization": req.cpu_utilization,
    }.items() if v is None]
    if missing:
        raise HTTPException(status_code=422, detail=f"RL 추론에 필요한 필드 누락: {missing}")

    hour = req.timestamp.hour if req.timestamp else 12
    obs = np.array([
        hour,
        req.outdoor_temp_c,
        req.outdoor_humidity,
        req.cpu_utilization,
        req.zone_temp_c,
        req.supply_setpoint_c,
        req.it_power_kw,
    ], dtype=np.float32)

    try:
        from domain.controllers.rl_inference import RLInference
        setpoint = RLInference.get().predict(obs)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=f"RL 모델 로드 실패: {e}")

    cooling_mode = decide_cooling_mode(req.outdoor_temp_c)
    if cooling_mode == CoolingMode.FREE_COOLING:
        ratio = 1.0
    elif cooling_mode == CoolingMode.HYBRID:
        ratio = 1 - (req.outdoor_temp_c - FREE_COOLING_THRESHOLD_C) / (HYBRID_THRESHOLD_C - FREE_COOLING_THRESHOLD_C)
    else:
        ratio = 0.0

    return ControlResponse(
        cooling_mode=cooling_mode.value,
        supply_air_temp_setpoint_c=setpoint,
        free_cooling_ratio=ratio,
    )
