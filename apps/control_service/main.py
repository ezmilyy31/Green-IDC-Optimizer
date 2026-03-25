from fastapi import FastAPI
from core.schemas.control import ControlRequest, ControlResponse
from domain.controllers.rule_based import run_rule_based

app = FastAPI(title="Control Service")

"""
POST /api/v1/control/optimize → 현재 최적 제어 결과 (rule_based, 추후 RL로 교체)
POST /control/rule-based      → Rule-based 제어 결과
POST /control/rl              → RL 에이전트 결과 (Week 4)
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
    # TODO: Week 4 RL 에이전트 연동, 임의값 넣어 둠
    return ControlResponse(
        cooling_mode = "hybrid",
        supply_air_temp_setpoint_c=20.0,
        free_cooling_ratio=0.5
    )