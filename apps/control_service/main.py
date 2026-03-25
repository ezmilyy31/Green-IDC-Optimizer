from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime
from domain.controllers.rule_based import run_rule_based

app = FastAPI(title="Control Service")

"""
POST /control/rule-based    → Rule-based 제어 결과
POST /control/mpc           → MPC 최적 설정값
POST /control/rl            → RL 에이전트 결과
POST /control/scenario      → 시나리오 주입
GET  /control/status        → 현재 제어 상태
GET  /esg/summary           → 탄소·WUE 요약
"""

class ControlRequest(BaseModel):
    outdoor_temp_c: float # 외기 온도, 냉각 모드 결정
    outdoor_humidity_pct: float = 50.0 # 외기 습도, free cooling 효율 보정용
    it_power_kw: float # 현재 IT 전력, 냉각 부하 계산용
    timestamp: datetime | None = None # 시각 - RL에서 time_of_day 특성
    #server_inlet_temp_c: float, 서버 입구 온도, PID가 현재 온도로 받아야 하는 값
    #server_outlet_temp_c: float, 서버 출구 온도, 냉각 부하 계산시 온도 변화량 계산 

class ControlResponse(BaseModel): # 대시 보드에 들어가야 하는 값
    cooling_mode: str # 냉각 방식
    supply_air_temp_setpoint_c: float # 공조기 목표 온도
    free_cooling_ratio: float # ESG 지표 계산 
    expected_pue: float = 1.35 # TODO: simulation_service 연동 후 교체, 예상 PUE
    # chw_flow_setpoint_kg_s: float, 냉각수 유량, simulation_service 연동 시 추가


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "control-service"}

@app.post("/api/v1/control/optimize") # 추후 가장 최적화한 방식이 들어갈 부분, 현재는 구현된 rule_based
def optimize(req: ControlRequest) -> ControlResponse:
    result = run_rule_based(
        outdoor_temp_c=req.outdoor_temp_c,
        outdoor_humidity_pct=req.outdoor_humidity_pct,
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
        outdoor_humidity_pct = req.outdoor_humidity_pct,
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