from pydantic import BaseModel
from datetime import datetime


class ControlRequest(BaseModel):
    outdoor_temp_c: float           # 외기 온도 (°C), 냉각 모드 결정
    outdoor_humidity: float = 50.0  # 외기 습도 (%), free cooling 효율 보정
    it_power_kw: float              # 현재 IT 전력 (kW), 냉각 부하 계산
    timestamp: datetime | None = None  # 시각, RL에서 time_of_day 특성
    # PID 연동 시 추가
    # server_inlet_temp_c: float     서버 입구 온도, PID가 현재 온도로 받아야 하는 값
    # server_outlet_temp_c: float   서버 출구 온도, 냉각 부하 계산시 온도 변화량 계산 


class ControlResponse(BaseModel):
    cooling_mode: str                        # 냉각 방식 (free_cooling/hybrid/chiller)
    supply_air_temp_setpoint_c: float        # 공조기 목표 온도 (°C)
    free_cooling_ratio: float                # 자연냉각 비율 (0.0~1.0), ESG 지표 계산
    expected_pue: float = 1.35              # 예상 PUE, TODO: simulation_service 연동 후 교체
    # simulation_service 연동 시 추가
    # chw_flow_setpoint_kg_s: float 냉각수 유량
    # expected_zone_temp_c: float 서버실 예상 온도

