"""RL 환경 인터페이스 스키마 - Sinergym datacenter_dx-mixed-continuous-stochastic-v1 환경 기반"""
# obs/action 인덱스 매핑 정의

from pydantic import BaseModel, Field

# Sinergym obs 37개 중 래퍼가 사용할 인덱스 매핑
# sinergym이 제공하는 observation 변수 중 필요한 것만 골라서 설정
OBS_INDEX = {
    "month":                        0,
    "hour":                         2,
    "outdoor_temperature":          3,
    "outdoor_humidity":             4,
    "east_zone_air_temperature":    9,
    "west_zone_air_temperature":    10,
    "cooling_setpoint":             15,
    "cpu_loading_fraction":         26,
    "HVAC_electricity_demand_rate": 36,
}

# 래퍼가 agent에게 넘기는 filtered obs 순서
FILTERED_OBS_KEYS = list(OBS_INDEX.keys())

# 대시보드, API에서 RL 상태를 JSON으로 주고받을 때 쓰는 형식(RL 엔드포인트)
class RLState(BaseModel):
    # Agent가 관측하는 상태

    month: float = Field(..., ge=1, le=12, description="월 (1~12)")
    hour: float = Field(..., ge=0, le=23, description="시 (0~23)")
    outdoor_temperature: float = Field(..., description="외기 온도 (°C)")
    outdoor_humidity: float = Field(..., ge=0.0, le=100.0, description="외기 습도 (%)")
    east_zone_air_temperature: float = Field(..., description="East 서버실 온도 (°C)")
    west_zone_air_temperature: float = Field(..., description="West 서버실 온도 (°C)")
    cooling_setpoint: float = Field(..., ge=20.0, le=30.0, description="냉각 설정 온도 (°C)")
    cpu_loading_fraction: float = Field(..., ge=0.0, le=1.0, description="CPU 부하율 (0~1)")
    HVAC_electricity_demand_rate: float = Field(..., ge=0.0, description="전체 HVAC 전력 (W)")

# 실험에서 정해지는 action이 하나 
class RLAction(BaseModel):
    # Agent가 결정하는 행동

    cooling_setpoint: float = Field(
        ..., ge=20.0, le=30.0,
        description="냉각 설정 온도 (°C) — Sinergym action space [20, 30]",
    )

class RLStepResult(BaseModel):
    # 환경 step 결과
    
    state: RLState
    reward: float
    terminated: bool
    truncated: bool
    energy_term: float = Field(..., description="에너지 소비 패널티")
    comfort_term: float = Field(..., description="온도 위반 패널티")
    total_power_demand: float = Field(..., description="전체 전력 (W)")
    total_temperature_violation: float = Field(..., description="온도 위반량")