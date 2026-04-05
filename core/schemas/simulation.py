"""
열역학 시뮬레이션 서비스 스키마

simulation_service (/api/v1/simulation/calculate) 의 Request / Response 구조.
대시보드 및 다른 서비스에서 import하여 재사용한다.
"""

from pydantic import BaseModel, Field


class SimulationRequest(BaseModel):
    """열역학 계산 요청"""

    outdoor_temp_c: float = Field(..., description="외기 온도 (°C)")
    it_power_kw: float = Field(..., gt=0, description="IT 장비 전력 소비량 (kW)")


class SimulationResponse(BaseModel):
    """열역학 계산 응답"""

    # 냉각 부하
    cooling_load_kw: float = Field(..., description="냉각 부하 Q (kW) — IT 발열 기준")

    # 칠러
    cooling_mode: str = Field(..., description="냉각 모드: chiller / free_cooling / hybrid")
    cop: float = Field(..., description="칠러 성능계수 COP (무차원)")
    chiller_power_kw: float = Field(..., description="칠러 전력 소비량 (kW)")

    # PUE
    pue: float = Field(..., description="PUE (Power Usage Effectiveness)")
    total_power_kw: float = Field(..., description="데이터센터 총 전력 (kW)")
    it_power_kw: float = Field(..., description="IT 장비 전력 (kW)")
    cooling_power_kw: float = Field(..., description="냉각 전력 (kW)")
    other_power_kw: float = Field(..., description="기타 전력 — UPS 손실, 조명 등 (kW)")
