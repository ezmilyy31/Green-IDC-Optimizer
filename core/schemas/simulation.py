"""
열역학 시뮬레이션 서비스 스키마

simulation_service 엔드포인트별 Request / Response 구조.
대시보드 및 다른 서비스에서 import하여 재사용한다.

엔드포인트:
  POST /api/v1/simulation/calculate  — 단일 포인트 계산
  POST /api/v1/simulation/24h        — 24시간 시뮬레이션
"""

from pydantic import BaseModel, Field


class SimulationRequest(BaseModel):
    """열역학 계산 요청 (단일 포인트)"""

    outdoor_temp_c: float = Field(..., description="외기 온도 (°C)")
    it_power_kw: float = Field(..., gt=0, description="IT 장비 전력 소비량 (kW)")


class SimulationResponse(BaseModel):
    """열역학 계산 응답 (단일 포인트)"""

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


# ── 24시간 시뮬레이션 ─────────────────────────────────────────────────────────

class Simulate24hRequest(BaseModel):
    """24시간 열역학 시뮬레이션 요청"""

    scenario_name: str = Field(
        ...,
        description="온도 프로파일 시나리오 이름. "
                    "선택지: '여름 (Summer)', '봄/가을 (Spring)', '겨울 (Winter)'",
    )
    num_cpu: int = Field(..., gt=0, description="CPU 서버 대수")
    num_gpu: int = Field(..., ge=0, description="GPU 서버 대수")
    base_util: float = Field(..., ge=0.0, le=1.0, description="기준 CPU 사용률 (0.0~1.0)")
    supply_temp_c: float = Field(..., description="CRAH 공급 온도 설정값 (°C)")
    crisis: str | None = Field(
        None,
        description="위기 시나리오 키. "
                    "None(정상) / 'server_surge' / 'chiller_failure' / 'heat_wave'",
    )


class HourlyResult(BaseModel):
    """시간별 시뮬레이션 결과 (1개 행)"""

    hour: int = Field(..., description="시각 (0~23)")
    outdoor_temp_c: float = Field(..., description="외기 온도 (°C)")
    cpu_utilization: float = Field(..., description="CPU 사용률 (0.0~1.0)")
    it_power_kw: float = Field(..., description="IT 전력 소비량 (kW)")
    cooling_load_kw: float = Field(..., description="냉각 부하 (kW)")
    chiller_power_kw: float = Field(..., description="칠러 전력 소비량 (kW)")
    total_power_kw: float = Field(..., description="데이터센터 총 전력 (kW)")
    return_temp_c: float = Field(..., description="환기 온도 — 서버 배기 온도 (°C)")
    cop: float = Field(..., description="칠러 COP (무차원)")
    pue: float = Field(..., description="PUE")
    cooling_mode: str = Field(..., description="냉각 모드: chiller / free_cooling / hybrid")


class Simulate24hResponse(BaseModel):
    """24시간 열역학 시뮬레이션 응답"""

    hourly: list[HourlyResult] = Field(..., description="시간별 결과 (24개)")
