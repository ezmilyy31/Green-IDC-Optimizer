import math
import sys
import os

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from fastapi import FastAPI, HTTPException

from core.config.constants import CRISIS_CONFIGS, SCENARIO_TEMP_PROFILES, WORKLOAD_PROFILE
from core.schemas.simulation import (
    HourlyResult,
    Simulate24hRequest,
    Simulate24hResponse,
    SimulationRequest,
    SimulationResponse,
)
from domain.thermodynamics.chiller import calculate_chiller_power_kw
from domain.thermodynamics.cooling_load import (
    AIR_SPECIFIC_HEAT_KJ_PER_KG_K,
    calculate_cooling_load_from_it_power_kw,
    calculate_m_air_for_servers,
)
from domain.thermodynamics.it_power import calculate_total_it_power_kw
from domain.thermodynamics.pue import calculate_pue

app = FastAPI(title="Simulation Service")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "simulation-service"}


@app.post("/api/v1/simulation/calculate", response_model=SimulationResponse)
def calculate(req: SimulationRequest) -> SimulationResponse:
    """
    열역학 모델 계산 엔드포인트 (단일 포인트).

    IT 전력과 외기 온도를 입력받아 냉각 부하, 칠러 전력, PUE를 반환한다.
    """
    cooling_load_kw = calculate_cooling_load_from_it_power_kw(req.it_power_kw)
    chiller_result = calculate_chiller_power_kw(cooling_load_kw, req.outdoor_temp_c)
    pue_result = calculate_pue(req.it_power_kw, chiller_result.chiller_power_kw)

    return SimulationResponse(
        cooling_load_kw=cooling_load_kw,
        cooling_mode=chiller_result.cooling_mode.value,
        cop=chiller_result.cop,
        chiller_power_kw=chiller_result.chiller_power_kw,
        pue=pue_result.pue,
        total_power_kw=pue_result.total_power_kw,
        it_power_kw=pue_result.it_power_kw,
        cooling_power_kw=pue_result.cooling_power_kw,
        other_power_kw=pue_result.other_power_kw,
    )


@app.post("/api/v1/simulation/24h", response_model=Simulate24hResponse)
def simulate_24h(req: Simulate24hRequest) -> Simulate24hResponse:
    """
    24시간 열역학 시뮬레이션 엔드포인트.

    시나리오별 외기 온도 프로파일과 시간대별 워크로드를 기반으로
    매 시간(0~23시) IT 전력 → 냉각 부하 → 칠러 전력 → PUE를 계산한다.

    - 외기 온도: sin 곡선 (새벽 4시 최저, 오후 4시 최고)
    - 워크로드: 시간대별 비율 × 기준 사용률 (WORKLOAD_PROFILE)
    - 위기 시나리오: util_multiplier로 부하 증폭, outdoor_override로 외기 고정, chiller_ratio로 칠러 용량 조절
    """
    if req.scenario_name not in SCENARIO_TEMP_PROFILES:
        raise HTTPException(
            status_code=422,
            detail=f"알 수 없는 시나리오: '{req.scenario_name}'. "
                   f"선택지: {list(SCENARIO_TEMP_PROFILES.keys())}",
        )
    if req.crisis not in CRISIS_CONFIGS:
        raise HTTPException(
            status_code=422,
            detail=f"알 수 없는 위기 시나리오: '{req.crisis}'. "
                   f"선택지: {[k for k in CRISIS_CONFIGS if k is not None]}",
        )

    profile = SCENARIO_TEMP_PROFILES[req.scenario_name]
    cfg = CRISIS_CONFIGS[req.crisis]
    m_air = calculate_m_air_for_servers(req.num_cpu + req.num_gpu)

    hourly: list[HourlyResult] = []
    for hour in range(24):
        # 외기 온도: sin 곡선 (새벽 4시 최저, 오후 4시 최고)
        raw_outdoor = profile["base"] + profile["amplitude"] * math.sin(
            math.radians((hour - 4) * 15)
        )
        outdoor_temp = cfg["outdoor_override"] if cfg["outdoor_override"] else raw_outdoor

        # IT 전력
        util = min(1.0, req.base_util * WORKLOAD_PROFILE[hour] / 0.6 * cfg["util_multiplier"])
        it_power_kw = calculate_total_it_power_kw(util, req.num_cpu, req.num_gpu)

        # 냉각 부하
        cooling_load_kw = calculate_cooling_load_from_it_power_kw(it_power_kw)

        # 칠러 전력 (위기 시 chiller_ratio로 용량 축소)
        chiller = calculate_chiller_power_kw(cooling_load_kw, outdoor_temp)
        actual_chiller_power = chiller.chiller_power_kw * cfg["chiller_ratio"]

        # PUE
        pue_result = calculate_pue(it_power_kw, actual_chiller_power)

        # 환기 온도: 냉각 부하에서 온도차 역산 + 미충족 냉각분 온도 상승
        unmet_cooling = cooling_load_kw * (1.0 - cfg["chiller_ratio"])
        delta_t = cooling_load_kw / (m_air * AIR_SPECIFIC_HEAT_KJ_PER_KG_K)
        delta_t_unmet = unmet_cooling / (m_air * AIR_SPECIFIC_HEAT_KJ_PER_KG_K)
        return_temp_c = req.supply_temp_c + delta_t + delta_t_unmet

        hourly.append(HourlyResult(
            hour=hour,
            outdoor_temp_c=round(outdoor_temp, 1),
            cpu_utilization=round(util, 3),
            it_power_kw=round(it_power_kw, 1),
            cooling_load_kw=round(cooling_load_kw, 1),
            chiller_power_kw=round(actual_chiller_power, 1),
            total_power_kw=round(pue_result.total_power_kw, 1),
            return_temp_c=round(return_temp_c, 1),
            cop=round(chiller.cop, 2),
            pue=round(pue_result.pue, 3),
            cooling_mode=chiller.cooling_mode.value,
        ))

    return Simulate24hResponse(hourly=hourly)
