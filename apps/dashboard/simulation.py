"""시뮬레이션 및 ESG 계산 로직"""

import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from apps.dashboard.api_client import simulate_24h as _api_simulate_24h
from apps.dashboard.constants import (
    CARBON_FACTOR_TCO2_PER_MWH,
    CRISIS_CONFIGS,
    ELECTRICITY_COST_KRW_PER_KWH,
    NUM_RACKS,
    SCENARIO_TEMP_PROFILES,
    TEMP_WARNING_THRESHOLD_C,
    WORKLOAD_PROFILE,
)

# Simulation Service 미기동 시 fallback용 직접 계산 모듈
from domain.thermodynamics.chiller import calculate_chiller_power_kw
from domain.thermodynamics.cooling_load import (
    AIR_SPECIFIC_HEAT_KJ_PER_KG_K,
    calculate_cooling_load_from_it_power_kw,
    calculate_m_air_for_servers,
)
from domain.thermodynamics.it_power import calculate_total_it_power_kw
from domain.thermodynamics.pue import calculate_pue


def _hourly_to_df(hourly: list[dict]) -> pd.DataFrame:
    """API 응답의 hourly 리스트를 대시보드용 DataFrame으로 변환한다."""
    return pd.DataFrame([
        {
            "시간":           f"{r['hour']:02d}:00",
            "외기온도 (°C)":  r["outdoor_temp_c"],
            "CPU 사용률 (%)": round(r["cpu_utilization"] * 100, 1),
            "IT 전력 (kW)":   r["it_power_kw"],
            "냉각 부하 (kW)": r["cooling_load_kw"],
            "칠러 전력 (kW)": r["chiller_power_kw"],
            "총 전력 (kW)":   r["total_power_kw"],
            "환기 온도 (°C)": r["return_temp_c"],
            "COP":            r["cop"],
            "PUE":            r["pue"],
            "냉각 모드":      r["cooling_mode"],
        }
        for r in hourly
    ])


def _run_simulation_local(
    scenario_name: str,
    num_cpu: int,
    num_gpu: int,
    base_util: float,
    supply_temp_c: float,
    crisis: str | None,
) -> pd.DataFrame:
    """Simulation Service 미기동 시 domain 함수로 직접 계산하는 fallback."""
    profile = SCENARIO_TEMP_PROFILES[scenario_name]
    m_air   = calculate_m_air_for_servers(num_cpu + num_gpu)
    cfg     = CRISIS_CONFIGS[crisis]

    rows = []
    for hour in range(24):
        raw_outdoor  = profile["base"] + profile["amplitude"] * math.sin(math.radians((hour - 4) * 15))
        outdoor_temp = cfg["outdoor_override"] if cfg["outdoor_override"] else raw_outdoor

        util         = min(1.0, base_util * WORKLOAD_PROFILE[hour] / 0.6 * cfg["util_multiplier"])
        it_power_kw  = calculate_total_it_power_kw(util, num_cpu, num_gpu)
        cooling_load = calculate_cooling_load_from_it_power_kw(it_power_kw)

        chiller              = calculate_chiller_power_kw(cooling_load, outdoor_temp)
        actual_chiller_power = chiller.chiller_power_kw * cfg["chiller_ratio"]

        unmet_cooling = cooling_load * (1.0 - cfg["chiller_ratio"])
        delta_t_unmet = unmet_cooling / (m_air * AIR_SPECIFIC_HEAT_KJ_PER_KG_K)

        pue_result  = calculate_pue(it_power_kw, actual_chiller_power)
        delta_t     = cooling_load / (m_air * AIR_SPECIFIC_HEAT_KJ_PER_KG_K)
        return_temp = supply_temp_c + delta_t + delta_t_unmet

        rows.append({
            "시간":           f"{hour:02d}:00",
            "외기온도 (°C)":  round(outdoor_temp, 1),
            "CPU 사용률 (%)": round(util * 100, 1),
            "IT 전력 (kW)":   round(it_power_kw, 1),
            "냉각 부하 (kW)": round(cooling_load, 1),
            "칠러 전력 (kW)": round(actual_chiller_power, 1),
            "총 전력 (kW)":   round(pue_result.total_power_kw, 1),
            "환기 온도 (°C)": round(return_temp, 1),
            "COP":            round(chiller.cop, 2),
            "PUE":            round(pue_result.pue, 3),
            "냉각 모드":      chiller.cooling_mode.value,
        })

    return pd.DataFrame(rows)


def run_simulation(
    scenario_name: str,
    num_cpu: int,
    num_gpu: int,
    base_util: float,
    supply_temp_c: float,
    crisis: str | None,
) -> pd.DataFrame:
    """24시간 열역학 시뮬레이션을 실행하고 결과 DataFrame을 반환한다.

    Simulation Service(port 8003)가 기동 중이면 REST API를 호출하고,
    응답 실패 시 domain 함수로 직접 계산하는 fallback을 수행한다.
    """
    result = _api_simulate_24h(scenario_name, num_cpu, num_gpu, base_util, supply_temp_c, crisis)

    if "error" not in result and "hourly" in result:
        return _hourly_to_df(result["hourly"])

    return _run_simulation_local(scenario_name, num_cpu, num_gpu, base_util, supply_temp_c, crisis)


def calculate_esg(df: pd.DataFrame) -> dict:
    """시뮬레이션 결과 DataFrame으로 ESG 지표를 계산한다."""
    total_energy_kwh = df["총 전력 (kW)"].sum()
    it_energy_kwh    = df["IT 전력 (kW)"].sum()

    carbon_tco2 = total_energy_kwh * CARBON_FACTOR_TCO2_PER_MWH / 1000
    cost_krw    = total_energy_kwh * ELECTRICITY_COST_KRW_PER_KWH
    cue         = carbon_tco2 / it_energy_kwh if it_energy_kwh > 0 else 0.0  # kgCO₂/kWh

    return {
        "carbon_tco2_day":   round(carbon_tco2, 3),
        "carbon_tco2_month": round(carbon_tco2 * 30, 1),
        "cost_krw_day":      round(cost_krw / 1e6, 2),        # 만원
        "cost_krw_month":    round(cost_krw * 30 / 1e6, 1),   # 만원
        "cue":               round(cue * 1000, 4),             # tCO₂/kWh → kgCO₂/kWh
    }


def simulate_rack_temperatures(peak_return_temp: float) -> tuple[list, list, list]:
    """피크 환기 온도 기준으로 랙별 온도 분포를 시뮬레이션한다."""
    random.seed(42)
    temps  = [round(peak_return_temp + random.gauss(0, 1.5), 1) for _ in range(NUM_RACKS)]
    labels = [f"R{i+1:02d}" for i in range(NUM_RACKS)]
    from apps.dashboard.constants import CLR_DANGER, CLR_GOOD, CLR_WARN
    colors = [
        CLR_DANGER if t > TEMP_WARNING_THRESHOLD_C else
        CLR_WARN   if t > TEMP_WARNING_THRESHOLD_C - 2 else
        CLR_GOOD
        for t in temps
    ]
    return labels, temps, colors
