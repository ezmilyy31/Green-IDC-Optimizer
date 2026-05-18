"""시뮬레이션 및 ESG 계산 로직"""

import math
import sys
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import streamlit as st

from apps.dashboard.constants import (
    CARBON_FACTOR_KG_PER_KWH,
    CARBON_FACTOR_TCO2_PER_MWH,
    CRISIS_CONFIGS,
    DEFAULT_HUMIDITY_PCT,
    ELECTRICITY_COST_KRW_PER_KWH,
    SCENARIO_TEMP_PROFILES,
    WORKLOAD_PROFILE,
)

# 합성 1년치 데이터 — RL 학습/A/B 비교/Forecast 학습이 사용하는 동일 데이터셋과 일치시켜
# 화면 간 정합성을 확보한다 (5분 간격, 105120 step).
_PARQUET_PATH = Path(__file__).resolve().parents[2] / "data" / "weather" / "synthetic_idc_1year_noisy.parquet"

# 시즌 시나리오 → parquet 대표 24h 시작 시각 (2019년 데이터)
_SEASON_START: dict[str, str] = {
    "여름 (Summer)":         "2019-07-15 00:00:00",
    "봄/가을 (Spring/Fall)": "2019-10-15 00:00:00",
    "겨울 (Winter)":         "2019-01-15 00:00:00",
}

# parquet 분포의 cpu_utilization 평균 (정규화 기준)
_PARQUET_UTIL_MEAN = 0.47
# 사이드바 cpu_util 슬라이더 기본값 (이 값일 때 parquet util 그대로 사용)
_BASE_UTIL_DEFAULT = 0.60

# 도메인 열역학 모듈
from domain.thermodynamics.chiller import calculate_chiller_power_kw
from domain.thermodynamics.cooling_load import (
    AIR_SPECIFIC_HEAT_KJ_PER_KG_K,
    calculate_cooling_load_from_airflow_kw,
    calculate_cooling_load_from_it_power_kw,
    calculate_m_air_for_servers,
)
from domain.thermodynamics.free_cooling import calculate_free_cooling
from domain.thermodynamics.it_power import calculate_total_it_power_kw
from domain.thermodynamics.pue import calculate_pue

# parquet 생성 로직(data/data_pipeline.py)에서 팬 전력 계산에 사용하는 공기유량 기반 냉각 부하.
# supply=18°C / return=27°C 고정으로 산출 (data_pipeline.py:402-404).
_FAN_AIRFLOW_SUPPLY_C = 18.0
_FAN_AIRFLOW_RETURN_C = 27.0


@lru_cache(maxsize=1)
def _load_parquet() -> pd.DataFrame:
    """1년치 합성 데이터 1회 로드 후 캐시."""
    df = pd.read_parquet(_PARQUET_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def _slice_24h(scenario_name: str) -> pd.DataFrame:
    """시즌 시나리오에 해당하는 24시간 슬라이스를 시간 단위로 리샘플."""
    df = _load_parquet()
    start = pd.Timestamp(_SEASON_START.get(scenario_name, _SEASON_START["여름 (Summer)"]))
    end   = start + pd.Timedelta(hours=24)
    window = df[(df["timestamp"] >= start) & (df["timestamp"] < end)].copy()
    window = window.set_index("timestamp")
    hourly = window.resample("1h").agg({
        "outside_temp_c":       "mean",
        "outside_humidity_pct": "mean",
        "cpu_utilization":      "mean",
    }).reset_index()
    return hourly


def _compute_row(
    *,
    hour_label: str,
    outdoor_temp: float,
    humidity: float,
    util: float,
    num_cpu: int,
    num_gpu: int,
    supply_temp_c: float,
    cfg: dict,
    m_air: float,
) -> dict:
    """단일 시간 슬라이스에 대해 도메인 열역학으로 IT/냉각/PUE 계산.

    냉각 전력 = 칠러 + CRAH 팬. 팬 전력은 free_cooling 효율에 따라 비율이 바뀌므로
    `calculate_free_cooling`으로 산출. PUE는 (IT + 칠러 + 팬 + other)/IT 로 정의되며
    이는 data_pipeline.py의 parquet PUE 정의와 일치한다.
    """
    it_power_kw       = calculate_total_it_power_kw(util, num_cpu, num_gpu)
    cooling_load      = calculate_cooling_load_from_it_power_kw(it_power_kw)
    # 팬 전력은 공기유량 기반 냉각 부하로 계산 (parquet 생성 로직과 일치)
    cooling_load_air  = calculate_cooling_load_from_airflow_kw(m_air, _FAN_AIRFLOW_SUPPLY_C, _FAN_AIRFLOW_RETURN_C)
    chiller           = calculate_chiller_power_kw(cooling_load, outdoor_temp, supply_temp_c, humidity)
    fc                = calculate_free_cooling(cooling_load_air, outdoor_temp, humidity, supply_temp_c)

    actual_chiller_power = chiller.chiller_power_kw * cfg["chiller_ratio"]
    fan_power_kw         = fc.fan_power_kw
    cooling_power_kw     = actual_chiller_power + fan_power_kw

    unmet_cooling = cooling_load * (1.0 - cfg["chiller_ratio"])
    delta_t       = cooling_load  / (m_air * AIR_SPECIFIC_HEAT_KJ_PER_KG_K)
    delta_t_unmet = unmet_cooling / (m_air * AIR_SPECIFIC_HEAT_KJ_PER_KG_K)
    return_temp   = supply_temp_c + delta_t + delta_t_unmet

    pue_result = calculate_pue(it_power_kw, cooling_power_kw)
    return {
        "시간":           hour_label,
        "외기온도 (°C)":  round(outdoor_temp, 1),
        "외기 습도 (%)":  round(humidity, 1),
        "CPU 사용률 (%)": round(util * 100, 1),
        "IT 전력 (kW)":   round(it_power_kw, 1),
        "냉각 부하 (kW)": round(cooling_load, 1),
        "칠러 전력 (kW)": round(actual_chiller_power, 1),
        "팬 전력 (kW)":   round(fan_power_kw, 1),
        "총 전력 (kW)":   round(pue_result.total_power_kw, 1),
        "환기 온도 (°C)": round(return_temp, 1),
        "COP":            round(chiller.cop, 2),
        "PUE":            round(pue_result.pue, 3),
        "냉각 모드":      chiller.cooling_mode.value,
    }


def _run_simulation_from_parquet(
    scenario_name: str,
    num_cpu: int,
    num_gpu: int,
    base_util: float,
    supply_temp_c: float,
    crisis: str | None,
) -> pd.DataFrame:
    """1년치 합성 parquet에서 시즌 대표 24h를 슬라이스해 시뮬레이션.

    외기/습도/CPU 패턴은 parquet에서, 서버 구성·공급 온도·위기 배율은 사이드바에서.
    사이드바의 base_util 슬라이더는 parquet util에 대한 글로벌 배율로 작동
    (슬라이더 기본 60% 일 때 parquet 값 그대로 적용됨).
    """
    hourly = _slice_24h(scenario_name)
    cfg    = CRISIS_CONFIGS[crisis]
    m_air  = calculate_m_air_for_servers(num_cpu + num_gpu)
    util_scale = base_util / _BASE_UTIL_DEFAULT  # 슬라이더 기본값 기준 보정 배율

    rows = []
    for _, r in hourly.iterrows():
        outdoor_temp = cfg["outdoor_override"] if cfg["outdoor_override"] else float(r["outside_temp_c"])
        humidity     = float(r["outside_humidity_pct"])
        util         = min(1.0, float(r["cpu_utilization"]) * util_scale * cfg["util_multiplier"])

        rows.append(_compute_row(
            hour_label=r["timestamp"].strftime("%H:%M"),
            outdoor_temp=outdoor_temp,
            humidity=humidity,
            util=util,
            num_cpu=num_cpu,
            num_gpu=num_gpu,
            supply_temp_c=supply_temp_c,
            cfg=cfg,
            m_air=m_air,
        ))
    return pd.DataFrame(rows)


def _run_simulation_sin_fallback(
    scenario_name: str,
    num_cpu: int,
    num_gpu: int,
    base_util: float,
    supply_temp_c: float,
    crisis: str | None,
) -> pd.DataFrame:
    """parquet 미존재 시에만 사용되는 sin 곡선 폴백 (구버전 로직 보존)."""
    profile = SCENARIO_TEMP_PROFILES[scenario_name]
    m_air   = calculate_m_air_for_servers(num_cpu + num_gpu)
    cfg     = CRISIS_CONFIGS[crisis]

    rows = []
    for hour in range(24):
        raw_outdoor  = profile["base"] + profile["amplitude"] * math.sin(math.radians((hour - 4) * 15))
        outdoor_temp = cfg["outdoor_override"] if cfg["outdoor_override"] else raw_outdoor
        util         = min(1.0, base_util * WORKLOAD_PROFILE[hour] / 0.6 * cfg["util_multiplier"])
        rows.append(_compute_row(
            hour_label=f"{hour:02d}:00",
            outdoor_temp=outdoor_temp,
            humidity=DEFAULT_HUMIDITY_PCT,
            util=util,
            num_cpu=num_cpu,
            num_gpu=num_gpu,
            supply_temp_c=supply_temp_c,
            cfg=cfg,
            m_air=m_air,
        ))
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def run_simulation(
    scenario_name: str,
    num_cpu: int,
    num_gpu: int,
    base_util: float,
    supply_temp_c: float,
    crisis: str | None,
) -> pd.DataFrame:
    """24시간 시뮬레이션 결과 DataFrame을 반환.

    1년치 noisy parquet(RL/A-B/Forecast가 공유하는 동일 데이터셋)에서 시즌 대표
    24h를 슬라이스해 도메인 열역학으로 PUE/COP를 계산한다. parquet 미존재 시
    sin 곡선 폴백.

    @st.cache_data: 동일 입력으로 페이지 전환·재실행 시 즉시 캐시 반환 (crisis_mode
    토글로 base와 비교 시 두 번 호출되는 비용도 캐시로 흡수).
    """
    try:
        return _run_simulation_from_parquet(
            scenario_name, num_cpu, num_gpu, base_util, supply_temp_c, crisis,
        )
    except FileNotFoundError:
        return _run_simulation_sin_fallback(
            scenario_name, num_cpu, num_gpu, base_util, supply_temp_c, crisis,
        )


# ── RL vs Rule-based 비교 (IDCEnv 기반) ───────────────────────────────────

def _scenario_start_idx(scenario_name: str) -> int:
    """시즌 시나리오 → parquet idx (5분 step, 2019-01-01 00:00 = 0)."""
    ts = pd.Timestamp(_SEASON_START.get(scenario_name, _SEASON_START["여름 (Summer)"]))
    base = pd.Timestamp("2019-01-01 00:00:00")
    return int((ts - base).total_seconds() // 300)


def run_rl_vs_rule(scenario_name: str, n_steps: int = 288) -> dict:
    """IDCEnv에서 Rule-based / RL-best 컨트롤러를 동시 평가.

    반환:
        {"rule": DataFrame(시간별 PUE/온도/모드), "rl": DataFrame(...),
         "rule_pue_mean": float, "rl_pue_mean": float,
         "savings_pct": float, "rule_viol": int, "rl_viol": int}
    """
    import numpy as np
    from domain.controllers.idc_env import IDCEnv
    from domain.controllers.rule_based import calculate_setpoint, decide_cooling_mode
    from domain.controllers.rl_inference import predict_best

    start_idx = _scenario_start_idx(scenario_name)

    def _rule(obs):
        od, hum = float(obs[1]), float(obs[3])
        return calculate_setpoint(decide_cooling_mode(od, hum), od)

    def _rl(obs):
        return float(predict_best(obs))

    def _run(controller):
        env = IDCEnv(max_episode_steps=n_steps)
        env.reset(seed=42)
        env._data_idx = start_idx
        env._zone_temp = 25.0
        env._outdoor_history.clear()
        obs = env._get_obs()
        rows = []
        viol = 0
        for step in range(n_steps):
            sp = controller(obs)
            obs, _, _, trunc, info = env.step(np.array([sp], dtype=np.float32))
            if info.get("temp_violation", 0) > 0:
                viol += 1
            rows.append({
                "step":          step,
                "minute":        step * 5,
                "PUE":           info["pue"],
                "zone_temp_c":   info["zone_temp_c"],
                "supply_temp_c": float(obs[6]),
                "냉각 모드":      info["cooling_mode"],
            })
            if trunc:
                break
        return pd.DataFrame(rows), viol

    df_rule, v_rule = _run(_rule)
    df_rl,   v_rl   = _run(_rl)
    pue_rule = float(df_rule["PUE"].mean())
    pue_rl   = float(df_rl["PUE"].mean())
    savings  = (pue_rule - pue_rl) / pue_rule * 100.0 if pue_rule > 0 else 0.0

    return {
        "rule":          df_rule,
        "rl":            df_rl,
        "rule_pue_mean": pue_rule,
        "rl_pue_mean":   pue_rl,
        "savings_pct":   savings,
        "rule_viol":     v_rule,
        "rl_viol":       v_rl,
    }


def calculate_esg(df: pd.DataFrame) -> dict:
    """시뮬레이션 결과 DataFrame으로 ESG 지표를 계산한다."""
    total_energy_kwh = df["총 전력 (kW)"].sum()
    it_energy_kwh    = df["IT 전력 (kW)"].sum()

    carbon_tco2 = total_energy_kwh * CARBON_FACTOR_TCO2_PER_MWH / 1000
    cost_krw    = total_energy_kwh * ELECTRICITY_COST_KRW_PER_KWH
    # Green Grid CUE = 총탄소 / IT에너지 (kgCO₂eq/kWh). PUE × 배출계수와 동치.
    cue_kg_per_kwh = (total_energy_kwh / it_energy_kwh) * CARBON_FACTOR_KG_PER_KWH if it_energy_kwh > 0 else 0.0

    return {
        "carbon_tco2_day":   round(carbon_tco2, 3),
        "carbon_tco2_month": round(carbon_tco2 * 30, 1),
        "cost_krw_day":      round(cost_krw / 1e4, 2),        # 만원 (1만원 = 1e4원)
        "cost_krw_month":    round(cost_krw * 30 / 1e4, 1),   # 만원
        "cue":               round(cue_kg_per_kwh, 4),         # kgCO₂/kWh
    }
