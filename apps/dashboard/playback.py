"""AI vs 규칙 비교 — 시나리오 프리셋 + A/B 시뮬레이션 헬퍼.

같은 외기/IT부하 시계열에서 두 컨트롤러(Rule-based, RL)를 동시에 돌려서
타임랩스 재생용 DataFrame을 생성한다.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from core.config.constants import (
    CARBON_FACTOR_KG_PER_KWH,
    ELECTRICITY_COST_KRW_PER_KWH,
)
from domain.controllers.idc_env import IDCEnv, EPISODE_STEPS
from domain.controllers.rule_based import calculate_setpoint, decide_cooling_mode

# 5분 간격, 288 step = 1일
STEPS_PER_DAY = EPISODE_STEPS
PARQUET_PATH = Path(__file__).resolve().parents[2] / "data" / "weather" / "synthetic_idc_1year_noisy.parquet"

# 디스크 캐시: 시나리오별 시뮬레이션 결과를 pickle로 보관 → 컨테이너 재시작에도 유지
# 시나리오 fingerprint(start_idx, n_steps, cpu_boost, outdoor_offset_c)가 바뀌면 자동 재계산
# 코드/모델 변경 시 강제 무효화하려면 CACHE_VERSION 숫자만 올리면 됨
CACHE_VERSION = "v2"
CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache" / "playback"


@dataclass(frozen=True)
class ScenarioPreset:
    key: str
    label: str
    description: str
    start_idx: int
    n_steps: int
    cpu_boost: float = 1.0       # IT부하 배율 (스파이크 시나리오용)
    outdoor_offset_c: float = 0.0  # 외기 온도 오프셋


def _find_extreme_window(df: pd.DataFrame, col: str, mode: str, window_days: int) -> int:
    """col 컬럼이 window_days 기간에서 가장 hot/cold인 시작 idx를 반환."""
    win = window_days * STEPS_PER_DAY
    rolling = df[col].rolling(win).mean()
    if mode == "max":
        end_idx = int(rolling.idxmax())
    else:
        end_idx = int(rolling.idxmin())
    return max(0, end_idx - win + 1)


def list_scenarios() -> list[ScenarioPreset]:
    """1년치 데이터를 스캔해 시나리오 프리셋을 동적으로 생성."""
    df = pd.read_parquet(PARQUET_PATH, columns=["outside_temp_c", "cpu_utilization"])

    heatwave_start = _find_extreme_window(df, "outside_temp_c", "max", 2)
    coldsnap_start = _find_extreme_window(df, "outside_temp_c", "min", 2)
    shoulder_start = _find_extreme_window(df, "outside_temp_c", "max", 2)
    # 환절기: 외기 평균이 15°C 근처인 첫 구간
    shoulder_target = (df["outside_temp_c"].rolling(STEPS_PER_DAY).mean() - 15.0).abs()
    shoulder_start = int(shoulder_target.idxmin()) - STEPS_PER_DAY + 1
    shoulder_start = max(0, shoulder_start)

    # IT 부하 스파이크: cpu가 가장 높은 1일
    spike_start = _find_extreme_window(df, "cpu_utilization", "max", 1)

    return [
        ScenarioPreset(
            key="heatwave",
            label="혹서기 폭염 (2일)",
            description="1년 중 외기 가장 더운 48시간 — 칠러 풀가동 구간에서 RL의 절감 여력 확인",
            start_idx=heatwave_start,
            n_steps=2 * STEPS_PER_DAY,
        ),
        ScenarioPreset(
            key="coldsnap",
            label="한파 (2일)",
            description="1년 중 외기 가장 추운 48시간 — 자연공조 100% 활용 가능 구간",
            start_idx=coldsnap_start,
            n_steps=2 * STEPS_PER_DAY,
        ),
        ScenarioPreset(
            key="shoulder",
            label="환절기 (1일)",
            description="외기 15°C 근처 — Free/Hybrid/Chiller 모드가 모두 등장하는 가장 까다로운 구간",
            start_idx=shoulder_start,
            n_steps=STEPS_PER_DAY,
        ),
        ScenarioPreset(
            key="it_spike",
            label="IT 부하 스파이크 (1일)",
            description="가장 바쁜 24시간 + 추가 부하 부스트(×1.2) — 과부하 시 안전 제어 확인",
            start_idx=spike_start,
            n_steps=STEPS_PER_DAY,
            cpu_boost=1.2,
        ),
    ]


def _rule_based_setpoint_local(obs: np.ndarray) -> float:
    """fallback: control-service 불가 시 도메인 함수로 직접 계산."""
    outdoor_temp = float(obs[1])
    humidity = float(obs[3])
    mode = decide_cooling_mode(outdoor_temp, humidity)
    return calculate_setpoint(mode, outdoor_temp)


def _obs_to_payload(obs: np.ndarray) -> dict:
    """IDCEnv obs(9차원) → control_service payload 매핑.

    obs 순서: [hour, outdoor_temp, outdoor_trend, humidity, cpu_util,
              zone_temp, supply_temp, it_power, wet_bulb]
    """
    return {
        "outdoor_temp_c":   float(obs[1]),
        "it_power_kw":      max(float(obs[7]), 0.1),  # ControlRequest는 gt=0.0 강제
        "outdoor_humidity": float(obs[3]),
        "outdoor_temp_trend_c_per_s": float(obs[2]),
        "zone_temp_c":      float(obs[5]),
        "supply_setpoint_c": float(obs[6]),
        "cpu_utilization":  float(obs[4]),
    }


def _make_rule_based_http_controller():
    """control-service /control/rule-based 호출 wrapper. 실패 시 local fallback."""
    from apps.dashboard.api_client import rule_based_control
    failures = {"count": 0}

    def controller(obs: np.ndarray) -> float:
        if failures["count"] >= 3:
            return _rule_based_setpoint_local(obs)  # 3회 실패 후 영구 fallback
        p = _obs_to_payload(obs)
        result = rule_based_control(
            outdoor_temp_c=p["outdoor_temp_c"],
            it_power_kw=p["it_power_kw"],
            outdoor_humidity=p["outdoor_humidity"],
        )
        if "error" in result:
            failures["count"] += 1
            return _rule_based_setpoint_local(obs)
        return float(result["supply_air_temp_setpoint_c"])
    return controller


def _make_rl_best_http_controller():
    """control-service /control/rl 호출 wrapper (효율 우선 best 모델만 사용).

    실패 시 local predict_best()로 fallback.
    """
    from apps.dashboard.api_client import rl_control
    failures = {"count": 0}

    def _local_predict(obs: np.ndarray) -> float:
        from domain.controllers.rl_inference import predict_best
        return float(predict_best(obs))

    def controller(obs: np.ndarray) -> float:
        if failures["count"] >= 3:
            return _local_predict(obs)
        p = _obs_to_payload(obs)
        result = rl_control(
            outdoor_temp_c=p["outdoor_temp_c"],
            it_power_kw=p["it_power_kw"],
            outdoor_humidity=p["outdoor_humidity"],
            zone_temp_c=p["zone_temp_c"],
            supply_setpoint_c=p["supply_setpoint_c"],
            cpu_utilization=p["cpu_utilization"],
            outdoor_temp_trend_c_per_s=p["outdoor_temp_trend_c_per_s"],
        )
        if "error" in result:
            failures["count"] += 1
            return _local_predict(obs)
        return float(result["supply_air_temp_setpoint_c"])
    return controller


def _make_rl_hybrid_http_controller():
    """control-service /control/rl-hybrid 호출 wrapper.

    위기 신호(부하/온도) 감지 시 control_service가 safety 모델로 자동 전환 →
    server_surge·폭염 등 OOD 상황에서도 0% 위반 달성.
    실패 시 local predict_hybrid()로 fallback.
    """
    from apps.dashboard.api_client import rl_hybrid_control
    failures = {"count": 0}

    def _local_predict(obs: np.ndarray) -> float:
        from domain.controllers.rl_inference import predict_hybrid
        return float(predict_hybrid(obs))

    def controller(obs: np.ndarray) -> float:
        if failures["count"] >= 3:
            return _local_predict(obs)
        p = _obs_to_payload(obs)
        result = rl_hybrid_control(
            outdoor_temp_c=p["outdoor_temp_c"],
            it_power_kw=p["it_power_kw"],
            outdoor_humidity=p["outdoor_humidity"],
            zone_temp_c=p["zone_temp_c"],
            supply_setpoint_c=p["supply_setpoint_c"],
            cpu_utilization=p["cpu_utilization"],
            outdoor_temp_trend_c_per_s=p["outdoor_temp_trend_c_per_s"],
        )
        if "error" in result:
            failures["count"] += 1
            return _local_predict(obs)
        return float(result["supply_air_temp_setpoint_c"])
    return controller


def _build_env(scenario: ScenarioPreset) -> IDCEnv:
    """시나리오 시작 idx로 고정된 IDCEnv 인스턴스 생성."""
    env = IDCEnv(max_episode_steps=scenario.n_steps)
    env.reset(seed=42)
    env._data_idx = scenario.start_idx
    env._zone_temp = 25.0
    env._outdoor_history.clear()

    if scenario.cpu_boost != 1.0 or scenario.outdoor_offset_c != 0.0:
        # 사본을 만들어 시나리오 변조 적용 (다른 IDCEnv 인스턴스에 영향 X)
        env._data = env._data.copy()
        end = scenario.start_idx + scenario.n_steps
        if scenario.outdoor_offset_c != 0.0:
            env._data[scenario.start_idx:end, 0] += scenario.outdoor_offset_c
        if scenario.cpu_boost != 1.0:
            env._data[scenario.start_idx:end, 2] = np.clip(
                env._data[scenario.start_idx:end, 2] * scenario.cpu_boost, 0.0, 1.0
            )
    return env


def _run_episode(env: IDCEnv, controller, n_steps: int) -> pd.DataFrame:
    """주어진 env에서 controller(obs→setpoint)로 n_steps 시뮬레이션."""
    obs = env._get_obs()
    rows = []
    for step in range(n_steps):
        setpoint = controller(obs)
        obs, _reward, _term, trunc, info = env.step(np.array([setpoint], dtype=np.float32))
        # 5분 단위 step → kWh 환산 (전력 kW × 5/60 h)
        total_power_kw = info["it_power_kw"] + info["cooling_power_kw"]
        rows.append({
            "step":            step,
            "minute":          step * 5,
            "외기온도":         float(obs[1]),
            "습도":            float(obs[3]),
            "CPU 사용률":       float(obs[4]) * 100.0,
            "공급 온도":        float(obs[6]),
            "서버실 온도":      info["zone_temp_c"],
            "IT 전력":          info["it_power_kw"],
            "냉각 전력":        info["cooling_power_kw"],
            "총 전력":          total_power_kw,
            "PUE":             info["pue"],
            "냉각 모드":        info["cooling_mode"],
            "온도 위반":        info["temp_violation"],
            "누적 kWh":         total_power_kw * (5.0 / 60.0),
        })
        if trunc:
            break

    df = pd.DataFrame(rows)
    df["누적 kWh"] = df["누적 kWh"].cumsum()
    df["누적 원"] = df["누적 kWh"] * ELECTRICITY_COST_KRW_PER_KWH
    df["누적 kgCO₂"] = df["누적 kWh"] * CARBON_FACTOR_KG_PER_KWH
    return df


def simulate_compare(scenario: ScenarioPreset) -> dict:
    """시나리오에 대해 세 컨트롤러(Rule / RL Best / RL Hybrid)를 돌리고 결과를 반환.

    - Rule-based: 외기 wet-bulb로 3단 setpoint 결정 (22/20/18°C)
    - RL Best: sac-wetbulb-1m 단독, PUE 최적화
    - RL Hybrid: 위기 신호 감지 시 safety 모델(sac-dr-fresh-1m) 자동 전환
    """
    rule_env = _build_env(scenario)
    df_rule = _run_episode(rule_env, _make_rule_based_http_controller(), scenario.n_steps)

    rl_loaded = False
    df_best = None
    df_hybrid = None
    try:
        best_env = _build_env(scenario)
        df_best = _run_episode(best_env, _make_rl_best_http_controller(), scenario.n_steps)
        hybrid_env = _build_env(scenario)
        df_hybrid = _run_episode(hybrid_env, _make_rl_hybrid_http_controller(), scenario.n_steps)
        rl_loaded = True
    except Exception as exc:
        # RL 호출 완전 실패 시 rule-based로 모두 채워서 UI 안정성 유지
        print(f"[playback] RL 시뮬레이션 실패 → rule-only 모드: {exc}")
        df_best = df_rule.copy()
        df_hybrid = df_rule.copy()

    def _summary(df: pd.DataFrame, label: str) -> dict:
        total = float(df["누적 kWh"].iloc[-1])
        savings = float(df_rule["누적 kWh"].iloc[-1]) - total
        pct = (savings / float(df_rule["누적 kWh"].iloc[-1]) * 100.0) if df_rule["누적 kWh"].iloc[-1] > 0 else 0.0
        return {
            f"{label}_total_kwh":   total,
            f"{label}_savings_kwh": savings,
            f"{label}_savings_pct": pct,
            f"{label}_savings_krw": savings * ELECTRICITY_COST_KRW_PER_KWH,
            f"{label}_savings_co2": savings * CARBON_FACTOR_KG_PER_KWH,
            f"{label}_avg_pue":     float(df["PUE"].mean()),
            f"{label}_violations":  int((df["온도 위반"] > 0).sum()),
        }

    summary: dict = {
        "rule_total_kwh": float(df_rule["누적 kWh"].iloc[-1]),
        "rule_avg_pue":   float(df_rule["PUE"].mean()),
        "rule_violations": int((df_rule["온도 위반"] > 0).sum()),
        "rl_loaded":      rl_loaded,
    }
    summary.update(_summary(df_best, "best"))
    summary.update(_summary(df_hybrid, "hybrid"))

    return {
        "rule":     df_rule,
        "best":     df_best,
        "hybrid":   df_hybrid,
        "scenario": scenario,
        "summary":  summary,
    }


# ── 디스크 캐시 래퍼 ────────────────────────────────────────────────────────

def _disk_cache_path(scenario: ScenarioPreset) -> Path:
    """시나리오 fingerprint 기반 캐시 파일 경로."""
    fp = (
        f"{CACHE_VERSION}_{scenario.key}"
        f"_idx{scenario.start_idx}"
        f"_n{scenario.n_steps}"
        f"_cpu{scenario.cpu_boost}"
        f"_temp{scenario.outdoor_offset_c}"
    )
    return CACHE_DIR / f"{fp}.pkl"


def simulate_compare_cached(scenario: ScenarioPreset) -> dict:
    """simulate_compare의 디스크 캐시 버전.

    같은 시나리오 fingerprint면 컨테이너 재시작에도 즉시 로드.
    모델/코드 변경 시 CACHE_VERSION을 올리거나 data/cache/playback/ 를 비운다.
    """
    cache_path = _disk_cache_path(scenario)
    if cache_path.exists():
        try:
            with cache_path.open("rb") as f:
                return pickle.load(f)
        except Exception as exc:
            print(f"[playback] 캐시 로드 실패 → 재계산: {exc}")

    result = simulate_compare(scenario)
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("wb") as f:
            pickle.dump(result, f)
    except Exception as exc:
        print(f"[playback] 캐시 저장 실패 (계속 진행): {exc}")
    return result
