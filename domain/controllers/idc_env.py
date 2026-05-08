"""커스텀 IDC Gym 환경 — 우리 thermodynamics 모델 기반.

Sinergym 대신 직접 만든 열역학 모델로 서버실 온도를 시뮬레이션한다.
- 날씨: 실측 데이터 (Google Cluster Trace 2019 + 기상청 ASOS)
- IT 부하: 실측 CPU 사용률 + 노이즈
- 제어: CRAH 공급 온도 setpoint (18~27°C)
- 보상: PUE 최적화 + 온도 위반 안전 페널티

사용법:
    from domain.controllers.idc_env import IDCEnv
    env = IDCEnv()
"""

from collections import deque

import numpy as np
from pathlib import Path

import gymnasium as gym
from gymnasium import spaces

from core.config.constants import (
    M_AIR_DESIGN_KG_S,
    M_AIR_MAX_KG_S,
    MIN_DELTA_T_C,
    T_SUPPLY_DESIGN_C,
    FAN_POWER_DESIGN_KW,
    FAN_AFFINITY_EXP,
)
from core.config.enums import CoolingMode
from domain.thermodynamics.chiller import calculate_chiller_power_kw, calculate_wet_bulb_c
from domain.thermodynamics.it_power import calculate_total_it_power_kw
from domain.thermodynamics.pue import calculate_pue

# 물리 상수
TIMESTEP_SEC = 300              # 5분 간격 (초)
C_P_KJ_PER_KG_K = 1.005        # 공기 비열 (kJ/kg·K)
C_EFF_KJ_PER_K = 9009.0        # 서버실 유효 열용량 (kJ/K)

# 서버 설정 (명세서 기준)
NUM_CPU_SERVERS = 400
NUM_GPU_SERVERS = 20

# 서버실 온도 제약 (명세서: 18~27°C)
T_ZONE_UPPER = 27.0
T_ZONE_LOWER = 18.0

# 공급 온도 제어 범위
T_SUPPLY_MIN = 18.0
T_SUPPLY_MAX = 25.0

# 에피소드 길이
EPISODE_STEPS = 288  # 1일 (5분 × 288)

# 외기 추세 윈도우 (1시간 = 12스텝)
TREND_WINDOW = 12

# 실측 IDC 데이터 (Google Cluster Trace 2019 + 기상청 ASOS, 5분 단위 365일)
REAL_DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "weather" / "synthetic_idc_1year_noisy.parquet"


def _load_real_data(data_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """data_pipeline.py 산출물에서 외기온도/습도/CPU 사용률 로드. 실패 시 None.

    각 배열 길이는 105,120 (365일 × 288 스텝, 5분 단위).
    """
    if not data_path.exists():
        return None
    try:
        import pandas as pd
        df = pd.read_parquet(data_path)
        outdoor_temp = df["outside_temp_c"].to_numpy(dtype=np.float32)
        humidity = df["outside_humidity_pct"].to_numpy(dtype=np.float32)
        cpu_util = df["cpu_utilization"].to_numpy(dtype=np.float32)
        return outdoor_temp, humidity, cpu_util
    except Exception:
        return None


class IDCEnv(gym.Env):
    """커스텀 IDC 강화학습 환경.

    관측 (9개):
        [hour, outdoor_temp, outdoor_trend, humidity, cpu_utilization, zone_temp, supply_setpoint, it_power_kw, wet_bulb]
        outdoor_trend: 현재 외기온도 - 최근 1시간 평균 (양수: 더워지는 중, 음수: 식는 중)
        wet_bulb: 습구온도 — 냉각 모드 전환 기준 (free <10°C, hybrid <18°C) 직접 노출

    행동 (1개):
        supply_temp_setpoint: CRAH 공급 온도 [18, 25]°C

    보상:
        weighted   (기본): 안전 우선 hierarchical 변형
            - 위반 시: -temp_violation × 3 (w_energy 무관, 안전 절대 우선)
            - 안전 시: -w_energy × pue_overhead + (0.20 - pue_overhead) × 3
              (NAVER 1.20 기준 선형 신호, 1.20 이상에서도 음수 영역 유지)
        hierarchical: 위반 시 -temp_violation, 안전 시 -pue_overhead (목표 분리)
    """

    metadata = {"render_modes": []}

    def __init__(self, max_episode_steps: int = EPISODE_STEPS, w_energy: float = 0.5, reward_type: str = "weighted"):
        super().__init__()

        self._max_episode_steps = max_episode_steps
        self._w_energy = w_energy
        self._reward_type = reward_type
        self._step_count = 0
        self._supply_temp = T_SUPPLY_DESIGN_C
        self._data_idx = 0
        self._outdoor_history: deque = deque(maxlen=TREND_WINDOW)
        # zone_temp/supply_temp는 reset()에서 재설정됨 (gym 표준상 reset 후 step 보장)

        # 관측: [hour, outdoor_temp, outdoor_trend, humidity, cpu_util, zone_temp, supply_temp, it_power, wet_bulb]
        # zone_temp 범위 5~50: 안전 위반 영역(<18, >27)도 obs로 허용 → 위반 페널티가 학습 신호
        # wet_bulb 범위 -15~35: 한국 연간 습구온도 커버 (겨울 -15°C ~ 여름 35°C)
        obs_low  = np.array([0,  -15, -15, 20, 0.0,  5.0, T_SUPPLY_MIN,   0.0, -15], dtype=np.float32)
        obs_high = np.array([23,  45,  15, 95, 1.0, 50.0, T_SUPPLY_MAX, 400.0,  35], dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        self.action_space = spaces.Box(
            low=np.array([T_SUPPLY_MIN], dtype=np.float32),
            high=np.array([T_SUPPLY_MAX], dtype=np.float32),
            dtype=np.float32,
        )

        self._data = self._generate_year_data()

    def _generate_year_data(self) -> np.ndarray:
        n = 365 * EPISODE_STEPS
        steps = np.arange(n)
        hours = (steps * 5 / 60) % 24

        real_data = _load_real_data(REAL_DATA_PATH)
        if real_data is None:
            raise FileNotFoundError(
                f"실측 IDC 데이터 파일이 필요합니다: {REAL_DATA_PATH}\n"
                "data_pipeline.py로 생성하거나 팀 공유 경로에서 받아오세요."
            )
        print(f"[IDCEnv] 실측 데이터 로드: {REAL_DATA_PATH.name}")

        outdoor_temp = real_data[0][:n]
        humidity = real_data[1][:n]
        cpu_util = real_data[2][:n]
        return np.stack([outdoor_temp, humidity, cpu_util, hours.astype(np.float32)], axis=1)

    def _compute_it_power(self, cpu_util: float) -> float:
        return calculate_total_it_power_kw(
            cpu_utilization=float(cpu_util),
            num_cpu_servers=NUM_CPU_SERVERS,
            num_gpu_servers=NUM_GPU_SERVERS,
        )

    def _get_obs(self) -> np.ndarray:
        row = self._data[self._data_idx % len(self._data)]
        outdoor_temp = float(row[0])
        humidity = float(row[1])
        trend = outdoor_temp - float(np.mean(self._outdoor_history)) if self._outdoor_history else 0.0
        wet_bulb = calculate_wet_bulb_c(outdoor_temp, humidity)
        it_power = self._compute_it_power(row[2])
        return np.array(
            [row[3], outdoor_temp, trend, humidity, row[2], self._zone_temp, self._supply_temp, it_power, wet_bulb],
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0
        rng = np.random.default_rng(seed)
        self._data_idx = int(rng.integers(0, len(self._data) - self._max_episode_steps - 1))
        self._zone_temp = float(rng.uniform(24.0, 26.0))
        self._supply_temp = T_SUPPLY_DESIGN_C
        self._outdoor_history.clear()
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        self._supply_temp = float(np.clip(action[0], T_SUPPLY_MIN, T_SUPPLY_MAX))

        row = self._data[self._data_idx % len(self._data)]
        outdoor_temp = float(row[0])
        humidity = float(row[1])
        cpu_util = float(row[2])
        self._outdoor_history.append(outdoor_temp)
        it_power_kw = self._compute_it_power(cpu_util)

        # 냉각 용량 계산 (가변 풍량 모델: CRAH가 부하에 맞춰 풍량 조절)
        # supply ↑ → ΔT ↓ → 풍량 ↑ → 팬 전력 ↑↑↑ (affinity law)
        t_return = self._zone_temp + 2.0
        delta_t = max(MIN_DELTA_T_C, t_return - self._supply_temp)
        m_air_required = it_power_kw / (C_P_KJ_PER_KG_K * delta_t)
        m_air_actual = min(m_air_required, M_AIR_MAX_KG_S)
        actual_cooling_kw = m_air_actual * C_P_KJ_PER_KG_K * delta_t

        # 서버실 온도 변화 (zone < supply는 물리적으로 불가 → supply가 하한)
        excess_heat_kw = it_power_kw - actual_cooling_kw
        new_zone = self._zone_temp + excess_heat_kw * TIMESTEP_SEC / C_EFF_KJ_PER_K
        self._zone_temp = float(np.clip(new_zone, self._supply_temp, 50.0))

        # 칠러 + 팬 전력 (cooling mode별 차별화)
        chiller_result = calculate_chiller_power_kw(
            actual_cooling_kw, outdoor_temp, self._supply_temp, humidity
        )
        fan_ratio = m_air_actual / M_AIR_DESIGN_KG_S
        if chiller_result.cooling_mode == CoolingMode.FREE_COOLING:
            # 외기 economizer: 베이스 60% + 약한 풍량 의존 (현실적 외기 도입 비용)
            fan_power_kw = FAN_POWER_DESIGN_KW * (0.6 + 0.3 * fan_ratio)
        else:
            # Hybrid/Chiller: 폐쇄 순환, Affinity Law
            fan_power_kw = FAN_POWER_DESIGN_KW * (fan_ratio ** FAN_AFFINITY_EXP)
        pue_result = calculate_pue(it_power_kw, chiller_result.chiller_power_kw + fan_power_kw)

        # 보상
        pue_overhead = pue_result.pue - 1.0
        temp_violation = max(0.0, self._zone_temp - T_ZONE_UPPER) + max(0.0, T_ZONE_LOWER - self._zone_temp)

        if self._reward_type == "hierarchical":
            # 안전/효율 목표 완전 분리
            # 위반 시: PUE 신호 없이 위반 크기만큼 패널티 → 온도 먼저 학습
            # 안전 시: violation 신호 없이 PUE만 최적화 → 에너지 효율 학습
            if temp_violation > 0.0:
                reward = -temp_violation
            else:
                reward = -pue_overhead
        else:
            # weighted: 안전 우선 hierarchical 변형 (스케일 다운: VecNormalize와 함께 사용)
            # - 위반 시: PUE 무시, 페널티
            # - 안전 시: PUE 최적화 (NAVER 1.20 기준 선형 신호)
            if temp_violation > 0.0:
                reward = -temp_violation * 1.5  # 1°C 위반 = -1.5
            else:
                # 임계값 0.25로 확장 (PUE 1.25까지 양의 신호 → 신호 강도 ↑)
                pue_signal = (0.25 - pue_overhead) * 1.5
                reward = -self._w_energy * pue_overhead + pue_signal

        self._step_count += 1
        self._data_idx += 1

        return self._get_obs(), reward, False, self._step_count >= self._max_episode_steps, {
            "pue": pue_result.pue,
            "zone_temp_c": self._zone_temp,
            "it_power_kw": it_power_kw,
            "cooling_power_kw": chiller_result.chiller_power_kw + fan_power_kw,
            "temp_violation": temp_violation,
            "cooling_mode": chiller_result.cooling_mode.value,
        }
