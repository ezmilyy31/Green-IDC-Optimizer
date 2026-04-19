"""커스텀 IDC Gym 환경 — 우리 thermodynamics 모델 기반.

Sinergym 대신 직접 만든 열역학 모델로 서버실 온도를 시뮬레이션한다.
- 날씨: 한국 월별 기후 패턴 (실측 기반)
- IT 부하: WORKLOAD_PROFILE + 계절 노이즈
- 제어: CRAH 공급 온도 setpoint (20~24°C)
- 보상: PUE overhead 최소화 + 온도 위반 패널티 (팬 전력 포함)

사용법:
    from domain.controllers.idc_env import IDCEnv
    env = IDCEnv()
"""

import numpy as np
from pathlib import Path

import gymnasium as gym
from gymnasium import spaces

from core.config.constants import (
    M_AIR_DESIGN_KG_S,
    T_SUPPLY_DESIGN_C,
    WORKLOAD_PROFILE,
    FAN_POWER_RATIO_FREE,
    FAN_POWER_RATIO_CHILLER,
)
from core.config.enums import CoolingMode
from domain.thermodynamics.chiller import calculate_chiller_power_kw
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
T_SUPPLY_MIN = 20.0
T_SUPPLY_MAX = 24.0

# 에피소드 길이
EPISODE_STEPS = 288  # 1일 (5분 × 288)

# 한국 월별 외기온도/습도 평균 (춘천 기준, 기상청 1991-2020 평년값)
KOR_MONTHLY_TEMP_MEAN = [-3.8, -1.5, 4.5, 11.8, 17.3, 21.8, 25.1, 25.9, 20.3, 13.4, 5.8, -0.8]
KOR_MONTHLY_HUMID_MEAN = [60.0, 60.0, 61.0, 58.0, 64.0, 72.0, 82.0, 80.0, 75.0, 69.0, 67.0, 64.0]


class IDCEnv(gym.Env):
    """커스텀 IDC 강화학습 환경.

    관측 (7개):
        [hour, outdoor_temp, humidity, cpu_utilization, zone_temp, supply_setpoint, it_power_kw]

    행동 (1개):
        supply_temp_setpoint: CRAH 공급 온도 [16, 27]°C

    보상:
        -w_energy * (PUE - 1.0) - (1 - w_energy) * temp_violation
    """

    metadata = {"render_modes": []}

    def __init__(self, max_episode_steps: int = EPISODE_STEPS, w_energy: float = 0.5):
        super().__init__()

        self._max_episode_steps = max_episode_steps
        self._w_energy = w_energy
        self._step_count = 0
        self._zone_temp = 22.0
        self._supply_temp = T_SUPPLY_DESIGN_C
        self._data_idx = 0

        obs_low  = np.array([0,  -15, 20, 0.0, 10.0, T_SUPPLY_MIN,   0.0], dtype=np.float32)
        obs_high = np.array([23,  45, 95, 1.0, 40.0, T_SUPPLY_MAX, 400.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        self.action_space = spaces.Box(
            low=np.array([T_SUPPLY_MIN], dtype=np.float32),
            high=np.array([T_SUPPLY_MAX], dtype=np.float32),
            dtype=np.float32,
        )

        self._data = self._generate_year_data()

    def _generate_year_data(self) -> np.ndarray:
        n = 365 * EPISODE_STEPS
        rng = np.random.default_rng(42)

        steps = np.arange(n)
        hours = (steps * 5 / 60) % 24
        month_idx = ((steps * 5 / 60 / 24) % 365 / 30.4).astype(int).clip(0, 11)

        temp_mean = np.array([KOR_MONTHLY_TEMP_MEAN[m] for m in month_idx])
        temp_daily = 6.0 * np.sin(2 * np.pi * hours / 24 - np.pi / 2)
        outdoor_temp = (temp_mean + temp_daily + rng.normal(0, 1.5, n)).astype(np.float32)

        humid_mean = np.array([KOR_MONTHLY_HUMID_MEAN[m] for m in month_idx])
        humidity = np.clip(humid_mean + rng.normal(0, 5, n), 20, 95).astype(np.float32)

        cpu_base = np.array([WORKLOAD_PROFILE[int(h)] for h in hours])
        cpu_util = np.clip(cpu_base + rng.normal(0, 0.05, n), 0.1, 1.0).astype(np.float32)

        return np.stack([outdoor_temp, humidity, cpu_util, hours.astype(np.float32)], axis=1)

    def _compute_it_power(self, cpu_util: float) -> float:
        return calculate_total_it_power_kw(
            cpu_utilization=float(cpu_util),
            num_cpu_servers=NUM_CPU_SERVERS,
            num_gpu_servers=NUM_GPU_SERVERS,
        )

    def _get_obs(self) -> np.ndarray:
        row = self._data[self._data_idx % len(self._data)]
        it_power = self._compute_it_power(row[2])
        return np.array([row[3], row[0], row[1], row[2], self._zone_temp, self._supply_temp, it_power], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0
        rng = np.random.default_rng(seed)
        self._data_idx = int(rng.integers(0, len(self._data) - self._max_episode_steps - 1))
        self._zone_temp = float(rng.uniform(20.0, 24.0))
        self._supply_temp = T_SUPPLY_DESIGN_C
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        self._supply_temp = float(np.clip(action[0], T_SUPPLY_MIN, T_SUPPLY_MAX))

        row = self._data[self._data_idx % len(self._data)]
        outdoor_temp = float(row[0])
        cpu_util = float(row[2])
        it_power_kw = self._compute_it_power(cpu_util)

        # 냉각 용량 계산
        t_return = self._zone_temp + 2.0
        cooling_capacity_kw = M_AIR_DESIGN_KG_S * C_P_KJ_PER_KG_K * max(0.0, t_return - self._supply_temp)
        actual_cooling_kw = min(cooling_capacity_kw, it_power_kw)

        # 서버실 온도 변화
        excess_heat_kw = it_power_kw - actual_cooling_kw
        self._zone_temp = float(np.clip(self._zone_temp + excess_heat_kw * TIMESTEP_SEC / C_EFF_KJ_PER_K, 5.0, 50.0))

        # 칠러 + 팬 전력
        chiller_result = calculate_chiller_power_kw(actual_cooling_kw, outdoor_temp, self._supply_temp)
        if chiller_result.cooling_mode == CoolingMode.FREE_COOLING:
            fan_power_kw = actual_cooling_kw * FAN_POWER_RATIO_FREE
        else:
            fan_power_kw = actual_cooling_kw * FAN_POWER_RATIO_CHILLER
        pue_result = calculate_pue(it_power_kw, chiller_result.chiller_power_kw + fan_power_kw)

        # 보상
        # temp_violation을 pue_overhead와 같은 스케일(0~1)로 정규화 (최대 위반 9°C 기준)
        pue_overhead = pue_result.pue - 1.0
        temp_violation = max(0.0, self._zone_temp - T_ZONE_UPPER) + max(0.0, T_ZONE_LOWER - self._zone_temp)
        temp_violation_norm = temp_violation / 9.0
        # 온도 위반 없을 때 PUE가 낮을수록 양의 보너스 (에이전트가 높은 setpoint 탐색하도록 유도)
        pue_bonus = max(0.0, 0.4 - pue_overhead) if temp_violation == 0.0 else 0.0
        reward = -(self._w_energy * pue_overhead + (1 - self._w_energy) * temp_violation_norm) + pue_bonus

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
