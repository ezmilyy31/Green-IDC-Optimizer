"""Sinergym datacenter_dx-mixed 환경 래퍼. [obs 필터링 + 에피소드 길이 제한 + 커스텀 보상]

Sinergym의 obs 37개를 핵심 9개로 필터링하고,
커스텀 보상 함수로 East+West 양존 온도 패널티를 반영한다.
"""

import gymnasium as gym
import numpy as np
import sinergym  # noqa: F401 — 환경 등록용

from core.schemas.rl_interface import FILTERED_OBS_KEYS, OBS_INDEX

ENV_ID = "Eplus-datacenter_dx-mixed-continuous-stochastic-v1"

# 서버실 온도 제약 (명세서: 18~27°C)
TEMP_UPPER_LIMIT = 27.0
TEMP_LOWER_LIMIT = 18.0

# gym.Wrapper -> Sinergym 환경 감싸서 입출력만 변경(물리 시뮬레이션은 변경 X)
class DataCenterRLEnv(gym.Wrapper):
    """Sinergym 데이터센터 환경을 프로젝트에 맞게 래핑.

    변경점:
    - obs: 37개 → 9개 (핵심 변수만 필터링)
    - action: 그대로 (cooling_setpoint 1개, [20, 30])
    - reward: 커스텀 보상 (East+West 양존 온도 패널티 + 에너지 항)
    """

    def __init__(self, max_episode_steps: int | None = None, w_energy: float = 0.5):
        """
        Args:
            max_episode_steps: 에피소드 최대 길이 (96 = 1일)
            w_energy: 에너지 항 가중치 (0~1). 높을수록 에너지 절감 중시.
                      온도 항 가중치 = 1 - w_energy
        """
        env = gym.make(ENV_ID)
        super().__init__(env) # gym.Wrapper에 등록

        self._max_episode_steps = max_episode_steps
        self._step_count = 0
        self._w_energy = w_energy

        # filtered obs space 재정의 (변수별 실제 범위)
        obs_low = np.array([1, 0, -30, 0, 15, 15, 20, 0, 0], dtype=np.float32)
        obs_high = np.array([12, 23, 50, 100, 40, 40, 30, 1, 5e5], dtype=np.float32)
        # 순서: month, hour, outdoor_temp, humidity, east_temp, west_temp, cooling_setpoint, cpu_loading, hvac_power
        self.observation_space = gym.spaces.Box(
            low=obs_low, high=obs_high, dtype=np.float32,
        )

    # 총 37개의 obs를 필요한 9개로 필터링
    def _filter_obs(self, obs: np.ndarray) -> np.ndarray:
        indices = [OBS_INDEX[k] for k in FILTERED_OBS_KEYS]
        return obs[indices].astype(np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._step_count = 0
        return self._filter_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._step_count += 1

        # 커스텀 보상: East+West 양존 온도 패널티 + 에너지 항
        filtered = self._filter_obs(obs)
        east_temp  = filtered[4]   # east_zone_air_temperature
        west_temp  = filtered[5]   # west_zone_air_temperature
        hvac_power = filtered[8]   # HVAC_electricity_demand_rate

        east_penalty = max(0, east_temp - TEMP_UPPER_LIMIT) + max(0, TEMP_LOWER_LIMIT - east_temp)
        west_penalty = max(0, west_temp - TEMP_UPPER_LIMIT) + max(0, TEMP_LOWER_LIMIT - west_temp)

        cpu_loading = filtered[7]
        it_power = cpu_loading * 250_000                            # 최대 IT 전력 250kW (CPU 400 + GPU 20대)
        pue_overhead = hvac_power / (it_power + 1e-6)
        energy_term  = -(pue_overhead / 5.0)                        # PUE overhead 5 이하 기준 정규화
        comfort_term = -((east_penalty + west_penalty) / 9.0)      # 최대 위반 9°C(=27-18) 기준
        comfort_bonus = 0.1 if (east_penalty == 0 and west_penalty == 0) else 0.0

        reward = self._w_energy * energy_term + (1 - self._w_energy) * (comfort_term + comfort_bonus)

        # 에피소드 길이 제한 (선택)
        if self._max_episode_steps and self._step_count >= self._max_episode_steps:
            truncated = True

        return filtered, reward, terminated, truncated, info