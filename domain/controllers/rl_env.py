"""Sinergym datacenter_dx 환경 래퍼. [obs 필터링 + 에피소드 길이 제한]

Sinergym의 obs 37개를 핵심 9개로 필터링하고,
보상 함수는 Sinergym 기본 제공(energy + comfort)을 그대로 사용한다.
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
    - reward: Sinergym 기본 보상 사용 (energy_term + comfort_term)
    """

    def __init__(self, max_episode_steps: int | None = None):
        env = gym.make(ENV_ID) # Sinergym 환경 생성
        super().__init__(env) # gym.Wrapper에 등록

        # observation space를 필터링된 크기로 재정의

        self._max_episode_steps = max_episode_steps
        self._step_count = 0

        # filtered obs space 재정의
        n_obs = len(FILTERED_OBS_KEYS)
        self.observation_space = gym.spaces.Box(
            low=-5e7, high=5e7, shape=(n_obs,), dtype=np.float32,
        )

    # 총 37개의 obs를 필요한 9개의 배열로 변경
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

        """
        커스텀 보상: PUE 최소화 + 온도 위반 패널티
        east_temp = obs[4]       # east_zone_air_temperature
        west_temp = obs[5]       # west_zone_air_temperature
        hvac_power = obs[8]      # HVAC_electricity_demand_rate

        zone_temp = max(east_temp, west_temp)
        temp_penalty = max(0, zone_tep - 27) * 10
        custom_reward = -hvac_power / 1e4 - temp_penalty
        
        """

        # 에피소드 길이 제한 (선택)
        if self._max_episode_steps and self._step_count >= self._max_episode_steps:
            truncated = True

        return self._filter_obs(obs), reward, terminated, truncated, info