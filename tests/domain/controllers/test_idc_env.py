"""IDCEnv 단위 테스트.

reward_type 은 현재 best 인 'weighted' 만 검증한다.
'hierarchical' 등 다른 보상 모드는 plan 범위에서 제외.
"""

import math

import numpy as np
import pytest

from domain.controllers.idc_env import (
    IDCEnv,
    T_SUPPLY_MAX,
    T_SUPPLY_MIN,
    T_ZONE_LOWER,
    T_ZONE_UPPER,
)


REQUIRED_INFO_KEYS = {
    "pue",
    "zone_temp_c",
    "it_power_kw",
    "cooling_power_kw",
    "cooling_mode",
}


class TestReset:
    def test_obs_in_observation_space(self, env):
        obs, info = env.reset(seed=0)
        assert env.observation_space.contains(obs)
        assert info == {}

    def test_obs_length_is_nine(self, env):
        # 회귀 방지: wet_bulb 추가로 9-dim (hour, outdoor_temp, outdoor_trend, humidity,
        #                                    cpu_util, zone_temp, supply_temp, it_power, wet_bulb)
        obs, _ = env.reset(seed=0)
        assert obs.shape == (9,)

    def test_reset_is_deterministic_with_seed(self, env):
        obs1, _ = env.reset(seed=42)
        obs2, _ = env.reset(seed=42)
        assert np.array_equal(obs1, obs2)


class TestStep:
    def test_reward_is_finite(self, env):
        env.reset(seed=0)
        action = np.array([20.0], dtype=np.float32)
        _, reward, _, _, _ = env.step(action)
        assert math.isfinite(reward)

    def test_info_has_required_keys(self, env):
        env.reset(seed=0)
        _, _, _, _, info = env.step(np.array([20.0], dtype=np.float32))
        assert REQUIRED_INFO_KEYS.issubset(info.keys())

    def test_info_pue_is_positive(self, env):
        env.reset(seed=0)
        _, _, _, _, info = env.step(np.array([20.0], dtype=np.float32))
        assert info["pue"] > 1.0  # PUE = 1 + overhead >= 1

    def test_action_below_min_is_clamped(self, env):
        env.reset(seed=0)
        obs, _, _, _, _ = env.step(np.array([T_SUPPLY_MIN - 5.0], dtype=np.float32))
        # obs[6] = supply_temp 은 클램프 후 값
        assert obs[6] >= T_SUPPLY_MIN

    def test_action_above_max_is_clamped(self, env):
        env.reset(seed=0)
        obs, _, _, _, _ = env.step(np.array([T_SUPPLY_MAX + 5.0], dtype=np.float32))
        assert obs[6] <= T_SUPPLY_MAX

    def test_truncated_at_max_episode_steps(self, env):
        env.reset(seed=0)
        action = np.array([20.0], dtype=np.float32)
        truncated = False
        for _ in range(env._max_episode_steps):
            _, _, terminated, truncated, _ = env.step(action)
            assert not terminated
        assert truncated is True

    def test_obs_remains_in_space_throughout_episode(self, env):
        env.reset(seed=0)
        action = np.array([20.0], dtype=np.float32)
        for _ in range(env._max_episode_steps):
            obs, _, _, _, _ = env.step(action)
            assert env.observation_space.contains(obs)


class TestRewardWeighted:
    def test_weighted_reward_no_nan(self):
        # best 보상 타입 = weighted
        env = IDCEnv(max_episode_steps=10, reward_type="weighted")
        env.reset(seed=0)
        for _ in range(10):
            _, reward, _, _, _ = env.step(np.array([20.0], dtype=np.float32))
            assert math.isfinite(reward), f"weighted reward NaN/inf: {reward}"
