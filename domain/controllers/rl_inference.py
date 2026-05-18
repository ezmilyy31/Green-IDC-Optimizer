"""IDCEnv 기반 SAC 모델 추론 — control_service / 대시보드 / 시뮬레이션 연동용.

path-key 기반 캐시로 여러 모델을 동시 로드할 수 있으나, 현재는 best 단독 운용.

2-tier 안전 시스템 (predict_best):
  Tier 1 (효율): best 모델 RL 정책
  Tier 2 (강제 cap): safe fallback — zone > 26.5°C 시 T_SUPPLY_MIN, zone < 19.0°C 시 T_SUPPLY_MAX
"""

from pathlib import Path
from typing import Optional

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from core.config.settings import settings
from domain.controllers.idc_env import (
    IDCEnv,
    T_SUPPLY_MIN,
    T_SUPPLY_MAX,
    T_ZONE_LOWER,
    T_ZONE_UPPER,
)

# Safe fallback 임계값: zone temp 위반 한계에서 여유를 두고 오버라이드 작동
SAFE_HIGH_C = T_ZONE_UPPER - 0.5  # 26.5°C: 한계 직전에 강력 냉각
SAFE_LOW_C = T_ZONE_LOWER + 1.0   # 19.0°C: 한계 직전에 냉각 완화

# IDCEnv obs 인덱스
ZONE_TEMP_OBS_INDEX = 5
CPU_UTIL_OBS_INDEX = 4
IT_POWER_OBS_INDEX = 7


class RLInference:
    """SAC + VecNormalize 페어 로드 및 추론. path-key 캐시로 여러 모델 동시 로드 지원.

    obs 순서 (IDCEnv와 동일):
        [hour, outdoor_temp, outdoor_trend, humidity, cpu_util, zone_temp, supply_temp, it_power, wet_bulb]
    action: supply_temp_setpoint (°C)
    """

    # path → RLInference 인스턴스 매핑. 여러 모델 동시 로드 가능.
    _instances: dict[str, "RLInference"] = {}

    def __init__(self, model_path: str):
        zip_path = Path(model_path)
        stats_path = Path(str(zip_path).replace(".zip", "") + "_vecnorm.pkl")
        if not zip_path.exists():
            raise FileNotFoundError(f"RL 모델 없음: {zip_path}")
        if not stats_path.exists():
            raise FileNotFoundError(f"VecNormalize 통계 없음: {stats_path}")

        dummy_env = DummyVecEnv([lambda: IDCEnv()])
        self._vec_env = VecNormalize.load(str(stats_path), dummy_env)
        self._vec_env.training = False
        self._vec_env.norm_reward = False
        self._model = SAC.load(str(zip_path))
        self.model_path = str(zip_path)

    @classmethod
    def get(cls, model_path: Optional[str] = None) -> "RLInference":
        """path별 캐시 반환. None이면 settings.rl_model_path 사용 (기존 동작 호환).

        같은 path로 여러 번 호출해도 모델은 1회만 로드된다.
        """
        path = model_path or settings.rl_model_path
        if path not in cls._instances:
            cls._instances[path] = cls(path)
        return cls._instances[path]

    def predict(self, obs: np.ndarray) -> float:
        """단일 모델 추론 + safe fallback.

        zone > 26.5°C: T_SUPPLY_MIN 강제 (긴급 냉각)
        zone < 19.0°C: T_SUPPLY_MAX 강제 (냉각 완화)
        그 외: RL 정책 결과
        """
        obs_batch = obs.reshape(1, -1).astype(np.float32)
        obs_norm = self._vec_env.normalize_obs(obs_batch)
        action, _ = self._model.predict(obs_norm, deterministic=True)
        rl_action = float(action.flatten()[0])

        zone_temp = float(obs[ZONE_TEMP_OBS_INDEX])
        if zone_temp > SAFE_HIGH_C:
            return T_SUPPLY_MIN
        if zone_temp < SAFE_LOW_C:
            return T_SUPPLY_MAX
        return rl_action


def predict_best(obs: np.ndarray, model_path: Optional[str] = None) -> float:
    """효율 우선 best 모델 추론 + safe fallback.

    Args:
        obs: IDCEnv 9-dim 관측 벡터
        model_path: 모델 경로 override. None이면 settings.rl_model_path 사용.

    Returns:
        supply_temp_setpoint (°C, T_SUPPLY_MIN ~ T_SUPPLY_MAX)
    """
    return RLInference.get(model_path).predict(obs)


