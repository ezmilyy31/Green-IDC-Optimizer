"""IDCEnv 기반 SAC 모델 추론 — control_service 연동용.

best 모델 + VecNormalize 통계 페어를 1회 로드하고 캐시한다 (싱글톤).
모델 경로는 settings.rl_model_path로 관리되며, 환경변수 RL_MODEL_PATH로 override 가능.

위기 시나리오(서버 급증/폭염/칠러 고장 등) OOD 상황에서 catastrophic 실패를
막기 위해, predict() 결과에 rule-based safe fallback을 적용한다.
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

ZONE_TEMP_OBS_INDEX = 5


class RLInference:
    """SAC + VecNormalize 페어 로드 및 추론 캐시.

    obs 순서 (IDCEnv와 동일):
        [hour, outdoor_temp, outdoor_trend, humidity, cpu_util, zone_temp, supply_temp, it_power, wet_bulb]
    action: supply_temp_setpoint (°C)
    """

    _instance: Optional["RLInference"] = None
    _loaded_path: Optional[str] = None

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
    def get(cls) -> "RLInference":
        path = settings.rl_model_path
        if cls._instance is None or cls._loaded_path != path:
            cls._instance = cls(path)
            cls._loaded_path = path
        return cls._instance

    def predict(self, obs: np.ndarray) -> float:
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
