"""IDCEnv 기반 SAC 모델 추론 — control_service / 대시보드 / 시뮬레이션 연동용.

여러 모델을 동시 로드하기 위해 path-key 기반 캐시를 사용한다 (multi-model singleton).
- best 모델     : settings.rl_model_path        (효율 우선, /control/rl 엔드포인트)
- safety 모델   : settings.rl_safety_model_path (안전 우선, /control/rl-hybrid 엔드포인트)

위기 시나리오 OOD 상황에서 catastrophic 실패를 막기 위해, predict() 결과에 zone temp
기반 safe fallback을 적용한다. predict_hybrid()는 부하/온도 신호 기반으로 두 모델을
자동 전환하며, safe fallback은 그 위 layer로 동작한다.

다층 안전 시스템 (predict_hybrid + safe fallback):
  Layer 1 (효율): best 모델     — 정상 부하 + zone < 26.0°C
  Layer 2 (안전): safety 모델   — 부하 위기 OR 26.0 ≤ zone < 26.5°C
  Layer 3 (강제 cap): T_SUPPLY_MIN — zone ≥ 26.5°C (predict() 내부 safe fallback)
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

# Hybrid switch 임계 — 실측 데이터 분포 기반 (eval_crisis.py와 동일):
#   normal:       cpu mean=0.40 max=0.52, it_power mean=144 max=161
#   server_surge: cpu mean=0.52 max=0.67, it_power mean=161 max=183
HYBRID_SWITCH_C = 26.0          # zone temp 임계 (사후 신호, fallback 26.5보다 낮게)
HYBRID_CPU_THRESH = 0.50        # cpu_util 임계 (사전 신호)
HYBRID_IT_POWER_THRESH = 165.0  # it_power 임계 (사전 신호)

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


def predict_hybrid(
    obs: np.ndarray,
    efficient_path: Optional[str] = None,
    safety_path: Optional[str] = None,
    switch_c: float = HYBRID_SWITCH_C,
    cpu_thresh: float = HYBRID_CPU_THRESH,
    it_power_thresh: float = HYBRID_IT_POWER_THRESH,
) -> float:
    """다중 신호 기반 best/safety 자동 전환 + safe fallback.

    세 가지 신호 중 하나라도 위기로 판정되면 safety 모델 사용:
      - 부하 신호 (사전): cpu_util > cpu_thresh
      - 부하 신호 (사전): it_power > it_power_thresh
      - 온도 신호 (사후): zone_temp >= switch_c

    각 모델의 predict() 내부에 safe fallback이 이미 적용되어, 선택된 모델 결과가
    그대로 반환되어도 zone temp 한계(26.5°C 초과)에서는 T_SUPPLY_MIN 강제.

    Args:
        obs: IDCEnv 9-dim 관측 벡터
        efficient_path: best 모델 경로. None이면 settings.rl_model_path.
        safety_path: safety 모델 경로. None이면 settings.rl_safety_model_path.
        switch_c, cpu_thresh, it_power_thresh: 임계값 override.

    Returns:
        supply_temp_setpoint (°C, T_SUPPLY_MIN ~ T_SUPPLY_MAX)
    """
    cpu_util = float(obs[CPU_UTIL_OBS_INDEX])
    it_power = float(obs[IT_POWER_OBS_INDEX])
    zone_temp = float(obs[ZONE_TEMP_OBS_INDEX])

    is_load_crisis = cpu_util > cpu_thresh or it_power > it_power_thresh
    is_temp_warning = zone_temp >= switch_c

    if is_load_crisis or is_temp_warning:
        path = safety_path or settings.rl_safety_model_path
    else:
        path = efficient_path or settings.rl_model_path

    return RLInference.get(path).predict(obs)
