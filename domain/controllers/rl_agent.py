"""PPO 학습 스크립트 — Sinergym datacenter_dx-mixed-continuous-stochastic-v1 환경.

사용법:
    python -m domain.controllers.rl_agent --total-timesteps 500000 --run-name exp-01
    python -m domain.controllers.rl_agent --lr 1e-4 --w-energy 0.3 --run-name exp-w03
"""

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from domain.controllers.idc_env import IDCEnv

try:
    from domain.controllers.rl_env import DataCenterRLEnv
except ImportError:
    DataCenterRLEnv = None  # type: ignore — sinergym 없는 환경에서는 사용 불가


MODEL_DIR = Path("data/models")


def make_env(max_episode_steps: int = 96, w_energy: float = 0.5, custom_env: bool = False) -> VecNormalize:
    """env 생성 + Monitor + VecNormalize 래핑."""
    if custom_env:
        env_fn = lambda: Monitor(IDCEnv(max_episode_steps=max_episode_steps, w_energy=w_energy))
    else:
        env_fn = lambda: Monitor(DataCenterRLEnv(max_episode_steps=max_episode_steps, w_energy=w_energy))
    vec_env = DummyVecEnv([env_fn])
    return VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0)


def train(
    lr: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    gamma: float = 0.9,
    total_timesteps: int = 50_000,
    max_episode_steps: int = 96,
    custom_env: bool = False,
    w_energy: float = 0.5,
    run_name: str = "ppo-baseline",
    device: str = "auto",
    resume: str | None = None,
) -> Path:
    """PPO 학습 실행.

    Args:
        lr: learning rate
        n_steps: rollout buffer 길이
        batch_size: 미니배치 크기
        gamma: discount factor
        total_timesteps: 총 학습 step 수
        max_episode_steps: 에피소드 최대 길이 (96 = 1일)
        w_energy: 보상 함수 에너지 항 가중치 (0~1)
        run_name: 실험 이름 (로그/모델 저장 경로에 사용)
        device: "auto" | "cuda" | "cpu"
        resume: 이어서 학습할 모델 경로 (없으면 처음부터)

    Returns:
        저장된 모델 경로
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    env = make_env(max_episode_steps, w_energy, custom_env)

    checkpoint_cb = CheckpointCallback(
        save_freq=50_000,
        save_path=str(MODEL_DIR / "checkpoints" / run_name),
        name_prefix="ppo",
    )

    if resume:
        print(f"[rl_agent] 모델 이어서 학습: {resume}")
        stats_path = str(resume).replace(".zip", "") + "_vecnorm.pkl"
        if Path(stats_path).exists():
            env = VecNormalize.load(stats_path, env)
            print(f"[rl_agent] VecNormalize 통계 복원: {stats_path}")
        model = PPO.load(resume, env=env, device=device, learning_rate=lr)
    else:
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=lr,
            n_steps=n_steps,
            batch_size=batch_size,
            gamma=gamma,
            n_epochs=10,
            ent_coef=0.0,
            verbose=1,
            tensorboard_log=None,
            device=device,
        )

    print(f"[rl_agent] 학습 시작: {run_name}")
    print(f"  lr={lr}, n_steps={n_steps}, batch_size={batch_size}")
    print(f"  gamma={gamma}, total_timesteps={total_timesteps}")
    print(f"  max_episode_steps={max_episode_steps}, w_energy={w_energy}, device={device}")

    model.learn(total_timesteps=total_timesteps, callback=checkpoint_cb)

    save_path = MODEL_DIR / run_name
    model.save(str(save_path))
    print(f"[rl_agent] 모델 저장 완료: {save_path}.zip")

    vecnorm_path = str(save_path) + "_vecnorm.pkl"
    env.save(vecnorm_path)
    print(f"[rl_agent] VecNormalize 통계 저장 완료: {vecnorm_path}")

    env.close()
    return save_path


def load_and_predict(model_path: str, observation: np.ndarray) -> np.ndarray:
    """저장된 PPO 모델로 행동 추론 (control_service 연동용).

    Args:
        model_path: 모델 zip 파일 경로 (.zip 확장자 생략 가능)
        observation: filtered obs 9개 (np.float32)

    Returns:
        action: cooling_setpoint 값 [20, 30] 범위
    """
    model = PPO.load(model_path)

    stats_path = str(model_path).replace(".zip", "") + "_vecnorm.pkl"
    obs = observation.reshape(1, -1).astype(np.float32)
    if Path(stats_path).exists():
        dummy_env = DummyVecEnv([lambda: DataCenterRLEnv()])
        vec_env = VecNormalize.load(stats_path, dummy_env)
        vec_env.training = False
        vec_env.norm_reward = False
        obs = vec_env.normalize_obs(obs)

    action, _ = model.predict(obs, deterministic=True)
    return action.flatten()


def parse_args():
    parser = argparse.ArgumentParser(description="PPO 학습 — Sinergym datacenter_dx 환경")
    parser.add_argument("--lr", type=float, default=3e-4, help="learning rate")
    parser.add_argument("--n-steps", type=int, default=2048, help="rollout steps")
    parser.add_argument("--batch-size", type=int, default=64, help="mini-batch size")
    parser.add_argument("--gamma", type=float, default=0.9, help="discount factor")
    parser.add_argument("--total-timesteps", type=int, default=50_000, help="총 학습 step")
    parser.add_argument("--max-episode-steps", type=int, default=96, help="에피소드 길이 (96=1일, 15분 간격)")
    parser.add_argument("--w-energy", type=float, default=0.5, help="보상 에너지 항 가중치 (0~1)")
    parser.add_argument("--run-name", type=str, default="ppo-baseline", help="실험 이름")
    parser.add_argument("--device", type=str, default="auto", help="auto|cuda|cpu")
    parser.add_argument("--resume", type=str, default=None, help="이어서 학습할 모델 경로")
    parser.add_argument("--custom-env", action="store_true", help="커스텀 IDC 환경 사용 (Sinergym 대신)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        lr=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=args.gamma,
        total_timesteps=args.total_timesteps,
        max_episode_steps=args.max_episode_steps,
        w_energy=args.w_energy,
        run_name=args.run_name,
        device=args.device,
        resume=args.resume,
        custom_env=args.custom_env,
    )
