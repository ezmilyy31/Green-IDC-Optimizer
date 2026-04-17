"""RL 모델 평가 스크립트 — RL vs 베이스라인 PUE overhead 비교.

사용법:
    python -m scripts.eval_rl --model data/models/exp-300k-pue --episodes 10
    python -m scripts.eval_rl --baseline-only --episodes 10
"""

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from domain.controllers.rl_env import DataCenterRLEnv

IT_POWER_MAX = 250_000  # W, CPU 400 + GPU 20대 기준


def evaluate(model_path: str, n_episodes: int = 10) -> None:
    """학습된 모델 평가 — PUE overhead 및 온도 위반율 출력."""
    env = DummyVecEnv([lambda: DataCenterRLEnv(max_episode_steps=96)])
    stats_path = str(model_path).replace(".zip", "") + "_vecnorm.pkl"
    if Path(stats_path).exists():
        env = VecNormalize.load(stats_path, env)
    else:
        env = VecNormalize(env, norm_obs=False, norm_reward=False)
    env.training = False
    env.norm_reward = False

    model = PPO.load(model_path, env=env)
    hvac_means, pue_means, viol_means = [], [], []

    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        ep_hvac, ep_pue, ep_viol = [], [], []

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, dones, _ = env.step(action)
            done = dones[0]

            raw = env.get_original_obs()[0]
            east, west = raw[4], raw[5]
            cpu_load, hvac = raw[7], raw[8]

            pue_oh = hvac / (cpu_load * IT_POWER_MAX + 1e-6)
            in_range = (18.0 <= east <= 27.0) and (18.0 <= west <= 27.0)

            ep_hvac.append(hvac)
            ep_pue.append(pue_oh)
            ep_viol.append(0 if in_range else 1)

        hvac_means.append(np.mean(ep_hvac))
        pue_means.append(np.mean(ep_pue))
        viol_means.append(np.mean(ep_viol))
        print(f"  ep {ep+1:2d} | hvac={np.mean(ep_hvac)/1000:.1f}kW  pue_overhead={np.mean(ep_pue):.3f}  violation={np.mean(ep_viol)*100:.1f}%")

    print("\n=== RL 평균 결과 ===")
    print(f"  HVAC 전력    : {np.mean(hvac_means)/1000:.2f} kW")
    print(f"  PUE overhead : {np.mean(pue_means):.4f}  (PUE ≈ {1 + np.mean(pue_means):.4f})")
    print(f"  온도 위반율  : {np.mean(viol_means)*100:.2f}%")
    env.close()


def evaluate_baseline(n_episodes: int = 10, setpoint: float = 24.0) -> None:
    """고정 setpoint 베이스라인 평가 — RL 비교용."""
    env = DummyVecEnv([lambda: DataCenterRLEnv(max_episode_steps=96)])
    hvac_means, pue_means, viol_means = [], [], []
    action = np.array([[setpoint]], dtype=np.float32)

    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        ep_hvac, ep_pue, ep_viol = [], [], []

        while not done:
            obs, _, dones, _ = env.step(action)
            done = dones[0]

            raw = obs[0]
            east, west = raw[4], raw[5]
            cpu_load, hvac = raw[7], raw[8]

            pue_oh = hvac / (cpu_load * IT_POWER_MAX + 1e-6)
            in_range = (18.0 <= east <= 27.0) and (18.0 <= west <= 27.0)

            ep_hvac.append(hvac)
            ep_pue.append(pue_oh)
            ep_viol.append(0 if in_range else 1)

        hvac_means.append(np.mean(ep_hvac))
        pue_means.append(np.mean(ep_pue))
        viol_means.append(np.mean(ep_viol))
        print(f"  ep {ep+1:2d} | hvac={np.mean(ep_hvac)/1000:.1f}kW  pue_overhead={np.mean(ep_pue):.3f}  violation={np.mean(ep_viol)*100:.1f}%")

    print("\n=== 베이스라인 평균 결과 ===")
    print(f"  HVAC 전력    : {np.mean(hvac_means)/1000:.2f} kW")
    print(f"  PUE overhead : {np.mean(pue_means):.4f}  (PUE ≈ {1 + np.mean(pue_means):.4f})")
    print(f"  온도 위반율  : {np.mean(viol_means)*100:.2f}%")
    env.close()


def parse_args():
    parser = argparse.ArgumentParser(description="RL 모델 평가 — PUE overhead 비교")
    parser.add_argument("--model", type=str, default=None, help="모델 경로 (.zip 생략 가능)")
    parser.add_argument("--episodes", type=int, default=10, help="평가 에피소드 수")
    parser.add_argument("--baseline-only", action="store_true", help="베이스라인만 평가")
    parser.add_argument("--setpoint", type=float, default=24.0, help="베이스라인 고정 setpoint (°C)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not args.baseline_only:
        if not args.model:
            raise ValueError("--model 경로를 지정하세요.")
        print(f"=== RL ({args.model}) ===")
        evaluate(args.model, args.episodes)
        print()

    print(f"=== 베이스라인 (고정 {args.setpoint}°C) ===")
    evaluate_baseline(args.episodes, args.setpoint)
