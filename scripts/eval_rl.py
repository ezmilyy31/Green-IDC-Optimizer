"""RL 모델 평가 스크립트 — RL vs 베이스라인 PUE overhead 비교.

사용법:
    # IDCEnv (커스텀)
    python scripts/eval_rl.py --custom-env --model data/models/chuncheon-w05.zip
    # Sinergym
    python scripts/eval_rl.py --model data/models/exp-300k-pue.zip
    # 베이스라인만
    python scripts/eval_rl.py --custom-env --baseline-only
"""

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from domain.controllers.idc_env import IDCEnv


def _load_sinergym_env():
    from domain.controllers.rl_env import DataCenterRLEnv
    return DataCenterRLEnv


def make_vec_env(custom_env: bool, max_episode_steps: int = 288):
    if custom_env:
        return DummyVecEnv([lambda: IDCEnv(max_episode_steps=max_episode_steps)])
    else:
        Env = _load_sinergym_env()
        return DummyVecEnv([lambda: Env(max_episode_steps=max_episode_steps)])


def evaluate(model_path: str, vecnorm_path: str | None, n_episodes: int, custom_env: bool) -> None:
    vec_env = make_vec_env(custom_env)

    stats_path = vecnorm_path or str(model_path).replace(".zip", "_vecnorm.pkl")
    if Path(stats_path).exists():
        vec_env = VecNormalize.load(stats_path, vec_env)
    else:
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False)
    vec_env.training = False
    vec_env.norm_reward = False

    model = PPO.load(model_path, env=vec_env)
    pue_means, viol_means = [], []

    for ep in range(n_episodes):
        obs = vec_env.reset()
        done = False
        ep_pue, ep_viol = [], []

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, dones, infos = vec_env.step(action)
            done = dones[0]
            info = infos[0]

            if custom_env:
                ep_pue.append(info["pue"] - 1.0)
                ep_viol.append(1 if info["temp_violation"] > 0 else 0)
            else:
                raw = vec_env.get_original_obs()[0]
                cpu_load, hvac = raw[7], raw[8]
                east, west = raw[4], raw[5]
                ep_pue.append(hvac / (cpu_load * 250_000 + 1e-6))
                ep_viol.append(0 if (18.0 <= east <= 27.0 and 18.0 <= west <= 27.0) else 1)

        pue_means.append(np.mean(ep_pue))
        viol_means.append(np.mean(ep_viol))
        print(f"  ep {ep+1:2d} | pue_overhead={np.mean(ep_pue):.4f}  violation={np.mean(ep_viol)*100:.1f}%")

    print("\n=== RL 평균 결과 ===")
    print(f"  PUE overhead : {np.mean(pue_means):.4f}  (PUE ≈ {1 + np.mean(pue_means):.4f})")
    print(f"  온도 위반율  : {np.mean(viol_means)*100:.2f}%")
    vec_env.close()


def evaluate_baseline(n_episodes: int, setpoint: float, custom_env: bool) -> None:
    vec_env = make_vec_env(custom_env)
    pue_means, viol_means = [], []
    action = np.array([[setpoint]], dtype=np.float32)

    for ep in range(n_episodes):
        obs = vec_env.reset()
        done = False
        ep_pue, ep_viol = [], []

        while not done:
            obs, _, dones, infos = vec_env.step(action)
            done = dones[0]
            info = infos[0]

            if custom_env:
                ep_pue.append(info["pue"] - 1.0)
                ep_viol.append(1 if info["temp_violation"] > 0 else 0)
            else:
                raw = obs[0]
                cpu_load, hvac = raw[7], raw[8]
                east, west = raw[4], raw[5]
                ep_pue.append(hvac / (cpu_load * 250_000 + 1e-6))
                ep_viol.append(0 if (18.0 <= east <= 27.0 and 18.0 <= west <= 27.0) else 1)

        pue_means.append(np.mean(ep_pue))
        viol_means.append(np.mean(ep_viol))
        print(f"  ep {ep+1:2d} | pue_overhead={np.mean(ep_pue):.4f}  violation={np.mean(ep_viol)*100:.1f}%")

    print("\n=== 베이스라인 평균 결과 ===")
    print(f"  PUE overhead : {np.mean(pue_means):.4f}  (PUE ≈ {1 + np.mean(pue_means):.4f})")
    print(f"  온도 위반율  : {np.mean(viol_means)*100:.2f}%")
    vec_env.close()


def parse_args():
    parser = argparse.ArgumentParser(description="RL 모델 평가 — PUE overhead 비교")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--vecnorm", type=str, default=None, help="VecNormalize 통계 경로 (.pkl)")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--custom-env", action="store_true", help="IDCEnv 사용 (기본: Sinergym)")
    parser.add_argument("--setpoint", type=float, default=22.0, help="베이스라인 고정 setpoint (°C)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not args.baseline_only:
        if not args.model:
            raise ValueError("--model 경로를 지정하세요.")
        print(f"=== RL ({args.model}) ===")
        evaluate(args.model, args.vecnorm, args.episodes, args.custom_env)
        print()

    print(f"=== 베이스라인 (고정 {args.setpoint}°C) ===")
    evaluate_baseline(args.episodes, args.setpoint, args.custom_env)
