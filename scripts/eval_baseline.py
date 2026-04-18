"""베이스라인 비교 평가 스크립트.

Rule-based, Random, RL 모델을 같은 IDCEnv에서 평가하여 비교한다.

사용법:
    python scripts/eval_baseline.py
    python scripts/eval_baseline.py --model data/models/exp-custom-idc.zip
"""

import argparse
import numpy as np
from pathlib import Path

from domain.controllers.idc_env import IDCEnv, T_SUPPLY_MIN, T_SUPPLY_MAX
from domain.controllers.rule_based import run_rule_based


def evaluate(env: IDCEnv, policy_fn, n_episodes: int = 20, seed: int = 42) -> dict:
    rewards, pues, violations, zone_temps = [], [], [], []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        ep_reward, ep_pues, ep_violations, ep_zone_temps = 0.0, [], [], []
        while True:
            action = policy_fn(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            ep_pues.append(info["pue"])
            ep_violations.append(info["temp_violation"])
            ep_zone_temps.append(info["zone_temp_c"])
            if terminated or truncated:
                break
        rewards.append(ep_reward)
        pues.append(np.mean(ep_pues))
        violations.append(np.mean(ep_violations))
        zone_temps.append(np.mean(ep_zone_temps))
    return {
        "ep_rew_mean": np.mean(rewards),
        "ep_rew_std": np.std(rewards),
        "pue_mean": np.mean(pues),
        "temp_violation_mean": np.mean(violations),
        "zone_temp_mean": np.mean(zone_temps),
    }


def rule_based_policy(obs):
    result = run_rule_based(float(obs[1]), float(obs[2]), float(obs[6]))
    return np.array([np.clip(result.supply_air_temp_setpoint_c, T_SUPPLY_MIN, T_SUPPLY_MAX)], dtype=np.float32)


def random_policy(obs):
    return np.array([np.random.uniform(T_SUPPLY_MIN, T_SUPPLY_MAX)], dtype=np.float32)


def fixed_policy(setpoint):
    return lambda obs: np.array([setpoint], dtype=np.float32)


def rl_policy(model_path):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from stable_baselines3.common.monitor import Monitor
    model = PPO.load(model_path)
    vecnorm_path = model_path.replace(".zip", "_vecnorm.pkl")
    vec_env = None
    if Path(vecnorm_path).exists():
        dummy = DummyVecEnv([lambda: Monitor(IDCEnv())])
        vec_env = VecNormalize.load(vecnorm_path, dummy)
        vec_env.training = False
        vec_env.norm_reward = False
    def policy(obs):
        o = vec_env.normalize_obs(obs.reshape(1, -1).astype(np.float32)) if vec_env else obs.reshape(1, -1)
        action, _ = model.predict(o, deterministic=True)
        return action.flatten()
    return policy


def print_result(name, result):
    print(f"\n{'─' * 50}")
    print(f"  {name}")
    print(f"{'─' * 50}")
    print(f"  ep_rew_mean:      {result['ep_rew_mean']:.3f} ± {result['ep_rew_std']:.3f}")
    print(f"  PUE 평균:         {result['pue_mean']:.4f}")
    print(f"  온도 위반 평균:    {result['temp_violation_mean']:.4f} °C")
    print(f"  서버실 온도 평균:  {result['zone_temp_mean']:.2f} °C")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="data/models/exp-custom-idc.zip")
    parser.add_argument("--episodes", type=int, default=20)
    args = parser.parse_args()

    env = IDCEnv(w_energy=0.5)
    print(f"\n{'=' * 50}")
    print("  IDCEnv 베이스라인 비교 평가 (평가 환경: w_energy=0.5 고정)")
    print(f"  에피소드 수: {args.episodes} (1일 × {args.episodes})")
    print(f"{'=' * 50}")

    print_result("Rule-based", evaluate(env, rule_based_policy, args.episodes))
    print_result("고정 setpoint 20°C (설계값)", evaluate(env, fixed_policy(20.0), args.episodes))
    print_result("고정 setpoint 24°C", evaluate(env, fixed_policy(24.0), args.episodes))
    print_result("Random", evaluate(env, random_policy, args.episodes))
    if Path(args.model).exists():
        print_result(f"PPO RL ({Path(args.model).stem})", evaluate(env, rl_policy(args.model), args.episodes))
    else:
        print(f"\n⚠ RL 모델 없음: {args.model}")
    print(f"\n{'=' * 50}\n")


if __name__ == "__main__":
    main()
