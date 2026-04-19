"""학습된 RL 모델이 실제로 어떤 supply_temp를 출력하는지 확인."""
import argparse
from pathlib import Path
import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from domain.controllers.idc_env import IDCEnv


def check(model_path: str, n_episodes: int = 3):
    vecnorm_path = model_path.replace(".zip", "_vecnorm.pkl")
    is_sac = not Path(vecnorm_path).exists()

    if is_sac:
        model = SAC.load(model_path)
        vec_env = None
        print("[check_action] SAC 모델 감지 (vecnorm 없음)")
    else:
        model = PPO.load(model_path)
        dummy = DummyVecEnv([lambda: Monitor(IDCEnv())])
        vec_env = VecNormalize.load(vecnorm_path, dummy)
        vec_env.training = False
        vec_env.norm_reward = False

    env = IDCEnv(w_energy=0.5)
    all_actions = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=42 + ep)
        ep_actions = []
        while True:
            if vec_env is not None:
                o = vec_env.normalize_obs(obs.reshape(1, -1).astype(np.float32))
            else:
                o = obs.reshape(1, -1).astype(np.float32)
            action, _ = model.predict(o, deterministic=True)
            supply = float(action.flatten()[0])
            ep_actions.append(supply)
            obs, _, terminated, truncated, _ = env.step(action.flatten())
            if terminated or truncated:
                break
        all_actions.extend(ep_actions)
        print(f"ep{ep+1}: mean={np.mean(ep_actions):.2f} min={np.min(ep_actions):.2f} "
              f"max={np.max(ep_actions):.2f} std={np.std(ep_actions):.2f}")

    a = np.array(all_actions)
    print(f"\n전체 {len(a)}스텝 | mean={a.mean():.3f} min={a.min():.3f} "
          f"max={a.max():.3f} std={a.std():.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args()
    check(args.model, args.episodes)
