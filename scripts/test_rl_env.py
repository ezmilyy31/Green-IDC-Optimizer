"""래퍼 동작 검증 스크립트.

실행: docker-compose run --rm simulation-service python3 -m scripts.test_rl_env
"""

from domain.controllers.rl_env import DataCenterRLEnv
from core.schemas.rl_interface import FILTERED_OBS_KEYS

# 1일 = 144 steps (10분 간격)
env = DataCenterRLEnv(max_episode_steps=144)

print(f"Filtered obs keys ({len(FILTERED_OBS_KEYS)}): {FILTERED_OBS_KEYS}")
print(f"Observation space: {env.observation_space}")
print(f"Action space: {env.action_space}")

# 3 에피소드 돌려서 reward 스케일 확인
for ep in range(3):
    obs, info = env.reset()
    ep_reward = 0.0
    steps = 0
    max_temp = 0.0

    while True:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        ep_reward += reward
        steps += 1

        # east/west zone temp (인덱스 4, 5)
        zone_temp = max(obs[4], obs[5])
        max_temp = max(max_temp, zone_temp)

        if terminated or truncated:
            break

    print(
        f"Episode {ep + 1}: steps={steps}, "
        f"total_reward={ep_reward:.2f}, "
        f"avg_reward={ep_reward / steps:.4f}, "
        f"max_zone_temp={max_temp:.1f}°C"
    )

env.close()
print("\nDone!")
