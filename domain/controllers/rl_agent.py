"""
# rl_agent.py — 학습 스크립트
from stable_baselines3 import PPO
from domain.controllers.rl_env import DataCenterRLEnv

env = DataCenterRLEnv()
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=50000)
model.save("data/models/ppo_datacenter")
"""