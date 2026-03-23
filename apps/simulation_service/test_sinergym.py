import gymnasium as gym
import sinergym

# '7. 제약사항'에 언급된 시뮬레이터 환경
env_name = 'Eplus-datacenter_dx-mixed-continuous-stochastic-v1'
env = gym.make(env_name)

# 제대로 로드되었는지 테스트
print(f"Sinergym 환경 '{env_name}' 로드 완료.")
obs, info = env.reset()
print(f"초기 관측치: {obs}") 

env.close()