import numpy as np
import pandas as pd
from pathlib import Path

"""
현실적인 모델 평가를 위한 합성 데이터 노이즈 추가 스크립트

결정론적 합성 데이터가 유발하는 모델의 100% 역산(과적합) 문제를 방지하기 위해, 
현실적인 IDC 센서 오차(가우시안 노이즈)를 타겟 및 주요 피처에 주입

- chiller_power_kw: 타겟 변수, 무작위 오차 주입 (std: 0.4)
- it_power_kw, outside_temp_c 등: 주요 센서 오차 모사
- 물리적 한계치(음수 방지, 비율 0~1) 클리핑(Clipping) 적용 완료
"""

SRC = Path("data/processed/synthetic_idc_1year.parquet")
DST = Path("data/processed/synthetic_idc_1year_noisy.parquet")

df = pd.read_parquet(SRC)
rng = np.random.default_rng(seed=42)  # 재현성

noise_spec = {
    "chiller_power_kw":        {"abs_std": 0.4},   

    "it_power_kw":             {"abs_std": 1.0},   
    
    "outside_temp_c":          {"abs_std": 0.3},   
    "outside_humidity_pct":    {"abs_std": 2.0},   
}

for col, spec in noise_spec.items():
    if col not in df.columns:
        continue
    noise = rng.normal(0.0, spec["abs_std"], size=len(df))
    df[col] = df[col] + noise

# 물리적 제약 복원 (음수 전력 금지, 습도/사용률 범위 등)
df["chiller_power_kw"]     = df["chiller_power_kw"].clip(lower=0.0)
df["it_power_kw"]          = df["it_power_kw"].clip(lower=0.0)
df["fan_power_kw"]         = df["fan_power_kw"].clip(lower=0.0)
df["total_power_kw"]       = df["total_power_kw"].clip(lower=0.0)
df["outside_humidity_pct"] = df["outside_humidity_pct"].clip(0.0, 100.0)
df["cpu_utilization"]      = df["cpu_utilization"].clip(0.0, 1.0)

df.to_parquet(DST, index=False)
print(f"Saved: {DST}  shape={df.shape}")