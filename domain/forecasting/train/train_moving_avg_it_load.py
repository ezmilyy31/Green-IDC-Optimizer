import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error

from domain.forecasting.moving_avg import MovingAverageForecaster

"""
IT 부하 예측용 이동평균 베이스라인 학습/평가 실행 파일

LGBM 대비 baseline 성능을 측정하기 위한 스크립트.
Simple MA / Seasonal MA 두 가지 모드를 모두 평가하고 비교한다.

<실행 명령어>
uv run python -m domain.forecasting.train.train_moving_avg_it_load
"""

DATA_PATH = "data/processed/synthetic_idc_1year_noisy.parquet"
TARGET_COL = "it_power_kw"
TIMESTAMP_COL = "timestamp"

# 5분 단위 데이터: 하루 = 288 스텝
SEASONAL_PERIOD = 288
# 같은 시간대의 최근 7일 평균
WINDOW = 7
# 마지막 2주를 테스트용으로 분리
TEST_SIZE = 288 * 14

# =========================================================
# 1. 데이터 로드
# =========================================================

if not Path(DATA_PATH).exists():
    raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {DATA_PATH}")

df = pd.read_parquet(DATA_PATH)
df = df[[TIMESTAMP_COL, TARGET_COL]].dropna()
print(f"데이터 로드 완료: {df.shape}")

train_df = df.iloc[:-TEST_SIZE].reset_index(drop=True)
test_df = df.iloc[-TEST_SIZE:].reset_index(drop=True)

print(f"Train: {len(train_df)} rows / Test: {len(test_df)} rows\n")
# (1번 섹션 df.dropna() 아래에 추가)
month_periods = df[TIMESTAMP_COL].dt.to_period("M")

# =========================================================
# 2. 월별 평가 함수
# =========================================================
def evaluate_monthly(model: MovingAverageForecaster, label: str) -> dict:
    print(f"\n[진행] {label} 1~12월 평가 시작...")
    monthly_results = []
    
    for period in sorted(month_periods.unique()):
        month_mask = month_periods == period
        month_df = df[month_mask]
        
        # 이전 모든 데이터를 history로 사용
        history = df.loc[df.index < month_df.index[0], TARGET_COL].to_numpy(dtype=float)
        if len(history) == 0:
            continue
            
        test_series = month_df[TARGET_COL].to_numpy(dtype=float)
        preds = model._forecast(history, len(test_series))
        
        mae_m = mean_absolute_error(test_series, preds)
        mape_all = float(np.mean(np.abs((test_series - preds) / (test_series + 1e-8))) * 100)
        mape_24h = float(np.mean(np.abs((test_series[:288] - preds[:288]) / (test_series[:288] + 1e-8))) * 100)
        
        end_168h = min(2016, len(test_series))
        mape_168h = float(np.mean(np.abs((test_series[:end_168h] - preds[:end_168h]) / (test_series[:end_168h] + 1e-8))) * 100)
        
        month_str = str(period).split('-')[1]
        monthly_results.append((month_str, mae_m, mape_all, mape_24h, mape_168h))

    # --- 출력 ---
    print(f"\n=== 1~12월 모델 평가 비교 요약 ({label}) ===")
    print(f"{'월 (Month)':<10} {'MAE':>10} {'MAPE (전체)':>15} {'MAPE (24h)':>15} {'MAPE (168h)':>15}")
    print("-" * 73)

    total_mae, total_mape_24h, total_mape_168h = [], [], []

    for month_str, mae, mape_all, mape_24h, mape_168h in monthly_results:
        total_mae.append(mae)
        total_mape_24h.append(mape_24h)
        total_mape_168h.append(mape_168h)
        print(f"{month_str}월{' ':<8} {mae:>7.2f} kW {mape_all:>13.2f} % {mape_24h:>13.2f} % {mape_168h:>13.2f} %")

    print("-" * 73)
    mean_mae = np.mean(total_mae) if total_mae else 0.0
    mean_mape_24h = np.mean(total_mape_24h) if total_mape_24h else 0.0
    mean_mape_168h = np.mean(total_mape_168h) if total_mape_168h else 0.0
    
    print(f"연간 평균 MAE       : {mean_mae:.2f} kW")
    print(f"연간 평균 MAPE(24h) : {mean_mape_24h:.2f} %")
    print(f"연간 평균 MAPE(168h): {mean_mape_168h:.2f} %\n")
    
    return {
        "label": label,
        "mae": mean_mae,
        "mape_24h": mean_mape_24h,
        "mape_168h": mean_mape_168h
    }

# =========================================================
# 3. 모델 평가 실행
# =========================================================
simple_model = MovingAverageForecaster(target_name=TARGET_COL, kind="simple", window=WINDOW, seasonal_period=SEASONAL_PERIOD)
simple_result = evaluate_monthly(simple_model, "Simple MA (window=7)")

seasonal_model = MovingAverageForecaster(target_name=TARGET_COL, kind="seasonal", window=WINDOW, seasonal_period=SEASONAL_PERIOD)
seasonal_result = evaluate_monthly(seasonal_model, "Seasonal MA (window=7, period=288)")

# =========================================================
# 4. 비교 요약
# =========================================================
print("=== 베이스라인 모델 비교 요약 ===")
print(f"{'모델':<40} {'연간 MAPE(24h)':>15} {'연간 MAPE(168h)':>15}")
print("-" * 73)
for r in [simple_result, seasonal_result]:
    print(f"{r['label']:<40} {r['mape_24h']:>13.2f}% {r['mape_168h']:>13.2f}%")
print()
print("※ LGBM 결과와 비교하려면 train_lgbm_it_load.py 결과를 참고하세요.")