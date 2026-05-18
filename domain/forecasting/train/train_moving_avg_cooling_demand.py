import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error

from domain.forecasting.moving_avg import MovingAverageForecaster

"""
냉각 수요 예측용 이동평균 베이스라인 학습/평가 실행 파일

각 월의 마지막 7일을 테스트 구간으로, 그 이전 전체 데이터를 history로 사용한다.
계절별 성능(여름/겨울 포함)을 종합 평가하며, free cooling 구간(칠러값=0)이 있는
월은 nMAE 대신 "-"로 표시된다.

chiller_power_kw는 free cooling 구간에서 0이 되므로
MAPE 대신 명세서 요구사항 지표인 nMAE(Normalized MAE)를 주 지표로 사용한다.
  nMAE = MAE / mean(actual)  (요구사항: 10% 이내)

<실행 명령어>
uv run python -m domain.forecasting.train.train_moving_avg_cooling_demand
"""

DATA_PATH = "data/weather/synthetic_idc_1year_noisy.parquet"
TARGET_COL = "chiller_power_kw"
TIMESTAMP_COL = "timestamp"

# 5분 단위 데이터: 하루 = 288 스텝
SEASONAL_PERIOD = 288
WINDOW = 7
TEST_DAYS = 7  # 각 월의 마지막 7일을 테스트 구간으로 사용

# =========================================================
# 1. 데이터 로드
# =========================================================

if not Path(DATA_PATH).exists():
    raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {DATA_PATH}")

df = pd.read_parquet(DATA_PATH)
df = df[[TIMESTAMP_COL, TARGET_COL]].dropna()
df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL])
df = df.reset_index(drop=True)
print(f"데이터 로드 완료: {df.shape}")
print(f"chiller_power_kw 분포: 0인 비율 = {(df[TARGET_COL] == 0).mean():.1%} (free cooling 구간)")
print(f"전체 기간: {df[TIMESTAMP_COL].iloc[0].date()} ~ {df[TIMESTAMP_COL].iloc[-1].date()}\n")

month_periods = df[TIMESTAMP_COL].dt.to_period("M")


# =========================================================
# 2. 월별 평가 함수
# =========================================================

def evaluate_monthly(model: MovingAverageForecaster, label: str) -> dict:
    """
    각 월의 마지막 7일을 테스트 구간으로, 그 이전 전체 데이터를 history로 예측한다.
    데이터 누수 없이 계절별 성능을 종합 평가하며, LGBM과 동일한 양식으로 출력한다.
    """
    test_size = SEASONAL_PERIOD * TEST_DAYS
    monthly_results = []
    
    for period in sorted(month_periods.unique()):
        month_mask = month_periods == period
        month_df = df[month_mask]

        if len(month_df) < test_size + 1:
            continue

        # 이 월의 마지막 7일 시작 위치 (reset_index 후 정수 인덱스 기준)
        test_start_idx = month_df.index[-test_size]
        history = df.loc[df.index < test_start_idx, TARGET_COL].to_numpy(dtype=float)

        if len(history) == 0:
            continue

        test_series = month_df.iloc[-test_size:][TARGET_COL].to_numpy(dtype=float)
        preds = model._forecast(history, test_size)

        # --- 월별 평가 지표 계산 ---
        mae_m = mean_absolute_error(test_series, preds)
        mean_act_m = float(np.mean(test_series))
        nmae_m = (mae_m / mean_act_m * 100) if mean_act_m > 0 else 0.0
        
        # 비영 구간 MAPE
        nonzero_mask = test_series > 1.0
        mape_nonzero_m = (
            float(np.mean(np.abs((test_series[nonzero_mask] - preds[nonzero_mask]) / test_series[nonzero_mask])) * 100)
            if nonzero_mask.sum() > 0 else 0.0
        )
        
        # 칠러 미가동률
        free_cooling_ratio = (test_series < 1.0).mean()
        
        # period(YYYY-MM)에서 월만 추출 (예: '2019-01' -> '01')
        month_str = str(period).split('-')[1]
        
        monthly_results.append((month_str, mae_m, nmae_m, mape_nonzero_m, free_cooling_ratio))

    # --- 테이블 출력 (LGBM 포맷 통일) ---
    print(f"\n=== 1~12월 모델 평가 비교 요약 ({label}) ===")
    print(f"{'월 (Month)':<7} {'MAE':>10} {'nMAE (전체)':>18} {'MAPE (비영)':>15} {'칠러 미가동률':>15}")
    print("-" * 75)

    total_mae = []
    valid_nmae_list = []

    for month_str, mae, nmae, mape_nonzero, fc_ratio in monthly_results:
        total_mae.append(mae)
        
        if fc_ratio >= 0.90:
            nmae_str = "(free cooling)"
            mape_str = "- "
        else:
            nmae_str = f"{nmae:.2f} %"
            mape_str = f"{mape_nonzero:.2f} %"
            valid_nmae_list.append(nmae)
            
        print(f"{month_str}월{' ':<8} {mae:>7.2f} kW {nmae_str:>18} {mape_str:>15}  {fc_ratio:>15.1%}")

    print("-" * 75)
    
    mean_mae = np.mean(total_mae) if total_mae else 0.0
    print(f"연간 평균 MAE : {mean_mae:.2f} kW")
    
    if valid_nmae_list:
        mean_valid_nmae = np.mean(valid_nmae_list)
        print(f"유효 nMAE 평균 : {mean_valid_nmae:.2f} % (프리쿨링 월 제외)")
    else:
        mean_valid_nmae = float('inf')
    print()

    return {
        "label": label,
        "mae": mean_mae,
        "nmae": mean_valid_nmae,
    }


# =========================================================
# 3. Simple MA 평가
# =========================================================

simple_model = MovingAverageForecaster(
    target_name=TARGET_COL,
    kind="simple",
    window=WINDOW,
    seasonal_period=SEASONAL_PERIOD,
)
simple_result = evaluate_monthly(simple_model, "Simple MA (window=7)")

# =========================================================
# 4. Seasonal MA 평가
# =========================================================

seasonal_model = MovingAverageForecaster(
    target_name=TARGET_COL,
    kind="seasonal",
    window=WINDOW,
    seasonal_period=SEASONAL_PERIOD,
)
seasonal_result = evaluate_monthly(seasonal_model, "Seasonal MA (window=7, period=288)")

# =========================================================
# 5. 비교 요약
# =========================================================

print("=== 모델 비교 요약 ===")
print(f"{'모델':<40} {'유효 nMAE (평균)':>15}")
print("-" * 57)
for r in [simple_result, seasonal_result]:
    print(f"{r['label']:<40} {r['nmae']:>14.2f}%")
print()
print("※ LGBM 결과와 비교하려면 train_lgbm_cooling_demand.py 결과를 참고하세요.")

# =========================================================
# 6. 최종 모델 저장 (전체 데이터로 재학습)
# =========================================================

best = (
    seasonal_model
    if seasonal_result["nmae"] <= simple_result["nmae"]
    else simple_model
)
best_label = (
    "Seasonal MA" if seasonal_result["nmae"] <= simple_result["nmae"] else "Simple MA"
)

best.fit(train_df=df)

save_dir = Path("data/models")
save_dir.mkdir(parents=True, exist_ok=True)
model_path = save_dir / "cooling_demand_moving_avg.joblib"
best.save(str(model_path))

print(f"최종 저장 모델: {best_label} → {model_path}")
