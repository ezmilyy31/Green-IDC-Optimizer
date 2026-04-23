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

DATA_PATH = "data/processed/synthetic_idc_1year.parquet"
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
    데이터 누수 없이 계절별 성능을 종합 평가한다.
    """
    test_size = SEASONAL_PERIOD * TEST_DAYS
    all_actual: list[float] = []
    all_preds: list[float] = []
    monthly_rows: list[tuple[str, str]] = []

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

        all_actual.extend(test_series)
        all_preds.extend(preds)

        mae_m = mean_absolute_error(test_series, preds)
        mean_act_m = float(np.mean(test_series))
        if mean_act_m > 0:
            nmae_m_str = f"{mae_m / mean_act_m * 100:.2f} %"
        else:
            nmae_m_str = "-  (free cooling)"
        monthly_rows.append((str(period), nmae_m_str))

    actual_arr = np.array(all_actual)
    preds_arr = np.array(all_preds)

    mae = mean_absolute_error(actual_arr, preds_arr)
    rmse = np.sqrt(mean_squared_error(actual_arr, preds_arr))
    mean_actual = float(np.mean(actual_arr))
    nmae = (mae / mean_actual * 100) if mean_actual > 0 else float("inf")

    nonzero = actual_arr > 1.0
    mape_nonzero = (
        float(np.mean(np.abs((actual_arr[nonzero] - preds_arr[nonzero]) / actual_arr[nonzero])) * 100)
        if nonzero.sum() > 0 else float("inf")
    )

    print(f"=== {label} ===")
    print(f"  MAE                      : {mae:.2f} kW")
    print(f"  RMSE                     : {rmse:.2f} kW")
    print(f"  nMAE (전체)              : {nmae:.2f} %  (요구사항: 10% 이내)")
    print(f"  MAPE (비영 구간 한정)    : {mape_nonzero:.2f} %")
    print()
    print("  [월별 nMAE]")
    for month_label, nmae_str in monthly_rows:
        print(f"    {month_label}: {nmae_str}")
    print()

    return {
        "label": label,
        "mae": mae,
        "rmse": rmse,
        "nmae": nmae,
        "mape_nonzero": mape_nonzero,
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
print(f"{'모델':<40} {'nMAE (전체)':>12}")
print("-" * 54)
for r in [simple_result, seasonal_result]:
    print(f"{r['label']:<40} {r['nmae']:>11.2f}%")
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
