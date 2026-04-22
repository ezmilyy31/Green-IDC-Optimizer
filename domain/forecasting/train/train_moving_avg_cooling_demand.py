import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error

from domain.forecasting.moving_avg import MovingAverageForecaster

"""
냉각 수요 예측용 이동평균 베이스라인 학습/평가 실행 파일

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
# 테스트 구간: 8~9월 2주 사용 (칠러 가동률이 높은 여름 구간)
# 12월 말로 고정하면 칠러 값이 전부 0이어서 nMAE 계산 불가
TEST_START = "2024-08-01"
TEST_END = "2024-08-14 23:55:00"

# =========================================================
# 1. 데이터 로드
# =========================================================

if not Path(DATA_PATH).exists():
    raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {DATA_PATH}")

df = pd.read_parquet(DATA_PATH)
df = df[[TIMESTAMP_COL, TARGET_COL]].dropna()
df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL])
print(f"데이터 로드 완료: {df.shape}")
print(f"chiller_power_kw 분포: 0인 비율 = {(df[TARGET_COL] == 0).mean():.1%} (free cooling 구간)")

# 테스트 구간: 8월 1~14일 (칠러 가동 구간)
# 학습: 테스트 시작 이전 전체 데이터
test_mask = (df[TIMESTAMP_COL] >= TEST_START) & (df[TIMESTAMP_COL] <= TEST_END)
test_df = df[test_mask].reset_index(drop=True)
train_df = df[df[TIMESTAMP_COL] < TEST_START].reset_index(drop=True)

print(f"\nTrain: {len(train_df)} rows ({train_df[TIMESTAMP_COL].iloc[0].date()} ~ {train_df[TIMESTAMP_COL].iloc[-1].date()})")
print(f"Test : {len(test_df)} rows ({test_df[TIMESTAMP_COL].iloc[0].date()} ~ {test_df[TIMESTAMP_COL].iloc[-1].date()})")
print(f"Test 칠러 평균: {test_df[TARGET_COL].mean():.2f} kW / 0인 비율: {(test_df[TARGET_COL] == 0).mean():.1%}\n")


def evaluate(model: MovingAverageForecaster, test_series: np.ndarray, label: str) -> dict:
    """
    냉각 수요 모델 평가.

    chiller_power_kw는 0값이 많아 MAPE가 불안정하므로
    nMAE를 주 지표, MAPE는 비영(non-zero) 구간에서만 보조 계산한다.
    """
    history = train_df[TARGET_COL].to_numpy(dtype=float)
    horizon = len(test_series)
    preds = model._forecast(history, horizon)

    mae = mean_absolute_error(test_series, preds)
    rmse = np.sqrt(mean_squared_error(test_series, preds))

    # nMAE: 명세서 요구사항 지표 (MAE / 실제값 평균)
    mean_actual = float(np.mean(test_series))
    nmae = (mae / mean_actual * 100) if mean_actual > 0 else float("inf")

    # MAPE: 0이 아닌 구간에서만 계산 (보조 지표)
    nonzero_mask = test_series > 1.0
    if nonzero_mask.sum() > 0:
        mape_nonzero = float(
            np.mean(
                np.abs((test_series[nonzero_mask] - preds[nonzero_mask]) / test_series[nonzero_mask])
            ) * 100
        )
    else:
        mape_nonzero = float("inf")

    # 24h / 168h nMAE
    def nmae_slice(actual, predicted):
        m = float(np.mean(actual))
        return (mean_absolute_error(actual, predicted) / m * 100) if m > 0 else float("inf")

    nmae_24h = nmae_slice(test_series[:288], preds[:288])
    end_168h = min(2016, len(test_series))
    nmae_168h = nmae_slice(test_series[:end_168h], preds[:end_168h])

    print(f"=== {label} ===")
    print(f"  MAE                      : {mae:.2f} kW")
    print(f"  RMSE                     : {rmse:.2f} kW")
    print(f"  nMAE (전체)              : {nmae:.2f} %  (요구사항: 10% 이내)")
    print(f"  nMAE (24h)               : {nmae_24h:.2f} %")
    print(f"  nMAE (168h)              : {nmae_168h:.2f} %")
    print(f"  MAPE (비영 구간 한정)    : {mape_nonzero:.2f} %")
    print()

    return {
        "label": label,
        "mae": mae,
        "rmse": rmse,
        "nmae": nmae,
        "nmae_24h": nmae_24h,
        "nmae_168h": nmae_168h,
        "mape_nonzero": mape_nonzero,
    }


# =========================================================
# 2. Simple MA 평가
# =========================================================

simple_model = MovingAverageForecaster(
    target_name=TARGET_COL,
    kind="simple",
    window=WINDOW,
    seasonal_period=SEASONAL_PERIOD,
)
simple_model.fit(train_df=train_df)
simple_result = evaluate(simple_model, test_df[TARGET_COL].to_numpy(), "Simple MA (window=7)")

# =========================================================
# 3. Seasonal MA 평가
# =========================================================

seasonal_model = MovingAverageForecaster(
    target_name=TARGET_COL,
    kind="seasonal",
    window=WINDOW,
    seasonal_period=SEASONAL_PERIOD,
)
seasonal_model.fit(train_df=train_df)
seasonal_result = evaluate(seasonal_model, test_df[TARGET_COL].to_numpy(), "Seasonal MA (window=7, period=288)")

# =========================================================
# 4. 비교 요약
# =========================================================

print("=== 모델 비교 요약 ===")
print(f"{'모델':<40} {'nMAE 24h':>10} {'nMAE 168h':>11}")
print("-" * 63)
for r in [simple_result, seasonal_result]:
    print(f"{r['label']:<40} {r['nmae_24h']:>9.2f}% {r['nmae_168h']:>10.2f}%")
print()
print("※ LGBM 결과와 비교하려면 train_lgbm_cooling_demand.py 결과를 참고하세요.")

# =========================================================
# 5. 최종 모델 저장
# =========================================================

best = (
    seasonal_model
    if seasonal_result["nmae"] <= simple_result["nmae"]
    else simple_model
)
best_label = (
    "Seasonal MA" if seasonal_result["nmae"] <= simple_result["nmae"] else "Simple MA"
)

best.fit(train_df=df.reset_index(drop=True))

save_dir = Path("data/models")
save_dir.mkdir(parents=True, exist_ok=True)
model_path = save_dir / "cooling_demand_moving_avg.joblib"
best.save(str(model_path))

print(f"최종 저장 모델: {best_label} → {model_path}")
