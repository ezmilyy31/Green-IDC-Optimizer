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

DATA_PATH = "data/processed/synthetic_idc_1year.parquet"
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


def evaluate(model: MovingAverageForecaster, test_series: np.ndarray, label: str) -> dict:
    """
    모델의 one-shot 예측 성능을 출력하고 지표 dict를 반환한다.

    모델을 train 이력으로 fit한 뒤 test 전체를 한 번에 예측한다.
    24h(288 스텝) / 168h(2016 스텝) ahead MAPE를 별도 계산한다.
    """
    history = train_df[TARGET_COL].to_numpy(dtype=float)
    horizon = len(test_series)

    # _forecast는 내부 메서드지만 평가 편의상 직접 호출
    preds = model._forecast(history, horizon)

    mae = mean_absolute_error(test_series, preds)
    rmse = np.sqrt(mean_squared_error(test_series, preds))
    mape_all = float(np.mean(np.abs((test_series - preds) / (test_series + 1e-8))) * 100)

    # 24h ahead (첫 288 스텝)
    mape_24h = float(
        np.mean(np.abs((test_series[:288] - preds[:288]) / (test_series[:288] + 1e-8))) * 100
    )
    # 168h ahead (첫 2016 스텝, test가 2016 스텝 이상인 경우)
    end_168h = min(2016, len(test_series))
    mape_168h = float(
        np.mean(np.abs((test_series[:end_168h] - preds[:end_168h]) / (test_series[:end_168h] + 1e-8))) * 100
    )

    print(f"=== {label} ===")
    print(f"  MAE            : {mae:.2f} kW")
    print(f"  RMSE           : {rmse:.2f} kW")
    print(f"  MAPE (전체)    : {mape_all:.2f} %")
    print(f"  MAPE (24h)     : {mape_24h:.2f} %  (요구사항: 5% 이내)")
    print(f"  MAPE (168h)    : {mape_168h:.2f} %  (요구사항: 8% 이내)")
    print()

    return {
        "label": label,
        "mae": mae,
        "rmse": rmse,
        "mape_all": mape_all,
        "mape_24h": mape_24h,
        "mape_168h": mape_168h,
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
print(f"{'모델':<35} {'MAPE 24h':>10} {'MAPE 168h':>10}")
print("-" * 57)
for r in [simple_result, seasonal_result]:
    print(f"{r['label']:<35} {r['mape_24h']:>9.2f}% {r['mape_168h']:>9.2f}%")
print()
print("※ LGBM 결과와 비교하려면 train_lgbm_it_load.py 결과를 참고하세요.")

# =========================================================
# 5. 최종 모델 저장 (Seasonal MA가 성능이 더 좋으면 저장)
# =========================================================

best = (
    seasonal_model
    if seasonal_result["mape_24h"] <= simple_result["mape_24h"]
    else simple_model
)
best_label = (
    "Seasonal MA" if seasonal_result["mape_24h"] <= simple_result["mape_24h"] else "Simple MA"
)

# 저장 전 전체 데이터로 재fit
best.fit(train_df=df)

save_dir = Path("data/models")
save_dir.mkdir(parents=True, exist_ok=True)
model_path = save_dir / "it_load_moving_avg.joblib"
best.save(str(model_path))

print(f"최종 저장 모델: {best_label} → {model_path}")
