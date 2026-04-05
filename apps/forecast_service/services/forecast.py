"""
Forecast Service orchestration layer.

이 파일의 책임
--------------
- API 요청(payload)을 받아 예측 흐름을 조합한다.
- target(it_load / cooling_demand / both)에 따라 필요한 모델을 선택한다.
- 요청의 model_type(lgbm / lstm)에 따라 적절한 모델을 고른다.
- feature frame 생성 함수를 호출한다.
- 모델 추론 결과를 ForecastResponse 형태로 조립한다.
- cooling mode를 rule-based로 판정한다.
- 필요 시 prediction interval을 결과에 포함한다.
"""

import inspect
import math
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core.config.enums import CoolingMode, ModelType, PredictionTarget
from core.config.constants import FREE_COOLING_THRESHOLD_C, HYBRID_THRESHOLD_C
from core.schemas.forecast import ForecastPoint, ForecastRequest, ForecastResponse
<<<<<<< Updated upstream
=======

from domain.forecasting.features.builder import _build_next_feature_row


>>>>>>> Stashed changes
STEP_MINUTES = 5

# 내부 학습 타깃 컬럼명
IT_TARGET_COL = "it_power_kw"
COOLING_TARGET_COL = "chiller_power_kw"  # 필요시 "cooling_load_kw"로 변경
DATA_PATH = "./data/processed/synthetic_idc_1year.parquet"

# =========================================================
# Public entrypoint
# =========================================================
def run_forecast(
    model_bundle: dict[str, Any],
    request: ForecastRequest,
) -> ForecastResponse:
    """
    Forecast Service orchestration layer.

    역할
    ----
    - request 해석
    - target / model_type별 모델 선택
    - 최근 이력 데이터 로드
    - IT 부하 / 냉각 수요 forecast 수행
    - prediction interval / cooling mode 포함하여 ForecastResponse 반환
    """
    current_ts = _align_timestamp(request.current_timestamp or datetime.utcnow())
    horizon_hours = request.forecast_horizon_hours
    horizon_steps = int(horizon_hours * 60 / STEP_MINUTES)

    history_df = _load_recent_history_window(
        current_timestamp=current_ts,
        lookback_steps=400,  # lag_288, rolling_288 고려해서 여유 있게
    )
    if history_df.empty:
        raise ValueError("No recent history data available for forecast.")

    defaults = model_bundle.get("defaults", {}) or {}
    interval_cfg = model_bundle.get("interval", {}) or {}
    weather_df = _prepare_weather_df(model_bundle.get("weather", {}), STEP_MINUTES)

    prediction_target = request.prediction_target
    model_type = request.model_type

    it_result_df: pd.DataFrame | None = None
    cooling_result_df: pd.DataFrame | None = None

    # cooling forecast에는 future predicted_it_power_kw가 필요할 수 있으므로
    # target == cooling_demand 여도 내부적으로 IT forecast를 먼저 돌릴 수 있음.
    if prediction_target in {
        PredictionTarget.IT_LOAD,
        PredictionTarget.BOTH,
        PredictionTarget.COOLING_DEMAND,
    }:
        it_model = _select_model(model_bundle, "it_load", model_type)
        if it_model is None and prediction_target in {
            PredictionTarget.IT_LOAD,
            PredictionTarget.BOTH,
            PredictionTarget.COOLING_DEMAND,
        }:
            raise ValueError(f"No model found for target='it_load', model_type='{model_type.value}'.")

        if it_model is not None:
            it_result_df = _forecast_one_target(
                target_name="it_load",
                target_col=IT_TARGET_COL,
                model=it_model,
                history_df=history_df.copy(),
                horizon_steps=horizon_steps,
                weather_df=weather_df,
                defaults=defaults,
                auxiliary_it_map=None,
            )

    if prediction_target in {PredictionTarget.COOLING_DEMAND, PredictionTarget.BOTH}:
        cooling_model = _select_model(model_bundle, "cooling_demand", model_type)
        if cooling_model is None:
            raise ValueError(f"No model found for target='cooling_demand', model_type='{model_type.value}'.")

        cooling_history_df = history_df.copy()

        # 냉각 모델이 predicted_it_power_kw를 요구할 수 있으므로
        # 과거 구간에서는 observed it_power_kw를 채워둠
        if "predicted_it_power_kw" not in cooling_history_df.columns and IT_TARGET_COL in cooling_history_df.columns:
            cooling_history_df["predicted_it_power_kw"] = cooling_history_df[IT_TARGET_COL]

        auxiliary_it_map = None
        if it_result_df is not None and not it_result_df.empty:
            auxiliary_it_map = {
                pd.Timestamp(row["timestamp"]): float(row["prediction"])
                for _, row in it_result_df.iterrows()
            }

        cooling_result_df = _forecast_one_target(
            target_name="cooling_demand",
            target_col=COOLING_TARGET_COL,
            model=cooling_model,
            history_df=cooling_history_df,
            horizon_steps=horizon_steps,
            weather_df=weather_df,
            defaults=defaults,
            auxiliary_it_map=auxiliary_it_map,
        )

    response_points = _merge_predictions_to_points(
        prediction_target=prediction_target,
        it_result_df=it_result_df if prediction_target in {PredictionTarget.IT_LOAD, PredictionTarget.BOTH} else None,
        cooling_result_df=cooling_result_df,
        include_prediction_interval=request.include_prediction_interval,
        interval_cfg=interval_cfg,
    )

    return ForecastResponse(
        prediction_target=prediction_target,
        model_type_used=model_type,
        generated_at=datetime.utcnow(),
        horizon_hours=horizon_hours,
        predictions=response_points,
    )


# =========================================================
# Forecast execution
# =========================================================
def _forecast_one_target(
    target_name: str,
    target_col: str,
    model: Any,
    history_df: pd.DataFrame,
    horizon_steps: int,
    weather_df: pd.DataFrame,
    defaults: dict[str, Any],
    auxiliary_it_map: dict[pd.Timestamp, float] | None,
) -> pd.DataFrame:
    """
    단일 target forecast 실행
    (IT_load / cooling_demand 중 하나에 대해 예측 수행)

    반환 컬럼:
    - timestamp
    - prediction
    - target
    - outdoor_temp_c (가능한 경우)
    """
    history_df = history_df.sort_values("timestamp").reset_index(drop=True)

    make_next_feature_row = lambda simulated_history, step: _build_next_feature_row(
        target_name=target_name,
        target_col=target_col,
        model=model,
        simulated_history=simulated_history,
        weather_df=weather_df,
        defaults=defaults,
        auxiliary_it_map=auxiliary_it_map,
    )

    raw_df = _run_model_forecast(
        model=model,
        history_df=history_df,
        horizon_steps=horizon_steps,
        make_next_feature_row=make_next_feature_row,
        target_col=target_col,
    )

    prediction_col = _find_prediction_column(raw_df, target_col)

    result = raw_df.rename(columns={prediction_col: "prediction"}).copy()
    result["target"] = target_name

    if "timestamp" not in result.columns:
        last_ts = pd.Timestamp(history_df["timestamp"].iloc[-1])
        result["timestamp"] = [
            last_ts + timedelta(minutes=STEP_MINUTES * i)
            for i in range(1, len(result) + 1)
        ]

    keep_cols = ["timestamp", "prediction", "target"]
    if "outdoor_temp_c" in result.columns:
        keep_cols.append("outdoor_temp_c")

    return result[keep_cols].reset_index(drop=True)

# model 객체가 가진 attribute에 따라 적절한 예측 방식 선택
def _run_model_forecast(
    model: Any,
    history_df: pd.DataFrame,
    horizon_steps: int,
    make_next_feature_row,
    target_col: str,
) -> pd.DataFrame:
    """
    모델별 forecast 실행.
    우선순위:
    1) forecast_recursive(...)
    2) forecast(...)
    3) predict(...) 기반 수동 recursive loop
    """
    if hasattr(model, "forecast_recursive"):
        return model.forecast_recursive(
            history_df=history_df,
            horizon=horizon_steps,
            make_next_feature_row=make_next_feature_row,
            timestamp_col="timestamp",
        )

    if hasattr(model, "forecast"):
        forecast_fn = getattr(model, "forecast")
        sig = inspect.signature(forecast_fn)

        kwargs: dict[str, Any] = {}
        if "history_df" in sig.parameters:
            kwargs["history_df"] = history_df
        if "horizon" in sig.parameters:
            kwargs["horizon"] = horizon_steps
        if "horizon_steps" in sig.parameters:
            kwargs["horizon_steps"] = horizon_steps
        if "make_next_feature_row" in sig.parameters:
            kwargs["make_next_feature_row"] = make_next_feature_row
        if "timestamp_col" in sig.parameters:
            kwargs["timestamp_col"] = "timestamp"

        if kwargs:
            output = forecast_fn(**kwargs)
        else:
            output = forecast_fn(history_df, horizon_steps)

        if isinstance(output, pd.DataFrame):
            return output

        raise TypeError("Model.forecast(...) must return a pandas DataFrame.")

    if hasattr(model, "predict"):
        return _manual_recursive_predict(
            model=model,
            history_df=history_df,
            horizon_steps=horizon_steps,
            make_next_feature_row=make_next_feature_row,
            target_col=target_col,
        )

    raise TypeError("Selected model does not support forecast/predict interface.")


def _manual_recursive_predict(
    model: Any,
    history_df: pd.DataFrame,
    horizon_steps: int,
    make_next_feature_row,
    target_col: str,
) -> pd.DataFrame:
    """
    forecast_recursive가 없는 모델을 위한 fallback.
    1스텝 예측 후, 그 결과를 simulated_history에 넣고, 또 다음 스텝을 예측하는 방식
    """
    simulated_history = history_df.copy()
    rows: list[dict[str, Any]] = []

    for step in range(1, horizon_steps + 1):
        next_features = make_next_feature_row(simulated_history, step)
        pred_arr = model.predict(next_features) # 추론 실행
        pred = float(np.asarray(pred_arr).reshape(-1)[0])

        row = {
            "step": step,
            "timestamp": next_features.iloc[0]["timestamp"],
            f"predicted_{target_col}": pred,
        }
        rows.append(row)

        appended_row = next_features.copy()
        appended_row[target_col] = pred
        simulated_history = pd.concat([simulated_history, appended_row], ignore_index=True)

    return pd.DataFrame(rows)


# =========================================================
# Feature row builder
# =========================================================
def _build_next_feature_row(
    target_name: str,
    target_col: str,
    model: Any,
    simulated_history: pd.DataFrame,
    weather_df: pd.DataFrame,
    defaults: dict[str, Any],
    auxiliary_it_map: dict[pd.Timestamp, float] | None,
) -> pd.DataFrame:
    """
    다음 시점 1개 row의 feature 생성. 파생변수까지 도출.
    LGBM처럼 tabular feature를 요구하는 모델을 위해 lag / rolling / interaction 생성.
    """
    feature_columns = list(getattr(model, "feature_columns", []) or [])
    if not feature_columns:
        raise ValueError("Model has empty feature_columns.")

    next_ts = pd.Timestamp(simulated_history["timestamp"].iloc[-1]) + timedelta(minutes=STEP_MINUTES)

    row: dict[str, Any] = {"timestamp": next_ts}
    row.update(_build_calendar_features(next_ts))

    weather_row = _lookup_weather_row(next_ts, weather_df)
    if weather_row:
        row.update(weather_row)

    row.setdefault("outdoor_temp_c", _last_or_default(simulated_history, "outdoor_temp_c", defaults.get("outdoor_temp_c", 20.0)))
    row.setdefault("outdoor_humidity", _last_or_default(simulated_history, "outdoor_humidity", defaults.get("outdoor_humidity", 50.0)))
    row.setdefault("outdoor_wind_speed", _last_or_default(simulated_history, "outdoor_wind_speed", defaults.get("outdoor_wind_speed", 0.0)))

    row.setdefault("free_cooling_available", row["outdoor_temp_c"] <= FREE_COOLING_TEMP_C)
    row.setdefault("cooling_degree_days", max(float(row["outdoor_temp_c"]) - 18.0, 0.0))

    if target_name == "cooling_demand":
        predicted_it_value = None
        if auxiliary_it_map is not None:
            predicted_it_value = auxiliary_it_map.get(next_ts)

        if predicted_it_value is None:
            predicted_it_value = _last_or_default(
                simulated_history,
                "predicted_it_power_kw",
                _last_or_default(simulated_history, IT_TARGET_COL, defaults.get(IT_TARGET_COL, 0.0)),
            )

        row["predicted_it_power_kw"] = float(predicted_it_value)

    for feature in feature_columns:
        if feature in row:
            continue

        lag_match = _parse_lag_feature(feature)
        if lag_match is not None:
            base_col, lag = lag_match
            row[feature] = _lag_value(simulated_history, base_col, lag)
            continue

        roll_match = _parse_rolling_feature(feature)
        if roll_match is not None:
            base_col, stat_name, window = roll_match
            row[feature] = _rolling_value(simulated_history, base_col, stat_name, window)
            continue

        if feature == "cpu_mem_ratio_lag_1":
            cpu_lag_1 = row.get("avg_cpu_lag_1", _lag_value(simulated_history, "avg_cpu", 1))
            mem_lag_1 = row.get("avg_mem_lag_1", _lag_value(simulated_history, "avg_mem", 1))
            row[feature] = float(cpu_lag_1) / max(float(mem_lag_1), 1e-6)
            continue

        if feature == "assigned_mem_gap_lag_1":
            assigned_mem_lag_1 = row.get("avg_assigned_mem_lag_1", _lag_value(simulated_history, "avg_assigned_mem", 1))
            mem_lag_1 = row.get("avg_mem_lag_1", _lag_value(simulated_history, "avg_mem", 1))
            row[feature] = float(assigned_mem_lag_1) - float(mem_lag_1)
            continue

        if feature == "it_power_diff_1":
            lag_1 = _lag_value(simulated_history, IT_TARGET_COL, 1)
            lag_2 = _lag_value(simulated_history, IT_TARGET_COL, 2)
            row[feature] = float(lag_1) - float(lag_2)
            continue

        if feature == "it_power_diff_12":
            lag_1 = _lag_value(simulated_history, IT_TARGET_COL, 1)
            lag_12 = _lag_value(simulated_history, IT_TARGET_COL, 12)
            row[feature] = float(lag_1) - float(lag_12)
            continue

        if feature == "temp_above_15c":
            row[feature] = max(float(row["outdoor_temp_c"]) - FREE_COOLING_TEMP_C, 0.0)
            continue

        if feature == "temp_below_15c":
            row[feature] = max(FREE_COOLING_THRESHOLD_C - float(row["outdoor_temp_c"]), 0.0)
            continue

        if feature == "it_power_x_outdoor_temp":
            base_it = row.get("predicted_it_power_kw", _last_or_default(simulated_history, IT_TARGET_COL, 0.0))
            row[feature] = float(base_it) * float(row["outdoor_temp_c"])
            continue

        if feature == "it_power_x_humidity":
            base_it = row.get("predicted_it_power_kw", _last_or_default(simulated_history, IT_TARGET_COL, 0.0))
            row[feature] = float(base_it) * float(row["outdoor_humidity"])
            continue

        if feature == "humidity_temp_index":
            row[feature] = float(row["outdoor_temp_c"]) * float(row["outdoor_humidity"])
            continue

        if feature == "free_cooling_x_it_power":
            base_it = row.get("predicted_it_power_kw", _last_or_default(simulated_history, IT_TARGET_COL, 0.0))
            row[feature] = float(bool(row["free_cooling_available"])) * float(base_it)
            continue

        if feature in simulated_history.columns:
            row[feature] = _last_or_default(simulated_history, feature, defaults.get(feature, 0.0))
            continue

        row[feature] = defaults.get(feature, 0.0)

    feature_df = pd.DataFrame([row])

    missing = [col for col in feature_columns if col not in feature_df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    return feature_df[["timestamp", *feature_columns]]


# =========================================================
# Response builder
# =========================================================
def _merge_predictions_to_points(
    prediction_target: PredictionTarget,
    it_result_df: pd.DataFrame | None,
    cooling_result_df: pd.DataFrame | None,
    include_prediction_interval: bool,
    interval_cfg: dict[str, Any],
) -> list[ForecastPoint]:
    """
    IT / Cooling 결과를 timestamp 기준으로 merge해서 ForecastPoint 리스트 생성.
    """
    by_ts: dict[pd.Timestamp, dict[str, Any]] = {}

    if it_result_df is not None and not it_result_df.empty:
        margin_ratio = float(interval_cfg.get("it_load_margin_ratio", 0.10))
        for _, row in it_result_df.iterrows():
            ts = pd.Timestamp(row["timestamp"])
            pred = float(row["prediction"])

            item = by_ts.setdefault(ts, {"timestamp": ts})
            item["predicted_it_load_kw"] = pred

            if include_prediction_interval:
                item["lower_bound_it_load_kw"] = max(0.0, pred * (1.0 - margin_ratio))
                item["upper_bound_it_load_kw"] = pred * (1.0 + margin_ratio)

    if cooling_result_df is not None and not cooling_result_df.empty:
        margin_ratio = float(interval_cfg.get("cooling_demand_margin_ratio", 0.12))
        for _, row in cooling_result_df.iterrows():
            ts = pd.Timestamp(row["timestamp"])
            pred = float(row["prediction"])

            item = by_ts.setdefault(ts, {"timestamp": ts})
            item["predicted_cooling_load_kw"] = pred
            item["cooling_mode"] = _rule_based_cooling_mode(row.get("outdoor_temp_c"))

            if include_prediction_interval:
                item["lower_bound_cooling_load_kw"] = max(0.0, pred * (1.0 - margin_ratio))
                item["upper_bound_cooling_load_kw"] = pred * (1.0 + margin_ratio)

    points: list[ForecastPoint] = []
    for ts in sorted(by_ts.keys()):
        item = by_ts[ts]
        points.append(
            ForecastPoint(
                timestamp=ts.to_pydatetime(),
                predicted_it_load_kw=item.get("predicted_it_load_kw"),
                predicted_cooling_load_kw=item.get("predicted_cooling_load_kw"),
                cooling_mode=item.get("cooling_mode"),
                lower_bound_it_load_kw=item.get("lower_bound_it_load_kw"),
                upper_bound_it_load_kw=item.get("upper_bound_it_load_kw"),
                lower_bound_cooling_load_kw=item.get("lower_bound_cooling_load_kw"),
                upper_bound_cooling_load_kw=item.get("upper_bound_cooling_load_kw"),
            )
        )

    return points


# =========================================================
# Model selection / request utilities
# =========================================================
def _select_model(
    model_bundle: dict[str, Any],
    target_name: str,
    model_type: ModelType,
) -> Any | None:
    """
    전달받은 model_bundle에서 target_name(IT_load / cooling demand)와
    model_type(lgbm, lstm)에 맞는 학습된 모델 객체를 찾아 반환
    """
    return (
        model_bundle.get("models", {})
        .get(target_name, {})
        .get(model_type.value)
    )


def _find_prediction_column(df: pd.DataFrame, target_col: str) -> str:
    candidates = [
        f"predicted_{target_col}",
        "prediction",
        "predicted_it_power_kw",
        "predicted_chiller_power_kw",
        "predicted_cooling_load_kw",
    ]
    for col in candidates:
        if col in df.columns:
            return col

    numeric_cols = [
        c for c in df.columns
        if c not in {"step", "timestamp"} and pd.api.types.is_numeric_dtype(df[c])
    ]
    if len(numeric_cols) == 1:
        return numeric_cols[0]

    raise ValueError(f"Could not identify prediction column from: {list(df.columns)}")


def _align_timestamp(ts: datetime) -> datetime:
    """
    5분 단위로 floor.
    """
    return ts.replace(minute=(ts.minute // STEP_MINUTES) * STEP_MINUTES, second=0, microsecond=0)


# =========================================================
# History loading
# =========================================================
def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_feature_data_path() -> Path:
    raw_path = os.getenv("FEATURE_DATA_PATH", DATA_PATH)
    path = Path(raw_path)
    if not path.is_absolute():
        path = _project_root() / path
    return path.resolve()


def _load_recent_history_window(
    current_timestamp: datetime,
    lookback_steps: int,
) -> pd.DataFrame:
    """
    현재 시점 이전의 최근 lookback_steps rows를 parquet에서 읽어온다.
    (예측의 출발점이 될 최근 데이터 로드)

    주의:
    - 1차 버전은 요청마다 parquet를 읽는다.
    - 나중에는 캐시 또는 feature store로 바꾸는 게 좋다.
    """
    data_path = _resolve_feature_data_path()
    if not data_path.exists():
        raise FileNotFoundError(f"Feature data file not found: {data_path}")

    df = pd.read_parquet(data_path)
    if "timestamp" not in df.columns:
        raise ValueError("Feature data must contain 'timestamp' column.")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")
    df = df[df["timestamp"] <= pd.Timestamp(current_timestamp)].tail(lookback_steps).reset_index(drop=True)

    return df


# =========================================================
# Weather utilities 
# =========================================================
def _prepare_weather_df(weather_payload: Any, step_minutes: int) -> pd.DataFrame:
    if not weather_payload:
        return pd.DataFrame(columns=["timestamp", "outdoor_temp_c", "outdoor_humidity", "outdoor_wind_speed"])

    rows = None

    if isinstance(weather_payload, list):
        rows = weather_payload
    elif isinstance(weather_payload, dict):
        for key in ("rows", "data", "hourly", "forecasts", "items"):
            if isinstance(weather_payload.get(key), list):
                rows = weather_payload[key]
                break
        if rows is None and "timestamp" in weather_payload:
            rows = [weather_payload]

    if not rows:
        return pd.DataFrame(columns=["timestamp", "outdoor_temp_c", "outdoor_humidity", "outdoor_wind_speed"])

    df = pd.DataFrame(rows)
    if "timestamp" not in df.columns:
        return pd.DataFrame(columns=["timestamp", "outdoor_temp_c", "outdoor_humidity", "outdoor_wind_speed"])

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").drop_duplicates("timestamp")

    for col in ("outdoor_temp_c", "outdoor_humidity", "outdoor_wind_speed"):
        if col not in df.columns:
            df[col] = np.nan

    df = (
        df.set_index("timestamp")[["outdoor_temp_c", "outdoor_humidity", "outdoor_wind_speed"]]
        .resample(f"{step_minutes}min")
        .ffill()
        .reset_index()
    )

    return df


# =========================================================
# Cooling mode - Post-processing
# TODO: 나중에 domain/thermodynamics/freecooling.py에서 이 내용을 선언하는게 나을듯?
# =========================================================

def _rule_based_cooling_mode(outdoor_temp_c: Any) -> CoolingMode:
    """
    예측된 외기 온도를 기반으로 데이터 센터의 냉각 모드를 결정
    """
    if outdoor_temp_c is None or pd.isna(outdoor_temp_c):
        return CoolingMode.CHILLER

    outdoor_temp_c = float(outdoor_temp_c)

    if outdoor_temp_c <= FREE_COOLING_THRESHOLD_C:
        return CoolingMode.FREE_COOLING
    if outdoor_temp_c <= HYBRID_THRESHOLD_C:
        return CoolingMode.HYBRID
    return CoolingMode.CHILLER