"""
Forecast Service orchestration layer.

이 파일의 책임
--------------
- API 요청(payload)을 받아 예측 흐름을 조합한다.
- target(it_load / cooling_demand / both)에 따라 필요한 모델을 선택한다.
- 요청의 model_type(lgbm / lstm)에 따라 적절한 모델(또는 Quantile 모델 번들)을 고른다.
- feature frame 생성 함수를 호출한다.
- 모델 추론 결과를 ForecastResponse 형태로 조립한다.
- cooling mode를 rule-based로 판정한다.
- Quantile 모델을 활용하여 과학적인 Prediction Interval을 결과에 포함한다.
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
from domain.forecasting.features.builder import _build_next_feature_row
from domain.forecasting.intervals import build_quantile_interval

STEP_MINUTES = 5

IT_TARGET_COL = "it_power_kw"
COOLING_TARGET_COL = "chiller_power_kw" 
DATA_PATH = "./data/processed/synthetic_idc_1year.parquet"

# =========================================================
# Public entrypoint
# =========================================================
def run_forecast(
    model_bundle: dict[str, Any],
    request: ForecastRequest,
) -> ForecastResponse:
    current_ts = _align_timestamp(request.current_timestamp or datetime.utcnow())
    horizon_hours = request.forecast_horizon_hours
    horizon_steps = int(horizon_hours * 60 / STEP_MINUTES)

    history_df = _load_recent_history_window(
        current_timestamp=current_ts,
        lookback_steps=400,
    )
    if history_df.empty:
        raise ValueError("No recent history data available for forecast.")

    defaults = model_bundle.get("defaults", {}) or {}
    interval_cfg = model_bundle.get("interval", {}) or {}
    weather_df = _prepare_weather_df(model_bundle.get("weather", {}), STEP_MINUTES)

    prediction_target = request.prediction_target
    model_type = request.model_type
    include_interval = request.include_prediction_interval

    it_result_df: pd.DataFrame | None = None
    cooling_result_df: pd.DataFrame | None = None

    # --- IT Load 예측 ---
    if prediction_target in {
        PredictionTarget.IT_LOAD,
        PredictionTarget.BOTH,
        PredictionTarget.COOLING_DEMAND,
    }:
        it_model_bundle = _select_model(model_bundle, "it_load", model_type, include_interval)
        if it_model_bundle is None and prediction_target in {
            PredictionTarget.IT_LOAD,
            PredictionTarget.BOTH,
            PredictionTarget.COOLING_DEMAND,
        }:
            raise ValueError(f"No model found for target='it_load', model_type='{model_type.value}'.")

        if it_model_bundle is not None:
            it_margin = float(interval_cfg.get("it_load_margin_ratio", 0.10)) if include_interval else 0.0
            
            # ★ 3개 모델 번들 처리용 신규 함수 호출
            it_result_df = _forecast_target_bundle(
                target_name="it_load",
                target_col=IT_TARGET_COL,
                model_or_bundle=it_model_bundle,
                history_df=history_df.copy(),
                horizon_steps=horizon_steps,
                weather_df=weather_df,
                defaults=defaults,
                auxiliary_it_map=None,
                margin_ratio=it_margin
            )

    # --- Cooling Demand 예측 ---
    if prediction_target in {PredictionTarget.COOLING_DEMAND, PredictionTarget.BOTH}:
        cooling_model_bundle = _select_model(model_bundle, "cooling_demand", model_type, include_interval)
        if cooling_model_bundle is None:
            raise ValueError(f"No model found for target='cooling_demand', model_type='{model_type.value}'.")

        cooling_history_df = history_df.copy()
        if "predicted_it_power_kw" not in cooling_history_df.columns and IT_TARGET_COL in cooling_history_df.columns:
            cooling_history_df["predicted_it_power_kw"] = cooling_history_df[IT_TARGET_COL]

        auxiliary_it_map = None
        if it_result_df is not None and not it_result_df.empty:
            auxiliary_it_map = {
                pd.Timestamp(row["timestamp"]): float(row["prediction"])
                for _, row in it_result_df.iterrows()
            }

        cooling_margin = float(interval_cfg.get("cooling_demand_margin_ratio", 0.12)) if include_interval else 0.0
        
        # ★ 3개 모델 번들 처리용 신규 함수 호출
        cooling_result_df = _forecast_target_bundle(
            target_name="cooling_demand",
            target_col=COOLING_TARGET_COL,
            model_or_bundle=cooling_model_bundle,
            history_df=cooling_history_df,
            horizon_steps=horizon_steps,
            weather_df=weather_df,
            defaults=defaults,
            auxiliary_it_map=auxiliary_it_map,
            margin_ratio=cooling_margin
        )

    response_points = _merge_predictions_to_points(
        prediction_target=prediction_target,
        it_result_df=it_result_df if prediction_target in {PredictionTarget.IT_LOAD, PredictionTarget.BOTH} else None,
        cooling_result_df=cooling_result_df,
        include_prediction_interval=include_interval,
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

def _forecast_target_bundle(
    target_name: str,
    target_col: str,
    model_or_bundle: Any | dict[str, Any],
    history_df: pd.DataFrame,
    horizon_steps: int,
    weather_df: pd.DataFrame,
    defaults: dict[str, Any],
    auxiliary_it_map: dict[pd.Timestamp, float] | None,
    margin_ratio: float = 0.0
) -> pd.DataFrame:
    """
    모델이 단일 객체인지, Quantile 3종 세트(dict)인지 구분하여 추론을 오케스트레이션
    """
    if isinstance(model_or_bundle, dict) and "point" in model_or_bundle:
        # 1. Quantile 3개 모델 각각 예측 수행
        df_point = _forecast_one_target(target_name, target_col, model_or_bundle["point"], history_df.copy(), horizon_steps, weather_df, defaults, auxiliary_it_map)
        df_lower = _forecast_one_target(target_name, target_col, model_or_bundle["lower"], history_df.copy(), horizon_steps, weather_df, defaults, auxiliary_it_map)
        df_upper = _forecast_one_target(target_name, target_col, model_or_bundle["upper"], history_df.copy(), horizon_steps, weather_df, defaults, auxiliary_it_map)
        
        # 2. intervals.py 모듈을 통한 역전 교정 및 안전 마진 적용
        safe_lower, safe_upper = build_quantile_interval(
            lower_preds=df_lower["prediction"].values,
            upper_preds=df_upper["prediction"].values,
            margin_ratio=margin_ratio
        )
        
        # 3. 결과 DataFrame 통합
        result_df = df_point.copy()
        result_df["lower_bound"] = safe_lower
        result_df["upper_bound"] = safe_upper
        return result_df
    else:
        # 단일 모델일 경우 기존 방식 그대로 진행
        return _forecast_one_target(target_name, target_col, model_or_bundle, history_df, horizon_steps, weather_df, defaults, auxiliary_it_map)


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


def _run_model_forecast(
    model: Any,
    history_df: pd.DataFrame,
    horizon_steps: int,
    make_next_feature_row,
    target_col: str,
) -> pd.DataFrame:
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
    simulated_history = history_df.copy()
    rows: list[dict[str, Any]] = []

    for step in range(1, horizon_steps + 1):
        next_features = make_next_feature_row(simulated_history, step)
        pred_arr = model.predict(next_features) 
        pred = float(np.asarray(pred_arr).reshape(-1)[0])

        row = {
            "step": step,
            "timestamp": next_features.iloc[0]["timestamp"],
            f"predicted_{target_col}": pred,
        }

        if "outdoor_temp_c" in next_features.columns:
            row["outdoor_temp_c"] = next_features.iloc[0]["outdoor_temp_c"]

        rows.append(row)

        appended_row = next_features.copy()
        appended_row[target_col] = pred
        simulated_history = pd.concat([simulated_history, appended_row], ignore_index=True)

    return pd.DataFrame(rows)


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
    ★ 업데이트: DataFrame에 포함된 과학적 lower_bound, upper_bound를 바로 매핑합니다.
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
                if "lower_bound" in row and "upper_bound" in row:
                    item["lower_bound_it_load_kw"] = max(0.0, float(row["lower_bound"]))
                    item["upper_bound_it_load_kw"] = float(row["upper_bound"])
                else:
                    # Fallback: LSTM 등 단일 모델이 넘어왔을 때
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
                if "lower_bound" in row and "upper_bound" in row:
                    item["lower_bound_cooling_load_kw"] = max(0.0, float(row["lower_bound"]))
                    item["upper_bound_cooling_load_kw"] = float(row["upper_bound"])
                else:
                    # Fallback
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
    include_interval: bool = False
) -> Any | dict[str, Any] | None:
    """
    ★ 업데이트: LGBM이고 구간 예측이 요청되었을 때, lgbm_quantile 번들을 통째로 반환합니다.
    """
    models_for_target = model_bundle.get("models", {}).get(target_name, {})
    
    if model_type == ModelType.LGBM:
        quantile_bundle = models_for_target.get("lgbm_quantile")
        if quantile_bundle and include_interval:
            return quantile_bundle
        elif quantile_bundle and not include_interval:
            return quantile_bundle.get("point")
        else:
            return models_for_target.get("lgbm")
    elif model_type == ModelType.MOVING_AVG:
        return models_for_target.get("moving_avg")
    else:
        return models_for_target.get(model_type.value)


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
    return ts.replace(minute=(ts.minute // STEP_MINUTES) * STEP_MINUTES, second=0, microsecond=0)


# =========================================================
# History loading & Weather utilities
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

def _rule_based_cooling_mode(outdoor_temp_c: Any) -> CoolingMode:
    if outdoor_temp_c is None or pd.isna(outdoor_temp_c):
        return CoolingMode.CHILLER

    outdoor_temp_c = float(outdoor_temp_c)

    if outdoor_temp_c <= FREE_COOLING_THRESHOLD_C:
        return CoolingMode.FREE_COOLING
    if outdoor_temp_c <= HYBRID_THRESHOLD_C:
        return CoolingMode.HYBRID
    return CoolingMode.CHILLER