import math
import pandas as pd
import numpy as np
from datetime import timedelta
from typing import Any

from core.config.constants import FREE_COOLING_THRESHOLD_C

"""
Feature Engineering 모듈
학습(Train) 시의 벡터화 연산과 추론(Inference) 시의 실시간 Row 생성 로직을 통합 관리합니다.
"""

# =========================================================
# 1. Common Constants & Feature Engineering Helpers
# =========================================================

STEP_MINUTES = 5
IT_TARGET_COL = "it_power_kw"

def _build_calendar_features(ts: pd.Timestamp) -> dict[str, Any]:
    hour = ts.hour
    minute = ts.minute
    day_of_week = ts.dayofweek

    return {
        "hour": hour,
        "minute": minute,
        "day_of_week": day_of_week,
        "day_of_month": ts.day,
        "month": ts.month,
        "is_weekend": day_of_week in (5, 6),
        "hour_sin": math.sin(2.0 * math.pi * hour / 24.0),
        "hour_cos": math.cos(2.0 * math.pi * hour / 24.0),
        "dow_sin": math.sin(2.0 * math.pi * day_of_week / 7.0),
        "dow_cos": math.cos(2.0 * math.pi * day_of_week / 7.0),
    }


def _parse_lag_feature(feature_name: str) -> tuple[str, int] | None:
    if "_lag_" not in feature_name:
        return None
    base_col, lag_str = feature_name.rsplit("_lag_", 1)
    if not lag_str.isdigit():
        return None
    return base_col, int(lag_str)


def _parse_rolling_feature(feature_name: str) -> tuple[str, str, int] | None:
    for suffix, stat_name in (("_roll_mean_", "mean"), ("_roll_std_", "std")):
        if suffix in feature_name:
            base_col, window_str = feature_name.rsplit(suffix, 1)
            if window_str.isdigit():
                return base_col, stat_name, int(window_str)
    return None


def _lag_value(history_df: pd.DataFrame, base_col: str, lag: int) -> float:
    if base_col not in history_df.columns or history_df.empty:
        return 0.0

    series = pd.to_numeric(history_df[base_col], errors="coerce").dropna()
    if series.empty:
        return 0.0

    if len(series) < lag:
        return float(series.iloc[0])

    return float(series.iloc[-lag])


def _rolling_value(history_df: pd.DataFrame, base_col: str, stat_name: str, window: int) -> float:
    if base_col not in history_df.columns or history_df.empty:
        return 0.0

    series = pd.to_numeric(history_df[base_col], errors="coerce").dropna()
    if series.empty:
        return 0.0

    tail = series.tail(window)

    if stat_name == "mean":
        return float(tail.mean())
    if stat_name == "std":
        value = tail.std()
        return 0.0 if pd.isna(value) else float(value)

    raise ValueError(f"Unsupported rolling stat: {stat_name}")


def _last_or_default(history_df: pd.DataFrame, col: str, default: Any) -> Any:
    if col not in history_df.columns or history_df.empty:
        return default

    series = history_df[col].dropna()
    if series.empty:
        return default

    return series.iloc[-1]


# =========================================================
# 2. [추론용] 실시간 단일 Row 생성 (Online Feature Engineering)
# =========================================================

def _lookup_weather_row(ts: pd.Timestamp, weather_df: pd.DataFrame) -> dict[str, Any]:
    """지정된 시간(ts)에 가장 근접한 날씨 데이터를 추출합니다."""
    if weather_df.empty:
        return {}

    match = weather_df.loc[weather_df["timestamp"] == ts]
    if not match.empty:
        row = match.iloc[0]
        return {
            "outdoor_temp_c": float(row["outdoor_temp_c"]),
            "outdoor_humidity": float(row["outdoor_humidity"]),
            "outdoor_wind_speed": float(row["outdoor_wind_speed"]),
        }

    before = weather_df.loc[weather_df["timestamp"] <= ts]
    row = weather_df.iloc[0] if before.empty else before.iloc[-1]

    return {
        "outdoor_temp_c": float(row["outdoor_temp_c"]),
        "outdoor_humidity": float(row["outdoor_humidity"]),
        "outdoor_wind_speed": float(row["outdoor_wind_speed"]),
    }


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

    row.setdefault("free_cooling_available", row["outdoor_temp_c"] <= FREE_COOLING_THRESHOLD_C)
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
            row[feature] = max(float(row["outdoor_temp_c"]) - FREE_COOLING_THRESHOLD_C, 0.0)
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
# 3. [학습용] 전체 데이터프레임 벡터화 연산 (Offline Feature Engineering)
# =========================================================

# =========================================================
# 3. [학습용] 전체 데이터프레임 벡터화 연산 (Offline Feature Engineering)
# =========================================================

def build_train_features(df: pd.DataFrame, target_col: str = IT_TARGET_COL) -> pd.DataFrame:
    """
    [학습 시점] _build_next_feature_row에 정의된 파생 변수 규칙을
    학습용 데이터프레임 전체에 일괄 적용(벡터화 연산)하여 속도를 극대화합니다.
    """
    df = df.copy()
    
    # 컬럼명 동기화
    rename_map = {"outside_temp_c": "outdoor_temp_c", "outside_humidity_pct": "outdoor_humidity"}
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    
    # 1. Calendar Features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['hour_sin'] = np.sin(2.0 * np.pi * df['hour'] / 24.0)
    df['hour_cos'] = np.cos(2.0 * np.pi * df['hour'] / 24.0)
    df['dow_sin'] = np.sin(2.0 * np.pi * df['day_of_week'] / 7.0)
    df['dow_cos'] = np.cos(2.0 * np.pi * df['day_of_week'] / 7.0)
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

    # 2. Weather & Threshold Features
    df['temp_above_15c'] = (df['outdoor_temp_c'] - FREE_COOLING_THRESHOLD_C).clip(lower=0.0)
    df['temp_below_15c'] = (FREE_COOLING_THRESHOLD_C - df['outdoor_temp_c']).clip(lower=0.0)
    df['free_cooling_available'] = (df['outdoor_temp_c'] <= FREE_COOLING_THRESHOLD_C).astype(int)
    df['humidity_temp_index'] = df['outdoor_temp_c'] * df['outdoor_humidity']

    # 3. IT 부하 관련 변수 (★ 냉방 수요 예측의 핵심 피처)
    # forecast.py 추론 시 넘겨받는 'predicted_it_power_kw'를 학습 시에는 실제 정답 데이터로 세팅합니다.
    df['predicted_it_power_kw'] = df[IT_TARGET_COL] 
    
    df['it_power_kw_lag_1'] = df[IT_TARGET_COL].shift(1)
    df['it_power_x_outdoor_temp'] = df['it_power_kw_lag_1'] * df['outdoor_temp_c']
    df['free_cooling_x_it_power'] = df['free_cooling_available'] * df['it_power_kw_lag_1']

    # 4. 동적 Target Lag & Diff Features (IT 부하든 냉방 부하든 입력된 타겟에 맞춰 생성)
    df[f'{target_col}_lag_1'] = df[target_col].shift(1)
    df[f'{target_col}_lag_2'] = df[target_col].shift(2)
    df[f'{target_col}_lag_12'] = df[target_col].shift(12)
    df[f'{target_col}_diff_1'] = df[f'{target_col}_lag_1'] - df[f'{target_col}_lag_2']
    df[f'{target_col}_diff_12'] = df[f'{target_col}_lag_1'] - df[f'{target_col}_lag_12']

    return df.dropna().reset_index(drop=True)