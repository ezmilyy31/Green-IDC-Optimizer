"""학습된 forecast 모델들의 메트릭을 산출해 JSON으로 저장한다.

산출 항목 (명세 §4-4, §4-5)
- IT 부하 LGBM       — MAPE 24h / 168h  (기준 ≤5% / ≤8%)
- 냉각 수요 LGBM      — nMAE             (기준 ≤10%)
- 90% PI Coverage    — quantile 모델 사용 (기준 ≥85%)

데이터/모델
- parquet : data/weather/synthetic_idc_1year_noisy.parquet
- 모델   : data/models/{it_load,cooling_demand}_lgbm.joblib,
           data/models/lgbm_quantile_{it_load,cooling_demand}_{lower,point,upper}.pkl

평가 방법
- 마지막 1개월(12월)을 holdout으로 분리
- 모델별로 추론 → 메트릭 계산
- 모델 일부 누락 시 해당 섹션은 null로 표시

명세상 누수 주의: 현재 모델이 전체 데이터로 재학습된 prod 모델이므로 holdout 구간에
일부 데이터 누수 가능. 정식 검증은 train script의 월별 cross-validation 값을 사용.

실행
    uv run python -m scripts.eval_forecast_metrics
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from domain.forecasting.features.builder import build_train_features
from domain.forecasting.lgbm_model import LGBMForecaster


PARQUET = Path("data/weather/synthetic_idc_1year_noisy.parquet")
MODELS_DIR = Path("data/models")
OUT = Path("data/eval/forecast_metrics.json")

IT_TARGET = "it_power_kw"
COOL_TARGET = "chiller_power_kw"


def _mape(yt: np.ndarray, yp: np.ndarray) -> float:
    return float(np.mean(np.abs((yt - yp) / (yt + 1e-8))) * 100)


def _nmae(yt: np.ndarray, yp: np.ndarray) -> float:
    denom = float(np.mean(yt))
    if denom == 0:
        return float("nan")
    return float(np.mean(np.abs(yt - yp)) / denom * 100)


def _coverage(yt: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    inside = (yt >= lower) & (yt <= upper)
    return float(np.mean(inside) * 100)


def _safe_load(path: Path) -> "LGBMForecaster | None":
    if not path.exists():
        return None
    try:
        return LGBMForecaster.load(str(path))
    except Exception as exc:
        print(f"[warn] {path.name} 로드 실패: {exc}")
        return None


def _predict_or_none(
    model: "LGBMForecaster | None",
    holdout: pd.DataFrame,
    target_col: str,
) -> "np.ndarray | None":
    if model is None:
        return None
    if target_col not in holdout.columns:
        print(f"[warn] target '{target_col}' missing in holdout")
        return None
    try:
        pred_df = model.predict_frame(holdout.drop(columns=[target_col]), timestamp_col="timestamp")
    except Exception as exc:
        print(f"[warn] predict 실패 ({target_col}): {exc}")
        return None
    return pred_df[f"predicted_{target_col}"].values


def _it_load_metrics(holdout: pd.DataFrame) -> dict:
    model = _safe_load(MODELS_DIR / "it_load_lgbm.joblib")
    y_pred = _predict_or_none(model, holdout, IT_TARGET)
    if y_pred is None:
        return {"status": "unavailable", "reason": "it_load_lgbm.joblib 로드/추론 실패"}
    y_true = holdout[IT_TARGET].values
    end_168 = min(2016, len(y_true))
    return {
        "status":        "ok",
        "mape_24h_pct":  round(_mape(y_true[:288], y_pred[:288]), 2),
        "mape_168h_pct": round(_mape(y_true[:end_168], y_pred[:end_168]), 2),
        "mape_all_pct":  round(_mape(y_true, y_pred), 2),
        "spec_24h_pct":  5.0,
        "spec_168h_pct": 8.0,
    }


def _cooling_metrics(holdout: pd.DataFrame) -> dict:
    """냉각 수요 모델은 chiller_power_kw 타깃 기준으로 다시 feature build 필요."""
    # build_train_features의 lag 컬럼이 타깃 컬럼명에 의존하므로 별도 re-build
    raw = pd.read_parquet(PARQUET)
    cool_df = build_train_features(raw, target_col=COOL_TARGET)
    last_ts = cool_df["timestamp"].max()
    cool_holdout = cool_df[cool_df["timestamp"].dt.month == last_ts.month].reset_index(drop=True)

    model = _safe_load(MODELS_DIR / "cooling_demand_lgbm.joblib")
    y_pred = _predict_or_none(model, cool_holdout, COOL_TARGET)
    if y_pred is None:
        return {"status": "unavailable", "reason": "cooling_demand_lgbm.joblib 로드/추론 실패"}
    y_true = cool_holdout[COOL_TARGET].values
    return {
        "status":       "ok",
        "nmae_pct":     round(_nmae(y_true, y_pred), 2),
        "mape_pct":     round(_mape(y_true, y_pred), 2),
        "mean_kw_true": round(float(np.mean(y_true)), 2),
        "spec_nmae_pct": 10.0,
    }


def _coverage_metrics(holdout: pd.DataFrame, target: str, file_prefix: str) -> dict:
    """90% PI: lower(α=0.05) ~ upper(α=0.95) 사이에 실제값이 포함된 비율."""
    lower_m = _safe_load(MODELS_DIR / f"{file_prefix}_lower.pkl")
    upper_m = _safe_load(MODELS_DIR / f"{file_prefix}_upper.pkl")
    if lower_m is None or upper_m is None:
        return {"status": "unavailable", "reason": f"{file_prefix}_lower/upper 파일 누락"}

    # cooling의 경우 build_train_features를 cooling 타깃으로 다시 만들어야 lag 컬럼이 맞음
    if target == COOL_TARGET and IT_TARGET in holdout.columns:
        raw = pd.read_parquet(PARQUET)
        df = build_train_features(raw, target_col=COOL_TARGET)
        last_ts = df["timestamp"].max()
        holdout = df[df["timestamp"].dt.month == last_ts.month].reset_index(drop=True)

    y_lower = _predict_or_none(lower_m, holdout, target)
    y_upper = _predict_or_none(upper_m, holdout, target)
    if y_lower is None or y_upper is None:
        return {"status": "unavailable", "reason": f"{file_prefix} 추론 실패"}
    y_true = holdout[target].values
    cov = _coverage(y_true, y_lower, y_upper)
    avg_width = float(np.mean(y_upper - y_lower))
    return {
        "status":          "ok",
        "coverage_90_pct": round(cov, 2),
        "avg_width_kw":    round(avg_width, 2),
        "spec_coverage_pct": 85.0,
    }


def _load_cv(name: str) -> "dict | None":
    """train script가 저장한 월별 CV 결과 JSON을 로드."""
    p = Path("data/eval") / name
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception as exc:
        print(f"[warn] CV JSON 로드 실패 {p}: {exc}")
        return None


def main() -> int:
    cv_it      = _load_cv("cv_lgbm_it_load.json")
    cv_cool    = _load_cv("cv_lgbm_cooling_demand.json")
    cv_qit     = _load_cv("cv_lgbm_quantile_it_load.json")

    # 보조: prod 모델 holdout 평가 (참고용, 누수 가능성 명시)
    holdout_section = {"status": "skipped"}
    if PARQUET.exists():
        df = pd.read_parquet(PARQUET)
        df = build_train_features(df, target_col=IT_TARGET)
        last_ts = df["timestamp"].max()
        holdout = df[df["timestamp"].dt.month == last_ts.month].reset_index(drop=True)
        holdout_section = {
            "status":       "ok",
            "method":       "prod_model_on_last_month",
            "holdout_rows": int(len(holdout)),
            "holdout_month": int(last_ts.month),
            "warning":      "prod 모델은 전체 1년 학습이므로 누수 가능 — 참고용",
            "it_load":      _it_load_metrics(holdout),
            "cooling_demand": _cooling_metrics(holdout),
            "it_load_coverage": _coverage_metrics(holdout, IT_TARGET, "lgbm_quantile_it_load"),
            "cooling_demand_coverage": _coverage_metrics(holdout, COOL_TARGET, "lgbm_quantile_cooling_demand"),
        }

    # 메인 KPI 섹션은 CV 결과를 우선 사용 (누수 없음)
    overall_status = "ok" if (cv_it or cv_cool or cv_qit) else "unavailable"

    out = {
        "generated_at":   datetime.utcnow().isoformat() + "Z",
        "data_source":    str(PARQUET),
        "status":         overall_status,
        "primary_source": "monthly_cv",  # 대시보드가 우선 사용할 출처
        "note":           "CV 결과는 train script가 학습 시 산출 — 누수 없는 정식 메트릭",
        "cv": {
            "lgbm_it_load":          cv_it,
            "lgbm_cooling_demand":   cv_cool,
            "lgbm_quantile_it_load": cv_qit,
        },
        "holdout": holdout_section,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"saved {OUT}")
    print(json.dumps({k: v for k, v in out.items() if k != "holdout"}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
