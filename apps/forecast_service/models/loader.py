import json
import os
from pathlib import Path
from typing import Any

from domain.forecasting.lgbm_model import LGBMForecaster
from domain.forecasting.moving_avg import MovingAverageForecaster

"""
Forecast Service model loader.

이 파일의 책임
--------------
- FastAPI startup 시기 호출되어, IT Load / Cooling Demand용 LGBM, LSTM 모델을 모두 로드한다.
- 새롭게 추가된 Quantile Regression 모델(Lower, Point, Upper)도 함께 로드한다.
- 외기 예보, feature 기본값, interval 설정 등 부가 리소스를 로드한다.
- services/forecast.py에서 바로 사용할 수 있는 model_bundle 형태로 반환한다.
"""

def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_model_dir() -> Path:
    raw_model_dir = os.getenv("MODEL_DIR", "./data/models")
    model_dir = Path(raw_model_dir)

    if not model_dir.is_absolute():
        model_dir = _project_root() / model_dir

    return model_dir.resolve()


def _load_json_if_exists(
    file_path: Path,
    default: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not file_path.exists():
        return default or {}

    with file_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_moving_avg_if_exists(file_path: Path) -> MovingAverageForecaster | None:
    if not file_path.exists():
        return None
    return MovingAverageForecaster.load(file_path)


def _load_lgbm_if_exists(file_path: Path) -> LGBMForecaster | None:
    if not file_path.exists():
        return None
    # 확장자가 .pkl이든 .joblib이든 동일하게 joblib.load를 수행하도록 
    # LGBMForecaster.load() 내부에 구현되어 있다고 가정.
    return LGBMForecaster.load(file_path)


def _try_load_lstm_if_exists(file_path: Path) -> Any | None:
    if not file_path.exists():
        return None

    try:
        from domain.forecasting.lstm_model import LSTMForecaster
    except ImportError:
        return None

    if not hasattr(LSTMForecaster, "load"):
        return None

    return LSTMForecaster.load(file_path)


def _check_any_model_loaded(models_dict: dict) -> bool:
    """중첩된 딕셔너리를 순회하며 로드된 모델이 하나라도 있는지 확인합니다."""
    for value in models_dict.values():
        if isinstance(value, dict):
            if _check_any_model_loaded(value):
                return True
        elif value is not None:
            return True
    return False


def load_model_bundle() -> dict[str, Any]:
    model_dir = _resolve_model_dir()

    bundle: dict[str, Any] = {
        "model_dir": str(model_dir),
        "models": {
            "it_load": {
                "moving_avg": _load_moving_avg_if_exists(model_dir / "it_load_moving_avg.joblib"),
                "lgbm": _load_lgbm_if_exists(model_dir / "it_load_lgbm.joblib"),
                "lgbm_quantile": {
                    "lower": _load_lgbm_if_exists(model_dir / "lgbm_quantile_it_load_lower.pkl"),
                    "point": _load_lgbm_if_exists(model_dir / "lgbm_quantile_it_load_point.pkl"),
                    "upper": _load_lgbm_if_exists(model_dir / "lgbm_quantile_it_load_upper.pkl"),
                },
                "lstm": _try_load_lstm_if_exists(model_dir / "it_load_lstm.pt"),
            },
            "cooling_demand": {
                "moving_avg": _load_moving_avg_if_exists(model_dir / "cooling_demand_moving_avg.joblib"),
                "lgbm": _load_lgbm_if_exists(model_dir / "cooling_demand_lgbm.joblib"),
                "lgbm_quantile": {
                    "lower": _load_lgbm_if_exists(model_dir / "lgbm_quantile_cooling_demand_lower.pkl"),
                    "point": _load_lgbm_if_exists(model_dir / "lgbm_quantile_cooling_demand_point.pkl"),
                    "upper": _load_lgbm_if_exists(model_dir / "lgbm_quantile_cooling_demand_upper.pkl"),
                },
                "lstm": _try_load_lstm_if_exists(model_dir / "cooling_demand_lstm.pt"),
            },
        },
        "defaults": _load_json_if_exists(
            model_dir / "forecast_defaults.json",
            default={
                "cooling_degree_days": 10.0,
                "free_cooling_available": False,
            },
        ),
        "weather": _load_json_if_exists(
            model_dir / "weather_forecast.json",
            default={},
        ),
        "interval": _load_json_if_exists(
            model_dir / "interval_config.json",
            default={
                "it_load_margin_ratio": 0.10,
                "cooling_demand_margin_ratio": 0.12,
            },
        ),
    }

    # 변경된 로직: 중첩 딕셔너리 구조(lgbm_quantile)까지 안전하게 검사
    if not _check_any_model_loaded(bundle["models"]):
        raise FileNotFoundError(
            "No forecast model artifact found in MODEL_DIR. "
            f"Checked directory: {model_dir}"
        )

    return bundle