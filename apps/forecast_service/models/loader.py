import json
import os
from pathlib import Path
from typing import Any

from domain.forecasting.lgbm_model import LGBMForecaster

"""
Forecast Service model loader.

이 파일의 책임
--------------
- FastAPI startup시기 호출되어, IT Load / Cooling Demand용 LGBM, LSTM 모델을 모두 로드한다.
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


def _load_lgbm_if_exists(file_path: Path) -> LGBMForecaster | None:
    if not file_path.exists():
        return None
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


def load_model_bundle() -> dict[str, Any]:
    model_dir = _resolve_model_dir()

    bundle: dict[str, Any] = {
        "model_dir": str(model_dir),
        "models": {
            "it_load": {
                "lgbm": _load_lgbm_if_exists(model_dir / "it_load_lgbm.joblib"),
                "lstm": _try_load_lstm_if_exists(model_dir / "it_load_lstm.pt"),
            },
            "cooling_demand": {
                "lgbm": _load_lgbm_if_exists(model_dir / "cooling_demand_lgbm.joblib"),
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

    any_model_loaded = any(
        model is not None
        for target_models in bundle["models"].values()
        for model in target_models.values()
    )

    if not any_model_loaded:
        raise FileNotFoundError(
            "No forecast model artifact found in MODEL_DIR. "
            f"Checked directory: {model_dir}"
        )

    return bundle