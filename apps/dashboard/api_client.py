"""
API Gateway 및 내부 서비스 HTTP 클라이언트

대시보드에서 다른 마이크로서비스에 HTTP 요청을 보내는 클라이언트 모음.
서비스가 응답하지 않아도 대시보드가 죽지 않도록 모든 호출은 예외를 catch한다.
"""

import os

import requests

API_URL        = os.getenv("API_GATEWAY_URL",        "http://localhost:8000")
SIMULATION_URL = os.getenv("SIMULATION_SERVICE_URL", "http://localhost:8003")
CONTROL_URL    = os.getenv("CONTROL_SERVICE_URL",    "http://localhost:8002")
FORECAST_URL   = os.getenv("FORECAST_SERVICE_URL",   "http://localhost:8001")

_TIMEOUT = 3        # seconds — 일반 엔드포인트
_TIMEOUT_SLOW = 15  # seconds — LightGBM 추론 등 느린 엔드포인트


def _get(url: str) -> dict:
    try:
        resp = requests.get(url, timeout=_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"error": str(e)}


def _post(url: str, payload: dict, timeout: int = _TIMEOUT) -> dict:
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"error": str(e)}


# ── Health checks ─────────────────────────────────────────────────────────────

def health_api() -> dict:
    """API Gateway 헬스체크"""
    return _get(f"{API_URL}/health")


def health_simulation() -> dict:
    """Simulation Service 헬스체크"""
    return _get(f"{SIMULATION_URL}/health")


def health_control() -> dict:
    """Control Service 헬스체크"""
    return _get(f"{CONTROL_URL}/health")


def health_forecast() -> dict:
    """Forecast Service 헬스체크"""
    return _get(f"{FORECAST_URL}/health")


def get_all_service_status() -> dict[str, bool]:
    """모든 서비스 헬스 상태를 한 번에 조회한다. 값이 True면 정상."""
    results = {
        "API Gateway": health_api(),
        "Simulation":  health_simulation(),
        "Control":     health_control(),
        "Forecast":    health_forecast(),
    }
    return {name: "error" not in r for name, r in results.items()}


# ── Control Service endpoints ─────────────────────────────────────────────────
# 구현된 엔드포인트: POST /api/v1/control/optimize, /control/rule-based, /control/rl
# ControlRequest:  { outdoor_temp_c, it_power_kw, outdoor_humidity_pct?, timestamp? }
# ControlResponse: { cooling_mode, supply_air_temp_setpoint_c, free_cooling_ratio, expected_pue }

def _build_control_payload(
    outdoor_temp_c: float,
    it_power_kw: float,
    outdoor_humidity: float = 50.0,
    *,
    zone_temp_c: float | None = None,
    supply_setpoint_c: float | None = None,
    cpu_utilization: float | None = None,
    outdoor_temp_trend_c_per_s: float = 0.0,
    timestamp: str | None = None,
) -> dict:
    """ControlRequest 스키마와 정합되는 페이로드.

    rule-based 엔드포인트는 처음 3개만 사용. RL 엔드포인트는 zone/supply/cpu 추가 필요.
    """
    payload: dict = {
        "outdoor_temp_c":   outdoor_temp_c,
        "it_power_kw":      it_power_kw,
        "outdoor_humidity": outdoor_humidity,
        "outdoor_temp_trend_c_per_s": outdoor_temp_trend_c_per_s,
    }
    if zone_temp_c is not None:
        payload["zone_temp_c"] = zone_temp_c
    if supply_setpoint_c is not None:
        payload["supply_setpoint_c"] = supply_setpoint_c
    if cpu_utilization is not None:
        payload["cpu_utilization"] = cpu_utilization
    if timestamp is not None:
        payload["timestamp"] = timestamp
    return payload


def optimize_control(
    outdoor_temp_c: float,
    it_power_kw: float,
    outdoor_humidity: float = 50.0,
) -> dict:
    """POST /api/v1/control/optimize — 현재 Rule-based 라우팅."""
    return _post(
        f"{API_URL}/api/v1/control/optimize",
        _build_control_payload(outdoor_temp_c, it_power_kw, outdoor_humidity),
    )


def rule_based_control(
    outdoor_temp_c: float,
    it_power_kw: float,
    outdoor_humidity: float = 50.0,
) -> dict:
    """POST /control/rule-based — Rule-based 냉각 제어."""
    return _post(
        f"{API_URL}/control/rule-based",
        _build_control_payload(outdoor_temp_c, it_power_kw, outdoor_humidity),
    )


def rl_control(
    outdoor_temp_c: float,
    it_power_kw: float,
    outdoor_humidity: float = 50.0,
    *,
    zone_temp_c: float,
    supply_setpoint_c: float,
    cpu_utilization: float,
    outdoor_temp_trend_c_per_s: float = 0.0,
    timestamp: str | None = None,
) -> dict:
    """POST /control/rl — 효율 우선 best 모델 추론 + safe fallback.

    zone_temp_c, supply_setpoint_c, cpu_utilization 필드를 반드시 요구한다.
    누락 시 control_service가 HTTP 422를 반환하므로 caller가 모두 채워서 전달해야 함.
    """
    return _post(
        f"{API_URL}/control/rl",
        _build_control_payload(
            outdoor_temp_c, it_power_kw, outdoor_humidity,
            zone_temp_c=zone_temp_c,
            supply_setpoint_c=supply_setpoint_c,
            cpu_utilization=cpu_utilization,
            outdoor_temp_trend_c_per_s=outdoor_temp_trend_c_per_s,
            timestamp=timestamp,
        ),
    )


# ── Forecast Service endpoints ────────────────────────────────────────────────

def call_forecast(
    horizon_hours: int = 24,
    include_prediction_interval: bool = True,
    model_type: str = "lgbm",
    current_timestamp: str | None = None,
) -> dict:
    """POST /api/v1/forecast — API Gateway 경유 IT 부하/냉각 수요 예측.

    model_type: "lgbm" | "moving_avg" | "lstm" (ModelType enum과 정합).
    current_timestamp: ISO 8601 문자열. 미지정 시 서버 UTC 현재 시각 사용.
    시즌별 시뮬레이션과 동기화하려면 시즌 시작 timestamp를 전달한다.

    응답 predictions 리스트의 각 항목:
        timestamp, predicted_it_load_kw, predicted_cooling_load_kw,
        cooling_mode, lower/upper_bound_it_load_kw, lower/upper_bound_cooling_load_kw
    """
    payload: dict = {
        "forecast_horizon_hours": horizon_hours,
        "include_prediction_interval": include_prediction_interval,
        "model_type": model_type,
    }
    if current_timestamp is not None:
        payload["current_timestamp"] = current_timestamp
    return _post(
        f"{API_URL}/api/v1/forecast",
        payload,
        timeout=_TIMEOUT_SLOW,
    )


# ── Simulation endpoints (서비스 구현 후 활성화) ──────────────────────────────
# TODO(Simulation Service): Simulation Service FastAPI 서버 구현 후 활성화.
#   현재 docker-compose.yml의 simulation-service command가 test_sinergym.py(일회성 스크립트)로
#   설정되어 있어 헬스체크 항상 실패 — FastAPI 서버 실행 명령으로 교체 필요 (api_spec.md §5).

def simulate_24h(
    scenario: str,
    num_cpu: int,
    num_gpu: int,
    base_util: float,
    supply_temp_c: float,
    crisis: str | None = None,
) -> dict:
    """POST /api/v1/simulation/24h — Simulation Service 24시간 시뮬레이션."""
    return _post(
        f"{SIMULATION_URL}/api/v1/simulation/24h",
        {
            "scenario_name": scenario,
            "num_cpu":       num_cpu,
            "num_gpu":       num_gpu,
            "base_util":     base_util,
            "supply_temp_c": supply_temp_c,
            "crisis":        crisis,
        },
    )
