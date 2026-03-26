"""
API Gateway 및 내부 서비스 HTTP 클라이언트

대시보드에서 다른 마이크로서비스에 HTTP 요청을 보내는 클라이언트 모음.
서비스가 응답하지 않아도 대시보드가 죽지 않도록 모든 호출은 예외를 catch한다.
"""

import os

import requests

<<<<<<< Updated upstream
API_URL = os.getenv("API_GATEWAY_URL", "http://localhost:8000")
SIMULATION_URL = os.getenv("SIMULATION_SERVICE_URL", "http://localhost:8003")
CONTROL_URL = os.getenv("CONTROL_SERVICE_URL", "http://localhost:8002")
FORECAST_URL = os.getenv("FORECAST_SERVICE_URL", "http://localhost:8001")
=======
API_URL        = os.getenv("API_GATEWAY_URL",       "http://localhost:8000")
SIMULATION_URL = os.getenv("SIMULATION_SERVICE_URL", "http://localhost:8003")
CONTROL_URL    = os.getenv("CONTROL_SERVICE_URL",    "http://localhost:8002")
FORECAST_URL   = os.getenv("FORECAST_SERVICE_URL",   "http://localhost:8001")
>>>>>>> Stashed changes

_TIMEOUT = 3  # seconds


def _get(url: str) -> dict:
    try:
        resp = requests.get(url, timeout=_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"error": str(e)}


def _post(url: str, payload: dict) -> dict:
    try:
        resp = requests.post(url, json=payload, timeout=_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"error": str(e)}


# ── Health checks ────────────────────────────────────────────────────────────

def health_api() -> dict:
<<<<<<< Updated upstream
    """API Gateway 헬스체크"""
=======
>>>>>>> Stashed changes
    return _get(f"{API_URL}/health")


def health_simulation() -> dict:
<<<<<<< Updated upstream
    """Simulation Service 헬스체크"""
=======
>>>>>>> Stashed changes
    return _get(f"{SIMULATION_URL}/health")


def health_control() -> dict:
<<<<<<< Updated upstream
    """Control Service 헬스체크"""
=======
>>>>>>> Stashed changes
    return _get(f"{CONTROL_URL}/health")


def health_forecast() -> dict:
<<<<<<< Updated upstream
    """Forecast Service 헬스체크"""
=======
>>>>>>> Stashed changes
    return _get(f"{FORECAST_URL}/health")


def get_all_service_status() -> dict[str, bool]:
    """모든 서비스 헬스 상태를 한 번에 조회한다. 값이 True면 정상."""
    results = {
        "API Gateway": health_api(),
<<<<<<< Updated upstream
        "Simulation": health_simulation(),
        "Control": health_control(),
        "Forecast": health_forecast(),
=======
        "Simulation":  health_simulation(),
        "Control":     health_control(),
        "Forecast":    health_forecast(),
>>>>>>> Stashed changes
    }
    return {name: "error" not in r for name, r in results.items()}


<<<<<<< Updated upstream
# ── Simulation endpoints (서비스 구현 후 활성화) ──────────────────────────────

def simulate_step(outdoor_temp: float, it_power_kw: float, supply_temp_setpoint: float = 18.0) -> dict:
    """
    단일 타임스텝 시뮬레이션 요청.
    Simulation Service의 POST /simulate/step 호출 예정.
    """
    return _post(
        f"{SIMULATION_URL}/simulate/step",
        {
            "outdoor_temp_c": outdoor_temp,
            "it_power_kw": it_power_kw,
            "supply_temp_setpoint_c": supply_temp_setpoint,
=======
# ── Control Service endpoints ─────────────────────────────────────────────────
# 구현된 엔드포인트: POST /api/v1/control/optimize, /control/rule-based, /control/rl
# ControlRequest: { outdoor_temp_c, it_power_kw, outdoor_humidity_pct?, timestamp? }
# ControlResponse: { cooling_mode, supply_air_temp_setpoint_c, free_cooling_ratio, expected_pue }

def _build_control_payload(
    outdoor_temp_c: float,
    it_power_kw: float,
    outdoor_humidity_pct: float = 50.0,
) -> dict:
    return {
        "outdoor_temp_c":       outdoor_temp_c,
        "it_power_kw":          it_power_kw,
        "outdoor_humidity_pct": outdoor_humidity_pct,
    }


def optimize_control(
    outdoor_temp_c: float,
    it_power_kw: float,
    outdoor_humidity_pct: float = 50.0,
) -> dict:
    """POST /api/v1/control/optimize — API Gateway 경유 Rule-based 제어."""
    return _post(
        f"{API_URL}/api/v1/control/optimize",
        _build_control_payload(outdoor_temp_c, it_power_kw, outdoor_humidity_pct),
    )


def rule_based_control(
    outdoor_temp_c: float,
    it_power_kw: float,
    outdoor_humidity_pct: float = 50.0,
) -> dict:
    """POST /control/rule-based — API Gateway 경유 Rule-based 냉각 제어."""
    return _post(
        f"{API_URL}/control/rule-based",
        _build_control_payload(outdoor_temp_c, it_power_kw, outdoor_humidity_pct),
    )


def rl_control(
    outdoor_temp_c: float,
    it_power_kw: float,
    outdoor_humidity_pct: float = 50.0,
) -> dict:
    """POST /control/rl — API Gateway 경유 RL 에이전트 제어 (Week 4 구현 예정)."""
    return _post(
        f"{API_URL}/control/rl",
        _build_control_payload(outdoor_temp_c, it_power_kw, outdoor_humidity_pct),
    )


# ── Simulation / Forecast (stub — /health 외 미구현) ─────────────────────────

def simulate_step(outdoor_temp: float, it_power_kw: float, supply_temp_setpoint: float = 18.0) -> dict:
    """POST /simulate/step — Simulation Service 연동 예정."""
    return _post(
        f"{SIMULATION_URL}/simulate/step",
        {
            "outdoor_temp_c":          outdoor_temp,
            "it_power_kw":             it_power_kw,
            "supply_temp_setpoint_c":  supply_temp_setpoint,
>>>>>>> Stashed changes
        },
    )


def simulate_scenario(hours: int = 24, scenario: str = "summer") -> dict:
<<<<<<< Updated upstream
    """
    시나리오 시뮬레이션 요청.
    Simulation Service의 GET /simulate/scenario 호출 예정.
    """
=======
    """GET /simulate/scenario — Simulation Service 연동 예정."""
>>>>>>> Stashed changes
    return _get(f"{SIMULATION_URL}/simulate/scenario?hours={hours}&scenario={scenario}")
