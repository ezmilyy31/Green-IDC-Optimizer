"""
API Gateway 및 내부 서비스 HTTP 클라이언트

대시보드에서 다른 마이크로서비스에 HTTP 요청을 보내는 클라이언트 모음.
서비스가 응답하지 않아도 대시보드가 죽지 않도록 모든 호출은 예외를 catch한다.
"""

import os

import requests

API_URL = os.getenv("API_GATEWAY_URL", "http://localhost:8000")
SIMULATION_URL = os.getenv("SIMULATION_SERVICE_URL", "http://localhost:8003")
CONTROL_URL = os.getenv("CONTROL_SERVICE_URL", "http://localhost:8002")
FORECAST_URL = os.getenv("FORECAST_SERVICE_URL", "http://localhost:8001")

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
        "Simulation": health_simulation(),
        "Control": health_control(),
        "Forecast": health_forecast(),
    }
    return {name: "error" not in r for name, r in results.items()}


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
        },
    )


def simulate_scenario(hours: int = 24, scenario: str = "summer") -> dict:
    """
    시나리오 시뮬레이션 요청.
    Simulation Service의 GET /simulate/scenario 호출 예정.
    """
    return _get(f"{SIMULATION_URL}/simulate/scenario?hours={hours}&scenario={scenario}")
