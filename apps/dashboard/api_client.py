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
    """POST /api/v1/control/optimize — API Gateway 경유 최적 제어.

    TODO(Simulation Service): 응답의 `expected_pue` 필드가 현재 고정값 1.35를 반환.
        Simulation Service /simulate/step 연동 후 실측 PUE로 교체 필요 (api_spec.md §2).
    TODO(Control Service): 현재 Rule-based 로직 고정.
        MPC(POST /api/v1/control/mpc) 또는 RL(POST /control/rl) 구현 완료 후
        가장 성능이 좋은 방식으로 라우팅 변경 (api_spec.md §2 교체 필요 항목).
    """
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
    """POST /control/rl — API Gateway 경유 RL 에이전트 제어.

    TODO(RL): 현재 Control Service가 고정값을 반환 중:
        { cooling_mode: "hybrid", supply_air_temp_setpoint_c: 20.0,
          free_cooling_ratio: 0.5, expected_pue: 1.35 }
        Week 4 PPO/DQN RL 에이전트 구현 완료 후 실제 추론값으로 교체 (api_spec.md §2).
    """
    return _post(
        f"{API_URL}/control/rl",
        _build_control_payload(outdoor_temp_c, it_power_kw, outdoor_humidity_pct),
    )


# ── Simulation endpoints (서비스 구현 후 활성화) ──────────────────────────────
# TODO(Simulation Service): 아래 두 함수는 Simulation Service FastAPI 서버 구현 후 활성화.
#   현재 docker-compose.yml의 simulation-service command가 test_sinergym.py(일회성 스크립트)로
#   설정되어 있어 헬스체크 항상 실패 — FastAPI 서버 실행 명령으로 교체 필요 (api_spec.md §5).

def simulate_step(outdoor_temp: float, it_power_kw: float, supply_temp_setpoint: float = 18.0) -> dict:
    """POST /simulate/step — Simulation Service 연동 예정 (api_spec.md §5).

    TODO(Simulation Service): 서비스 구현 후 simulation.py의 단일 스텝 계산을 이 호출로 교체.
    """
    return _post(
        f"{SIMULATION_URL}/simulate/step",
        {
            "outdoor_temp_c":         outdoor_temp,
            "it_power_kw":            it_power_kw,
            "supply_temp_setpoint_c": supply_temp_setpoint,
        },
    )


def simulate_24h(
    scenario: str,
    num_cpu: int,
    num_gpu: int,
    base_util: float,
    supply_temp_c: float,
    crisis: str | None = None,
) -> dict:
    """POST /simulate/24h — Simulation Service 연동 예정 (api_spec.md §5).

    TODO(Simulation Service): 서비스 구현 후 simulation.py의 run_simulation()을 이 호출로 교체.
        현재 simulation.py에서 domain.* 직접 import로 처리 중 (명세서 아키텍처 원칙 위반).
    """
    # TODO(Simulation Service): /simulate/24h 엔드포인트 스키마 확정 후 payload 수정
    return _post(
        f"{SIMULATION_URL}/simulate/24h",
        {
            "scenario":       scenario,
            "num_cpu":        num_cpu,
            "num_gpu":        num_gpu,
            "base_util":      base_util,
            "supply_temp_c":  supply_temp_c,
            "crisis":         crisis,
        },
    )
