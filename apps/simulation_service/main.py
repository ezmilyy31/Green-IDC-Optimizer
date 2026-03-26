import sys
import os

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from fastapi import FastAPI

from core.schemas.simulation import SimulationRequest, SimulationResponse
from domain.thermodynamics.chiller import calculate_chiller_power_kw
from domain.thermodynamics.cooling_load import calculate_cooling_load_from_it_power_kw
from domain.thermodynamics.pue import calculate_pue

app = FastAPI(title="Simulation Service")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "simulation-service"}


@app.post("/api/v1/simulation/calculate", response_model=SimulationResponse)
def calculate(req: SimulationRequest) -> SimulationResponse:
    """
    열역학 모델 계산 엔드포인트.

    IT 전력과 외기 온도를 입력받아 냉각 부하, 칠러 전력, PUE를 반환한다.
    """
    cooling_load_kw = calculate_cooling_load_from_it_power_kw(req.it_power_kw)
    chiller_result = calculate_chiller_power_kw(cooling_load_kw, req.outdoor_temp_c)
    pue_result = calculate_pue(req.it_power_kw, chiller_result.chiller_power_kw)

    return SimulationResponse(
        cooling_load_kw=cooling_load_kw,
        cooling_mode=chiller_result.cooling_mode.value,
        cop=chiller_result.cop,
        chiller_power_kw=chiller_result.chiller_power_kw,
        pue=pue_result.pue,
        total_power_kw=pue_result.total_power_kw,
        it_power_kw=pue_result.it_power_kw,
        cooling_power_kw=pue_result.cooling_power_kw,
        other_power_kw=pue_result.other_power_kw,
    )
