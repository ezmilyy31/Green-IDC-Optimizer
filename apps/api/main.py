import os

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

app = FastAPI(title="AI Green IDC API Gateway")

_CONTROL_URL  = os.getenv("CONTROL_SERVICE_URL",  "http://control-service:8002")
_FORECAST_URL = os.getenv("FORECAST_SERVICE_URL", "http://forecast-service:8001")

_TIMEOUT = 10.0


async def _proxy_post(target_url: str, request: Request) -> JSONResponse:
    body = await request.json()
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        try:
            resp = await client.post(target_url, json=body)
            resp.raise_for_status()
            return JSONResponse(content=resp.json(), status_code=resp.status_code)
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=str(e))
        except httpx.RequestError as e:
            raise HTTPException(status_code=503, detail=f"서비스 연결 실패: {e}")


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "api-gateway"}


# ── Control Service ───────────────────────────────────────────────────────────

@app.post("/api/v1/control/optimize")
async def control_optimize(request: Request) -> JSONResponse:
    return await _proxy_post(f"{_CONTROL_URL}/api/v1/control/optimize", request)


@app.post("/control/rule-based")
async def control_rule_based(request: Request) -> JSONResponse:
    return await _proxy_post(f"{_CONTROL_URL}/control/rule-based", request)


@app.post("/control/rl")
async def control_rl(request: Request) -> JSONResponse:
    return await _proxy_post(f"{_CONTROL_URL}/control/rl", request)
