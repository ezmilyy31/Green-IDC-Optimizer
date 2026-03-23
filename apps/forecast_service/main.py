from fastapi import FastAPI

app = FastAPI(title="Forecast Service")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "forecast-service"}