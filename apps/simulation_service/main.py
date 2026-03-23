from fastapi import FastAPI

app = FastAPI(title="Simulation Service")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "simulation-service"}