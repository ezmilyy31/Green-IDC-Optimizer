from fastapi import FastAPI

app = FastAPI(title="Control Service")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "control-service"}