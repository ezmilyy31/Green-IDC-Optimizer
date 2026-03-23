from fastapi import FastAPI

app = FastAPI(title="AI Green IDC API Gateway")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "api-gateway"}