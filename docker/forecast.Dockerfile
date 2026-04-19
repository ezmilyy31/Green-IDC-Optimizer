FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# 1. OS 종속성 설치 (캐시 활용을 위해 위쪽에 배치)
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*
    
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY pyproject.toml uv.lock .python-version /app/
RUN uv sync --frozen --no-dev --extra forecast

COPY . /app

CMD ["uv", "run", "--frozen", "--no-dev", "--extra", "forecast", "uvicorn", "apps.forecast_service.main:app", "--host", "0.0.0.0", "--port", "8001"]