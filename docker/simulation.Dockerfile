FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY pyproject.toml uv.lock .python-version /app/
RUN uv sync --frozen --no-dev

COPY . /app

CMD ["uv", "run", "uvicorn", "apps.simulation_service.main:app", "--host", "0.0.0.0", "--port", "8003"]