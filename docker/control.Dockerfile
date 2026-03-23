FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY pyproject.toml uv.lock .python-version /app/
RUN uv sync --frozen --no-dev --extra control

COPY . /app

CMD ["uv", "run", "--extra", "control", "uvicorn", "apps.control_service.main:app", "--host", "0.0.0.0", "--port", "8002"]