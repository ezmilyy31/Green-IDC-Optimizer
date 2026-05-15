FROM python:3.11-bookworm

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# libgomp1은 non-slim 이미지에 이미 포함되어 있어 별도 apt 단계 불필요
# (slim 사용 시 deb.debian.org 미러가 KR/arm64에서 매우 느려 빌드 실패 빈발)

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY pyproject.toml uv.lock .python-version /app/
RUN uv sync --frozen --no-dev --extra forecast

COPY . /app

CMD ["uv", "run", "--frozen", "--no-dev", "--extra", "forecast", "uvicorn", "apps.forecast_service.main:app", "--host", "0.0.0.0", "--port", "8001"]