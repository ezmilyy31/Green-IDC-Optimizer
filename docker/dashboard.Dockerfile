FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY pyproject.toml uv.lock .python-version /app/
RUN uv sync --frozen --no-dev --extra dashboard

COPY . /app

CMD ["uv", "run", "--frozen", "--no-dev", "--extra", "dashboard", "streamlit", "run", "apps/dashboard/app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]