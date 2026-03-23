FROM sailugr/sinergym:latest

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Sinergym 이미지는 비루트(non-root) 사용자가 기본일 수 있으므로 설치를 위해 root로 전환
USER root

COPY pyproject.toml uv.lock .python-version /app/
RUN uv pip install --system --break-system-packages .[control]

COPY . /app

CMD ["uvicorn", "apps.simulation_service.main:app", "--host", "0.0.0.0", "--port", "8003"]