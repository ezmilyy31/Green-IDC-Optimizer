# Runbook

## 개발환경 구성

1. **저장소 클론 및 폴더 이동**

   ```bash
   git clone <레포지토리_URL>
   cd AI-Green-IDC
   ```

2. **환경변수 파일 생성 및 설정**

   ```bash
   cp .env.example .env
   ```

   > `.env` 파일을 열어 `KMA_API_KEY` 등 각자의 로컬 환경에 맞는 값을 기입합니다.

3. **가상환경 생성 및 패키지 설치 (`uv` 사용)**
   ```bash
   uv venv
   source .venv/bin/activate  # Windows의 경우: .venv\Scripts\activate
   uv sync
   ```

---

## Docker 실행

### 전체 실행

```bash
# 전체 실행
docker compose up --build

# 백그라운드 실행
docker compose up -d --build

# 종료
docker compose down

# 로그 확인
docker compose logs -f api
docker compose logs -f forecast-service
docker compose logs -f control-service
docker compose logs -f simulation-service
docker compose logs -f dashboard
```

### simulation-service (Sinergym)

```bash
docker compose up --build simulation-service
```

### forecast-service

1. LGBM 모델 생성

   ```bash
   uv run python -m domain.forecasting.train.train_lgbm_it_load
   ```

   > `data/models/`에 joblib 파일이 생성됩니다.

2. 서비스 실행

   ```bash
   docker compose up --build forecast-service
   ```

3. 요청 예시

   <details>
   <summary>예시 Request</summary>

   ```bash
   # health check
   curl http://localhost:8001/health

   # LGBM IT_load 예측
   curl -X POST http://localhost:8001/api/v1/forecast \
      -H "Content-Type: application/json" \
      -d '{
         "prediction_target": "it_load",
         "model_type": "lgbm",
         "forecast_horizon_hours": 24,
         "include_prediction_interval": true
      }'
   ```

   </details>

---

## 로컬환경에서 서비스 실행

```bash
# API Gateway
uv run uvicorn apps.api.main:app --reload --host 0.0.0.0 --port 8000

# Forecast Service
uv run --extra forecast uvicorn apps.forecast_service.main:app --reload --host 0.0.0.0 --port 8001

# Control Service
uv run --extra control uvicorn apps.control_service.main:app --reload --host 0.0.0.0 --port 8002

# Simulation Service
uv run uvicorn apps.simulation_service.main:app --reload --host 0.0.0.0 --port 8003

# Dashboard
uv run --extra dashboard streamlit run apps/dashboard/app.py --server.port 8501
```

---

## RL 학습 및 평가

커스텀 IDCEnv 기반 (Sinergym 불필요, 로컬에서 실행 가능).

### 학습

```bash
# 포그라운드
uv run python -m domain.controllers.rl_agent \
  --custom-env \
  --total-timesteps 300000 \
  --run-name exp-300k \
  --w-energy 0.5 \
  --gamma 0.9

# 백그라운드
nohup uv run python -m domain.controllers.rl_agent \
  --custom-env \
  --total-timesteps 300000 \
  --run-name exp-300k \
  > logs/exp-300k.log 2>&1 &
```

주요 파라미터:

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `--custom-env` | False | IDCEnv 사용 (커스텀 열역학 모델) |
| `--algo` | ppo | 알고리즘 (`ppo` \| `sac`) |
| `--total-timesteps` | 50000 | 총 학습 step |
| `--run-name` | ppo-baseline | 모델 저장 이름 |
| `--w-energy` | 0.5 | 에너지 항 가중치 (0~1) |
| `--gamma` | 0.9 | 할인율 |
| `--lr` | 3e-4 | learning rate |
| `--ent-coef` | 0.0 | entropy 보너스 계수 (탐색 강도) |
| `--resume` | None | 이어서 학습할 모델 경로 |

학습 로그는 `data/logs/{run-name}.csv`에 자동 저장됩니다.

### Docker로 학습

```bash
# 빌드
docker compose build rl-service

# 학습 실행
docker compose run --rm rl-service uv run python -m domain.controllers.rl_agent \
  --custom-env \
  --total-timesteps 300000 \
  --run-name exp-300k \
  --w-energy 0.5 \
  --gamma 0.9

# 백그라운드 실행
nohup docker compose run --rm rl-service uv run python -m domain.controllers.rl_agent \
  --custom-env \
  --total-timesteps 300000 \
  --run-name exp-300k \
  > logs/exp-300k.log 2>&1 &
```

> `rl-service`는 `profiles: [rl]`로 설정되어 있어 `docker compose up --build` 시 자동 포함되지 않습니다.

### 평가

```bash
uv run python scripts/eval_baseline.py \
  --model data/models/exp-300k.zip \
  --episodes 20
```
