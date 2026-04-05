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

## docker 실행

### simulation-service 실행 (sinergym)

```
docker compose up --build simulation-service
```

테스트를 위해 `docker-compose.yml`에서 `app/simulation_service/test_sinergym.py`를 실행하도록 설정해 둠.

### 그 외 실행

```
// 전체 실행
docker compose up --build

// 백그라운드 실행
docker compose up -d --build

// 종료
docker compose down

// 로그 확인
docker compose logs -f api
docker compose logs -f forecast-service
docker compose logs -f control-service
docker compose logs -f simulation-service
docker compose logs -f dashboard
```

## 로컬환경에서 서비스 실행

API Gateway

```
uv run uvicorn apps.api.main:app --reload --host 0.0.0.0 --port 8000
```

Forecast Service

```
uv run --extra forecast uvicorn apps.forecast_service.main:app --reload --host 0.0.0.0 --port 8001
```

Control Service

```
uv run --extra control uvicorn apps.control_service.main:app --reload --host 0.0.0.0 --port 8002
```

Simulation Service

```
uv run uvicorn apps.simulation_service.main:app --reload --host 0.0.0.0 --port 8003
```

Dashboard

```
uv run --extra dashboard streamlit run apps/dashboard/app.py --server.port 8
```

## 프로젝트 폴더 구성

apps = 실행 단위
core = 공용 코드
domain = 핵심 문제 해결 로직
data = 입출력 자산
docs = 설명서
tests = 검증 코드

```
AI-Green-IDC/                           # 프로젝트 루트 디렉토리. 전체 소스, 데이터, 문서, 실행 설정을 포함하는 최상위 폴더
├─ apps/                                # 실제 실행 가능한 애플리케이션 계층 모음. 각 서비스는 독립 프로세스/컨테이너로 실행 가능
│  ├─ api/                              # 외부에서 가장 먼저 진입하는 통합 API 게이트웨이(FastAPI). 요청을 받아 내부 서비스로 전달
│  ├─ forecast_service/                 # 부하 예측 및 냉각 수요 예측 서비스. ML/DL 모델 추론 API 제공
│  ├─ control_service/                  # 제어 서비스. Rule-based, PID, RL, 향후 MPC 등의 제어 로직을 API 형태로 제공
│  ├─ simulation_service/               # 시뮬레이션 서비스. 열역학 계산, PUE 계산, 운영 시나리오 평가 등을 수행
│  └─ dashboard/                        # Streamlit 기반 사용자 대시보드. 운영 현황, 예측 결과, 제어 결과를 시각화
│
├─ core/                                # 여러 서비스에서 공통으로 사용하는 핵심 공용 모듈 모음
│  ├─ config/                           # 환경변수, 설정값, 상수 정의. 예: API URL, 포트, 모델 경로, 기준 온도 범위
│  ├─ schemas/                          # Pydantic 기반 데이터 스키마. 요청/응답 형식, 검증 규칙, 직렬화 구조 정의
│  ├─ clients/                          # 서비스 간 REST 호출용 클라이언트 코드. 예: api→forecast_service 호출 래퍼
│  └─ utils/                            # 범용 유틸리티 함수 모음. 날짜 처리, 로깅, 파일 입출력, 공통 계산 함수 등
│
├─ domain/                              # 비즈니스 로직/문제 해결 로직이 위치하는 핵심 도메인 계층
│  ├─ data_pipeline/                    # 데이터 수집·정제·가공 파이프라인 관련 모듈
│  │  ├─ google_trace/                  # Google Cluster Trace 데이터 수집/파싱/전처리 로직
│  │  ├─ weather/                       # 기상청 API 또는 기상 데이터 수집/정제 로직
│  │  └─ feature_engineering/           # 예측 모델 입력용 feature 생성 로직. lag, rolling, 시간 파생변수 등 생성
│  ├─ thermodynamics/                   # 데이터센터 냉각 및 전력 계산을 위한 열역학 모델 구현 모듈
│  │  ├─ it_power.py                    # IT 장비(서버/랙) 전력 사용량 계산 로직
│  │  ├─ cooling_load.py                # 냉각 부하(Q) 계산 로직. 유량, 비열, 온도차 기반 계산 등 담당
│  │  ├─ chiller.py                     # 칠러 소비전력 및 COP 계산 로직
│  │  ├─ free_cooling.py                # 외기 조건 기반 Free Cooling 가능 여부 및 절감 효과 계산 로직
│  │  └─ pue.py                         # PUE 계산 로직. 총 전력 대비 IT 전력 비율 산출
│  ├─ forecasting/                      # 예측 모델 정의 및 학습/추론 관련 도메인 로직
│  │  ├─ lgbm_model.py                  # LightGBM 기반 예측 모델 구현 파일
│  │  ├─ lstm_model.py                  # LSTM 기반 시계열 예측 모델 구현 파일
│  │  └─ intervals.py                   # 예측 구간(예: 90% Prediction Interval) 계산 로직
│  ├─ controllers/                      # 제어 알고리즘 관련 핵심 로직 모음
│  │  ├─ rule_based.py                  # 고정 규칙 기반 제어 로직. 임계치 조건에 따라 냉각 모드/설정값 결정
│  │  ├─ pid.py                         # PID 제어기 구현. 목표 온도와 실제 온도 차이를 기반으로 연속 제어
│  │  ├─ rl_env.py                      # 강화학습 환경 정의. 상태, 행동, 보상 함수, step/reset 로직 포함
│  │  └─ rl_agent.py                    # 강화학습 에이전트 학습/추론 로직. PPO 등 RL 알고리즘 연동
│  └─ esg/                              # ESG/탄소배출/에너지 절감 효과 분석 관련 로직
│
├─ data/                                # 프로젝트에서 사용하는 데이터 저장 공간
│  ├─ raw/                              # 원본 데이터 저장 폴더. 수집 직후의 비가공 데이터 보관
│  ├─ interim/                          # 1차 정제 또는 병합된 중간 데이터 저장 폴더
│  ├─ processed/                        # 모델 학습/평가/추론에 바로 사용할 수 있도록 가공 완료된 데이터 저장 폴더
│  └─ models/                           # 학습 완료된 모델 파일, scaler, tokenizer, 메타데이터 저장 폴더
│
├─ docs/                                # 프로젝트 문서화 자료 저장 폴더
│  ├─ architecture.md                   # 전체 시스템 아키텍처 문서. 서비스 구성, 데이터 흐름, 배포 구조 설명
│  ├─ thermodynamic_model.md            # 열역학 모델 수식, 가정, 변수 정의, 계산 방식 설명 문서
│  └─ api_spec.md                       # REST API 명세 문서. 엔드포인트, 요청/응답 구조, 에러 코드 정의
│
├─ tests/                               # 단위 테스트, 통합 테스트, API 테스트 코드 저장 폴더
├─ docker/                              # Docker 관련 설정 파일 저장 폴더. 각 서비스별 Dockerfile 또는 보조 스크립트 포함 가능
├─ docker-compose.yml                   # 여러 컨테이너(api, forecast, control, dashboard 등)를 한 번에 띄우는 오케스트레이션 파일
├─ .env.example                         # 환경변수 예시 파일. 실제 실행 시 필요한 키/설정값의 템플릿 제공
└─ README.md                            # 프로젝트 소개, 실행 방법, 폴더 구조, 개발 가이드 등을 설명하는 메인 문서
```
