# AI GREEN IDC — CLAUDE.md

> Claude Code(및 AI 어시스턴트)가 본 프로젝트를 이해하고 올바르게 지원하기 위한
> 프로젝트 컨텍스트 파일입니다. 프로젝트 루트에 위치해야 합니다.

---

## 프로젝트 개요

- **프로젝트명**: AI GREEN IDC — 데이터센터 냉각 최적화 시스템
- **과목**: CSE4186 / AIE4090 (CD) Agentic AI, 2026 Spring, Sogang University
- **트랙**: Track 2 (AI GREEN IDC)
- **목표**: Sinergym 시뮬레이터 기반 가상 IDC 환경에서 PUE를 최소화하는 냉각 제어 시스템 구축
- **제출 형태**: GitHub Public Repository URL
- **선택 모듈**: 미정 (MPC / RL 중 결정 예정)
- **보너스 구현**: 미구현

---

## 핵심 제약 사항 (절대 준수)

1. **열역학 모델은 직접 구현 필수** — 라이브러리로 대체 불가
2. **모듈 간 통신은 REST API만 허용** — 직접 함수 호출 절대 금지
3. **유료 클라우드(AWS, GCP 등) 및 유료 LLM API 사용 금지**
4. **외부 유료 데이터셋 사용 금지**
5. **docker-compose up 단일 명령으로 전체 시스템 기동 필수**
6. **GitHub 레포지토리는 제출 시점에 Public 상태 필수**

---

## 개발 환경

```
OS       : Ubuntu 22.04 LTS (WSL2 가능)
Language : Python 3.10 ~ 3.12
Container: Docker 24.0+, Docker Compose 2.0+
VCS      : GitHub (Public Repository)
```

---

## 시스템 아키텍처

```
┌──────────────────────────────────────────────────────┐
│                   docker-compose                      │
│                                                      │
│  [data_pipeline] ──► [thermo_model :8001]            │
│         │                    │                       │
│  [forecast_api :8002] ──► [control_api :8003]        │
│                            [optional_api :8004] ◄미정 │
│         └──────────────────────┘                     │
│              [dashboard :8501]                        │
└──────────────────────────────────────────────────────┘
```

| 서비스 | 포트 | 역할 |
|--------|------|------|
| data_pipeline | 내부 | 워크로드/기상 데이터 수집 및 전처리 |
| thermo_model | 8001 | 냉각 부하, 칠러 전력, PUE 계산 (직접 구현) |
| forecast_api | 8002 | POST /api/v1/forecast |
| control_api | 8003 | POST /api/v1/control/optimize |
| optional_api | 8004 | POST /api/v1/control/rl 또는 /mpc (미정) |
| dashboard | 8501 | Streamlit 통합 관제 UI |

---

## API 명세

### POST /api/v1/forecast
```json
// Request
{ "horizon_hours": 24 }

// Response
{
  "predictions": [...],
  "confidence_interval_90": { "lower": [...], "upper": [...] }
}
```

### POST /api/v1/control/optimize
```json
// Request
{
  "current_it_power_kw": 6200,
  "outdoor_temp_c": 28,
  "server_temp_c": 24.5,
  "timestamp": "2026-03-27T10:00:00"
}
// Response
{
  "supply_temp_setpoint_c": 20.0,
  "cooling_mode": "chiller",
  "expected_pue": 1.38,
  "hourly_plan": [...]
}
```

### POST /api/v1/control/rl  (선택 — 미정)
- 입력: 서버 온도, IT 전력, 외기 조건, 시간대
- 출력: 공급 온도, 냉각수 유량, Free Cooling 비율

---

## 핵심 수식 (열역학 모델 — 직접 구현 필수)

```python
# 서버 전력 (SPECpower 공식)
P_server = P_idle + (P_max - P_idle) * cpu_utilization
# CPU 서버: P_idle=200W, P_max=500W
# GPU 서버: P_idle=300W, P_max=1500W (A100 4장 기준)

# 냉각 부하
Q_cooling = m_dot * c_p * delta_T
# m_dot: 공기 유량(kg/s), c_p: 1.005 kJ/kg·K

# 칠러 COP (외기 온도 함수)
COP = max(2.0, 6.0 - 0.1 * (T_outdoor - 15))

# 칠러 전력
P_chiller = Q_cooling / COP

# PUE
PUE = (P_IT + P_cooling + P_other) / P_IT

# Free Cooling 전환 조건
free_cooling_available = (T_outdoor < 15)

# 탄소 배출량
carbon_tco2 = energy_mwh * 0.459  # tCO2/MWh
```

---

## 데이터셋

### 1. Google Cluster Trace 2019 (필수)
- **권장**: Kaggle 샘플 → https://www.kaggle.com/datasets/derrickmwiti/google-2019-cluster-sample
- **대안**: Zenodo 전처리 → https://zenodo.org/records/14564935
- **요구사항**: 최소 100개 VM, 24h+, 5분 단위
- **주요 컬럼**: `timestamp`, `avg_cpu`, `avg_mem`

### 2. 기상청 공공데이터포털 API (필수)
- **발급**: https://data.go.kr → "기상청_지상(종관, ASOS) 시간자료 조회서비스"
- **관측소**: `stnIds=119` (수원)
- **수집**: 기온(ta), 습도(hm), 풍속(ws)

### 3. Sinergym (필수)
```bash
docker pull sailugr/sinergym:latest
# 환경: Eplus-datacenter-mixed-continuous-stochastic-v1
```
- **목적**: 열역학 모델 검증 (오차 10% 이내)

### 4. SPECpower_ssj2008 (필수)
- https://www.spec.org/power_ssj2008/results/
- P_idle, P_max 파라미터 참조

---

## 성능 기준

### 필수 항목
| 항목 | 기준 |
|------|------|
| IT 부하 예측 MAPE (24h) | 5% 이내 |
| IT 부하 예측 MAPE (168h) | 8% 이내 |
| 냉각 수요 예측 nMAE | 10% 이내 |
| 90% PI Coverage | 85% 이상 |
| 열역학 모델 오차 vs Sinergym | 10% 이내 |
| Rule-based 온도 유지율 | 95% 이상 (18~27°C) |
| 위기 시나리오 온도 유지 | 27°C 이하 |
| API 응답 시간 | 10초 이내 (p95) |
| 단위 테스트 Coverage | 60% 이상 (pytest) |

### 선택 모듈 기준 (미정)
| 항목 | 기준 | 모듈 |
|------|------|------|
| MPC 에너지 절감률 | 10% 이상 vs Rule-based | 선택 A |
| MPC 최적화 계산 시간 | 5분 이내 | 선택 A |
| RL PUE 개선율 | 10% 이상 vs Rule-based | 선택 B |
| RL 온도 위반율 | 5% 미만 | 선택 B |

---

## 위기 시나리오 (3개 이상 필수)

| 시나리오 | 조건 | 제약 |
|----------|------|------|
| 서버 급증 | IT 부하 +30% | 27°C 이하 유지 |
| 냉각기 고장 | 칠러 1대 탈락 (용량 50%) | 27°C 이하 유지 |
| 폭염 | 외기 온도 38°C 이상 | 27°C 이하 유지 |

---

## 디렉토리 구조 (권장)

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

---

## 코드 품질 기준

```python
# PEP 8 준수 (black 포맷터 사용)
# 핵심 함수에 Type Hinting 필수

def calculate_cooling_load(
    m_dot: float,
    c_p: float,
    supply_temp: float,
    return_temp: float
) -> float:
    """
    열역학 기반 냉각 부하 계산.

    Args:
        m_dot: 공기 유량 (kg/s)
        c_p: 공기 비열 (kJ/kg·K)
        supply_temp: 공급 온도 (°C)
        return_temp: 환기 온도 (°C)

    Returns:
        냉각 부하 (kW)
    """
    return m_dot * c_p * (return_temp - supply_temp)
```

---

## 주요 일정

| 날짜 | 이벤트 |
|------|--------|
| 3/27 | 팀 구성원/주제 제안 발표 |
| 4/10, 4/17 | 프로젝트 계획 발표 (Peer-Review) |
| **4/24** | **중간고사: Tech Report 개인 제출** (docx, TA 이메일, 파일명: TR-X조-이름-학번.docx) |
| 5/8, 5/15 | 중간 점검 1차 발표 |
| **5/15** | **경진대회 참가 신청 마감** |
| 5/22, 5/29 | 중간 점검 2차 발표 (예선) |
| 6/5 | 서강 캡스톤 경진대회 본선 (다산관 103호, 13:30) |
| 6/12, 6/19 | 최종 발표 (Peer-Review) |
| 학기말 | GitHub Public URL 최종 제출 |

---

## 참고 링크

- Sinergym: https://github.com/ugr-sail/sinergym
- Google Cluster Trace (Kaggle): https://www.kaggle.com/datasets/derrickmwiti/google-2019-cluster-sample
- Zenodo 전처리 버전: https://zenodo.org/records/14564935
- 기상청 API: https://data.go.kr
- SPECpower: https://www.spec.org/power_ssj2008/results/
- Stable-Baselines3: https://stable-baselines3.readthedocs.io
- FastAPI: https://fastapi.tiangolo.com
- Streamlit: https://docs.streamlit.io

---

## AI 어시스턴트 활용 지침

- **유료 LLM API 호출 코드는 프로덕션 코드에 포함 불가** (과제 규정)
- 코드 리뷰, 알고리즘 설계, 디버깅 보조 용도로만 활용
- 열역학 수식 구현은 AI 제안을 참고하되 팀원이 직접 검증하고 구현
- 생성된 코드는 반드시 팀원이 이해하고 설명할 수 있어야 함
