# README.md

## 1. 프로젝트 개요

본 프로젝트에서는 IT 부하 예측을 통해 강화학습을 기반으로 냉각 시스템을 제어합니다. 이를 통해 Data Center의 PUE를 최적화함으로써 에너지 효율을 개선하는 것을 목표로 합니다.

## 2. 팀 소개

팀명: 짱돌

| **팀원** | **학번** | **담당 역할** |
| --- | --- | --- |
| **김효림 (팀장)** | 20221549 | • 제어 스키마 설계 및 Control Service 구축<br>• 커스텀 RL 환경 구현<br>• PPO/SAC 학습 파이프라인 구현<br>• 베이스라인 평가 프레임워크 구축<br>• RL 학습 전반 총괄 |
| **심하연** | 20221573 | • MSA 설계 및 Docker 환경 구성<br>• Streamlit 대시보드 프론트엔드 구현 및 API 연동<br>• RL 학습<br>• 커스텀 보상 함수 보완 |
| **김다은** | 20231515 | • Forecast Service 스키마 설계 및 구축<br>• LightGBM 기반 IT 부하·냉각 수요 예측 모델 구현<br>• RL 학습 |
| **김성현** | 20201564 | • 열역학 모델 구현 및 검증<br>• PID 시뮬레이터 구현<br>• 데이터 물리학적 검증<br>• RL 학습 |
| **심서연** | 20221571 | • 데이터 수집 및 전처리<br>• 데이터 파이프라인 구축 및 파생변수<br>• 모델 비교용 이동평균 예측 모델 구현<br>• RL 학습 |

## 3. 프로젝트 실행

```bash
#가상환경 생성 및 패키지 설치
Green-IDC-Optimizer$ uv venv
Green-IDC-Optimizer$ source .venv/bin/activate
Green-IDC-Optimizer$ uv sync

# 전체 실행
Green-IDC-Optimizer$ docker compose up --build 
# 백그라운드 전체 실행
Green-IDC-Optimizer$ docker compose up -d --build
# 전체 종료
Green-IDC-Optimizer$ docker compose down
```

→ 자세한 내용은 [docs/RUNBOOK.md](docs/RUNBOOK.md) 파일 참고

## 4. 프로젝트 상세 내용

![System Architecture.png](docs/system_architecture.png)

### 4-1. 기술 스택

#### **4-1-1. Backend / API**

- Python 3.11 (uv 패키지 매니저)
- FastAPI + Uvicorn - API Gateway, Forecast/Control/Simulation 서비스
- Pydantic - 요청/응답 Schema

#### **4-1-2. ML / AI**

- PyTorch - LSTM 시계열 예측 모델
- LightGBM - 그래디언트 부스팅 예측 모델
- Stable-Baselines3 - RL 에이전트 (SAC)
- Gymnasium - RL 환경 인터페이스
- Scikit-learn - 스케일러, 전처리
- Sinergym (sailugr/sinergym)??(다른거 쓴 걸루 기억) - EnergyPlus 기반 데이터센터 시뮬레이터

#### **4-1-3. Frontend / 시각화**

- Streamlit - 대시보드
- Plotly / Altair - 차트

#### **4-1-4. 인프라**

- Docker + Docker Compose - 마이크로서비스 컨테이너화
- Pandas, Numpy - 데이터 전처리

### 4-2. 프로젝트 상세

#### 4-2-1. 열역학 기반 냉각 부하 모델

**`domain/thermodynamics/`**

- **SPECpower_ssj2008 기반 서버 전력 모델**: `P = P_idle + (P_max - P_idle) × cpu_util` (CPU서버: idle 200W / max 500W, GPU서버: idle 300W / max 1500W)
- **냉각 부하 계산**: `Q = ṁ × cp × ΔT` 직접 구현
- **칠러 COP 모델**: 외기온도 + 공급온도 + 부분부하율(PLR) 기반 비선형 효율 모델
- **습구온도 기반 냉각 모드 자동 전환**: Free Cooling(습구 <10°C) / Hybrid(<18°C)??? / Chiller(≥18°C)
- **PUE 계산**: 구글 데이터센터 PUE 1.10 벤치마크 기준

#### 4-2-2. IT 부하 / 냉각 수요 예측

**`domain/forecasting/`**

- **LightGBM**: lag feature + rolling 통계 + 캘린더 특성, 24h/168h ahead 예측
- 출력: 예측값 + **90% 신뢰구간**

#### 4-2-3. RL 기반 냉각 제어

**`domain/controllers/`**

- **커스텀 IDC Gym 환경** (`idc_env.py`): 실측 Google Cluster Trace 2019 + 기상청 ASOS 데이터 기반 (365일 × 288스텝)
- **관측 공간**: 9차원 (시간, 외기온도, 습도, CPU 사용률, 존 온도, 공급온도, IT전력, 습구온도 등)
- **행동 공간**: 공급온도 설정값 [18, 25]°C
- **알고리즘**: SAC (Soft Actor-Critic) + PPO (Proximal Policy Optimization)???? (Stable-Baselines3)
- **도메인 랜덤화**: 학습 중 위기 시나리오 자동 주입 — 서버급증(CPU ×1.3) / 폭염(외기 +5~10°C) / 칠러 효율 저하
- **2-tier 안전 시스템**: RL 추론 + 존 온도 26.5°C 초과 시 강제 냉각 fallback
- **Rule-based 컨트롤러** + **PID 제어기** (Anti-windup, Incremental PID) 구현

#### 4-2-4. 통합 관제 대시보드

**`apps/dashboard/`**

- **실시간 대시보드**: PUE 게이지, 온도, 전력 KPI
- **운영 관리**: 파라미터 조작 → 24h 시뮬레이션 즉시 실행
- **ESG 지표**: 탄소 배출량 (0.459 tCO₂/MWh), WUE, 에너지 비용
- **분석 도구**: 예측 모델 성능 비교, feature importance
- **RL vs Rule-based**: 두 제어 방식 PUE/온도 위반율 비교

#### 4-2-5. 위기 시나리오 시뮬레이터

- 상황 1 ) 서버 급증: IT 부하 +30%
- 상황 2 ) 칠러 고장: 냉각 용량 50% 저하
- 상황 3 ) 폭염: 외기 38°C 이상
