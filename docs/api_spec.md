# AI Green IDC — API 명세서

> 각 서비스의 REST API 엔드포인트, 요청/응답 스키마, 호출 예시를 정의합니다.
> 모든 서비스는 JSON을 입출력으로 사용하며, Content-Type은 `application/json`입니다.

---

## 표기 규칙

| 마크            | 의미                                               |
| --------------- | -------------------------------------------------- |
| ✅ 구현 완료    | 현재 동작하는 코드                                 |
| 🔧 값 교체 필요 | 동작하지만 하드코딩된 임시값 — 연동 후 반드시 수정 |
| 🚧 구현 예정    | 스켈레톤만 있거나 엔드포인트 미존재                |
| ⚠️ 규칙 위반    | 명세서 제약 조건과 불일치 — 추후 수정 필요         |

---

## 목차

1. [시스템 아키텍처](#1-시스템-아키텍처)
2. [API Gateway](#2-api-gateway-port-8000)
3. [Control Service](#3-control-service-port-8002)
4. [Forecast Service](#4-forecast-service-port-8001)
5. [Simulation Service](#5-simulation-service-port-8003)
6. [Dashboard](#6-dashboard-port-8501)
7. [공통 규격](#7-공통-규격)
8. [전체 TODO 목록](#8-전체-todo-목록)

---

## 1. 시스템 아키텍처

```
Browser (8501)
    └── Dashboard (Streamlit)
            └── API Gateway (8000)  ←── 모든 외부 요청 진입점
                    ├── Control Service  (8002)
                    ├── Forecast Service (8001)
                    └── Simulation Service (8003)
```

**서비스 간 통신 원칙**

- 모듈 간 직접 함수 호출 금지 — REST API 통신만 허용
- 대시보드는 API Gateway를 경유하여 내부 서비스에 접근
- 각 서비스는 독립적으로 실행 가능

> ⚠️ **현재 위반 사항**: `apps/dashboard/simulation.py`가 `domain.thermodynamics.*`를 직접 import하여 시뮬레이션을 수행 중.
> Simulation Service REST API 구현 완료 후 `POST /simulate/24h` 호출로 교체 필요.
> 관련 코드: `apps/dashboard/simulation.py` — `run_simulation()` 함수, `apps/dashboard/api_client.py` — `simulate_24h()` (미활성화)

---

## 2. API Gateway (port: 8000)

내부 마이크로서비스로의 요청을 프록시하는 단일 진입점입니다.

### `GET /health` ✅ 구현 완료

```json
{ "status": "ok", "service": "api-gateway" }
```

---

### `POST /api/v1/control/optimize` ✅ 구현 완료

최적 냉각 제어 설정값을 반환합니다. Control Service로 프록시됩니다.

> 🔧 **교체 필요**: 현재 Rule-based 로직 고정. MPC(`POST /api/v1/control/mpc`) 또는 RL(`POST /control/rl`) 구현 완료 후 가장 성능이 좋은 방식으로 교체.

**요청 (ControlRequest)**

| 필드                   | 타입           | 필수 | 설명                                  |
| ---------------------- | -------------- | ---- | ------------------------------------- |
| `outdoor_temp_c`       | float          | ✅   | 외기 온도 (°C)                        |
| `it_power_kw`          | float          | ✅   | 현재 IT 전력 (kW)                     |
| `outdoor_humidity_pct` | float          | —    | 외기 습도 (%), 기본값 50.0            |
| `timestamp`            | string \| null | —    | ISO 8601 시각 (RL time_of_day 특성용) |

> 🚧 **추가 예정 필드**: `server_inlet_temp_c` (서버 입구 온도), `server_outlet_temp_c` (서버 출구 온도) — Simulation Service 연동 시 추가.

```json
{
  "outdoor_temp_c": 32.0,
  "it_power_kw": 6200.0,
  "outdoor_humidity_pct": 60.0,
  "timestamp": "2024-07-16T14:00:00"
}
```

**응답 (ControlResponse)**

| 필드                         | 타입   | 상태 | 설명                                                                          |
| ---------------------------- | ------ | ---- | ----------------------------------------------------------------------------- |
| `cooling_mode`               | string | ✅   | 냉각 방식 (`"chiller"` \| `"free_cooling"` \| `"hybrid"`)                     |
| `supply_air_temp_setpoint_c` | float  | ✅   | CRAH 공급 온도 설정값 (°C)                                                    |
| `free_cooling_ratio`         | float  | ✅   | 자연공조 비율 (0.0 ~ 1.0)                                                     |
| `expected_pue`               | float  | 🔧   | 예상 PUE — **현재 고정값 `1.35`**, Simulation Service 연동 후 실측값으로 교체 |

> 🚧 **추가 예정 필드**: `chw_flow_setpoint_kg_s` (냉각수 유량) — Simulation Service 연동 시 추가.

```json
{
  "cooling_mode": "chiller",
  "supply_air_temp_setpoint_c": 19.5,
  "free_cooling_ratio": 0.0,
  "expected_pue": 1.35
}
```

**오류 응답**

| 코드 | 설명                      |
| ---- | ------------------------- |
| 503  | Control Service 연결 실패 |
| 422  | 요청 스키마 오류          |

---

### `POST /control/rule-based` ✅ 구현 완료

Rule-based 냉각 제어 결과를 반환합니다.
요청/응답 스키마는 `/api/v1/control/optimize`와 동일합니다.

---

### `POST /control/rl` 🔧 임시값 반환 중

RL 에이전트 제어 결과를 반환합니다.

> 🔧 **교체 필요**: 현재 아래 고정값을 반환. Week 4 RL 에이전트(PPO/DQN) 구현 후 실제 추론값으로 교체.

**현재 고정 응답값 (임시)**

```json
{
  "cooling_mode": "hybrid",
  "supply_air_temp_setpoint_c": 20.0,
  "free_cooling_ratio": 0.5,
  "expected_pue": 1.35
}
```

---

### `POST /api/v1/control/mpc` 🚧 구현 예정

MPC 기반 최적 제어 결과를 반환합니다. (명세서 선택 A)

> 🚧 API Gateway에 프록시 라우트 추가 및 Control Service 엔드포인트 구현 필요.
> 라이브러리: CasADi / CVXPY / scipy.optimize 중 선택.

---

## 3. Control Service (port: 8002)

외기 조건과 IT 전력을 입력받아 최적 냉각 설정값을 계산합니다.
API Gateway를 통해서만 접근하는 것을 권장합니다.

### `GET /health` ✅ 구현 완료

```json
{ "status": "ok", "service": "control-service" }
```

### `POST /api/v1/control/optimize` ✅ 구현 완료

### `POST /control/rule-based` ✅ 구현 완료

### `POST /control/rl` 🔧 임시값 반환 중

스키마는 [API Gateway 섹션](#post-apiv1controloptimize)과 동일합니다.

**냉각 모드 전환 기준 (Rule-based)**

| 외기 온도   | 냉각 모드                      |
| ----------- | ------------------------------ |
| < 15°C      | `free_cooling` (완전 자연공조) |
| 15°C ~ 22°C | `hybrid` (혼합 냉각)           |
| ≥ 22°C      | `chiller` (기계식 냉방)        |

### `POST /api/v1/control/mpc` 🚧 구현 예정

### `POST /control/scenario` 🚧 구현 예정

위기 시나리오를 제어 서비스에 주입합니다.

### `GET /control/status` 🚧 구현 예정

현재 제어 상태를 반환합니다.

### `GET /esg/summary` 🚧 구현 예정

탄소 배출량 및 WUE 요약을 반환합니다.

---

## 4. Forecast Service (port: 8001)

IT 부하 및 냉각 수요 예측 서비스입니다.

### `GET /health` ✅ 구현 완료

```json
{ "status": "ok", "service": "forecast-service" }
```

### `POST /api/v1/forecast` ✅ 구현 완료

> API Gateway(`POST /api/v1/forecast`) → Forecast Service 프록시

**요청 (ForecastRequest)**

| 필드                          | 타입           | 필수 | 설명                                                           |
| ----------------------------- | -------------- | ---- | -------------------------------------------------------------- |
| `forecast_horizon_hours`      | int            | ✅   | 예측 기간 (1 ~ 168시간)                                        |
| `prediction_target`           | string         | —    | `"it_load"` \| `"cooling_demand"` \| `"both"` (기본: `"both"`) |
| `model_type`                  | string         | —    | `"lgbm"` \| `"lstm"` (기본: `"lgbm"`)                          |
| `current_timestamp`           | string \| null | —    | 예측 기준 시각 (ISO 8601), null 시 현재 시각                   |
| `include_prediction_interval` | bool           | —    | 신뢰구간 포함 여부 (기본: `true`)                              |

**응답 (ForecastResponse)**

| 필드                                        | 타입   | 설명                      |
| ------------------------------------------- | ------ | ------------------------- |
| `prediction_target`                         | string | 예측 대상                 |
| `model_type_used`                           | string | 사용된 모델 타입          |
| `generated_at`                              | string | 예측 생성 시각 (ISO 8601) |
| `horizon_hours`                             | int    | 예측 기간 (h)             |
| `predictions`                               | array  | 시간별 예측값 리스트      |
| `predictions[].timestamp`                   | string | 예측 시각                 |
| `predictions[].predicted_it_load_kw`        | float  | IT 전력 예측값 (kW)       |
| `predictions[].predicted_cooling_load_kw`   | float  | 냉각 수요 예측값 (kW)     |
| `predictions[].cooling_mode`                | string | 예측 냉각 모드            |
| `predictions[].lower_bound_it_load_kw`      | float  | IT 부하 신뢰구간 하한     |
| `predictions[].upper_bound_it_load_kw`      | float  | IT 부하 신뢰구간 상한     |
| `predictions[].lower_bound_cooling_load_kw` | float  | 냉각 수요 신뢰구간 하한   |
| `predictions[].upper_bound_cooling_load_kw` | float  | 냉각 수요 신뢰구간 상한   |

**성능 목표 (명세서 4절)**

| 지표                     | 목표     |
| ------------------------ | -------- |
| IT 부하 예측 MAPE (24h)  | 5% 이내  |
| IT 부하 예측 MAPE (168h) | 8% 이내  |
| 90% PI Coverage          | 85% 이상 |
| 냉각 수요 예측 nMAE      | 10% 이내 |

---

## 5. Simulation Service (port: 8003)

Sinergym(EnergyPlus 기반) 환경을 통한 데이터센터 시뮬레이션 서비스입니다.

### `GET /health` ✅ 구현 완료

```json
{ "status": "ok", "service": "simulation-service" }
```

> ⚠️ **현재 상태**: `docker-compose.yml`의 `command`가 `test_sinergym.py`(일회성 테스트 스크립트)로 설정되어 있어 헬스체크가 항상 실패함. FastAPI 서버로 교체 필요.

### `POST /simulate/step` 🚧 구현 예정

단일 타임스텝 시뮬레이션을 실행합니다.

**요청**

| 필드                     | 타입  | 설명                                    |
| ------------------------ | ----- | --------------------------------------- |
| `outdoor_temp_c`         | float | 외기 온도 (°C)                          |
| `it_power_kw`            | float | IT 전력 (kW)                            |
| `supply_temp_setpoint_c` | float | CRAH 공급 온도 설정값 (°C), 기본값 18.0 |

**응답**

| 필드               | 타입  | 설명              |
| ------------------ | ----- | ----------------- |
| `zone_temp_c`      | float | 서버 존 온도 (°C) |
| `cooling_power_kw` | float | 냉각 전력 (kW)    |
| `pue`              | float | PUE               |
| `cop`              | float | COP               |

### `POST /simulate/24h` 🚧 구현 예정

24시간 시뮬레이션을 실행하고 시간별 결과를 반환합니다.

> 🚧 이 엔드포인트가 구현되면 `apps/dashboard/app.py`의 `domain.thermodynamics.*` 직접 import를 이 API 호출로 교체해야 함 (명세서 11절 위반 해소).

**성능 목표 (명세서 3절)**

| 지표                           | 목표     |
| ------------------------------ | -------- |
| 열역학 모델 오차 (vs Sinergym) | 10% 이내 |

---

## 6. Dashboard (port: 8501)

웹 브라우저에서 `http://localhost:8501` 접속합니다.

### 화면 구성

| 영역      | 컴포넌트         | 상태 | 설명                                       |
| --------- | ---------------- | ---- | ------------------------------------------ |
| Row 1     | PUE 게이지       | ✅   | 24시간 평균 PUE, NAVER 1.09 벤치마크 delta |
| Row 1     | 전력 소비 추이   | ✅   | IT / 칠러 / 총 전력 (24h 시계열)           |
| Row 1     | 서버 온도 분포   | ✅   | 랙별 환기 온도 바차트, 27°C 경고선         |
| Row 2     | 냉각 모드        | ✅   | 현재 모드, 외기/공급 온도, 도넛차트        |
| Row 2     | 제어 서비스 추천 | ✅   | Rule-Based / RL 설정값 (API Gateway 경유)  |
| Row 2     | ESG 지표         | ✅   | 탄소 배출, WUE, 물 사용량, 전력 비용       |
| Row 2     | 알람 이력        | ✅   | ERROR / WARN / INFO 최대 15건              |
| 위기 분석 | 시나리오 분석    | ✅   | 위기 모드 시 정상 대비 증가분 + 대응 전략  |
| Row 3     | KPI 카드         | ✅   | PUE / COP / IT 전력 / 칠러 전력 / 총 전력  |
| Row 4     | PUE 추이         | ✅   | 24시간 PUE 라인차트, 벤치마크 참고표       |
| Row 4     | 상세 데이터      | ✅   | 시간별 전체 데이터 테이블                  |

### 하드코딩된 값 — 교체 필요

| 위치                 | 현재값                 | 교체 조건                  | 교체 대상                      |
| -------------------- | ---------------------- | -------------------------- | ------------------------------ |
| `WUE_BY_MODE[]`      | 성현오빠 md 기준       |
| 시뮬레이션 로직 전체 | `domain.*` 직접 import | Simulation Service 구현 후 | `POST /simulate/24h` REST 호출 |

### 사이드바 파라미터

| 파라미터        | 범위             | 기본값 |
| --------------- | ---------------- | ------ |
| 시나리오        | 여름/봄가을/겨울 | 여름   |
| CPU 서버 대수   | 100 ~ 1000       | 400    |
| GPU 서버 대수   | 0 ~ 200          | 20     |
| 평균 CPU 사용률 | 10 ~ 100%        | 60%    |
| CRAH 공급 온도  | 14 ~ 24°C        | 18°C   |

### 위기 시나리오 버튼

| 버튼     | 적용 변경                 | 대응 전략                                |
| -------- | ------------------------- | ---------------------------------------- |
| 정상     | —                         | —                                        |
| 서버급증 | IT 부하 ×1.3 (+30%)       | 칠러 추가 가동 + 공급 온도 1도 하향      |
| 냉각고장 | 칠러 용량 ×0.5 (1대 탈락) | IT 부하 분산 요청 + Free Cooling 최대화  |
| 폭염     | 외기 온도 38°C 고정       | 칠러 전력 증가 + 공급 온도 최저화 (16°C) |

### 알람 조건

| 레벨  | 조건                       |
| ----- | -------------------------- |
| ERROR | 서버 최고 환기 온도 > 27°C |
| WARN  | 위기 시나리오 주입         |
| INFO  | 정상 모드 복귀             |

---

## 7. 공통 규격

### 오류 응답 형식

```json
{ "detail": "오류 설명 메시지" }
```

### 환경변수 (`.env`)

| 변수                     | 기본값                           | 설명                          |
| ------------------------ | -------------------------------- | ----------------------------- |
| `API_GATEWAY_URL`        | `http://api:8000`                | API Gateway URL (Docker 내부) |
| `CONTROL_SERVICE_URL`    | `http://control-service:8002`    | Control Service URL           |
| `FORECAST_SERVICE_URL`   | `http://forecast-service:8001`   | Forecast Service URL          |
| `SIMULATION_SERVICE_URL` | `http://simulation-service:8003` | Simulation Service URL        |
| `KMA_API_KEY`            | —                                | 기상청 공공데이터포털 API 키  |

### API 응답 시간 기준

| 항목                               | 기준      |
| ---------------------------------- | --------- |
| 응답 시간 (95th percentile)        | 10초 이내 |
| API Gateway 프록시 타임아웃        | 10초      |
| 대시보드 서비스 상태 조회 타임아웃 | 3초       |

---

## 8. 전체 TODO 목록

### 🔧 값 교체 필요 (즉시)

- [ ] `ControlResponse.expected_pue` 고정값 `1.35` → Simulation Service 연동 후 실측값으로 교체
- [ ] `POST /control/rl` 고정 응답값 → Week 4 RL 에이전트 추론값으로 교체
- [ ] `WUE_BY_MODE["chiller"]` = `1.5` → `1.8`로 수정 (`thermodynamic_model.md` 기준)
- [ ] `WUE_BY_MODE["hybrid"]` = `0.8` → 팀 협의 후 값 결정

### 🚧 구현 예정 (우선순위 순)

- [ ] **Simulation Service** `POST /simulate/step`, `POST /simulate/24h` 구현
  - `docker-compose.yml` command를 `test_sinergym.py` → FastAPI 서버 실행으로 교체
- [ ] **Dashboard** `domain.*` 직접 import → `POST /simulate/24h` REST 호출로 교체 (명세서 11절 위반 해소)
- [x] **Forecast Service** `POST /api/v1/forecast` 구현 완료 — Dashboard 연결 완료 (`3_분석_도구.py`)
- [ ] **Control Service** `POST /api/v1/control/mpc` 구현 (명세서 선택 A)
  - API Gateway에 프록시 라우트 `/api/v1/control/mpc` 추가 필요
- [ ] **Control Service** `POST /control/scenario`, `GET /control/status`, `GET /esg/summary` 구현
- [ ] **ControlRequest** 필드 추가: `server_inlet_temp_c`, `server_outlet_temp_c`
- [ ] **ControlResponse** 필드 추가: `chw_flow_setpoint_kg_s`

---
