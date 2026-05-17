# 제어 방식 성능 비교: Rule-based vs PID vs RL (SAC)

> **AI Green IDC — 데이터센터 냉각 최적화 시스템**  
> 평가 환경: 커스텀 IDCEnv (Google Cluster Trace 2019 + 기상청 ASOS, 365일 × 5분 단위)  
> 기준 모델: `sac-wetbulb-1m.zip` (RL Best), `sac-dr-fresh-1m.zip` (RL Safety)  
> **수치 출처 표기**: (측정) = eval 스크립트 실제 실행 결과 / (추정) = 코드·물리 모델 기반 추정

---

## 목차

1. [개요](#1-개요)
2. [시뮬레이션 환경 설정](#2-시뮬레이션-환경-설정)
3. [제어 방식 상세](#3-제어-방식-상세)
4. [정상 운영 성능 비교](#4-정상-운영-성능-비교)
5. [계절별 · 외기 조건별 성능](#5-계절별--외기-조건별-성능)
6. [RL 학습 곡선](#6-rl-학습-곡선)
7. [위기 시나리오 대응 성능](#7-위기-시나리오-대응-성능)
8. [3-Tier 안전 시스템 (RL-Hybrid)](#8-3-tier-안전-시스템-rl-hybrid)
9. [냉각 모드 활용 분포](#9-냉각-모드-활용-분포)
10. [에너지 비용 · ESG 분석](#10-에너지-비용--esg-분석)
11. [Trade-off 분석](#11-trade-off-분석)
12. [구현 파일 참조](#12-구현-파일-참조)

---

## 1. 개요

이 문서는 데이터센터 냉각 최적화를 위해 구현한 세 가지 제어 방식의 성능을 비교 분석한다.

| 제어 방식 | 알고리즘 분류 | 의사결정 근거 | 적응 능력 | 구현 복잡도 |
|-----------|---------------|---------------|-----------|-------------|
| **Rule-based** | 휴리스틱 / 임계값 기반 | 습구온도 구간별 고정 설정값 | ✗ 없음 | 낮음 |
| **PID** | 고전 제어 (피드백) | 오차 비례·적분·미분 제어 | △ 제한적 | 중간 |
| **RL (SAC)** | 심층강화학습 | 데이터 기반 정책 학습 | ✓ 높음 | 높음 |

### 핵심 결과 요약 (실측 데이터 IDCEnv, ep=20)

> 출처: `eval_baseline.py` (ep=20), `eval_crisis.py` (ep=20)

```
평균 PUE (정상 운영):
  Rule-based  : 1.1894  (overhead 0.1894, 기준선)
  PID         : 1.1752  (overhead 0.1752, -7.5%)
  RL (hybrid) : 1.1747  (overhead 0.1747, -7.7%)  ← best와 사실상 동급
  RL (best)   : 1.1743  (overhead 0.1743, -7.9%)  ← cooling overhead 최저

Hybrid 정책 발동 통계 (평상시 normal):
  부하 신호 (cpu>0.55 OR it>170): 11.84% 시간 발동
  온도 신호 (zone>=27.0):            0% 발동 (안전망 미발동)
  → 평상시에도 부하 spike 시 safety 정책 일부 활용

중요 인사이트 — PUE 상한:
  Rule-based baseline(1.189)이 이미 NAVER 각 춘천(1.09) 수준에 근접.
  추가 효율 개선 여지가 제한적 → 본 시스템의 핵심 가치는 효율보다 robustness.

온도 위반 평균 (위기 server_surge):
  Rule-based   : 0.0000°C
  RL-best+fb   : 0.1128°C (zone 26.05°C, 효율 우선) ⚠
  RL-hybrid+fb : 0.0099°C (zone 25.45°C, 91% 감소) ← 최적
```

### 최종 시스템 구성

| 컴포넌트 | 역할 | 모델/로직 |
|----------|------|-----------|
| **효율 정책** | 평상시 PUE 최적화 | `sac-wetbulb-1m` (SAC, 1M timesteps) |
| **안전 정책** | 위기 시 zone 마진 확보 | `sac-dr-fresh-1m` (SAC, 1M timesteps, 위기 시나리오 40% 혼합 학습) |
| **Hybrid 분기** | 두 정책 자동 전환 | `cpu>0.55` OR `it_power>170 kW` OR `zone≥27°C` → safety |
| **Safe Fallback** | 극한 상황 방지 안전망 | `zone>26.5°C` → supply 18°C 강제 |

---

## 2. 시뮬레이션 환경 설정

### 물리 환경 파라미터

| 항목 | 값 | 출처 |
|------|----|------|
| 서버 구성 | CPU 400대 + GPU 20대 (A100×4) | 설계 명세 |
| IT 최대 전력 | 230 kW (CPU 200kW + GPU 30kW) | SPECpower_ssj2008 |
| IT 평균 전력 | 144 kW (cpu_util ~40%) | eval 실측 기반 추정 |
| 유효 열용량 C_eff | 9,009 kJ/K | 열역학 모델 |
| CRAH 공기 유량 | 33 kg/s (3대 정상 운영) | N+1 이중화 설계 |
| 시간 스텝 | 300 s (5분) | Google Cluster Trace 기준 |
| 에피소드 길이 | 288 스텝 (1일) | IDCEnv |
| 서버실 온도 제약 | 18 ~ 27°C (ASHRAE Class A1) | 명세서 요구사항 |
| CRAH 공급 온도 범위 | 18 ~ 25°C (제어 범위) | IDCEnv 행동 공간 |

### 열역학 핵심 수식

```
# 서버실 온도 동역학
T_zone[t+1] = T_zone[t] + (Q_in - Q_out) × Δt / C_eff

# IT 전력 (SPECpower)
P_IT = (CPU 400대 × (200 + 300 × util)) + (GPU 20대 × (300 + 1200 × util))

# 칠러 COP
COP = max(2.0, 6.0 - 0.1 × (T_outdoor - 15) + 0.25 × (T_supply - 20))

# 팬 전력 (Affinity Law, VFD 제어)
P_fan = 18.5 kW × (m_air / 33.0)²

# PUE
PUE = (P_IT + P_chiller + P_fan + P_other) / P_IT
```

### 냉각 모드 전환 기준 (습구온도 기반)

```
wet-bulb < 10°C  → FREE_COOLING  (칠러 OFF, 팬만 구동)
10°C ≤ wb < 18°C → HYBRID       (칠러 부분 가동, 선형 보간)
wb ≥ 18°C        → CHILLER      (기계식 전면 가동)
```

> 습구온도(wet-bulb)를 건구온도 대신 사용하는 이유: 잠열 부하를 반영해야 자연공조 가용 시간을 정확히 예측 가능.  
> 근사식: `T_wb ≈ T_dry - (1 - RH/100) × (T_dry - 4)` (정확도 ±1°C)

---

## 3. 제어 방식 상세

### 3.1 Rule-based 컨트롤러

**구현 파일**: `domain/controllers/rule_based.py`

#### 알고리즘

```python
def decide_cooling_mode(outdoor_temp_c, outdoor_humidity_pct):
    wet_bulb = calculate_wet_bulb_c(outdoor_temp_c, outdoor_humidity_pct)
    if wet_bulb < 10.0:   return CoolingMode.FREE_COOLING
    elif wet_bulb < 18.0: return CoolingMode.HYBRID
    else:                 return CoolingMode.CHILLER

def calculate_setpoint(cooling_mode, outdoor_temp_c):
    if cooling_mode == FREE_COOLING: return 22.0  # 고정값
    elif cooling_mode == HYBRID:     return 20.0  # 고정값
    else:                            return 18.0  # 고정값 (최저 냉방)
```

#### 특성 분석

| 속성 | 내용 |
|------|------|
| 공급 온도 | **고정** (18 / 20 / 22°C 중 1개 선택) |
| IT 부하 반응 | ✗ 부하량 무시 (Chiller 모드는 항상 18°C) |
| 적응 학습 | ✗ 없음 |
| 계산 비용 | 매우 낮음 (O(1)) |
| 설명 가능성 | ✓ 높음 (if-else 로직) |
| 주요 문제점 | Chiller 모드에서 18°C 고정 → 과냉각 → COP 저하 → PUE 상승 |

**COP 손실 예시 (Chiller 모드, 외기 25°C)**:

| 공급 온도 | COP | 칠러 전력 (100 kW 냉각) |
|-----------|-----|-------------------------|
| Rule-based: 18°C | 5.25 | 19.0 kW |
| 최적: 24°C | 6.75 | **14.8 kW** ← RL 학습 목표 |
| 차이 | -1.50 | +4.2 kW (28% 손실) |

> COP 공식: `6.0 - 0.1×(25-15) + 0.25×(T_supply-20) → 18°C: 5.25, 24°C: 6.75`

---

### 3.2 PID 컨트롤러

**구현 파일**: `domain/controllers/pid.py`

#### 알고리즘 (Incremental PID)

```
supply[t] = supply[t-1] + Kp·Δe + Ki·e·Δt + Kd·(Δe/Δt)

여기서:
  e = T_setpoint - T_zone (오차, 현재)
  Δe = e[t] - e[t-1]    (오차 변화량)
  T_setpoint = 24°C      (서버실 목표 온도, 고정)
```

#### 튜닝된 게인값 (IMC 기반 + Sweep 검증)

| 게인 | 값 | 물리적 의미 |
|------|----|-------------|
| **Kp = 1.0** | IMC 공식 도출 | 오차 1°C당 공급 온도 1°C/s 조정 |
| **Ki = 0.001** | Sweep 선택 (IMC: 0.0037) | 정상상태 오차 제거 (Ti = 1000s ≈ 3.7τ) |
| **Kd = 0.5** | S3 복구 수렴 기준 | 칠러 고장 복구 시 과냉각(언더슛) 방지 |

**열적 시정수 도출**:

```
ṁ·cp = 33 kg/s × 1.005 kJ/kg·K = 33.2 kW/K
C_eff = 9,009 kJ/K
τ = C_eff / (ṁ·cp) = 9,009 / 33.2 = 272 s  (약 4.5분)
```

#### Anti-Windup 전략

```python
if supply <= output_min or supply >= output_max:
    delta = kp * delta_e + kd * (delta_e / dt)   # Ki 누적 중단
else:
    delta = kp * delta_e + ki * e * dt + kd * (delta_e / dt)
```

#### 특성 분석

| 속성 | 내용 |
|------|------|
| 공급 온도 | **적응형** (16~27°C, 오차에 따라 연속 조절) |
| 목표 서버실 온도 | 24°C 고정 |
| IT 부하 반응 | △ 간접적 (온도 상승 → 오차 감지 → 냉각 증가) |
| 적응 학습 | △ 제한적 (게인은 고정, 오차에 반응) |
| 설명 가능성 | ✓ 높음 (물리 수식 기반) |
| 주요 문제점 | 게인은 시스템 파라미터 변화에 재튜닝 필요; 예측 불가 |

---

### 3.3 강화학습 (SAC) 컨트롤러

**구현 파일**: `domain/controllers/rl_agent.py`, `domain/controllers/rl_inference.py`

#### 알고리즘: Soft Actor-Critic (SAC)

```
Objective: π* = argmax_π Σ_t [ E[r(s,a)] + α·H(π(·|s)) ]
  - r: 보상 (PUE 최적화 + 온도 위반 패널티)
  - α: 엔트로피 계수 (자동 조정)
  - H(π): 정책 엔트로피 (탐색-활용 균형)
```

#### 관측 공간 (9차원)

| 인덱스 | 변수 | 범위 | 의미 |
|--------|------|------|------|
| 0 | `hour` | [0, 23] | 시간대 (주야 패턴) |
| 1 | `outdoor_temp` | [-15, 45] °C | 외기 건구온도 |
| 2 | `outdoor_trend` | [-15, 15] °C | 최근 1시간 외기 온도 변화 추세 |
| 3 | `humidity` | [20, 95] % | 상대습도 |
| 4 | `cpu_utilization` | [0, 1] | CPU 사용률 |
| 5 | `zone_temp` | [5, 50] °C | 서버실 현재 온도 |
| 6 | `supply_setpoint` | [18, 25] °C | 이전 스텝의 공급 온도 |
| 7 | `it_power_kw` | [0, 400] kW | IT 전력 소비량 |
| 8 | `wet_bulb` | [-15, 35] °C | 습구온도 (냉각 모드 전환 기준) |

#### 행동 공간 (1차원 연속)

```
a = supply_temp_setpoint ∈ [18.0, 25.0] °C
```

#### 보상 함수

```python
if temp_violation > 0.0:
    reward = -temp_violation × 1.5      # 1°C 위반 = -1.5 패널티
else:
    pue_signal = (0.25 - pue_overhead) × 1.5
    reward = -w_energy × pue_overhead + pue_signal
    # w_energy = 0.5 (학습), 0.8 (평가)
```

#### 학습 설정

| 하이퍼파라미터 | 값 |
|----------------|-----|
| 알고리즘 | SAC |
| 총 학습 스텝 | 1,000,000 |
| 에피소드 길이 | 288 스텝 (1일) |
| 에너지 가중치 (w_energy) | 0.5 (학습) / 0.8 (평가) |
| VecNormalize | ✓ 적용 |
| 학습률 | 3×10⁻⁴ |

#### 도메인 랜덤화 (안전 파인튜닝 모델)

| 시나리오 | 비중 | 증폭 내용 |
|----------|------|-----------|
| normal | 40% | 실측 데이터 그대로 |
| server_surge | 20% | CPU 사용률 × 1.3 |
| heat_wave | 20% | 외기 온도 +5~10°C |
| chiller_derate | 20% | 칠러 전력 × 1.5~3.3 |

---

## 4. 정상 운영 성능 비교

> 평가 조건: `eval_baseline.py`, IDCEnv `w_energy=0.8`, **20 에피소드** (각 1일)  
> **(측정)** = `eval_baseline.py` 실제 실행 결과

### 4.1 주요 지표 비교 (측정)

| 지표 | Rule-based | 고정 20°C | PID | RL-best | RL-hybrid |
|------|:----------:|:---------:|:---:|:-------:|:---------:|
| **평균 PUE** | **1.1894** | 1.1847 | **1.1752** | **1.1743** | **1.1747** |
| PUE overhead | 0.1894 | 0.1847 | 0.1752 | 0.1743 | 0.1747 |
| Rule-based overhead 대비 | 기준 | -2.5% | -7.5% | **-7.9%** | **-7.7%** |
| **온도 위반 (°C)** | **0.0000** | 0.0000 | **0.0000** | 0.0000 | 0.0000 |
| **평균 서버실 온도** | 25.01°C | 25.01°C | 25.55°C | 25.76°C | 25.64°C |
| **에피소드 보상** | -17.43 | -14.37 | -8.03 | **-7.72** | -11.00 |

> **핵심 인사이트 4가지**:
> 1. RL이 모든 baseline 능가 — Rule-based 대비 cooling overhead **7.7% 감소**
> 2. Hybrid가 best와 사실상 동급 — PUE 차이 0.0004 (0.04%, 시뮬 noise 수준)
> 3. Hybrid의 평상시 동작: 부하 신호(cpu>0.55 또는 it>170kW) **11.84% 시간 발동** → safety 정책 활용. 이로 인한 미미한 PUE 손해는 위기 사전 감지의 trade-off
> 4. PID가 강력한 baseline — RL-best와 0.5% 격차로 근접

> **PUE 상한 한계**: baseline(1.189)이 이미 세계적 수준에 근접 → 효율 개선 여지 제한.  
> 본 시스템의 핵심 가치는 **효율보다 위기 상황 robustness**.

### 4.2 에너지 비용 비교 (일간, IT 평균 전력 144 kW 기준)

> IT 전력 3,456 kWh/일, 전기요금 120원/kWh

| 방식 | PUE (측정) | 일간 총 전력 | 냉각 전력 | 일간 비용 | Rule-based 대비 |
|------|:----------:|:----------:|:---------:|:--------:|:---------------:|
| Rule-based | 1.1894 | 4,111 kWh | 655 kWh | **493,320원** | 기준 |
| 고정 20°C | 1.1847 | 4,095 kWh | 639 kWh | 491,400원 | -1,920원/일 |
| PID | 1.1752 | 4,062 kWh | 606 kWh | 487,440원 | -5,880원/일 |
| RL-best | 1.1743 | 4,059 kWh | 603 kWh | **487,080원** | **-6,240원/일** |
| RL-hybrid | 1.1747 | 4,060 kWh | 604 kWh | 487,200원 | -6,120원/일 |

> **연간 절감액 (RL-best vs Rule-based)**: 약 **228만 원/년**  
> **RL-best vs RL-hybrid 격차**: 약 **4.4만 원/년** (0.04%) — server_surge 위반 91% 감소의 보험료  
> ※ PUE 차이가 작아 절감액이 이전 추정(2,930만 원)과 크게 다름. 실측값 반영 결과.

---

## 5. 계절별 · 외기 조건별 성능

> **(추정)** — eval 스크립트가 계절별로 분리 측정하지 않음. 물리 모델(COP 수식) + 측정 연간 평균 PUE 기반 역산 추정.

### 5.1 외기 온도별 예상 PUE (추정)

| 외기 온도 | 냉각 모드 | COP | Rule-based PUE | PID PUE | RL PUE |
|-----------|-----------|-----|:--------------:|:-------:|:------:|
| **5°C** | Free Cooling | — | 1.10 | 1.08 | 1.08 |
| **10°C** | Free→Hybrid | 6.5 | 1.13 | 1.11 | 1.10 |
| **15°C** | Hybrid | 6.0 | 1.18 | 1.15 | 1.14 |
| **20°C** | Hybrid→Chiller | 5.5 | 1.22 | 1.19 | 1.18 |
| **25°C** | Chiller | 5.0 | 1.29 | 1.25 | 1.22 |
| **30°C** | Chiller | 4.5 | 1.35 | 1.30 | 1.27 |
| **35°C** | Chiller | 4.0 | 1.41 | 1.36 | 1.32 |
| **38°C** (폭염) | Chiller | 3.7 | 1.44 | 1.39 | 1.35 |

> 연간 평균이 ~1.19인 것은 한국 기후(겨울 Free Cooling 비중 높음)에서 저온 구간이 상당 시간 유지되기 때문.

### 5.2 계절별 성능 요약 (추정)

| 계절 | 평균 외기 | Rule-based PUE | PID PUE | RL PUE |
|------|----------|:--------------:|:-------:|:------:|
| 겨울 (12~2월) | 2°C | 1.10 | 1.09 | 1.08 |
| 봄 (3~5월) | 13°C | 1.19 | 1.16 | 1.14 |
| 여름 (6~8월) | 28°C | 1.37 | 1.31 | 1.27 |
| 가을 (9~11월) | 16°C | 1.21 | 1.18 | 1.15 |
| **연간 평균** | — | **~1.19** | **~1.18** | **~1.17** |

---

## 6. RL 학습 곡선

> 데이터: `data/logs/sac-1m.csv` (SAC, w_energy=0.5, 커스텀 IDCEnv)

### 6.1 에피소드 보상 진행 (학습 로그 기반)

| 학습 단계 | 타임스텝 | ep_rew_mean | 상태 |
|-----------|----------|-------------|------|
| 초기 | 4,096 | -33.9 | 랜덤 정책, 온도 위반 다수 |
| 초기 수렴 | ~50,000 | -22.4 | 온도 위반 감소 시작 |
| 중간 수렴 | ~200,000 | -5.2 | 온도 제약 대부분 충족 |
| 안정화 | ~400,000 | +35.2 | PUE 최적화 집중 |
| 최종 (1M) | 1,000,000 | +38.1 | 수렴 완료 |

> 학습 보상(w_energy=0.5)과 평가 보상(w_energy=0.8)의 스케일이 다름.  
> 평가 기준 ep_rew (측정): Rule-based -17.43 / PID -8.03 / RL-best -7.72

### 6.2 학습 단계별 행동 변화

```
Phase 1 (0~50k): 랜덤 탐색
  공급 온도 무작위 설정 → 잦은 온도 위반 → 큰 패널티

Phase 2 (50k~200k): 안전 학습
  온도 위반 최소화 전략 습득 (낮은 공급 온도 → 안전하지만 비효율)

Phase 3 (200k~500k): 효율 학습
  zone 온도 여유 파악 → 공급 온도를 점진적으로 높임
  Free Cooling 구간 감지 및 활용 증가

Phase 4 (500k~1M): 정책 정제
  시간대별·계절별 패턴 학습 완성
  outdoor_trend 활용한 사전 대응
```

---

## 7. 위기 시나리오 대응 성능

> 평가: `eval_crisis.py`, IDCEnv `force_scenario` 사용, **20 에피소드**  
> **(측정)** = 실제 eval_crisis.py 실행 결과 / PID는 eval_crisis.py 미포함 (별도 시뮬레이터 결과)

### 7.1 시나리오 정의

| 코드 | 시나리오 | 증폭 내용 |
|------|----------|-----------|
| **normal** | 정상 운영 | 실측 데이터 |
| **server_surge** | 서버 급증 | CPU 사용률 ×1.3 → IT 전력 상승 |
| **heat_wave** | 폭염 | 외기 온도 +5~10°C |
| **chiller_derate** | 칠러 효율 저하 | 칠러 전력 ×1.5~3.3 (COP 30~67% 저하) |

### 7.2 평균 PUE (시나리오별, 측정)

| 시나리오 | Rule-based | RL-best+fb | RL-hybrid+fb |
|----------|:----------:|:----------:|:------------:|
| **normal** | 1.1894 | **1.1743** | 1.1747 |
| **server_surge** | 1.1906 | **1.1734** | 1.1754 |
| **heat_wave** | 1.2292 | **1.2171** | 1.2175 |
| **chiller_derate** | 1.2499 | **1.2375** | 1.2379 |

> RL이 모든 시나리오에서 Rule-based보다 효율적 (PUE 1.0~1.5%p 개선).

### 7.3 온도 위반 (°C, 시나리오별, 측정)

| 시나리오 | Rule-based | RL-best+fb | RL-hybrid+fb | 위반 감소율 |
|----------|:----------:|:----------:|:------------:|:-----------:|
| **normal** | 0.0000 | 0.0000 | 0.0000 | — |
| **server_surge** | 0.0000 | **0.1128 ⚠** | **0.0099 ✓** | **91.2%** |
| **heat_wave** | 0.0000 | 0.0000 | 0.0000 | — |
| **chiller_derate** | 0.0000 | 0.0000 | 0.0000 | — |

> **server_surge 분석**:
> - IT 부하(~300 kW) > 칠러 최대 용량(250 kW) → 물리적 한계로 zone 26.05°C까지 상승
> - RL-best+fb: fallback(zone>26.5°C) 발동 전에도 칠러 max 상태 → supply 18°C 효과 제한 → 위반 발생
> - RL-hybrid+fb: cpu 신호 사전 감지 → safety 모델 조기 전환 → zone 25.45°C 마진 확보 → 91.2% 감소
> - 모든 위반은 ASHRAE A3 클래스 내 (server_surge 0.0099°C → zone 27.01°C 순간 도달)

### 7.4 fallback만으로는 server_surge를 못 막는 이유

```
[fallback 단독 한계]
  server_surge → IT 300kW → 칠러 이미 max 가동
  → zone 26.5°C 도달 시 fallback이 supply 18°C로 낮춤
  → 칠러가 이미 최대 출력 → 추가 냉각 불가
  → fallback 발동 시점에 이미 위반 임박

[Hybrid의 해결책]
  cpu_util > 0.55 감지 (zone 상승 전, 사전 신호)
  → 즉시 safety 모델로 전환 → 보수적 supply 온도
  → zone이 26.5°C에 도달하지 않으므로 fallback 불필요
```

### 7.5 PID 위기 시나리오 검증 결과 (별도 시뮬레이션, 1600s)

> 출처: `docs/pid_tuning.md` — `apps/simulation_service/pid_simulator.py` 별도 실행 결과  
> ※ eval_crisis.py에 PID 미포함; 아래는 PID 튜닝 검증 시뮬레이션 결과

| 시나리오 | 내용 | 18~27°C 유지율 | 최대 서버실 온도 |
|----------|------|:--------------:|:----------------:|
| **S1: 부하 급증** | IT 144→187 kW (+30%), t=300s | **100.0%** | 25.8°C |
| **S2: 폭염** | 외기 22→42°C, t=300s | **100.0%** | 26.3°C |
| **S3: 칠러 고장** | 칠러 OFF 160s → 예비 투입 | **100.0%** | 26.7°C |

> S3 물리적 근거: C_eff=9,009 kJ/K, 144 kW → 27°C 도달까지 `(27-24)×9009/144 = 188s` 필요  
> → 160s 공백 구간 내 27°C 미도달, 예비 칠러 투입 후 정상 복구.

---

## 8. 3-Tier 안전 시스템 (RL-Hybrid)

> 구현: `domain/controllers/rl_inference.py`, `scripts/eval_crisis.py`

### 8.1 시스템 구조

```
관측(obs) 입력
    │
    ├─ [Tier 1: 효율 우선] RL-best (sac-wetbulb-1m)
    │     조건: cpu_util ≤ 0.55 AND it_power ≤ 170 kW AND zone_temp < 27°C
    │     "지금이 가장 효율적인 상태"
    │
    ├─ [Tier 2: 안전 우선] RL-finetune (sac-dr-fresh-1m)
    │     조건: cpu_util > 0.55 OR it_power > 170 kW OR zone_temp ≥ 27°C
    │     "위기 감지, 안전 우선으로 전환" (DR 학습으로 강건성 확보)
    │
    └─ [Tier 3: 강제 냉방] T_SUPPLY_MIN = 18°C
          조건: zone_temp > 26.5°C (Tier 2와 독립, 최후 안전망)
          "마지막 안전망"
```

### 8.2 전환 임계값 근거

| 신호 | 임계값 | 유형 | 근거 |
|------|--------|------|------|
| `cpu_util > 0.55` | 사전 신호 | 부하 | normal 최대 ~0.52, surge 평균 구간 위 |
| `it_power > 170 kW` | 사전 신호 | 부하 | normal 최대 ~161 kW, surge 최대 183 kW 사이 |
| `zone_temp ≥ 27°C` | 사후 신호 | 온도 | ASHRAE 한계 (실질적으로 fallback 26.5°C가 먼저 발동) |
| `zone_temp > 26.5°C` | 사후 신호 | 온도 | Tier 3 fallback — 27°C 한계 0.5°C 전 강제 개입 |

> 핵심: cpu/it_power 사전 신호로 조기 감지해야 Safety 모델이 마진 확보 가능.  
> 실증: server_surge에서 hybrid가 위반 91.2% 감소 (0.1128°C → 0.0099°C).

### 8.3 설명 가능성

Hybrid 전환 신호(cpu/it_power/zone)는 모두 **해석 가능**:
- 운영자가 "지금 어떤 정책이 작동 중인지" 실시간 확인 가능
- 대시보드 연계 시 의사결정 지원 도구로 발전 가능
- DeepMind Google DC, Microsoft Azure 등 유사 다층 supervisor 구조와 일치

### 8.4 Hybrid 성능 요약 (측정)

| 지표 | RL-best+fb | RL-hybrid+fb |
|------|:---------:|:------------:|
| 정상 PUE | 1.1743 | 1.1747 (+0.0004) |
| server_surge 위반 (°C) | 0.1128 ⚠ | **0.0099 ✓** |
| server_surge 위반 감소율 | 기준 | **91.2%** |
| heat_wave PUE | 1.2171 | 1.2175 (+0.0004) |
| chiller_derate PUE | 1.2375 | 1.2379 (+0.0004) |

> **결론**: Hybrid가 정상 best 대비 PUE 0.0004 손해 (0.04%, 시뮬 noise 수준). 이 손해는 평상시 11.84% 시간 safety 정책 발동(CPU 피크)에서 비롯되며, trade-off로 server_surge 위반 91% 감소 확보 — **"공짜 안전"은 아니지만 매우 저렴한 보험**.

---

## 9. 냉각 모드 활용 분포

> **(추정)** — eval 스크립트는 냉각 모드 비중을 직접 출력하지 않음.  
> 측정된 연간 PUE(~1.189)와 물리 모델을 역산하여 추정.

### 9.1 연간 냉각 모드 비중 (%, 추정)

| 냉각 모드 | Rule-based | PID | RL (SAC) |
|-----------|:----------:|:---:|:--------:|
| Free Cooling | ~45% | ~48% | ~55% |
| Hybrid | ~30% | ~28% | ~27% |
| Chiller Only | ~25% | ~24% | ~18% |

> RL이 Free Cooling 비중을 높이는 방식:
> - `outdoor_trend`로 외기 하강 추세를 사전 감지
> - zone 온도 여유가 있을 때 공급 온도를 높여 Free Cooling 조건 충족 유도

---

## 10. 에너지 비용 · ESG 분석

> 측정된 PUE 기반 계산. IT 평균 전력 144 kW, 전기요금 120원/kWh, 탄소계수 0.459 tCO2/MWh.

### 10.1 일간 전력 소비 비교 (측정 PUE 기반)

| 항목 | Rule-based | PID | RL-best |
|------|:----------:|:---:|:-------:|
| PUE (측정) | 1.1894 | 1.1752 | 1.1747 |
| IT 전력 | 3,456 kWh | 3,456 kWh | 3,456 kWh |
| 냉각+기타 전력 | 655 kWh | 606 kWh | 604 kWh |
| **총 전력** | **4,111 kWh** | **4,062 kWh** | **4,060 kWh** |
| **전기요금** | 493,320원 | 487,440원 | **487,200원** |
| Rule-based 대비 절감 | 기준 | -5,880원/일 | **-6,120원/일** |

### 10.2 연간 탄소 배출량 (측정 PUE 기반)

| 방식 | 연간 총 전력 | 탄소 배출량 | Rule-based 대비 감축 |
|------|:----------:|:----------:|:-------------------:|
| Rule-based | 1,500,515 kWh | **688.7 tCO2** | 기준 |
| PID | 1,482,630 kWh | 680.5 tCO2 | -8.2 tCO2 (-1.2%) |
| RL-best | 1,481,900 kWh | **680.1 tCO2** | **-8.6 tCO2 (-1.2%)** |

> 절감량이 작은 이유: baseline PUE(1.189)가 이미 세계 수준에 근접 → 추가 절감 여지 제한.  
> 연간 절감액: 6,120원/일 × 365 ≈ **223만 원/년**

### 10.3 WUE (Water Usage Effectiveness) 비교 (추정)

| 방식 | Free Cooling 비중 (추정) | WUE (L/kWh, 추정) |
|------|:--------------------:|:-----------------:|
| Rule-based | ~45% | ~0.99 |
| PID | ~48% | ~0.94 |
| RL-best | ~55% | ~0.85 |

---

## 11. Trade-off 분석

### 11.1 방식별 특성 매트릭스

| 평가 항목 | Rule-based | PID | RL-best | RL-hybrid+fb |
|-----------|:----------:|:---:|:-------:|:------------:|
| **에너지 효율 (PUE)** | ★★★☆☆ | ★★★★☆ | ★★★★★ | ★★★★★ |
| **정상 온도 안전성** | ★★★★★ | ★★★★★ | ★★★★★ | ★★★★★ |
| **위기 안전성 (server_surge)** | ★★★★★ | ★★★★★ | ★★★☆☆ | ★★★★★ |
| **설명 가능성** | ★★★★★ | ★★★★☆ | ★☆☆☆☆ | ★★★☆☆ |
| **구현 복잡도** | ★☆☆☆☆ | ★★☆☆☆ | ★★★★☆ | ★★★★★ |
| **위기 사전 감지** | ✗ | △ 간접 | △ obs 기반 | ✓ 다중 신호 |
| **학습 데이터 필요** | 없음 | 없음 | 1년치 시뮬 | 1년치 × 2모델 |

### 11.2 적용 시나리오별 권장 방식

| 상황 | 권장 방식 | 이유 |
|------|-----------|------|
| 빠른 프로토타이핑 | Rule-based | 즉시 가동, 설명 용이 |
| 안전 규정 + 설명 가능성 | **PID** | 100% 온도 유지율, 물리 기반 해석 |
| 에너지 비용 최소화 | **RL-best** | overhead 7.9% 개선, 연 228만원 절감 |
| 효율 + 위기 안전성 균형 | **RL-hybrid+fb** | server_surge 위반 91.2% 감소, PUE 0.04% 손해 (매우 저렴한 trade-off) |
| **실제 운영 권장** | **RL-hybrid+fb** | 효율·안전·설명가능성(전환 신호) 모두 확보 |

### 11.3 PUE vs 안전성 Trade-off

```
[Rule-based]   Chiller 18°C 고정 → PUE overhead 0.1894, 위기 안전
[PID]          zone 24°C 유지 → overhead 0.1752, 위기 100% 안전 (별도 시뮬)
[RL-best]      zone 여유 최대 활용 → overhead 0.1743, server_surge 0.1128°C 위반
[RL-hybrid]    사전 신호 조기 전환 → overhead 0.1747, server_surge 0.0099°C

효율 (PUE):  Best > Hybrid > PID > Rule-based
안전 (위반): Hybrid > Best ≈ Rule-based (server_surge에서)
→ 효율 최강은 Best (overhead 0.0004 차이), 안전 강한 운영 원하면 Hybrid
→ 평상시 11.84% 시간 safety 정책 발동(CPU 피크)이 미미한 PUE 손해의 원인
→ 연 4.4만 원 수준 PUE 비용으로 잠재 수억원 server_surge 리스크 헤지
```

---

## 12. 구현 파일 참조

| 모듈 | 파일 | 내용 |
|------|------|------|
| Rule-based | `domain/controllers/rule_based.py` | 습구온도 기반 냉각 모드 + 고정 설정값 |
| PID | `domain/controllers/pid.py` | Incremental PID, Anti-windup |
| RL 환경 | `domain/controllers/idc_env.py` | 커스텀 IDC Gymnasium 환경 |
| RL 학습 | `domain/controllers/rl_agent.py` | SAC/PPO 학습 파이프라인 |
| RL 추론 | `domain/controllers/rl_inference.py` | 3-Tier 안전 시스템, Hybrid 전환 |
| 평가 (기본) | `scripts/eval_baseline.py` | Rule/PID/RL 통합 비교 |
| 평가 (위기) | `scripts/eval_crisis.py` | 4개 위기 시나리오 비교 |
| PID 튜닝 | `docs/pid_tuning.md` | IMC 기반 게인 도출 과정 |
| 학습 로그 | `data/logs/sac-1m.csv` | 1M 스텝 에피소드 보상 기록 |
| 학습 모델 (효율) | `data/models/sac-wetbulb-1m.zip` | RL Best |
| 학습 모델 (안전) | `data/models/sac-dr-fresh-1m.zip` | RL Safety (DR 학습) |

### 실행 방법

```bash
# 기본 성능 비교 (Rule-based / PID / RL)
PYTHONPATH=. python scripts/eval_baseline.py \
    --model data/models/sac-wetbulb-1m.zip --episodes 20

# 위기 시나리오 비교
PYTHONPATH=. python scripts/eval_crisis.py \
    --best data/models/sac-wetbulb-1m.zip \
    --finetune data/models/sac-dr-fresh-1m.zip \
    --episodes 20
```

---

## 부록: 성능 지표 정의

| 지표 | 수식 | 설명 |
|------|------|------|
| **PUE** | 총전력 / IT전력 | 1.0 이상, 낮을수록 효율적. NAVER 각 춘천 1.09 = 세계 최고 |
| **PUE overhead** | PUE - 1.0 | 냉각·기타에 사용된 오버헤드 비율 |
| **온도 위반** | max(0, T_zone - 27) + max(0, 18 - T_zone) | °C 단위, 0이 이상적 |
| **COP** | 제거 열량 / 칠러 전력 | 높을수록 효율, 외기 온도에 반비례 |
| **WUE** | 물 사용량(L) / IT전력(kWh) | 낮을수록 물 절약 (추정값) |

---

## 부록: 발표 핵심 메시지

> "Hybrid 정책은 평상시 best와 사실상 동급 PUE (0.04% 손해) +  
> 위기 시나리오 위반 91% 감소. **매우 저렴한 trade-off로 다층 안전 시스템 완성**."

---

*작성: AI Green IDC Team*  
*수치 출처: `eval_baseline.py` ep=20 (측정), `eval_crisis.py` ep=20 (측정), 그 외 물리 모델 기반 추정*  
*참조 규격: ASHRAE TC 9.9, NAVER SASB 보고서, Google Cluster Trace 2019, 기상청 ASOS*
