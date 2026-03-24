### 파일 구조

```
domain/thermodynamics/
├── it_power.py        ← 서버 전력 (SPECpower 공식)
├── cooling_load.py    ← 냉각 부하 Q 계산
├── chiller.py         ← 칠러 COP 및 전력, 냉각 모드 결정
├── free_cooling.py    ← 자연공조 효율 및 팬 전력
└── pue.py             ← PUE 계산 및 벤치마크 비교
```

### 계산 흐름

```
[cpu_utilization]
      │
      ▼
 it_power.py
 calculate_total_it_power_kw()
      │ it_power_kw
      ▼
 cooling_load.py
 calculate_cooling_load_from_it_power_kw()
      │ cooling_load_kw  (≈ it_power_kw, overhead_factor=1.0)
      ├──────────────────────────────────┐
      ▼                                  ▼
 chiller.py                        free_cooling.py
 calculate_chiller_power_kw()      calculate_free_cooling()
 (냉각 모드 자동 결정)               (참고용 효율 계산)
      │ chiller_power_kw
      ▼
 pue.py
 calculate_pue()
      │
      ▼
 PUEResult.pue
```

---

### `it_power.py` — 서버 전력 모델

**역할**: CPU 사용률 → 서버 전력 소비량 (SPECpower_ssj2008 공식)

```python
from domain.thermodynamics.it_power import (
    ServerSpec, ServerType,
    calculate_server_power_w,
    calculate_total_it_power_kw,
)
```

#### 주요 함수

**`calculate_server_power_w(cpu_utilization, server_type, custom_spec)`**

| 인자 | 타입 | 설명 |
|------|------|------|
| `cpu_utilization` | `float` | CPU 사용률 (0.0 ~ 1.0) |
| `server_type` | `ServerType` | `CPU` 또는 `GPU` |
| `custom_spec` | `ServerSpec \| None` | 커스텀 사양 (None이면 기본값) |

반환: 서버 1대의 전력 (W)

**`calculate_total_it_power_kw(cpu_utilization, num_cpu_servers, num_gpu_servers, ...)`**

| 인자 | 타입 | 설명 |
|------|------|------|
| `cpu_utilization` | `float` | 평균 CPU 사용률 |
| `num_cpu_servers` | `int` | CPU 서버 대수 **(필수, 기본값 없음)** |
| `num_gpu_servers` | `int` | GPU 서버 대수 **(필수, 기본값 없음)** |

반환: 전체 IT 전력 (kW)

#### 기본값 (SPECpower_ssj2008)

```python
CPU_SERVER = ServerSpec(p_idle_w=200.0, p_max_w=500.0)   # Intel Xeon
GPU_SERVER = ServerSpec(p_idle_w=300.0, p_max_w=1500.0)  # NVIDIA A100 × 4
```

#### Sinergym 검증 시 IT 전력 계산

`sinergym_validator.py`에서는 `calculate_total_it_power_kw()` 대신 EnergyPlus ITE 공식을 직접 사용한다.

```python
# IDF 설계 전력: East(259.08 m² × 200 W/m²) + West(232.26 m² × 200 W/m²) = 98,268 W
ITE_P_DESIGN_KW = 98.268
it_power_kw = max(0.0, ITE_P_DESIGN_KW * (-1.0 + cpu_fraction + 0.0667 * t_inlet))
# t_inlet: CRAH 공급 온도 obs (동서 구역 평균)
```

> **이유**: SPECpower는 온도항이 없어 Sinergym 실측 대비 IT 전력을 과소 추정한다.
> 검증 맥락에서만 EnergyPlus ITE 공식을 적용하고, `it_power.py`의 SPECpower는 그대로 유지한다.

---

### `cooling_load.py` — 냉각 부하 계산

**역할**: IT 전력 또는 공기 유량으로 냉각 부하(Q) 계산

```python
from domain.thermodynamics.cooling_load import (
    calculate_cooling_load_from_it_power_kw,
    calculate_cooling_load_from_airflow_kw,
    calculate_m_air_for_servers,
    M_AIR_DESIGN_KG_S, NUM_SERVERS_DESIGN, T_SUPPLY_DESIGN_C, T_RETURN_DESIGN_C,
)
```

#### 주요 함수

**`calculate_cooling_load_from_it_power_kw(it_power_kw, overhead_factor=1.0)`**

에너지 보존 법칙 기반: 서버에 공급한 전력은 모두 열로 변환됩니다.

```python
Q_cooling = it_power_kw * overhead_factor
# overhead_factor=1.0 → Q = IT 전력과 동일
# overhead_factor=1.05 → UPS 손실 5% 추가 반영 가능
```

**`calculate_cooling_load_from_airflow_kw(m_dot, supply_temp_c, return_temp_c, c_p=1.005)`**

공기 유량과 온도 차이 기반 (Q = ṁ × c_p × ΔT):

```python
Q = m_dot * c_p * (return_temp_c - supply_temp_c)
```

> `return_temp_c < supply_temp_c`이면 `ValueError`를 발생시킵니다.

**`calculate_m_air_for_servers(num_servers)`**

명세서 설계 기준(500대 → 50 kg/s)을 실제 서버 수에 선형 스케일:

```python
m_air = M_AIR_DESIGN_KG_S * (num_servers / NUM_SERVERS_DESIGN)
# 예: 197대 → 50 × (197/500) = 19.7 kg/s
```

#### 설계 상수 (명세서 SyntheticIDCBuilder, p.15-16)

```python
M_AIR_DESIGN_KG_S = 50.0   # 설계 공기 유량 (500대 기준, kg/s)
NUM_SERVERS_DESIGN = 500    # 설계 기준 서버 수
T_SUPPLY_DESIGN_C  = 18.0  # CRAH 공급 온도 설계값 (°C)
T_RETURN_DESIGN_C  = 27.0  # 환기 온도 설계값 (°C)
```

#### Sinergym 검증 시 Q 계산 방식

```python
# 설계 m_air (고정) + Sinergym 관측 ΔT (가변) → Q가 실제 열 상태에 따라 변동
m_air_design = calculate_m_air_for_servers(197)  # → 19.7 kg/s
Q_model = calculate_cooling_load_from_airflow_kw(m_air_design, avg_supply_obs, avg_return_obs)
```

---

### `chiller.py` — 칠러 전력 및 냉각 모드

**역할**: 외기 온도에 따라 냉각 모드를 결정하고 칠러 전력을 계산

```python
from domain.thermodynamics.chiller import (
    CoolingMode, ChillerResult,
    calculate_cop,
    calculate_chiller_power_kw,
)
```

#### 냉각 모드 결정 기준

| 외기 온도 | 냉각 모드 | 칠러 가동 |
|-----------|-----------|-----------|
| T < 15°C | `FREE_COOLING` | 미가동 (팬 3.5%만) |
| 15°C ≤ T < 22°C | `HYBRID` | 부분 가동 |
| T ≥ 22°C | `CHILLER` | 전면 가동 |

#### 주요 함수

**`calculate_cop(outdoor_temp_c)`**

```python
COP = max(2.0, 6.0 - 0.1 * (outdoor_temp_c - 15))
```

**`calculate_chiller_power_kw(cooling_load_kw, outdoor_temp_c)`**

반환: `ChillerResult(cop, chiller_power_kw, cooling_mode)`

```python
# FREE_COOLING 모드: 부하 비례 팬 전력 (최대 4.49 kW에서 포화)
chiller_power_kw = min(4.49, cooling_load_kw * 0.044)

# HYBRID 모드: 온도에 따라 칠러 비중 선형 보간
chiller_fraction = (T - 15) / (22 - 15)
chiller_power_kw = (Q * chiller_fraction) / COP + Q * (1 - chiller_fraction) * 0.035

# CHILLER 모드: 칠러 전면 가동
chiller_power_kw = cooling_load_kw / COP
```

> **Sinergym과의 차이**: Sinergym의 CRAH 팬은 일정 속도로 동작하여 고정 전력(~4.49 kW)을
> 소비하지만, 우리 모델은 냉각 부하에 비례하는 변속 팬을 가정합니다.

---

### `free_cooling.py` — 자연공조 효율

**역할**: 외기 조건에 따른 자연공조 가능 여부 및 팬 전력 계산

```python
from domain.thermodynamics.free_cooling import (
    FreeCoolingResult,
    calculate_free_cooling_efficiency,
    calculate_free_cooling,
)
```

#### 주요 함수

**`calculate_free_cooling_efficiency(outdoor_temp_c, outdoor_humidity_pct=50.0, supply_temp_setpoint_c=18.0)`**

반환: 자연공조 효율 (0.0 ~ 1.0)

**`calculate_free_cooling(cooling_load_kw, outdoor_temp_c, ...)`**

반환: `FreeCoolingResult(is_available, efficiency, fan_power_kw, effective_cooling_kw, mode_description)`

```python
fan_ratio = FAN_POWER_RATIO_FREE * efficiency + FAN_POWER_RATIO_CHILLER * (1 - efficiency)
# FAN_POWER_RATIO_FREE    = 0.035  (3.5%, Sinergym 실측 보정)
# FAN_POWER_RATIO_CHILLER = 0.08   (8%)
```

> `free_cooling.py`는 독립 분석용입니다. 실제 PUE 계산 파이프라인에서는
> `chiller.py`의 `calculate_chiller_power_kw()`가 자연공조 모드를 내부적으로 처리합니다.

---

### `pue.py` — PUE 계산

**역할**: 총 전력 / IT 전력으로 PUE 계산, 벤치마크 비교

```python
from domain.thermodynamics.pue import (
    PUEResult, PUE_BENCHMARK,
    calculate_pue,
)
```

#### 주요 함수

**`calculate_pue(it_power_kw, cooling_power_kw, other_power_kw=None)`**

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `it_power_kw` | 필수 | IT 장비 전력 |
| `cooling_power_kw` | 필수 | 냉각 전력 (칠러 + 팬) |
| `other_power_kw` | `None` | None이면 `it_power × 0.05`로 자동 계산 |

반환: `PUEResult(pue, total_power_kw, it_power_kw, cooling_power_kw, other_power_kw, efficiency_vs_benchmark)`

#### 벤치마크 상수

```python
PUE_BENCHMARK = {
    "naver_chuncheon": 1.09,   # 목표 기준 (세계 최고 수준)
    "google_global":   1.10,
    "global_average":  1.58,
    "korea_private":   2.03,
    "korea_public":    3.13,
    "green_dc_standard": 1.66,
}
```

---

### 전체 사용 예시

```python
from domain.thermodynamics.it_power import calculate_total_it_power_kw
from domain.thermodynamics.cooling_load import calculate_cooling_load_from_it_power_kw
from domain.thermodynamics.chiller import calculate_chiller_power_kw
from domain.thermodynamics.pue import calculate_pue, PUE_BENCHMARK

# 입력값
cpu_utilization  = 0.75   # CPU 사용률 75%
num_cpu_servers  = 500    # CPU 서버 500대
num_gpu_servers  = 0
outdoor_temp_c   = 10.0   # 외기 온도 10°C

# 1. IT 전력
it_power_kw = calculate_total_it_power_kw(
    cpu_utilization, num_cpu_servers, num_gpu_servers
)
# → 83.7 kW (197대 기준) 또는 실제 서버 수에 맞게

# 2. 냉각 부하
cooling_load_kw = calculate_cooling_load_from_it_power_kw(it_power_kw)
# → it_power_kw × 1.0 = 냉각 부하

# 3. 칠러 (10°C → FREE_COOLING 모드)
chiller = calculate_chiller_power_kw(cooling_load_kw, outdoor_temp_c)
# chiller.cooling_mode → CoolingMode.FREE_COOLING
# chiller.chiller_power_kw → cooling_load_kw × 0.035 (팬 전력만)
# chiller.cop → 6.5

# 4. PUE
result = calculate_pue(it_power_kw, chiller.chiller_power_kw)
# result.pue → (IT + 팬 + 기타5%) / IT ≈ 1.09

print(f"PUE: {result.pue:.3f}")
print(f"NAVER 각 춘천 대비: ×{result.efficiency_vs_benchmark:.2f}")
```

---

### Sinergym 검증 (`apps/simulation_service/sinergym_validator.py`)

검증 스크립트는 위 함수들을 사용해 Sinergym 시뮬레이션 결과와 비교합니다.

```
Sinergym obs 추출
  ├─ outdoor_temp_c  (Site:OutdoorAirDrybulbTemperature)
  ├─ cpu_fraction    (Data Center CPU Loading Schedule)
  ├─ east/west flow  (Zone Mechanical Ventilation Mass Flow Rate)
  └─ east/west temps (Zone Supply/Return Air Temperature)
         │
         ├──[우리 모델]──────────────────────────────────────────
         │  it_power = calculate_total_it_power_kw(cpu_fraction, 197, 0)
         │  Q_model  = calculate_cooling_load_from_it_power_kw(it_power)
         │  chiller  = calculate_chiller_power_kw(Q_model, outdoor_temp_c)
         │  pue      = calculate_pue(it_power, chiller.chiller_power_kw)
         │
         └──[Sinergym 기준값]──────────────────────────────────
            Q_real   = Σ (flow × 1.005 × ΔT)  ← 공기 유량으로 측정
            P_real   = hvac_power - basin_heater_power
            PUE_real = (Q_real + hvac_power) / Q_real

오차율 = |모델 - 실측| / 실측 × 100%
목표: 냉각 부하 < 10%, 칠러 전력 < 10%, PUE < 10%
```

#### 모델 보정 (Sinergym 실측 기반)

| 항목 | 보정 방식 | 근거 |
|------|-----------|------|
| 냉각 부하 (Q) | `overhead_factor=1.22` — IT 전력의 22% 비IT 발열 포함 | Sinergym Q실측 / SPECpower 비율 1.15~1.29 → 중간값 |
| 칠러 전력 (P) | `min(4.49, Q × 0.044)` — 부하 비례 팬, 최대 4.49 kW 포화 | Q≈102 kW(cpu=0.75)에서 Sinergym 팬 4.49 kW 포화 실측 |
| PUE | Q·P 보정 후 오차 ~3% | — |

---

*작성: AI Green IDC Team | 참조: SPECpower_ssj2008, ASHRAE TC 9.9, Sinergym (EnergyPlus 24.1.0)*
