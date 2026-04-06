"""
서버실 열역학 수치 시뮬레이터

ODE 기반으로 서버실 온도를 시간에 따라 적분한다.

핵심 수식:
  dT/dt = (Q_in - Q_out) / C_eff
  T_zone[t+1] = T_zone[t] + dT/dt × Δt

  - Q_in   : IT 발열량 (kW = kJ/s)
  - Q_out  : 냉각량 (kW = kJ/s), 기존 열역학 모델에서 계산
  - C_eff  : 유효 열용량 (kJ/K) = 공기 + 서버 장비 + 랙 + 구조물
  - Δt     : 시간 스텝 (s)

유효 열용량 구성 (CPU 400대 + GPU 20대, 중규모 IDC 1홀):
  - 공기:        1.2 kg/m³ × 1,500 m³ × 1.005 kJ/kg·K ≈ 1,809 kJ/K
  - CPU 서버:    400대 × 20 kg × 0.5 kJ/kg·K           = 4,000 kJ/K
  - GPU 서버:     20대 × 60 kg × 0.5 kJ/kg·K           =   600 kJ/K  (A100×4, 약 60kg)
  - 랙:          12개 × 100 kg × 0.5 kJ/kg·K           =   600 kJ/K  (CPU 10 + GPU 2)
  - 구조물:      바닥·벽 (얇게 반영)                    = 2,000 kJ/K
  합계: 약 9,009 kJ/K  (열적 시정수 τ = C_eff / (ṁ·cp) ≈ 272초, ṁ=33 kg/s 기준)
"""

import sys
import os

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from dataclasses import dataclass

from domain.thermodynamics.cooling_load import AIR_SPECIFIC_HEAT_KJ_PER_KG_K


# 서버실 기본 설계 파라미터 (CPU 400대 + GPU 20대, 중규모 IDC 1홀 500m²×3m)
DEFAULT_ROOM_VOLUME_M3 = 1500.0      # 서버실 부피 (m³), 500m² × 천장 3m
DEFAULT_AIR_DENSITY_KG_M3 = 1.2     # 공기 밀도 (kg/m³), 20°C 1atm 기준
DEFAULT_T_INITIAL_C = 24.0          # 초기 서버실 온도 (°C) — 정상 운영 평형 온도 가정
DEFAULT_DT_S = 1.0                  # 시간 스텝 (초)

# CRAH 설계 파라미터 (N+1 이중화: 3대 운용 + 1대 예비)
# CRAH 1대당 냉각 용량 100kW (열), ΔT=9°C 기준:
#   ṁ_per = 100 / (1.005 × 9) ≈ 11.1 kg/s
DEFAULT_N_CRAH_UNITS = 4           # 설치 CRAH 총 대수 (예비 1대 포함)
DEFAULT_N_CRAH_ACTIVE = 3          # 정상 운영 CRAH 대수 (예비 1대 제외)
DEFAULT_M_DOT_KG_PER_S = 44.0     # 전체 공기 유량 (kg/s), 4대 전부 가동 시

# 칠러 설계 파라미터 (2N 이중화: 1대 운용 + 1대 예비, CRAH 대수와 독립)
# 칠러 1대 = 300kW 열 / COP(22°C=5.3) ≈ 56.6kW 전기
DEFAULT_CHILLER_DESIGN_KW = 56.6  # 칠러 1대 전기 용량 (kW)

# 유효 열용량 구성 (kJ/K)
_C_AIR     = DEFAULT_AIR_DENSITY_KG_M3 * DEFAULT_ROOM_VOLUME_M3 * AIR_SPECIFIC_HEAT_KJ_PER_KG_K
_C_SERVERS = 400 * 20 * 0.5 + 20 * 60 * 0.5  # CPU 400대×20kg + GPU 20대×60kg, 0.5 kJ/kg·K
_C_RACKS   = 12 * 100 * 0.5                   # 랙 12개×100kg×0.5 kJ/kg·K (CPU 10 + GPU 2)
_C_STRUCTURE = 2000.0                          # 바닥·벽 구조물 (1500m³ 기준)
DEFAULT_C_EFF_KJ_PER_K = _C_AIR + _C_SERVERS + _C_RACKS + _C_STRUCTURE  # ≈ 9,009 kJ/K


@dataclass
class ThermalSimulatorConfig:
    """시뮬레이터 설정 파라미터"""

    room_volume_m3: float = DEFAULT_ROOM_VOLUME_M3
    air_density_kg_m3: float = DEFAULT_AIR_DENSITY_KG_M3
    c_p_kj_per_kg_k: float = AIR_SPECIFIC_HEAT_KJ_PER_KG_K
    t_initial_c: float = DEFAULT_T_INITIAL_C
    dt_s: float = DEFAULT_DT_S
    c_eff_kj_per_k: float = DEFAULT_C_EFF_KJ_PER_K
    n_crah_units: int = DEFAULT_N_CRAH_UNITS          # 설계 CRAH 총 대수
    m_dot_kg_per_s: float = DEFAULT_M_DOT_KG_PER_S   # 전체 공기 유량 (전 CRAH 가동 시, kg/s)
    chiller_design_kw: float = DEFAULT_CHILLER_DESIGN_KW  # 전체 칠러 설계 전력 (전 CRAH 가동 시, kW)

    @property
    def m_dot_per_crah(self) -> float:
        """CRAH 1대당 공기 유량 (kg/s)"""
        return self.m_dot_kg_per_s / self.n_crah_units

    @property
    def chiller_kw_per_crah(self) -> float:
        """CRAH 1대당 칠러 설계 전력 (kW)"""
        return self.chiller_design_kw / self.n_crah_units


@dataclass
class ThermalStepResult:
    """단일 스텝 계산 결과"""

    t_zone_c: float       # 현재 서버실 온도 (°C)
    dT_dt: float          # 온도 변화율 (°C/s)
    q_in_kw: float        # IT 발열량 (kW)
    q_out_kw: float       # 냉각량 (kW)
    q_net_kw: float       # 순 열량 (kW), 양수면 가열, 음수면 냉각


class ThermalSimulator:
    """
    서버실 온도 수치 시뮬레이터.

    매 스텝마다 Q_in과 Q_out을 받아 서버실 온도를 갱신한다.
    Q_in, Q_out은 기존 열역학 모델(it_power.py, chiller.py 등)에서 계산해서 넣는다.

    사용 예시:
        sim = ThermalSimulator()
        for t in range(steps):
            q_in = calculate_cooling_load_from_it_power_kw(it_power_kw)
            q_out = calculate_chiller_power_kw(q_in, outdoor_temp_c).chiller_power_kw * cop
            result = sim.step(q_in, q_out)
            print(result.t_zone_c)
    """

    def __init__(self, config: ThermalSimulatorConfig | None = None):
        self.config = config or ThermalSimulatorConfig()
        self.t_zone_c = self.config.t_initial_c

    @property
    def m_air_kg(self) -> float:
        """서버실 공기 총 질량 (kg)"""
        return self.config.air_density_kg_m3 * self.config.room_volume_m3

    @property
    def c_eff(self) -> float:
        """유효 열용량 (kJ/K)"""
        return self.config.c_eff_kj_per_k

    def step(self, q_in_kw: float, q_out_kw: float) -> ThermalStepResult:
        """
        한 시간 스텝 진행.

        Args:
            q_in_kw:  IT 발열량 (kW)
            q_out_kw: 냉각량 (kW)

        Returns:
            ThermalStepResult
        """
        if q_in_kw < 0:
            raise ValueError(f"q_in_kw는 0 이상이어야 합니다. 입력값: {q_in_kw}")
        if q_out_kw < 0:
            raise ValueError(f"q_out_kw는 0 이상이어야 합니다. 입력값: {q_out_kw}")

        q_net_kw = q_in_kw - q_out_kw
        dT_dt = q_net_kw / self.config.c_eff_kj_per_k
        self.t_zone_c += dT_dt * self.config.dt_s

        return ThermalStepResult(
            t_zone_c=self.t_zone_c,
            dT_dt=dT_dt,
            q_in_kw=q_in_kw,
            q_out_kw=q_out_kw,
            q_net_kw=q_net_kw,
        )

    def reset(self) -> None:
        """시뮬레이터 초기 상태로 리셋"""
        self.t_zone_c = self.config.t_initial_c
