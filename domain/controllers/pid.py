# PID 제어기 구현

from dataclasses import dataclass, field

"""
Incremental (velocity) PID:
  supply[t] = supply[t-1] + Kp·Δe + Ki·e·dt + Kd·(Δe/dt)

위치형(position) PID 대신 incremental PID를 사용하는 이유:
  - 위치형: output = Kp·e + Ki·∫e·dt + Kd·de/dt
    → output이 공급 온도 절댓값을 직접 계산
    → 큰 초기 오차에서 항상 clamp에 걸려 Kp 크기에 무관하게 bang-bang 동작
  - incremental: output = 이전 공급 온도 + PID가 계산한 변화량
    → 공급 온도가 점진적으로 조절됨
    → Kp가 "온도 변화량에 대한 공급 온도 변화율"로 물리적 의미를 가짐

게인값 (IMC 기반 + sweep 튜닝, 정상 운영 CRAH 3대 기준):
  Kp = 1.0   → 오차 1°C당 공급 온도 1°C/s 조정
  Ki = 0.001 → 정상 상태 오차 제거 (IMC 기준 Ki_imc ≈ 0.0035, sweep 결과 0.001 채택)
  Kd = 0.5   → supply_min=16°C 확장 후 S3 칠러 고장 복구 시 과냉각 방지.
              회복 구간(delta_error > 0)에서 supply를 살짝 올려 setpoint 근접 수렴

  도출 근거:
  - 시스템 유효 열용량: C_eff = 9,009 kJ/K
    (공기 1,809 + CPU서버 4,000 + GPU서버 600 + 랙 600 + 구조물 2,000)
  - 정상 운영 공기 유량: ṁ·cp = 33 kg/s × 1.005 kJ/kg·K = 33.2 kW/K  (CRAH 3대)
  - 열적 시정수: τ = C_eff / (ṁ·cp) = 9,009 / 33.2 = 272초
  - IMC 공식: Kp_imc = 1.0, Ki_imc = Kp / τ ≈ 0.0037

고정 setpoint:
  - 서버실 목표 온도 24°C 고정 (zone_target_c 파라미터로 주입)
  - 초기화: supply = T_zone - Q_in/(ṁ·cp) (평형 공급온도), _prev_error = 0

Anti-windup: 조건부 적분
  - supply가 output_min/max에 도달하면 적분 누적 중단
"""

# 검증된 게인값 (incremental PID, IMC 기반 + sweep 튜닝, CRAH 3대 정상 운영 기준)
DEFAULT_KP = 1.0
DEFAULT_KI = 0.001
DEFAULT_KD = 1.0

# CRAH 공급 온도 물리적 한계 (°C)
SUPPLY_TEMP_MIN_C = 16.0
SUPPLY_TEMP_MAX_C = 27.0


@dataclass
class PIDController:
    kp: float = DEFAULT_KP
    ki: float = DEFAULT_KI
    kd: float = DEFAULT_KD
    setpoint: float = 24.0       # 목표 서버실 온도 (°C), run_pid_loop에서 zone_target_c로 고정 주입
    output_min: float = SUPPLY_TEMP_MIN_C
    output_max: float = SUPPLY_TEMP_MAX_C

    _prev_error: float = field(default=0.0, init=False)
    _supply: float = field(default=SUPPLY_TEMP_MAX_C, init=False)  # 이전 공급 온도, 초기값=최대

    def compute(self, current_value: float, dt: float = 1.0) -> float:
        """
        Incremental PID로 공급 온도 변화량을 계산하고 공급 온도를 갱신한다.

        Args:
            current_value: 현재 서버실 온도 (°C)
            dt: 시간 스텝 (초)

        Returns:
            CRAH 공급 온도 설정값 (°C)
        """
        error = self.setpoint - current_value
        delta_error = error - self._prev_error

        delta_supply = self.kp * delta_error + self.ki * error * dt + self.kd * (delta_error / dt)

        # anti-windup: supply가 한계에 도달하면 적분항만 중단
        new_supply = self._supply + delta_supply
        if new_supply <= self.output_min or new_supply >= self.output_max:
            delta_supply = self.kp * delta_error + self.kd * (delta_error / dt)
            new_supply = self._supply + delta_supply

        self._supply = max(self.output_min, min(self.output_max, new_supply))
        self._prev_error = error

        return self._supply

    def reset(self) -> None:
        self._prev_error = 0.0
        self._supply = self.output_max


@dataclass
class PIDLoopResult:
    """단일 스텝 PID 제어 시뮬레이션 결과"""

    step: int
    t_zone_c: float           # 서버실 온도 (°C)
    supply_temp_c: float      # CRAH 공급 온도 설정값 (°C)
    q_in_kw: float            # IT 발열량 (kW)
    q_out_kw: float           # 실제 냉각량 (kW)
    error_c: float            # 오차 = setpoint - T_zone (°C)
    outdoor_temp_c: float     # 해당 스텝의 외기 온도 (°C)
    n_crah: int               # 해당 스텝의 가동 CRAH 대수 (대수 감소 시 ṁ·칠러 용량 동시 감소)
