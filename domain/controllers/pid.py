# PID 제어기 구현

from dataclasses import dataclass, field

"""
`u(t) = Kp·e(t) + Ki·∫e(t)dt + Kd·de(t)/dt`

- `e(t)` : 목표 온도 - 현재 온도 (오차)
- `Kp` (비례): 오차에 즉각 반응. 너무 크면 진동 발생
- `Ki` (적분): 누적 오차 제거. 정상 상태 오차 보정
- `Kd` (미분): 오차 변화율 예측. 과도 반응 억제
- `rule_based.py`에서 공급 온도를 조절해 서버 온도 18~27°C 유지 (유지율 95%+ 필수)
- 비례 게인은 추후 시뮬레이션 돌려보면서 최적값 찾아야 함 
"""

@dataclass
class PIDController:
    kp: float # 비례 게인, 각 값들은 초기값 정한 후 추후 Sinergym 시뮬레이션 -> 최적값 찾기 
    ki: float # 적분 게인
    kd: float # 미분 게인
    setpoint: float # 목표 값
    
    _prev_error: float = field(default = 0.0, init = False) # 이전 오차
    _integral: float = field(default = 0.0, init = False) # 누적 오차

    def compute(self, current_value: float, dt: float = 1.0) -> float:
        error = self.setpoint - current_value # e(t)
        self._integral += error * dt # 누적 오차 (Ki·∫e(t)dt )
        derivative = (error - self._prev_error) / dt # 변화율 (Kd·de(t)/dt)
        self._prev_error = error

        return self.kp * error + self.ki * self._integral + self.kd * derivative

    def reset(self) -> None:
        self._prev_error = 0.0
        self._integral = 0.0
