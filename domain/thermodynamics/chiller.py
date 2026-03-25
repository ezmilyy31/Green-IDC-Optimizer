"""
칠러(Chiller) 전력 소비 모델

칠러는 냉각수를 냉각시키는 기계식 냉동기다.
외기 온도가 높을수록 칠러가 더 많은 전력을 소비한다.

핵심 지표: COP (Coefficient Of Performance, 성능계수)
  - COP = 제거한 열량 / 소비한 전력
  - COP가 높을수록 효율적 (같은 전력으로 더 많은 열을 제거)
  - 외기 온도가 낮을수록 COP 상승 → 전력 절감
"""

from dataclasses import dataclass
from enum import Enum


class CoolingMode(str, Enum):
    CHILLER = "chiller"           # 기계식 냉방 (칠러 가동)
    FREE_COOLING = "free_cooling" # 자연공조 (외기로 직접 냉각)
    HYBRID = "hybrid"             # 혼합 (부분 자연공조 + 칠러 보조)


@dataclass
class ChillerResult:
    """칠러 계산 결과"""

    cop: float                  # 성능계수 (무차원)
    chiller_power_kw: float     # 칠러 전력 소비량 (kW)
    cooling_mode: CoolingMode   # 냉각 모드


# 냉각 모드 전환 온도 기준 (°C) — 명세서 기준
FREE_COOLING_THRESHOLD_C = 15.0   # 이하: 완전 자연공조
HYBRID_THRESHOLD_C = 22.0         # 이하: 혼합 모드


def calculate_cop(outdoor_temp_c: float) -> float:
    """
    외기 온도에 따른 칠러 COP(성능계수)를 계산한다.

    공식: COP = max(2.0, 6.0 - 0.1 × (T_outdoor - 15))

    물리적 의미:
      - 외기가 15°C일 때 COP = 6.0 (설계 기준점)
      - 외기가 1°C 오를 때마다 COP가 0.1씩 감소
      - COP 최솟값은 2.0 (매우 더운 날에도 이 이하로 떨어지지 않는다고 가정)

    예시:
      - 외기 5°C  → COP = 7.0 → 전력 1kW로 7kW 냉각 가능
      - 외기 15°C → COP = 6.0 → 전력 1kW로 6kW 냉각 가능
      - 외기 35°C → COP = 4.0 → 전력 1kW로 4kW 냉각 가능
      - 외기 55°C → COP = 2.0 (하한) → 전력 1kW로 2kW 냉각 가능

    Args:
        outdoor_temp_c: 외기 온도 (°C)

    Returns:
        칠러 COP (무차원, 최솟값 2.0)
    """
    return max(2.0, 6.0 - 0.1 * (outdoor_temp_c - 15.0))


def calculate_chiller_power_kw(
    cooling_load_kw: float,
    outdoor_temp_c: float,
) -> ChillerResult:
    """
    냉각 부하와 외기 온도로 칠러 전력 소비량을 계산한다.

    냉각 모드 결정 (명세서 기준):
      - 외기 < 15°C: Free Cooling (자연공조) — 칠러 미사용
      - 15°C ≤ 외기 < 22°C: Hybrid (혼합) — 칠러 일부 사용
      - 외기 ≥ 22°C: Chiller (기계식) — 칠러 전면 가동

    공식 (기계식 모드):
      P_chiller = Q_cooling / COP

    Args:
        cooling_load_kw: 제거해야 할 열량 (kW)
        outdoor_temp_c: 외기 온도 (°C)

    Returns:
        ChillerResult (COP, 칠러 전력, 냉각 모드)

    Raises:
        ValueError: 냉각 부하가 음수일 때
    """
    if cooling_load_kw < 0:
        raise ValueError(f"냉각 부하는 0 이상이어야 합니다. 입력값: {cooling_load_kw}")

    cop = calculate_cop(outdoor_temp_c)

    if outdoor_temp_c < FREE_COOLING_THRESHOLD_C:
        mode = CoolingMode.FREE_COOLING
        # 자연공조: 기계식 칠러 미사용
        chiller_power_kw = 0.0

    elif outdoor_temp_c < HYBRID_THRESHOLD_C:
        # 혼합 모드: 외기 온도에 따라 칠러 비중을 선형 보간
        # 15°C에서 0% 칠러, 22°C에서 100% 칠러
        chiller_fraction = (outdoor_temp_c - FREE_COOLING_THRESHOLD_C) / (
            HYBRID_THRESHOLD_C - FREE_COOLING_THRESHOLD_C
        )
        mode = CoolingMode.HYBRID
        chiller_power_kw = (cooling_load_kw * chiller_fraction) / cop

    else:
        # 기계식 냉방: 칠러 전면 가동
        mode = CoolingMode.CHILLER
        chiller_power_kw = cooling_load_kw / cop

    return ChillerResult(cop=cop, chiller_power_kw=chiller_power_kw, cooling_mode=mode)
