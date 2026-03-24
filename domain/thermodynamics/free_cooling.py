"""
자연공조(Free Cooling) 효율 모델

자연공조는 외기가 충분히 차가울 때 기계식 냉동기(칠러) 없이
외부 공기나 냉각탑을 이용해 서버실을 냉각하는 방식이다.

외기 조건(온도·습도)에 따라 자연공조 효율이 달라진다.
"""

from dataclasses import dataclass


@dataclass
class FreeCoolingResult:
    """자연공조 계산 결과"""

    is_available: bool           # 자연공조 가능 여부
    efficiency: float            # 효율 (0.0 ~ 1.0, 1.0 = 완전 자연공조)
    fan_power_kw: float          # 팬 전력 소비량 (kW)
    effective_cooling_kw: float  # 실제 냉각 가능 열량 (kW)
    mode_description: str        # 냉각 모드 설명


# 자연공조 전환 온도 기준 (명세서 기준: T_outdoor < 15°C)
FREE_COOLING_FULL_THRESHOLD_C = 15.0   # 이하: 완전 자연공조
FREE_COOLING_PARTIAL_THRESHOLD_C = 22.0  # 이하: 부분 자연공조

# 팬 전력 비율 (냉각 부하 대비, ASHRAE TC 9.9 참조)
FAN_POWER_RATIO_FREE = 0.035   # 자연공조 시 팬 전력 (3.5%)
FAN_POWER_RATIO_CHILLER = 0.08  # 기계식 냉방 시 팬 전력 (8%)


def calculate_free_cooling_efficiency(
    outdoor_temp_c: float,
    outdoor_humidity_pct: float = 50.0,
    supply_temp_setpoint_c: float = 18.0,
) -> float:
    """
    외기 조건에 따른 자연공조 효율을 계산한다.

    효율 결정 요소:
      1. 온도: 외기가 낮을수록 효율 높음
      2. 습도: 습도가 낮을수록 증발 냉각 효과 높음 (간접 효과)
      3. 공급 온도 설정값: 목표 온도와 외기 온도의 차이가 클수록 유리

    효율 계산:
      - 완전 자연공조 (T < 15°C): efficiency = 1.0
      - 혼합 구간 (15°C ≤ T < 22°C): 선형 감소
      - 기계식 전환 (T ≥ 22°C): efficiency = 0.0

    Args:
        outdoor_temp_c: 외기 온도 (°C)
        outdoor_humidity_pct: 외기 상대 습도 (%, 0~100)
        supply_temp_setpoint_c: CRAH 공급 온도 설정값 (°C)

    Returns:
        자연공조 효율 (0.0 ~ 1.0)
    """
    # 기본 온도 기반 효율
    if outdoor_temp_c < FREE_COOLING_FULL_THRESHOLD_C:
        temp_efficiency = 1.0
    elif outdoor_temp_c < FREE_COOLING_PARTIAL_THRESHOLD_C:
        # 15°C ~ 22°C 구간: 선형 감소
        temp_efficiency = 1.0 - (outdoor_temp_c - FREE_COOLING_FULL_THRESHOLD_C) / (
            FREE_COOLING_PARTIAL_THRESHOLD_C - FREE_COOLING_FULL_THRESHOLD_C
        )
    else:
        temp_efficiency = 0.0

    # 습도 보정: 높은 습도는 효율을 약간 감소 (증발 냉각 효과 감소)
    # 50% 습도를 기준으로, 100%일 때 최대 10% 효율 감소
    humidity_factor = 1.0 - max(0.0, (outdoor_humidity_pct - 50.0) / 500.0)

    # 공급 온도 여유도 보정: 외기와 목표 온도 차이가 클수록 효율 향상
    temp_margin = supply_temp_setpoint_c - outdoor_temp_c
    margin_factor = min(1.0, max(0.8, 1.0 + temp_margin * 0.02))

    return min(1.0, max(0.0, temp_efficiency * humidity_factor * margin_factor))


def calculate_free_cooling(
    cooling_load_kw: float,
    outdoor_temp_c: float,
    outdoor_humidity_pct: float = 50.0,
    supply_temp_setpoint_c: float = 18.0,
) -> FreeCoolingResult:
    """
    자연공조로 처리 가능한 냉각량과 팬 전력을 계산한다.

    자연공조가 처리하지 못한 나머지 냉각 부하는 칠러가 담당한다.

    Args:
        cooling_load_kw: 전체 냉각 부하 (kW)
        outdoor_temp_c: 외기 온도 (°C)
        outdoor_humidity_pct: 외기 상대 습도 (%)
        supply_temp_setpoint_c: 공급 온도 설정값 (°C)

    Returns:
        FreeCoolingResult (가용 여부, 효율, 팬 전력, 냉각량)
    """
    if cooling_load_kw < 0:
        raise ValueError(f"냉각 부하는 0 이상이어야 합니다. 입력값: {cooling_load_kw}")

    efficiency = calculate_free_cooling_efficiency(
        outdoor_temp_c, outdoor_humidity_pct, supply_temp_setpoint_c
    )

    is_available = efficiency > 0.0
    effective_cooling_kw = cooling_load_kw * efficiency

    # 팬 전력: 자연공조 비율에 비례
    fan_ratio = FAN_POWER_RATIO_FREE * efficiency + FAN_POWER_RATIO_CHILLER * (1 - efficiency)
    fan_power_kw = cooling_load_kw * fan_ratio

    # 모드 설명
    if efficiency >= 0.99:
        mode_description = f"완전 자연공조 (외기 {outdoor_temp_c:.1f}°C, 효율 100%)"
    elif efficiency > 0.0:
        mode_description = (
            f"혼합 냉각 (외기 {outdoor_temp_c:.1f}°C, "
            f"자연공조 {efficiency * 100:.0f}% / 칠러 {(1 - efficiency) * 100:.0f}%)"
        )
    else:
        mode_description = f"기계식 냉방 (외기 {outdoor_temp_c:.1f}°C, 자연공조 불가)"

    return FreeCoolingResult(
        is_available=is_available,
        efficiency=efficiency,
        fan_power_kw=fan_power_kw,
        effective_cooling_kw=effective_cooling_kw,
        mode_description=mode_description,
    )
