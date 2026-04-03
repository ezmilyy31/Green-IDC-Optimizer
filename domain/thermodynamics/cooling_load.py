"""
냉각 부하 계산 모델

데이터센터에서 제거해야 할 열량(냉각 부하)을 계산한다.

핵심 원리:
  - 서버에 공급된 전기 에너지는 결국 모두 열로 변환된다 (에너지 보존 법칙).
  - 따라서 냉각 부하 ≈ IT 전력 소비량 (정상 상태 기준).
  - 공기 냉각 방정식: Q = ṁ × c_p × ΔT
"""


from core.config.constants import (
    T_SUPPLY_DESIGN_C,
    T_RETURN_DESIGN_C,
    M_AIR_DESIGN_KG_S,
    NUM_SERVERS_DESIGN,
)

# 공기의 물리 상수
AIR_SPECIFIC_HEAT_KJ_PER_KG_K = 1.005  # 공기 정압 비열 (kJ/kg·K)


def calculate_m_air_for_servers(num_servers: int) -> float:
    """
    서버 수에 비례하여 설계 공기 유량을 반환한다.

    명세서 설계 기준(500대 → 50 kg/s)을 실제 서버 수에 선형 스케일한다.

    Args:
        num_servers: 실제 CPU 서버 대수

    Returns:
        스케일된 설계 공기 유량 (kg/s)
    """
    return M_AIR_DESIGN_KG_S * (num_servers / NUM_SERVERS_DESIGN)


def calculate_cooling_load_from_airflow_kw(
    m_dot_kg_per_s: float,
    supply_temp_c: float,
    return_temp_c: float,
    c_p: float = AIR_SPECIFIC_HEAT_KJ_PER_KG_K,
) -> float:
    """
    공기 유량과 온도 차이로 냉각 부하를 계산한다.

    공식: Q = ṁ × c_p × ΔT
      - ṁ  : 공기 질량 유량 (kg/s)
      - c_p : 공기 비열 (kJ/kg·K), 상온 기준 1.005
      - ΔT  : 환기 온도 - 공급 온도 (°C)

    Args:
        m_dot_kg_per_s: 공기 질량 유량 (kg/s)
        supply_temp_c: CRAH 공급 온도 — 서버실로 들어가는 차가운 공기 온도 (°C)
        return_temp_c: 환기 온도 — 서버를 통과해 나온 뜨거운 공기 온도 (°C)
        c_p: 공기 비열 (kJ/kg·K), 기본값 1.005

    Returns:
        냉각 부하 (kW)

    Raises:
        ValueError: 환기 온도가 공급 온도보다 낮을 때 (물리적으로 불가)
    """
    if return_temp_c < supply_temp_c:
        raise ValueError(
            f"환기 온도({return_temp_c}°C)가 공급 온도({supply_temp_c}°C)보다 낮습니다. "
            "서버를 통과한 공기는 반드시 더 뜨거워야 합니다."
        )
    delta_t = return_temp_c - supply_temp_c
    return m_dot_kg_per_s * c_p * delta_t  # kW (kJ/s)


def calculate_cooling_load_from_it_power_kw(
    it_power_kw: float,
    overhead_factor: float = 1.0,
) -> float:
    """
    IT 전력으로부터 냉각 부하를 계산한다 (에너지 보존 법칙).

    서버에 공급된 전기 에너지는 결국 모두 열로 바뀐다.
    따라서 냉각 시스템이 제거해야 할 열량 = IT 전력 소비량.

    Args:
        it_power_kw: 전체 IT 장비 전력 소비량 (kW)
        overhead_factor: 추가 열 발생 계수 (기본 1.0 = IT 전력과 동일)
                         UPS 손실 등을 포함하려면 1.02~1.05 사용

    Returns:
        냉각 부하 (kW)
    """
    if it_power_kw < 0:
        raise ValueError(f"IT 전력은 0 이상이어야 합니다. 입력값: {it_power_kw}")
    return it_power_kw * overhead_factor


