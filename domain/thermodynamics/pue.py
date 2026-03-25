"""
PUE(Power Usage Effectiveness) 계산 모델

PUE = 데이터센터 총 전력 / IT 장비 전력

PUE = 1.0  : 이상적 (냉각·조명 등 부가 전력이 전혀 없음)
PUE = 1.09 : NAVER 각 춘천 (세계 최고 수준)
PUE = 1.58 : 글로벌 평균
PUE = 2.03 : 국내 민간 평균
PUE가 낮을수록 에너지 효율이 높다.
"""

from dataclasses import dataclass


# 벤치마크 PUE 값 (참고용) — NAVER 각 춘천 1.09가 목표 기준
PUE_BENCHMARK = {
    "naver_chuncheon": 1.09,   # NAVER 각 춘천 (세계 최고 수준)
    "google_global": 1.10,      # Google 글로벌 평균 (2023)
    "global_average": 1.58,     # 글로벌 평균
    "korea_private": 2.03,      # 국내 민간 평균
    "korea_public": 3.13,       # 국내 공공기관 평균
    "green_dc_standard": 1.66,  # 그린데이터센터 인증 기준
}


@dataclass
class PUEResult:
    """PUE 계산 결과"""

    pue: float                      # Power Usage Effectiveness
    total_power_kw: float           # 데이터센터 총 전력 (kW)
    it_power_kw: float              # IT 장비 전력 (kW)
    cooling_power_kw: float         # 냉각 전력 (kW)
    other_power_kw: float           # 기타 전력 (kW, 조명·UPS 손실 등)
    efficiency_vs_benchmark: float  # NAVER 각 춘천(1.09) 대비 효율 비율


def calculate_pue(
    it_power_kw: float,
    cooling_power_kw: float,
    other_power_kw: float | None = None,
) -> PUEResult:
    """
    PUE(Power Usage Effectiveness)를 계산한다.

    공식: PUE = (P_IT + P_냉각 + P_기타) / P_IT

    P_기타: 조명, UPS 손실, 배전 손실 등 (보통 IT 전력의 3~8%)

    Args:
        it_power_kw: IT 장비 전력 소비량 (kW) — 서버, 스토리지, 네트워크 장비
        cooling_power_kw: 냉각 시스템 전력 (kW) — 칠러, CRAH 팬, 냉각탑
        other_power_kw: 기타 전력 (kW). None이면 IT 전력의 5%로 추정

    Returns:
        PUEResult (PUE 값 및 세부 전력 분류)

    Raises:
        ValueError: IT 전력이 0 이하일 때
    """
    if it_power_kw <= 0:
        raise ValueError(f"IT 전력은 0보다 커야 합니다. 입력값: {it_power_kw}")
    if cooling_power_kw < 0:
        raise ValueError(f"냉각 전력은 0 이상이어야 합니다. 입력값: {cooling_power_kw}")

    # 기타 전력: 미지정 시 IT 전력의 5%로 추정 (UPS 손실, 조명, 배전 등)
    if other_power_kw is None:
        other_power_kw = it_power_kw * 0.05

    total_power_kw = it_power_kw + cooling_power_kw + other_power_kw
    pue = total_power_kw / it_power_kw

    # NAVER 각 춘천 PUE 1.09 대비 효율 비율 (낮을수록 좋음)
    efficiency_vs_benchmark = pue / PUE_BENCHMARK["naver_chuncheon"]

    return PUEResult(
        pue=pue,
        total_power_kw=total_power_kw,
        it_power_kw=it_power_kw,
        cooling_power_kw=cooling_power_kw,
        other_power_kw=other_power_kw,
        efficiency_vs_benchmark=efficiency_vs_benchmark,
    )


