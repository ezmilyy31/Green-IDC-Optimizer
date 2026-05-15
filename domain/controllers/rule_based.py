from dataclasses import dataclass
from core.config.enums import CoolingMode
from core.config.constants import WET_BULB_FREE_THRESHOLD_C, WET_BULB_HYBRID_THRESHOLD_C
from domain.thermodynamics.chiller import calculate_wet_bulb_c
# CoolingMode.FREE_COOLING / HYBRID / CHILLER (이미 정의된 ENUM type 냉각 모드)
# 환경(chiller.py)과 동일하게 wet-bulb 기준 사용 (잠열 부하 반영)
# wb < 10°C -> 자연 공조 | wb < 18°C -> HYBRID | wb >= 18°C -> CHILLER

@dataclass
class RuleBasedResult: # RuleBased 기반 제어 모델
    # RuleBased에 따른 냉각 방식 결정 로직
    cooling_mode: CoolingMode # 냉각 방식 결정(자연 공조/ 혼합/ 기계식 냉방)
    supply_air_temp_setpoint_c: float # 공조기(CRAH)가 서버실로 불어넣는 공기 목표 온도
    free_cooling_ratio: float # 자연 냉각을 얼마나 활용할지 비율(hybrid)

"""
wet-bulb(습구온도) 기준 cooling mode 결정 — 환경(chiller.py)과 일치
잠열 부하 반영해야 자유공조 가능 시간 정확 (한국 건조 기후 wb << db)
"""

def decide_cooling_mode(outdoor_temp_c: float, outdoor_humidity_pct: float = 50.0) -> CoolingMode:
    # 습구 온도 기반 Free_COOLING / HYBRID / CHILLER
    wet_bulb = calculate_wet_bulb_c(outdoor_temp_c, outdoor_humidity_pct)
    if wet_bulb < WET_BULB_FREE_THRESHOLD_C:
        return CoolingMode.FREE_COOLING
    elif wet_bulb < WET_BULB_HYBRID_THRESHOLD_C:
        return CoolingMode.HYBRID
    else:
        return CoolingMode.CHILLER
    
"""
공급 온도 실험값 X
가장 냉각이 필요한 부분 | 중간 | 자연 공조 온도 나눠서 분배하여 정한 임의 값 
sinergym 검증 후 튜닝 필요
"""

def calculate_setpoint(cooling_mode: CoolingMode, outdoor_temp_c: float) -> float:
    # 냉각 모드에 따라 공급 온도 설정값 반환
    if (cooling_mode == CoolingMode.FREE_COOLING):
        return 22.0
    elif (cooling_mode == CoolingMode.HYBRID):
        return 20.0 # Free Cooling과 Chiller 중간 값. 
    else: return 18.0

"""
추후 it_power_kw, outdoor_humidity_pct 데이터 정리되면 해당 값 적용
"""

def run_rule_based(
    outdoor_temp_c: float,
    outdoor_humidity: float,
    it_power_kw: float) -> RuleBasedResult:
    # 함수 조합 -> 최종 결과 반환

    cooling_mode = decide_cooling_mode(outdoor_temp_c, outdoor_humidity)
    setpoint = calculate_setpoint(cooling_mode, outdoor_temp_c)

    if cooling_mode == CoolingMode.FREE_COOLING:
        ratio = 1.0
    elif cooling_mode == CoolingMode.HYBRID:
        # wet-bulb 기준 hybrid 영역 내 위치를 ratio로 (FC쪽 1.0 → chiller쪽 0.0)
        wet_bulb = calculate_wet_bulb_c(outdoor_temp_c, outdoor_humidity)
        ratio = 1.0 - (wet_bulb - WET_BULB_FREE_THRESHOLD_C) / (
            WET_BULB_HYBRID_THRESHOLD_C - WET_BULB_FREE_THRESHOLD_C
        )
    else:
        ratio = 0.0


    return RuleBasedResult(
        cooling_mode = cooling_mode,
        supply_air_temp_setpoint_c = setpoint,
        free_cooling_ratio = ratio
    )
