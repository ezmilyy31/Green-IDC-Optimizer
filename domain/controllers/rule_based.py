from dataclasses import dataclass
from core.config.enums import CoolingMode
from core.config.constants import FREE_COOLING_THRESHOLD_C, HYBRID_THRESHOLD_C
# CoolingMode.FREE_COOLING / HYBRID / CHILLER (이미 정의된 ENUM type 냉각 모드)
# 15도 이하 -> 자연 공조 / 15도 초과 22도 이하 HYBRID / 22도 이상 기계식 냉방 <free cooling> 기준 

@dataclass
class RuleBasedResult: # RuleBased 기반 제어 모델 
    # RuleBased에 따른 냉각 방식 결정 로직
    cooling_mode: CoolingMode # 냉각 방식 결정(자연 공조/ 혼합/ 기계식 냉방)
    supply_air_temp_setpoint_c: float # 공조기(CRAH)가 서버실로 불어넣는 공기 목표 온도
    free_cooling_ratio: float # 자연 냉각을 얼마나 활용할지 비율(hybrid)

"""
free_cooling.py에서 제시하는 온도전환 기준으로 CoolingMode 설정
15도 미만 -> 자연 공조 | 22도 이하 -> HYBRID | 22도 초과 - CHILLER(외부 냉방) 필요
"""

def decide_cooling_mode(outdoor_temp_c: float) -> CoolingMode:
    # 온도 임계 값 기반 Free_COOLING / HYBRID / CHILLER
    if (outdoor_temp_c < FREE_COOLING_THRESHOLD_C):
        return CoolingMode.FREE_COOLING
    elif (outdoor_temp_c <= HYBRID_THRESHOLD_C):
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

    cooling_mode = decide_cooling_mode(outdoor_temp_c)
    setpoint = calculate_setpoint(cooling_mode, outdoor_temp_c)

    if cooling_mode == CoolingMode.FREE_COOLING:
        ratio =  1.0
    elif cooling_mode == CoolingMode.HYBRID:
        ratio = 1 - (outdoor_temp_c - 15) / (22 - 15)
    else: ratio = 0.0


    return RuleBasedResult(
        cooling_mode = cooling_mode,
        supply_air_temp_setpoint_c = setpoint,
        free_cooling_ratio = ratio
    )
