from enum import Enum

# 기본값 관련 ################################################
class CoolingMode(str, Enum):
    CHILLER = "chiller"           # 기계식 냉방 (칠러 가동)
    FREE_COOLING = "free_cooling" # 자연공조 (외기로 직접 냉각)
    HYBRID = "hybrid"             # 혼합 (부분 자연공조 + 칠러 보조)

# Forecast 관련 ############################################

# 예측 목표 선택지:     1. IT 부하 예측 /   2. 냉각 수요 예측
class PredictionTarget(str, Enum):
    IT_LOAD = "it_load"
    COOLING_DEMAND = "cooling_demand"
    BOTH = "both"

# 예측에 사용할 model 선택지
class ModelType(str, Enum): 
    LGBM = "lgbm"
    LSTM = "lstm"