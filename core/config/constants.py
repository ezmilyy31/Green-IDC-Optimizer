
# 냉각 모드 전환 온도 기준 (°C) — 명세서 기준
FREE_COOLING_THRESHOLD_C = 15.0   # 이하: 완전 자연공조
HYBRID_THRESHOLD_C = 22.0         # 이하: 혼합 모드

# 팬 전력 비율 (냉각 부하 대비, ASHRAE TC 9.9 참조)
FAN_POWER_RATIO_FREE = 0.035      # 자연공조 시 팬 전력 (3.5%)
FAN_POWER_RATIO_CHILLER = 0.08    # 기계식 냉방 시 팬 전력 (8%)

# 냉각 부하 설계 파라미터 (SyntheticIDCBuilder 기준, p.15-16)
T_SUPPLY_DESIGN_C = 18.0          # CRAH 공급 온도 설계값 (°C)
T_RETURN_DESIGN_C = 27.0          # 환기 온도 설계값 (°C)
M_AIR_DESIGN_KG_S = 50.0          # 설계 공기 유량 (kg/s) — 서버 500대 기준
NUM_SERVERS_DESIGN = 500           # 설계 기준 서버 수

# 서버 전력 스펙 (SPECpower_ssj2008 기준)
CPU_SERVER_P_IDLE_W = 200.0        # CPU 서버 유휴 전력 (W, Intel Xeon 기준)
CPU_SERVER_P_MAX_W = 500.0         # CPU 서버 최대 전력 (W)
GPU_SERVER_P_IDLE_W = 300.0        # GPU 서버 유휴 전력 (W, NVIDIA A100 × 4 기준)
GPU_SERVER_P_MAX_W = 1500.0        # GPU 서버 최대 전력 (W)

# PUE 벤치마크 (참고용) — NAVER 각 춘천 1.09가 목표 기준
PUE_BENCHMARK = {
    "naver_chuncheon": 1.09,    # NAVER 각 춘천 (세계 최고 수준)
    "google_global": 1.10,      # Google 글로벌 평균 (2023)
    "global_average": 1.58,     # 글로벌 평균
    "korea_private": 2.03,      # 국내 민간 평균
    "korea_public": 3.13,       # 국내 공공기관 평균
    "green_dc_standard": 1.66,  # 그린데이터센터 인증 기준
}
