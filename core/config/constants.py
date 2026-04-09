
# 냉각 모드 전환 온도 기준 (°C) — 명세서 기준
FREE_COOLING_THRESHOLD_C = 15.0   # 이하: 완전 자연공조
HYBRID_THRESHOLD_C = 22.0         # 이하: 혼합 모드

# 팬 전력 비율 (냉각 부하 대비, ASHRAE TC 9.9 참조)
FAN_POWER_RATIO_FREE = 0.035      # 자연공조 시 팬 전력 (3.5%)
FAN_POWER_RATIO_CHILLER = 0.08    # 기계식 냉방 시 팬 전력 (8%)

# 냉각 부하 설계 파라미터 (SyntheticIDCBuilder 기준, p.15-16)
T_SUPPLY_DESIGN_C = 20.0          # CRAH 공급 온도 설계값 (°C)
T_RETURN_DESIGN_C = 27.0          # 환기 온도 설계값 (°C)
M_AIR_DESIGN_KG_S = 50.0          # 설계 공기 유량 (kg/s) — 서버 500대 기준
NUM_SERVERS_DESIGN = 500           # 설계 기준 서버 수

# 서버 전력 스펙 (SPECpower_ssj2008 기준)
CPU_SERVER_P_IDLE_W = 200.0        # CPU 서버 유휴 전력 (W, Intel Xeon 기준)
CPU_SERVER_P_MAX_W = 500.0         # CPU 서버 최대 전력 (W)
GPU_SERVER_P_IDLE_W = 300.0        # GPU 서버 유휴 전력 (W, NVIDIA A100 × 4 기준)
GPU_SERVER_P_MAX_W = 1500.0        # GPU 서버 최대 전력 (W)

# ESG 계수 나중에 쓴다면 ..
# TODO(업데이트 필요): 한국에너지공단(KEA) 연도별 전력 배출계수 갱신 시 수정 — 현재 2023년 기준 0.459 tCO₂/MWh
CARBON_FACTOR_TCO2_PER_MWH   = 0.459   # 한국 전력 탄소 배출계수
# TODO(업데이트 필요): 산업용 전기요금 변경 시 수정 — 현재 120원/kWh (한전 산업용 갑 기준 임시값)
ELECTRICITY_COST_KRW_PER_KWH = 120.0   # 산업용 전기요금 (원/kWh)
WUE_BY_MODE = {
    "chiller":      1.8,
    # TODO(팀 협의): hybrid WUE 값 미확정 — thermodynamic_model.md에 명시 없음
    "hybrid":       0.0,
    "free_cooling": 0.2,
}

# 위기 시나리오 정의 (명세서 기준)
CRISIS_CONFIGS: dict = {
    None: {
        "label":           "정상",
        "util_multiplier":    1.0,
        "outdoor_override":   None,
        "chiller_ratio":      1.0,
    },
    "server_surge": {
        "label":           "서버 급증 (+30%)",
        "util_multiplier":    1.3,
        "outdoor_override":   None,
        "chiller_ratio":      1.0,
    },
    "chiller_failure": {
        "label":           "냉각기 고장 (칠러 1대 탈락)",
        "util_multiplier":    1.0,
        "outdoor_override":   None,
        "chiller_ratio":      0.5,
    },
    "heat_wave": {
        "label":           "폭염 (외기 38°C+)",
        "util_multiplier":    1.0,
        "outdoor_override":   38.0,
        "chiller_ratio":      1.0,
    },
}

CRISIS_STRATEGIES: dict[str, str] = {
    "server_surge":    "칠러 추가 가동 + 공급 온도 1도 하향",
    "chiller_failure": "IT 부하 분산 요청 + Free Cooling 최대화",
    "heat_wave":       "칠러 전력 증가 + 공급 온도 최저화 (16°C)",
}

SCENARIO_TEMP_PROFILES: dict[str, dict] = {
    "여름 (Summer)":    {"base": 30.0, "amplitude": 6.0},
    "봄/가을 (Spring/Fall)": {"base": 15.0, "amplitude": 8.0},
    "겨울 (Winter)":    {"base": 2.0,  "amplitude": 6.0},
}

# 시간대별 워크로드 비율 (00~23시)
WORKLOAD_PROFILE: list[float] = [
    0.35, 0.30, 0.28, 0.27, 0.28, 0.32,   # 00~05: 야간 저부하
    0.45, 0.60, 0.75, 0.85, 0.90, 0.92,   # 06~11: 오전 증가
    0.90, 0.88, 0.92, 0.93, 0.88, 0.82,   # 12~17: 주간 피크
    0.72, 0.65, 0.58, 0.50, 0.42, 0.38,   # 18~23: 야간 감소
]

# PUE 벤치마크 (참고용) — NAVER 각 춘천 1.09가 목표 기준
PUE_BENCHMARK = {
    "naver_chuncheon": 1.09,    # NAVER 각 춘천 (세계 최고 수준)
    "google_global": 1.10,      # Google 글로벌 평균 (2023)
    "global_average": 1.58,     # 글로벌 평균
    "korea_private": 2.03,      # 국내 민간 평균
    "korea_public": 3.13,       # 국내 공공기관 평균
    "green_dc_standard": 1.66,  # 그린데이터센터 인증 기준
}
