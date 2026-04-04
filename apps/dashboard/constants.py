"""대시보드 공통 상수 정의 대부분이 디자인적인것들."""

from core.config.constants import (
    CARBON_FACTOR_TCO2_PER_MWH as CARBON_FACTOR_TCO2_PER_MWH,
    CRISIS_CONFIGS as CRISIS_CONFIGS,
    CRISIS_STRATEGIES as CRISIS_STRATEGIES,
    ELECTRICITY_COST_KRW_PER_KWH as ELECTRICITY_COST_KRW_PER_KWH,
    PUE_BENCHMARK,
    SCENARIO_TEMP_PROFILES as SCENARIO_TEMP_PROFILES,
    T_RETURN_DESIGN_C,
    WORKLOAD_PROFILE as WORKLOAD_PROFILE,
    WUE_BY_MODE as WUE_BY_MODE,
)

TEMP_WARNING_THRESHOLD_C = T_RETURN_DESIGN_C  # 경고 임계값 = 설계 환기 온도 (27°C)

# ── 통합 색상 팔레트 ──────────────────────────────────────────────────────────
# 메인 블루 팔레트: #5D90FF (강조 블루) · #7BD2F7 (스카이) · #A3ABFB (인디고)
CLR_IT        = "#FBBF24"   # IT 전력    (앰버 옐로)
CLR_CHILLER   = "#A3ABFB"   # 칠러 전력  (인디고)
CLR_TOTAL     = "#F59E0B"   # 총 전력    (딥 앰버)
CLR_GOOD      = "#2ECC71"   # 정상 / 양호  (의미 색상 유지)
CLR_WARN      = "#F39C12"   # 경고          (의미 색상 유지)
CLR_DANGER    = "#E74C3C"   # 위험          (의미 색상 유지)
CLR_CARBON    = "#7BD2F7"   # 탄소 차트 bar
CLR_COST      = "#A3ABFB"   # 비용 차트 bar
CLR_COST_LINE = "#5D90FF"   # 비용 누적선

NUM_RACKS           = 40
NAVER_PUE_BENCHMARK = PUE_BENCHMARK["naver_chuncheon"]

# core PUE_BENCHMARK → 대시보드 표시용 한국어 레이블 매핑
_PUE_LABEL_MAP: dict[str, str] = {
    "naver_chuncheon":   "NAVER 각 춘천",
    "google_global":     "Google 글로벌",
    "green_dc_standard": "그린DC 인증",
    "global_average":    "글로벌 평균",
    "korea_private":     "국내 민간",
    "korea_public":      "국내 공공기관",
}


COOLING_MODE_LABELS = {"chiller": "기계식", "free_cooling": "자연공조", "hybrid": "혼합"}
COOLING_MODE_COLORS = ["#5D90FF", "#7BD2F7", "#A3ABFB"]   # 블루 팔레트: 기계식 / 자연공조 / 혼합
MODE_COLOR_MAP      = {"chiller": "#5D90FF", "hybrid": "#A3ABFB", "free_cooling": "#7BD2F7"}

PUE_GAUGE_STEPS = [
    {"range": [1.0, 1.4], "color": "#d5f5e3"},
    {"range": [1.4, 1.8], "color": "#fef9e7"},
    {"range": [1.8, 2.5], "color": "#fdecea"},
]

PUE_BENCHMARKS = [
    (_PUE_LABEL_MAP[k], str(v))
    for k, v in PUE_BENCHMARK.items()
    if k in _PUE_LABEL_MAP
]

ANIM_SPEED_OPTIONS = {
    "매우 빠르게 (0.1초)": 0.1,
    "빠르게 (0.2초)":      0.2,
    "보통 (0.4초)":        0.4,
    "느리게 (0.8초)":      0.8,
}
