"""
AI Green IDC 통합 관제 대시보드

열역학 기반 시뮬레이션 결과와 서비스 상태를 시각화한다.
domain 모듈을 직접 import해서 HTTP 서비스 없이도 동작한다.
"""

import math
import random
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from apps.dashboard.api_client import (
    get_all_service_status,
    optimize_control,
    rl_control,
)
from domain.thermodynamics.chiller import calculate_chiller_power_kw
from domain.thermodynamics.cooling_load import (
    AIR_SPECIFIC_HEAT_KJ_PER_KG_K,
    calculate_cooling_load_from_it_power_kw,
    calculate_m_air_for_servers,
)
from domain.thermodynamics.it_power import calculate_total_it_power_kw
from domain.thermodynamics.pue import calculate_pue


# ════════════════════════════════════════════════════════════════════════════
# 상수
# ════════════════════════════════════════════════════════════════════════════

TEMP_WARNING_THRESHOLD_C = 27.0   # 서버 환기 온도 경고 기준 (ASHRAE A1)
NUM_RACKS                = 40
NAVER_PUE_BENCHMARK      = 1.09

# ESG 계수 (명세서 기준)
CARBON_FACTOR_TCO2_PER_MWH  = 0.459   # 한국 전력 탄소 배출계수
ELECTRICITY_COST_KRW_PER_KWH = 120.0  # 산업용 전기요금 (원/kWh)
WUE_BY_MODE = {                        # 냉각 모드별 물 사용량 (L/kWh IT)
    "chiller":      1.5,
    "hybrid":       0.8,
    "free_cooling": 0.2,
}

# 위기 시나리오 정의 (명세서 9항 기준)
CRISIS_CONFIGS: dict[str, dict] = {
    None: {
        "label":          "정상",
        "util_multiplier":   1.0,
        "outdoor_override":  None,   # 외기 온도 강제 지정 없음
        "chiller_ratio":     1.0,
    },
    "server_surge": {
        "label":          "서버 급증 (+30%)",
        "util_multiplier":   1.3,    # IT 부하 +30%
        "outdoor_override":  None,
        "chiller_ratio":     1.0,
    },
    "chiller_failure": {
        "label":          "냉각기 고장 (칠러 1대 탈락)",
        "util_multiplier":   1.0,
        "outdoor_override":  None,
        "chiller_ratio":     0.5,    # 칠러 용량 50%만 가동
    },
    "heat_wave": {
        "label":          "폭염 (외기 38°C+)",
        "util_multiplier":   1.0,
        "outdoor_override":  38.0,   # 외기 온도 38°C 고정
        "chiller_ratio":     1.0,
    },
}

SCENARIO_TEMP_PROFILES = {
    "여름 (Summer)":    {"base": 30.0, "amplitude": 6.0},
    "봄/가을 (Spring)": {"base": 15.0, "amplitude": 8.0},
    "겨울 (Winter)":    {"base": 2.0,  "amplitude": 6.0},
}

# 시간대별 워크로드 비율 (00~23시, 기준값 0.6 기준 정규화)
WORKLOAD_PROFILE = [
    0.35, 0.30, 0.28, 0.27, 0.28, 0.32,   # 00~05: 야간 저부하
    0.45, 0.60, 0.75, 0.85, 0.90, 0.92,   # 06~11: 오전 증가
    0.90, 0.88, 0.92, 0.93, 0.88, 0.82,   # 12~17: 주간 피크
    0.72, 0.65, 0.58, 0.50, 0.42, 0.38,   # 18~23: 야간 감소
]

COOLING_MODE_LABELS = {"chiller": "기계식", "free_cooling": "자연공조", "hybrid": "혼합"}
COOLING_MODE_COLORS = ["#E74C3C", "#2ECC71", "#F39C12"]

PUE_GAUGE_STEPS = [
    {"range": [1.0, 1.4], "color": "#d5f5e3"},
    {"range": [1.4, 1.8], "color": "#fef9e7"},
    {"range": [1.8, 2.5], "color": "#fdecea"},
]

PUE_BENCHMARKS = [
    ("NAVER 각 춘천", "1.09"),
    ("Google 글로벌",  "1.10"),
    ("글로벌 평균",    "1.58"),
    ("국내 민간",      "2.03"),
]


# ════════════════════════════════════════════════════════════════════════════
# 시뮬레이션
# ════════════════════════════════════════════════════════════════════════════

def run_simulation(
    scenario_name: str,
    num_cpu: int,
    num_gpu: int,
    base_util: float,
    supply_temp_c: float,
    crisis: str | None,
) -> pd.DataFrame:
    """24시간 열역학 시뮬레이션을 실행하고 결과 DataFrame을 반환한다."""
    profile = SCENARIO_TEMP_PROFILES[scenario_name]
    m_air   = calculate_m_air_for_servers(num_cpu + num_gpu)
    cfg     = CRISIS_CONFIGS[crisis]

    rows = []
    for hour in range(24):
        # 외기 온도 계산 (폭염 시나리오는 38°C 강제 고정)
        raw_outdoor = (
            profile["base"]
            + profile["amplitude"] * math.sin(math.radians((hour - 4) * 15))
        )
        outdoor_temp = cfg["outdoor_override"] if cfg["outdoor_override"] else raw_outdoor

        # IT 부하 (서버급증 시나리오는 +30% 적용)
        util        = min(1.0, base_util * WORKLOAD_PROFILE[hour] / 0.6 * cfg["util_multiplier"])
        it_power_kw = calculate_total_it_power_kw(util, num_cpu, num_gpu)
        cooling_load = calculate_cooling_load_from_it_power_kw(it_power_kw)

        # 칠러 계산 (냉각기 고장 시나리오는 용량 50% 제한)
        chiller              = calculate_chiller_power_kw(cooling_load, outdoor_temp)
        actual_chiller_power = chiller.chiller_power_kw * cfg["chiller_ratio"]

        # 미처리 냉각 부하 → 서버 온도 상승
        unmet_cooling = cooling_load * (1.0 - cfg["chiller_ratio"])
        delta_t_unmet = unmet_cooling / (m_air * AIR_SPECIFIC_HEAT_KJ_PER_KG_K)

        pue_result  = calculate_pue(it_power_kw, actual_chiller_power)
        delta_t     = cooling_load / (m_air * AIR_SPECIFIC_HEAT_KJ_PER_KG_K)
        return_temp = supply_temp_c + delta_t + delta_t_unmet  # 환기 온도

        rows.append({
            "시간":           f"{hour:02d}:00",
            "외기온도 (°C)":  round(outdoor_temp, 1),
            "CPU 사용률 (%)": round(util * 100, 1),
            "IT 전력 (kW)":   round(it_power_kw, 1),
            "냉각 부하 (kW)": round(cooling_load, 1),
            "칠러 전력 (kW)": round(actual_chiller_power, 1),
            "총 전력 (kW)":   round(pue_result.total_power_kw, 1),
            "환기 온도 (°C)": round(return_temp, 1),
            "COP":            round(chiller.cop, 2),
            "PUE":            round(pue_result.pue, 3),
            "냉각 모드":      chiller.cooling_mode.value,
        })

    return pd.DataFrame(rows)


def calculate_esg(df: pd.DataFrame) -> dict:
    """24시간 시뮬레이션 결과로 ESG 지표를 계산한다."""
    total_energy_kwh = df["총 전력 (kW)"].sum()   # 시간당 → 24h 합계 (kWh)
    it_energy_kwh    = df["IT 전력 (kW)"].sum()

    carbon_tco2_day = total_energy_kwh * CARBON_FACTOR_TCO2_PER_MWH / 1000
    cost_krw_day    = total_energy_kwh * ELECTRICITY_COST_KRW_PER_KWH

    water_l_day = sum(
        row["IT 전력 (kW)"] * WUE_BY_MODE.get(row["냉각 모드"], 1.0)
        for _, row in df.iterrows()
    )
    wue = water_l_day / it_energy_kwh if it_energy_kwh > 0 else 0.0

    return {
        "carbon_tco2_day":   round(carbon_tco2_day, 3),
        "carbon_tco2_month": round(carbon_tco2_day * 30, 1),
        "cost_krw_day":      round(cost_krw_day / 1e6, 2),   # 만원 단위
        "wue":               round(wue, 2),
        "water_l_day":       round(water_l_day, 0),
    }


def simulate_rack_temperatures(peak_return_temp: float) -> tuple[list, list, list]:
    """피크 환기 온도 기준으로 랙별 온도 분포를 시뮬레이션한다."""
    random.seed(42)
    temps  = [round(peak_return_temp + random.gauss(0, 1.5), 1) for _ in range(NUM_RACKS)]
    labels = [f"R{i+1:02d}" for i in range(NUM_RACKS)]
    colors = [
        "#E74C3C" if t > TEMP_WARNING_THRESHOLD_C else
        "#F39C12" if t > TEMP_WARNING_THRESHOLD_C - 2 else
        "#2ECC71"
        for t in temps
    ]
    return labels, temps, colors


# ════════════════════════════════════════════════════════════════════════════
# 차트 빌더
# ════════════════════════════════════════════════════════════════════════════

def build_pue_gauge(avg_pue: float) -> go.Figure:
    color = "#2ECC71" if avg_pue < 1.4 else ("#F39C12" if avg_pue < 1.8 else "#E74C3C")
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=avg_pue,
        title={"text": "PUE (24h 평균)", "font": {"size": 14}},
        delta={"reference": NAVER_PUE_BENCHMARK, "valueformat": ".3f"},
        number={"valueformat": ".3f"},
        gauge={
            "axis":      {"range": [1.0, 2.5], "tickformat": ".1f"},
            "bar":       {"color": color},
            "steps":     PUE_GAUGE_STEPS,
            "threshold": {
                "line": {"color": "gray", "width": 2},
                "thickness": 0.75,
                "value": NAVER_PUE_BENCHMARK,
            },
        },
    ))
    fig.update_layout(height=240, margin=dict(t=50, b=10, l=20, r=20))
    return fig


def build_power_trend(df: pd.DataFrame) -> go.Figure:
    hours = list(range(24))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hours, y=df["IT 전력 (kW)"],
        name="IT 전력", fill="tozeroy", line=dict(color="#4C9BE8"),
    ))
    fig.add_trace(go.Scatter(
        x=hours, y=df["칠러 전력 (kW)"],
        name="칠러 전력", fill="tozeroy", line=dict(color="#F4845F"),
    ))
    fig.add_trace(go.Scatter(
        x=hours, y=df["총 전력 (kW)"],
        name="총 전력", line=dict(color="#2C3E50", dash="dot", width=2),
    ))
    fig.update_layout(
        title="전력 소비 추이 (kW)",
        height=280,
        margin=dict(t=40, b=40, l=10, r=10),
        xaxis_title="시간 (h)",
        yaxis_title="전력 (kW)",
        legend=dict(orientation="h", y=-0.3),
    )
    return fig


def build_rack_temp_chart(
    labels: list, temps: list, colors: list,
    y_min: float, y_max: float,
) -> go.Figure:
    fig = go.Figure(go.Bar(x=labels, y=temps, marker_color=colors))
    fig.add_hline(
        y=TEMP_WARNING_THRESHOLD_C,
        line_dash="dash", line_color="red",
        annotation_text=f"경고 {TEMP_WARNING_THRESHOLD_C}°C",
    )
    fig.update_layout(
        title="서버 온도 분포 (피크 시간)",
        height=280,
        margin=dict(t=40, b=40, l=10, r=10),
        xaxis_title="랙",
        yaxis_title="온도 (°C)",
        yaxis=dict(range=[y_min, y_max]),
    )
    return fig


def render_ctrl_recommendation(result: dict, label: str) -> None:
    """제어 서비스 추천값을 렌더링한다. 오프라인이면 경고 표시."""
    if "error" in result:
        st.warning(f"{label}: 서비스 오프라인")
    else:
        mode_ko = COOLING_MODE_LABELS.get(result.get("cooling_mode", ""), result.get("cooling_mode", "-"))
        ratio   = result.get("free_cooling_ratio", 0.0)
        st.write(
            f"**{label}** — 모드: `{mode_ko}` | "
            f"설정 온도: `{result.get('supply_air_temp_setpoint_c', '-')}°C` | "
            f"Free Cooling 비율: `{ratio:.0%}`"
        )


def build_pue_trend(df: pd.DataFrame) -> go.Figure:
    hours = list(range(24))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hours, y=df["PUE"],
        line=dict(color="#2ECC71", width=2), mode="lines+markers",
    ))
    fig.add_hline(
        y=NAVER_PUE_BENCHMARK, line_dash="dash", line_color="gray",
        annotation_text=f"{NAVER_PUE_BENCHMARK} (NAVER)",
    )
    fig.update_layout(
        height=240,
        margin=dict(t=20, b=40, l=10, r=10),
        xaxis_title="시간 (h)",
        yaxis_title="PUE",
        showlegend=False,
    )
    return fig


# ════════════════════════════════════════════════════════════════════════════
# 알람 관리
# ════════════════════════════════════════════════════════════════════════════

def add_alarm(level: str, message: str) -> None:
    """알람을 세션 이력에 추가한다."""
    st.session_state.alarms.insert(0, {
        "time":    datetime.now().strftime("%H:%M:%S"),
        "level":   level,
        "message": message,
    })
    if len(st.session_state.alarms) > 15:
        st.session_state.alarms.pop()


# ════════════════════════════════════════════════════════════════════════════
# 페이지 & 세션 초기화
# ════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="AI Green IDC Dashboard",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "crisis_mode" not in st.session_state:
    st.session_state.crisis_mode = None
if "alarms" not in st.session_state:
    st.session_state.alarms = []
if "prev_crisis" not in st.session_state:
    st.session_state.prev_crisis = None


# ════════════════════════════════════════════════════════════════════════════
# 사이드바 — 파라미터 입력
# ════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.header("⚙️ 시뮬레이션 파라미터")

    scenario = st.selectbox("시나리오", list(SCENARIO_TEMP_PROFILES.keys()), index=0)

    st.divider()
    st.subheader("서버 구성")
    num_cpu_servers = st.slider("CPU 서버 대수", 100, 1000, 400, step=50)
    num_gpu_servers = st.slider("GPU 서버 대수", 0, 200, 20, step=10)
    cpu_utilization = st.slider("평균 CPU 사용률 (%)", 10, 100, 60) / 100.0

    st.divider()
    st.subheader("냉각 설정")
    supply_temp = st.slider("CRAH 공급 온도 (°C)", 14, 24, 18)

    st.divider()
    st.subheader("🚨 위기 시나리오 주입")

    # 명세서 기준: 서버급증 / 냉각기고장 / 폭염 3가지
    c0, c1, c2, c3 = st.columns(4)
    if c0.button("정상",     use_container_width=True):
        st.session_state.crisis_mode = None
    if c1.button("서버급증", use_container_width=True, type="primary"):
        st.session_state.crisis_mode = "server_surge"
    if c2.button("냉각고장", use_container_width=True, type="primary"):
        st.session_state.crisis_mode = "chiller_failure"
    if c3.button("폭염",     use_container_width=True, type="primary"):
        st.session_state.crisis_mode = "heat_wave"

    crisis_mode = st.session_state.crisis_mode
    cfg         = CRISIS_CONFIGS[crisis_mode]

    if crisis_mode:
        st.error(f"🔴 위기 모드: {cfg['label']}")
    else:
        st.success("🟢 정상 운영 중")

    st.divider()
    st.subheader("🔌 서비스 상태")
    for name, is_ok in get_all_service_status().items():
        st.write(f"{'🟢' if is_ok else '🔴'} {name}")


# ════════════════════════════════════════════════════════════════════════════
# 데이터 계산
# ════════════════════════════════════════════════════════════════════════════

df = run_simulation(
    scenario, num_cpu_servers, num_gpu_servers,
    cpu_utilization, supply_temp, crisis_mode,
)

avg_pue     = df["PUE"].mean()
avg_cop     = df["COP"].mean()
avg_it      = df["IT 전력 (kW)"].mean()
avg_cooling = df["칠러 전력 (kW)"].mean()
avg_total   = df["총 전력 (kW)"].mean()

peak_idx         = df["IT 전력 (kW)"].idxmax()
peak_return_temp = df.loc[peak_idx, "환기 온도 (°C)"]
max_return_temp  = df["환기 온도 (°C)"].max()
current_mode     = df.loc[peak_idx, "냉각 모드"]
current_outdoor  = df.loc[peak_idx, "외기온도 (°C)"]

rack_labels, rack_temps, rack_colors = simulate_rack_temperatures(peak_return_temp)
over_threshold = sum(1 for t in rack_temps if t > TEMP_WARNING_THRESHOLD_C)

esg = calculate_esg(df)

# Control Service 호출 — 피크 시간 기준 외기온도 + IT 전력으로 제어 추천 요청
peak_it_power = df.loc[peak_idx, "IT 전력 (kW)"]
ctrl_rule = optimize_control(current_outdoor, peak_it_power)
ctrl_rl   = rl_control(current_outdoor, peak_it_power)

# 위기 모드 전환 시 알람 생성
if crisis_mode != st.session_state.prev_crisis:
    if crisis_mode:
        add_alarm("WARN", f"위기 시나리오 주입: {cfg['label']}")
    else:
        add_alarm("INFO", "정상 모드로 복귀")
    st.session_state.prev_crisis = crisis_mode

# 온도 경고 알람
if max_return_temp > TEMP_WARNING_THRESHOLD_C:
    add_alarm("ERROR", f"서버 온도 경고: 최고 {max_return_temp:.1f}°C (기준 {TEMP_WARNING_THRESHOLD_C}°C 초과)")


# ════════════════════════════════════════════════════════════════════════════
# UI 렌더링
# ════════════════════════════════════════════════════════════════════════════

# ── 헤더 ─────────────────────────────────────────────────────────────────

col_title, col_mode = st.columns([4, 1])
with col_title:
    st.title("🌿 AI Green IDC — 통합 관제 대시보드")
    st.caption("열역학 기반 냉각 부하 시뮬레이션 · PUE/COP/ESG 최적화")
with col_mode:
    st.markdown("<br>", unsafe_allow_html=True)
    if crisis_mode:
        st.error(f"🔴 {cfg['label']}")
    else:
        st.success("🟢 AUTO MODE")

# ── 온도 경고 배너 ────────────────────────────────────────────────────────

if max_return_temp > TEMP_WARNING_THRESHOLD_C:
    st.error(
        f"⚠️ **서버 온도 경고**: 최고 **{max_return_temp:.1f}°C** "
        f"(기준 {TEMP_WARNING_THRESHOLD_C}°C 초과) | "
        f"경고 랙 **{over_threshold}개** — 즉시 냉각 강화 필요"
    )

st.divider()

# ── Row 1: PUE 게이지 | 전력 소비 추이 | 서버 온도 분포 ─────────────────

col_gauge, col_power, col_temp = st.columns(3)

with col_gauge:
    st.plotly_chart(build_pue_gauge(avg_pue), width="stretch")

with col_power:
    st.plotly_chart(build_power_trend(df), width="stretch")

with col_temp:
    fig_rack = build_rack_temp_chart(
        rack_labels, rack_temps, rack_colors,
        y_min=supply_temp - 2, y_max=max(rack_temps) + 3,
    )
    st.plotly_chart(fig_rack, width="stretch")

st.divider()

# ── Row 2: 냉각 모드 | ESG 지표 | 알람 이력 ──────────────────────────────

col_cooling, col_esg, col_alarm = st.columns(3)

with col_cooling:
    st.subheader("❄️ 냉각 모드")
    mode_label = COOLING_MODE_LABELS.get(current_mode, current_mode)
    free_cool_ok = current_outdoor < 15.0
    st.metric("현재 모드",    mode_label)
    st.metric("외기 온도",    f"{current_outdoor} °C")
    st.metric("공급 온도",    f"{supply_temp} °C")
    st.metric("피크 환기온도", f"{peak_return_temp:.1f} °C")
    if free_cool_ok:
        st.success("✅ Free Cooling 가용")
    else:
        st.info("ℹ️ Free Cooling 불가 (외기 > 15°C)")

    # 냉각 모드 분포 (도넛 차트)
    mode_counts = df["냉각 모드"].value_counts()
    fig_pie = go.Figure(go.Pie(
        labels=[COOLING_MODE_LABELS.get(m, m) for m in mode_counts.index],
        values=mode_counts.values,
        marker=dict(colors=COOLING_MODE_COLORS),
        textinfo="label+percent",
        hole=0.5,
    ))
    fig_pie.update_layout(
        height=200,
        margin=dict(t=10, b=10, l=10, r=10),
        showlegend=False,
    )
    st.plotly_chart(fig_pie, width="stretch")

    # ── Control Service 추천값 ───────────────────────────────────────────
    st.markdown(f"**🤖 제어 서비스 추천** (피크 기준: 외기 {current_outdoor}°C / IT {peak_it_power:.0f}kW)")
    render_ctrl_recommendation(ctrl_rule, "Rule-Based")
    render_ctrl_recommendation(ctrl_rl,   "RL Agent")

with col_esg:
    st.subheader("🌱 ESG 지표")
    e1, e2 = st.columns(2)
    e1.metric("탄소 배출 (일)",  f"{esg['carbon_tco2_day']:.3f} tCO₂")
    e2.metric("탄소 배출 (월)",  f"{esg['carbon_tco2_month']:.1f} tCO₂")
    e3, e4 = st.columns(2)
    e3.metric("WUE",             f"{esg['wue']:.2f} L/kWh")
    e4.metric("물 사용량 (일)",   f"{esg['water_l_day']:.0f} L")
    e5, e6 = st.columns(2)
    e5.metric("전력 비용 (일)",  f"{esg['cost_krw_day']:.1f} 만원")
    e6.metric("전력 비용 (월)",  f"{esg['cost_krw_day'] * 30:.0f} 만원")

    # 비용-에너지-탄소 Trade-off 바 차트
    st.caption("냉각 방식별 탄소 배출 (kgCO₂/h)")
    _MODE_COLOR_MAP = {"chiller": "#E74C3C", "hybrid": "#F39C12", "free_cooling": "#2ECC71"}
    mode_carbon = df.groupby("냉각 모드").apply(
        lambda g: (g["총 전력 (kW)"].mean() * CARBON_FACTOR_TCO2_PER_MWH)
    ).reset_index()
    mode_carbon.columns = ["모드_key", "탄소(kgCO₂/h)"]
    mode_carbon["색상"] = mode_carbon["모드_key"].map(_MODE_COLOR_MAP)
    mode_carbon["모드"] = mode_carbon["모드_key"].map(COOLING_MODE_LABELS)
    fig_carbon = go.Figure(go.Bar(
        x=mode_carbon["모드"],
        y=mode_carbon["탄소(kgCO₂/h)"],
        marker_color=mode_carbon["색상"],
    ))
    fig_carbon.update_layout(
        height=160,
        margin=dict(t=10, b=30, l=10, r=10),
        yaxis_title="kgCO₂/h",
        showlegend=False,
    )
    st.plotly_chart(fig_carbon, width="stretch")

with col_alarm:
    st.subheader("🔔 알람 이력")
    if not st.session_state.alarms:
        st.info("알람 없음")
    else:
        for alarm in st.session_state.alarms[:10]:
            icon = {"ERROR": "🔴", "WARN": "🟡", "INFO": "🔵"}.get(alarm["level"], "⚪")
            st.write(f"{icon} `{alarm['time']}` {alarm['message']}")

    if st.button("알람 초기화"):
        st.session_state.alarms = []
        st.rerun()

st.divider()

# ── Row 3: KPI 카드 ───────────────────────────────────────────────────────

st.subheader("📊 24시간 평균 KPI")
k1, k2, k3, k4, k5 = st.columns(5)

pue_delta = f"{((avg_pue - NAVER_PUE_BENCHMARK) / NAVER_PUE_BENCHMARK * 100):+.1f}% vs NAVER"
k1.metric("PUE",      f"{avg_pue:.3f}",    pue_delta, delta_color="inverse")
k2.metric("COP",      f"{avg_cop:.2f}")
k3.metric("IT 전력",  f"{avg_it:.0f} kW")
k4.metric("칠러 전력", f"{avg_cooling:.0f} kW")
k5.metric("총 전력",  f"{avg_total:.0f} kW")

st.divider()

# ── Row 4: PUE 추이 | 상세 테이블 ────────────────────────────────────────

col_pue, col_table = st.columns([1, 2])

with col_pue:
    st.subheader("PUE 추이")
    st.plotly_chart(build_pue_trend(df), width="stretch")

    # PUE 벤치마크 참고
    st.caption("**PUE 벤치마크**")
    for label, val in PUE_BENCHMARKS:
        st.write(f"- {label}: **{val}**")

with col_table:
    st.subheader("🗂️ 시간별 상세 데이터")
    st.dataframe(
        df.style.format({"PUE": "{:.3f}", "COP": "{:.2f}"}),
        width="stretch",
        height=320,
    )
