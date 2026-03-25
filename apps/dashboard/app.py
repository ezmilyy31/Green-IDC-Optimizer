"""
AI Green IDC Dashboard

열역학 기반 시뮬레이션 결과와 서비스 상태를 시각화한다.
domain 모듈을 직접 import해서 HTTP 서비스 없이도 동작한다.
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from apps.dashboard.api_client import get_all_service_status
from domain.thermodynamics.chiller import calculate_chiller_power_kw
from domain.thermodynamics.cooling_load import calculate_cooling_load_from_it_power_kw
from domain.thermodynamics.it_power import calculate_total_it_power_kw
from domain.thermodynamics.pue import calculate_pue

# ── 페이지 설정 ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AI Green IDC Dashboard",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🌿 AI Green IDC Dashboard")
st.caption("열역학 기반 냉각 부하 시뮬레이션 · PUE/COP 최적화")

# ── 사이드바: 파라미터 설정 ───────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ 시뮬레이션 파라미터")

    scenario = st.selectbox(
        "시나리오",
        ["여름 (Summer)", "봄/가을 (Spring)", "겨울 (Winter)"],
        index=0,
    )

    st.divider()
    st.subheader("서버 구성")
    num_cpu_servers = st.slider("CPU 서버 대수", 100, 1000, 400, step=50)
    num_gpu_servers = st.slider("GPU 서버 대수", 0, 200, 20, step=10)
    cpu_utilization = st.slider("평균 CPU 사용률 (%)", 10, 100, 60) / 100.0

    st.divider()
    st.subheader("냉각 설정")
    supply_temp = st.slider("CRAH 공급 온도 설정값 (°C)", 14, 24, 18)

# ── 시뮬레이션 로직 ──────────────────────────────────────────────────────────

SCENARIO_TEMP_PROFILES = {
    "여름 (Summer)":   {"base": 30.0, "amplitude": 6.0,  "offset": 0},
    "봄/가을 (Spring)": {"base": 15.0, "amplitude": 8.0,  "offset": 0},
    "겨울 (Winter)":   {"base": 2.0,  "amplitude": 6.0,  "offset": 0},
}

WORKLOAD_PROFILE = [
    0.35, 0.30, 0.28, 0.27, 0.28, 0.32,   # 00~05시: 야간 저부하
    0.45, 0.60, 0.75, 0.85, 0.90, 0.92,   # 06~11시: 오전 증가
    0.90, 0.88, 0.92, 0.93, 0.88, 0.82,   # 12~17시: 주간 피크
    0.72, 0.65, 0.58, 0.50, 0.42, 0.38,   # 18~23시: 야간 감소
]


def run_simulation(
    scenario_name: str,
    num_cpu: int,
    num_gpu: int,
    base_util: float,
    supply_temp_c: float,
) -> pd.DataFrame:
    profile = SCENARIO_TEMP_PROFILES[scenario_name]
    rows = []

    for hour in range(24):
        # 외기 온도: 새벽 4시 최저, 오후 2시 최고
        outdoor_temp = (
            profile["base"]
            + profile["amplitude"] * math.sin(math.radians((hour - 4) * 15))
        )
        # IT 부하: 시간대별 워크로드 패턴 반영
        util = min(1.0, base_util * WORKLOAD_PROFILE[hour] / 0.6)
        it_power_kw = calculate_total_it_power_kw(util, num_cpu, num_gpu)

        cooling_load_kw = calculate_cooling_load_from_it_power_kw(it_power_kw)
        chiller = calculate_chiller_power_kw(cooling_load_kw, outdoor_temp)
        pue_result = calculate_pue(it_power_kw, chiller.chiller_power_kw)

        rows.append(
            {
                "시간": f"{hour:02d}:00",
                "외기온도 (°C)": round(outdoor_temp, 1),
                "IT 전력 (kW)": round(it_power_kw, 1),
                "냉각 부하 (kW)": round(cooling_load_kw, 1),
                "칠러 전력 (kW)": round(chiller.chiller_power_kw, 1),
                "COP": round(chiller.cop, 2),
                "PUE": round(pue_result.pue, 3),
                "냉각 모드": chiller.cooling_mode.value,
                "총 전력 (kW)": round(pue_result.total_power_kw, 1),
            }
        )

    return pd.DataFrame(rows)


df = run_simulation(scenario, num_cpu_servers, num_gpu_servers, cpu_utilization, supply_temp)

# ── Row 1: KPI 카드 ──────────────────────────────────────────────────────────

st.subheader("📊 주요 지표 (24시간 평균)")
k1, k2, k3, k4, k5 = st.columns(5)

avg_pue = df["PUE"].mean()
avg_cop = df["COP"].mean()
avg_it = df["IT 전력 (kW)"].mean()
avg_cooling = df["칠러 전력 (kW)"].mean()
avg_total = df["총 전력 (kW)"].mean()

pue_delta = f"{((avg_pue - 1.09) / 1.09 * 100):+.1f}% vs NAVER 각 춘천(1.09)"

k1.metric("PUE", f"{avg_pue:.3f}", pue_delta, delta_color="inverse")
k2.metric("COP (칠러 효율)", f"{avg_cop:.2f}")
k3.metric("IT 전력", f"{avg_it:.0f} kW")
k4.metric("칠러 전력", f"{avg_cooling:.0f} kW")
k5.metric("총 전력", f"{avg_total:.0f} kW")

st.divider()

# ── Row 2: 시계열 차트 ───────────────────────────────────────────────────────

st.subheader("📈 24시간 시뮬레이션 결과")

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        "전력 분포 (kW)",
        "PUE 추이",
        "외기 온도 & COP",
        "냉각 모드별 비율",
    ),
    specs=[
        [{"type": "xy"}, {"type": "xy"}],
        [{"type": "xy"}, {"type": "domain"}],
    ],
    vertical_spacing=0.22,
    horizontal_spacing=0.12,
)

hours = list(range(24))

# 전력 분포 (stacked area)
fig.add_trace(
    go.Scatter(
        x=hours, y=df["IT 전력 (kW)"], name="IT 전력",
        fill="tozeroy", line=dict(color="#4C9BE8"),
    ),
    row=1, col=1,
)
fig.add_trace(
    go.Scatter(
        x=hours, y=df["칠러 전력 (kW)"], name="칠러 전력",
        fill="tozeroy", line=dict(color="#F4845F"),
    ),
    row=1, col=1,
)

# PUE 추이
fig.add_trace(
    go.Scatter(
        x=hours, y=df["PUE"], name="PUE",
        line=dict(color="#2ECC71", width=2),
        mode="lines+markers",
    ),
    row=1, col=2,
)
# NAVER 기준선
fig.add_hline(
    y=1.09, line_dash="dash", line_color="gray",
    annotation_text="NAVER 각 춘천 1.09", row=1, col=2,
)

# 외기 온도 & COP (이중 축)
fig.add_trace(
    go.Scatter(
        x=hours, y=df["외기온도 (°C)"], name="외기 온도 (°C)",
        line=dict(color="#E74C3C"), yaxis="y3",
    ),
    row=2, col=1,
)
fig.add_trace(
    go.Scatter(
        x=hours, y=df["COP"], name="COP",
        line=dict(color="#9B59B6", dash="dot"),
    ),
    row=2, col=1,
)

# 냉각 모드 비율 (파이 차트)
mode_counts = df["냉각 모드"].value_counts()
mode_labels = {"chiller": "기계식", "free_cooling": "자연공조", "hybrid": "혼합"}
fig.add_trace(
    go.Pie(
        labels=[mode_labels.get(m, m) for m in mode_counts.index],
        values=mode_counts.values,
        name="냉각 모드",
        marker=dict(colors=["#E74C3C", "#2ECC71", "#F39C12"]),
        textinfo="label+percent",
    ),
    row=2, col=2,
)

fig.update_layout(
    height=720,
    showlegend=True,
    legend=dict(orientation="h", y=-0.12, x=0.5, xanchor="center"),
    margin=dict(t=60, b=80),
)
fig.update_xaxes(title_text="시간 (h)", row=1, col=1)
fig.update_xaxes(title_text="시간 (h)", row=1, col=2)
fig.update_xaxes(title_text="시간 (h)", row=2, col=1)

st.plotly_chart(fig, width='stretch')

st.divider()

# ── Row 3: 상세 데이터 테이블 + 서비스 상태 ─────────────────────────────────

col_table, col_status = st.columns([3, 1])

with col_table:
    st.subheader("🗂️ 시간별 상세 데이터")
    st.dataframe(
        df.style.format({"PUE": "{:.3f}", "COP": "{:.2f}"}),
        width='stretch',
        height=320,
    )

with col_status:
    st.subheader("🔌 서비스 상태")
    statuses = get_all_service_status()
    for name, is_ok in statuses.items():
        icon = "🟢" if is_ok else "🔴"
        st.write(f"{icon} **{name}**")

    st.divider()
    st.caption("**PUE 기준값**")
    benchmarks = {
        "NAVER 각 춘천": "1.09",
        "Google 글로벌": "1.10",
        "글로벌 평균": "1.58",
        "국내 민간": "2.03",
    }
    for label, val in benchmarks.items():
        st.write(f"- {label}: **{val}**")
