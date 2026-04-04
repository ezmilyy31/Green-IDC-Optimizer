"""
AI Green IDC 대시보드 — 메인

핵심 KPI · PUE 게이지 · 전력 추이 · 서버 랙 온도 · PUE 추이 · 시간별 데이터
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import streamlit as st

from apps.dashboard.charts import (
    build_pue_gauge,
    build_pue_trend,
    build_power_trend,
    build_rack_temp_chart,
    render_pue_benchmarks,
)
from apps.dashboard.constants import NAVER_PUE_BENCHMARK, TEMP_WARNING_THRESHOLD_C
from apps.dashboard.sidebar import render_sidebar

d = render_sidebar()

# ── 헤더 ──────────────────────────────────────────────────────────────────

st.title(":material/dashboard: AI Green IDC — 통합 관제")
st.caption("열역학 기반 냉각 부하 시뮬레이션 · PUE / COP / ESG 최적화 · 24시간 기준")

if d["crisis_mode"]:
    st.error(f"⚠️ 위기 모드 활성: {d['cfg']['label']}")
else:
    st.success("정상 운영 중")

if d["max_return_temp"] > TEMP_WARNING_THRESHOLD_C:
    st.error(
        f"🔴 서버 온도 경고: 최고 **{d['max_return_temp']:.1f}°C** "
        f"(기준 {TEMP_WARNING_THRESHOLD_C}°C 초과) | "
        f"경고 랙 **{d['over_threshold']}개** — 즉시 냉각 강화 필요"
    )

st.divider()

# ── KPI 카드 ──────────────────────────────────────────────────────────────

pue_delta = f"{((d['avg_pue'] - NAVER_PUE_BENCHMARK) / NAVER_PUE_BENCHMARK * 100):+.1f}% vs NAVER"
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("PUE",        f"{d['avg_pue']:.3f}",    pue_delta, delta_color="inverse")
k2.metric("COP",        f"{d['avg_cop']:.2f}")
k3.metric("IT 전력",    f"{d['avg_it']:.0f} kW")
k4.metric("탄소 (일)",  f"{d['esg']['carbon_tco2_day']:.3f} tCO₂")
k5.metric("비용 (일)",  f"{d['esg']['cost_krw_day']:.1f} 만원")

st.markdown("<br>", unsafe_allow_html=True)

# ── Row 1: PUE 게이지 | 전력 추이 | 랙 온도 ─────────────────────────────

col_gauge, col_power, col_rack = st.columns(3)

with col_gauge:
    with st.container(border=True):
        st.plotly_chart(build_pue_gauge(d["avg_pue"]), width="stretch")

with col_power:
    with st.container(border=True):
        st.plotly_chart(build_power_trend(d["df"]), width="stretch")

with col_rack:
    with st.container(border=True):
        st.plotly_chart(
            build_rack_temp_chart(
                d["rack_labels"], d["rack_temps"], d["rack_colors"],
                y_min=d["supply_temp"] - 2,
                y_max=max(d["rack_temps"]) + 3,
            ),
            width="stretch",
        )

# ── Row 2: PUE 추이 | 시간별 테이블 ─────────────────────────────────────

col_pue, col_table = st.columns([1, 2])

with col_pue:
    with st.container(border=True):
        st.subheader("PUE 추이")
        st.plotly_chart(build_pue_trend(d["df"]), width="stretch")
        render_pue_benchmarks(d["avg_pue"])

with col_table:
    with st.container(border=True):
        st.subheader("시간별 상세 데이터")
        st.dataframe(
            d["df"].style.format({"PUE": "{:.3f}", "COP": "{:.2f}"}),
            width="stretch",
            height=635,
        )
