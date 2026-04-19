"""
ESG 지표

탄소 배출 · 전력 비용 · 냉각 방식별 탄소 분석
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import plotly.graph_objects as go
import streamlit as st

from apps.dashboard.charts import build_carbon_bar
from apps.dashboard.constants import (
    CARBON_FACTOR_TCO2_PER_MWH,
    CLR_CARBON,
    CLR_COST,
    CLR_COST_LINE,
    CLR_TOTAL,
    ELECTRICITY_COST_KRW_PER_KWH,
)
from apps.dashboard.sidebar import render_sidebar

d = render_sidebar()

st.title(":material/eco: ESG 지표")
st.caption("탄소 배출 · 전력 비용 · CUE 분석")
st.divider()

esg = d["esg"]

# ── ESG 요약 카드 ─────────────────────────────────────────────────────────

st.subheader("요약")
e1, e2, e3, e4, e5 = st.columns(5)
e1.metric("탄소 배출 (일)",  f"{esg['carbon_tco2_day']:.3f} tCO₂")
e2.metric("탄소 배출 (월)",  f"{esg['carbon_tco2_month']:.1f} tCO₂")
e3.metric("CUE",             f"{esg['cue']:.4f} kgCO₂/kWh")
e4.metric("전력 비용 (일)",  f"{esg['cost_krw_day']:.1f} 만원")
e5.metric("전력 비용 (월)",  f"{esg['cost_krw_month']:.0f} 만원")

st.divider()

# ── 시간별 탄소/비용 누적 차트 ────────────────────────────────────────────

col_carbon, col_cost = st.columns(2)

with col_carbon:
    with st.container(border=True):
        st.subheader("시간별 탄소 배출 누적")
        df = d["df"]
        hourly_carbon = df["총 전력 (kW)"] * CARBON_FACTOR_TCO2_PER_MWH / 1000  # tCO₂/h
        cumulative    = hourly_carbon.cumsum()
        hours         = list(range(len(df)))

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=hours, y=hourly_carbon,
            name="시간별", marker_color=CLR_CARBON,
        ))
        fig.add_trace(go.Scatter(
            x=hours, y=cumulative,
            name="누적", line=dict(color=CLR_TOTAL, width=2),
            yaxis="y2",
        ))
        fig.update_layout(
            height=320, margin=dict(t=20, b=40, l=10, r=60),
            xaxis_title="시간 (h)",
            yaxis=dict(title="tCO₂/h"),
            yaxis2=dict(title="누적 tCO₂", overlaying="y", side="right"),
            legend=dict(orientation="h", y=-0.3),
        )
        st.plotly_chart(fig, width="stretch")

with col_cost:
    with st.container(border=True):
        st.subheader("시간별 전력 비용 누적")
        hourly_cost_man = df["총 전력 (kW)"] * ELECTRICITY_COST_KRW_PER_KWH / 1e6  # 만원/h
        cumulative_cost = hourly_cost_man.cumsum()

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=hours, y=hourly_cost_man,
            name="시간별", marker_color=CLR_COST,
        ))
        fig2.add_trace(go.Scatter(
            x=hours, y=cumulative_cost,
            name="누적", line=dict(color=CLR_COST_LINE, width=2),
            yaxis="y2",
        ))
        fig2.update_layout(
            height=320, margin=dict(t=20, b=40, l=10, r=60),
            xaxis_title="시간 (h)",
            yaxis=dict(title="만원/h"),
            yaxis2=dict(title="누적 만원", overlaying="y", side="right"),
            legend=dict(orientation="h", y=-0.3),
        )
        st.plotly_chart(fig2, width="stretch")

st.divider()

# ── 냉각 방식별 탄소 배출 ─────────────────────────────────────────────────

with st.container(border=True):
    st.subheader("냉각 방식별 탄소 배출 (시간 평균 kgCO₂/h)")
    st.plotly_chart(build_carbon_bar(d["df"]), width="stretch")

st.divider()
st.caption(
    f"**계수 기준** — 탄소 배출계수: {CARBON_FACTOR_TCO2_PER_MWH} tCO₂/MWh (한국 전력) · "
    f"전기요금: {ELECTRICITY_COST_KRW_PER_KWH} 원/kWh (산업용)"
)
