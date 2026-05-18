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
    CLR_DANGER,
    CLR_GOOD,
    CLR_TOTAL,
    ELECTRICITY_COST_KRW_PER_KWH,
)
from apps.dashboard.sidebar import render_sidebar

d = render_sidebar()

st.title(":material/eco: ESG 지표")
st.caption(
    "탄소 배출 · 전력 비용 · CUE 분석 — "
    f"계수: {CARBON_FACTOR_TCO2_PER_MWH} tCO₂/MWh (한국 전력) · "
    f"{ELECTRICITY_COST_KRW_PER_KWH} 원/kWh (산업용)"
)
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

df    = d["df"]
hours = list(range(len(df)))

col_carbon, col_cost = st.columns(2)

with col_carbon:
    with st.container(border=True):
        st.subheader("시간별 탄소 배출 누적")
        hourly_carbon = df["총 전력 (kW)"] * CARBON_FACTOR_TCO2_PER_MWH / 1000  # tCO₂/h
        cumulative    = hourly_carbon.cumsum()

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
        hourly_cost_man = df["총 전력 (kW)"] * ELECTRICITY_COST_KRW_PER_KWH / 1e4  # 만원/h (1만원 = 1e4원)
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
    st.plotly_chart(build_carbon_bar(df), width="stretch")

st.divider()

# ── ROI 계산기 ────────────────────────────────────────────────────────────

st.subheader("ROI 계산기 — 최적화 도입 시 회수 분석")
st.caption(
    "현재 운영 비용 기준으로, 냉각 최적화(RL 등) 도입 시 연간 절감액과 "
    "투자 회수 기간을 추정한다."
)

# 현재 연간 비용/탄소 (월 × 12)
annual_cost_man    = esg["cost_krw_month"]   * 12    # 만원/년
annual_carbon_tco2 = esg["carbon_tco2_month"] * 12   # tCO₂/년

roi_left, roi_right = st.columns([1, 1.4])

with roi_left:
    with st.container(border=True):
        st.markdown("**입력**")
        capex_man = st.number_input(
            "도입 비용 (CapEx, 만원)",
            min_value=0, value=5000, step=500,
            help="냉각 최적화 시스템(RL 컨트롤러·센서·서버 등) 초기 투자비",
        )
        savings_pct = st.slider(
            "예상 전력 절감률 (%)",
            min_value=0.0, max_value=30.0, value=10.0, step=0.5,
            help="최적화 도입 시 총 전력 소비 감소 비율. 일반적 RL 개선폭은 5~15%.",
        )
        opex_man_per_year = st.number_input(
            "연간 운영비 (OpEx, 만원/년)",
            min_value=0, value=200, step=50,
            help="유지보수·라이선스·재학습 등 연간 발생 비용",
        )

with roi_right:
    with st.container(border=True):
        st.markdown("**연간 효과**")
        savings_ratio = savings_pct / 100.0
        annual_saving_man    = annual_cost_man * savings_ratio - opex_man_per_year
        annual_saving_carbon = annual_carbon_tco2 * savings_ratio

        if annual_saving_man <= 0:
            payback_years = float("inf")
            payback_label = "회수 불가 (절감액 ≤ 운영비)"
        else:
            payback_years = capex_man / annual_saving_man
            if payback_years >= 100:
                payback_label = f"{payback_years:.0f}년"
            elif payback_years >= 1:
                payback_label = f"{payback_years:.1f}년"
            else:
                payback_label = f"{payback_years * 12:.1f}개월"

        r1, r2, r3 = st.columns(3)
        r1.metric("연 절감 비용",     f"{annual_saving_man:+,.0f} 만원",
                  help="연간 전력비용 절감 − 연간 운영비")
        r2.metric("연 절감 탄소",     f"{annual_saving_carbon:.1f} tCO₂")
        r3.metric("투자 회수 기간",   payback_label,
                  help="CapEx ÷ 연 절감 비용")

        # 5년 누적 차트 — 누적 절감액이 CapEx를 넘어서는 시점 시각화
        years     = list(range(6))
        cum_saving_man = [annual_saving_man * y for y in years]
        capex_line     = [capex_man] * len(years)

        fig_roi = go.Figure()
        fig_roi.add_trace(go.Scatter(
            x=years, y=cum_saving_man,
            name="누적 절감액",
            line=dict(color=CLR_GOOD, width=2), mode="lines+markers",
            fill="tozeroy", fillcolor="rgba(46,204,113,0.12)",
        ))
        fig_roi.add_trace(go.Scatter(
            x=years, y=capex_line,
            name="CapEx (도입비)",
            line=dict(color=CLR_DANGER, width=2, dash="dash"),
            mode="lines",
        ))
        if 0 < payback_years <= max(years):
            fig_roi.add_vline(
                x=payback_years, line_color=CLR_DANGER, line_width=1.5,
                annotation_text=f"회수 시점 {payback_label}",
                annotation_position="top right",
            )
        fig_roi.update_layout(
            height=240,
            margin=dict(t=10, b=40, l=10, r=10),
            xaxis=dict(title="연차 (year)", dtick=1),
            yaxis_title="누적 (만원)",
            legend=dict(orientation="h", y=-0.3),
        )
        st.plotly_chart(fig_roi, width="stretch")

st.caption(
    f"**계산 기준** — 현재 연 비용 {annual_cost_man:,.0f} 만원 · "
    f"연 탄소 {annual_carbon_tco2:.1f} tCO₂ (월간 값을 12개월 단순 환산)"
)
