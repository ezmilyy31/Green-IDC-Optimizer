"""
AI Green IDC 대시보드 — 메인

핵심 KPI · PUE 게이지 · 전력 추이 · 서버 랙 온도 · PUE 추이 · 시간별 데이터
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import plotly.graph_objects as go
import streamlit as st

from apps.dashboard.charts import (
    build_pue_gauge,
    build_power_trend,
    render_pue_benchmarks,
)
from apps.dashboard.constants import CLR_GOOD, CLR_TOTAL, NAVER_PUE_BENCHMARK, TEMP_WARNING_THRESHOLD_C
from apps.dashboard.sidebar import render_sidebar
from apps.dashboard.simulation import run_rl_vs_rule

d = render_sidebar()


@st.cache_data(ttl=3600, show_spinner="RL 컨트롤러로 24h 시뮬레이션 중…")
def _rl_compare(scenario: str) -> dict:
    """시즌 키별 RL vs Rule 비교를 캐싱 (DataFrame 직렬화 비용은 1회만)."""
    return run_rl_vs_rule(scenario)


cmp = _rl_compare(d["scenario"])
pue_rule = cmp["rule_pue_mean"]
pue_rl   = cmp["rl_pue_mean"]
savings  = cmp["savings_pct"]

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
        f"(기준 {TEMP_WARNING_THRESHOLD_C}°C 초과) — 즉시 냉각 강화 필요"
    )

st.divider()

# ── KPI 카드 ──────────────────────────────────────────────────────────────

pue_delta = f"{((pue_rule - NAVER_PUE_BENCHMARK) / NAVER_PUE_BENCHMARK * 100):+.1f}% vs NAVER"
k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("PUE (Rule)", f"{pue_rule:.3f}", pue_delta, delta_color="inverse",
          help="규칙기반 컨트롤러 24h 평균. NAVER 각 춘천(1.09) 대비 비교")
k2.metric("PUE (RL)",   f"{pue_rl:.3f}",  f"{savings:+.2f}% vs Rule",
          delta_color="normal" if savings > 0 else "inverse",
          help=f"RL 컨트롤러 24h 평균. 온도 위반: Rule {cmp['rule_viol']}건 / RL {cmp['rl_viol']}건")
k3.metric("COP",        f"{d['avg_cop']:.2f}")
k4.metric("IT 전력",    f"{d['avg_it']:.0f} kW")
k5.metric("탄소 (일)",  f"{d['esg']['carbon_tco2_day']:.3f} tCO₂")
k6.metric("비용 (일)",  f"{d['esg']['cost_krw_day']:.1f} 만원")

st.markdown("<br>", unsafe_allow_html=True)

# ── Row 1: PUE 게이지 | 전력 추이 | PUE 추이 (Rule vs RL) ────────────────

col_gauge, col_power, col_pue = st.columns(3)

with col_gauge:
    with st.container(border=True):
        st.plotly_chart(
            build_pue_gauge(pue_rl, title="PUE — RL (24h 평균)"),
            width="stretch",
        )

with col_power:
    with st.container(border=True):
        st.plotly_chart(build_power_trend(d["df"]), width="stretch")

with col_pue:
    with st.container(border=True):
        # IDCEnv 스텝(5분)에서 시간 단위로 리샘플 → 24개 포인트로 압축
        df_rule_h = cmp["rule"].assign(hour=cmp["rule"]["step"] // 12).groupby("hour")["PUE"].mean()
        df_rl_h   = cmp["rl"  ].assign(hour=cmp["rl"  ]["step"] // 12).groupby("hour")["PUE"].mean()
        fig_pue = go.Figure()
        fig_pue.add_trace(go.Scatter(
            x=df_rule_h.index, y=df_rule_h.values,
            name="Rule-based", line=dict(color=CLR_TOTAL, width=2),
        ))
        fig_pue.add_trace(go.Scatter(
            x=df_rl_h.index, y=df_rl_h.values,
            name="RL", line=dict(color=CLR_GOOD, width=2.5, dash="dot"),
            fill="tonexty", fillcolor="rgba(46,204,113,0.12)",
        ))
        fig_pue.add_hline(y=NAVER_PUE_BENCHMARK, line_dash="dash",
                          line_color="rgba(128,128,128,0.5)",
                          annotation_text="NAVER 1.09",
                          annotation_position="bottom right")
        # legend를 차트 위로 → x축 라벨과 겹치지 않음. height는 옆 카드(게이지/전력추이)와 동일 300.
        fig_pue.update_layout(
            title=dict(text="PUE 추이 — Rule vs RL", x=0.02, font=dict(size=14)),
            height=300, margin=dict(t=55, b=40, l=10, r=10),
            xaxis_title="시간 (h)", yaxis_title="PUE",
            legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1.0),
        )
        st.plotly_chart(fig_pue, width="stretch")

render_pue_benchmarks(pue_rl)
