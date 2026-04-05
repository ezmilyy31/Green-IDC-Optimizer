"""
운영 관리 — 냉각 제어 · 위기 시나리오

냉각 모드 현황 · Rule-Based / RL 제어 추천 · 알람 이력 · 위기 시나리오 비교 분석
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import plotly.graph_objects as go
import streamlit as st

from apps.dashboard.charts import build_cooling_donut
from apps.dashboard.constants import (
    CLR_DANGER,
    CLR_GOOD,
    CLR_IT,
    COOLING_MODE_LABELS,
    CRISIS_STRATEGIES,
)
from core.config.constants import FREE_COOLING_THRESHOLD_C
from apps.dashboard.sidebar import render_sidebar

d = render_sidebar()

# ── 헤더 ──────────────────────────────────────────────────────────────────

st.title(":material/tune: 운영 관리")
st.caption("냉각 제어 현황 · 위기 시나리오 분석")
st.divider()

# ── Section 1: 냉각 제어 ─────────────────────────────────────────────────

mode_label   = COOLING_MODE_LABELS.get(d["current_mode"], d["current_mode"])
free_cool_ok = d["current_outdoor"] < FREE_COOLING_THRESHOLD_C

# Free Cooling 상태 배지
fc_color = CLR_GOOD if free_cool_ok else "#64748b"
fc_label = "Free Cooling 가용" if free_cool_ok else "Free Cooling 불가"
st.markdown(
    f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">'
    f'<span style="font-size:1.1rem;font-weight:700;">냉각 제어</span>'
    f'<span style="border:1.5px solid {fc_color};color:{fc_color};'
    f'padding:2px 10px;border-radius:999px;font-size:0.75rem;font-weight:600;">'
    f'{fc_label}</span>'
    f'</div>',
    unsafe_allow_html=True,
)

col_kpi, col_ctrl, col_alarm = st.columns([2, 2, 1.5])

with col_kpi:
    with st.container(border=True):
        # KPI 2×2
        r1c1, r1c2 = st.columns(2)
        r1c1.metric("현재 모드",     mode_label)
        r1c2.metric("외기 온도",     f"{d['current_outdoor']} °C")
        r2c1, r2c2 = st.columns(2)
        r2c1.metric("공급 온도",     f"{d['supply_temp']} °C")
        r2c2.metric("피크 환기온도", f"{d['peak_return_temp']:.1f} °C")

        st.markdown("<br>", unsafe_allow_html=True)

        # 냉각 모드 도넛
        st.caption("냉각 모드 분포 (24h)")
        st.plotly_chart(build_cooling_donut(d["df"]), width="stretch")

with col_ctrl:
    with st.container(border=True):
        st.markdown(
            f'<p style="font-size:0.85rem;font-weight:700;opacity:0.7;'
            f'text-transform:uppercase;letter-spacing:0.06em;margin-bottom:2px;">'
            f'제어 서비스 추천</p>'
            f'<p style="font-size:0.82rem;opacity:0.55;margin-bottom:14px;">'
            f'피크 기준 — 외기 {d["current_outdoor"]}°C / IT {d["peak_it_power"]:.0f} kW</p>',
            unsafe_allow_html=True,
        )

        def _ctrl_card(result: dict, label: str, accent: str) -> str:
            online_dot = (
                '<span style="display:inline-block;width:6px;height:6px;border-radius:50%;'
                'background:#22c55e;margin-right:4px;vertical-align:middle;"></span>'
                '<span style="font-size:0.68rem;color:#22c55e;font-weight:600;">온라인</span>'
            )
            offline_dot = (
                '<span style="display:inline-block;width:6px;height:6px;border-radius:50%;'
                'background:#ef4444;margin-right:4px;vertical-align:middle;"></span>'
                '<span style="font-size:0.68rem;color:#ef4444;font-weight:600;">오프라인</span>'
            )
            if "error" in result:
                return (
                    f'<div style="background:{accent}0d;border:1px solid {accent}33;'
                    f'border-radius:10px;padding:12px 14px;margin-bottom:10px;">'
                    f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">'
                    f'<span style="font-size:0.78rem;font-weight:700;color:{accent};'
                    f'text-transform:uppercase;letter-spacing:0.05em;">{label}</span>'
                    f'<span style="margin-left:auto;">{offline_dot}</span>'
                    f'</div>'
                    f'<div style="font-size:0.8rem;opacity:0.45;">서비스 연결 불가</div>'
                    f'</div>'
                )
            mode_ko = COOLING_MODE_LABELS.get(result.get("cooling_mode", ""), result.get("cooling_mode", "-"))
            temp    = result.get("supply_air_temp_setpoint_c", "-")
            ratio   = f'{result.get("free_cooling_ratio", 0.0):.0%}'
            metrics = [("냉각 모드", mode_ko), ("설정 온도", f"{temp}°C"), ("Free Cooling", ratio)]
            cells = "".join(
                f'<div>'
                f'<div style="font-size:0.62rem;opacity:0.45;text-transform:uppercase;'
                f'letter-spacing:0.04em;margin-bottom:4px;">{k}</div>'
                f'<div style="font-size:0.95rem;font-weight:700;">{v}</div>'
                f'</div>'
                for k, v in metrics
            )
            return (
                f'<div style="background:{accent}0d;border:1px solid {accent}33;'
                f'border-radius:10px;padding:12px 14px;margin-bottom:10px;">'
                f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:10px;">'
                f'<span style="font-size:0.78rem;font-weight:700;color:{accent};'
                f'text-transform:uppercase;letter-spacing:0.05em;">{label}</span>'
                f'<span style="margin-left:auto;">{online_dot}</span>'
                f'</div>'
                f'<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;">'
                f'{cells}'
                f'</div>'
                f'</div>'
            )

        st.markdown(
            _ctrl_card(d["ctrl_rule"], "Rule-Based", "#5D90FF") +
            _ctrl_card(d["ctrl_rl"],   "RL Agent",   "#7BD2F7"),
            unsafe_allow_html=True,
        )

with col_alarm:
    with st.container(border=True):
        alarm_count = len(st.session_state.alarms)
        st.markdown(
            f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">'
            f'<span style="font-size:0.85rem;font-weight:700;opacity:0.7;'
            f'text-transform:uppercase;letter-spacing:0.06em;">알람 이력</span>'
            f'<span style="background:{"#ef4444" if alarm_count else "#94a3b8"};'
            f'color:#fff;font-size:0.7rem;font-weight:700;'
            f'padding:1px 7px;border-radius:999px;">{alarm_count}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        if not st.session_state.alarms:
            st.markdown(
                '<div style="opacity:0.4;font-size:0.82rem;text-align:center;padding:20px 0;">알람 없음</div>',
                unsafe_allow_html=True,
            )
        else:
            level_bg = {"ERROR": "rgba(239,68,68,0.08)", "WARN": "rgba(243,156,18,0.08)", "INFO": "rgba(76,155,232,0.08)"}
            level_border = {"ERROR": "#ef4444", "WARN": "#F39C12", "INFO": "#4C9BE8"}
            icons = {"ERROR": "🔴", "WARN": "🟡", "INFO": "🔵"}

            for alarm in st.session_state.alarms[:10]:
                lvl = alarm["level"]
                bg  = level_bg.get(lvl, "rgba(128,128,128,0.06)")
                bd  = level_border.get(lvl, "#94a3b8")
                ico = icons.get(lvl, "⚪")
                st.markdown(
                    f'<div style="background:{bg};border-left:2px solid {bd};'
                    f'border-radius:0 6px 6px 0;padding:5px 8px;margin-bottom:4px;">'
                    f'<div style="font-size:0.75rem;opacity:0.5;">{alarm["time"]}</div>'
                    f'<div style="font-size:0.88rem;">{ico} {alarm["message"]}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("알람 초기화", use_container_width=True):
            st.session_state.alarms = []
            st.rerun()

st.divider()

# ── Section 2: 위기 시나리오 ─────────────────────────────────────────────

st.markdown(
    '<span style="font-size:1.1rem;font-weight:700;">위기 시나리오 분석</span>',
    unsafe_allow_html=True,
)

if not d["crisis_mode"]:
    st.markdown(
        '<div style="opacity:0.45;font-size:0.88rem;padding:16px 0;">'
        '사이드바에서 위기 시나리오를 선택하면 정상 대비 영향 분석이 표시됩니다.'
        '</div>',
        unsafe_allow_html=True,
    )
else:
    cfg          = d["cfg"]
    esg          = d["esg"]
    esg_base     = d["esg_base"]
    avg_pue      = d["avg_pue"]
    avg_pue_base = d["avg_pue_base"]
    df           = d["df"]
    df_base      = d["df_base"]

    pue_delta    = avg_pue - avg_pue_base
    cost_delta   = esg["cost_krw_day"]    - esg_base["cost_krw_day"]
    carbon_delta = esg["carbon_tco2_day"] - esg_base["carbon_tco2_day"]
    it_delta_pct = (df["IT 전력 (kW)"].mean() / df_base["IT 전력 (kW)"].mean() - 1) * 100

    # 시나리오 풀 배너
    st.markdown(
        f'<div style="background:rgba(239,68,68,0.08);border:1px solid rgba(239,68,68,0.3);'
        f'border-radius:8px;padding:10px 16px;margin:8px 0 16px;">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;">'
        f'<span style="font-weight:700;color:#ef4444;">⚠️ {cfg["label"]}</span>'
        f'<span style="font-size:0.8rem;opacity:0.65;">'
        f'권장 대응 — {CRISIS_STRATEGIES.get(d["crisis_mode"], "-")}</span>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # KPI 델타 카드
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("PUE 변화",       f"{avg_pue:.3f}",                      f"{pue_delta:+.3f} vs 정상",  delta_color="inverse")
    s2.metric("전력 비용 증가",  f"{esg['cost_krw_day']:.1f} 만원",     f"{cost_delta:+.2f} 만원",    delta_color="inverse")
    s3.metric("탄소 배출 증가",  f"{esg['carbon_tco2_day']:.3f} tCO₂", f"{carbon_delta:+.3f} tCO₂", delta_color="inverse")
    s4.metric("IT 부하 변화",    f"{df['IT 전력 (kW)'].mean():.0f} kW", f"{it_delta_pct:+.1f}%",      delta_color="inverse")

    st.markdown("<br>", unsafe_allow_html=True)

    hours = list(range(24))

    col_power, col_pue = st.columns(2)

    with col_power:
        with st.container(border=True):
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=hours, y=df_base["총 전력 (kW)"],
                name="정상", line=dict(color=CLR_GOOD, dash="dot", width=2),
            ))
            fig.add_trace(go.Scatter(
                x=hours, y=df["총 전력 (kW)"],
                name=f"위기 ({cfg['label']})", fill="tonexty",
                line=dict(color=CLR_DANGER, width=2),
            ))
            fig.update_layout(
                title="총 전력 (kW)", height=280,
                margin=dict(t=40, b=50, l=10, r=10),
                xaxis_title="시간 (h)", yaxis_title="kW",
                legend=dict(orientation="h", y=-0.25),
            )
            st.plotly_chart(fig, width="stretch")

    with col_pue:
        with st.container(border=True):
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=hours, y=df_base["PUE"],
                name="정상", line=dict(color=CLR_GOOD, dash="dot", width=2),
            ))
            fig2.add_trace(go.Scatter(
                x=hours, y=df["PUE"],
                name=f"위기 ({cfg['label']})",
                line=dict(color=CLR_DANGER, width=2),
            ))
            fig2.update_layout(
                title="PUE", height=280,
                margin=dict(t=40, b=50, l=10, r=10),
                xaxis_title="시간 (h)", yaxis_title="PUE",
                legend=dict(orientation="h", y=-0.25),
            )
            st.plotly_chart(fig2, width="stretch")

    col_it, col_temp = st.columns(2)

    with col_it:
        with st.container(border=True):
            fig3 = go.Figure()
            fig3.add_trace(go.Bar(x=hours, y=df_base["IT 전력 (kW)"],
                                  name="정상", marker_color=CLR_IT))
            fig3.add_trace(go.Bar(x=hours, y=df["IT 전력 (kW)"],
                                  name="위기", marker_color=CLR_DANGER, opacity=0.7))
            fig3.update_layout(
                title="IT 전력 (kW)", height=280, barmode="overlay",
                margin=dict(t=40, b=50, l=10, r=10),
                xaxis_title="시간 (h)", yaxis_title="kW",
                legend=dict(orientation="h", y=-0.25),
            )
            st.plotly_chart(fig3, width="stretch")

    with col_temp:
        with st.container(border=True):
            fig4 = go.Figure()
            fig4.add_trace(go.Scatter(
                x=hours, y=df_base["환기 온도 (°C)"],
                name="정상", line=dict(color=CLR_GOOD, dash="dot"),
            ))
            fig4.add_trace(go.Scatter(
                x=hours, y=df["환기 온도 (°C)"],
                name="위기", line=dict(color=CLR_DANGER, width=2),
            ))
            fig4.add_hline(y=27.0, line_dash="dash", line_color="red",
                           annotation_text="경고 27°C")
            fig4.update_layout(
                title="환기 온도 (°C)", height=280,
                margin=dict(t=40, b=50, l=10, r=10),
                xaxis_title="시간 (h)", yaxis_title="°C",
                legend=dict(orientation="h", y=-0.25),
            )
            st.plotly_chart(fig4, width="stretch")
