"""
분석 도구 — 예측 애니메이션 · 존 히트맵

24시간 예측 타임랩스 · 데이터센터 플로어 존별 온도/전력 2D 히트맵
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from apps.dashboard.charts import build_power_trend, build_pue_trend
from apps.dashboard.constants import ANIM_SPEED_OPTIONS, CLR_DANGER, COOLING_MODE_LABELS
from apps.dashboard.sidebar import add_alarm, render_sidebar

d = render_sidebar()

st.title(":material/analytics: 분석 도구")
st.caption("24h 예측 애니메이션 · Multi-Zone 2D Heatmap")
st.divider()

tab_anim, tab_heatmap = st.tabs(["예측 애니메이션", "Multi-Zone Heatmap"])

# ── 예측 애니메이션 탭 ────────────────────────────────────────────────────

with tab_anim:
    st.info(
        "**목업 데이터**: 현재 열역학 시뮬레이션 결과를 표시 중입니다. "
        "Forecast Service 구현 후 실제 예측값(`POST /api/v1/forecast`)으로 교체됩니다.",
    )

    # 인라인 컨트롤
    ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([3, 1, 1, 1])
    with ctrl1:
        speed_label = st.select_slider(
            "재생 속도",
            options=list(ANIM_SPEED_OPTIONS.keys()),
            value="보통 (0.4초)",
            label_visibility="collapsed",
        )
        st.session_state.anim_speed = ANIM_SPEED_OPTIONS[speed_label]

    if ctrl2.button("▶ 재생", use_container_width=True,
                    disabled=st.session_state.anim_running):
        st.session_state.anim_running = True
        st.session_state.anim_hour   = 0
        st.rerun()

    if ctrl3.button("⏸ 정지", use_container_width=True,
                    disabled=not st.session_state.anim_running):
        st.session_state.anim_running = False
        st.rerun()

    if ctrl4.button("⟲ 초기화", use_container_width=True):
        st.session_state.anim_running = False
        st.session_state.anim_hour   = 0
        st.rerun()

    anim_running = st.session_state.anim_running
    anim_hour    = st.session_state.anim_hour
    anim_active  = anim_running or anim_hour > 0

    df_full  = d["df_full"]
    # 애니메이션 중엔 현재 시각까지만 슬라이싱 — 차트가 실시간으로 누적됨
    df_shown = df_full.iloc[:anim_hour + 1] if anim_active else df_full

    if anim_active:
        st.markdown(
            f'<div style="display:flex;align-items:baseline;gap:12px;margin-bottom:4px;">'
            f'<span style="font-size:1.4rem;font-weight:700;">T+{anim_hour}h &nbsp;'
            f'<span style="font-size:1rem;opacity:0.6;">({anim_hour:02d}:00)</span></span>'
            f'<span style="font-size:0.82rem;opacity:0.5;">'
            f'{"▶ 재생 중" if anim_running else "⏸ 일시정지"}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.caption("▶ 재생 버튼을 눌러 T+0h ~ T+23h 예측 흐름을 확인하세요.")

    st.progress((anim_hour + 1) / 24 if anim_active else 0)

    if anim_active:
        cur = df_shown.iloc[-1]
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("PUE",       f"{cur['PUE']:.3f}")
        k2.metric("IT 전력",   f"{cur['IT 전력 (kW)']:.0f} kW")
        k3.metric("칠러 전력", f"{cur['칠러 전력 (kW)']:.0f} kW")
        k4.metric("냉각 모드", COOLING_MODE_LABELS.get(cur["냉각 모드"], cur["냉각 모드"]))
        k5.metric("외기 온도", f"{cur['외기온도 (°C)']} °C")

    col_power, col_pue = st.columns(2)
    with col_power:
        with st.container(border=True):
            fig_power = build_power_trend(df_shown, df_full=df_full, anim_hour=anim_hour if anim_active else None)
            fig_power.update_layout(height=320)
            st.plotly_chart(fig_power, width="stretch")
    with col_pue:
        with st.container(border=True):
            st.plotly_chart(
                build_pue_trend(df_shown, df_full=df_full, anim_hour=anim_hour if anim_active else None),
                width="stretch",
            )

    if anim_active:
        with st.expander("누적 데이터", expanded=False):
            st.dataframe(
                df_shown.style.format({"PUE": "{:.3f}", "COP": "{:.2f}"}),
                width="stretch",
                height=240,
            )

# ── 존 히트맵 탭 ──────────────────────────────────────────────────────────

with tab_heatmap:
    st.info(
        "**목업 데이터**: 실제 센서 데이터 없이 랜덤 시뮬레이션 값을 표시 중입니다. "
        "멀티존 제어 서비스 구현 후 실측값으로 교체됩니다.",
    )

    ZONES      = ["Zone A", "Zone B", "Zone C"]
    ZONE_ROWS  = 2
    COLS       = 10
    N_ZONES    = len(ZONES)
    TOTAL_ROWS = N_ZONES * ZONE_ROWS  # 6

    view = st.radio(
        "표시 항목",
        ["온도 (°C)", "IT 전력 (kW)", "칠러 전력 (kW)"],
        horizontal=True,
    )

    rng         = np.random.default_rng(seed=42)
    supply_temp = d["supply_temp"]
    data        = np.zeros((TOTAL_ROWS, COLS))

    if view == "온도 (°C)":
        for zone_i in range(N_ZONES):
            for row_in_zone in range(ZONE_ROWS):
                g    = zone_i * ZONE_ROWS + row_in_zone
                heat = 10.0 if row_in_zone == 1 else 2.0
                grad = np.linspace(0, 3.0, COLS)
                data[g] = supply_temp + heat + grad
        data      += rng.normal(0, 0.8, data.shape)
        unit       = "°C"
        cmin, cmax = supply_temp, supply_temp + 18
        colorscale = "RdYlBu_r"
        warning_val = 27.0
        title      = "Zone별 온도 분포 (°C)"

    elif view == "IT 전력 (kW)":
        peak_it      = d["avg_it"]
        per_rack     = peak_it / (TOTAL_ROWS * COLS)
        zone_factors = [1.2, 1.0, 0.8]
        for zone_i in range(N_ZONES):
            for row_in_zone in range(ZONE_ROWS):
                g       = zone_i * ZONE_ROWS + row_in_zone
                data[g] = rng.uniform(per_rack * 0.7, per_rack * 1.3, COLS) * zone_factors[zone_i]
        unit       = "kW"
        cmin, cmax = 0, data.max() * 1.1
        colorscale = "YlOrBr"
        warning_val = None
        title      = "Zone별 IT 전력 분포 (kW)"

    else:
        peak_cool  = d["avg_cooling"]
        per_rack   = peak_cool / (TOTAL_ROWS * COLS)
        data       = rng.uniform(per_rack * 0.5, per_rack * 1.5, (TOTAL_ROWS, COLS))
        unit       = "kW"
        cmin, cmax = 0, data.max() * 1.1
        colorscale = [[0.0, "#ede9fe"], [0.5, "#A3ABFB"], [1.0, "#4338ca"]]
        warning_val = None
        title      = "Zone별 칠러 전력 분포 (kW)"

    y_labels = []
    for zone_name in ZONES:
        y_labels.append(f"{zone_name}  Cold Aisle")
        y_labels.append(f"{zone_name}  Hot Aisle")

    x_labels = [f"R{c+1:02d}" for c in range(COLS)]

    fig = go.Figure(go.Heatmap(
        z=data,
        x=x_labels,
        y=y_labels,
        colorscale=colorscale,
        zmin=cmin,
        zmax=cmax,
        text=np.round(data, 1),
        texttemplate="%{text}",
        textfont={"size": 9},
        colorbar=dict(title=unit, thickness=15),
    ))

    for zone_i in range(1, N_ZONES):
        boundary_y = zone_i * ZONE_ROWS - 0.5
        fig.add_shape(
            type="line",
            xref="paper", x0=0, x1=1,
            yref="y",     y0=boundary_y, y1=boundary_y,
            line=dict(color="white", width=3),
        )

    if warning_val is not None:
        for r in range(TOTAL_ROWS):
            for c in range(COLS):
                if data[r, c] > warning_val:
                    fig.add_shape(
                        type="rect",
                        x0=c - 0.5, x1=c + 0.5,
                        y0=r - 0.5, y1=r + 0.5,
                        line=dict(color=CLR_DANGER, width=2),
                    )

    fig.update_layout(
        title=title,
        height=500,
        margin=dict(t=50, b=40, l=145, r=20),
        xaxis=dict(title="랙 번호", side="top"),
        yaxis=dict(title="", autorange="reversed"),
    )

    with st.container(border=True):
        st.plotly_chart(fig, width="stretch")

    st.divider()
    st.subheader("Zone별 요약")

    ZONE_ACCENTS = ["#5D90FF", "#7BD2F7", "#A3ABFB"]

    zone_cols = st.columns(N_ZONES)
    for zone_i, (zone_name, col_ui) in enumerate(zip(ZONES, zone_cols)):
        zone_data = data[zone_i * ZONE_ROWS : (zone_i + 1) * ZONE_ROWS]
        accent    = ZONE_ACCENTS[zone_i]
        with col_ui:
            with st.container(border=True):
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:10px;">'
                    f'<div style="width:8px;height:8px;border-radius:50%;background:{accent};"></div>'
                    f'<span style="font-size:1rem;font-weight:700;color:{accent};">{zone_name}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                if view == "온도 (°C)":
                    over = int((zone_data > 27.0).sum())
                    st.metric("평균 온도", f"{zone_data.mean():.1f} °C")
                    st.metric("최고 온도", f"{zone_data.max():.1f} °C")
                    st.metric("경고 랙",   f"{over}개",
                              delta=f"+{over}" if over else "정상",
                              delta_color="inverse")
                else:
                    st.metric("평균", f"{zone_data.mean():.1f} {unit}")
                    st.metric("최대", f"{zone_data.max():.1f} {unit}")
                    st.metric("합계", f"{zone_data.sum():.1f} {unit}")

# ── 애니메이션 프레임 전진 (탭 밖에서 실행) ───────────────────────────────

if st.session_state.anim_running:
    time.sleep(st.session_state.anim_speed)
    if st.session_state.anim_hour < 23:
        st.session_state.anim_hour += 1
    else:
        st.session_state.anim_running = False
        add_alarm("INFO", "24시간 예측 애니메이션 재생 완료")
    st.rerun()
