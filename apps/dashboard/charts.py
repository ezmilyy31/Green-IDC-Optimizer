"""Plotly 차트 빌더 모음"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from apps.dashboard.constants import (
    CARBON_FACTOR_TCO2_PER_MWH,
    CLR_CHILLER,
    CLR_DANGER,
    CLR_GOOD,
    CLR_IT,
    CLR_TOTAL,
    CLR_WARN,
    COOLING_MODE_LABELS,
    COOLING_MODE_COLORS,
    MODE_COLOR_MAP,
    NAVER_PUE_BENCHMARK,
    PUE_BENCHMARKS,
    PUE_GAUGE_STEPS,
    TEMP_WARNING_THRESHOLD_C,
)


def build_pue_gauge(avg_pue: float, title: str = "PUE (24h 평균)") -> go.Figure:
    color = CLR_GOOD if avg_pue < 1.4 else (CLR_WARN if avg_pue < 1.8 else CLR_DANGER)
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=avg_pue,
        title={"text": title, "font": {"size": 13}},
        delta={"reference": NAVER_PUE_BENCHMARK, "valueformat": ".3f"},
        number={"valueformat": ".3f"},
        domain={"x": [0, 1], "y": [0, 0.85]},
        gauge={
            "axis":      {"range": [1.0, 2.5], "tickformat": ".1f", "ticklen": 12},
            "bar":       {"color": color},
            "steps":     PUE_GAUGE_STEPS,
            "threshold": {
                "line": {"color": "gray", "width": 2},
                "thickness": 0.75,
                "value": NAVER_PUE_BENCHMARK,
            },
        },
    ))
    fig.update_layout(height=300, margin=dict(t=20, b=20, l=20, r=20))
    return fig


def build_power_trend(
    df: pd.DataFrame,
    df_full: pd.DataFrame | None = None,
    anim_hour: int | None = None,
) -> go.Figure:
    """전력 소비 추이. 애니메이션 모드면 df_full을 흐린 배경으로 표시."""
    hours = list(range(len(df)))
    fig = go.Figure()

    if df_full is not None and anim_hour is not None:
        full_hours = list(range(24))
        for col, color in [("IT 전력 (kW)", CLR_IT), ("총 전력 (kW)", CLR_TOTAL)]:
            fig.add_trace(go.Scatter(
                x=full_hours, y=df_full[col],
                line=dict(color=color, width=1, dash="dot"),
                opacity=0.2, showlegend=False, hoverinfo="skip",
            ))

    fig.add_trace(go.Scatter(
        x=hours, y=df["IT 전력 (kW)"],
        name="IT 전력", fill="tozeroy", line=dict(color=CLR_IT),
    ))
    fig.add_trace(go.Scatter(
        x=hours, y=df["칠러 전력 (kW)"],
        name="칠러 전력", fill="tozeroy", line=dict(color=CLR_CHILLER),
    ))
    fig.add_trace(go.Scatter(
        x=hours, y=df["총 전력 (kW)"],
        name="총 전력", line=dict(color=CLR_TOTAL, dash="dot", width=2),
    ))

    if anim_hour is not None and len(hours) > 0:
        fig.add_vline(
            x=hours[-1], line_dash="solid", line_color="#E74C3C", line_width=2,
            annotation_text=f"{hours[-1]:02d}:00", annotation_position="top right",
        )

    fig.update_layout(
        title="전력 소비 추이 (kW)", height=300,
        margin=dict(t=40, b=70, l=10, r=10),
        xaxis=dict(title="시간 (h)", range=[-0.5, 23.5], dtick=4),
        yaxis_title="전력 (kW)",
        legend=dict(orientation="h", y=-0.32),
    )
    return fig


def build_pue_trend(
    df: pd.DataFrame,
    df_full: pd.DataFrame | None = None,
    anim_hour: int | None = None,
) -> go.Figure:
    hours = list(range(len(df)))
    fig = go.Figure()

    if df_full is not None and anim_hour is not None:
        fig.add_trace(go.Scatter(
            x=list(range(24)), y=df_full["PUE"],
            line=dict(color=CLR_GOOD, width=1, dash="dot"),
            opacity=0.2, showlegend=False, hoverinfo="skip",
        ))

    fig.add_trace(go.Scatter(
        x=hours, y=df["PUE"],
        line=dict(color=CLR_GOOD, width=2), mode="lines+markers",
        showlegend=False,
    ))
    fig.add_hline(
        y=NAVER_PUE_BENCHMARK, line_dash="dash", line_color="gray",
        annotation_text=f"{NAVER_PUE_BENCHMARK} (NAVER)",
    )

    if anim_hour is not None and len(hours) > 0:
        fig.add_vline(x=hours[-1], line_dash="solid", line_color="#E74C3C", line_width=2)

    fig.update_layout(
        height=320, margin=dict(t=20, b=50, l=10, r=10),
        xaxis=dict(title="시간 (h)", range=[-0.5, 23.5]),
        yaxis_title="PUE", showlegend=False,
    )
    return fig


def build_rack_temp_chart(
    labels: list, temps: list, colors: list,
    y_min: float, y_max: float,
) -> go.Figure:
    fig = go.Figure(go.Bar(x=labels, y=temps, marker_color=colors))
    fig.add_hline(
        y=TEMP_WARNING_THRESHOLD_C, line_dash="dash", line_color="red",
        annotation_text=f"경고 {TEMP_WARNING_THRESHOLD_C}°C",
    )
    fig.update_layout(
        title="서버 온도 분포 (피크 시간)", height=300,
        margin=dict(t=40, b=50, l=10, r=10),
        xaxis_title="랙", yaxis_title="온도 (°C)",
        yaxis=dict(range=[y_min, y_max]),
    )
    return fig


def build_cooling_donut(df: pd.DataFrame) -> go.Figure:
    mode_counts = df["냉각 모드"].value_counts()
    fig = go.Figure(go.Pie(
        labels=[COOLING_MODE_LABELS.get(m, m) for m in mode_counts.index],
        values=mode_counts.values,
        marker=dict(colors=COOLING_MODE_COLORS),
        textinfo="label+percent",
        hole=0.5,
    ))
    fig.update_layout(height=200, margin=dict(t=10, b=10, l=10, r=10), showlegend=False)
    return fig


def build_carbon_bar(df: pd.DataFrame) -> go.Figure:
    mode_carbon = df.groupby("냉각 모드").apply(
        lambda g: g["총 전력 (kW)"].mean() * CARBON_FACTOR_TCO2_PER_MWH
    ).reset_index()
    mode_carbon.columns = ["모드_key", "탄소(kgCO₂/h)"]
    mode_carbon["색상"] = mode_carbon["모드_key"].map(MODE_COLOR_MAP)
    mode_carbon["모드"] = mode_carbon["모드_key"].map(COOLING_MODE_LABELS)

    fig = go.Figure(go.Bar(
        x=mode_carbon["모드"],
        y=mode_carbon["탄소(kgCO₂/h)"],
        marker_color=mode_carbon["색상"],
    ))
    fig.update_layout(
        height=200, margin=dict(t=10, b=30, l=10, r=10),
        yaxis_title="kgCO₂/h", showlegend=False,
    )
    return fig


def render_pue_benchmarks(avg_pue: float | None = None) -> None:
    """PUE 벤치마크 비교표를 렌더링한다."""
    PUE_MIN, PUE_MAX = 1.0, 2.2

    rows = [(label, float(val), False) for label, val in PUE_BENCHMARKS]
    if avg_pue is not None:
        rows.append(("현재", avg_pue, True))
    rows.sort(key=lambda x: x[1])

    items_html = ""
    for label, val, is_current in rows:
        bar_pct  = max(0, min(100, (val - PUE_MIN) / (PUE_MAX - PUE_MIN) * 100))
        if is_current:
            bar_color   = CLR_GOOD if val < 1.4 else (CLR_WARN if val < 1.8 else CLR_DANGER)
            row_style   = "background:rgba(128,128,128,0.08);border-radius:6px;padding:5px 8px;margin:3px 0;"
            label_style = "font-weight:700;"
            val_style   = f"font-weight:800;color:{bar_color};"
        else:
            bar_color   = "#94a3b8"
            row_style   = "padding:4px 8px;margin:2px 0;"
            label_style = "opacity:0.75;"
            val_style   = "font-weight:600;opacity:0.85;"

        items_html += f"""
        <div style="display:flex;flex-direction:column;{row_style}">
          <div style="display:flex;justify-content:space-between;font-size:0.76rem;margin-bottom:3px;">
            <span style="{label_style}">{label}</span>
            <span style="{val_style}">{val:.2f}</span>
          </div>
          <div style="background:rgba(128,128,128,0.15);border-radius:4px;height:4px;">
            <div style="width:{bar_pct:.1f}%;background:{bar_color};height:4px;border-radius:4px;"></div>
          </div>
        </div>"""

    st.markdown(
        f'<div style="margin-top:12px;">'
        f'<div style="font-size:0.7rem;font-weight:700;opacity:0.5;'
        f'text-transform:uppercase;letter-spacing:0.06em;margin-bottom:6px;">PUE 벤치마크</div>'
        f'{items_html}'
        f'<div style="font-size:0.68rem;opacity:0.4;margin-top:6px;text-align:right;">'
        f'기준 범위 {PUE_MIN} – {PUE_MAX}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


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
            # TODO(Simulation Service): expected_pue 필드가 현재 고정값 1.35를 반환 중이므로 표시 생략.
            # Simulation Service 연동 후 f" | 예상 PUE: `{result.get('expected_pue', '-'):.3f}`" 추가.
        )
