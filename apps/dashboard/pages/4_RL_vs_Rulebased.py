"""RL vs Rule-based — 폭염·환절기·혹한기·IT 스파이크 시나리오별 절감량 타임랩스 비교.

시나리오 버튼 선택 → 디스크 캐시 시뮬레이션 로드 → ▶ 재생으로
외기온도·공급온도·PUE·전력절감도 4개 차트가 애니메이션으로 동기화.
상단 카드: 데이터센터 규모 슬라이더 → 연간/시나리오별 절감액 실시간 반영.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from apps.dashboard.constants import (
    CLR_CHILLER,
    CLR_TOTAL,
)
from apps.dashboard.playback import (
    PARQUET_PATH,
    ScenarioPreset,
    SIM_BASE_SERVERS,
    list_scenarios,
    scale_savings,
    simulate_compare_cached,
)
from apps.dashboard.sidebar import render_sidebar
from core.config.constants import CARBON_FACTOR_KG_PER_KWH, ELECTRICITY_COST_KRW_PER_KWH

CLR_RULE = CLR_TOTAL
CLR_BEST = CLR_CHILLER

render_sidebar()


# ── 상태 초기화 ───────────────────────────────────────────────────────────
ss = st.session_state
ss.setdefault("ab_scenario_key", "summer")
ss.setdefault("ab_running", False)
ss.setdefault("ab_step", 0)
ss.setdefault("ab_last_advance", 0.0)
FIXED_SPEED = 0.05


@st.cache_resource(show_spinner="시나리오 시뮬레이션 중…")
def _load_scenarios() -> list[ScenarioPreset]:
    return list_scenarios()


@st.cache_resource(show_spinner="RL + Rule-based 시뮬레이션 (디스크 캐시 사용)…")
def _compute(scenario_key: str) -> dict:
    scenarios = {s.key: s for s in _load_scenarios()}
    return simulate_compare_cached(scenarios[scenario_key])


scenarios = _load_scenarios()
scenario_lookup = {s.key: s for s in scenarios}


@st.cache_data(ttl=3600, show_spinner="규모별 절감 효과 계산 중…")
def _scale(num_servers: int) -> dict:
    return scale_savings(num_servers)


def _fmt_krw(won: float) -> str:
    if abs(won) >= 1e8:
        return f"{won / 1e8:,.2f}억원"
    return f"{won / 1e4:,.0f}만원"


# ── 헤더 ─────────────────────────────────────────────────────────────────
st.title(":material/compare_arrows: RL vs Rule-based")
st.caption("같은 외기/IT부하 시계열에서 RL Best(효율+safe fallback)와 Rule-based 컨트롤러를 동시에 돌려 누적 절감량을 비교.")
st.divider()


# ── 도입 효과 계산기 ─────────────────────────────────────────────────────
st.markdown("##### 우리 데이터센터에 도입하면?")

scale_presets = {
    "현재 PoC 규모 (서버 500대)": 500,
    "중형 IDC (서버 5,000대)":     5_000,
    "대형 IDC (서버 50,000대)":    50_000,
    "하이퍼스케일 (서버 200,000대)": 200_000,
}
preset_label = st.select_slider(
    "데이터센터 규모를 선택하세요",
    options=list(scale_presets.keys()),
    value="중형 IDC (서버 5,000대)",
)
num_servers = scale_presets[preset_label]
eff = _scale(num_servers)

def _hero_card_html(num_servers: int, amount: str, kwh: str, co2: str) -> str:
    return (
        f"<div style='text-align:center;padding:10px 4px 6px;'>"
        f"<div style='font-size:0.72rem;color:#80cbc4;text-transform:uppercase;letter-spacing:.06em;font-weight:600;'>연환산 추정</div>"
        f"<div style='font-weight:700;font-size:0.9rem;color:#b2dfdb;margin:4px 0 8px;'>서버 {num_servers:,}대 · 연간</div>"
        f"<div style='font-size:2.3rem;font-weight:900;color:#26c6a6;line-height:1.1;'>{amount}</div>"
        f"<div style='border-top:1px solid rgba(38,198,166,.15);margin-top:10px;padding-top:8px;'>"
        f"<div style='font-size:0.83rem;color:#cfd2d6;'>{kwh}</div>"
        f"<div style='font-size:0.78rem;color:#7a9e8a;margin-top:2px;'>{co2}</div>"
        f"</div></div>"
    )


def _card_html(scenario: str, badge: str, amount: str, kwh: str, co2: str, positive: bool = True) -> str:
    clr = "#2ECC71" if positive else "#e74c3c"
    return (
        f"<div style='text-align:center;padding:10px 4px 6px;'>"
        f"<div style='font-weight:700;font-size:1.0rem;color:#e8ecf0;margin-bottom:4px;'>{scenario}</div>"
        f"<div style='font-size:0.72rem;color:#90a4ae;margin-bottom:10px;'>{badge}</div>"
        f"<div style='font-size:2.1rem;font-weight:900;color:{clr};line-height:1.1;'>{amount}</div>"
        f"<div style='border-top:1px solid rgba(255,255,255,.07);margin-top:10px;padding-top:8px;'>"
        f"<div style='font-size:0.83rem;color:#cfd2d6;'>{kwh}</div>"
        f"<div style='font-size:0.78rem;color:#7a9e8a;margin-top:2px;'>{co2}</div>"
        f"</div></div>"
    )


# 히어로 + 시나리오 카드 한 행 (5열)
hero_col, *card_cols = st.columns([1.2, 1, 1, 1, 1], gap="medium")

with hero_col:
    with st.container(border=True):
        st.markdown(_hero_card_html(
            num_servers=num_servers,
            amount=_fmt_krw(eff['annual_krw']),
            kwh=f"{eff['annual_kwh']:,.0f} kWh",
            co2=f"CO₂ {eff['annual_co2_t']:,.0f}톤",
        ), unsafe_allow_html=True)

for col, sc in zip(card_cols, scenarios):
    res = _compute(sc.key)
    sm = res["summary"]
    sc_kwh = sm["best_savings_kwh"] * (num_servers / SIM_BASE_SERVERS)
    sc_krw = sc_kwh * ELECTRICITY_COST_KRW_PER_KWH
    sc_co2 = sc_kwh * CARBON_FACTOR_KG_PER_KWH / 1000
    days = sc.n_steps // 288
    with col:
        with st.container(border=True):
            st.markdown(_card_html(
                scenario=sc.label,
                badge=f"{days}일 시뮬 · {sm['best_savings_pct']:+.1f}%",
                amount=_fmt_krw(sc_krw),
                kwh=f"{sc_kwh:,.0f} kWh",
                co2=f"CO₂ {sc_co2:.1f}톤",
                positive=sc_kwh >= 0,
            ), unsafe_allow_html=True)

st.divider()


# ── 시나리오 선택 ────────────────────────────────────────────────────────
st.markdown("##### :material/play_circle: 시나리오 선택 & 재생")

btn_cols = st.columns(len(scenarios))
for col, sc in zip(btn_cols, scenarios):
    with col:
        is_active = sc.key == ss.ab_scenario_key
        if st.button(
            sc.label,
            key=f"sc_btn_{sc.key}",
            type="primary" if is_active else "secondary",
            width="stretch",
        ):
            if not is_active:
                ss.ab_scenario_key = sc.key
                ss.ab_step = 0
                ss.ab_running = False
                st.rerun()

current_scenario = scenario_lookup[ss.ab_scenario_key]
st.caption(current_scenario.description)


# ── 시뮬레이션 결과 로드 ──────────────────────────────────────────────────
result = _compute(ss.ab_scenario_key)
df_rule:   pd.DataFrame = result["rule"]
df_best:   pd.DataFrame = result["best"]
summary = result["summary"]
n_steps = len(df_rule)

# 실제 날짜/시각 복원 (parquet timestamp + start_idx 기준)
@st.cache_data(show_spinner=False)
def _start_timestamp(scenario_key: str) -> pd.Timestamp:
    sc = scenario_lookup[scenario_key]
    ts = pd.read_parquet(PARQUET_PATH, columns=["timestamp"])
    return pd.to_datetime(ts["timestamp"].iloc[sc.start_idx])

_start_ts = _start_timestamp(ss.ab_scenario_key)

if not summary["rl_loaded"]:
    st.warning("RL 모델 로드 실패 — RL Best 트랙을 Rule-based 결과로 표시됩니다.", icon=":material/warning:")


# ── 차트 빌더 (타임랩스용) ────────────────────────────────────────────────
UIREV = f"ab-{ss.ab_scenario_key}"
_xfmt_anim = "%m/%d %H:%M"
_dt_full = _start_ts + pd.to_timedelta(df_rule["minute"], unit="min")
X_MIN_DT = _dt_full.iloc[0]
X_MAX_DT = _dt_full.iloc[-1]
PUE_MAX = max(df_rule["PUE"].max(), df_best["PUE"].max()) * 1.05
_savings_best = df_rule["누적 kWh"] - df_best["누적 kWh"]
SAVINGS_MIN = min(0.0, float(_savings_best.min())) * 1.1
SAVINGS_MAX = max(float(_savings_best.max()), 0.1) * 1.1
OUT_MIN = float(df_rule["외기온도"].min()) - 1
OUT_MAX = float(df_rule["외기온도"].max()) + 1
_sup_all = pd.concat([df_rule["공급 온도"], df_best["공급 온도"]])
SUP_MIN = float(_sup_all.min()) - 1
SUP_MAX = float(_sup_all.max()) + 1


def _build_temp_chart(sub_rule: pd.DataFrame, sub_best: pd.DataFrame) -> go.Figure:
    """외기 온도(상) + 공급 온도(하) subplot — 렌더 1회"""
    x_rule = _start_ts + pd.to_timedelta(sub_rule["minute"], unit="min")
    x_best = _start_ts + pd.to_timedelta(sub_best["minute"], unit="min")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.45, 0.55], vertical_spacing=0.08,
                        subplot_titles=("외기 온도 (°C)", "공급 온도 Setpoint (°C)"))
    fig.add_trace(go.Scatter(
        x=x_rule, y=sub_rule["외기온도"],
        name="외기 온도", line=dict(color="#F4A261", width=2.0),
        fill="tozeroy", fillcolor="rgba(244,162,97,0.12)",
        hovertemplate="%{x|" + _xfmt_anim + "} · %{y:.1f}°C<extra>외기 온도</extra>",
        showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=x_rule, y=sub_rule["공급 온도"],
        name="Rule-based", line=dict(color=CLR_RULE, width=1.5, dash="dot"),
        hovertemplate="%{x|" + _xfmt_anim + "} · %{y:.1f}°C<extra>Rule-based</extra>",
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=x_best, y=sub_best["공급 온도"],
        name="RL Best", line=dict(color=CLR_BEST, width=2.5),
        fill="tonexty", fillcolor="rgba(93,144,255,0.12)",
        hovertemplate="%{x|" + _xfmt_anim + "} · %{y:.1f}°C<extra>RL Best</extra>",
    ), row=2, col=1)
    _xax = dict(tickformat=_xfmt_anim, dtick=3 * 3600 * 1000, range=[X_MIN_DT, X_MAX_DT], fixedrange=True)
    fig.update_layout(
        height=380, margin=dict(l=10, r=10, t=40, b=10),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        uirevision=f"{UIREV}-temp", transition={"duration": 0},
    )
    fig.update_xaxes(**_xax)
    fig.update_yaxes(range=[OUT_MIN, OUT_MAX], fixedrange=True, row=1, col=1)
    fig.update_yaxes(range=[SUP_MIN, SUP_MAX], fixedrange=True, row=2, col=1)
    return fig


def _build_perf_chart(sub_rule: pd.DataFrame, sub_best: pd.DataFrame) -> go.Figure:
    """PUE 비교(상) + 전력절감도(하) subplot — 렌더 1회"""
    x_rule = _start_ts + pd.to_timedelta(sub_rule["minute"], unit="min")
    x_best = _start_ts + pd.to_timedelta(sub_best["minute"], unit="min")
    savings = sub_rule["누적 kWh"].values - sub_best["누적 kWh"].values
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.5, 0.5], vertical_spacing=0.08,
                        subplot_titles=("PUE 비교 (낮을수록 best)", "전력절감도"))
    fig.add_trace(go.Scatter(
        x=x_rule, y=sub_rule["PUE"],
        name="Rule-based", line=dict(color=CLR_RULE, width=2),
        hovertemplate="%{x|" + _xfmt_anim + "} · PUE %{y:.3f}<extra>Rule-based</extra>",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=x_best, y=sub_best["PUE"],
        name="RL Best", line=dict(color=CLR_BEST, width=2),
        hovertemplate="%{x|" + _xfmt_anim + "} · PUE %{y:.3f}<extra>RL Best</extra>",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=x_best, y=savings,
        name="RL Best 누적 절감", line=dict(color=CLR_BEST, width=2.5),
        fill="tozeroy", fillcolor="rgba(93,144,255,0.15)",
        hovertemplate="%{x|" + _xfmt_anim + "} · %{y:.2f} kWh<extra>절감</extra>",
        showlegend=False,
    ), row=2, col=1)
    _xax = dict(tickformat=_xfmt_anim, dtick=3 * 3600 * 1000, range=[X_MIN_DT, X_MAX_DT], fixedrange=True)
    fig.update_layout(
        height=380, margin=dict(l=10, r=10, t=40, b=10),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        uirevision=f"{UIREV}-perf", transition={"duration": 0},
    )
    fig.update_xaxes(**_xax)
    fig.update_yaxes(range=[1.0, PUE_MAX], fixedrange=True, row=1, col=1)
    fig.update_yaxes(range=[SAVINGS_MIN, SAVINGS_MAX], fixedrange=True, title_text="누적 절감 (kWh)", row=2, col=1)
    return fig


# ── 동적 영역 (fragment) ─────────────────────────────────────────────────
@st.fragment(run_every="100ms")
def playback_fragment():
    if ss.ab_running and ss.ab_step < n_steps - 1:
        now = time.time()
        if now - ss.ab_last_advance >= FIXED_SPEED:
            stride = max(1, n_steps // 200)
            ss.ab_step = min(ss.ab_step + stride, n_steps - 1)
            ss.ab_last_advance = now

    idx = min(ss.ab_step, n_steps - 1)
    cur_rule = df_rule.iloc[idx]
    cur_best = df_best.iloc[idx]

    elapsed_min = idx * 5
    progress_pct = (idx + 1) / n_steps
    save_best_kwh = float(cur_rule["누적 kWh"] - cur_best["누적 kWh"])
    save_best_krw = float(cur_rule["누적 원"] - cur_best["누적 원"])

    # 버튼 + 재생바 한 행
    bc1, bc2, bc3, pb_col = st.columns([1, 1, 1, 7])
    with bc1:
        if st.button("▶", disabled=ss.ab_running, width="stretch", key="frag_play"):
            if ss.ab_step >= n_steps - 1:
                ss.ab_step = 0
            ss.ab_running = True
            st.rerun()
    with bc2:
        if st.button("⏸", disabled=not ss.ab_running, width="stretch", key="frag_pause"):
            ss.ab_running = False
            st.rerun()
    with bc3:
        if st.button("⏹", width="stretch", key="frag_stop"):
            ss.ab_running = False
            ss.ab_step = 0
            st.rerun()
    with pb_col:
        st.progress(progress_pct, text=f"경과 {elapsed_min // 60:02d}:{elapsed_min % 60:02d}")

    rule_kwh = cur_rule['누적 kWh']
    best_kwh = cur_best['누적 kWh']
    diff_pct = (save_best_kwh / rule_kwh * 100) if rule_kwh > 0 else 0.0
    is_saving = save_best_kwh >= 0
    accent  = "#2ECC71" if is_saving else "#e74c3c"
    box_lbl = "절감액" if is_saving else "초과 사용"
    st.markdown(
        f"<div style='display:flex;align-items:center;gap:12px;padding:10px 4px;'>"
        f"<div style='flex:1;text-align:center;padding:8px 12px;background:rgba(255,255,255,0.04);border-radius:8px;'>"
        f"<div style='font-size:0.75rem;color:#90a4ae;margin-bottom:4px;'>Rule-based 누적</div>"
        f"<div style='font-size:1.35rem;font-weight:600;color:#b0bec5;'>{rule_kwh:,.1f} kWh</div>"
        f"</div>"
        f"<div style='font-size:1.2rem;color:#546e7a;'>→</div>"
        f"<div style='flex:1;text-align:center;padding:8px 12px;background:rgba(255,255,255,0.04);border-radius:8px;'>"
        f"<div style='font-size:0.75rem;color:#90a4ae;margin-bottom:4px;'>RL Best 누적</div>"
        f"<div style='font-size:1.35rem;font-weight:600;color:#b0bec5;'>{best_kwh:,.1f} kWh</div>"
        f"</div>"
        f"<div style='flex:2;text-align:center;padding:10px 16px;background:rgba(255,255,255,0.04);border-radius:8px;'>"
        f"<div style='font-size:0.78rem;color:#90a4ae;margin-bottom:4px;letter-spacing:.04em;text-transform:uppercase;'>{box_lbl}</div>"
        f"<div style='font-size:2.0rem;font-weight:900;color:{accent};line-height:1.1;'>{save_best_krw:+,.0f}원</div>"
        f"<div style='font-size:0.85rem;color:#90a4ae;margin-top:3px;'>{save_best_kwh:+,.1f} kWh · {diff_pct:+.1f}%</div>"
        f"</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    sub_rule = df_rule.iloc[: idx + 1]
    sub_best = df_best.iloc[: idx + 1]

    ch1, ch2 = st.columns(2)
    with ch1:
        st.plotly_chart(_build_temp_chart(sub_rule, sub_best), width="stretch", key="chart_temp")
    with ch2:
        st.plotly_chart(_build_perf_chart(sub_rule, sub_best), width="stretch", key="chart_perf")

    if ss.ab_step >= n_steps - 1:
        ss.ab_running = False
        best_pct  = summary["best_savings_pct"]
        best_viol = summary["best_violations"]
        st.success(
            f"**시뮬레이션 종료** — "
            f"RL Best **{best_pct:+.1f} %** 절감 (위반 {best_viol} 회). "
            f"누적 절감액 {summary['best_savings_krw']:+,.0f} 원 · "
            f"CO₂ {summary['best_savings_co2']:+,.1f} kg",
            icon=":material/celebration:",
        )


playback_fragment()

