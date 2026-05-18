"""RL vs Rule-based — 동일 시나리오에서 RL과 Rule-based 컨트롤러의 절감량을 타임랩스로 비교.

시나리오 프리셋 버튼 → 사전 시뮬레이션 → ▶ 재생으로 두 컨트롤러의 PUE/전력/온도가
실시간처럼 그려지며 누적 절감량(kWh, 원, kgCO₂)이 카운터로 증가한다.

부드러운 재생을 위해 @st.fragment로 동적 영역을 격리하고, 모든 Plotly 차트에
uirevision을 부여해 축 상태를 유지한다 → 페이지 전체 재렌더링 없이 fragment만 갱신.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from apps.dashboard.constants import (
    ANIM_SPEED_OPTIONS,
    CLR_CHILLER,
    CLR_GOOD,
    CLR_TOTAL,
)
from apps.dashboard.playback import (
    ScenarioPreset,
    list_scenarios,
    simulate_compare_cached,
)
from apps.dashboard.sidebar import render_sidebar

# 3-way 비교 색상: Rule(주황) · Best(인디고/효율) · Hybrid(녹색/안전)
CLR_RULE = CLR_TOTAL
CLR_BEST = CLR_CHILLER
CLR_HYBRID = CLR_GOOD

render_sidebar()


# ── 상태 초기화 ───────────────────────────────────────────────────────────
ss = st.session_state
ss.setdefault("ab_scenario_key", "shoulder")
ss.setdefault("ab_running", False)
ss.setdefault("ab_step", 0)
ss.setdefault("ab_speed", 0.1)
ss.setdefault("ab_last_advance", 0.0)  # 마지막 step 진행 시각 (throttle용)


@st.cache_resource(show_spinner="시나리오 시뮬레이션 중… (RL/Rule 두 컨트롤러 비교)")
def _load_scenarios() -> list[ScenarioPreset]:
    return list_scenarios()


@st.cache_resource(show_spinner="RL + Rule-based 시뮬레이션 (디스크 캐시 사용)…")
def _compute(scenario_key: str) -> dict:
    scenarios = {s.key: s for s in _load_scenarios()}
    # 디스크 캐시: 첫 호출만 시뮬레이션, 이후 즉시 로드 (컨테이너 재시작에도 유지)
    return simulate_compare_cached(scenarios[scenario_key])


scenarios = _load_scenarios()
scenario_lookup = {s.key: s for s in scenarios}


# ── 헤더 ─────────────────────────────────────────────────────────────────
st.title(":material/compare_arrows: RL vs Rule-based")
st.caption("같은 외기/IT부하 시계열에서 RL 컨트롤러(Best · Hybrid)와 Rule-based 컨트롤러를 동시에 돌려 누적 절감량을 비교.")
st.divider()


# ── 시나리오 선택 (프리셋 버튼) ───────────────────────────────────────────
st.markdown("##### 시나리오 프리셋")
cols = st.columns(len(scenarios))
for col, sc in zip(cols, scenarios):
    selected = ss.ab_scenario_key == sc.key
    # 라벨에서 "(N일)" 기간 표기 제거 → 버튼 더 깔끔
    short_label = sc.label.split(" (")[0].strip()
    if col.button(
        short_label,
        key=f"sc_{sc.key}",
        help=sc.description,
        width="stretch",
        type="primary" if selected else "secondary",
    ):
        ss.ab_scenario_key = sc.key
        ss.ab_step = 0
        ss.ab_running = False
        st.rerun()

current_scenario = scenario_lookup[ss.ab_scenario_key]
st.caption(f"**{current_scenario.label}** — {current_scenario.description}")


# ── 시뮬레이션 결과 로드 ──────────────────────────────────────────────────
result = _compute(ss.ab_scenario_key)
df_rule:   pd.DataFrame = result["rule"]
df_best:   pd.DataFrame = result["best"]
df_hybrid: pd.DataFrame = result["hybrid"]
summary = result["summary"]
n_steps = len(df_rule)

if not summary["rl_loaded"]:
    st.warning("RL 모델 로드 실패 — 세 트랙 모두 Rule-based 결과로 표시됩니다 (시연 안정성).", icon=":material/warning:")


# ── 시나리오 전체 결과: 2개 그룹 카드 (Best / Hybrid) ─────────────────────
st.markdown("##### 시나리오 전체 결과")

def _safety_delta(violations: int) -> tuple[str, str]:
    return ("안전", "normal") if violations == 0 else ("주의", "inverse")

card_best, card_hybrid = st.columns(2)

with card_best:
    with st.container(border=True):
        st.markdown(":blue[**RL Best**]  ·  효율 우선")
        sb1, sb2 = st.columns(2)
        sb1.metric(
            "Rule 대비 절감",
            f"{summary['best_savings_pct']:+.2f}%",
            f"{summary['best_savings_kwh']:+,.1f} kWh",
            delta_color="off",
        )
        delta_text, delta_color = _safety_delta(summary['best_violations'])
        sb2.metric(
            "온도 위반",
            f"{summary['best_violations']} 회",
            delta_text,
            delta_color=delta_color,
        )

with card_hybrid:
    with st.container(border=True):
        st.markdown(":green[**RL Hybrid**]  ·  안전 우선")
        sh1, sh2 = st.columns(2)
        sh1.metric(
            "Rule 대비 절감",
            f"{summary['hybrid_savings_pct']:+.2f}%",
            f"{summary['hybrid_savings_kwh']:+,.1f} kWh",
            delta_color="off",
        )
        delta_text, delta_color = _safety_delta(summary['hybrid_violations'])
        sh2.metric(
            "온도 위반",
            f"{summary['hybrid_violations']} 회",
            delta_text,
            delta_color=delta_color,
        )

st.divider()


# ── 차트 빌더 (한 번 만들고 fragment에서 슬라이스만 갱신) ──────────────────
UIREV = f"ab-{ss.ab_scenario_key}"
X_MAX = float(df_rule["minute"].max())
PUE_MAX = max(df_rule["PUE"].max(), df_best["PUE"].max(), df_hybrid["PUE"].max()) * 1.05
# 누적 절감 = 규칙 - RL  (RL이 더 적은 전력을 쓰면 양수)
_savings_best   = df_rule["누적 kWh"] - df_best["누적 kWh"]
_savings_hybrid = df_rule["누적 kWh"] - df_hybrid["누적 kWh"]
SAVINGS_MIN = min(0.0, float(_savings_best.min()), float(_savings_hybrid.min())) * 1.1
SAVINGS_MAX = max(float(_savings_best.max()), float(_savings_hybrid.max()), 0.1) * 1.1


def _build_pue_chart(sub_rule: pd.DataFrame, sub_best: pd.DataFrame, sub_hybrid: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sub_rule["minute"], y=sub_rule["PUE"],
        name="Rule-based", line=dict(color=CLR_RULE, width=2),
    ))
    fig.add_trace(go.Scatter(
        x=sub_best["minute"], y=sub_best["PUE"],
        name="RL Best (효율)", line=dict(color=CLR_BEST, width=2),
    ))
    fig.add_trace(go.Scatter(
        x=sub_hybrid["minute"], y=sub_hybrid["PUE"],
        name="RL Hybrid (안전)", line=dict(color=CLR_HYBRID, width=2.5),
    ))
    fig.update_layout(
        title="PUE 비교 — RL vs Rule-based (낮을수록 best)",
        height=280, margin=dict(l=10, r=10, t=40, b=10),
        xaxis_title="경과 시간 (분)", yaxis_title="PUE",
        xaxis=dict(range=[0, X_MAX], fixedrange=True),
        yaxis=dict(range=[1.0, PUE_MAX], fixedrange=True),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        uirevision=f"{UIREV}-pue",
        transition={"duration": 0},
    )
    return fig


def _build_race_chart(sub_rule: pd.DataFrame, sub_best: pd.DataFrame, sub_hybrid: pd.DataFrame) -> go.Figure:
    """Best/Hybrid 각각의 Rule-based 대비 누적 절감 (양수 = RL 우세)."""
    savings_best   = sub_rule["누적 kWh"].values - sub_best["누적 kWh"].values
    savings_hybrid = sub_rule["누적 kWh"].values - sub_hybrid["누적 kWh"].values
    fig = go.Figure()
    fig.add_hline(y=0, line_color="rgba(128,128,128,0.4)", line_width=1)
    fig.add_trace(go.Scatter(
        x=sub_best["minute"], y=savings_best,
        name="Best 누적 절감", line=dict(color=CLR_BEST, width=2.5),
        hovertemplate="경과 %{x}분<br>Best 절감 %{y:.2f} kWh<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=sub_hybrid["minute"], y=savings_hybrid,
        name="Hybrid 누적 절감", line=dict(color=CLR_HYBRID, width=3),
        fill="tozeroy", fillcolor="rgba(46,204,113,0.20)",
        hovertemplate="경과 %{x}분<br>Hybrid 절감 %{y:.2f} kWh<extra></extra>",
    ))
    fig.update_layout(
        title="누적 절감 전력 — Best vs Hybrid (Rule-based 기준)",
        height=300, margin=dict(l=10, r=10, t=40, b=10),
        xaxis_title="경과 시간 (분)", yaxis_title="누적 절감 (kWh)",
        xaxis=dict(range=[0, X_MAX], fixedrange=True),
        yaxis=dict(range=[SAVINGS_MIN, SAVINGS_MAX], fixedrange=True),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        uirevision=f"{UIREV}-race",
        transition={"duration": 0},
    )
    return fig


# ── 재생 컨트롤 (fragment 바깥 — 버튼은 즉시 반응해야 함) ──────────────────
c1, c2, c3, c4 = st.columns([3, 1, 1, 1])
with c1:
    speed_label = st.select_slider(
        "재생 속도",
        options=list(ANIM_SPEED_OPTIONS.keys()),
        value="매우 빠르게 (0.1초)",
        key="ab_speed_label",
    )
    ss.ab_speed = ANIM_SPEED_OPTIONS[speed_label]
with c2:
    if st.button("▶ 재생", disabled=ss.ab_running, width="stretch"):
        if ss.ab_step >= n_steps - 1:
            ss.ab_step = 0
        ss.ab_running = True
        st.rerun()
with c3:
    if st.button("⏸ 일시정지", disabled=not ss.ab_running, width="stretch"):
        ss.ab_running = False
        st.rerun()
with c4:
    if st.button("⏹ 정지", width="stretch"):
        ss.ab_running = False
        ss.ab_step = 0
        st.rerun()


# ── 동적 영역 (fragment) ─────────────────────────────────────────────────
# run_every=0.1초로 fragment만 자동 폴링 (페이지 전체 rerun X) → 부드러운 갱신.
# 사용자가 고른 ab_speed에 맞춰 fragment 내부에서 step 진행을 throttle.
@st.fragment(run_every="100ms")
def playback_fragment():
    """재생 중 갱신되는 모든 위젯을 이 fragment에 격리.

    Streamlit fragment의 run_every가 자동 폴링을 수행 →
    헤더/버튼/시나리오 카드는 가만히 있고 차트·카운터만 부드럽게 갱신됨.
    """
    # 재생 중이면 ab_speed 간격으로 step 진행 (time 기반 throttle)
    if ss.ab_running and ss.ab_step < n_steps - 1:
        now = time.time()
        if now - ss.ab_last_advance >= ss.ab_speed:
            stride = max(1, n_steps // 200)
            ss.ab_step = min(ss.ab_step + stride, n_steps - 1)
            ss.ab_last_advance = now

    idx = min(ss.ab_step, n_steps - 1)
    cur_rule   = df_rule.iloc[idx]
    cur_best   = df_best.iloc[idx]
    cur_hybrid = df_hybrid.iloc[idx]

    # 진행바
    progress_pct = (idx + 1) / n_steps
    elapsed_min = idx * 5
    st.progress(
        progress_pct,
        text=f"{idx + 1}/{n_steps} step  ·  경과 {elapsed_min // 60:02d}:{elapsed_min % 60:02d} (5분 간격)",
    )

    # 라이브 카운터: Rule/Best/Hybrid 누적 kWh + Hybrid 기준 절감액
    st.markdown("##### :material/speed: 누적 카운터")
    save_best_kwh   = float(cur_rule["누적 kWh"] - cur_best["누적 kWh"])
    save_hybrid_kwh = float(cur_rule["누적 kWh"] - cur_hybrid["누적 kWh"])
    save_hybrid_krw = float(cur_rule["누적 원"] - cur_hybrid["누적 원"])

    lc1, lc2, lc3, lc4 = st.columns(4)
    lc1.metric("Rule 누적 kWh",   f"{cur_rule['누적 kWh']:,.1f}")
    lc2.metric("Best 누적 kWh",   f"{cur_best['누적 kWh']:,.1f}",   f"{-save_best_kwh:+,.1f}")
    lc3.metric("Hybrid 누적 kWh", f"{cur_hybrid['누적 kWh']:,.1f}", f"{-save_hybrid_kwh:+,.1f}")
    lc4.metric("Hybrid 누적 절감액", f"₩ {save_hybrid_krw:,.0f}")

    # 시간축 비교 (현재 step까지만 슬라이스)
    sub_rule   = df_rule.iloc[: idx + 1]
    sub_best   = df_best.iloc[: idx + 1]
    sub_hybrid = df_hybrid.iloc[: idx + 1]

    st.markdown("##### :material/timeline: 시간축 비교")
    st.plotly_chart(_build_pue_chart(sub_rule, sub_best, sub_hybrid),  width="stretch", key="chart_pue")
    st.plotly_chart(_build_race_chart(sub_rule, sub_best, sub_hybrid), width="stretch", key="chart_race")

    # 종료 시 임팩트 메시지 — Best/Hybrid 비교 강조
    if ss.ab_step >= n_steps - 1:
        ss.ab_running = False
        best_pct   = summary["best_savings_pct"]
        hybrid_pct = summary["hybrid_savings_pct"]
        best_viol  = summary["best_violations"]
        hyb_viol   = summary["hybrid_violations"]
        st.success(
            f"**시뮬레이션 종료** — "
            f"Best **{best_pct:+.2f}%** 절감 (위반 {best_viol}회) · "
            f"Hybrid **{hybrid_pct:+.2f}%** 절감 (위반 {hyb_viol}회). "
            f"Hybrid 연 환산 절감액 추정 ₩{summary['hybrid_savings_krw']:,.0f} · "
            f"CO₂ {summary['hybrid_savings_co2']:.1f}kg",
            icon=":material/celebration:",
        )

    # 재생 루프는 @st.fragment(run_every=...)가 자동 처리. 명시 호출 불필요.


playback_fragment()
