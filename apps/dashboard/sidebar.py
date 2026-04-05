"""
공통 사이드바 렌더링 모듈

모든 페이지에서 `render_sidebar()` 를 호출하면:
  - 시뮬레이션 파라미터 UI를 렌더링하고
  - 시뮬레이션을 실행한 뒤
  - 파생 지표를 담은 dict를 반환한다.
"""

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

from apps.dashboard.styles import inject_global_styles
from apps.dashboard.api_client import (
    get_all_service_status,
    optimize_control,
    rl_control,
)
from apps.dashboard.constants import (
    CRISIS_CONFIGS,
    SCENARIO_TEMP_PROFILES,
    TEMP_WARNING_THRESHOLD_C,
)
from apps.dashboard.simulation import (
    calculate_esg,
    run_simulation,
    simulate_rack_temperatures,
)


# ── 세션 상태 초기화 ──────────────────────────────────────────────────────

def init_session_state() -> None:
    defaults = {
        "crisis_mode":    None,
        "prev_crisis":    None,
        "prev_temp_warn": False,
        "alarms":         [],
        "anim_running":   False,
        "anim_hour":      0,
        "anim_speed":     0.4,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ── 알람 ─────────────────────────────────────────────────────────────────

def add_alarm(level: str, message: str) -> None:
    st.session_state.alarms.insert(0, {
        "time":    datetime.now().strftime("%H:%M:%S"),
        "level":   level,
        "message": message,
    })
    if len(st.session_state.alarms) > 15:
        st.session_state.alarms.pop()


# ── 서비스 상태 (캐시: 10초 TTL) ─────────────────────────────────────────

@st.cache_data(ttl=10, show_spinner=False)
def _cached_service_status() -> dict[str, bool]:
    return get_all_service_status()


# ── 공통 사이드바 렌더링 ──────────────────────────────────────────────────

def render_sidebar() -> dict:
    """사이드바를 렌더링하고 시뮬레이션 결과 dict를 반환한다."""
    init_session_state()
    inject_global_styles()

    def _sec(title: str) -> None:
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:7px;margin:14px 0 6px;">'
            f'<div style="width:3px;height:13px;background:#5D90FF;border-radius:2px;flex-shrink:0;"></div>'
            f'<span style="font-size:0.85rem;font-weight:700;color:#5D90FF;'
            f'text-transform:uppercase;letter-spacing:0.07em;">{title}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with st.sidebar:
        st.markdown(
            '<div style="display:flex;align-items:center;gap:10px;padding:4px 0 12px;">'
            '<div style="width:10px;height:10px;border-radius:50%;background:#5D90FF;'
            'box-shadow:0 0 6px #5D90FF88;"></div>'
            '<span style="font-size:1.05rem;font-weight:700;letter-spacing:0.04em;">'
            '시뮬레이션 파라미터</span>'
            '</div>',
            unsafe_allow_html=True,
        )

        _sec("시나리오")
        scenario = st.selectbox("시나리오", list(SCENARIO_TEMP_PROFILES.keys()), index=0,
                                label_visibility="collapsed")

        st.divider()
        _sec("서버 구성")
        num_cpu  = st.slider("CPU 서버 대수", 100, 1000, 400, step=50)
        num_gpu  = st.slider("GPU 서버 대수", 0, 200, 20, step=10)
        cpu_util = st.slider("평균 CPU 사용률 (%)", 10, 100, 60) / 100.0

        st.divider()
        _sec("냉각 설정")
        supply_temp = st.slider("CRAH 공급 온도 (°C)", 14, 24, 18)

        st.divider()
        _sec("위기 시나리오")
        c0, c1 = st.columns(2)
        c2, c3 = st.columns(2)
        if c0.button("정상",     use_container_width=True):
            st.session_state.crisis_mode = None
        if c1.button("서버 급증", use_container_width=True):
            st.session_state.crisis_mode = "server_surge"
        if c2.button("냉각 고장", use_container_width=True):
            st.session_state.crisis_mode = "chiller_failure"
        if c3.button("폭염",     use_container_width=True):
            st.session_state.crisis_mode = "heat_wave"

        crisis_mode = st.session_state.crisis_mode
        cfg         = CRISIS_CONFIGS[crisis_mode]

        if crisis_mode:
            st.markdown(
                f'<div style="background:rgba(239,68,68,0.07);'
                f'border:1px solid rgba(239,68,68,0.3);border-left:3px solid #ef4444;'
                f'border-radius:6px;padding:7px 10px;margin-top:6px;">'
                f'<div style="font-size:0.7rem;color:#ef4444;font-weight:700;'
                f'text-transform:uppercase;letter-spacing:0.05em;margin-bottom:2px;">위기 모드 활성</div>'
                f'<div style="font-size:0.88rem;font-weight:600;">⚠️ {cfg["label"]}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="background:rgba(34,197,94,0.07);'
                'border:1px solid rgba(34,197,94,0.25);border-left:3px solid #22c55e;'
                'border-radius:6px;padding:7px 10px;margin-top:6px;">'
                '<div style="font-size:0.7rem;color:#22c55e;font-weight:700;'
                'text-transform:uppercase;letter-spacing:0.05em;margin-bottom:2px;">상태</div>'
                '<div style="font-size:0.88rem;font-weight:600;">정상 운영 중</div>'
                '</div>',
                unsafe_allow_html=True,
            )

        st.divider()
        _sec("서비스 상태")
        for name, is_ok in _cached_service_status().items():
            color  = "#22c55e" if is_ok else "#ef4444"
            label  = "Online"  if is_ok else "Offline"
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;align-items:center;'
                f'padding:5px 0;border-bottom:1px solid rgba(128,128,128,0.1);">'
                f'<span style="font-size:0.92rem;">{name}</span>'
                f'<span style="display:flex;align-items:center;gap:5px;font-size:0.85rem;'
                f'color:{color};font-weight:600;">'
                f'<span style="width:6px;height:6px;border-radius:50%;background:{color};'
                f'display:inline-block;"></span>{label}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ── 시뮬레이션 실행 ──────────────────────────────────────────────────

    df_full = run_simulation(scenario, num_cpu, num_gpu, cpu_util, supply_temp, crisis_mode)
    df      = df_full

    avg_pue     = df["PUE"].mean()
    avg_cop     = df["COP"].mean()
    avg_it      = df["IT 전력 (kW)"].mean()
    avg_cooling = df["칠러 전력 (kW)"].mean()
    avg_total   = df["총 전력 (kW)"].mean()

    peak_idx         = df["IT 전력 (kW)"].idxmax()
    peak_return_temp = df.loc[peak_idx, "환기 온도 (°C)"]
    max_return_temp  = df["환기 온도 (°C)"].max()
    current_mode     = df.loc[peak_idx, "냉각 모드"]
    current_outdoor  = df.loc[peak_idx, "외기온도 (°C)"]
    peak_it_power    = df.loc[peak_idx, "IT 전력 (kW)"]

    rack_labels, rack_temps, rack_colors = simulate_rack_temperatures(peak_return_temp)
    over_threshold = sum(1 for t in rack_temps if t > TEMP_WARNING_THRESHOLD_C)

    esg = calculate_esg(df)

    if crisis_mode:
        df_base      = run_simulation(scenario, num_cpu, num_gpu, cpu_util, supply_temp, None)
        esg_base     = calculate_esg(df_base)
        avg_pue_base = df_base["PUE"].mean()
    else:
        df_base = esg_base = avg_pue_base = None

    # TODO(RL): ctrl_rl 응답은 현재 고정값 반환 중 (cooling_mode="hybrid", expected_pue=1.35).
    #   Week 4 PPO/DQN RL 에이전트 구현 후 실제 추론값으로 자동 교체됨.
    # TODO(Simulation Service): ctrl_rule 응답의 expected_pue도 고정값 1.35.
    #   Simulation Service 연동 후 실측값으로 교체됨.
    ctrl_rule = optimize_control(current_outdoor, peak_it_power)
    ctrl_rl   = rl_control(current_outdoor, peak_it_power)

    # 위기 모드 전환 알람
    if crisis_mode != st.session_state.prev_crisis:
        if crisis_mode:
            add_alarm("WARN", f"위기 시나리오 주입: {cfg['label']}")
        else:
            add_alarm("INFO", "정상 모드로 복귀")
        st.session_state.prev_crisis = crisis_mode

    temp_warn = max_return_temp > TEMP_WARNING_THRESHOLD_C
    if temp_warn != st.session_state.prev_temp_warn:
        if temp_warn:
            add_alarm("ERROR", f"서버 온도 경고: 최고 {max_return_temp:.1f}°C (기준 {TEMP_WARNING_THRESHOLD_C}°C 초과)")
        else:
            add_alarm("INFO", "서버 온도 정상 범위로 복귀")
        st.session_state.prev_temp_warn = temp_warn

    return {
        "scenario":        scenario,
        "num_cpu":         num_cpu,
        "num_gpu":         num_gpu,
        "cpu_util":        cpu_util,
        "supply_temp":     supply_temp,
        "crisis_mode":     crisis_mode,
        "cfg":             cfg,
        "df":              df,
        "df_full":         df_full,
        "df_base":         df_base,
        "avg_pue":         avg_pue,
        "avg_cop":         avg_cop,
        "avg_it":          avg_it,
        "avg_cooling":     avg_cooling,
        "avg_total":       avg_total,
        "peak_idx":        peak_idx,
        "peak_return_temp": peak_return_temp,
        "max_return_temp": max_return_temp,
        "current_mode":    current_mode,
        "current_outdoor": current_outdoor,
        "peak_it_power":   peak_it_power,
        "rack_labels":     rack_labels,
        "rack_temps":      rack_temps,
        "rack_colors":     rack_colors,
        "over_threshold":  over_threshold,
        "esg":             esg,
        "esg_base":        esg_base,
        "avg_pue_base":    avg_pue_base,
        "ctrl_rule":       ctrl_rule,
        "ctrl_rl":         ctrl_rl,
    }
