"""
분석 도구 — 예측 결과 · 3D Datacenter View

LGBM 예측 결과 + 데이터센터 플로어 3D 시각화
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from apps.dashboard.api_client import call_forecast
from apps.dashboard.constants import CLR_CHILLER, CLR_IT, TEMP_WARNING_THRESHOLD_C
from apps.dashboard.sidebar import render_sidebar
from apps.dashboard.simulation import get_scenario_timestamp

d = render_sidebar()


@st.cache_data(ttl=300)
def _fetch_forecast(horizon_hours: int, current_timestamp: str, model_type: str = "lgbm") -> dict:
    return call_forecast(
        horizon_hours=horizon_hours,
        include_prediction_interval=True,
        model_type=model_type,
        current_timestamp=current_timestamp,
    )


st.title(":material/analytics: 분석 도구")
st.caption("LGBM 예측 결과 · 3D Datacenter View")
st.divider()

tab_anim, tab_3d = st.tabs(["예측 결과", "3D Datacenter View"])


@st.cache_data(ttl=3600, show_spinner=False)
def _load_forecast_metrics() -> dict | None:
    import json
    p = Path("data/eval/forecast_metrics.json")
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _kpi(col, label, value, spec, ok, unit="%", help_text=None):
    """기준 충족/미달을 색상 분기로 표시하는 KPI 카드."""
    if value is None:
        col.metric(label, "—", "측정 미실행", help=help_text)
        return
    col.metric(
        label,
        f"{value:.2f}{unit}" if unit else f"{value:.2f}",
        f"{'통과' if ok else '미달'} (기준 {spec}{unit})",
        delta_color="normal" if ok else "inverse",
        help=help_text,
    )


# ── 예측 결과 탭 ─────────────────────────────────────────────────────────

with tab_anim:
    # ── Forecast 모델 성능 KPI (명세 §4-4, §4-5 충족 가시화) ────────────
    _metrics = _load_forecast_metrics()
    st.markdown("##### :material/speed: 예측 모델 성능")

    fc_kpi_cols = st.columns(4)
    if _metrics is None or _metrics.get("status") == "unavailable":
        for col, lbl in zip(fc_kpi_cols, ["MAPE (24h)", "MAPE (168h)", "커버리지 (90% PI)", "냉각 nMAE"]):
            col.metric(lbl, "—", "측정 미실행")
        st.caption(
            ":material/info: `bash domain/forecasting/train/train_all.sh && uv run python -m scripts.eval_forecast_metrics`"
        )
    else:
        # CV(누수 없음) 우선 사용. CV 없으면 holdout(보조) 값으로 폴백
        cv = _metrics.get("cv", {}) or {}
        cv_it   = cv.get("lgbm_it_load") or {}
        cv_cool = cv.get("lgbm_cooling_demand") or {}
        cv_qit  = cv.get("lgbm_quantile_it_load") or {}

        holdout = _metrics.get("holdout", {}) or {}
        h_it    = holdout.get("it_load", {}) or {}
        h_cool  = holdout.get("cooling_demand", {}) or {}
        h_cov   = holdout.get("it_load_coverage", {}) or {}

        # CV 우선 / 없으면 holdout
        m24       = cv_it.get("mape_24h_pct")    if cv_it   else h_it.get("mape_24h_pct")
        m168      = cv_it.get("mape_168h_pct")   if cv_it   else h_it.get("mape_168h_pct")
        cool_nmae = cv_cool.get("nmae_valid_pct") if cv_cool else h_cool.get("nmae_pct")
        cov_pct   = cv_qit.get("coverage_90_pct") if cv_qit  else h_cov.get("coverage_90_pct")

        _kpi(fc_kpi_cols[0], "MAPE (24h)",  m24,  5.0,
             ok=(m24 is not None and m24 <= 5.0),
             help_text="IT 부하 LGBM 24h ahead 예측 평균 절대 백분율 오차 (월별 CV 평균)")
        _kpi(fc_kpi_cols[1], "MAPE (168h)", m168, 8.0,
             ok=(m168 is not None and m168 <= 8.0),
             help_text="IT 부하 LGBM 168h ahead 예측 평균 절대 백분율 오차 (월별 CV 평균)")
        _kpi(fc_kpi_cols[2], "커버리지 (90% PI)", cov_pct, 85.0,
             ok=(cov_pct is not None and cov_pct >= 85.0),
             help_text="실제값이 [5%, 95%] 예측 구간 안에 들어간 비율 (월별 CV 평균)")
        _kpi(fc_kpi_cols[3], "냉각 nMAE", cool_nmae, 10.0,
             ok=(cool_nmae is not None and cool_nmae <= 10.0),
             help_text="냉각 수요 LGBM 정규화 MAE — free cooling 압도적 월은 제외하고 평균")

    st.divider()

    fc_model_label = "LGBM"
    _fc    = _fetch_forecast(horizon_hours=24, current_timestamp=get_scenario_timestamp(d["scenario"]), model_type="lgbm")
    _fc_ok = "error" not in _fc

    if not _fc_ok:
        st.info(
            "**Forecast Service 오프라인**: 서비스 연결 후 실제 예측값(`POST /api/v1/forecast`)이 자동으로 표시됩니다.",
        )

    if _fc_ok:
        _preds   = _fc.get("predictions", [])
        _n       = len(_preds)
        # Forecast Service는 5분 step 단위로 응답하므로 step 인덱스를 시간으로 환산.
        _STEPS_PER_HOUR = 12
        _hours_fc = [i / _STEPS_PER_HOUR for i in range(_n)]
        _total_hours = _n / _STEPS_PER_HOUR
        if _preds:
            _it      = [p.get("predicted_it_load_kw")          or 0 for p in _preds]
            _it_lo   = [p.get("lower_bound_it_load_kw")        or 0 for p in _preds]
            _it_hi   = [p.get("upper_bound_it_load_kw")        or 0 for p in _preds]
            _cool    = [p.get("predicted_cooling_load_kw")     or 0 for p in _preds]
            _cool_lo = [p.get("lower_bound_cooling_load_kw")   or 0 for p in _preds]
            _cool_hi = [p.get("upper_bound_cooling_load_kw")   or 0 for p in _preds]

            _dtick = 4 if _total_hours <= 48 else (12 if _total_hours <= 96 else 24)
            _model_used = _fc.get("model_type_used", "")

            fig_fc = go.Figure()
            fig_fc.add_trace(go.Scatter(
                x=_hours_fc + _hours_fc[::-1], y=_it_hi + _it_lo[::-1],
                fill="toself", fillcolor="rgba(251,191,36,0.12)",
                line=dict(color="rgba(0,0,0,0)"), showlegend=False, hoverinfo="skip",
            ))
            fig_fc.add_trace(go.Scatter(
                x=_hours_fc, y=_it,
                name=f"IT 부하 — {fc_model_label}", line=dict(color=CLR_IT, width=2),
            ))
            fig_fc.add_trace(go.Scatter(
                x=_hours_fc + _hours_fc[::-1], y=_cool_hi + _cool_lo[::-1],
                fill="toself", fillcolor="rgba(163,171,251,0.12)",
                line=dict(color="rgba(0,0,0,0)"), showlegend=False, hoverinfo="skip",
            ))
            fig_fc.add_trace(go.Scatter(
                x=_hours_fc, y=_cool,
                name=f"냉각 수요 — {fc_model_label}", line=dict(color=CLR_CHILLER, width=2),
            ))
            fig_fc.update_layout(
                title=f"IT 부하 / 냉각 수요 예측 (kW) — Forecast Service · {_model_used} · {int(_total_hours)}h",
                height=320,
                margin=dict(t=40, b=70, l=10, r=10),
                xaxis=dict(title="시간 (h)", range=[-0.5, _total_hours - 0.5], dtick=_dtick),
                yaxis_title="전력 (kW)",
                legend=dict(orientation="h", y=-0.32),
            )
            with st.container(border=True):
                st.plotly_chart(fig_fc, width="stretch")

# ── 3D 데이터센터 뷰 탭 ───────────────────────────────────────────────────

def _build_rack_mesh3d(data: np.ndarray, cmin: float, cmax: float,
                       colorscale, unit: str, title: str,
                       n_zones: int = 3, zone_rows: int = 2) -> go.Figure:
    """Zone 격자 데이터를 3D 직육면체 랙 군집으로 시각화 + Zone 구분/라벨 추가.

    각 셀(r, c) = 하나의 랙 큐브. 색상 = data[r, c] 값. 8 vertices × 12 triangles per rack.
    실제 42U 랙 비율(0.6w × 1.1d × 2.0h), 통로 갭, 양 끝 CRAH 유닛 포함.
    """
    n_rows, n_cols = data.shape

    # 실제 42U 랙 비율 (m): 0.6 wide × 1.1 deep × 2.0 tall
    rack_dx, rack_dy, rack_dz = 0.6, 1.1, 2.0
    rack_gap_x = 0.3                       # 같은 줄 내 랙 사이 간격
    spacing_x  = rack_dx + rack_gap_x      # 0.9
    aisle_gap  = 0.9                       # zone 내 cold/hot 통로
    zone_gap   = 1.6                       # zone 간 분리

    all_x, all_y, all_z = [], [], []
    all_i, all_j, all_k = [], [], []
    all_intensity, all_text = [], []
    v = 0  # vertex offset

    # 8개 vertex 인덱스 패턴 (밑면 0~3, 윗면 4~7)
    BASE_I = [0, 0, 0, 0, 4, 4, 1, 1, 2, 2, 3, 3]
    BASE_J = [1, 2, 4, 3, 5, 6, 2, 5, 6, 3, 0, 7]
    BASE_K = [2, 3, 5, 4, 6, 7, 5, 6, 7, 7, 4, 4]

    ZONE_NAMES   = ["Zone A", "Zone B", "Zone C"]
    AISLE_LABELS = ["Cold", "Hot"]

    # y 위치 사전 계산: zone 내 행 간격(aisle_gap), zone 간 분리(zone_gap)
    y_positions = []
    y_cursor = 0.0
    for zone_i in range(n_zones):
        for ria in range(zone_rows):
            y_positions.append(y_cursor)
            if ria < zone_rows - 1:
                y_cursor += rack_dy + aisle_gap
        if zone_i < n_zones - 1:
            y_cursor += rack_dy + zone_gap

    for r in range(n_rows):
        zone_i = r // zone_rows
        aisle  = AISLE_LABELS[r % zone_rows] if zone_rows == 2 else f"Row {r % zone_rows + 1}"
        for c in range(n_cols):
            cx = c * spacing_x
            cy = y_positions[r]
            x0, x1 = cx - rack_dx / 2, cx + rack_dx / 2
            y0, y1 = cy - rack_dy / 2, cy + rack_dy / 2
            z0, z1 = 0.0, rack_dz

            all_x.extend([x0, x1, x1, x0, x0, x1, x1, x0])
            all_y.extend([y0, y0, y1, y1, y0, y0, y1, y1])
            all_z.extend([z0, z0, z0, z0, z1, z1, z1, z1])

            val = float(data[r, c])
            all_intensity.extend([val] * 8)
            zone_lab = ZONE_NAMES[zone_i] if zone_i < len(ZONE_NAMES) else f"Zone {zone_i+1}"
            all_text.extend([f"{zone_lab} · {aisle} aisle<br>R{c+1:02d}<br>{val:.2f} {unit}"] * 8)

            all_i.extend([i + v for i in BASE_I])
            all_j.extend([j + v for j in BASE_J])
            all_k.extend([k + v for k in BASE_K])
            v += 8

    fig = go.Figure(data=[go.Mesh3d(
        x=all_x, y=all_y, z=all_z,
        i=all_i, j=all_j, k=all_k,
        intensity=all_intensity,
        colorscale=colorscale,
        cmin=cmin, cmax=cmax,
        flatshading=True,
        showscale=True,
        colorbar=dict(title=unit, thickness=14, len=0.7),
        hovertext=all_text, hoverinfo="text",
        lighting=dict(ambient=0.55, diffuse=0.85, specular=0.15, roughness=0.7),
    )])

    # ── Zone별 색상 바닥 (Zone 영역을 가시화) ─────────────────────────────
    ZONE_COLORS = [
        "rgba(93, 144, 255, 0.18)",   # Zone A — 블루
        "rgba(163, 171, 251, 0.18)",  # Zone B — 인디고
        "rgba(123, 210, 247, 0.18)",  # Zone C — 스카이
    ]
    zone_x0 = -rack_dx - 0.5
    zone_x1 = (n_cols - 1) * spacing_x + rack_dx + 0.5
    margin_y = rack_dy / 2 + 0.3
    for zone_i in range(n_zones):
        rows_in_zone = list(range(zone_i * zone_rows, (zone_i + 1) * zone_rows))
        y_first = y_positions[rows_in_zone[0]] - margin_y
        y_last  = y_positions[rows_in_zone[-1]] + margin_y
        floor_color = ZONE_COLORS[zone_i % len(ZONE_COLORS)]
        fig.add_trace(go.Mesh3d(
            x=[zone_x0, zone_x1, zone_x1, zone_x0],
            y=[y_first, y_first, y_last, y_last],
            z=[0, 0, 0, 0],
            i=[0, 0], j=[1, 2], k=[2, 3],
            color=floor_color, opacity=1.0,
            showscale=False, hoverinfo="skip",
        ))

    # ── Zone 라벨 (3D 텍스트) ────────────────────────────────────────────
    label_x = zone_x1 + 0.6
    label_z = rack_dz + 0.5
    for zone_i in range(n_zones):
        if zone_i >= len(ZONE_NAMES):
            break
        rows_in_zone = list(range(zone_i * zone_rows, (zone_i + 1) * zone_rows))
        label_y = (y_positions[rows_in_zone[0]] + y_positions[rows_in_zone[-1]]) / 2
        accent = ZONE_COLORS[zone_i].replace("0.18", "1.0")
        fig.add_trace(go.Scatter3d(
            x=[label_x], y=[label_y], z=[label_z],
            mode="text",
            text=[f"<b>{ZONE_NAMES[zone_i]}</b>"],
            textfont=dict(size=15, color=accent),
            hoverinfo="skip", showlegend=False,
        ))

    # ── CRAH 유닛 (양 끝에 회색 큐브, 데이터센터 분위기) ──────────────────
    def _add_box(x0, x1, y0, y1, z0, z1, color="rgba(120,120,135,0.85)", label=None):
        verts_x = [x0, x1, x1, x0, x0, x1, x1, x0]
        verts_y = [y0, y0, y1, y1, y0, y0, y1, y1]
        verts_z = [z0, z0, z0, z0, z1, z1, z1, z1]
        fig.add_trace(go.Mesh3d(
            x=verts_x, y=verts_y, z=verts_z,
            i=BASE_I, j=BASE_J, k=BASE_K,
            color=color, opacity=0.95, flatshading=True,
            showscale=False,
            hovertext=label or "CRAH", hoverinfo="text",
            lighting=dict(ambient=0.6, diffuse=0.7),
        ))

    crah_y_full_start = y_positions[0] - margin_y
    crah_y_full_end   = y_positions[-1] + margin_y
    crah_h            = 2.6  # 랙보다 약간 더 높음
    crah_w            = 0.9

    # 좌측 CRAH 2대 (상/하 분할)
    mid_y = (crah_y_full_start + crah_y_full_end) / 2
    for y0, y1, idx in [
        (crah_y_full_start, mid_y - 0.3, 1),
        (mid_y + 0.3, crah_y_full_end, 2),
    ]:
        _add_box(
            zone_x0 - crah_w - 0.2, zone_x0 - 0.2,
            y0, y1, 0.0, crah_h,
            label=f"CRAH-{idx}",
        )
    # 우측 CRAH 2대 (상/하 분할)
    for y0, y1, idx in [
        (crah_y_full_start, mid_y - 0.3, 3),
        (mid_y + 0.3, crah_y_full_end, 4),
    ]:
        _add_box(
            zone_x1 + 0.2, zone_x1 + crah_w + 0.2,
            y0, y1, 0.0, crah_h,
            label=f"CRAH-{idx}",
        )

    # y 범위가 더 커졌으니 aspect 비율 재계산 (x: 0~9, y: 0~10+ 범위)
    scene_x = (n_cols - 1) * spacing_x + 2 * crah_w + 1.5
    scene_y = crah_y_full_end - crah_y_full_start
    aspect_x = scene_x / max(scene_x, scene_y) * 2.4
    aspect_y = scene_y / max(scene_x, scene_y) * 2.4

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(title="랙 열", showticklabels=False, showgrid=False, zeroline=False),
            yaxis=dict(title="Zone / 통로", showticklabels=False, showgrid=False, zeroline=False),
            zaxis=dict(title="", showticklabels=False, showgrid=False, zeroline=False),
            aspectmode="manual",
            aspectratio=dict(x=aspect_x, y=aspect_y, z=0.55),
            camera=dict(eye=dict(x=1.6, y=-2.0, z=1.15)),
            bgcolor="rgba(0,0,0,0)",
        ),
        height=640,
        margin=dict(l=0, r=0, t=40, b=0),
        uirevision="3d-datacenter",
    )
    return fig


with tab_3d:
    hour_3d = st.slider("시간 (h)", 0, 23, 14, key="3d_hour",
                        help="해당 시각의 시뮬레이션 스냅샷을 3D로 표시")

    # 시뮬레이션 결과에서 해당 시간 행 추출
    df_full        = d["df"]
    row_h          = df_full.iloc[hour_3d]
    supply_temp_3d = d["supply_temp"]
    hour_return_t  = float(row_h["환기 온도 (°C)"])

    N_ZONES_3D    = 3
    ZONE_ROWS_3D  = 2
    COLS_3D       = 10
    TOTAL_ROWS_3D = N_ZONES_3D * ZONE_ROWS_3D  # 6
    ZONE_FACTORS  = np.array([1.2, 1.0, 0.8])  # Zone별 부하 가중 (A 고밀도, C 저밀도)

    # 같은 hour엔 동일 seed → 슬라이더 움직일 때 깔끔하게 변화 (시간별 잡음 분리)
    rng_3d  = np.random.default_rng(seed=42 + hour_3d)
    data_3d = np.zeros((TOTAL_ROWS_3D, COLS_3D))

    # 온도 물리식: cold aisle 흡입 ≈ supply + 작은 혼합 손실, hot aisle 토출 ≈ return
    # 열 방향(column 0→9)으로 누적 가열 ↑
    delta_t = max(0.5, hour_return_t - supply_temp_3d)
    for zone_i in range(N_ZONES_3D):
        zf      = ZONE_FACTORS[zone_i]
        cold_in = supply_temp_3d + 1.5 * zf
        hot_out = supply_temp_3d + delta_t * zf
        grad    = np.linspace(0.0, 1.0, COLS_3D)
        data_3d[zone_i * ZONE_ROWS_3D]     = cold_in + grad * (1.0 * zf)
        data_3d[zone_i * ZONE_ROWS_3D + 1] = hot_out + grad * (2.0 * zf)
    data_3d += rng_3d.normal(0, 0.4, data_3d.shape)  # 센서 잡음 ±0.4°C

    unit_3d = "°C"
    cmin_3d = supply_temp_3d - 1
    cmax_3d = max(supply_temp_3d + 18.0, hour_return_t + 5.0)

    # 발산형 컬러 스케일 — supply_temp 부근은 흰색(=안전), 경고 임계에서 노랑, 그 위는 빨강
    def _norm(t: float) -> float:
        return max(0.0, min(1.0, (t - cmin_3d) / (cmax_3d - cmin_3d)))

    supply_anchor = _norm(supply_temp_3d)
    warn_anchor   = _norm(TEMP_WARNING_THRESHOLD_C)  # 안전 한계
    colorscale_3d = [
        [0.0,                                 "#1e3a8a"],   # 매우 차가움 (deep blue)
        [max(0.001, supply_anchor - 0.04),    "#93c5fd"],   # cold
        [supply_anchor,                        "#f8fafc"],  # supply ≈ 흰색
        [min(0.999, supply_anchor + 0.04),    "#fef3c7"],   # 살짝 따뜻
        [max(0.001, warn_anchor),              "#fbbf24"],  # 경고 임계 (amber)
        [min(0.999, warn_anchor + 0.10),      "#ef4444"],   # 위반 (red)
        [1.0,                                 "#7f1d1d"],   # 최대 (dark crimson)
    ]
    # plotly가 anchor 정렬을 요구하므로 정렬
    colorscale_3d = sorted(colorscale_3d, key=lambda x: x[0])

    title_3d = f"3D 온도 분포 — {hour_3d:02d}:00 (외기 {row_h['외기온도 (°C)']:.1f}°C)"

    fig_3d = _build_rack_mesh3d(data_3d, cmin_3d, cmax_3d, colorscale_3d, unit_3d, title_3d)
    st.plotly_chart(fig_3d, width="stretch")

    # 핫스팟 / 안전 알림
    hot_count = int((data_3d > TEMP_WARNING_THRESHOLD_C).sum())
    if hot_count:
        st.warning(
            f"{hour_3d:02d}:00 — {TEMP_WARNING_THRESHOLD_C:.0f}°C 초과 랙 **{hot_count}개** 감지. "
            f"Supply {supply_temp_3d}°C / Return {hour_return_t:.1f}°C / ΔT {hour_return_t - supply_temp_3d:+.1f}°C.",
            icon=":material/local_fire_department:",
        )
    else:
        st.success(
            f"{hour_3d:02d}:00 — 모든 랙 안전 영역 (≤ {TEMP_WARNING_THRESHOLD_C:.0f}°C). "
            f"ΔT {hour_return_t - supply_temp_3d:+.1f}°C, 평균 랙 온도 {data_3d.mean():.1f}°C.",
            icon=":material/check_circle:",
        )

    # ── Zone별 요약 ─────────────────────────────────────────────────────
    st.markdown("##### Zone Summary")

    ZONE_NAMES_3D   = ["Zone A", "Zone B", "Zone C"]
    ZONE_ACCENTS_3D = ["#5D90FF", "#A3ABFB", "#7BD2F7"]

    zone_cols = st.columns(N_ZONES_3D)
    for zone_i, (zone_name, col_ui) in enumerate(zip(ZONE_NAMES_3D, zone_cols)):
        zone_data = data_3d[zone_i * ZONE_ROWS_3D : (zone_i + 1) * ZONE_ROWS_3D]
        cold_data = zone_data[0]                # Cold Aisle row
        hot_data  = zone_data[1] if ZONE_ROWS_3D > 1 else zone_data[0]
        over      = int((zone_data > TEMP_WARNING_THRESHOLD_C).sum())
        accent    = ZONE_ACCENTS_3D[zone_i]
        with col_ui:
            with st.container(border=True):
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:10px;">'
                    f'<div style="width:8px;height:8px;border-radius:50%;background:{accent};"></div>'
                    f'<span style="font-size:1rem;font-weight:700;color:{accent};">{zone_name}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                m1, m2 = st.columns(2)
                m1.metric("Cold Aisle 평균", f"{cold_data.mean():.1f} °C")
                m2.metric("Hot Aisle 평균",  f"{hot_data.mean():.1f} °C")
                m3, m4 = st.columns(2)
                m3.metric("Max Temp", f"{zone_data.max():.1f} °C")
                m4.metric(
                    "At-risk Racks",
                    f"{over}",
                    delta=f"+{over}" if over else "Safe",
                    delta_color="inverse" if over else "normal",
                )
