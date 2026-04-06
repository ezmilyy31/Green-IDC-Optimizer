"""
PID 제어 시뮬레이터

ThermalSimulator(물리 환경) + PIDController(제어기)를 묶어 시간 루프를 실행한다.

역할:
  - run_pid_loop: PID 제어 루프 실행
  - run_fixed_supply_loop: 고정 공급 온도(No-PID) 루프 실행 (PID와의 비교용)

검증 방법론:
  Sinergym과의 직접 비교는 모델 범위 차이(solar gain, 외벽 등)로 불가하여,
  동일 ODE 환경 내에서 PID vs. 고정 공급온도(No-PID)를 대조 실험하는 방식으로 대체한다.
  → 외부 환경 차이가 아닌 제어기 유무만 변수로 분리된 검증.

`__main__` 블록:
  게인 튜닝 sweep (Ki → Kd) 후 최적 게인값으로 4개 위기 시나리오 검증 출력.
  시나리오: S1(부하 급증), S2(폭염), S3(CRAH 1대 고장), S4(칠러 1대 고장)
"""

import sys
import os

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from apps.simulation_service.thermal_simulator import (
    ThermalSimulator,
    ThermalSimulatorConfig,
    DEFAULT_M_DOT_KG_PER_S,
    DEFAULT_N_CRAH_ACTIVE,
    DEFAULT_C_EFF_KJ_PER_K,
)
from domain.controllers.pid import PIDController, PIDLoopResult
from domain.thermodynamics.cooling_load import (
    AIR_SPECIFIC_HEAT_KJ_PER_KG_K,
    calculate_cooling_load_from_airflow_kw,
)
from domain.thermodynamics.chiller import calculate_chiller_power_kw, calculate_cop
from domain.thermodynamics.it_power import calculate_total_it_power_kw


def _resolve_schedule(value: "list[float] | float", steps: int, name: str) -> list[float]:
    """단일 float 또는 list를 steps 길이의 list로 정규화한다."""
    if isinstance(value, list):
        if len(value) != steps:
            raise ValueError(f"{name} 길이({len(value)})가 steps({steps})와 다릅니다.")
        return value
    return [value] * steps


def run_pid_loop(
    steps: int,
    it_power_kw_schedule: list[float],
    pid: PIDController,
    sim: ThermalSimulator,
    outdoor_temp_c_schedule: "list[float] | float" = 22.0,
    n_crah_schedule: "list[int] | int | None" = None,
    zone_target_c: float = 24.0,
    chiller_scale_schedule: "list[float] | float" = 1.0,
) -> list[PIDLoopResult]:
    """
    PID 제어 루프를 steps만큼 실행한다.

    고정 setpoint: zone_target_c (서버실 목표 온도, 기본 24°C).

    Args:
        steps: 시뮬레이션 스텝 수
        it_power_kw_schedule: 스텝별 IT 발열량 목록 (길이 = steps)
        pid: PIDController 인스턴스
        sim: ThermalSimulator 인스턴스
        outdoor_temp_c_schedule: 스텝별 외기 온도 목록 또는 단일 고정값 (°C).
            COP 계산에 사용: COP = max(2.0, 6.0 - 0.1 × (T_out - 15))
        n_crah_schedule: 스텝별 가동 CRAH 대수 목록 또는 단일 고정값.
            None이면 sim.config.n_crah_units(전 대수 가동)을 사용.
            대수가 줄면 ṁ(공기 유량)과 칠러 용량이 비례해서 함께 감소한다.
        zone_target_c: 서버실 목표 온도 (°C). PID setpoint로 고정 사용. 기본 24.0°C.
        chiller_scale_schedule: 스텝별 칠러 용량 스케일 (0.0~1.0) 또는 단일 고정값.
            칠러 고장 시나리오에 사용. CRAH 대수와 독립적으로 칠러 전력 상한을 조절한다.
            예: 0.5 → 칠러 2대 중 1대 고장 (용량 50%).

    Returns:
        각 스텝의 PIDLoopResult 목록
    """
    if len(it_power_kw_schedule) != steps:
        raise ValueError(
            f"it_power_kw_schedule 길이({len(it_power_kw_schedule)})가 steps({steps})와 다릅니다."
        )

    outdoor_sched = _resolve_schedule(outdoor_temp_c_schedule, steps, "outdoor_temp_c_schedule")
    n_total = sim.config.n_crah_units
    _n_raw = n_crah_schedule if n_crah_schedule is not None else n_total
    n_sched = _resolve_schedule(_n_raw, steps, "n_crah_schedule")
    chiller_scale_sched = _resolve_schedule(chiller_scale_schedule, steps, "chiller_scale_schedule")

    # 정상 운영 평형 상태로 초기화
    # supply: T_zone = setpoint, Q_out = Q_in(첫 스텝) 기준 평형 공급온도
    # _prev_error = 0: 평형 상태에서 오차 변화 없음 → 첫 스텝 Kd 항 = 0
    m_dot_cp_init = sim.config.m_dot_kg_per_s * (DEFAULT_N_CRAH_ACTIVE / sim.config.n_crah_units) \
                    * AIR_SPECIFIC_HEAT_KJ_PER_KG_K
    supply_eq = zone_target_c - it_power_kw_schedule[0] / m_dot_cp_init
    pid._supply = max(pid.output_min, min(pid.output_max, supply_eq))
    pid._prev_error = 0.0

    results = []

    for t in range(steps):
        q_in_kw = it_power_kw_schedule[t]
        outdoor_temp_c = outdoor_sched[t]
        n_active = n_sched[t]
        chiller_scale = chiller_scale_sched[t]

        # CRAH 대수에 비례해 ṁ 계산 / 칠러는 CRAH 대수와 독립 (2N 설계)
        ratio = n_active / n_total
        m_dot_active = sim.config.m_dot_kg_per_s * ratio
        chiller_max_active = sim.config.chiller_design_kw * chiller_scale
        m_dot_cp_active = m_dot_active * AIR_SPECIFIC_HEAT_KJ_PER_KG_K

        # 고정 setpoint: 서버실 목표 온도
        pid.setpoint = zone_target_c

        supply_temp_c = pid.compute(current_value=sim.t_zone_c, dt=sim.config.dt_s)

        # Q_out: 유효 공기 유량 기반, 유효 칠러 COP × 최대 전력으로 상한 제한
        if sim.t_zone_c > supply_temp_c:
            q_out_air = calculate_cooling_load_from_airflow_kw(
                m_dot_kg_per_s=m_dot_active,
                supply_temp_c=supply_temp_c,
                return_temp_c=sim.t_zone_c,
            )
            chiller = calculate_chiller_power_kw(q_out_air, outdoor_temp_c)
            q_out_kw = min(q_out_air, chiller_max_active * chiller.cop)
        else:
            q_out_kw = 0.0

        error_c = pid.setpoint - sim.t_zone_c
        step_result = sim.step(q_in_kw=q_in_kw, q_out_kw=q_out_kw)

        results.append(PIDLoopResult(
            step=t + 1,
            t_zone_c=step_result.t_zone_c,
            supply_temp_c=supply_temp_c,
            q_in_kw=q_in_kw,
            q_out_kw=q_out_kw,
            error_c=error_c,
            outdoor_temp_c=outdoor_temp_c,
            n_crah=n_active,
        ))

    return results



if __name__ == "__main__":
    T_MIN, T_MAX = 18.0, 27.0
    STEPS = 1600       # 정상(300s) + 위기(1300s)
    CRISIS_START = 300  # 위기 시작 시점 (s)

    ELECTRICITY_RATE_KRW_PER_KWH = 120.0   # 산업용 전기요금 참고값 (원/kWh)
    NAVER_PUE_BENCHMARK = 1.09              # NAVER 권고 PUE 벤치마크 (명세서 #3)

    # ── 공통 스케줄 ──────────────────────────────────────────────────────
    def _sched(normal_val, crisis_val):
        return [normal_val if t < CRISIS_START else crisis_val for t in range(STEPS)]

    # 전환 구간 (예비 장비 투입까지 걸리는 시간)
    CHILLER_TRANSITION_S = 160   # 칠러 고장 감지 + 예비 투입까지 대기 시간 (초)

    # ── 서버 구성 (SPECpower 기반) ────────────────────────────────────────
    NUM_CPU   = 400   # CPU 서버 대수
    NUM_GPU   = 20    # GPU 서버 대수 (A100×4 기준, 전체의 ~5%)
    BASE_UTIL = 0.4   # 평균 CPU 사용률 (40%)

    # Q_in 기준: calculate_total_it_power_kw(util, num_cpu, num_gpu)
    # CPU 400대: (200 + 300×0.4) × 400 = 128kW
    # GPU  20대: (300 + 1200×0.4) × 20 = 15.6kW  → 합계 ≈ 144kW
    IT_BASE_KW   = calculate_total_it_power_kw(BASE_UTIL, NUM_CPU, NUM_GPU)
    IT_CRISIS_KW = round(IT_BASE_KW * 1.3, 1)   # S1: +30% 부하 급증

    it_s1      = _sched(IT_BASE_KW, IT_CRISIS_KW)
    outdoor_s2 = _sched(22.0, 42.0)              # S2: 폭염
    it_s2      = [IT_BASE_KW] * STEPS

    # S3: 칠러 고장 — 전환 구간(160s) 동안 냉각 불가, 예비 칠러 투입 후 복구
    it_s3 = [IT_BASE_KW] * STEPS
    chiller_scale_s3 = [
        1.0 if t < CRISIS_START else
        (0.0 if t < CRISIS_START + CHILLER_TRANSITION_S else 1.0)
        for t in range(STEPS)
    ]

    ZONE_TARGET_C = 24.0   # 서버실 목표 온도 (고정 setpoint)

    # ── 유틸 함수 ────────────────────────────────────────────────────────
    def _run(kp, ki, kd, it_sched, outdoor_sched=22.0, n_crah_sched=DEFAULT_N_CRAH_ACTIVE,
             chiller_scale_sched=1.0):
        sim = ThermalSimulator(ThermalSimulatorConfig())
        pid = PIDController(kp=kp, ki=ki, kd=kd)
        return run_pid_loop(STEPS, it_sched, pid, sim,
                            outdoor_temp_c_schedule=outdoor_sched,
                            n_crah_schedule=n_crah_sched,
                            zone_target_c=ZONE_TARGET_C,
                            chiller_scale_schedule=chiller_scale_sched)

    def _run_all(kp, ki, kd):
        r1 = _run(kp, ki, kd, it_s1)
        r2 = _run(kp, ki, kd, it_s2, outdoor_s2)
        r3 = _run(kp, ki, kd, it_s3, chiller_scale_sched=chiller_scale_s3)
        return r1, r2, r3

    def _crisis_ret(results):
        crisis = results[CRISIS_START:]
        ok = sum(1 for r in crisis if T_MIN <= r.t_zone_c <= T_MAX)
        return ok / len(crisis) * 100

    def _total_ret(results):
        ok = sum(1 for r in results if T_MIN <= r.t_zone_c <= T_MAX)
        return ok / STEPS * 100

    _M_DOT_CP_FULL = DEFAULT_M_DOT_KG_PER_S * AIR_SPECIFIC_HEAT_KJ_PER_KG_K  # 전 CRAH 가동 시

    # ────────────────────────────────────────────────────────────────────
    # [1단계] Ki sweep  (Kp=1.0, Kd=0.0 고정)
    #
    # 정상 운영 3대 기준: ṁ = 33.0 kg/s, ṁ·cp = 33.2 kW/K
    # τ = C_eff / (ṁ·cp) = 9009 / 33.2 ≈ 272초
    # IMC 기준: Ki_imc = Kp / τ = 1.0 / 272 ≈ 0.00368
    # ────────────────────────────────────────────────────────────────────
    _M_DOT_NORMAL = DEFAULT_M_DOT_KG_PER_S * DEFAULT_N_CRAH_ACTIVE / 4  # 3대 기준 ṁ
    _TAU = DEFAULT_C_EFF_KJ_PER_K / (_M_DOT_NORMAL * AIR_SPECIFIC_HEAT_KJ_PER_KG_K)

    KP = 1.0
    KI_SWEEP = [0.001, 0.002, 0.003, 0.004, 0.005, 0.007, 0.010, 0.015, 0.020]

    print("\n" + "=" * 74)
    print("  [1단계] Ki sweep — Kp=1.0, Kd=0.0 고정")
    print(f"  IMC 기준: Ki_imc = {KP:.1f} / {_TAU:.0f} = {KP/_TAU:.5f}  (τ={_TAU:.0f}s, ṁ={_M_DOT_NORMAL:.1f} kg/s, 3대 정상 운영)")
    print("=" * 74)
    print(f"{'Ki':>7}  {'정상최종T':>9}  {'SP':>7}  {'S1위기%':>8}  {'S2위기%':>8}  {'S3위기%':>8}  {'min%':>6}")
    print(f"  (S1: IT+30% / S2: 폭염 42°C / S3: 칠러 전환 {CHILLER_TRANSITION_S}s)")
    print("-" * 74)

    ki_best = KI_SWEEP[-1]
    ki_best_score = -1.0
    ki_sweep_data = {}

    for ki in KI_SWEEP:
        r1, r2, r3 = _run_all(KP, ki, 0.0)
        normal_last = r1[CRISIS_START - 1]
        t_final = normal_last.t_zone_c
        s1c = _crisis_ret(r1)
        s2c = _crisis_ret(r2)
        s3c = _crisis_ret(r3)
        min_c = min(s1c, s2c, s3c)
        ki_sweep_data[ki] = (s1c, s2c, s3c, min_c, t_final)

        mark = " ◎" if min_c >= 95 else (" ○" if min_c >= 80 else "")
        print(f"{ki:>7.3f}  {t_final:>8.2f}°C  {ZONE_TARGET_C:>6.2f}°C  {s1c:>7.1f}%  {s2c:>7.1f}%  {s3c:>7.1f}%  {min_c:>5.1f}%{mark}")

        score = min_c * 1000 - abs(t_final - ZONE_TARGET_C)
        if score > ki_best_score:
            ki_best_score = score
            ki_best = ki

    print(f"\n  → 최적 Ki = {ki_best}  (위기 min={ki_sweep_data[ki_best][3]:.1f}%,"
          f" 정상 최종T={ki_sweep_data[ki_best][4]:.2f}°C vs SP={ZONE_TARGET_C:.2f}°C)")

    # ────────────────────────────────────────────────────────────────────
    # [2단계] Kd sweep  (Kp=2.0, Ki=ki_best 고정)
    # ────────────────────────────────────────────────────────────────────
    # 음수 Kd: 회복 구간(delta_error > 0)에서 supply를 낮춰 냉각 유지 → 수렴 가속 가능
    # 양수 Kd: 회복 구간에서 supply를 올려 냉각 방해 → supply가 포화(18°C)된 상태에서 역효과
    KD_SWEEP = [-20.0, -10.0, -5.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]

    print("\n" + "=" * 74)
    print(f"  [2단계] Kd sweep — Kp=1.0, Ki={ki_best} 고정")
    print("=" * 74)
    print(f"{'Kd':>7}  {'S3회복T':>9}  {'S1위기%':>8}  {'S2위기%':>8}  {'S3위기%':>8}  {'min%':>6}")
    print("-" * 66)

    kd_best = 0.0
    kd_best_score = -1.0

    for kd in KD_SWEEP:
        r1, r2, r3 = _run_all(KP, ki_best, kd)
        s3_final_T = r3[-1].t_zone_c  # 칠러 백업 후 최종 T_zone (24°C에 가까울수록 좋음)
        s1c = _crisis_ret(r1)
        s2c = _crisis_ret(r2)
        s3c = _crisis_ret(r3)
        min_c = min(s1c, s2c, s3c)

        mark = " ◎" if min_c >= ki_sweep_data[ki_best][3] else (
               " ○" if min_c >= ki_sweep_data[ki_best][3] - 2 else "")
        print(f"{kd:>7.1f}  {s3_final_T:>8.2f}°C  {s1c:>7.1f}%  {s2c:>7.1f}%  {s3c:>7.1f}%  {min_c:>5.1f}%{mark}")

        score = min_c * 1000 - abs(s3_final_T - ZONE_TARGET_C)
        if score > kd_best_score:
            kd_best_score = score
            kd_best = kd

    print(f"\n  → 최적 Kd = {kd_best}")
    print(f"\n  ★ 최종 게인값: Kp={KP}, Ki={ki_best}, Kd={kd_best}")
    print(f"    (이전: Kp=2.0, Ki=0.002, Kd=100.0 — ṁ=18 kg/s, τ=347s 기준)")
    print(f"    (변경: C_eff 6281→9009 kJ/K, ṁ 18→44 kg/s, CPU 400대+GPU 20대 반영)")

    # ────────────────────────────────────────────────────────────────────
    # [3단계] 최종 게인값으로 4개 시나리오 상세 출력
    # ────────────────────────────────────────────────────────────────────
    def _print_scenario(name, desc, results, var_label, var_fn):
        print(f"\n{'='*62}")
        print(f"  {name}")
        print(f"  {desc}")
        print(f"{'='*62}")
        hdr = (
            f"{'t':>5}  {'Phase':<10}  {var_label:>8}  "
            f"{'T_zone':>8}  {'supply':>6}  {'Q_out':>7}  {'error':>7}"
        )
        print(hdr)
        print("-" * len(hdr))
        for t in sorted(set(list(range(0, STEPS, 50)) + [STEPS - 1])):
            r = results[t]
            phase = "정상 운영" if t < CRISIS_START else "위기"
            flag = " !" if not (T_MIN <= r.t_zone_c <= T_MAX) else "  "
            print(
                f"{t+1:>5}  {phase:<10}  {var_fn(r):>8}  "
                f"{r.t_zone_c:>6.2f}°C{flag}  {r.supply_temp_c:>5.1f}°C  "
                f"{r.q_out_kw:>6.1f}kW  {r.error_c:>+6.2f}°C"
            )
        ok = sum(1 for r in results if T_MIN <= r.t_zone_c <= T_MAX)
        print()
        print(f"  온도 유지율: {ok}/{STEPS} = {ok/STEPS*100:.1f}%")

    def _energy_analysis(name: str, results: list, dt_s: float = 1.0):
        """시나리오별 에너지·PUE·비용 분석 출력 (명세서 #9)"""
        it_kwh = sum(r.q_in_kw for r in results) * dt_s / 3600
        cool_kwh = sum(
            (r.q_out_kw / calculate_cop(r.outdoor_temp_c)) * dt_s / 3600
            for r in results if r.q_out_kw > 0
        )
        total_kwh = it_kwh + cool_kwh
        pue = total_kwh / it_kwh if it_kwh > 0 else float("inf")
        cost = total_kwh * ELECTRICITY_RATE_KRW_PER_KWH
        pue_gap = pue - NAVER_PUE_BENCHMARK

        print(f"\n  ── 에너지·비용 분석 ({name}) ──────────────────────")
        print(f"  {'IT 소비':>16}: {it_kwh:.4f} kWh")
        print(f"  {'냉각 전기 소비':>16}: {cool_kwh:.4f} kWh")
        print(f"  {'총 소비':>16}: {total_kwh:.4f} kWh")
        print(f"  {'평균 PUE':>16}: {pue:.3f}  (NAVER 벤치마크 {NAVER_PUE_BENCHMARK} 대비 {pue_gap:+.3f})")
        print(f"  {'추정 비용':>16}: {cost:.1f} 원")

    cfg = ThermalSimulatorConfig()
    cop_normal = calculate_cop(22.0)
    m_dot_normal = cfg.m_dot_per_crah * DEFAULT_N_CRAH_ACTIVE   # 정상 3대 기준 ṁ

    # 시나리오 1: 부하 급증 (CRAH 3대 정상 가동)
    sim1 = ThermalSimulator(ThermalSimulatorConfig())
    r1 = run_pid_loop(STEPS, it_s1, PIDController(kp=KP, ki=ki_best, kd=kd_best),
                      sim1, outdoor_temp_c_schedule=22.0, zone_target_c=ZONE_TARGET_C,
                      n_crah_schedule=DEFAULT_N_CRAH_ACTIVE)
    _print_scenario(
        "시나리오 1: 정상 운영 → 서버 부하 급증 (+30%)",
        f"IT {IT_BASE_KW:.0f}→{IT_CRISIS_KW:.0f}kW (CPU {NUM_CPU}대+GPU {NUM_GPU}대, +30%) / 외기 22°C / CRAH {DEFAULT_N_CRAH_ACTIVE}대 정상 | "
        f"ṁ={m_dot_normal:.1f}kg/s, 칠러 1대={cfg.chiller_design_kw*cop_normal:.0f}kW 열",
        r1, "IT부하", lambda r: f"{r.q_in_kw:.0f}kW",
    )
    _energy_analysis("S1: 부하 급증", r1)

    # 시나리오 2: 폭염 (CRAH 3대 정상 가동)
    cop_crisis_s2 = calculate_cop(42.0)
    sim2 = ThermalSimulator(ThermalSimulatorConfig())
    r2 = run_pid_loop(STEPS, it_s2, PIDController(kp=KP, ki=ki_best, kd=kd_best),
                      sim2, outdoor_temp_c_schedule=outdoor_s2, zone_target_c=ZONE_TARGET_C,
                      n_crah_schedule=DEFAULT_N_CRAH_ACTIVE)
    _print_scenario(
        "시나리오 2: 정상 운영 → 폭염 (외기 온도 급상승)",
        f"외기 22→42°C / IT {IT_BASE_KW:.0f}kW / CRAH {DEFAULT_N_CRAH_ACTIVE}대 정상 | "
        f"칠러 1대 COP 5.3→3.3, 냉각 한도 {cfg.chiller_design_kw*cop_normal:.0f}→{cfg.chiller_design_kw*cop_crisis_s2:.0f}kW 열",
        r2, "외기온도", lambda r: f"{r.outdoor_temp_c:.0f}°C",
    )
    _energy_analysis("S2: 폭염", r2)

    # 시나리오 3: 칠러 고장 — 전환 구간(160s) 동안 냉각 불가, 이후 예비 칠러 투입
    sim3 = ThermalSimulator(ThermalSimulatorConfig())
    r3 = run_pid_loop(STEPS, it_s3, PIDController(kp=KP, ki=ki_best, kd=kd_best),
                      sim3, outdoor_temp_c_schedule=22.0,
                      zone_target_c=ZONE_TARGET_C,
                      n_crah_schedule=DEFAULT_N_CRAH_ACTIVE,
                      chiller_scale_schedule=chiller_scale_s3)
    _print_scenario(
        f"시나리오 3: 칠러 고장 → 전환 구간 {CHILLER_TRANSITION_S}s → 예비 투입",
        f"칠러 scale 1→0→1 / CRAH {DEFAULT_N_CRAH_ACTIVE}대 유지 / IT {IT_BASE_KW:.0f}kW / 외기 22°C | "
        f"전환 중 Q_out=0 (냉각 불가)",
        r3, "칠러상태", lambda r: f"{'고장중' if CRISIS_START <= r.step - 1 < CRISIS_START + CHILLER_TRANSITION_S else '정상'}",
    )
    _energy_analysis("S3: 칠러 고장·전환", r3)
