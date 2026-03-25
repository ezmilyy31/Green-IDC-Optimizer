"""
열역학 모델 vs Sinergym 검증 스크립트

[보류] 2026-03-24 팀 합의로 Sinergym 오차 검증을 보류하기로 결정.

사유: 비교 대상의 물리적 범위 불일치
  - Q_model = IT 발열만 (에너지 보존)
    Q_시뮬  = IT 발열 + 일사 + 외벽 열전도 + 열적 관성
  - P_model = 압축기(냉동 사이클) 전력만
    P_시뮬  = 압축기 + 팬 + basin heater (HVAC 전체)
  범위가 다른 두 값의 오차를 "모델 정확도"로 해석할 수 없음.
  재개 시 비교 범위를 맞추는 작업(비IT 발열 모델링 또는 Sinergym 분리 측정)이 선행되어야 함.

──────────────────────────────────────────────────────────────────────────────
Sinergym(EnergyPlus 기반) 시뮬레이션 결과와 자체 열역학 모델을 비교하여
냉각 전력·PUE 오차가 10% 이내인지 검증한다.

실행 방법 (Docker 컨테이너 내부):
    python3 apps/simulation_service/sinergym_validator.py
"""

import glob
import os
import shutil
import sys

import numpy as np

# Docker(/app) 및 로컬 환경 모두에서 동작하도록 프로젝트 루트를 경로에 추가
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

try:
    import gymnasium as gym
    import sinergym  # noqa: F401 — 환경 등록을 위해 import 필요
except ImportError as e:
    print(f"[오류] Sinergym 또는 Gymnasium을 불러올 수 없습니다: {e}")
    print("       Docker 컨테이너(simulation_service) 내부에서 실행하세요.")
    sys.exit(1)

from domain.thermodynamics.chiller import calculate_chiller_power_kw
from domain.thermodynamics.cooling_load import calculate_cooling_load_from_it_power_kw
from domain.thermodynamics.it_power import calculate_total_it_power_kw
from domain.thermodynamics.pue import PUE_BENCHMARK, calculate_pue

# ── 설정 ──────────────────────────────────────────────────────────────────────
ENV_NAME = "Eplus-datacenter_dx-mixed-continuous-stochastic-v1"
VALIDATION_EPISODES = 120   # 샘플링 횟수 (73시간 간격으로 추출하여 모든 시간대 커버)
STEPS_PER_YEAR = 35040      # EnergyPlus 시간 해상도: 1스텝 = 15분 (8760시간 * 4)
SAMPLE_INTERVAL = STEPS_PER_YEAR // VALIDATION_EPISODES  # 292스텝 = 73시간 간격
ERROR_THRESHOLD_PCT = 10.0  # 허용 오차율 (%)

# Sinergym DX 환경 ITE 설계 전력 (IDF 직접 값)
# 근거: IDF 파일(2ZoneDataCenterHVAC_wEconomizer_DX.epJSON)
#   East Zone: 259.08 m² × 200 W/m² = 51,816 W
#   West Zone: 232.26 m² × 200 W/m² = 46,452 W
ITE_P_DESIGN_KW = 98.268  # 총 설계 IT 전력 (kW)

# EnergyPlus ITE(IT Equipment) 온도 보정 계수
# 공식: P = P_design × (-1 + cpu + ITE_TEMP_COEFF × T_inlet)
# 물리적 의미: 흡기 온도 1°C 상승 → 서버 냉각 팬 RPM 증가 → 전력 6.67% 증가
# 출처: EnergyPlus Engineering Reference, ITE Object (fLoadTemp bivariate curve)
ITE_TEMP_COEFF = 0.0667          # 1/°C
ITE_T_INLET_NEUTRAL_C = 20.0    # SPECpower 측정 기준 흡기 온도 (°C)


def _get_obs_variables(env: gym.Env) -> list[str]:
    """환경에서 관측 변수명 목록을 가져온다 (래퍼 중첩 고려)."""
    for attr in ("observation_variables", "obs_variables"):
        if hasattr(env, attr):
            return list(getattr(env, attr))
        if hasattr(env, "unwrapped") and hasattr(env.unwrapped, attr):
            return list(getattr(env.unwrapped, attr))
    return []


def _find_first(variables: list[str], *keyword_groups: tuple[str, ...]) -> int | None:
    """
    keyword_groups 순서대로 시도하여, 그룹 내 키워드를 모두 포함하는
    첫 번째 변수의 인덱스를 반환한다.

    ※ Python or 연산자는 0을 falsy로 처리하므로 사용하지 않는다.
    """
    for keywords in keyword_groups:
        for i, name in enumerate(variables):
            name_lower = name.lower()
            if all(kw.lower() in name_lower for kw in keywords):
                return i
    return None


def _detect_indices(variables: list[str]) -> dict[str, int | None]:
    """관측 변수 목록에서 검증에 필요한 변수의 인덱스를 탐지한다."""
    return {
        "outdoor_temp": _find_first(
            variables,
            ("outdoor", "drybulb"),
            ("outdoor", "temperature"),
            ("site", "outdoor"),
        ),
        "it_power": _find_first(
            variables,
            ("building", "electric"),
            ("it", "electric"),
            ("facility", "building"),
            ("total", "building"),
        ),
        "hvac_power": _find_first(
            variables,
            ("hvac", "electric"),
            ("facility", "hvac"),
            ("cooling", "electric"),
            ("total", "hvac"),
        ),
        "zone_temp": _find_first(
            variables,
            ("zone", "air", "temperature", "west"),
            ("zone", "air", "temperature", "east"),
            ("zone", "air", "temperature"),
            ("zone", "temperature"),
        ),
        # 공기 유량 기반 냉각 부하 계산용 (Q = m_dot × c_p × ΔT)
        "east_return_temp":  _find_first(variables, ("east", "return")),
        "east_supply_temp":  _find_first(variables, ("east", "supply")),
        "west_return_temp":  _find_first(variables, ("west", "return")),
        "west_supply_temp":  _find_first(variables, ("west", "supply")),
        "east_flow":         _find_first(variables, ("east", "mass", "flow")),
        "west_flow":         _find_first(variables, ("west", "mass", "flow")),
        # SPECpower 공식 입력: CPU 사용률
        "cpu_fraction":      _find_first(variables, ("cpu", "loading"), ("cpu", "fraction")),
    }


def _to_kw(raw: float) -> float:
    """EnergyPlus 전력 출력값을 kW로 변환한다 (1,000 이상이면 W로 판단)."""
    return raw / 1000.0 if abs(raw) >= 1000.0 else raw


def run_validation() -> bool:
    """
    Sinergym 시뮬레이션을 실행하고 자체 열역학 모델과 비교 검증한다.

    Returns:
        True: 검증 통과 (평균 오차 < 10%)
        False: 검증 실패
    """
    print(f"\nSinergym 환경 초기화: {ENV_NAME}")
    env = gym.make(ENV_NAME)

    variables = _get_obs_variables(env)
    print(f"\n관측 변수 목록 ({len(variables)}개):")
    for i, v in enumerate(variables):
        print(f"  [{i:2d}] {v}")

    indices = _detect_indices(variables)
    print(f"\n탐지된 인덱스: {indices}")

    # 탐지 실패 시 조기 종료
    missing = [k for k, v in indices.items() if v is None and k != "zone_temp"]
    if missing:
        print(f"\n[경고] 다음 변수를 탐지하지 못했습니다: {missing}")
        print("       위 관측 변수 목록을 보고 키워드 패턴을 수정하세요.")
        env.close()
        return False

    # 중간값 액션 (공급 온도 중간 설정)
    action = (env.action_space.high + env.action_space.low) / 2.0

    from tqdm import tqdm

    # 1개 에피소드를 연간 전체(8760h)로 실행하면서 SAMPLE_INTERVAL마다 샘플링
    # → 계절별 다양한 기온 조건(겨울~여름) 커버
    print(f"\n연간 시뮬레이션 시작 ({STEPS_PER_YEAR}스텝, {SAMPLE_INTERVAL}스텝마다 샘플링)...\n")
    obs, info = env.reset()

    results: list[dict] = []

    def _q_zone(obs: np.ndarray, flow_idx: str, return_idx: str, supply_idx: str) -> float:
        if any(indices.get(k) is None for k in (flow_idx, return_idx, supply_idx)):
            return 0.0
        q = (float(obs[indices[flow_idx]]) * 1.005
             * (float(obs[indices[return_idx]]) - float(obs[indices[supply_idx]])))
        return max(0.0, q)

    sample_idx = 0
    for step in tqdm(range(STEPS_PER_YEAR), desc="연간 시뮬레이션", unit="h"):
        obs, reward, terminated, truncated, info = env.step(action)

        # SAMPLE_INTERVAL마다 검증 데이터 수집
        if (step + 1) % SAMPLE_INTERVAL != 0:
            if terminated or truncated:
                break
            continue

        # ── 관측값 추출 ────────────────────────────────────────────────────
        outdoor_temp_c = float(obs[indices["outdoor_temp"]])
        hvac_power_kw  = _to_kw(float(obs[indices["hvac_power"]]))

        # ── Sinergym 기준값: Q = ṁ × c_p × ΔT (공기 유량 기반 실제 냉각 부하) ──
        sinergym_q_kw = (
            _q_zone(obs, "east_flow", "east_return_temp", "east_supply_temp")
            + _q_zone(obs, "west_flow", "west_return_temp", "west_supply_temp")
        )

        # ── IT 전력: SPECpower + EnergyPlus ITE 온도 보정 (검증 전용) ─────────
        # SPECpower는 온도항이 없어 Sinergym(EnergyPlus) 대비 IT 전력을 과소 추정한다.
        # 검증 맥락에서만 흡기 온도(CRAH 공급 온도 obs)를 이용해 EnergyPlus ITE 공식으로 보정.
        # 공식: P = P_design × (-1 + cpu + 0.0667 × T_inlet)
        # it_power.py의 SPECpower 공식은 그대로 유지 (다른 서비스에서 독립 사용).
        cpu_fraction = (
            float(obs[indices["cpu_fraction"]]) if indices["cpu_fraction"] is not None else 0.5
        )

        east_sup_t = float(obs[indices["east_supply_temp"]]) if indices["east_supply_temp"] is not None else None
        west_sup_t = float(obs[indices["west_supply_temp"]]) if indices["west_supply_temp"] is not None else None
        sup_obs = [t for t in (east_sup_t, west_sup_t) if t is not None]
        t_inlet = sum(sup_obs) / len(sup_obs) if sup_obs else ITE_T_INLET_NEUTRAL_C

        it_power_kw = max(0.0, ITE_P_DESIGN_KW * (-1.0 + cpu_fraction + ITE_TEMP_COEFF * t_inlet))

        # ── Sinergym 냉각 전력: HVAC 원시값 (보정 없음) ──────────────────────
        sinergym_cooling_elec_kw = hvac_power_kw

        # ── Q모델: 에너지 보존 법칙 (Q = IT_power) ───────────────────────────
        # 서버에 공급된 전기 에너지는 정상 상태에서 전부 열로 변환된다.
        # SPECpower는 온도항이 없어 Sinergym 대비 IT_power를 과소 추정하는 한계가 있음.
        cooling_load_kw = calculate_cooling_load_from_it_power_kw(it_power_kw)

        # 첫 샘플: 실제 값 확인용 디버그 출력
        if sample_idx == 0:
            print(f"\n[샘플1 디버그] outdoor={outdoor_temp_c:.1f}°C  cpu={cpu_fraction:.2f}  "
                  f"T_inlet={t_inlet:.1f}°C  our_IT={it_power_kw:.1f}kW  "
                  f"Q_model={cooling_load_kw:.1f}kW  sinergym_Q={sinergym_q_kw:.1f}kW  "
                  f"hvac={hvac_power_kw:.2f}kW\n")

        if sinergym_q_kw <= 0 or it_power_kw <= 0:
            sample_idx += 1
            if terminated or truncated:
                break
            continue

        # ── 자체 열역학 모델 계산 ──────────────────────────────────────────
        chiller_result  = calculate_chiller_power_kw(cooling_load_kw, outdoor_temp_c)
        pue_result      = calculate_pue(it_power_kw, chiller_result.chiller_power_kw)

        # Sinergym PUE: (실제 서버 발열 + 전체 HVAC) / 실제 서버 발열
        sinergym_total_kw = sinergym_q_kw + hvac_power_kw
        sinergym_pue      = sinergym_total_kw / sinergym_q_kw

        # ── 오차율 (검증 기준) ───────────────────────────────────────────
        cooling_load_error_pct = (
            abs(cooling_load_kw - sinergym_q_kw) / sinergym_q_kw * 100.0
            if sinergym_q_kw > 0 else 0.0
        )
        chiller_elec_error_pct = (
            abs(chiller_result.chiller_power_kw - sinergym_cooling_elec_kw) / sinergym_cooling_elec_kw * 100.0
            if sinergym_cooling_elec_kw > 0 else None
        )
        pue_error_pct = abs(pue_result.pue - sinergym_pue) / sinergym_pue * 100.0

        results.append({
            "episode":                    sample_idx,
            "outdoor_temp_c":             outdoor_temp_c,
            "it_power_kw":                it_power_kw,
            "our_cooling_load_kw":        cooling_load_kw,
            "sinergym_q_kw":              sinergym_q_kw,
            "our_cooling_kw":             chiller_result.chiller_power_kw,
            "sinergym_cooling_elec_kw":   sinergym_cooling_elec_kw,
            "our_pue":                    pue_result.pue,
            "sinergym_pue":               sinergym_pue,
            "cooling_mode":               chiller_result.cooling_mode.value,
            "cop":                        chiller_result.cop,
            "cooling_load_error_pct":     cooling_load_error_pct,
            "chiller_elec_error_pct":     chiller_elec_error_pct,  # None이면 비교 불가
            "pue_error_pct":              pue_error_pct,
        })

        sample_idx += 1
        if terminated or truncated:
            break

    env.close()

    # ── Sinergym/EnergyPlus 시뮬레이션 출력 디렉토리 정리 ────────────────────
    for d in glob.glob(os.path.join(_PROJECT_ROOT, "Eplus-*")):
        if os.path.isdir(d):
            shutil.rmtree(d)

    # ── 결과 출력 ─────────────────────────────────────────────────────────────
    if not results:
        print("[경고] 유효한 검증 데이터가 없습니다 (IT 전력이 모두 0).")
        return False

    avg_cooling_load_error  = np.mean([r["cooling_load_error_pct"] for r in results])
    # chiller_elec_error_pct가 None인 샘플(basin heater 초과)은 평균에서 제외
    valid_chiller = [r["chiller_elec_error_pct"] for r in results if r["chiller_elec_error_pct"] is not None]
    avg_chiller_elec_error  = np.mean(valid_chiller) if valid_chiller else float("nan")
    avg_pue_error           = np.mean([r["pue_error_pct"] for r in results])
    avg_our_cooling_load    = np.mean([r["our_cooling_load_kw"] for r in results])
    avg_sinergym_q          = np.mean([r["sinergym_q_kw"] for r in results])
    avg_our_cooling         = np.mean([r["our_cooling_kw"] for r in results])
    avg_sinergym_elec       = np.mean([r["sinergym_cooling_elec_kw"] for r in results])
    avg_our_pue             = np.mean([r["our_pue"] for r in results])
    avg_sinergym_pue        = np.mean([r["sinergym_pue"] for r in results])

    W = 56  # 테이블 너비

    # ── 에피소드별 상세 결과 ───────────────────────────────────────────────
    print(f"\n=== 에피소드별 검증 결과 ===\n")
    EW = 80
    print(f"{'ep':>3} {'T외기':>6} {'Q모델':>8} {'Q시뮬':>8} {'Q오차':>7}"
          f" {'P모델':>8} {'P시뮬':>8} {'P오차':>7} {'PUE오차':>7} {'판정':>4}")
    print("-" * EW)
    for r in results:
        chiller_err = r["chiller_elec_error_pct"]
        chiller_err_str = f"{chiller_err:>6.1f}%" if chiller_err is not None else "   N/A "
        # 칠러 오차가 None이면 판정에서 제외 (냉각 부하 + PUE만 기준)
        chiller_ok = (chiller_err is None) or (chiller_err < ERROR_THRESHOLD_PCT)
        ep_pass = (
            r["cooling_load_error_pct"] < ERROR_THRESHOLD_PCT
            and chiller_ok
            and r["pue_error_pct"] < ERROR_THRESHOLD_PCT
        )
        print(f"{r['episode']+1:>3} {r['outdoor_temp_c']:>5.1f}°"
              f" {r['our_cooling_load_kw']:>8.1f} {r['sinergym_q_kw']:>8.1f}"
              f" {r['cooling_load_error_pct']:>6.1f}%"
              f" {r['our_cooling_kw']:>8.2f} {r['sinergym_cooling_elec_kw']:>8.2f}"
              f" {chiller_err_str}"
              f" {r['pue_error_pct']:>6.1f}%"
              f" {'✓' if ep_pass else '✗':>4}")

    per_ep_pass_count = sum(
        1 for r in results
        if r["cooling_load_error_pct"] < ERROR_THRESHOLD_PCT
        and ((r["chiller_elec_error_pct"] is None) or (r["chiller_elec_error_pct"] < ERROR_THRESHOLD_PCT))
        and r["pue_error_pct"] < ERROR_THRESHOLD_PCT
    )
    n_na = sum(1 for r in results if r["chiller_elec_error_pct"] is None)
    print("-" * EW)
    print(f"에피소드 통과: {per_ep_pass_count} / {len(results)}"
          + (f"  (칠러 비교 N/A: {n_na}개)" if n_na else ""))

    # ── 평균 비교 ──────────────────────────────────────────────────────────
    print(f"\n=== 열역학 모델 vs Sinergym 시뮬레이션 비교 (평균) ===\n")
    print(f"{'구분':<16} {'자체 모델':>13} {'Sinergym':>13} {'오차율':>8}")
    print("-" * W)
    print(f"{'냉각 부하 (kW)':<16}"          # Q = ṁ·cp·ΔT 비교 (검증 기준)
          f" {avg_our_cooling_load:>13,.1f}"
          f" {avg_sinergym_q:>13,.1f}"
          f" {avg_cooling_load_error:>7.1f}%")
    print(f"{'칠러 전력 (kW)':<16}"          # 냉각 전력 비교 (anti-freeze 히터 제외)
          f" {avg_our_cooling:>13,.1f}"
          f" {avg_sinergym_elec:>13,.1f}"
          f" {avg_chiller_elec_error:>7.1f}%")
    print(f"{'PUE':<16}"
          f" {avg_our_pue:>13.2f}"
          f" {avg_sinergym_pue:>13.2f}"
          f" {avg_pue_error:>7.1f}%")

    # 검증 기준: 냉각 부하(Q = ṁcpΔT) 오차 + 칠러 전력 오차 + PUE 오차
    cooling_pass = avg_cooling_load_error < ERROR_THRESHOLD_PCT
    chiller_pass = (not np.isnan(avg_chiller_elec_error)) and (avg_chiller_elec_error < ERROR_THRESHOLD_PCT)
    pue_pass     = avg_pue_error < ERROR_THRESHOLD_PCT
    overall_pass = cooling_pass and chiller_pass and pue_pass

    print("\n" + "=" * W)
    print(f"  냉각 부하 오차 {ERROR_THRESHOLD_PCT:.0f}% 이내: "
          f"{'통과 ✓' if cooling_pass else '실패 ✗'}  ({avg_cooling_load_error:.2f}%)")
    print(f"  칠러 전력 오차 {ERROR_THRESHOLD_PCT:.0f}% 이내: "
          f"{'통과 ✓' if chiller_pass else '실패 ✗'}  ({avg_chiller_elec_error:.2f}%)")
    print(f"  PUE 오차 {ERROR_THRESHOLD_PCT:.0f}% 이내      : "
          f"{'통과 ✓' if pue_pass else '실패 ✗'}  ({avg_pue_error:.2f}%)")
    print("-" * W)
    print(f"  {'[검증 통과]' if overall_pass else '[검증 실패]'} "
          f"{'모든 항목 오차율 10% 이내' if overall_pass else '오차율이 허용 기준 초과'}")
    print("=" * W)

    naver_pue = PUE_BENCHMARK["naver_chuncheon"]
    print(f"\nNAVER 각 춘천 PUE {naver_pue} 대비: "
          f"{avg_our_pue:.3f} (×{avg_our_pue / naver_pue:.2f})")
    print(f"유효 샘플 수: {len(results)} / {VALIDATION_EPISODES}")

    return overall_pass


if __name__ == "__main__":
    passed = run_validation()
    sys.exit(0 if passed else 1)
