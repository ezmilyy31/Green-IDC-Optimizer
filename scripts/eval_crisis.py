"""위기 시나리오 평가 — best vs fine-tune vs hybrid vs rule-based 비교.

IDCEnv `force_scenario`로 4가지 시나리오를 강제 적용해 평가:
- normal: 평상시 (baseline 성능 보존 확인)
- server_surge: CPU util ×1.3
- heat_wave: 외기 온도 +5~10°C
- chiller_derate: 칠러 전력 ×1.5~3.3 (효율 저하)

평가 컨트롤러 (두 모델 모두 있을 때):
- Rule-based / RL-best (효율 우선) / RL-finetune (안전 우선) / RL-hybrid (zone temp 기반 자동 전환)
- 각각 +fb 버전: safe fallback wrap 적용

3-tier 안전 시스템 (hybrid + fb 사용 시):
  Layer 1: best 모델     (zone < 26.0°C, 효율 우선)
  Layer 2: dr-fresh 모델 (26.0 ≤ zone < 26.5°C, 안전 우선)
  Layer 3: safe fallback (zone ≥ 26.5°C, T_SUPPLY_MIN 강제)

사용법:
    python scripts/eval_crisis.py
    python scripts/eval_crisis.py --best data/models/sac-wetbulb-1m.zip \
        --finetune data/models/sac-dr-fresh-1m.zip --episodes 20
"""

import argparse
from pathlib import Path

import numpy as np

from domain.controllers.idc_env import IDCEnv, T_SUPPLY_MIN, T_SUPPLY_MAX
from domain.controllers.rl_inference import SAFE_HIGH_C, SAFE_LOW_C, ZONE_TEMP_OBS_INDEX
from scripts.eval_baseline import (
    evaluate,
    rule_based_policy,
    rl_policy,
)


def with_safe_fallback(rl_pol):
    """RL 정책에 zone_temp 기반 safe fallback 후처리 추가.

    zone > 26.5°C: T_SUPPLY_MIN(18°C) 강제 냉각
    zone < 19.0°C: T_SUPPLY_MAX(25°C) 냉각 완화
    그 외: RL 정책 그대로
    """
    def wrapped(obs):
        zone_temp = float(obs[ZONE_TEMP_OBS_INDEX])
        if zone_temp > SAFE_HIGH_C:
            return np.array([T_SUPPLY_MIN], dtype=np.float32)
        if zone_temp < SAFE_LOW_C:
            return np.array([T_SUPPLY_MAX], dtype=np.float32)
        return rl_pol(obs)
    return wrapped


# Hybrid switch 임계 — 다중 신호 기반 사전 감지로 안전 우선 정책 발동.
# 실측 데이터 분포 기준 (Google Cluster Trace + 기상청 ASOS):
#   normal:       cpu mean=0.40 max=0.52, it_power mean=144 max=161
#   server_surge: cpu mean=0.52 max=0.67, it_power mean=161 max=183
# 임계 1 (사후 신호): zone_temp ≥ 26.0°C  — fallback(26.5)보다 0.5°C 낮게
# 임계 2 (사전 신호): cpu_util > 0.50     — normal max(0.52) 직전, surge mean(0.52) 위
# 임계 3 (사전 신호): it_power > 165 kW   — normal max(161) 위, surge max(183) 아래
HYBRID_SWITCH_C = 26.0
HYBRID_CPU_THRESH = 0.50
HYBRID_IT_POWER_THRESH = 165.0

# IDCEnv obs 인덱스 (zone_temp는 rl_inference에서 가져옴)
CPU_UTIL_OBS_INDEX = 4
IT_POWER_OBS_INDEX = 7


def with_hybrid(
    efficient_pol,
    safety_pol,
    switch_c: float = HYBRID_SWITCH_C,
    cpu_thresh: float = HYBRID_CPU_THRESH,
    it_power_thresh: float = HYBRID_IT_POWER_THRESH,
):
    """다중 신호 기반 두 정책 자동 전환 wrapper.

    세 가지 신호 중 하나라도 위기로 판정되면 safety_pol 사용:
      - 부하 신호 (사전): cpu_util > cpu_thresh 또는 it_power > it_power_thresh
      - 온도 신호 (사후): zone_temp >= switch_c

    safe fallback과 조합 시 3-tier 안전 시스템:
      Layer 1 효율    : efficient_pol   (정상 부하 + zone < 26.0°C)
      Layer 2 안전    : safety_pol      (위기 부하 OR 26.0 ≤ zone < 26.5°C)
      Layer 3 강제 cap: T_SUPPLY_MIN    (zone ≥ 26.5°C, with_safe_fallback에서 적용)

    핵심 차이 (vs zone-only hybrid):
      zone temp는 사후 신호 → server_surge처럼 부하가 칠러 용량 초과하는 시나리오에선
      zone이 올라간 시점에 이미 위반 임박. 부하 신호로 사전 감지해야 dr-fresh가
      마진 확보할 시간 확보.
    """
    def wrapped(obs):
        zone_temp = float(obs[ZONE_TEMP_OBS_INDEX])
        cpu_util = float(obs[CPU_UTIL_OBS_INDEX])
        it_power = float(obs[IT_POWER_OBS_INDEX])

        is_load_crisis = cpu_util > cpu_thresh or it_power > it_power_thresh
        is_temp_warning = zone_temp >= switch_c

        if is_load_crisis or is_temp_warning:
            return safety_pol(obs)
        return efficient_pol(obs)
    return wrapped


SCENARIOS = ["normal", "server_surge", "heat_wave", "chiller_derate"]


def make_env(scenario: str) -> IDCEnv:
    return IDCEnv(w_energy=0.8, force_scenario=scenario)


def run_one(name: str, policy_fn, episodes: int) -> dict[str, dict]:
    """4개 시나리오에서 동일 정책 평가. 시나리오별 결과 dict 반환."""
    results = {}
    for sc in SCENARIOS:
        env = make_env(sc)
        results[sc] = evaluate(env, policy_fn, episodes)
    return results


def print_table(all_results: dict[str, dict[str, dict]]) -> None:
    """controller → scenario → metrics 구조의 dict를 표로 출력."""
    header_fmt = f"{'시나리오':<18}{'컨트롤러':<22}{'PUE':>9}{'위반(°C)':>12}{'zone(°C)':>11}{'rew':>10}"
    print("\n" + "=" * len(header_fmt))
    print(header_fmt)
    print("=" * len(header_fmt))
    for sc in SCENARIOS:
        first = True
        for name, sc_results in all_results.items():
            r = sc_results[sc]
            sc_label = sc if first else ""
            print(
                f"{sc_label:<18}{name:<22}"
                f"{r['pue_mean']:>9.4f}"
                f"{r['temp_violation_mean']:>12.4f}"
                f"{r['zone_temp_mean']:>11.2f}"
                f"{r['ep_rew_mean']:>10.2f}"
            )
            first = False
        print("-" * len(header_fmt))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--best", type=str, default="data/models/sac-wetbulb-1m.zip")
    parser.add_argument("--finetune", type=str, default="data/models/sac-dr-fresh-1m.zip")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--hybrid-switch-c", type=float, default=HYBRID_SWITCH_C,
                        help=f"Hybrid 정책 전환 zone temp 임계 (기본 {HYBRID_SWITCH_C}°C)")
    args = parser.parse_args()

    controllers: dict[str, callable] = {"Rule-based": rule_based_policy}
    best_pol = None
    ft_pol = None
    if Path(args.best).exists():
        best_pol = rl_policy(args.best)
        controllers["RL-best"] = best_pol
        controllers["RL-best+fb"] = with_safe_fallback(best_pol)
    else:
        print(f"⚠ best 모델 없음: {args.best}")
    if Path(args.finetune).exists():
        ft_pol = rl_policy(args.finetune)
        controllers["RL-finetune"] = ft_pol
        controllers["RL-finetune+fb"] = with_safe_fallback(ft_pol)
    else:
        print(f"⚠ fine-tune 모델 없음: {args.finetune}")

    # Hybrid: 두 모델 모두 있을 때만 활성화. best=효율 우선, finetune=안전 우선.
    if best_pol is not None and ft_pol is not None:
        hybrid_pol = with_hybrid(best_pol, ft_pol, switch_c=args.hybrid_switch_c)
        controllers["RL-hybrid"] = hybrid_pol
        controllers["RL-hybrid+fb"] = with_safe_fallback(hybrid_pol)
        print(
            f"  [hybrid] safety 발동: zone≥{args.hybrid_switch_c}°C OR "
            f"cpu>{HYBRID_CPU_THRESH} OR it_power>{HYBRID_IT_POWER_THRESH}kW"
        )

    print(f"\n{'=' * 60}")
    print(f"  위기 시나리오 평가 (시나리오당 {args.episodes} ep)")
    print(f"{'=' * 60}")

    all_results = {name: run_one(name, pol, args.episodes) for name, pol in controllers.items()}
    print_table(all_results)


if __name__ == "__main__":
    main()
