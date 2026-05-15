"""위기 시나리오 평가 — best 모델 vs fine-tune 모델 vs rule-based 비교.

IDCEnv `force_scenario`로 4가지 시나리오를 강제 적용해 평가:
- normal: 평상시 (baseline 성능 보존 확인)
- server_surge: CPU util ×1.3
- heat_wave: 외기 온도 +5~10°C
- chiller_derate: 칠러 전력 ×1.5~3.3 (효율 저하)

사용법:
    python scripts/eval_crisis.py
    python scripts/eval_crisis.py --best data/models/sac-wetbulb-1m.zip \
        --finetune data/models/sac-wetbulb-dr-100k.zip --episodes 20
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
    parser.add_argument("--finetune", type=str, default="data/models/sac-wetbulb-dr-100k.zip")
    parser.add_argument("--episodes", type=int, default=10)
    args = parser.parse_args()

    controllers: dict[str, callable] = {"Rule-based": rule_based_policy}
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

    print(f"\n{'=' * 60}")
    print(f"  위기 시나리오 평가 (시나리오당 {args.episodes} ep)")
    print(f"{'=' * 60}")

    all_results = {name: run_one(name, pol, args.episodes) for name, pol in controllers.items()}
    print_table(all_results)


if __name__ == "__main__":
    main()
