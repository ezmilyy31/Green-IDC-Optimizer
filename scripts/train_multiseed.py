"""SAC multi-seed 안정성 실험.

sac-wetbulb-1m과 동일한 하이퍼파라미터로 seed 4개를 병렬 학습한 뒤
각 모델의 PUE를 평가해 평균 ± 표준편차를 리포트한다.

사용법:
    # 학습 + 평가 (전체, ~1시간)
    python scripts/train_multiseed.py

    # 이미 학습한 모델을 평가만
    python scripts/train_multiseed.py --skip-train

    # seed 커스텀
    python scripts/train_multiseed.py --seeds 0 1 2 3
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가 (domain, core 모듈 접근)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

# ── 하이퍼파라미터 (sac-wetbulb-1m과 동일) ────────────────────────────────
SEEDS = [0, 1, 42, 100]
TOTAL_TIMESTEPS = 1_000_000
LR = 1e-4
BATCH_SIZE = 256
GAMMA = 0.99
W_ENERGY = 0.8
REWARD_TYPE = "weighted"
MAX_EPISODE_STEPS = 288   # idc_env.py EPISODE_STEPS와 동일

EVAL_SEED = 42            # 평가 시 고정 seed (공정 비교)
EVAL_STEPS = 288          # 평가 에피소드 길이 (1일)

MODEL_DIR = Path("data/models")
LOG_DIR = Path("data/logs")

RULE_PUE = 1.1894         # Rule-based 기준값 (비교용)


def _run_name(seed: int) -> str:
    return f"sac-multiseed-s{seed}"


# ── 학습 ──────────────────────────────────────────────────────────────────

def launch_training(seeds: list[int], total_timesteps: int = TOTAL_TIMESTEPS) -> list[tuple]:
    """seed별 학습 프로세스를 동시에 띄우고 (Popen, pid, log_file) 리스트 반환."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    procs = []
    for seed in seeds:
        cmd = [
            sys.executable, "-m", "domain.controllers.rl_agent",
            "--algo",              "sac",
            "--custom-env",
            "--total-timesteps",   str(total_timesteps),
            "--lr",                str(LR),
            "--batch-size",        str(BATCH_SIZE),
            "--gamma",             str(GAMMA),
            "--w-energy",          str(W_ENERGY),
            "--reward-type",       REWARD_TYPE,
            "--max-episode-steps", str(MAX_EPISODE_STEPS),
            "--device",            "cpu",
            "--run-name",          _run_name(seed),
            "--seed",              str(seed),
        ]
        log_path = LOG_DIR / f"multiseed-s{seed}.txt"
        log_file = open(log_path, "w", encoding="utf-8")
        proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
        procs.append((seed, proc, log_file))
        print(f"  [seed={seed:>3}] PID {proc.pid} 시작 → 로그: {log_path}")
    return procs


def wait_all(procs: list[tuple]) -> None:
    for seed, proc, log_file in procs:
        proc.wait()
        log_file.close()
        status = "완료" if proc.returncode == 0 else f"오류 (exit={proc.returncode})"
        print(f"  [seed={seed:>3}] {status}")


# ── 평가 ──────────────────────────────────────────────────────────────────

def evaluate(seed: int) -> dict:
    """모델 1개를 IDCEnv 1에피소드로 평가."""
    from domain.controllers.idc_env import IDCEnv
    from domain.controllers.rl_inference import RLInference

    model_path = MODEL_DIR / f"{_run_name(seed)}.zip"
    if not model_path.exists():
        return {"seed": seed, "pue": None, "violations": None, "note": "모델 없음"}

    try:
        inference = RLInference(str(model_path))
        env = IDCEnv(max_episode_steps=EVAL_STEPS)
        obs, _ = env.reset(seed=EVAL_SEED)

        pues, viols = [], []
        done = False
        while not done:
            setpoint = inference.predict(obs)
            obs, _, term, trunc, info = env.step(
                np.array([setpoint], dtype=np.float32)
            )
            pues.append(info["pue"])
            viols.append(info["temp_violation"])
            done = term or trunc

        return {
            "seed": seed,
            "pue": float(np.mean(pues)),
            "violations": int(sum(v > 0 for v in viols)),
            "note": "",
        }
    except Exception as exc:
        return {"seed": seed, "pue": None, "violations": None, "note": str(exc)}


# ── 결과 출력 ──────────────────────────────────────────────────────────────

def print_summary(results: list[dict]) -> None:
    valid = [r for r in results if r["pue"] is not None]
    pues = [r["pue"] for r in valid]

    W = 60
    print("\n" + "=" * W)
    print("  SAC Multi-seed 안정성 결과")
    print("=" * W)
    print(f"  {'Seed':>5}  {'PUE':>8}  {'위반':>6}  비고")
    print("-" * W)
    for r in results:
        if r["pue"] is not None:
            diff = r["pue"] - RULE_PUE
            print(f"  {r['seed']:>5}  {r['pue']:>8.4f}  {r['violations']:>6d}  "
                  f"(Rule 대비 {diff:+.4f})")
        else:
            print(f"  {r['seed']:>5}  {'N/A':>8}  {'N/A':>6}  {r['note']}")

    if len(pues) >= 2:
        mean, std = float(np.mean(pues)), float(np.std(pues))
        print("-" * W)
        print(f"  {'평균':>5}  {mean:>8.4f}")
        print(f"  {'±std':>5}  {std:>8.4f}")
        print(f"  {'min':>5}  {float(np.min(pues)):>8.4f}")
        print(f"  {'max':>5}  {float(np.max(pues)):>8.4f}")
        print(f"\n  Rule-based 기준: {RULE_PUE:.4f}")
        print(f"  SAC 평균 개선:  {RULE_PUE - mean:+.4f}  "
              f"(overhead 기준 {(RULE_PUE - mean) / (RULE_PUE - 1.0) * 100:.1f}%)")
    print("=" * W)


# ── 메인 ──────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS,
                        help="학습할 seed 목록 (기본: 0 1 42 100)")
    parser.add_argument("--total-timesteps", type=int, default=TOTAL_TIMESTEPS,
                        help=f"총 학습 스텝 수 (기본: {TOTAL_TIMESTEPS:,})")
    parser.add_argument("--skip-train", action="store_true",
                        help="학습을 건너뛰고 기존 모델 평가만 실행")
    args = parser.parse_args()

    seeds = args.seeds
    total_timesteps = args.total_timesteps

    if not args.skip_train:
        print(f"[multi-seed] {len(seeds)}개 seed 병렬 학습 시작: {seeds}")
        print(f"  하이퍼파라미터: lr={LR}, batch={BATCH_SIZE}, gamma={GAMMA}, "
              f"w_energy={W_ENERGY}, steps={total_timesteps:,}")
        t0 = time.time()
        procs = launch_training(seeds, total_timesteps)
        wait_all(procs)
        print(f"\n[multi-seed] 전체 학습 완료: {(time.time() - t0) / 60:.1f}분\n")

    print("[multi-seed] 평가 중...")
    results = [evaluate(s) for s in seeds]
    print_summary(results)


if __name__ == "__main__":
    main()
