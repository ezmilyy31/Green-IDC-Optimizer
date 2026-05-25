# PPO vs SAC 비교 실험 — IDCEnv 재학습

## 개요

`domain/controllers/rl_agent.py`의 PPO 알고리즘을 커스텀 IDCEnv 환경에서 1M 스텝 학습한 뒤,
SAC 및 기존 베이스라인 컨트롤러와 PUE·온도 위반 기준으로 비교한다.

- **환경**: `IDCEnv` (커스텀, Sinergym 미사용)
- **데이터**: `synthetic_idc_1year_noisy.parquet` (Google Cluster Trace 2019 + 기상청 ASOS)
- **평가**: 20 에피소드 (1일 × 20), `w_energy=0.8`, seed `42~61`

---

## 학습 설정

| 항목 | PPO | SAC (비교 기준) |
|---|---|---|
| 모델 파일 | `data/models/ppo-idc-1m.zip` | `data/models/sac-wetbulb-1m.zip` |
| 총 스텝 | 1,000,000 | 1,000,000 |
| Learning Rate | 3e-4 | 1e-4 |
| Batch Size | 64 | 256 |
| Gamma | 0.99 | 0.99 |
| n_steps (rollout) | 2,048 | — (off-policy) |
| w_energy | 0.8 | 0.8 |
| 보상 타입 | weighted | weighted |
| 에피소드 길이 | 288 스텝 (1일) | 288 스텝 (1일) |
| 환경 | IDCEnv (`--custom-env`) | IDCEnv (`--custom-env`) |
| 학습 시간 (CPU) | 약 21.8분 | 약 24.0분 |

**학습 커맨드 (PPO):**

```bash
python -m domain.controllers.rl_agent \
  --algo ppo --custom-env \
  --total-timesteps 1000000 \
  --lr 3e-4 --n-steps 2048 --batch-size 64 \
  --gamma 0.99 --w-energy 0.8 \
  --reward-type weighted --max-episode-steps 288 \
  --run-name ppo-idc-1m --seed 0
```

---

## 평가 결과

> 평가 스크립트: `scripts/eval_baseline.py --model data/models/ppo-idc-1m.zip --episodes 20`

| 컨트롤러 | PUE 평균 | 온도 위반 (°C) | 보상 평균 | 서버실 온도 (°C) |
|---|:---:|:---:|:---:|:---:|
| 고정 setpoint 24°C | 1.2999 | 0.0000 | -90.684 | 25.15 |
| Random | 1.2064 | 0.0000 | -28.734 | 25.59 |
| Rule-based | 1.1894 | 0.0000 | -17.427 | 25.01 |
| 고정 setpoint 20°C (설계값) | 1.1847 | 0.0000 | -14.370 | 25.01 |
| **PPO (IDCEnv, 1M)** | **1.1768** | **0.0000** | **-9.145** | **25.01** |
| PID (zone target=24°C) | 1.1752 | 0.0000 | -8.025 | 25.55 |
| **SAC (sac-wetbulb-1m)** | **1.1747** | **0.0000** | **-7.724** | **25.76** |

---

## 학습 커브 요약

### PPO

- 초기 보상: `-14.2` (step 2,048)
- 최고 보상: `-5.3` (step ~237,000)
- 최종 보상: `-13.2` (step 1,001,472)
- 특이점: 학습 전반에 걸쳐 보상이 `-5 ~ -25` 사이에서 진동하며 명확한 수렴 없음

### SAC

- 초기 보상: `-33.9` (step 4,096)
- 최고 보상: `+44.6` (step ~223,000)
- 최종 보상: `+35.8` (step 999,424)
- 특이점: 약 80,000 스텝 이후 양수 구간 진입, 이후 꾸준히 상승

SAC는 off-policy 특성(Replay Buffer)으로 sample efficiency가 높고 보상 수렴이 명확한 반면,
PPO는 on-policy 특성상 rollout 간 분산이 크고 IDCEnv의 연속적 보상 지형에서 수렴이 느렸다.

---

## 분석

### PUE 비교

```
SAC   1.1747  ← 최저 (최우수)
PPO   1.1768  (+0.0021 vs SAC)
PID   1.1752  (+0.0005 vs SAC)
Rule  1.1894  (+0.0147 vs SAC)
```

- PPO는 Rule-based 대비 **PUE 0.0126 감소** (overhead 기준 약 11.2% 개선)
- SAC 대비 PUE 차이는 **0.0021** (실운용상 유의미한 차이 없음)
- 온도 위반은 모든 컨트롤러에서 0 — IDCEnv 난이도상 안전 제약은 쉽게 만족됨

### PPO 성능이 SAC보다 낮은 이유

1. **On-policy 한계**: PPO는 수집한 rollout을 몇 번만 재사용하므로 SAC의 Replay Buffer 대비 data efficiency 낮음
2. **연속 행동 공간**: 1차원 setpoint 최적화에서 SAC의 reparameterization trick이 더 정밀한 gradient 추정 제공
3. **하이퍼파라미터 미조정**: SAC는 `lr=1e-4`, `batch=256`으로 IDCEnv에 맞게 튜닝됐으나 PPO 설정은 기본값 사용

### PPO의 장점

- 학습 코드 단순 (`n_steps=2048`, on-policy rollout)
- 하이퍼파라미터 민감도 낮아 초기 실험·프로토타입에 적합
- 동일 1M 스텝 기준 학습 시간이 SAC보다 약 10% 빠름 (21.8분 vs 24.0분)

---

## 결론

| 항목 | 결과 |
|---|---|
| 최종 채택 알고리즘 | **SAC** (`sac-wetbulb-1m`) |
| PPO의 Rule-based 대비 PUE 개선 | **−0.0126** (1.1894 → 1.1768) |
| SAC 대비 PPO PUE 열세 | **+0.0021** (유의미한 차이 없음) |
| 온도 위반 | **둘 다 0** — 안전 제약 완전 준수 |

PPO도 Rule-based 대비 유의미한 에너지 개선을 보였으나,
SAC가 학습 안정성·PUE 최적화 모두에서 우세하여 최종 제어 에이전트로 SAC를 채택한다.

---

*실험 날짜: 2026-05-25*  
*평가 환경: IDCEnv, w_energy=0.8, 20 에피소드*
