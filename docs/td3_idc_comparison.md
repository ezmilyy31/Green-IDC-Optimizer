# TD3 실험 결과 — IDCEnv 학습 및 비교

## 개요

`domain/controllers/rl_agent.py`에 TD3(Twin Delayed Deep Deterministic Policy Gradient)를
추가하여 커스텀 IDCEnv 환경에서 1M 스텝 학습하고, 기존 컨트롤러(SAC, PPO, Rule-based 등)와
PUE·온도 위반 기준으로 비교한다.

- **환경**: `IDCEnv` (커스텀, 실측 Google Cluster Trace 2019 + 기상청 ASOS)
- **평가**: 20 에피소드 (1일 × 20), `w_energy=0.8`, seed `42~61`

---

## 학습 설정

| 항목 | TD3 | SAC (비교) | PPO (비교) |
|---|---|---|---|
| 모델 파일 | `data/models/td3-idc-1m.zip` | `data/models/sac-wetbulb-1m.zip` | `data/models/ppo-idc-1m.zip` |
| 총 스텝 | 1,000,000 | 1,000,000 | 1,000,000 |
| Learning Rate | 1e-4 | 1e-4 | 3e-4 |
| Batch Size | 256 | 256 | 64 |
| Gamma | 0.99 | 0.99 | 0.99 |
| Buffer Size | 200,000 | 200,000 | — (on-policy) |
| Policy Delay | 2 | — | — |
| Target Policy Noise | 0.2 | — | — |
| Net Arch | [256, 256, 128] | [256, 256, 128] | 기본값 |
| w_energy | 0.8 | 0.8 | 0.8 |
| 보상 타입 | weighted | weighted | weighted |
| 에피소드 길이 | 288 스텝 (1일) | 288 스텝 | 288 스텝 |
| 순수 학습 시간 (추정) | 약 3~4시간 | 약 24분 | 약 22분 |

> **학습 시간 주의**: TD3 로그에 elapsed_sec 기준 대형 점프 3회 확인됨
> (step ~330K: +33분, step ~387K→422K: +9시간). 컴퓨터 절전/일시중단으로 추정.
> 순수 학습 구간 합산 기준 약 3~4시간 소요.

**학습 커맨드 (TD3):**

```bash
python -m domain.controllers.rl_agent \
  --algo td3 --custom-env \
  --total-timesteps 1000000 \
  --lr 1e-4 --batch-size 256 \
  --gamma 0.99 --w-energy 0.8 \
  --reward-type weighted --max-episode-steps 288 \
  --run-name td3-idc-1m --seed 0
```

---

## 평가 결과

> 평가 스크립트: `scripts/eval_baseline.py --model data/models/td3-idc-1m.zip --episodes 20`

| 컨트롤러 | PUE 평균 | 온도 위반 (°C) | 보상 평균 | 서버실 온도 (°C) |
|---|:---:|:---:|:---:|:---:|
| 고정 setpoint 24°C | 1.2999 | 0.0000 | -90.684 | 25.15 |
| Random | 1.2077 | 0.0000 | -29.571 | 25.59 |
| Rule-based | 1.1894 | 0.0000 | -17.427 | 25.01 |
| 고정 setpoint 20°C (설계값) | 1.1847 | 0.0000 | -14.370 | 25.01 |
| PPO (ppo-idc-1m) | 1.1768 | 0.0000 | -9.145 | 25.01 |
| **TD3 (td3-idc-1m)** | **1.1751** | **0.0000** | **-7.956** | **25.86** |
| PID (zone target=24°C) | 1.1752 | 0.0000 | -8.025 | 25.55 |
| SAC (sac-wetbulb-1m) | 1.1747 | 0.0000 | -7.724 | 25.76 |

---

## 학습 커브 분석

### TD3

| 구간 | 보상 범위 | 특이사항 |
|---|---|---|
| 초반 (0 ~ 50K) | -5.2 → -15.5 | 첫 rollout 후 악화 — replay buffer warm-up 기간 |
| 중반 (50K ~ 500K) | -8.7 ~ -21.7 | 진동 지속, 뚜렷한 수렴 없음 |
| 후반 (500K ~ 1M) | -5.0 ~ -27.9 | 최고 보상 -5.0 (step 864K), 이후 다시 진동 |

- 초기 보상: `-5.19` (step 2,048)
- 최고 보상: `-5.01` (step 864,256)
- 최종 보상: `-11.69` (step 999,424)

### SAC (비교)

- 초기 보상: `-33.9` → 80K 스텝 이후 양수 진입 → 최종 `+35.8`
- 명확한 단조 상승 수렴 곡선

### PPO (비교)

- 초기~최종 보상 모두 음수 (-5 ~ -25 진동), TD3와 유사한 패턴

---

## 분석

### RL 알고리즘별 PUE 비교

```
SAC   1.1747  ← 최저 (최우수)
TD3   1.1751  (+0.0004 vs SAC)
PID   1.1752  (+0.0005 vs SAC)
PPO   1.1768  (+0.0021 vs SAC)
Rule  1.1894  (+0.0147 vs SAC)
```

- TD3는 Rule-based 대비 **PUE 0.0143 감소** (overhead 기준 약 12.7% 개선)
- SAC 대비 PUE 차이는 **0.0004** — 실운용상 유의미한 차이 없음
- 보상 기준으로도 SAC(-7.724) > TD3(-7.956) > PID(-8.025) > PPO(-9.145) 순

### TD3 vs SAC — 성능 차이의 원인

| 항목 | SAC | TD3 |
|---|---|---|
| 정책 타입 | 확률적 (stochastic) | 결정론적 (deterministic) |
| 탐색 방식 | entropy 자동 조절 (`ent_coef="auto"`) | 학습 시 가우시안 노이즈 주입 |
| 학습 커브 | 명확한 수렴 (양수 도달) | 진동 지속, 수렴 불안정 |

IDCEnv는 1차원 연속 행동 공간(supply setpoint)이며 보상 신호가 PUE 기반으로 완만하게 변함.
이 환경에서 SAC의 entropy 정규화가 탐색-활용 균형을 자동으로 조절한 반면,
TD3의 고정 노이즈(σ=0.2) 방식은 초반 탐색에서 비효율적이었던 것으로 보임.

### TD3의 학습 시간 이슈

TD3는 off-policy임에도 불구하고 순수 학습 시간이 SAC(24분) 대비 훨씬 길었음(3~4시간).
이는 `policy_delay=2`로 인해 actor 업데이트 빈도가 절반으로 줄어들고,
target network 업데이트 연산이 추가되기 때문.
또한 로그에 3회의 대형 시간 점프(절전 추정)가 있어 실제 연속 학습 시간 측정이 어려움.

---

## 결론

| 항목 | 결과 |
|---|---|
| TD3의 Rule-based 대비 PUE 개선 | **−0.0143** (1.1894 → 1.1751) |
| TD3 vs SAC PUE 차이 | **+0.0004** (유의미한 차이 없음) |
| TD3 vs PPO PUE 차이 | **−0.0017** (TD3 우세) |
| 온도 위반 | **0** — 안전 제약 완전 준수 |
| 학습 안정성 | SAC > TD3 ≈ PPO (커브 진동 유사) |

TD3는 SAC와 거의 동등한 PUE 성능을 보였지만 학습 커브가 불안정하고 학습 시간이 길어,
IDCEnv 1차원 연속 제어 과제에서는 SAC 대비 실용적 이점이 없음.
최종 제어 에이전트는 **SAC(sac-wetbulb-1m)** 을 유지한다.

---

*실험 날짜: 2026-05-25*
*평가 환경: IDCEnv, w_energy=0.8, 20 에피소드*
*관련 문서: [ppo_idc_comparison.md](ppo_idc_comparison.md)*
