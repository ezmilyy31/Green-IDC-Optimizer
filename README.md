# AI Green IDC — 데이터센터 지능형 냉각/전력 최적화 시스템

IT 부하 예측, 강화학습 기반 냉각 제어, PUE 최적화를 통해 데이터센터 에너지 효율을 개선하는 시스템입니다.

## 빠른 시작

```bash
cp .env.example .env
docker compose up --build
```

> 상세 실행 방법은 [docs/RUNBOOK.md](docs/RUNBOOK.md)를 참고하세요.

---

## 프로젝트 구조

```
AI-Green-IDC/
├─ apps/                    # 실행 가능한 서비스 계층
│  ├─ api/                  # API 게이트웨이 (FastAPI)
│  ├─ forecast_service/     # 부하/냉각 수요 예측 서비스
│  ├─ control_service/      # 제어 서비스 (Rule-based / PID / RL)
│  ├─ simulation_service/   # 열역학 시뮬레이션 서비스
│  └─ dashboard/            # Streamlit 대시보드
│
├─ core/                    # 공용 모듈 (설정, 스키마, 클라이언트)
├─ domain/                  # 핵심 비즈니스 로직
│  ├─ thermodynamics/       # 냉각 부하, PUE, 칠러 계산
│  ├─ forecasting/          # LightGBM / LSTM 예측 모델
│  └─ controllers/          # Rule-based, PID, RL 제어기
│
├─ data/                    # 데이터 및 학습된 모델 저장
├─ docs/                    # 문서 (RUNBOOK, API 명세, 설계 문서)
├─ scripts/                 # 학습/평가 스크립트
└─ tests/                   # 테스트 코드
```

---

## 기여 방법

### 브랜치 전략

```
main       ← 배포 브랜치
develop    ← 통합 브랜치 (PR 대상)
feat/...   ← 기능 개발
fix/...    ← 버그 수정
chore/...  ← 설정, 문서, 빌드
```

### 커밋 컨벤션

```
FEAT:     새로운 기능
FIX:      버그 수정
DOCS:     문서 수정
CHORE:    빌드/설정 변경
REFACTOR: 리팩토링
```

---

## 팀원 소개
미정
