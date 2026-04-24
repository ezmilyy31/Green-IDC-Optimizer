#!/bin/bash

# =================================================================
# Green-IDC-Optimizer 통합 학습 스크립트
# 모든 Forecast 모델(IT Load, Cooling Demand)을 일괄 학습 및 평가합니다.
# =================================================================

# 색상 정의 (출력 가독성용)
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}====================================================${NC}"
echo -e "${GREEN}  Green-IDC-Optimizer 모델 통합 학습을 시작합니다.${NC}"
echo -e "${BLUE}====================================================${NC}"

# 1. IT Load (IT 부하) 모델 학습
echo -e "\n${GREEN}[Step 1/2] IT Load 예측 모델 학습 시작${NC}"
echo "----------------------------------------------------"

echo -e "${BLUE}1-1. IT Load: Point (LGBM)${NC}"
uv run python -m domain.forecasting.train.train_lgbm_it_load

echo -e "\n${BLUE}1-2. IT Load: Quantile Bundle (LGBM)${NC}"
uv run python -m domain.forecasting.train.train_lgbm_quantile_it_load

echo -e "\n${BLUE}1-3. IT Load: Baseline (Moving Average)${NC}"
uv run python -m domain.forecasting.train.train_moving_avg_it_load


# 2. Cooling Demand (냉방 수요) 모델 학습
echo -e "\n${GREEN}[Step 2/2] Cooling Demand 예측 모델 학습 시작${NC}"
echo "----------------------------------------------------"

echo -e "${BLUE}2-1. Cooling Demand: Point (LGBM)${NC}"
uv run python -m domain.forecasting.train.train_lgbm_cooling_demand

echo -e "\n${BLUE}2-2. Cooling Demand: Quantile Bundle (LGBM)${NC}"
uv run python -m domain.forecasting.train.train_lgbm_quantile_cooling_demand

echo -e "\n${BLUE}2-3. Cooling Demand: Baseline (Moving Average)${NC}"
uv run python -m domain.forecasting.train.train_moving_avg_cooling_demand

echo -e "\n${BLUE}====================================================${NC}"
echo -e "${GREEN}  모든 모델 학습 및 평가 리포트 작성이 완료되었습니다!${NC}"
echo -e "${BLUE}  생성된 모델 파일은 data/models/ 에서 확인 가능합니다.${NC}"
echo -e "${BLUE}====================================================${NC}"