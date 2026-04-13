#!/bin/bash
# GPU 서버 환경 설정 스크립트
# 사용법: bash scripts/setup_gpu_server.sh
set -e

echo "=== 1. 시스템 패키지 ==="
apt update && apt install -y python3 python3-pip git wget curl libx11-6
ln -sf /usr/bin/python3 /usr/bin/python
ln -sf /usr/bin/pip3 /usr/bin/pip

echo "=== 2. PyTorch (CUDA 12.1) ==="
pip install torch --index-url https://download.pytorch.org/whl/cu121

echo "=== 3. Python 패키지 ==="
pip install stable-baselines3 gymnasium numpy pandas pyyaml wandb

echo "=== 4. Sinergym (순서 중요) ==="
pip install git+https://github.com/AlejandroCN7/opyplus.git@master
pip install sinergym --no-deps
pip install eppy xlsxwriter

echo "=== 5. EnergyPlus 설치 확인 ==="
if [ ! -d "/usr/local/EnergyPlus-23.2.0-7636e6b3e9" ]; then
    echo "EnergyPlus 없음 — 설치 중..."
    wget https://github.com/NREL/EnergyPlus/releases/download/v23.2.0/EnergyPlus-23.2.0-7636e6b3e9-Linux-Ubuntu22.04-x86_64.sh
    chmod +x EnergyPlus-23.2.0-7636e6b3e9-Linux-Ubuntu22.04-x86_64.sh
    echo "y" | ./EnergyPlus-23.2.0-7636e6b3e9-Linux-Ubuntu22.04-x86_64.sh
else
    echo "EnergyPlus 이미 설치됨 — 스킵"
fi

echo "=== 6. EnergyPlus 경로 설정 ==="
export PYTHONPATH=$PYTHONPATH:/usr/local/EnergyPlus-23.2.0-7636e6b3e9
echo 'export PYTHONPATH=$PYTHONPATH:/usr/local/EnergyPlus-23.2.0-7636e6b3e9' >> ~/.bashrc

echo "=== 7. GPU 확인 ==="
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('PyTorch version:', torch.__version__)"

echo "=== 8. 동작 검증 ==="
mkdir -p logs
python -m domain.controllers.rl_agent --total-timesteps 2048 --run-name test
echo "=== 완료! '모델 저장 완료' 뜨면 성공 ==="
