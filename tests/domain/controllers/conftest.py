import sys
from pathlib import Path

# pytest pythonpath 미설정 우회: 프로젝트 루트를 이 테스트 디렉토리 한정으로 sys.path 에 주입
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pytest

from domain.controllers.idc_env import IDCEnv


@pytest.fixture
def env():
    return IDCEnv(max_episode_steps=10)
