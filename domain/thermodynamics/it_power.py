"""
IT 장비 전력 소비 모델 (SPECpower_ssj2008 기반)

서버의 CPU 사용률에 따른 전력 소비량을 계산한다.
공식: P = P_idle + (P_max - P_idle) × cpu_utilization
"""

from dataclasses import dataclass
from enum import Enum

from core.config.constants import (
    CPU_SERVER_P_IDLE_W,
    CPU_SERVER_P_MAX_W,
    GPU_SERVER_P_IDLE_W,
    GPU_SERVER_P_MAX_W,
)


class ServerType(str, Enum):
    CPU = "cpu"
    GPU = "gpu"


@dataclass(frozen=True)
class ServerSpec:
    """서버 전력 사양 (SPECpower_ssj2008 기준)"""

    p_idle_w: float  # 유휴 상태 전력 (Watts)
    p_max_w: float   # 최대 부하 전력 (Watts)
    count: int = 1   # 서버 대수


# SPECpower_ssj2008 기준 참조값
CPU_SERVER = ServerSpec(p_idle_w=CPU_SERVER_P_IDLE_W, p_max_w=CPU_SERVER_P_MAX_W)
GPU_SERVER = ServerSpec(p_idle_w=GPU_SERVER_P_IDLE_W, p_max_w=GPU_SERVER_P_MAX_W)


def calculate_server_power_w(
    cpu_utilization: float,
    server_type: ServerType = ServerType.CPU,
    custom_spec: ServerSpec | None = None,
) -> float:
    """
    서버 1대의 전력 소비량을 계산한다 (SPECpower 공식).

    공식: P = P_idle + (P_max - P_idle) × cpu_utilization

    Args:
        cpu_utilization: CPU 사용률 (0.0 ~ 1.0)
        server_type: 서버 종류 (CPU / GPU)
        custom_spec: 사용자 정의 서버 사양 (None이면 기본값 사용)

    Returns:
        서버 1대의 전력 소비량 (Watts)

    Raises:
        ValueError: cpu_utilization이 0~1 범위를 벗어날 때
    """
    if not 0.0 <= cpu_utilization <= 1.0:
        raise ValueError(f"cpu_utilization은 0.0~1.0 사이여야 합니다. 입력값: {cpu_utilization}")

    spec = custom_spec or (CPU_SERVER if server_type == ServerType.CPU else GPU_SERVER)
    return spec.p_idle_w + (spec.p_max_w - spec.p_idle_w) * cpu_utilization


def calculate_total_it_power_kw(
    cpu_utilization: float,
    num_cpu_servers: int,
    num_gpu_servers: int,
    cpu_spec: ServerSpec | None = None,
    gpu_spec: ServerSpec | None = None,
) -> float:
    """
    데이터센터 전체 IT 전력 소비량을 계산한다.

    Args:
        cpu_utilization: 평균 CPU 사용률 (0.0 ~ 1.0)
        num_cpu_servers: CPU 서버 대수 (반드시 실제 구성에 맞게 지정)
        num_gpu_servers: GPU 서버 대수
        cpu_spec: CPU 서버 사양 (None이면 SPECpower 기본값)
        gpu_spec: GPU 서버 사양 (None이면 SPECpower 기본값)

    Returns:
        전체 IT 전력 소비량 (kW)
    """
    cpu_spec = cpu_spec or CPU_SERVER
    gpu_spec = gpu_spec or GPU_SERVER

    cpu_power_w = calculate_server_power_w(cpu_utilization, ServerType.CPU, cpu_spec)
    gpu_power_w = calculate_server_power_w(cpu_utilization, ServerType.GPU, gpu_spec)

    total_w = cpu_power_w * num_cpu_servers + gpu_power_w * num_gpu_servers
    return total_w / 1000.0  # W → kW
