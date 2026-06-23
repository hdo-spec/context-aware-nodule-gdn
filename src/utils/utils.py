# src/utils/utils.py
#
# ─── 역할 ────────────────────────────────────────────────────────────────────
#   device 자동 선택 + seed 고정을 하나로 통합.
#   기존 device.py + seed.py를 합침.
#
# ─── 사용 방법 ───────────────────────────────────────────────────────────────
#   from src.utils.utils import get_device, set_seed
#
#   set_seed(42)
#   device = get_device()

import os
import random

import numpy as np
import torch


def get_device() -> torch.device:
    """
    사용 가능한 디바이스 자동 선택 및 정보 출력.

    우선순위: CUDA(NVIDIA GPU) > MPS(Apple Silicon) > CPU

    Returns:
        torch.device 인스턴스
    """
    if torch.cuda.is_available():
        device   = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)                      # 첫 번째 GPU 이름
        gpu_mem  = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f'[device] CUDA 사용: {gpu_name} ({gpu_mem:.1f}GB)')

    elif torch.backends.mps.is_available():
        device = torch.device('mps')                                   # Apple Silicon (M1/M2)
        print('[device] MPS 사용 (Apple Silicon)')

    else:
        device = torch.device('cpu')
        print('[device] CPU 사용 (GPU 없음 — 학습 느릴 수 있음)')

    return device


def set_seed(seed: int = 42) -> None:
    """
    모든 랜덤 엔진에 동일한 seed 설정.

    왜 여러 라이브러리에 모두 설정하는가:
      Python / NumPy / PyTorch는 각각 독립적인 random 엔진을 가짐.
      하나만 고정해도 다른 곳의 랜덤성이 남아 실험마다 결과가 달라짐.

    Args:
        seed: 정수 seed 값 (기본값: config.py의 SEED=42)
    """
    random.seed(seed)                              # Python 표준 라이브러리
    np.random.seed(seed)                           # NumPy
    torch.manual_seed(seed)                        # PyTorch CPU
    torch.cuda.manual_seed_all(seed)               # PyTorch GPU (다중 GPU 포함)

    # cudnn 결정적 실행 강제
    # benchmark=False    : 매번 동일한 알고리즘 선택 (재현성 ↑, 속도 약간 ↓)
    # deterministic=True : 비결정적 알고리즘 비활성화
    torch.backends.cudnn.benchmark     = False
    torch.backends.cudnn.deterministic = True

    os.environ['PYTHONHASHSEED'] = str(seed)       # Python 해시 랜덤화 비활성화
    print(f'[seed] 고정 완료: seed={seed}')
