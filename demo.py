# demo.py
#
# ─── 역할 ────────────────────────────────────────────────────────────────────
#   학습된 GDN(+CBAM) 모델로 폐결절 2.5D patch 1장의 악성도를 예측하는
#   가장 간단한 형태의 추론 데모. 전체 평가 파이프라인(evaluate.py)을
#   돌릴 필요 없이, 단일 patch에 대한 결과를 빠르게 확인하기 위한 스크립트.
#
# ─── 전제 ────────────────────────────────────────────────────────────────────
#   repo 루트(이 파일이 있는 위치)에서 실행해야 src.models.models를
#   import할 수 있음. CBAM은 별도 클래스가 아니라 GDN 생성자의
#   use_cbam1 / use_cbam2 / use_cbam_final 플래그로 켜고 끄는 구조이므로,
#   build_model()의 플래그 조합이 best_model.pth를 학습할 때와 동일해야
#   state_dict가 정확히 매칭됨 (다르면 load_state_dict()에서 즉시 에러).
#
# ─── patch 형식 ──────────────────────────────────────────────────────────────
#   (3, H, W) float32 .npy — 2.5D crop (k-1, k, k+1 슬라이스), [0,1] 정규화.
#   src/preprocessing/export_patches.py가 생성하는 raw patch와 동일한 형식.
#
# ─── 사용 방법 ───────────────────────────────────────────────────────────────
#   python demo.py --patch sample_results/example_patch.npy
#
#   threshold를 지정하지 않으면 sample_results/result.json에 저장된
#   Youden threshold를 자동으로 사용하고, 없으면 0.5로 fallback.
#   python demo.py --patch sample_results/example_patch.npy --threshold 0.42

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.models.models import GDN


# ─────────────────────────────────────────────────────────────────────────────
# 섹션 1. 모델 / patch 로드
# ─────────────────────────────────────────────────────────────────────────────

def build_model() -> nn.Module:
    """
    src/models/models.py의 GDN 인스턴스 생성.

    최종 채택 구성 (README 6.2 ablation 기준):
      dilation d2=3 (models.py GDLayer 기본값) +
      CBAM(gd3+gd4 이후, use_cbam2) + CBAM(gd5 이후, use_cbam_final)
      → use_cbam1=False, use_cbam2=True, use_cbam_final=True

    다른 실험(EXP-A~D) 가중치를 불러오려면 아래 플래그를 그에 맞게 바꿔야 함.
    플래그가 best_model.pth와 다르면 CBAM 관련 키가 없거나 남아서
    load_state_dict()에서 즉시 에러가 나므로, 모델이 안 맞으면 바로 알 수 있음.
    """
    return GDN(in_ch=3, num_classes=1,
              use_cbam1=False, use_cbam2=True, use_cbam_final=True)


def load_patch(patch_path: str) -> torch.Tensor:
    """
    (C, H, W) .npy → (1, C, H, W) 배치 텐서.
    export_patches.py가 저장하는 float16 raw patch도 그대로 처리 가능.
    """
    patch = np.load(patch_path).astype(np.float32)
    if patch.ndim != 3:
        raise ValueError(
            f'patch shape이 (C, H, W) 형식이 아님: {patch.shape}\n'
            f'src/preprocessing/export_patches.py로 생성한 raw patch를 사용해야 함.'
        )
    return torch.from_numpy(patch).unsqueeze(0)


def load_threshold(result_dir: Path, default: float = 0.5) -> float:
    """
    result_dir/result.json 또는 config.json에 Youden threshold가
    저장돼 있으면 그 값을 사용, 없으면 default(0.5).
    """
    for fname in ('result.json', 'config.json'):
        path = result_dir / fname
        if not path.exists():
            continue
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for key in ('youden_threshold', 'threshold'):
            if key in data:
                return float(data[key])
    return default


# ─────────────────────────────────────────────────────────────────────────────
# 섹션 2. 추론
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict(model: nn.Module, patch: torch.Tensor, device: torch.device) -> float:
    model.eval()
    patch = patch.to(device)
    logit = model(patch)
    prob  = torch.sigmoid(logit).item()
    return prob


# ─────────────────────────────────────────────────────────────────────────────
# 섹션 3. 메인
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description='폐결절 악성도 추론 데모 (GDN 단일 patch)')
    parser.add_argument('--patch', type=str, required=True,
                        help='2.5D patch .npy 파일 경로, shape=(3, H, W)')
    parser.add_argument('--model_path', type=str,
                        default='sample_results/best_model.pth')
    parser.add_argument('--threshold', type=float, default=None,
                        help='지정하지 않으면 sample_results/result.json의 '
                             'Youden threshold 사용, 없으면 0.5')
    args = parser.parse_args()

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f'모델 파일 없음: {model_path}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = build_model()
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)

    patch = load_patch(args.patch)

    threshold = args.threshold
    if threshold is None:
        threshold = load_threshold(model_path.parent)

    prob = predict(model, patch, device)
    pred = 'Malignant (악성)' if prob >= threshold else 'Benign (양성)'

    print('─' * 42)
    print(f'  입력 patch     : {args.patch}')
    print(f'  악성 확률      : {prob:.4f}')
    print(f'  적용 threshold : {threshold:.4f}  (Youden 기준)')
    print(f'  예측 결과      : {pred}')
    print('─' * 42)


if __name__ == '__main__':
    main()
