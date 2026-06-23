# src/evaluation/compare_gradcam.py
#
# ─── 역할 ────────────────────────────────────────────────────────────────────
#   ConvNeXt / DualConvNeXt(small=32px) / DualConvNeXt(large=96px) / GDN
#   네 행으로 나란히 비교.
#
#   - 세 모델이 모두 TP/TN으로 맞춘 공통 결절만 사용.
#   - DualConvNeXt는 small branch(32px)와 large branch(96px) Grad-CAM을 각각 행으로 분리.
#   - cherry-pick 방지: diameter_max_mm(또는 volume_mm3) 오름차순 정렬 후 상위 N개.
#
# ─── Grad-CAM 타겟 레이어 ─────────────────────────────────────────────────────
#   convnext           -> model.stage4[-1].dwconv
#   dual_convnext small-> model.small_branch.stage3[-1].dwconv  (down3 없음 → stage3)
#   dual_convnext large-> model.large_branch.stage4[-1].dwconv
#   gdn                -> model.gd5.conv_d1
#
# ─── 출력 구조 ───────────────────────────────────────────────────────────────
#   {output_dir}/
#     comparison/
#       TP_01_{id}_n{idx}_comparison.png  ← 샘플 1개, 4행(모델) × 2열(원본|CAM)
#       TP_grid.png                       ← 케이스 전체 4행 × N열 grid
#       TN_*.png
#     individual/
#       TP_01_{id}_n{idx}_convnext.png
#       TP_01_{id}_n{idx}_dual_convnext_small.png
#       TP_01_{id}_n{idx}_dual_convnext_large.png
#       TP_01_{id}_n{idx}_gdn.png
#     selection_log.json
#
# ─── 사용 예시 ───────────────────────────────────────────────────────────────
#   conda run -n resnet --no-capture-output python -m src.evaluation.compare_gradcam \
#     --gdn_dir       outputs/experiments/260611_gdn_32_ep50_aug0 \
#     --convnext_dir  outputs/experiments/260617_convnext_32_ep50_aug0 \
#     --dual_dir      outputs/experiments/260615_dual_convnext_32+96_ep50_aug0 \
#     --n_per_case 5 \
#     --output_dir outputs/gradcam_comparison

import argparse
import json
import csv as csv_module
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib
import matplotlib.lines
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from src.configs.config import PROCESSED_ROOT
from src.datasets.dataset import get_dataloaders, get_dual_dataloaders
from src.models.models import GDN, ConvNeXt, DualConvNeXt
from src.utils.utils import get_device

NPY_CACHE_ROOT = PROCESSED_ROOT / 'npy_cache'


# ─────────────────────────────────────────────────────────────────────────────
# 섹션 0. 행 순서 / 표시 이름 상수
#
#   비교 페이지 행은 4개:
#     'convnext'            ConvNeXt
#     'dual_convnext_small' DualConvNeXt (small=32px)
#     'dual_convnext_large' DualConvNeXt (large=96px)
#     'gdn'                 GDN
#
#   실제 모델 키(모델 인스턴스/results 딕셔너리 접근용)는 3개:
#     'convnext', 'dual_convnext', 'gdn'
# ─────────────────────────────────────────────────────────────────────────────

ROW_ORDER = ['convnext', 'dual_convnext_small', 'dual_convnext_large', 'gdn']

ROW_DISPLAY_NAME = {
    'convnext'           : 'ConvNeXt',
    'dual_convnext_small': 'DualConvNeXt\n(small=32px)',
    'dual_convnext_large': 'DualConvNeXt\n(large=96px)',
    'gdn'                : 'GDN',
}

ROW_DISPLAY_NAME_INLINE = {
    'convnext'           : 'ConvNeXt',
    'dual_convnext_small': 'DualConvNeXt (small=32px)',
    'dual_convnext_large': 'DualConvNeXt (large=96px)',
    'gdn'                : 'GDN',
}

# 실제 모델 인스턴스/results 딕셔너리 키
MODEL_KEYS = ['convnext', 'dual_convnext', 'gdn']


# ─────────────────────────────────────────────────────────────────────────────
# 섹션 1. 모델/설정 로드
# ─────────────────────────────────────────────────────────────────────────────

def load_exp_config(exp_dir: Path) -> dict:
    config_path = exp_dir / 'config.json'
    if not config_path.exists():
        raise FileNotFoundError(f'config.json 없음: {config_path}')
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_model_baseline(model_name: str, exp_dir: Path, device: torch.device) -> nn.Module:
    if model_name == 'gdn':
        model = GDN(in_ch=3, num_classes=1)
    elif model_name == 'convnext':
        model = ConvNeXt(in_ch=3, num_classes=1)
    elif model_name == 'dual_convnext':
        model = DualConvNeXt(num_classes=1)
    else:
        raise ValueError(f'알 수 없는 모델: {model_name}')

    model_path = exp_dir / 'best_model.pth'
    if not model_path.exists():
        raise FileNotFoundError(f'best_model.pth 없음: {model_path}')

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 섹션 2. test set 결절 단위 결과 수집 (threshold=0.5 고정)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def collect_test_results_baseline(model: nn.Module, loader, device: torch.device,
                                  is_dual: bool = False) -> dict:
    model.eval()
    nodule_data = defaultdict(lambda: {'probs': [], 'label': None})

    for batch in loader:
        if is_dual:
            patch_small, patch_large, labels, subject_ids, nodule_idxs, z_idxs = batch
            patch_small = patch_small.to(device)
            patch_large = patch_large.to(device)
            logits = model(patch_small, patch_large)
        else:
            patches, labels, subject_ids, nodule_idxs, z_idxs = batch
            patches = patches.to(device)
            logits  = model(patches)

        probs = torch.sigmoid(logits).cpu().numpy().flatten()
        lbls  = labels.numpy().flatten().astype(int)

        for j in range(len(lbls)):
            key   = (subject_ids[j], nodule_idxs[j])
            label = int(lbls[j])
            if nodule_data[key]['label'] is not None and nodule_data[key]['label'] != label:
                raise ValueError(f'같은 결절 key에 서로 다른 label: {key}')
            nodule_data[key]['probs'].append(float(probs[j]))
            nodule_data[key]['label'] = label

    nodule_keys   = list(nodule_data.keys())
    nodule_probs  = {k: float(np.mean(nodule_data[k]['probs'])) for k in nodule_keys}
    nodule_labels = {k: nodule_data[k]['label'] for k in nodule_keys}
    nodule_preds  = {k: (1 if nodule_probs[k] >= 0.5 else 0) for k in nodule_keys}

    return {'probs': nodule_probs, 'labels': nodule_labels, 'preds': nodule_preds}


def get_case(nodule_results: dict, key: tuple) -> str:
    label = nodule_results['labels'][key]
    pred  = nodule_results['preds'][key]
    if label == 1 and pred == 1:
        return 'TP'
    if label == 0 and pred == 0:
        return 'TN'
    if label == 0 and pred == 1:
        return 'FP'
    return 'FN'


# ─────────────────────────────────────────────────────────────────────────────
# 섹션 3. 결절 메타데이터 로드
# ─────────────────────────────────────────────────────────────────────────────

def load_nodule_meta(crop_size: int) -> dict:
    raw_csv = NPY_CACHE_ROOT / f'labels_raw_{crop_size}.csv'
    meta = {}
    if not raw_csv.exists():
        return meta
    with open(raw_csv, 'r', encoding='utf-8') as f:
        for row in csv_module.DictReader(f):
            key = (row['subject_id'], row['nodule_idx'])
            if key in meta:
                continue
            try:
                diam = float(row['diameter_max_mm']) if row.get('diameter_max_mm') else None
            except ValueError:
                diam = None
            try:
                vol = float(row['volume_mm3']) if row.get('volume_mm3') else None
            except ValueError:
                vol = None
            meta[key] = {'diameter_max_mm': diam, 'volume_mm3': vol}
    return meta


# ─────────────────────────────────────────────────────────────────────────────
# 섹션 4. Grad-CAM
# ─────────────────────────────────────────────────────────────────────────────

class GradCAM:
    """
    GDN의 gd5.conv_d1 뒤 inplace ReLU로 인한 RuntimeError 방지를 위해
    forward 전에 모든 inplace ReLU를 비활성화하고 hook 제거 시 복원.
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model        = model
        self.activations  = None
        self.gradients    = None
        self._relu_states = []

        self._disable_inplace_relu()
        self._fwd_hook = target_layer.register_forward_hook(self._save_activation)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _disable_inplace_relu(self):
        for module in self.model.modules():
            if isinstance(module, nn.ReLU) and module.inplace:
                self._relu_states.append((module, True))
                module.inplace = False

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, *model_inputs: torch.Tensor, target_class: int = 1) -> np.ndarray:
        self.model.zero_grad()
        logit = self.model(*model_inputs)
        logit[0, 0].backward()

        alpha   = self.gradients.mean(dim=[2, 3], keepdim=True)
        heatmap = (alpha * self.activations).sum(dim=1, keepdim=True)
        heatmap = torch.relu(heatmap)
        heatmap = heatmap.squeeze().cpu().numpy()

        if heatmap.max() > heatmap.min():
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        else:
            heatmap = np.zeros_like(heatmap)
        return heatmap

    def remove_hooks(self):
        self._fwd_hook.remove()
        self._bwd_hook.remove()
        for module, inplace in self._relu_states:
            module.inplace = inplace
        self._relu_states = []


def get_target_layer(model: nn.Module, row_key: str) -> nn.Module:
    """
    행 키(row_key) 기준 타겟 레이어 반환.
      convnext            -> model.stage4[-1].dwconv
      dual_convnext_small -> model.small_branch.stage3[-1].dwconv
                            (ConvNeXtSmall은 down3 없이 stage3까지)
      dual_convnext_large -> model.large_branch.stage4[-1].dwconv
      gdn                 -> model.gd5.conv_d1
    """
    if row_key == 'convnext':
        return model.stage4[-1].dwconv
    elif row_key == 'dual_convnext_small':
        return model.small_branch.stage3[-1].dwconv
    elif row_key == 'dual_convnext_large':
        return model.large_branch.stage4[-1].dwconv
    elif row_key == 'gdn':
        return model.gd5.conv_d1
    else:
        raise ValueError(f'알 수 없는 row_key: {row_key}')


def overlay_heatmap(image: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    from matplotlib.cm import jet
    if heatmap.shape != image.shape:
        from PIL import Image as PILImage
        heatmap_resized = np.array(
            PILImage.fromarray((heatmap * 255).astype(np.uint8)).resize(
                (image.shape[1], image.shape[0]), PILImage.BILINEAR
            )
        ) / 255.0
    else:
        heatmap_resized = heatmap

    image_rgb   = np.stack([image, image, image], axis=-1)
    heatmap_rgb = jet(heatmap_resized)[:, :, :3]
    overlay     = 0.4 * image_rgb + 0.6 * heatmap_rgb
    return np.clip(overlay, 0, 1)


# ─────────────────────────────────────────────────────────────────────────────
# 섹션 5. 결절 → 패치 매핑
# ─────────────────────────────────────────────────────────────────────────────

def build_center_sample_map(test_loader, is_dual: bool) -> dict:
    """{(subject_id, nodule_idx): sample}. 결절당 처음 등장한 슬라이스 사용."""
    center_map = {}
    for batch in test_loader:
        if is_dual:
            patch_small, patch_large, labels, subject_ids, nodule_idxs, z_idxs = batch
            for i in range(len(labels)):
                key = (subject_ids[i], nodule_idxs[i])
                if key not in center_map:
                    center_map[key] = (patch_small[i], patch_large[i], int(labels[i]))
        else:
            patches, labels, subject_ids, nodule_idxs, z_idxs = batch
            for i in range(len(labels)):
                key = (subject_ids[i], nodule_idxs[i])
                if key not in center_map:
                    center_map[key] = (patches[i], int(labels[i]))
    return center_map


def compute_row_results(row_key: str, model: nn.Module, sample, device) -> tuple:
    """
    행 키 하나에 대한 (patch_np, heatmap, label) 반환.

    dual_convnext_small/large는 같은 DualConvNeXt 인스턴스에서
    타겟 레이어만 다르게 지정해 각각 독립적으로 Grad-CAM 계산.
    patch_np 기준:
      dual_convnext_small -> patch_small (32px)
      dual_convnext_large -> patch_large (96px)
      convnext / gdn      -> 단일 patch
    """
    is_dual = row_key in ('dual_convnext_small', 'dual_convnext_large')

    target_layer = get_target_layer(model, row_key)
    gradcam      = GradCAM(model, target_layer)

    if is_dual:
        patch_small, patch_large, label = sample
        input_small = patch_small.unsqueeze(0).to(device)
        input_large = patch_large.unsqueeze(0).to(device)
        input_small.requires_grad_(True)
        input_large.requires_grad_(True)
        heatmap  = gradcam.generate(input_small, input_large, target_class=1)
        patch_np = patch_small.numpy() if row_key == 'dual_convnext_small' \
                   else patch_large.numpy()
    else:
        patch, label = sample
        input_tensor = patch.unsqueeze(0).to(device)
        input_tensor.requires_grad_(True)
        heatmap  = gradcam.generate(input_tensor, target_class=1)
        patch_np = patch.numpy()

    gradcam.remove_hooks()
    return patch_np, heatmap, label


# ─────────────────────────────────────────────────────────────────────────────
# 섹션 6. 시각화
# ─────────────────────────────────────────────────────────────────────────────

CENTER_CH = 1   # patch_np[0]=k-1, [1]=k(center), [2]=k+1

LEGEND_PATCHES = [
    mpatches.Patch(color='red',  label='High attention'),
    mpatches.Patch(color='blue', label='Low attention'),
]

# dual_convnext_small/large가 ROW_ORDER 안에서 차지하는 인덱스
_DUAL_ROW_INDICES = (
    ROW_ORDER.index('dual_convnext_small'),
    ROW_ORDER.index('dual_convnext_large'),
)


def draw_dual_bracket(fig: plt.Figure, axes_col: np.ndarray,
                      n_rows: int, label_x_fig: float = 0.01) -> None:
    """
    DualConvNeXt 두 행(dual_convnext_small, dual_convnext_large)을
    figure 좌표계에서 세로 선 + 양 끝 가로 티크 + 레이블로 묶어 표시.

    axes_col: shape (n_rows,) — 맨 왼쪽 열의 axes 배열.
              tight_layout 이후 호출해야 axes 위치가 확정됨.
    label_x_fig: 브라켓 텍스트의 figure x 좌표 (0=left edge).
                 tight_layout의 left rect에 맞게 조정.
    """
    top_row_idx, bot_row_idx = _DUAL_ROW_INDICES

    # axes bbox는 figure 좌표(0~1)로 get_position() 반환
    top_bbox = axes_col[top_row_idx].get_position()
    bot_bbox = axes_col[bot_row_idx].get_position()

    y_top = top_bbox.y1   # 위쪽 행의 top
    y_bot = bot_bbox.y0   # 아래쪽 행의 bottom
    y_mid = (y_top + y_bot) / 2.0

    x_tick  = label_x_fig + 0.012   # 세로 선 x
    tick_w  = 0.008                  # 가로 티크 길이
    lw      = 1.8

    # 세로 선
    fig.lines.append(matplotlib.lines.Line2D(
        [x_tick, x_tick], [y_bot, y_top],
        transform=fig.transFigure, color='#444444',
        linewidth=lw, clip_on=False
    ))
    # 위 티크
    fig.lines.append(matplotlib.lines.Line2D(
        [x_tick, x_tick + tick_w], [y_top, y_top],
        transform=fig.transFigure, color='#444444',
        linewidth=lw, clip_on=False
    ))
    # 아래 티크
    fig.lines.append(matplotlib.lines.Line2D(
        [x_tick, x_tick + tick_w], [y_bot, y_bot],
        transform=fig.transFigure, color='#444444',
        linewidth=lw, clip_on=False
    ))

    # 레이블
    fig.text(
        label_x_fig, y_mid, 'DualConvNeXt',
        transform=fig.transFigure,
        fontsize=18, fontweight='bold', color='#222222',
        va='center', ha='left', rotation=90,
    )


def save_comparison_page(row_results: dict, label: int, case: str,
                         rank: int, key: tuple, meta: dict,
                         out_dir: Path) -> Path:
    """
    샘플 1개 × 4행(모델) 비교 페이지.
    열: [원본(k) | Grad-CAM 오버레이]
    행: ConvNeXt / DualConvNeXt(small) / DualConvNeXt(large) / GDN

    dual 두 행은 figure 왼쪽에 세로 브라켓 + "DualConvNeXt" 레이블로 묶어 표시.
    row_results: {row_key: (patch_np, heatmap)}
    """
    # dual 행의 짧은 레이블 (브라켓이 "DualConvNeXt" 역할)
    _short_label = {
        'convnext'           : 'ConvNeXt',
        'dual_convnext_small': '(small=32px)',
        'dual_convnext_large': '(large=96px)',
        'gdn'                : 'GDN',
    }

    n_rows = len(ROW_ORDER)
    fig, axes = plt.subplots(n_rows, 2, figsize=(11, 5.0 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for row, row_key in enumerate(ROW_ORDER):
        entry = row_results.get(row_key)
        if entry is None:
            axes[row, 0].axis('off')
            axes[row, 1].axis('off')
            continue

        patch_np, heatmap = entry
        img     = patch_np[CENTER_CH]
        overlay = overlay_heatmap(img, heatmap)

        axes[row, 0].imshow(img, cmap='gray', vmin=0, vmax=1)
        axes[row, 0].axis('off')
        axes[row, 1].imshow(overlay)
        axes[row, 1].axis('off')

        if row == 0:
            axes[row, 0].set_title('Original (k center)', fontsize=22,
                                   fontweight='bold', pad=10)
            axes[row, 1].set_title('Grad-CAM Overlay', fontsize=22,
                                   fontweight='bold', pad=10)

        axes[row, 0].text(
            -0.18, 0.5, _short_label[row_key],
            transform=axes[row, 0].transAxes,
            fontsize=18, fontweight='bold',
            va='center', ha='right', rotation=90
        )

    actual = 'Malignant' if label == 1 else 'Benign'
    diam   = meta.get('diameter_max_mm')
    title  = (f'{case} — Sample #{rank+1} | {key[0]} n{key[1]} | Actual: {actual}'
              + (f' | {diam:.1f} mm' if diam is not None else ''))
    fig.suptitle(title, fontsize=24, fontweight='bold')
    fig.legend(handles=LEGEND_PATCHES, loc='lower center', ncol=2, fontsize=20,
               bbox_to_anchor=(0.5, 0.0))

    # tight_layout 먼저 → axes 위치 확정 후 브라켓 그리기
    fig.tight_layout(rect=[0.10, 0.04, 1, 0.94])
    draw_dual_bracket(fig, axes[:, 0], n_rows, label_x_fig=0.01)

    fname = f'{case}_{rank+1:02d}_{key[0]}_n{key[1]}_comparison.png'
    save_path = out_dir / fname
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return save_path


def save_case_grid_page(case: str, sample_keys: list, results_by_key: dict,
                        meta_map: dict, out_dir: Path) -> Path:
    """
    케이스 전체 grid:
    행 = 4개 (ConvNeXt / DualConvNeXt small / DualConvNeXt large / GDN)
    열 = N개 샘플
    k(center) Grad-CAM 오버레이만 표시.

    dual 두 행은 figure 왼쪽에 세로 브라켓 + "DualConvNeXt" 레이블로 묶어 표시.
    results_by_key: {key: {row_key: (patch_np, heatmap)}}
    """
    # dual 행의 짧은 레이블
    _short_label = {
        'convnext'           : 'ConvNeXt',
        'dual_convnext_small': '(small=32px)',
        'dual_convnext_large': '(large=96px)',
        'gdn'                : 'GDN',
    }

    n_cols = len(sample_keys)
    n_rows = len(ROW_ORDER)
    if n_cols == 0:
        return None

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.8 * n_cols, 4.8 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    if n_cols == 1:
        axes = axes.reshape(-1, 1)

    for row, row_key in enumerate(ROW_ORDER):
        for col, key in enumerate(sample_keys):
            ax    = axes[row, col]
            entry = results_by_key[key].get(row_key)
            if entry is None:
                ax.axis('off')
                continue

            patch_np, heatmap = entry
            ax.imshow(overlay_heatmap(patch_np[CENTER_CH], heatmap))
            ax.axis('off')

            if row == 0:
                meta  = meta_map.get(key, {})
                diam  = meta.get('diameter_max_mm')
                title = f'#{col+1}'
                if diam is not None:
                    title += f'\n{diam:.1f} mm'
                ax.set_title(title, fontsize=22, fontweight='bold', pad=8)

        axes[row, 0].text(
            -0.25, 0.5, _short_label[row_key],
            transform=axes[row, 0].transAxes,
            fontsize=20, fontweight='bold',
            va='center', ha='right', rotation=90
        )

    fig.suptitle(
        f'{case} — all 3 models correct (k=center slice) | '
        f'n={n_cols} samples, sorted by diameter_max_mm',
        fontsize=26, fontweight='bold'
    )
    fig.legend(handles=LEGEND_PATCHES, loc='lower center', ncol=2, fontsize=20,
               bbox_to_anchor=(0.5, 0.0))

    # tight_layout 먼저 → axes 위치 확정 후 브라켓 그리기
    fig.tight_layout(rect=[0.09, 0.04, 1, 0.93])
    draw_dual_bracket(fig, axes[:, 0], n_rows, label_x_fig=0.01)

    fname = f'{case}_grid.png'
    save_path = out_dir / fname
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return save_path


def save_individual_png(row_key: str, patch_np: np.ndarray, heatmap: np.ndarray,
                        label: int, case: str, rank: int, key: tuple,
                        out_dir: Path) -> Path:
    """
    행 1개, 결절 1개의 개별 Grad-CAM PNG.
    k-1 / k / k+1 3채널 모두 포함.
    """
    slice_names = ['k-1 (prev)', 'k (center)', 'k+1 (next)']
    n_channels  = patch_np.shape[0]

    fig, axes = plt.subplots(2, n_channels, figsize=(6 * n_channels, 9))
    for ch in range(n_channels):
        img = patch_np[ch]

        axes[0, ch].imshow(img, cmap='gray', vmin=0, vmax=1)
        axes[0, ch].set_title(slice_names[ch], fontsize=20, fontweight='bold', pad=8)
        axes[0, ch].axis('off')

        axes[1, ch].imshow(overlay_heatmap(img, heatmap))
        axes[1, ch].set_title('Grad-CAM', fontsize=20, fontweight='bold', pad=8)
        axes[1, ch].axis('off')

    actual = 'Malignant' if label == 1 else 'Benign'
    fig.suptitle(
        f'{ROW_DISPLAY_NAME_INLINE[row_key]} | {case} (Sample {rank+1}) | '
        f'{key[0]} n{key[1]} | Actual: {actual}',
        fontsize=22, fontweight='bold'
    )
    fig.legend(handles=LEGEND_PATCHES, loc='lower center', ncol=2, fontsize=18,
               bbox_to_anchor=(0.5, 0.0))

    fname     = f'{case}_{rank+1:02d}_{key[0]}_n{key[1]}_{row_key}.png'
    save_path = out_dir / fname
    fig.tight_layout(rect=[0, 0.04, 1, 0.94])
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return save_path


# ─────────────────────────────────────────────────────────────────────────────
# 섹션 7. 메인
# ─────────────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    device = get_device()

    model_dirs = {
        'gdn'          : Path(args.gdn_dir),
        'convnext'     : Path(args.convnext_dir),
        'dual_convnext': Path(args.dual_dir),
    }
    configs = {name: load_exp_config(d) for name, d in model_dirs.items()}

    # ── 1. 모델별 DataLoader + test 결과 수집 ──────────────────────────────
    print('[1/5] 모델 로드 + test set 결절 단위 예측 수집...')
    nodule_results     = {}
    center_sample_maps = {}
    models             = {}

    for name, exp_dir in model_dirs.items():
        cfg     = configs[name]
        is_dual = (name == 'dual_convnext')
        model   = load_model_baseline(name, exp_dir, device)
        models[name] = model

        if is_dual:
            parts      = str(cfg['crop_size']).split('+')
            small_crop = int(parts[0])
            _, _, test_loader, _ = get_dual_dataloaders(
                crop_size_small=small_crop, crop_size_large=96,
                batch_size=cfg['batch_size'], num_workers=cfg['num_workers']
            )
        else:
            _, _, test_loader, _ = get_dataloaders(
                crop_size=int(cfg['crop_size']),
                batch_size=cfg['batch_size'], num_workers=cfg['num_workers']
            )

        nodule_results[name]     = collect_test_results_baseline(
            model, test_loader, device, is_dual=is_dual)
        center_sample_maps[name] = build_center_sample_map(test_loader, is_dual=is_dual)
        print(f'  - {name}: {len(nodule_results[name]["labels"])}개 결절 예측 완료')

    # ── 2. 세 모델 공통 결절 키 ────────────────────────────────────────────
    common_keys = (set(nodule_results['gdn']['labels'])
                   & set(nodule_results['convnext']['labels'])
                   & set(nodule_results['dual_convnext']['labels']))
    print(f'[2/5] 세 모델 공통 결절: {len(common_keys)}개')

    # ── 3. 케이스별 세 모델이 모두 동일하게 맞춘 결절 ─────────────────────
    cases_to_export = ['TP', 'TN']
    selected = {c: [] for c in cases_to_export}

    for key in sorted(common_keys):
        case_per_model = {name: get_case(nodule_results[name], key) for name in MODEL_KEYS}
        cases          = set(case_per_model.values())
        if len(cases) == 1 and next(iter(cases)) in cases_to_export:
            selected[next(iter(cases))].append(key)

    for c in cases_to_export:
        print(f'  - 세 모델 모두 {c}: {len(selected[c])}개')

    # ── 4. 객관적 정렬 후 상위 N개 채택 ──────────────────────────────────
    base_crop = int(str(configs['gdn']['crop_size']))
    meta_map  = load_nodule_meta(base_crop)

    def sort_key(key):
        m   = meta_map.get(key, {})
        val = m.get(args.sort_by)
        return (val is None, val if val is not None else 0.0)

    for c in cases_to_export:
        selected[c].sort(key=sort_key)
        selected[c] = selected[c][:args.n_per_case]

    # ── 5. 출력 폴더 준비 ────────────────────────────────────────────────
    output_dir     = Path(args.output_dir)
    comparison_dir = output_dir / 'comparison'
    individual_dir = output_dir / 'individual'
    comparison_dir.mkdir(parents=True, exist_ok=True)
    individual_dir.mkdir(parents=True, exist_ok=True)

    selection_log = {
        'sort_by'         : args.sort_by,
        'n_per_case'      : args.n_per_case,
        'model_dirs'      : {k: str(v) for k, v in model_dirs.items()},
        'common_n_nodules': len(common_keys),
        'selected'        : {},
    }

    # ── 6. Grad-CAM 생성 + 시각화 저장 ──────────────────────────────────
    print('[3/5] Grad-CAM 생성 + 시각화 저장...')
    for case in cases_to_export:
        # results_by_key: {key: {row_key: (patch_np, heatmap)}}
        results_by_key = {}

        for rank, key in enumerate(selected[case]):
            label       = nodule_results['gdn']['labels'][key]
            row_results = {}   # {row_key: (patch_np, heatmap)}

            for row_key in ROW_ORDER:
                # row_key → 실제 모델 키 매핑
                model_key = ('dual_convnext'
                             if row_key in ('dual_convnext_small', 'dual_convnext_large')
                             else row_key)
                sample = center_sample_maps[model_key].get(key)
                if sample is None:
                    print(f'  [WARN] {row_key}에 {key} 샘플 없음, 건너뜀')
                    continue

                patch_np, heatmap, _ = compute_row_results(
                    row_key, models[model_key], sample, device)
                row_results[row_key] = (patch_np, heatmap)

                # 개별 PNG (k-1/k/k+1 모두)
                save_individual_png(row_key, patch_np, heatmap, label,
                                    case, rank, key, individual_dir)

            if len(row_results) < len(ROW_ORDER):
                missing = set(ROW_ORDER) - set(row_results)
                print(f'  [WARN] {key}: 누락된 행 {missing}')

            results_by_key[key] = row_results

            # 샘플 1개 × 4행 비교 페이지
            meta = meta_map.get(key, {})
            save_comparison_page(row_results, label, case, rank, key,
                                 meta, comparison_dir)

        # 케이스 전체 grid (4행 × N열)
        grid_path = save_case_grid_page(case, selected[case], results_by_key,
                                        meta_map, comparison_dir)
        if grid_path:
            print(f'  [{case}] grid 저장: {grid_path.name}')

        selection_log['selected'][case] = [
            {'subject_id': k[0], 'nodule_idx': k[1], **meta_map.get(k, {})}
            for k in selected[case]
        ]

    with open(output_dir / 'selection_log.json', 'w', encoding='utf-8') as f:
        json.dump(selection_log, f, ensure_ascii=False, indent=2)

    print(f'[4/5] selection_log.json 저장: {output_dir / "selection_log.json"}')
    print(f'[5/5] 완료.')
    print(f'  비교 페이지 (샘플별): {comparison_dir}')
    print(f'  케이스 grid         : {comparison_dir}/TP_grid.png, TN_grid.png')
    print(f'  개별 PNG            : {individual_dir}')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='ConvNeXt/DualConvNeXt(small+large)/GDN 공통 결절 Grad-CAM 비교')
    parser.add_argument('--gdn_dir',      type=str, required=True)
    parser.add_argument('--convnext_dir', type=str, required=True)
    parser.add_argument('--dual_dir',     type=str, required=True)
    parser.add_argument('--n_per_case',   type=int, default=5)
    parser.add_argument('--sort_by',      type=str, default='diameter_max_mm',
                        choices=['diameter_max_mm', 'volume_mm3'])
    parser.add_argument('--output_dir',   type=str, default='outputs/gradcam_comparison')
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())
