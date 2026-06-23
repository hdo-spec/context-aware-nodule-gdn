# src/utils/visualize.py
#
# ─── 역할 ────────────────────────────────────────────────────────────────────
#   evaluate.py에서 호출하는 시각화 함수 모음.
#   ROC curve, 혼동 행렬, 학습 곡선, Grad-CAM을 PNG로 저장.
#
# ─── Grad-CAM 설계 ───────────────────────────────────────────────────────────
#   [활성] 방법 1: 채널별 개별 시각화
#     [k-1, k, k+1] 3채널을 나란히 3장 출력
#     어느 슬라이스에서 모델이 반응했는지 볼 수 있음
#
#   [주석] 방법 2: 채널 평균 시각화
#     3채널 평균 이미지 1장 + heatmap 1장
#     슬라이스별 차이가 사라지지만 단순함

import csv
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use('Agg')   # GUI 없는 서버에서 PNG 저장용 백엔드
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_curve, auc, confusion_matrix

# ─────────────────────────────────────────────────────────────────────────────
# 섹션 1. ROC Curve
# ─────────────────────────────────────────────────────────────────────────────

def plot_roc_curve(labels: list, probs: list, save_path: Path, title: str = '') -> None:
    """
    ROC curve를 그리고 PNG로 저장.

    ROC curve: threshold를 0~1로 변화시키면서 FPR(x축) vs TPR(y축) 그래프.
    AUC: ROC curve 아래 면적. 1.0에 가까울수록 좋음. 목표: ≥ 0.90.
    대각선(점선): AUC=0.5, 랜덤 분류 수준.

    Args:
        labels   : 정답 레이블 리스트 (0 또는 1)
        probs    : sigmoid 확률값 리스트 ([0, 1])
        save_path: 저장 경로 (.png)
        title    : 그래프 제목
    """
    fpr, tpr, _ = roc_curve(labels, probs)       # FPR, TPR, threshold 계산
    roc_auc     = auc(fpr, tpr)                  # AUC 계산

    fig, ax = plt.subplots(figsize=(7, 6))

    # ROC curve
    ax.plot(fpr, tpr, color='steelblue', lw=2,
            label=f'ROC curve (AUC = {roc_auc:.4f})')

    # 대각선: 랜덤 분류 기준선
    ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random (AUC = 0.50)')

    # 목표 AUC 기준선 (0.90)
    ax.axhline(y=0.90, color='red', lw=1, linestyle=':', alpha=0.7, label='Target AUC = 0.90')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    ax.set_ylabel('True Positive Rate (Sensitivity)',      fontsize=12)
    ax.set_title(title or 'ROC Curve', fontsize=13)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[VIZ] ROC curve 저장: {save_path.name}')


# ─────────────────────────────────────────────────────────────────────────────
# 섹션 2. 혼동 행렬 (Confusion Matrix)
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(labels: list, preds: list, save_path: Path, title: str = '') -> None:
    """
    혼동 행렬을 그리고 PNG로 저장.

    혼동 행렬 구조:
               예측 Benign  예측 Malignant
    실제 Benign    TN           FP
    실제 Malignant FN           TP

    의료 분류에서 FN(악성을 놓침)이 FP(양성을 오탐)보다 위험.
    sensitivity(=TPR=TP/(TP+FN))가 높은지 반드시 확인.
    """
    cm = confusion_matrix(labels, preds)         # [[TN, FP], [FN, TP]]

    fig, ax = plt.subplots(figsize=(5, 4))

    # 색상: 대각선(TN, TP)은 파란계열, 나머지(FP, FN)는 빨간계열
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar(im, ax=ax)

    classes = ['Benign (0)', 'Malignant (1)']
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks);  ax.set_xticklabels(classes, rotation=15)
    ax.set_yticks(tick_marks);  ax.set_yticklabels(classes)

    # 셀 안에 숫자 표시
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]),
                    ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black',
                    fontsize=14, fontweight='bold')

    ax.set_ylabel('Actual Label',    fontsize=11)
    ax.set_xlabel('Predicted Label', fontsize=11)
    ax.set_title(title or 'Confusion Matrix', fontsize=12)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[VIZ] 혼동 행렬 저장: {save_path.name}')


# ─────────────────────────────────────────────────────────────────────────────
# 섹션 3. 학습 곡선 (Learning Curve)
# ─────────────────────────────────────────────────────────────────────────────

def plot_learning_curve(history_csv: Path, save_path: Path, title: str = '') -> None:
    """
    history.csv를 읽어서 train/val 학습 곡선을 그리고 PNG로 저장.

    2개 서브플롯:
      상단: train_loss vs val_loss (과적합 시 val_loss 증가)
      하단: train_auc  vs val_auc  (목표: val_auc ≥ 0.90)

    best_auc_flag=1인 epoch에 수직선 표시 → best checkpoint 시점 확인.

    과적합 판단:
      train_loss 내려가는데 val_loss 올라가면 과적합.
      train_auc >> val_auc 이면 과적합.
    """
    if not history_csv.exists():
        print(f'[WARN] history.csv 없음: {history_csv}')
        return

    # CSV 읽기
    epochs, train_loss, val_loss = [], [], []
    train_auc, val_auc = [], []
    best_epochs = []

    with open(history_csv, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            epochs.append(int(row['epoch']))
            train_loss.append(float(row['train_loss']))
            val_loss.append(float(row['val_loss']))
            train_auc.append(float(row['train_auc']))
            val_auc.append(float(row['val_auc']))
            if int(row['best_auc_flag']) == 1:
                best_epochs.append(int(row['epoch']))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    # ── Loss 곡선 ─────────────────────────────────────────────
    ax1.plot(epochs, train_loss, label='Train Loss', color='steelblue', lw=2)
    ax1.plot(epochs, val_loss,   label='Val Loss',   color='orange',    lw=2)
    for be in best_epochs:
        ax1.axvline(x=be, color='green', lw=1, linestyle='--', alpha=0.5)
    ax1.set_ylabel('Loss', fontsize=11)
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)
    ax1.set_title(title or 'Learning Curve', fontsize=13)

    # ── AUC 곡선 ──────────────────────────────────────────────
    ax2.plot(epochs, train_auc, label='Train AUC', color='steelblue', lw=2)
    ax2.plot(epochs, val_auc,   label='Val AUC',   color='orange',    lw=2)
    ax2.axhline(y=0.90, color='red', lw=1, linestyle=':', alpha=0.7, label='Target 0.90')
    for be in best_epochs:
        ax2.axvline(x=be, color='green', lw=1, linestyle='--', alpha=0.5,
                    label='Best checkpoint' if be == best_epochs[0] else '')
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('AUC',   fontsize=11)
    ax2.set_ylim([0.4, 1.05])
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[VIZ] 학습 곡선 저장: {save_path.name}')


# ─────────────────────────────────────────────────────────────────────────────
# 섹션 4. Grad-CAM
# ─────────────────────────────────────────────────────────────────────────────

class GradCAM:
    """
    Grad-CAM 구현.

    원리:
      1. 타겟 레이어의 feature map (activation)을 forward hook으로 저장
      2. 타겟 클래스 logit에 대한 feature map의 gradient를 backward hook으로 저장
      3. gradient를 채널별로 global average pooling → 채널 중요도 alpha 계산
      4. alpha × feature map을 채널 방향으로 합산 → heatmap
      5. ReLU: 양의 기여만 남김 (음의 값은 목표 클래스에 반하는 부위)
      6. 입력 크기로 resize → 원본 이미지에 오버레이

    Args:
        model      : 평가 모드 모델
        target_layer: Grad-CAM을 적용할 레이어 (nn.Module)
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model        = model
        self.activations  = None   # forward hook이 저장하는 feature map
        self.gradients    = None   # backward hook이 저장하는 gradient
        self._relu_states = []

        # forward hook에서 activation과 tensor gradient hook을 함께 등록한다.
        # module full backward hook은 GDN의 inplace ReLU와 충돌할 수 있다.
        self._disable_inplace_relu()
        self._fwd_hook = target_layer.register_forward_hook(self._save_activation)
        self._tensor_hook = None

    def _disable_inplace_relu(self):
        """Grad-CAM backward 중 autograd hook과 충돌하지 않도록 inplace ReLU를 임시 해제."""
        for module in self.model.modules():
            if isinstance(module, nn.ReLU) and module.inplace:
                self._relu_states.append((module, module.inplace))
                module.inplace = False

    def _save_activation(self, module, input, output):
        """forward pass 시 feature map 저장."""
        if self._tensor_hook is not None:
            self._tensor_hook.remove()
            self._tensor_hook = None
        self.activations = output.detach().clone()
        self._tensor_hook = output.register_hook(self._save_gradient)

    def _save_gradient(self, grad):
        """backward pass 시 feature map gradient 저장."""
        self.gradients = grad.detach().clone()

    def generate(self, *model_inputs: torch.Tensor, target_class: int = 1) -> np.ndarray:
        """
        Grad-CAM heatmap 생성.

        Args:
            model_inputs: 모델 입력. 단일 모델은 (input,), DualConvNeXt는 (small, large).
            target_class: 시각화할 클래스 (1=악성)

        Returns:
            np.ndarray: heatmap, shape (H, W), 범위 [0, 1]
        """
        self.model.zero_grad()
        self.activations = None
        self.gradients = None

        # forward
        logit = self.model(*model_inputs)          # (1, 1)

        # backward: target_class=1(악성) logit에 대한 gradient 계산
        logit[0, 0].backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError('Grad-CAM activation/gradient가 저장되지 않았습니다.')

        # gradient global average pooling → 채널 중요도 alpha
        alpha = self.gradients.mean(dim=[2, 3], keepdim=True)   # (1, C, 1, 1)

        # 가중합 heatmap
        heatmap = (alpha * self.activations).sum(dim=1, keepdim=True)   # (1, 1, h, w)
        heatmap = torch.relu(heatmap)              # 음의 기여 제거
        heatmap = heatmap.squeeze().cpu().numpy()  # (h, w)

        # [0, 1] 정규화
        if heatmap.max() > heatmap.min():
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        else:
            heatmap = np.zeros_like(heatmap)

        return heatmap

    def remove_hooks(self):
        """메모리 누수 방지: hook 제거."""
        self._fwd_hook.remove()
        if self._tensor_hook is not None:
            self._tensor_hook.remove()
            self._tensor_hook = None
        for module, inplace in self._relu_states:
            module.inplace = inplace
        self._relu_states = []


def _get_target_layer(model: nn.Module, model_name: str) -> nn.Module:
    """
    모델별 Grad-CAM 타겟 레이어 반환.

    선택 기준: 마지막 conv 레이어 (공간 정보가 남아있는 가장 깊은 레이어).
    너무 앞 레이어는 저수준 특징(엣지 등), 마지막 레이어는 고수준 의미(결절 패턴).
    """
    if model_name == 'gdn':
        return model.gd5                # GDN 마지막 GDLayer 출력
    elif model_name == 'convnext':
        return model.stage4[-1].dwconv  # ConvNeXt 마지막 블록의 depthwise conv
    elif model_name == 'dual_convnext':
        # DualConvNeXt: large branch 기준 (96px 패치가 맥락 정보 풍부)
        return model.large_branch.stage4[-1].dwconv
    else:
        raise ValueError(f'알 수 없는 모델: {model_name}')


def _overlay_heatmap(image: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    """
    원본 이미지에 heatmap을 오버레이한 RGB 이미지 반환.

    Args:
        image  : (H, W) 그레이스케일 이미지, 범위 [0, 1]
        heatmap: (H, W) Grad-CAM heatmap, 범위 [0, 1]

    Returns:
        np.ndarray: (H, W, 3) RGB 이미지
    """
    from matplotlib.cm import jet   # jet colormap: 파랑(낮음) → 빨강(높음)

    # heatmap을 입력 크기로 resize (feature map은 보통 입력보다 작음)
    if heatmap.shape != image.shape:
        from PIL import Image as PILImage
        heatmap_resized = np.array(
            PILImage.fromarray((heatmap * 255).astype(np.uint8)).resize(
                (image.shape[1], image.shape[0]), PILImage.BILINEAR
            )
        ) / 255.0
    else:
        heatmap_resized = heatmap

    # 원본 이미지를 RGB로 변환
    image_rgb  = np.stack([image, image, image], axis=-1)   # (H, W, 3) 그레이

    # heatmap을 컬러맵 적용 후 RGB로 변환
    heatmap_rgb = jet(heatmap_resized)[:, :, :3]             # (H, W, 3), RGBA→RGB

    # 알파 블렌딩: 원본 40% + heatmap 60%
    overlay = 0.4 * image_rgb + 0.6 * heatmap_rgb
    return np.clip(overlay, 0, 1)


def save_gradcam(model: nn.Module, model_name: str, test_loader, selected_indices: Dict[str, List[tuple]], save_dir: Path, device: torch.device, is_dual: bool = False) -> None:
    """
    선택된 결절에 대해 Grad-CAM을 생성하고 케이스별 PNG로 저장.

    [활성] 방법 1: 3채널 개별 시각화
      한 샘플당 [k-1 슬라이스, k 슬라이스, k+1 슬라이스] 3장 나란히 출력.

    [주석] 방법 2: 채널 평균 시각화

    Args:
        model           : 평가 모드 모델
        model_name      : 'gdn', 'convnext', 'dual_convnext'
        test_loader     : test DataLoader
        selected_indices: {'TP': [(subject_id, nodule_idx), ...], ...}
                          결절 단위 key 리스트 (기존 int 인덱스에서 변경)
        save_dir        : gradcam/ 폴더
        device          : 'cuda' 또는 'cpu'
        is_dual         : DualConvNeXt 여부
    """
    target_layer = _get_target_layer(model, model_name)
    gradcam      = GradCAM(model, target_layer)

    # test_loader에서 전체 샘플 수집
    # key: (subject_id, nodule_idx) → 결절의 대표 슬라이스 1개 (중심 z)
    # 결절당 여러 슬라이스 중 중심 슬라이스를 Grad-CAM에 사용
    from collections import defaultdict
    nodule_center_sample = {}   # {(subject_id, nodule_idx): sample}

    for batch in test_loader:
        if is_dual:
            patch_small, patch_large, labels, subject_ids, nodule_idxs, z_idxs = batch
            for i in range(len(labels)):
                key = (subject_ids[i], nodule_idxs[i])
                # 이미 저장된 슬라이스보다 중심에 가까운 슬라이스 우선 (단순히 처음 등장한 것 사용)
                if key not in nodule_center_sample:
                    nodule_center_sample[key] = (
                        patch_small[i], patch_large[i], int(labels[i])
                    )
        else:
            patches, labels, subject_ids, nodule_idxs, z_idxs = batch
            for i in range(len(labels)):
                key = (subject_ids[i], nodule_idxs[i])
                if key not in nodule_center_sample:
                    nodule_center_sample[key] = (patches[i], int(labels[i]))

    # 케이스별 Grad-CAM 생성
    for case_name, nodule_keys in selected_indices.items():
        if not nodule_keys:
            print(f'[Grad-CAM] {case_name}: 샘플 없음, 건너뜀')
            continue

        for rank, key in enumerate(nodule_keys):
            if key not in nodule_center_sample:
                print(f'[Grad-CAM] {case_name} key={key} 샘플 없음, 건너뜀')
                continue

            sample = nodule_center_sample[key]

            if is_dual:
                patch_small, patch_large, label = sample
                input_small = patch_small.unsqueeze(0).to(device)
                input_large = patch_large.unsqueeze(0).to(device)
                input_small.requires_grad_(True)
                input_large.requires_grad_(True)
                patch_np     = patch_large.numpy()
            else:
                patch, label = sample
                input_tensor = patch.unsqueeze(0).to(device)
                input_tensor.requires_grad_(True)
                patch_np     = patch.numpy()

            # Grad-CAM heatmap 생성
            try:
                if is_dual:
                    heatmap = gradcam.generate(input_small, input_large, target_class=1)
                else:
                    heatmap = gradcam.generate(input_tensor, target_class=1)
            except RuntimeError as e:
                print(f'[Grad-CAM][WARN] {case_name} key={key} 생성 실패: {e}')
                continue

            # ── [활성] 방법 1: 3채널 개별 시각화 ─────────────────────
            # 슬라이스명: k-1, k (중심), k+1
            slice_names = ['k-1 (prev)', 'k (center)', 'k+1 (next)']
            n_channels  = patch_np.shape[0]   # 3

            fig, axes = plt.subplots(2, n_channels, figsize=(5 * n_channels, 8))

            for ch in range(n_channels):
                img = patch_np[ch]   # (H, W), [0, 1]

                # 상단: 원본 이미지
                axes[0, ch].imshow(img, cmap='gray', vmin=0, vmax=1)
                axes[0, ch].set_title(slice_names[ch], fontsize=9)
                axes[0, ch].axis('off')

                # 하단: 원본 + heatmap 오버레이
                overlay = _overlay_heatmap(img, heatmap)
                axes[1, ch].imshow(overlay)
                axes[1, ch].set_title(f'Grad-CAM', fontsize=9)
                axes[1, ch].axis('off')

            # 레이블/예측 정보 표시
            actual    = 'Malignant' if label == 1 else 'Benign'
            predicted = case_name.split('_')[0]   # 'TP', 'TN', 'FP', 'FN'
            fig.suptitle(
                f'{case_name} (Sample {rank+1}) | Actual: {actual}',
                fontsize=12, fontweight='bold'
            )

            # 범례: 빨강=높은 attention, 파랑=낮은 attention
            legend = [
                mpatches.Patch(color='red',  label='High attention'),
                mpatches.Patch(color='blue', label='Low attention'),
            ]
            fig.legend(handles=legend, loc='lower center', ncol=2, fontsize=9)

            save_path = save_dir / f'gradcam_{case_name}_{rank+1:02d}.png'
            fig.tight_layout()
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)

        print(f'[Grad-CAM] {case_name}: {len(nodule_keys)}장 저장')

    # ── [주석] 방법 2: 채널 평균 시각화 ─────────────────────────
    # 팀 회의 후 아래 주석 해제하여 활성화 가능.
    # 방법 1과 동시에 저장하거나 대체하여 사용.
    #
    # for case_name, indices in selected_indices.items():
    #     for rank, sample_idx in enumerate(indices):
    #         patch_np = ...   # (3, H, W)
    #         img_mean = patch_np.mean(axis=0)   # (H, W): 3채널 평균
    #         heatmap  = gradcam.generate(input_tensor, target_class=1)
    #         overlay  = _overlay_heatmap(img_mean, heatmap)
    #
    #         fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    #         ax1.imshow(img_mean, cmap='gray'); ax1.set_title('평균 이미지')
    #         ax2.imshow(overlay);               ax2.set_title('Grad-CAM')
    #         ax1.axis('off'); ax2.axis('off')
    #
    #         save_path = save_dir / f'gradcam_avg_{case_name}_{rank+1:02d}.png'
    #         fig.savefig(save_path, dpi=150, bbox_inches='tight')
    #         plt.close(fig)

    # hook 제거 (메모리 누수 방지)
    gradcam.remove_hooks()
    print(f'[Grad-CAM] 전체 저장 완료: {save_dir}')
