# src/training/engine.py
#
# ─── 변경 사항 ────────────────────────────────────────────────────────────────
#   train_one_epoch: 슬라이스 단위 학습 (변경 없음)
#   validate_one_epoch: 슬라이스별 prob 수집 후 결절별 평균 → AUC 계산
#
#   결절별 평균 집계:
#     배치에서 (prob, label, subject_id, nodule_idx) 수집
#     (subject_id, nodule_idx) 키로 그룹핑
#     슬라이스별 prob 평균 → 결절 단위 확률
#     결절 단위 AUC 계산

import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
from sklearn.metrics import roc_auc_score


def _append_nodule_result(nodule_results: dict, key: tuple, prob: float, label: int) -> None:
    data = nodule_results[key]
    if data['label'] is not None and data['label'] != label:
        raise ValueError(
            f'같은 결절 key에 서로 다른 label이 섞였습니다: '
            f'{key}, previous={data["label"]}, current={label}'
        )
    data['probs'].append(prob)
    data['label'] = label


def _compute_metrics(all_labels, all_probs, all_preds, total_loss, n_batches) -> dict:
    """공통 지표 계산."""
    avg_loss = total_loss / n_batches if n_batches > 0 else 0.0
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.0

    all_labels = np.array(all_labels)
    all_preds  = np.array(all_preds)
    accuracy    = float((all_preds == all_labels).mean())
    tp = float(((all_preds == 1) & (all_labels == 1)).sum())
    tn = float(((all_preds == 0) & (all_labels == 0)).sum())
    fp = float(((all_preds == 1) & (all_labels == 0)).sum())
    fn = float(((all_preds == 0) & (all_labels == 1)).sum())
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        'loss'       : round(avg_loss,    4),
        'auc'        : round(auc,         4),
        'accuracy'   : round(accuracy,    4),
        'sensitivity': round(sensitivity, 4),
        'specificity': round(specificity, 4),
    }


def train_one_epoch(model: nn.Module, loader, criterion, optimizer,
                    device: torch.device, is_dual: bool = False) -> dict:
    """
    1 epoch 학습.

    슬라이스 단위로 독립 학습.
    train AUC도 슬라이스 단위로 계산 (과적합 모니터링용).
    """
    model.train()

    total_loss = 0.0
    all_labels, all_probs, all_preds = [], [], []

    for batch in loader:
        if is_dual:
            patch_small, patch_large, labels, _, _, _ = batch
            patch_small = patch_small.to(device)
            patch_large = patch_large.to(device)
        else:
            patches, labels, _, _, _ = batch
            patches = patches.to(device)

        labels = labels.to(device).float().unsqueeze(1)

        optimizer.zero_grad()
        logits = model(patch_small, patch_large) if is_dual else model(patches)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        probs = torch.sigmoid(logits).detach().cpu().numpy().flatten()
        preds = (probs >= 0.5).astype(int)
        lbls  = labels.detach().cpu().numpy().flatten().astype(int)

        all_probs.extend(probs.tolist())
        all_preds.extend(preds.tolist())
        all_labels.extend(lbls.tolist())

    return _compute_metrics(all_labels, all_probs, all_preds, total_loss, len(loader))


@torch.no_grad()
def validate_one_epoch(model: nn.Module, loader, criterion,
                       device: torch.device, is_dual: bool = False) -> tuple[dict, dict]:
    """
    1 epoch 검증/테스트.

    슬라이스별 prob을 수집한 후 (subject_id, nodule_idx) 기준으로 그룹핑하여
    결절별 평균 확률로 AUC 계산.

    왜 결절별 평균인가:
      라벨이 결절 단위이고, 과제가 결절 전체의 morphology 판단.
      슬라이스 단위 AUC는 같은 결절의 슬라이스가 독립 샘플처럼 취급되어
      실제 결절 분류 성능을 과대평가할 수 있음.
    """
    model.eval()

    total_loss = 0.0
    alpha_storage = {
        'gd1': [],
        'gd2': [],
        'gd3': [],
        'gd4': [],
        'gd5': [],
    }

    # 결절별 prob 누적: {(subject_id, nodule_idx): {'probs': [], 'label': int}}
    nodule_results = defaultdict(lambda: {'probs': [], 'label': None})

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

        # 모델 forward 이후 alpha 수집
        for layer_name in alpha_storage:
            if hasattr(model, layer_name):
                layer = getattr(model, layer_name)

                if layer.last_alpha is not None:
                    alpha_storage[layer_name].append(
                        layer.last_alpha.mean().item()
                    )

        labels_dev = labels.to(device).float().unsqueeze(1)
        loss = criterion(logits, labels_dev)
        total_loss += loss.item()

        probs = torch.sigmoid(logits).cpu().numpy().flatten()
        lbls  = labels.numpy().flatten().astype(int)

        # 결절별 prob 누적
        for j in range(len(lbls)):
            key = (subject_ids[j], nodule_idxs[j])   # (str, str)
            _append_nodule_result(
                nodule_results,
                key,
                prob=float(probs[j]),
                label=int(lbls[j]),
            )

    # 결절별 평균 확률 계산
    nodule_probs  = []
    nodule_labels = []
    nodule_preds  = []

    for key, data in nodule_results.items():
        mean_prob = float(np.mean(data['probs']))   # 슬라이스별 prob 평균
        nodule_probs.append(mean_prob)
        nodule_labels.append(data['label'])
        nodule_preds.append(1 if mean_prob >= 0.5 else 0)

    avg_loss = total_loss / len(loader) if len(loader) > 0 else 0.0

    # AUC는 결절 단위로 계산
    try:
        auc = roc_auc_score(nodule_labels, nodule_probs)
    except ValueError:
        auc = 0.0

    nodule_labels = np.array(nodule_labels)
    nodule_preds  = np.array(nodule_preds)
    accuracy    = float((nodule_preds == nodule_labels).mean())
    tp = float(((nodule_preds == 1) & (nodule_labels == 1)).sum())
    tn = float(((nodule_preds == 0) & (nodule_labels == 0)).sum())
    fp = float(((nodule_preds == 1) & (nodule_labels == 0)).sum())
    fn = float(((nodule_preds == 0) & (nodule_labels == 1)).sum())
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    alpha_means = {
        k: round(float(np.mean(v)), 4)
        for k, v in alpha_storage.items()
        if len(v) > 0
    }
    metrics = {
        'loss'          : round(avg_loss,    4),
        'auc'           : round(auc,         4),
        'accuracy'      : round(accuracy,    4),
        'sensitivity'   : round(sensitivity, 4),
        'specificity'   : round(specificity, 4),
        'n_nodules'     : len(nodule_results),   # 평가된 결절 수 (디버깅용)
    }

    return metrics, alpha_means
