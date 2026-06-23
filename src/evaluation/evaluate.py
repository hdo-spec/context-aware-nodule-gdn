# src/evaluation/evaluate.py
#
# ─── 변경 사항 ────────────────────────────────────────────────────────────────
#   collect_test_results: 슬라이스별 결과 수집 후 결절별 평균 집계
#   subgroup_auc: 결절 단위로 계산
#   result.json: 결절 단위 평가 지표 저장

import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, roc_curve

from src.configs.config import PROCESSED_ROOT, OUTPUT_ROOT
from src.datasets.dataset import get_dataloaders, get_dual_dataloaders
from src.models.models import GDN, ConvNeXt, DualConvNeXt
from src.utils.utils import get_device
from src.utils.visualize import (plot_roc_curve, plot_confusion_matrix,
                                  plot_learning_curve, save_gradcam)

NPY_CACHE_ROOT = PROCESSED_ROOT / 'npy_cache'


def load_exp_config(exp_dir: Path) -> dict:
    config_path = exp_dir / 'config.json'
    if not config_path.exists():
        raise FileNotFoundError(f'config.json 없음: {config_path}')
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_model(model_name: str, exp_dir: Path, device: torch.device, config: dict) -> nn.Module:
    if model_name == 'gdn':
        model = GDN(
            in_ch=3,
            num_classes=1,
            use_cbam1=config.get('gdn_cbam1', False),
            use_cbam2=config.get('gdn_cbam2', False),
            use_cbam_final=config.get('gdn_cbam_final', False),
            use_gd4b=config.get('gdn_gd4b', False),
            use_gd6=config.get('gdn_gd6', False),)
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

# val set에서 최적 threshold 찾기
@torch.no_grad()
def find_optimal_threshold(model: nn.Module, val_loader, device: torch.device,
                           is_dual: bool = False) -> float:
    """val set에서 Youden index 기준 최적 threshold 탐색."""
    nodule_data = defaultdict(lambda: {'probs': [], 'label': None})

    for batch in val_loader:
        if is_dual:
            patch_small, patch_large, labels, subject_ids, nodule_idxs, z_idxs = batch
            patch_small = patch_small.to(device)
            patch_large = patch_large.to(device)
            logits = model(patch_small, patch_large)
        else:
            patches, labels, subject_ids, nodule_idxs, z_idxs = batch
            patches = patches.to(device)
            logits = model(patches)

        probs = torch.sigmoid(logits).cpu().numpy().flatten()
        lbls  = labels.numpy().flatten().astype(int)

        for j in range(len(lbls)):
            key = (subject_ids[j], nodule_idxs[j])
            nodule_data[key]['probs'].append(float(probs[j]))
            nodule_data[key]['label'] = int(lbls[j])

    val_probs  = [float(np.mean(v['probs'])) for v in nodule_data.values()]
    val_labels = [v['label'] for v in nodule_data.values()]

    fpr, tpr, thresholds = roc_curve(val_labels, val_probs)
    youden    = tpr - fpr
    best_idx  = int(np.argmax(youden))
    best_thr  = float(thresholds[best_idx])

    print(f'[Threshold] val 최적 threshold={best_thr:.4f} '
          f'(TPR={tpr[best_idx]:.4f}, FPR={fpr[best_idx]:.4f})')
    return best_thr

@torch.no_grad()
def collect_test_results(model: nn.Module, loader, device: torch.device,
                         criterion=None, is_dual: bool = False, threshold=0.5) -> dict:
    """
    test set 전체 순회 → 슬라이스별 결과 수집 → 결절별 평균 집계.

    결절별 평균 이유:
      라벨이 결절 단위. 슬라이스 단위 평가는 같은 결절 내 상관관계로
      실제 성능을 과대평가할 수 있음.

    Returns:
        dict:
          'nodule_probs'  : 결절별 평균 확률 리스트
          'nodule_labels' : 결절별 레이블 리스트
          'nodule_preds'  : 결절별 예측 리스트 (threshold=0.5)
          'nodule_keys'   : (subject_id, nodule_idx) 리스트
          'nodule_diameter': 결절별 diameter_max_mm (subgroup 분석용)
          'nodule_volume' : 결절별 volume_mm3 (subgroup 분석용)
    """
    model.eval()

    # 결절별 데이터 누적
    nodule_data = defaultdict(lambda: {
        'probs': [], 'label': None,
        'diameter_max_mm': '', 'volume_mm3': ''
    })

    # CSV에서 결절별 메타데이터 미리 로드 (diameter, volume)
    # → DataLoader가 반환하지 않으므로 CSV에서 직접 읽음
    csv_crop_size = None  # 아래에서 설정

    total_loss = 0.0
    n_batches = 0

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

        if criterion is not None:
            labels_dev = labels.to(device).float().unsqueeze(1)
            total_loss += criterion(logits, labels_dev).item()
            n_batches += 1

        probs = torch.sigmoid(logits).cpu().numpy().flatten()
        lbls  = labels.numpy().flatten().astype(int)

        for j in range(len(lbls)):
            key = (subject_ids[j], nodule_idxs[j])
            label = int(lbls[j])
            if nodule_data[key]['label'] is not None and nodule_data[key]['label'] != label:
                raise ValueError(
                    f'같은 결절 key에 서로 다른 label이 섞였습니다: '
                    f'{key}, previous={nodule_data[key]["label"]}, current={label}'
                )
            nodule_data[key]['probs'].append(float(probs[j]))
            nodule_data[key]['label'] = label

    # 결절별 평균 집계
    nodule_keys    = list(nodule_data.keys())
    nodule_probs   = [float(np.mean(nodule_data[k]['probs'])) for k in nodule_keys]
    nodule_labels  = [nodule_data[k]['label'] for k in nodule_keys]
    nodule_preds   = [1 if p >= threshold else 0 for p in nodule_probs]

    return {
        'nodule_probs'  : nodule_probs,
        'nodule_labels' : nodule_labels,
        'nodule_preds'  : nodule_preds,
        'nodule_keys'   : nodule_keys,
        'slice_loss'    : round(total_loss / n_batches, 4) if n_batches > 0 else None,
    }


def _get_size_group(volume_mm3: float) -> str:
    """Fleischner Society 기준 크기 그룹."""
    if volume_mm3 < 100.0:
        return 'small'
    elif volume_mm3 <= 250.0:
        return 'intermediate'
    else:
        return 'large'


def compute_subgroup_auc(nodule_results: dict, crop_size: int) -> dict:
    """
    결절별 volume_mm3로 subgroup 분류 후 AUC 계산.
    CSV에서 (subject_id, nodule_idx) 기준으로 volume_mm3 조회.
    """
    raw_csv = NPY_CACHE_ROOT / f'labels_raw_{crop_size}.csv'
    if not raw_csv.exists():
        return {}

    import csv as csv_module
    # (subject_id, nodule_idx) → volume_mm3 매핑
    nodule_volume = {}
    with open(raw_csv, 'r', encoding='utf-8') as f:
        for row in csv_module.DictReader(f):
            key = (row['subject_id'], row['nodule_idx'])
            if key not in nodule_volume:
                nodule_volume[key] = row.get('volume_mm3', '')

    groups = {
        'small'       : {'labels': [], 'probs': []},
        'intermediate': {'labels': [], 'probs': []},
        'large'       : {'labels': [], 'probs': []},
        'unknown'     : {'labels': [], 'probs': []},
    }

    for i, key in enumerate(nodule_results['nodule_keys']):
        vol_str = nodule_volume.get(key, '')
        label   = nodule_results['nodule_labels'][i]
        prob    = nodule_results['nodule_probs'][i]

        if not vol_str:
            groups['unknown']['labels'].append(label)
            groups['unknown']['probs'].append(prob)
            continue

        group = _get_size_group(float(vol_str))
        groups[group]['labels'].append(label)
        groups[group]['probs'].append(prob)

    subgroup_result = {}
    for g_name, data in groups.items():
        n = len(data['labels'])
        if n == 0:
            subgroup_result[g_name] = {'auc': None, 'n': 0, 'n_benign': 0, 'n_malignant': 0}
            continue
        n_b = data['labels'].count(0)
        n_m = data['labels'].count(1)
        try:
            auc = round(roc_auc_score(data['labels'], data['probs']), 4)
        except ValueError:
            auc = None
        subgroup_result[g_name] = {'auc': auc, 'n': n, 'n_benign': n_b, 'n_malignant': n_m}

    print(f'\n[Subgroup AUC] Fleischner Society 기준 (volume_mm3)')
    print(f'  {"그룹":12s} {"AUC":>6s}  {"n":>5s}  benign  malignant')
    print(f'  {"-"*50}')
    for g in ['small', 'intermediate', 'large', 'unknown']:
        d   = subgroup_result.get(g, {})
        auc = f'{d["auc"]:.4f}' if d.get('auc') is not None else '  N/A '
        print(f'  {g:12s} {auc:>6s}  {d.get("n",0):>5d}  '
              f'{d.get("n_benign",0):>6d}  {d.get("n_malignant",0):>9d}')

    return subgroup_result


def select_gradcam_samples(nodule_results: dict, n_per_case: int = 5) -> dict:
    """TP/TN/FP/FN 케이스별 결절 선택 (결절 단위)."""
    labels = np.array(nodule_results['nodule_labels'])
    preds  = np.array(nodule_results['nodule_preds'])
    keys   = nodule_results['nodule_keys']

    tp = [keys[i] for i in range(len(keys)) if preds[i]==1 and labels[i]==1][:n_per_case]
    tn = [keys[i] for i in range(len(keys)) if preds[i]==0 and labels[i]==0][:n_per_case]
    fp = [keys[i] for i in range(len(keys)) if preds[i]==1 and labels[i]==0][:n_per_case]
    fn = [keys[i] for i in range(len(keys)) if preds[i]==0 and labels[i]==1][:n_per_case]

    print(f'[Grad-CAM] TP={len(tp)}, TN={len(tn)}, FP={len(fp)}, FN={len(fn)}')
    return {'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn}


def evaluate(args: argparse.Namespace) -> None:
    exp_dir    = Path(args.exp_dir)
    device     = get_device()
    config     = load_exp_config(exp_dir)
    model_name = config['model']
    crop_size  = config['crop_size']
    is_dual    = (model_name == 'dual_convnext')
    # dual_convnext: crop_size는 '32+96', '48+96', '64+96' 형식으로 저장됨
    # small_crop: DataLoader용, large_crop(=96): subgroup AUC CSV용
    if is_dual:
        parts = str(crop_size).split('+')
        small_crop = int(parts[0])
        csv_crop_size = int(parts[1]) if len(parts) > 1 else 96
    else:
        small_crop = None
        csv_crop_size = int(crop_size)

    print(f'[INFO] 실험 폴더: {exp_dir}')

    model = load_model(model_name, exp_dir, device, config)
    print(f'[INFO] 모델 로드: {model_name}')

    if is_dual:
        _, val_loader, test_loader, _ = get_dual_dataloaders(
            crop_size_small=small_crop, crop_size_large=96,
            batch_size=config['batch_size'], num_workers=config['num_workers']
        )
    else:
        _, val_loader, test_loader, _ = get_dataloaders(
            crop_size=csv_crop_size,
            batch_size=config['batch_size'],
            num_workers=config['num_workers']
        )

    pos_weight = torch.tensor([config['pos_weight']]).to(device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # val set에서 최적 threshold 탐색 → test set에 적용
    best_threshold = find_optimal_threshold(model, val_loader, device, is_dual=is_dual)

    # test set 단일 순회로 지표 + 상세 결과 동시 수집
    # (기존: validate_one_epoch + collect_test_results 두 번 순회 → 통합)
    nodule_results = collect_test_results(
        model, test_loader, device, criterion=criterion, is_dual=is_dual,
        threshold=best_threshold
    )

    # nodule_results에서 test_metrics 직접 계산
    import numpy as _np
    _labels = nodule_results['nodule_labels']
    _probs  = nodule_results['nodule_probs']
    _preds  = nodule_results['nodule_preds']
    try:
        _auc = roc_auc_score(_labels, _probs)
    except ValueError:
        _auc = 0.0
    _labels_arr = _np.array(_labels)
    _preds_arr  = _np.array(_preds)
    _tp = float(((_preds_arr==1)&(_labels_arr==1)).sum())
    _tn = float(((_preds_arr==0)&(_labels_arr==0)).sum())
    _fp = float(((_preds_arr==1)&(_labels_arr==0)).sum())
    _fn = float(((_preds_arr==0)&(_labels_arr==1)).sum())
    test_metrics = {
        'loss'        : nodule_results['slice_loss'],
        'auc'         : round(_auc, 4),
        'accuracy'    : round(float((_preds_arr==_labels_arr).mean()), 4),
        'sensitivity' : round(_tp/(_tp+_fn) if (_tp+_fn)>0 else 0.0, 4),
        'specificity' : round(_tn/(_tn+_fp) if (_tn+_fp)>0 else 0.0, 4),
        'n_nodules'   : len(_labels),
    }

    subgroup_auc = compute_subgroup_auc(nodule_results, csv_crop_size)

    # result.json 저장
    result = {
        'model'            : model_name,
        'crop_size'        : crop_size,
        'threshold': round(best_threshold, 4),  # 추가
        'threshold_method': 'youden_val',  # 추가
        'eval_unit'        : 'nodule',   # 결절 단위 평가
        'test_loss'        : test_metrics['loss'],
        'test_loss_unit'   : 'slice',
        'test_auc'         : test_metrics['auc'],
        'test_accuracy'    : test_metrics['accuracy'],
        'test_sensitivity' : test_metrics['sensitivity'],
        'test_specificity' : test_metrics['specificity'],
        'n_nodules'        : test_metrics.get('n_nodules', len(nodule_results['nodule_keys'])),
        'n_malignant'      : sum(nodule_results['nodule_labels']),
        'n_benign'         : nodule_results['nodule_labels'].count(0),
        'subgroup_auc'     : subgroup_auc,
        'subgroup_basis'   : 'volume_mm3',
    }
    # dual_convnext: small/large/metadata crop 출처를 명시적으로 기록
    # (small ablation 실험 A/B/C 결과를 나중에 비교할 때 혼동 방지)
    if is_dual:
        result['small_crop_size']   = small_crop
        result['large_crop_size']   = 96
        result['metadata_csv_crop'] = csv_crop_size  # subgroup AUC lookup에 쓴 CSV
    with open(exp_dir / 'result.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f'\n[RESULT] threshold={best_threshold:.4f} | ' 
          f'auc={test_metrics["auc"]:.4f} | '
          f'accuracy={test_metrics["accuracy"]:.4f} | '
          f'sensitivity={test_metrics["sensitivity"]:.4f} | '
          f'specificity={test_metrics["specificity"]:.4f} | '
          f'n_nodules={result["n_nodules"]}')

    # 시각화
    print('\n[INFO] 시각화 생성 중...')
    plot_roc_curve(labels=nodule_results['nodule_labels'],
                   probs=nodule_results['nodule_probs'],
                   save_path=exp_dir / 'roc_curve.png',
                   title=f'{model_name} | crop={crop_size} | AUC={test_metrics["auc"]:.4f} (nodule-level)')

    plot_confusion_matrix(labels=nodule_results['nodule_labels'],
                          preds=nodule_results['nodule_preds'],
                          save_path=exp_dir / 'confusion_matrix.png',
                          title=f'{model_name} | crop={crop_size} | thr={best_threshold:.4f}')

    plot_learning_curve(history_csv=exp_dir / 'history.csv',
                        save_path=exp_dir / 'learning_curve.png',
                        title=f'{model_name} | crop={crop_size}')

    # Grad-CAM: 결절 단위 TP/TN/FP/FN 샘플 선택
    gradcam_samples = select_gradcam_samples(nodule_results, n_per_case=5)
    gradcam_dir = exp_dir / 'gradcam'
    gradcam_dir.mkdir(exist_ok=True)

    save_gradcam(model=model, model_name=model_name, test_loader=test_loader,
                 selected_indices=gradcam_samples, save_dir=gradcam_dir,
                 device=device, is_dual=is_dual)

    print(f'\n[DONE] 평가 완료. 결과 폴더: {exp_dir}')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='LIDC-IDRI test set 최종 평가 (결절 단위)')
    parser.add_argument('--exp_dir', type=str, required=True)
    return parser.parse_args()


if __name__ == '__main__':
    evaluate(parse_args())
