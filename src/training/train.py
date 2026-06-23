# src/training/train.py
#
# ─── 역할 ────────────────────────────────────────────────────────────────────
#   실험 설정 관리 + 학습 루프 실행.
#   실제 epoch 연산은 engine.py에 위임.
#
# ─── 사용 방법 ───────────────────────────────────────────────────────────────
#   conda run -n resnet --no-capture-output \
#     python -m src.training.train --model gdn --crop_size 32
#
#   conda run -n resnet --no-capture-output \
#     python -m src.training.train --model convnext --crop_size 64
#
#   conda run -n resnet --no-capture-output \
#     python -m src.training.train --model dual_convnext
#
# ─── CBAM 실험 실행 예시 ─────────────────────────────────────────────────────
# EXP-A (baseline, CBAM 없음)
# conda run -n resnet --no-capture-output python -m src.training.train \
#   --model gdn --crop_size 32 --augment hflip rot90 \
#   --experiment_group cbam_search --variant cbam_none
#
# EXP-B (cbam1만)
# conda run -n resnet --no-capture-output python -m src.training.train \
#   --model gdn --crop_size 32 --augment hflip rot90 --cbam1 \
#   --experiment_group cbam_search --variant cbam1_only
#
# EXP-C (cbam2만)
# conda run -n resnet --no-capture-output python -m src.training.train \
#   --model gdn --crop_size 32 --augment hflip rot90 --cbam2 \
#   --experiment_group cbam_search --variant cbam2_only
#
# EXP-D (cbam1+cbam2)
# conda run -n resnet --no-capture-output python -m src.training.train \
#   --model gdn --crop_size 32 --augment hflip rot90 --cbam1 --cbam2 \
#   --experiment_group cbam_search --variant cbam_both
#
# EXP-C2 (cbam2만, gd5 이후)
# conda run -n resnet --no-capture-output python -m src.training.train \
#   --model gdn --crop_size 32 --augment hflip rot90 --cbam2 \
#   --experiment_group cbam_search --variant cbam2_after_gd5
#
# ─── 출력 구조 ───────────────────────────────────────────────────────────────
#   outputs/experiments/260608_gdn_32x32_ep50_aug1/
#     config.json     ← 실험 설정 전체 기록
#     history.csv     ← epoch별 train/val 지표
#     best_model.pth  ← val_auc 최고 시점
#     last_model.pth  ← 마지막 epoch

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn

from src.configs.config import PROCESSED_ROOT, OUTPUT_ROOT, SEED
from src.datasets.dataset import get_dataloaders, get_dual_dataloaders
from src.models.models import GDN, ConvNeXt, DualConvNeXt
from src.training.engine import train_one_epoch, validate_one_epoch
from src.utils.utils import get_device, set_seed


# ─────────────────────────────────────────────────────────────────────────────
# 경로 설정
# ─────────────────────────────────────────────────────────────────────────────

NPY_CACHE_ROOT = PROCESSED_ROOT / 'npy_cache'
EXP_ROOT       = OUTPUT_ROOT / 'experiments'


# ─────────────────────────────────────────────────────────────────────────────
# 섹션 1. 실험 폴더 생성
# ─────────────────────────────────────────────────────────────────────────────

def make_exp_dir(model_name: str, crop_size: str, epochs: int, n_aug: int) -> Path:
    """
    실험 결과 저장 폴더 자동 생성.

    네이밍: 날짜_모델_crop_ep_aug수
    예: 260608_gdn_32_ep50_aug2
    OUTPUT_ROOT 기준으로 생성 → 실행 위치와 무관하게 항상 동일 경로.
    """
    date_str = datetime.now().strftime('%y%m%d')
    exp_name = (f'{date_str}_{model_name}_'
                f'{crop_size}_'
                f'ep{epochs}_aug{n_aug}')
    exp_dir  = EXP_ROOT / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


def normalize_augmentations(args: argparse.Namespace) -> list[str]:
    """CLI augmentation 옵션을 학습에 사용할 리스트로 정규화."""
    if args.no_aug:
        return []
    return [a for a in args.augment if a != 'none']


# ─────────────────────────────────────────────────────────────────────────────
# 섹션 2. 모델 빌드
# ─────────────────────────────────────────────────────────────────────────────

def build_model(args: argparse.Namespace) -> nn.Module:
    """
    args에서 모델 인스턴스 생성.

    모든 모델 출력: (B, 1) logit.
    BCEWithLogitsLoss 사용 → forward에 sigmoid 붙이지 않음.

    GDN 전용 옵션:
      --cbam1 : drop1 이후 CBAM 적용 (채널+공간 어텐션)
      --cbam2 : drop2 이후 CBAM 적용
    """
    if args.model == 'gdn':
        return GDN(in_ch=3, num_classes=1,
                   use_cbam1=args.cbam1,
                   use_cbam2=args.cbam2,
                   use_cbam_final=args.cbam_final,
                   use_gd4b=args.gd4b,
                   use_gd6=args.gd6)
    elif args.model == 'convnext':
        return ConvNeXt(in_ch=3, num_classes=1)
    elif args.model == 'dual_convnext':
        return DualConvNeXt(num_classes=1)
    else:
        raise ValueError(f'알 수 없는 모델: {args.model}\n'
                         f'선택 가능: gdn, convnext, dual_convnext')


# ─────────────────────────────────────────────────────────────────────────────
# 섹션 3. 결과 저장 유틸리티
# ─────────────────────────────────────────────────────────────────────────────

def extract_gdn_config(model: nn.Module) -> dict:
    """
    실제 학습에 사용된 GDN 인스턴스에서 구조 설정값을 추출.
    다른 모델이 들어오면 빈 dict 반환 (안전).

    model 인스턴스에서 직접 읽으므로 args 기본값과 불일치 위험 없음.
    cbam1/cbam2는 None 여부로 실제 적용 여부를 확인.
    """
    if not hasattr(model, 'gd1'):
        return {}
    return {
        'gdn_dilation_d1': model.gd1.conv_d1.dilation[0],
        'gdn_dilation_d2': model.gd1.conv_d2.dilation[0],
        'gdn_dropout': {
            'drop1': model.drop1.p,
            'drop2': model.drop2.p,
            'drop3': model.drop3.p,
        },
        'gdn_cbam1': model.cbam1 is not None,   # 실제 인스턴스 확인
        'gdn_cbam2': model.cbam2 is not None,
        'gdn_cbam_final' : model.cbam_final is not None,    # ← 추가
        'gdn_gd4b': model.gd4b is not None,
        'gdn_gd6': model.gd6 is not None,
    }


def save_config(exp_dir: Path, args: argparse.Namespace, pos_weight: float,
                crop_info: str, augmentations: list[str], model: nn.Module) -> None:
    """실험 설정 전체를 config.json으로 저장."""
    config = {
        'created_at'       : datetime.now().isoformat(timespec='seconds'),
        'experiment_group' : args.experiment_group,
        'variant'          : args.variant,
        'model'            : args.model,
        'crop_size'        : crop_info,
        'epochs'           : args.epochs,
        'batch_size'       : args.batch_size,
        'num_workers'      : args.num_workers,
        'lr'               : args.lr,
        'weight_decay'     : args.weight_decay,
        'pos_weight'       : round(pos_weight, 4),
        'optimizer'        : 'AdamW',
        'scheduler'        : 'CosineAnnealingLR',
        'loss'             : 'BCEWithLogitsLoss',
        'augmentations'    : augmentations,
        'aug_prob'         : args.aug_prob,
        'n_slices'         : 1,
        'stride'           : 1,
        'seed'             : args.seed,
    }
    # 실제 모델 인스턴스에서 GDN 구조 추출 (gdn 외 모델은 빈 dict)
    config.update(extract_gdn_config(model))

    with open(exp_dir / 'config.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    print(f'[INFO] config.json 저장: {exp_dir / "config.json"}')


class HistoryWriter:
    """
    history.csv에 epoch별 지표를 한 줄씩 기록.
    매 epoch 즉시 기록 → 학습 중단 시에도 기록 보존.
    """
    FIELDNAMES = [
        'epoch',
        'train_loss', 'train_auc', 'train_accuracy', 'train_sensitivity', 'train_specificity',
        'val_loss',   'val_auc',   'val_accuracy',   'val_sensitivity',   'val_specificity',
        'best_auc_flag', 'lr',
    ]

    def __init__(self, exp_dir: Path):
        self.csv_path = exp_dir / 'history.csv'
        with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
            csv.DictWriter(f, fieldnames=self.FIELDNAMES).writeheader()

    def write(self, epoch: int, train_metrics: dict, val_metrics: dict,
              is_best: bool, lr: float) -> None:
        row = {
            'epoch'             : epoch,
            'train_loss'        : train_metrics['loss'],
            'train_auc'         : train_metrics['auc'],
            'train_accuracy'    : train_metrics['accuracy'],
            'train_sensitivity' : train_metrics['sensitivity'],
            'train_specificity' : train_metrics['specificity'],
            'val_loss'          : val_metrics['loss'],
            'val_auc'           : val_metrics['auc'],
            'val_accuracy'      : val_metrics['accuracy'],
            'val_sensitivity'   : val_metrics['sensitivity'],
            'val_specificity'   : val_metrics['specificity'],
            'best_auc_flag'     : 1 if is_best else 0,
            'lr'                : lr,
        }
        with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
            csv.DictWriter(f, fieldnames=self.FIELDNAMES).writerow(row)


class AlphaWriter:
    LAYER_NAMES = ['gd1', 'gd2', 'gd3', 'gd4', 'gd5']

    def __init__(self, exp_dir: Path):
        self.csv_path = exp_dir / 'alpha_history.csv'
        with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
            csv.DictWriter(f, fieldnames=['epoch'] + self.LAYER_NAMES).writeheader()

    def write(self, epoch: int, alpha_means: dict) -> None:
        row = {'epoch': epoch}
        row.update({k: round(alpha_means.get(k, 0.0), 4) for k in self.LAYER_NAMES})
        with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
            csv.DictWriter(f, fieldnames=['epoch'] + self.LAYER_NAMES).writerow(row)


# ─────────────────────────────────────────────────────────────────────────────
# 섹션 4. 메인 학습 루프
# ─────────────────────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    """
    전체 학습 파이프라인 실행.

    순서:
      1. seed 고정 + device 설정
      2. 실험 폴더 생성
      3. DataLoader 빌드
      4. 모델 / optimizer / scheduler / loss 설정
      5. epoch 루프
      6. best_model.pth / last_model.pth 저장
    """
    set_seed(args.seed)
    device  = get_device()
    is_dual = (args.model == 'dual_convnext')

    augmentations = normalize_augmentations(args)
    n_aug         = len(augmentations)
    display_crop  = f'{args.small_crop}+96' if is_dual else str(args.crop_size)
    exp_dir       = make_exp_dir(args.model, display_crop, args.epochs, n_aug)
    print(f'[INFO] 실험 폴더: {exp_dir}')

    # DataLoader
    if is_dual:
        train_loader, val_loader, _, _ = get_dual_dataloaders(
            crop_size_small=args.small_crop, crop_size_large=96,
            batch_size=args.batch_size, num_workers=args.num_workers,
            augmentations=augmentations, aug_prob=args.aug_prob,
        )
        pos_weight = torch.tensor([args.pos_weight])
    else:
        train_loader, val_loader, _, _ = get_dataloaders(
            crop_size=args.crop_size,
            batch_size=args.batch_size, num_workers=args.num_workers,
            augmentations=augmentations, aug_prob=args.aug_prob,
        )
        pos_weight = torch.tensor([args.pos_weight])

    # 모델 — args 전체를 넘겨서 cbam1/cbam2 포함
    model = build_model(args).to(device)
    print(f'[INFO] 모델: {args.model} | 파라미터: '
          f'{sum(p.numel() for p in model.parameters() if p.requires_grad):,}')

    pos_weight = pos_weight.to(device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer  = torch.optim.AdamW(model.parameters(),
                                   lr=args.lr, weight_decay=args.weight_decay)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    save_config(exp_dir, args, float(pos_weight.item()), display_crop, augmentations, model)
    history_writer = HistoryWriter(exp_dir)
    alpha_writer   = AlphaWriter(exp_dir) if args.model == 'gdn' else None

    best_val_auc = 0.0

    for epoch in range(1, args.epochs + 1):
        train_metrics             = train_one_epoch(
            model, train_loader, criterion, optimizer, device, is_dual=is_dual)
        val_metrics, alpha_means  = validate_one_epoch(
            model, val_loader,   criterion,           device, is_dual=is_dual)

        if alpha_writer is not None and alpha_means:
            alpha_writer.write(epoch, alpha_means)
            print('[ALPHA] ' + ' '.join(f'{k}={v:.3f}' for k, v in alpha_means.items()))

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        is_best = val_metrics['auc'] > best_val_auc
        if is_best:
            best_val_auc = val_metrics['auc']
            torch.save(model.state_dict(), exp_dir / 'best_model.pth')

        history_writer.write(epoch, train_metrics, val_metrics, is_best, lr=current_lr)

        print(f'[Epoch {epoch:3d}/{args.epochs}] '
              f'train_loss={train_metrics["loss"]:.4f} train_auc={train_metrics["auc"]:.4f} | '
              f'val_loss={val_metrics["loss"]:.4f} val_auc={val_metrics["auc"]:.4f}'
              f'{" ← best" if is_best else ""}')

    torch.save(model.state_dict(), exp_dir / 'last_model.pth')
    print(f'\n[DONE] 학습 완료 | best val_auc: {best_val_auc:.4f}')
    print(f'평가 실행: python -m src.evaluation.evaluate --exp_dir {exp_dir}')


# ─────────────────────────────────────────────────────────────────────────────
# 섹션 5. 명령줄 인터페이스
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='LIDC-IDRI 결절 분류 학습')
    parser.add_argument('--model',        type=str,   default='convnext',
                        choices=['gdn', 'convnext', 'dual_convnext'])
    parser.add_argument('--no_aug',       action='store_true',
                        help='증강 없이 raw 데이터만 사용')
    parser.add_argument('--augment',      type=str,   nargs='+', default=['none'],
                        choices=['none', 'hflip', 'vflip', 'rot90', 'hu_shift', 'gaussian_noise'],
                        help='online augmentation. 예: --augment hflip rot90')
    parser.add_argument('--aug_prob',     type=float, default=0.5)
    parser.add_argument('--crop_size',    type=int,   default=64,
                        help='gdn=32, convnext=64 권장')
    parser.add_argument('--small_crop',   type=int,   default=32,
                        choices=[32, 48, 64],
                        help='dual_convnext small branch crop size')
    parser.add_argument('--epochs',       type=int,   default=50)
    parser.add_argument('--batch_size',   type=int,   default=16)
    parser.add_argument('--num_workers',  type=int,   default=4)
    parser.add_argument('--lr',           type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--pos_weight', type=float, default=1.0,
                    help='BCEWithLogitsLoss pos_weight (기본값 1.0)')
    # ── GDN CBAM 실험 옵션 ───────────────────────────────────────────────────
    parser.add_argument('--cbam1',        action='store_true',
                        help='GDN: drop1 이후 CBAM 적용 (EXP-B, EXP-D)')
    parser.add_argument('--cbam2',        action='store_true',
                        help='GDN: drop2 이후 CBAM 적용 (EXP-C, EXP-D)')
    parser.add_argument('--cbam_final',   action='store_true',
                        help='GDN: drop3 이후 CBAM 적용 (EXP-E)')    # ← 추가
    parser.add_argument('--gd4b', action='store_true',
                    help='GDN: gd4 이후 GDLayer 추가 (EXP-F1)')
    parser.add_argument('--gd6', action='store_true',
                    help='GDN: gd5 이후 GDLayer 추가 (EXP-F2)')
    # ── 실험 메타 ────────────────────────────────────────────────────────────
    parser.add_argument('--experiment_group', type=str, default='',
                        help='실험 그룹명. 예: cbam_search, dilation_search')
    parser.add_argument('--variant',          type=str, default='',
                        help='변형 이름. 예: cbam_none / cbam1_only / cbam_both')
    parser.add_argument('--seed',         type=int,   default=SEED)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    crop_info = f'{args.small_crop}+96' if args.model == 'dual_convnext' else str(args.crop_size)
    print(f'[CONFIG] model={args.model} | crop={crop_info} | '
          f'epochs={args.epochs} | batch={args.batch_size} | lr={args.lr} | '
          f'augment={normalize_augmentations(args)} | '
          f'cbam1={args.cbam1} cbam2={args.cbam2}')
    train(args)
