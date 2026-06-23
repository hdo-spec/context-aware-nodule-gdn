# src/datasets/dataset.py
#
# ─── 변경 사항 ────────────────────────────────────────────────────────────────
#   __getitem__ 반환 형식 변경:
#     기존: (patch_tensor, label_tensor)
#     변경: (patch_tensor, label_tensor, subject_id, nodule_idx, z_idx)
#
#   확장성:
#     subject_id, nodule_idx: val/test에서 결절별 평균 집계에 사용
#     z_idx: 특정 슬라이스 추적, Grad-CAM 디버깅에 사용

import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from src.configs.config import PROCESSED_ROOT

NPY_CACHE_ROOT = PROCESSED_ROOT / 'npy_cache'
SUPPORTED_AUGMENTATIONS = {'hflip', 'vflip', 'rot90', 'hu_shift', 'gaussian_noise'}


class PatchAugment:
    """학습 시점에 raw patch tensor에 적용하는 online augmentation."""

    def __init__(self, augmentations: list[str] | None = None,
                 prob: float = 0.5,
                 hu_shift: float = 0.05,
                 noise_sigma: float = 0.02):
        self.augmentations = [a for a in (augmentations or []) if a and a != 'none']
        unknown = sorted(set(self.augmentations) - SUPPORTED_AUGMENTATIONS)
        if unknown:
            raise ValueError(f'지원하지 않는 augmentation: {unknown}')
        self.prob = float(prob)
        self.hu_shift = float(hu_shift)
        self.noise_sigma = float(noise_sigma)

    def __bool__(self) -> bool:
        return bool(self.augmentations)

    def __call__(self, patch: torch.Tensor) -> torch.Tensor:
        patch = patch.clone()
        for aug in self.augmentations:
            if torch.rand(()) > self.prob:
                continue
            if aug == 'hflip':
                patch = torch.flip(patch, dims=(2,))
            elif aug == 'vflip':
                patch = torch.flip(patch, dims=(1,))
            elif aug == 'rot90':
                k = int(torch.randint(1, 4, ()).item())
                patch = torch.rot90(patch, k=k, dims=(1, 2))
            elif aug == 'hu_shift':
                shift = (torch.rand(()) * 2.0 - 1.0) * self.hu_shift
                patch = torch.clamp(patch + shift, 0.0, 1.0)
            elif aug == 'gaussian_noise':
                noise = torch.randn_like(patch) * self.noise_sigma
                patch = torch.clamp(patch + noise, 0.0, 1.0)
        return patch.contiguous()


class DualPatchAugment(PatchAugment):
    """DualConvNeXt용 online augmentation. 두 crop에 같은 기하 변환을 적용한다."""

    def __call__(self, patch_small: torch.Tensor, patch_large: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        patch_small = patch_small.clone()
        patch_large = patch_large.clone()

        for aug in self.augmentations:
            if torch.rand(()) > self.prob:
                continue
            if aug == 'hflip':
                patch_small = torch.flip(patch_small, dims=(2,))
                patch_large = torch.flip(patch_large, dims=(2,))
            elif aug == 'vflip':
                patch_small = torch.flip(patch_small, dims=(1,))
                patch_large = torch.flip(patch_large, dims=(1,))
            elif aug == 'rot90':
                k = int(torch.randint(1, 4, ()).item())
                patch_small = torch.rot90(patch_small, k=k, dims=(1, 2))
                patch_large = torch.rot90(patch_large, k=k, dims=(1, 2))
            elif aug == 'hu_shift':
                shift = (torch.rand(()) * 2.0 - 1.0) * self.hu_shift
                patch_small = torch.clamp(patch_small + shift, 0.0, 1.0)
                patch_large = torch.clamp(patch_large + shift, 0.0, 1.0)
            elif aug == 'gaussian_noise':
                patch_small = torch.clamp(
                    patch_small + torch.randn_like(patch_small) * self.noise_sigma,
                    0.0, 1.0
                )
                patch_large = torch.clamp(
                    patch_large + torch.randn_like(patch_large) * self.noise_sigma,
                    0.0, 1.0
                )

        return patch_small.contiguous(), patch_large.contiguous()


def _sample_key(row: dict) -> tuple[str, str, str, str]:
    """small/large patch CSV를 정확히 매칭하기 위한 안정 키."""
    return (
        row['subject_id'],
        row['nodule_idx'],
        row['z_idx'],
        row.get('aug_type', 'raw'),
    )


class NoduleDataset(Dataset):
    """
    LIDC-IDRI 결절 분류용 PyTorch Dataset.

    __getitem__ 반환:
        patch_tensor : FloatTensor (C, H, W)
        label_tensor : LongTensor scalar (0=양성, 1=악성)
        subject_id   : str (예: 'LIDC-IDRI-0001')
        nodule_idx   : str (예: '0')
        z_idx        : int (슬라이스 인덱스)
    """

    def __init__(self, csv_path: Path, fold: str | None = None,
                 transform=None, preload: bool = False, augment=None):
        self.transform = transform
        self.preload   = preload
        self.augment   = augment
        self.rows      = self._load_rows(csv_path, fold)

        # preload: 전체 patch를 RAM에 미리 올려둠
        # 장점: 학습 중 디스크 IO 없음 → epoch 빠름
        # 단점: 메모리 사용량 증가 (2910샘플 × 3×64×64 × 4B ≈ 약 270MB)
        # 사용 조건: RAM 충분한 경우에만 활성화 (기본값: False)
        if self.preload:
            print(f'[Dataset] preload 시작: {len(self.rows)}개 patch → RAM 로드 중...')
            self.patches = np.stack([
                np.load(r['patch_path']).astype(np.float32) for r in self.rows
            ])   # (N, C, H, W)
            print(f'[Dataset] preload 완료: {self.patches.nbytes / 1e6:.1f}MB')

    def _load_rows(self, csv_path: Path, fold: str | None) -> list[dict]:
        if not csv_path.exists():
            raise FileNotFoundError(
                f'Patch CSV 없음: {csv_path}\n'
                f'먼저 실행: python -m src.preprocessing.export_patches'
            )
        with open(csv_path, 'r', newline='', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        if fold is not None:
            rows = [r for r in rows if r['fold'] == fold]
        return rows

    def __len__(self) -> int:
        return len(self.rows)
    
    @property
    def pos_weight(self) -> torch.Tensor:
        n_neg = sum(1 for r in self.rows if r['label'] == '0')
        n_pos = sum(1 for r in self.rows if r['label'] == '1')
        if n_pos == 0:
            raise ValueError('malignant 샘플이 없음')
        return torch.tensor([n_neg / n_pos], dtype=torch.float32)

    def __getitem__(self, idx: int) -> tuple:
        row = self.rows[idx]

        if self.preload:
            patch = self.patches[idx]   # RAM에서 직접 접근 (idx 사용 필수)
        else:
            patch = np.load(row['patch_path']).astype(np.float32)

        patch_tensor = torch.from_numpy(patch)
        label_tensor = torch.tensor(int(row['label']), dtype=torch.long)

        if self.transform is not None:
            patch_tensor = self.transform(patch_tensor)
        if self.augment is not None:
            patch_tensor = self.augment(patch_tensor)

        return (patch_tensor, label_tensor,
                row['subject_id'],         # str
                row['nodule_idx'],         # str
                int(row['z_idx']))         # int


class DualNoduleDataset(Dataset):
    """
    DualConvNeXt 전용 Dataset.
    같은 idx에서 small/large crop을 동시에 로드하여 매칭 보장.

    __getitem__ 반환:
        patch_small  : FloatTensor (3, 32, 32)
        patch_large  : FloatTensor (3, 96, 96)
        label_tensor : LongTensor scalar
        subject_id   : str
        nodule_idx   : str
        z_idx        : int
    """

    def __init__(self, csv_path_small: Path, csv_path_large: Path,
                 fold: str | None = None, augment=None):
        self.augment = augment
        self.rows_small = self._load_rows(csv_path_small, fold)
        self.rows_large = self._load_rows(csv_path_large, fold)
        self.rows_large_by_key = self._build_large_lookup(self.rows_large)
        self._validate_all_pairs()

    def _load_rows(self, csv_path: Path, fold: str | None) -> list[dict]:
        if not csv_path.exists():
            raise FileNotFoundError(f'Patch CSV 없음: {csv_path}')
        with open(csv_path, 'r', newline='', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        if fold is not None:
            rows = [r for r in rows if r['fold'] == fold]
        return rows

    def _build_large_lookup(self, rows: list[dict]) -> dict[tuple[str, str, str, str], dict[str, Any]]:
        lookup = {}
        for row in rows:
            key = _sample_key(row)
            if key in lookup:
                raise ValueError(f'large CSV 중복 샘플 key: {key}')
            lookup[key] = row
        return lookup

    def _validate_all_pairs(self) -> None:
        small_keys = [_sample_key(row) for row in self.rows_small]
        large_keys = set(self.rows_large_by_key)
        missing = [key for key in small_keys if key not in large_keys]
        extra = sorted(large_keys - set(small_keys))

        if missing or extra:
            msg = [
                'small/large CSV 샘플 매칭 실패',
                f'  small rows={len(self.rows_small)}, large rows={len(self.rows_large)}',
                f'  missing_in_large={len(missing)}, extra_in_large={len(extra)}',
            ]
            if missing:
                msg.append(f'  missing 예시: {missing[:5]}')
            if extra:
                msg.append(f'  extra 예시: {extra[:5]}')
            raise ValueError('\n'.join(msg))

        for row_s in self.rows_small:
            row_l = self.rows_large_by_key[_sample_key(row_s)]
            if row_s['label'] != row_l['label'] or row_s['fold'] != row_l['fold']:
                raise ValueError(
                    'small/large CSV label/fold 불일치: '
                    f'{_sample_key(row_s)} small(label={row_s["label"]}, fold={row_s["fold"]}) '
                    f'large(label={row_l["label"]}, fold={row_l["fold"]})'
                )

    def __len__(self) -> int:
        return len(self.rows_small)
    
    @property
    def pos_weight(self) -> torch.Tensor:
        n_neg = sum(1 for r in self.rows_small if r['label'] == '0')
        n_pos = sum(1 for r in self.rows_small if r['label'] == '1')
        if n_pos == 0:
            raise ValueError('malignant 샘플이 없음')
        return torch.tensor([n_neg / n_pos], dtype=torch.float32)

    def __getitem__(self, idx: int) -> tuple:
        row_s = self.rows_small[idx]
        row_l = self.rows_large_by_key[_sample_key(row_s)]

        patch_small  = torch.from_numpy(np.load(row_s['patch_path']).astype(np.float32))
        patch_large  = torch.from_numpy(np.load(row_l['patch_path']).astype(np.float32))
        label_tensor = torch.tensor(int(row_s['label']), dtype=torch.long)

        if self.augment is not None:
            patch_small, patch_large = self.augment(patch_small, patch_large)

        return (patch_small, patch_large, label_tensor,
                row_s['subject_id'],
                row_s['nodule_idx'],
                int(row_s['z_idx']))


def get_dataloaders(crop_size: int, batch_size: int = 16, num_workers: int = 4,
                    transform=None, augmentations: list[str] | None = None,
                    aug_prob: float = 0.5) -> tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    """
    train/val/test DataLoader 생성.

    train: labels_raw_{crop_size}.csv, shuffle=True, online augmentation optional
    val  : labels_raw_{crop_size}.csv, shuffle=False
    test : labels_raw_{crop_size}.csv, shuffle=False
    """
    raw_csv = NPY_CACHE_ROOT / f'labels_raw_{crop_size}.csv'

    augment = PatchAugment(augmentations, prob=aug_prob) if augmentations else None
    raw_train_ds = NoduleDataset(raw_csv, fold='train', transform=transform, augment=augment)
    val_ds   = NoduleDataset(raw_csv, fold='val',   transform=None)
    test_ds  = NoduleDataset(raw_csv, fold='test',  transform=None)

    print(f'[Dataset] crop_size={crop_size} | augmentations={augmentations or []}')
    print(f'  train : {len(raw_train_ds)} samples (슬라이스 단위)')
    print(f'  val   : {len(val_ds)} samples (슬라이스 단위)')
    print(f'  test  : {len(test_ds)} samples (슬라이스 단위)')

    n_b = sum(1 for r in raw_train_ds.rows if r['label'] == '0')
    n_m = sum(1 for r in raw_train_ds.rows if r['label'] == '1')
    n_nodules = len(set((r['subject_id'], r['nodule_idx']) for r in raw_train_ds.rows))
    print(f'  train 결절 수: {n_nodules}개 | 레이블: benign={n_b}, malignant={n_m}')

    train_loader = DataLoader(raw_train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True,
                              persistent_workers=(num_workers > 0), drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True,
                              persistent_workers=(num_workers > 0), drop_last=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True,
                              persistent_workers=(num_workers > 0), drop_last=False)

    return train_loader, val_loader, test_loader, raw_train_ds.pos_weight


def get_dual_dataloaders(crop_size_small: int = 32, crop_size_large: int = 96,
                         batch_size: int = 16,
                         num_workers: int = 4,
                         augmentations: list[str] | None = None,
                         aug_prob: float = 0.5) -> tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    """DualConvNeXt용 DataLoader."""
    raw_csv_s = NPY_CACHE_ROOT / f'labels_raw_{crop_size_small}.csv'
    raw_csv_l = NPY_CACHE_ROOT / f'labels_raw_{crop_size_large}.csv'

    augment = DualPatchAugment(augmentations, prob=aug_prob) if augmentations else None

    raw_train_ds = DualNoduleDataset(raw_csv_s, raw_csv_l, fold='train', augment=augment)
    val_ds   = DualNoduleDataset(raw_csv_s, raw_csv_l, fold='val')
    test_ds  = DualNoduleDataset(raw_csv_s, raw_csv_l, fold='test')

    print(f'[DualDataset] small={crop_size_small}, large={crop_size_large} | augmentations={augmentations or []}')
    print(f'  train: {len(raw_train_ds)} | val: {len(val_ds)} | test: {len(test_ds)}')

    train_loader = DataLoader(raw_train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True,
                              persistent_workers=(num_workers > 0), drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True,
                              persistent_workers=(num_workers > 0), drop_last=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True,
                              persistent_workers=(num_workers > 0), drop_last=False)

    return train_loader, val_loader, test_loader, raw_train_ds.pos_weight
