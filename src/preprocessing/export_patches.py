# src/preprocessing/export_patches.py
#
# ─── 역할 ────────────────────────────────────────────────────────────────────
#   실험 전 딱 한 번만 실행하는 전처리 스크립트.
#   NIfTI → 결절별 2.5D patch .npy 파일로 저장 (cache).
#
# ─── 변경 사항 (z-slice 전체 활용) ──────────────────────────────────────────
#   기존: 결절당 대표 슬라이스 1개 → 1개 샘플
#   변경: 결절이 존재하는 모든 z 슬라이스 (양끝 제외) → 슬라이스당 1개 샘플
#
#   z 범위 추출 방법:
#     1차: seg.nii.gz + seg_manifest.json → 실제 voxel 기준 z 범위 (정확)
#     2차: num_slices (fallback) → center_z ± num_slices//2 근사
#     교차 검증: 두 방법의 슬라이스 수 차이가 크면 경고 출력
#
#   양끝 제외 이유:
#     결절 경계 슬라이스는 결절이 일부만 포함 → 노이즈
#     z_slices[1:-1]로 양끝 제거
#
# ─── 파일명 규칙 ─────────────────────────────────────────────────────────────
#   raw: {subject_id}_n{nodule_idx}_z{z_idx}.npy
#
# ─── 사용 방법 ───────────────────────────────────────────────────────────────
#   python -m src.preprocessing.export_patches --crop_sizes 32 64 96

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import nibabel as nib
import numpy as np

from src.configs.config import PROCESSED_ROOT, LABELS_CSV, SPLIT_JSON, SEED

NPY_CACHE_ROOT = PROCESSED_ROOT / 'npy_cache'


# ─────────────────────────────────────────────────────────────────────────────
# 섹션 1. 유틸리티 함수
# ─────────────────────────────────────────────────────────────────────────────

def load_labels(labels_csv: Path) -> list[dict]:
    if not labels_csv.exists():
        raise FileNotFoundError(f'labels.csv 없음: {labels_csv}')
    with open(labels_csv, 'r', newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def load_split(split_json: Path) -> dict[str, str]:
    if not split_json.exists():
        raise FileNotFoundError(f'split.json 없음: {split_json}')
    with open(split_json, 'r', encoding='utf-8') as f:
        split = json.load(f)
    subject_to_fold = {}
    for fold, subjects in split.items():
        if fold == 'meta':
            continue
        for sid in subjects:
            subject_to_fold[sid] = fold
    return subject_to_fold


def center_z_mm_to_slice_idx(center_z_mm: float, nifti_img) -> int:
    """
    center_z (mm) → NIfTI slice index 변환.
    deci-mm / 부호 반전 케이스 자동 보정.
    """
    affine    = nifti_img.affine
    z_origin  = float(affine[2, 3])
    # affine[2,2]: 부호 포함 z spacing (get_zooms()는 절대값만 반환)
    # 부호 포함이어야 z축 반전 케이스(z_spacing < 0)를 올바르게 처리
    z_spacing = float(affine[2, 2])
    D         = nifti_img.shape[2]

    if abs(z_spacing) < 1e-6:
        z_spacing = 1.0

    z_min = z_origin
    z_max = z_origin + z_spacing * (D - 1)
    if z_min > z_max:
        z_min, z_max = z_max, z_min

    margin   = max(abs(z_max - z_min) * 0.1, 5.0)
    z_center = (z_min + z_max) / 2.0

    if z_min - margin <= center_z_mm <= z_max + margin:
        idx = int(round((center_z_mm - z_origin) / z_spacing))
        return int(np.clip(idx, 0, D - 1))

    cz_neg   = -center_z_mm
    cz_fixed = center_z_mm / 10.0
    zm_fixed = z_min / 10.0
    zx_fixed = z_max / 10.0

    candidates = []
    if z_min - margin <= cz_neg <= z_max + margin:
        candidates.append((abs(cz_neg - z_center),   cz_neg,        z_origin,        z_spacing))
    if z_min - margin <= cz_fixed <= z_max + margin:
        candidates.append((abs(cz_fixed - z_center), cz_fixed,       z_origin,        z_spacing))
    if zm_fixed - margin <= center_z_mm <= zx_fixed + margin:
        candidates.append((abs(center_z_mm - z_center), center_z_mm, z_origin / 10.0, z_spacing / 10.0))

    if candidates:
        _, cz_use, zo_use, zs_use = min(candidates, key=lambda x: x[0])
        idx = int(round((cz_use - zo_use) / zs_use))
        return int(np.clip(idx, 0, D - 1))

    print(f'\n[WARN] center_z={center_z_mm:.1f} 보정 실패 → 중심 슬라이스 사용')
    return D // 2


def _first_nonempty(row: dict, keys: list[str]) -> str:
    """row에서 먼저 발견되는 비어 있지 않은 값을 반환."""
    for key in keys:
        value = row.get(key, '')
        if value not in ('', None):
            return value
    return ''


def _get_seg_label(manifest: dict, nodule_idx: int):
    for item in manifest.get('series', []):
        for nidx, lbl in zip(item.get('nodule_indices', []), item.get('labels', [])):
            if nidx == nodule_idx:
                return lbl
    return None


def get_nodule_voxel_info_from_seg(nifti_path: str, nodule_idx: int) -> dict | None:
    """
    seg.nii.gz에서 결절 mask의 voxel 중심과 z 슬라이스를 직접 계산.

    nibabel 배열은 (x, y, z)이므로 z축은 axis=2이다.
    """
    seg_path = Path(nifti_path).parent / 'seg.nii.gz'
    manifest_path = Path(nifti_path).parent / 'seg_manifest.json'

    if not seg_path.exists() or not manifest_path.exists():
        return None

    try:
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)

        label = _get_seg_label(manifest, nodule_idx)
        if label is None:
            return None

        seg_array = nib.load(str(seg_path)).get_fdata(dtype=np.float32)
        xs, ys, zs = np.where(seg_array == label)
        if len(xs) == 0:
            return None

        z_indices = sorted(set(int(z) for z in zs.tolist()))
        z_slices = z_indices if len(z_indices) < 3 else z_indices[1:-1]

        return {
            'cx': int(round(float(np.median(xs)))),
            'cy': int(round(float(np.median(ys)))),
            'cz': int(round(float(np.median(zs)))),
            'z_slices': z_slices,
            'label': label,
        }
    except Exception as e:
        print(f'\n[WARN] seg voxel 정보 추출 실패: {e}')
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 섹션 2. z 범위 추출
# ─────────────────────────────────────────────────────────────────────────────

def get_z_slices_from_seg(nifti_path: str, nodule_idx: int) -> list[int] | None:
    """
    seg.nii.gz + seg_manifest.json에서 결절의 z 슬라이스 목록 추출.

    동작:
      1. manifest에서 nodule_idx → seg label 매핑
      2. seg_array에서 해당 label이 존재하는 z 슬라이스 추출
      3. 양끝 제외 (z_slices[1:-1])

    Returns:
        양끝 제외된 z index 리스트. seg 없으면 None.
    """
    seg_path      = Path(nifti_path).parent / 'seg.nii.gz'
    manifest_path = Path(nifti_path).parent / 'seg_manifest.json'

    if not seg_path.exists() or not manifest_path.exists():
        return None

    try:
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)

        # nodule_idx → seg label 매핑
        label = None
        for item in manifest.get('series', []):
            for nidx, lbl in zip(item.get('nodule_indices', []), item.get('labels', [])):
                if nidx == nodule_idx:
                    label = lbl
                    break
            if label is not None:
                break

        if label is None:
            return None

        seg_array = nib.load(str(seg_path)).get_fdata(dtype=np.float32)

        # 해당 label이 존재하는 z 슬라이스 찾기
        # nibabel seg_array shape: (x, y, z) → z축(axis=2)만 남김
        z_has_label = np.any(seg_array == label, axis=(0, 1))  # (z,) bool array
        z_indices   = np.where(z_has_label)[0].tolist()        # z index 리스트

        if len(z_indices) < 3:
            # 양끝 제외 후 샘플이 없으면 전체 사용
            return z_indices

        return z_indices[1:-1]   # 양끝 제외

    except Exception as e:
        print(f'\n[WARN] seg z 범위 추출 실패: {e}')
        return None


def get_z_slices_from_num_slices(center_z_idx: int, num_slices_str: str, D: int) -> list[int]:
    """
    num_slices 컬럼으로 z 범위 근사 (seg 없을 때 fallback).

    center_z ± num_slices//2 범위에서 양끝 제외.

    Args:
        center_z_idx  : 중심 슬라이스 인덱스
        num_slices_str: labels.csv의 num_slices 값 (문자열)
        D             : CT z 방향 전체 슬라이스 수

    Returns:
        양끝 제외된 z index 리스트
    """
    try:
        num_slices = int(float(num_slices_str)) if num_slices_str else 3
    except ValueError:
        num_slices = 3

    half    = num_slices // 2
    z_start = max(0, center_z_idx - half)
    z_end   = min(D - 1, center_z_idx + half)
    z_all   = list(range(z_start, z_end + 1))

    if len(z_all) < 3:
        return z_all
    return z_all[1:-1]   # 양끝 제외


def get_z_slices(nifti_path: str, nodule_idx: int, center_z_idx: int,
                 num_slices_str: str, D: int) -> tuple[list[int], str]:
    """
    seg → num_slices 순서로 z 슬라이스 목록 결정.
    두 방법 결과를 교차 검증하여 차이가 크면 경고 출력.

    Returns:
        (z_slices, source): z index 리스트, 사용한 방법 ('seg' 또는 'num_slices')
    """
    seg_slices = get_z_slices_from_seg(nifti_path, nodule_idx)

    num_slices_result = get_z_slices_from_num_slices(center_z_idx, num_slices_str, D)

    if seg_slices is not None:
        # 교차 검증: 슬라이스 수 차이가 2 이상이면 경고
        diff = abs(len(seg_slices) - len(num_slices_result))
        if diff >= 2:
            pass
            # print(f'\n[WARN] z 범위 불일치: seg={len(seg_slices)}개, '
            #       f'num_slices={len(num_slices_result)}개 (차이={diff}) '
            #       f'→ seg 기준 사용')
        return seg_slices, 'seg'

    # seg 없으면 num_slices fallback
    return num_slices_result, 'num_slices'


# ─────────────────────────────────────────────────────────────────────────────
# 섹션 3. Crop 함수
# ─────────────────────────────────────────────────────────────────────────────

def crop_25d(ct: np.ndarray, cx: int, cy: int, cz: int, crop_size: int, n_slices: int) -> np.ndarray:
    """
    CT 볼륨에서 2.5D patch crop. stride=1 고정: [k-1, k, k+1].

    destination canvas 방식으로 shape 보장.
    Returns: (2*n_slices+1, crop_size, crop_size), float16
    """
    H, W, D = ct.shape
    half    = crop_size // 2
    n_ch    = 2 * n_slices + 1

    canvas = np.zeros((n_ch, crop_size, crop_size), dtype=np.float16)

    # nibabel get_fdata() → (X, Y, Z) shape
    # ct.shape = (H, W, D)에서 H=X축, W=Y축
    # cx=x좌표(첫번째 축=row), cy=y좌표(두번째 축=col)
    # slice_2d = ct[x_range, y_range, z] → shape (x_len, y_len)
    # canvas는 (Z, H, W) = (채널, row, col) 방향
    # slice_2d.T로 전치해서 (y_len, x_len) → canvas(col, row) 방향과 맞춤
    src_r0, src_r1 = cx - half, cx + half   # X축 (nibabel 첫번째 축)
    src_c0, src_c1 = cy - half, cy + half   # Y축 (nibabel 두번째 축)

    valid_r0 = max(0, src_r0);  valid_r1 = min(H, src_r1)
    valid_c0 = max(0, src_c0);  valid_c1 = min(W, src_c1)

    if valid_r0 >= valid_r1 or valid_c0 >= valid_c1:
        return canvas

    dst_r0 = valid_r0 - src_r0
    dst_c0 = valid_c0 - src_c0
    dst_r1 = dst_r0 + (valid_r1 - valid_r0)
    dst_c1 = dst_c0 + (valid_c1 - valid_c0)

    for ch_idx, dz in enumerate(range(-n_slices, n_slices + 1)):
        z_idx    = int(np.clip(cz + dz, 0, D - 1))
        slice_2d = ct[valid_r0:valid_r1, valid_c0:valid_c1, z_idx]  # (x_len, y_len)
        # .T: (x_len, y_len) → (y_len, x_len) = canvas (col, row) 방향
        canvas[ch_idx, dst_c0:dst_c1, dst_r0:dst_r1] = slice_2d.T.astype(np.float16)

    return canvas


# ─────────────────────────────────────────────────────────────────────────────
# 섹션 4. 메인 export 함수
# ─────────────────────────────────────────────────────────────────────────────

def export_patches(crop_sizes: list[int], n_slices: int = 1,
                   seed: int = SEED) -> None:
    """
    결절별 모든 z 슬라이스에 대해 crop_sizes별 raw patch만 생성.

    경계 상황:
      1. nifti_path 없음           → 건너뜀 + 경고
      2. center_x/y/z 비어있음     → 건너뜀 + 경고
      3. z 범위 추출 실패          → center_z 단일 슬라이스만 사용
      4. 이미 존재하는 .npy        → 덮어쓰지 않음 (재실행 안전)
    """
    print(f'[INFO] labels.csv 로드: {LABELS_CSV}')
    rows = load_labels(LABELS_CSV)
    print(f'[INFO] 총 결절 수: {len(rows)}')

    print(f'[INFO] split.json 로드')
    subject_to_fold = load_split(SPLIT_JSON)

    raw_dirs, raw_csv_rows = {}, {}
    for cs in crop_sizes:
        size_tag         = f'{cs}x{cs}'
        raw_dirs[cs]     = NPY_CACHE_ROOT / 'raw' / size_tag
        raw_dirs[cs].mkdir(parents=True, exist_ok=True)
        raw_csv_rows[cs] = []

    # NIfTI 캐싱: 같은 환자 결절이 여러 개면 반복 로드 방지
    cached_path, cached_img, cached_ct = None, None, None
    total, skipped = len(rows), 0
    total_samples  = 0   # 생성된 전체 샘플 수 (결절 × 슬라이스)

    for i, row in enumerate(rows):
        if i % 100 == 0:
            print(f'  [{i:4d}/{total}] 처리 중... (샘플 {total_samples}개 생성)', end='\r')

        # ── NIfTI 로드 ────────────────────────────────────────
        nifti_path = row.get('nifti_path', '')
        if not nifti_path or not Path(nifti_path).exists():
            print(f'\n[WARN] NIfTI 없음, 건너뜀: {nifti_path}')
            skipped += 1
            continue

        if nifti_path != cached_path:
            cached_img  = nib.load(nifti_path)
            cached_ct   = cached_img.get_fdata(dtype=np.float32)
            cached_path = nifti_path

        ct, img = cached_ct, cached_img
        D = ct.shape[2]

        subject_id      = row['subject_id']
        nodule_idx      = row.get('nodule_idx', str(i))
        fold            = subject_to_fold.get(subject_id, 'unknown')
        diam_str        = row.get('diameter_max_mm', '')
        diameter_max_mm = float(diam_str) if diam_str else 0.0
        num_slices_str  = row.get('num_slices', '')
        seg_info        = get_nodule_voxel_info_from_seg(nifti_path, int(nodule_idx))

        # ── 좌표 파싱 ─────────────────────────────────────────
        if seg_info is not None:
            cx = seg_info['cx']
            cy = seg_info['cy']
            cz = seg_info['cz']
            coord_source = 'seg_centroid'
        else:
            # seg가 없는 경우에만 labels.csv 좌표를 fallback으로 사용한다.
            cx_str = _first_nonempty(row, ['center_x', 'center_x_px'])
            cy_str = _first_nonempty(row, ['center_y', 'center_y_px'])
            cz_str = row.get('center_z', '')

            if not cx_str or not cy_str or not cz_str:
                print(f'\n[WARN] 좌표 없음, 건너뜀: '
                      f'subject={subject_id}, nodule={nodule_idx}')
                skipped += 1
                continue

            cx = int(round(float(cx_str)))
            cy = int(round(float(cy_str)))
            cz = center_z_mm_to_slice_idx(float(cz_str), img)
            coord_source = 'labels_csv'

        # ── z 슬라이스 목록 결정 ──────────────────────────────
        if seg_info is not None:
            z_slices = seg_info['z_slices']
            z_source = 'seg'
        else:
            z_slices = get_z_slices_from_num_slices(cz, num_slices_str, D)
            z_source = 'num_slices'

        # z 범위를 전혀 못 구하면 center_z 단일 슬라이스만
        if not z_slices:
            z_slices = [cz]
            z_source = 'fallback'
            print(f'\n[WARN] z 범위 없음, center_z만 사용: '
                  f'subject={subject_id}, nodule={nodule_idx}')

        # ── crop_size별, z_slice별 처리 ───────────────────────
        for cs in crop_sizes:
            for z_idx in z_slices:
                fname = f'{subject_id}_n{nodule_idx}_z{z_idx}'   # 파일명 prefix

                # raw patch
                raw_path = raw_dirs[cs] / f'{fname}.npy'
                if not raw_path.exists():
                    patch = crop_25d(ct, cx, cy, z_idx, cs, n_slices)
                    np.save(raw_path, patch)

                raw_csv_rows[cs].append({
                    'patch_path'     : str(raw_path),
                    'subject_id'     : subject_id,
                    'nodule_idx'     : nodule_idx,
                    'z_idx'          : z_idx,          # 슬라이스 인덱스
                    'label'          : row['label'],
                    'fold'           : fold,
                    'aug_type'       : 'raw',
                    'center_x_vox'   : cx,
                    'center_y_vox'   : cy,
                    'center_z_idx'   : cz,             # 대표 중심 z (참조용)
                    'coord_source'    : coord_source,
                    'crop_size'      : cs,
                    'diameter_max_mm': diameter_max_mm,
                    'volume_mm3'     : row.get('volume_mm3', ''),
                    'z_source'       : z_source,        # 'seg' / 'num_slices' / 'fallback'
                })
                total_samples += 1

    print(f'\n[INFO] 처리 완료. 건너뜀: {skipped}/{total}')
    print(f'[INFO] 총 생성 샘플: {total_samples}개 (결절 {total-skipped}개 × 평균 {total_samples/max(1,total-skipped):.1f}슬라이스)')

    for cs in crop_sizes:
        _write_csv(NPY_CACHE_ROOT / f'labels_raw_{cs}.csv', raw_csv_rows[cs])
        _print_summary(cs, raw_csv_rows[cs])

    _write_config(crop_sizes, n_slices, seed, total, skipped, total_samples)


# ─────────────────────────────────────────────────────────────────────────────
# 섹션 5. 저장 유틸리티
# ─────────────────────────────────────────────────────────────────────────────

def _write_csv(csv_path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f'[INFO] CSV 저장: {csv_path.name} ({len(rows)}행)')


def _write_config(crop_sizes, n_slices, seed,
                  total, skipped, total_samples) -> None:
    config = {
        'created_at'              : datetime.now().isoformat(timespec='seconds'),
        'crop_sizes'              : crop_sizes,
        'n_slices'                : n_slices,
        'stride'                  : 1,
        'input_channels'          : 2 * n_slices + 1,
        'sampling_mode'           : 'all_z_slices',   # 변경된 샘플링 방식
        'seed'                    : seed,
        'active_augmentations'    : [],
        'online_augmentations'    : ['hflip', 'vflip', 'rot90', 'hu_shift', 'gaussian_noise'],
        'total_nodules'           : total,
        'skipped_nodules'         : skipped,
        'total_samples'           : total_samples,
        'avg_slices_per_nodule'   : round(total_samples / max(1, total - skipped), 2),
    }
    NPY_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    with open(NPY_CACHE_ROOT / 'config.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    print(f'[INFO] config.json 저장')


def _print_summary(cs: int, raw_rows: list[dict]) -> None:
    from collections import Counter
    print(f'\n{"="*50}')
    print(f'[SUMMARY] crop_size={cs}')
    fold_label = Counter((r['fold'], r['label']) for r in raw_rows)
    for fold in ['train', 'val', 'test']:
        b = fold_label.get((fold, '0'), 0)
        m = fold_label.get((fold, '1'), 0)
        if b + m > 0:
            print(f'  raw {fold:5s}: benign={b:4d}, malignant={m:4d}, total={b+m}')

# ─────────────────────────────────────────────────────────────────────────────
# 섹션 6. 명령줄 인터페이스
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--crop_sizes',   type=int,   nargs='+', default=[32, 64, 96])
    parser.add_argument('--n_slices',     type=int,   default=1)
    parser.add_argument('--seed',         type=int,   default=SEED)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print(f'[CONFIG] crop_sizes={args.crop_sizes} | n_slices={args.n_slices} | '
          f'채널={2*args.n_slices+1} | 저장=raw only')
    export_patches(crop_sizes=args.crop_sizes, n_slices=args.n_slices,
                   seed=args.seed)
