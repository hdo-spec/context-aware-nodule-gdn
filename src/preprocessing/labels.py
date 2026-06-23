# src/preprocessing/labels.py
#
# ─── 역할 ────────────────────────────────────────────────────────────────────
#   nodule_info_clean.json → labels.csv 생성
#
# ─── 분류 기준 ───────────────────────────────────────────────────────────────
#   판독자별 악성도 점수(1~5점)의 평균을 계산하여 분류
#     mean < 3.0  →  0 (양성, benign)
#     mean == 3.0 →  None (불확실, CSV 제외)
#     mean > 3.0  →  1 (악성, malignant)
#
# ─── 변경 사항 ───────────────────────────────────────────────────────────────
#   [추가] series_idx 컬럼
#
#   [왜 series_idx가 필요한가?]
#     export_nifti.py는 환자당 시리즈가 2개 이상이면 서브폴더로 저장:
#       시리즈 1개: nifti/{subject_id}/ct_norm.nii.gz
#       시리즈 N개: nifti/{subject_id}/series_0/ct_norm.nii.gz
#                  nifti/{subject_id}/series_1/ct_norm.nii.gz
#     이 인덱스는 export_nifti의 valid_metas 리스트 순서와 동일.
#
#     series_idx 없이 export_patches를 실행하면:
#       다중 시리즈 환자는 루트에 ct_norm.nii.gz가 없으므로 전부 스킵됨.
#       → 해당 환자의 모든 결절이 누락됨.
#
#   [series_idx 결정 방법]
#     nodule['series_uid'] → ct_meta 리스트에서 같은 series_uid 위치 탐색
#     ct_meta에 series_uid가 없는 경우 → series_idx = 0 (단일 시리즈 가정)
#
# ─── 출력 컬럼 ───────────────────────────────────────────────────────────────
#   subject_id, nodule_idx, series_uid, series_idx,
#   malignancy_scores, num_readers, mean_score, std_score,
#   label,
#   center_x, center_y, center_z,
#   center_x_px, center_y_px,   ← [추가] pixel 좌표 원본 (export_patches 사용)
#   num_slices, diameter_max_mm,
#   volume_mm3                   ← [추가] seg mask voxel 카운팅 (1mm iso → mm³)
#
# ─── 실행 ────────────────────────────────────────────────────────────────────
#   python -m src.preprocessing.labels

import os
import csv
import json
import numpy as np
import nibabel as nib       # seg.nii.gz 로드 (volume_mm3 계산용)
from pathlib import Path    # seg 경로 계산용

from src.configs.config import PROCESSED_ROOT


CLEAN_JSON_PATH  = os.path.join(PROCESSED_ROOT, 'nodule_info_clean.json')
LABELS_CSV_PATH  = os.path.join(PROCESSED_ROOT, 'labels.csv')
MALIGNANCY_THRESHOLD = 3.0


# ══════════════════════════════════════════════════════════════════════════════
# 섹션 1. 분류 함수
# ══════════════════════════════════════════════════════════════════════════════

def get_label(malignancy_scores: list):
    """
    판독자 점수 리스트 → 레이블 반환.

        mean < 3.0  → 0    (양성)
        mean == 3.0 → None (불확실, CSV에서 제외)
        mean > 3.0  → 1    (악성)
    """
    mean = float(np.mean(malignancy_scores))
    if mean < MALIGNANCY_THRESHOLD:   return 0     # 양성
    if mean == MALIGNANCY_THRESHOLD:  return None  # 불확실 → 제외
    return 1                                        # 악성


# ══════════════════════════════════════════════════════════════════════════════
# 섹션 2. series_idx 결정
# ══════════════════════════════════════════════════════════════════════════════

def resolve_series_idx(nodule_series_uid: str, ct_meta_list: list) -> int:
    """
    nodule의 series_uid → ct_meta 리스트 인덱스 반환.

    export_nifti.py는 valid_metas(ct_meta에서 error 없는 항목) 기준으로
    series_0, series_1 폴더를 만든다.
    labels.py도 동일하게 error 없는 항목만 필터링한 뒤 인덱스를 계산해야
    export_nifti의 폴더 번호와 일치한다.

    [예시]
        ct_meta_list = [
            {'series_uid': 'A', ...},           # error 없음 → valid_idx=0
            {'series_uid': 'B', 'error': ...},  # error 있음 → 제외
            {'series_uid': 'C', ...},           # error 없음 → valid_idx=1
        ]
        nodule series_uid='C' → series_idx=1 → 폴더 series_1/

    [series_uid가 ct_meta에 없는 경우]
        단일 시리즈이면 ct_meta에 series_uid 필드가 없을 수 있음.
        이때는 series_idx=0 반환 (루트 또는 series_0/).

    Args:
        nodule_series_uid: nodule['series_uid']
        ct_meta_list:      patient['ct_meta'] 전체 리스트

    Returns:
        valid_metas 기준 인덱스 (0-based)
    """
    # error 없는 meta만 추출 (export_nifti의 valid_metas와 동일 로직)
    valid_metas = [m for m in ct_meta_list if 'error' not in m]

    if not valid_metas:
        return 0

    # series_uid로 매칭
    for idx, meta in enumerate(valid_metas):
        if meta.get('series_uid') == nodule_series_uid:
            return idx

    # 매칭 실패: 단일 시리즈이거나 uid 불일치 → 0 반환
    return 0


# ══════════════════════════════════════════════════════════════════════════════
# 섹션 2-1. seg mask에서 volume_mm3 계산
# ══════════════════════════════════════════════════════════════════════════════

def compute_volume_mm3(nifti_path: str, nodule_idx: int) -> float | None:
    """
    seg.nii.gz에서 nodule_idx에 해당하는 결절의 부피를 계산.

    계산 원리:
      1mm isotropic 리샘플 완료 상태이므로 voxel 수 = 부피(mm³).

    label 매핑 원칙:
      export_nifti.py는 series 내 enumerate 순서로 label을 부여함.
        for i, nodule in enumerate(nodules_s): label = i + 1
      따라서 JSON의 nodule_idx와 seg label이 직접 대응하지 않음.
      예: series 내 nodule_idx=[3,7,12] → seg label=[1,2,3]
      → seg_manifest.json에서 nodule_idx → label 매핑을 읽어야 함.

    seg 경로:
      단일/다중 시리즈 모두: Path(nifti_path).parent / 'seg.nii.gz'
      (다중 시리즈는 series_N 폴더 안에 ct_norm.nii.gz와 seg.nii.gz가 함께 있음)

    Args:
        nifti_path: ct_norm.nii.gz 절대경로
        nodule_idx: JSON의 nodule_idx

    Returns:
        float: 부피 (mm³). seg 없거나 해당 nodule이 seg에 없으면 None.
    """
    seg_path      = Path(nifti_path).parent / 'seg.nii.gz'
    manifest_path = Path(nifti_path).parent / 'seg_manifest.json'

    if not seg_path.exists() or not manifest_path.exists():
        return None

    try:
        # seg_manifest.json에서 nodule_idx → seg label 매핑
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)

        label = None
        for item in manifest.get('series', []):
            indices = item.get('nodule_indices', [])
            labels  = item.get('labels', [])
            for nidx, lbl in zip(indices, labels):
                if nidx == nodule_idx:
                    label = lbl
                    break
            if label is not None:
                break

        if label is None:
            return None   # 이 결절은 seg mask에 없음 (annotation 없음 등)

        seg_array   = nib.load(str(seg_path)).get_fdata(dtype=np.float32)
        voxel_count = int(np.sum(seg_array == label))
        return float(voxel_count)   # 1mm³/voxel → mm³

    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# 섹션 3. NIfTI 경로 결정
# ══════════════════════════════════════════════════════════════════════════════

def resolve_nifti_path(subject_id: str, series_idx: int,
                       ct_meta_list: list, nifti_root: str) -> str:
    """
    subject_id + series_idx → ct_norm.nii.gz 절대경로 반환.

    export_nifti.py 저장 규칙:
        valid_metas 개수 == 1 → {nifti_root}/{subject_id}/ct_norm.nii.gz
        valid_metas 개수 >= 2 → {nifti_root}/{subject_id}/series_{idx}/ct_norm.nii.gz

    이 함수는 labels.csv의 nifti_path 컬럼에 저장되며,
    export_patches가 이 경로를 직접 사용한다.
    """
    valid_metas = [m for m in ct_meta_list if 'error' not in m]
    n_series    = len(valid_metas)

    if n_series <= 1:
        # 단일 시리즈: 루트 경로
        return os.path.join(nifti_root, subject_id, 'ct_norm.nii.gz')
    else:
        # 다중 시리즈: series_N 서브폴더
        return os.path.join(nifti_root, subject_id,
                            f'series_{series_idx}', 'ct_norm.nii.gz')


def load_seg_manifest_series_map(subject_id: str, nifti_root: str) -> dict:
    """
    Return nodule_idx -> series tag from export_nifti.py's seg_manifest.json.

    This is the source of truth after export_nifti filters duplicate or
    annotation-less series.  Falling back to series_uid alone can point labels
    at a skipped folder such as series_1.
    """
    manifest_path = os.path.join(nifti_root, subject_id, 'seg_manifest.json')
    if not os.path.exists(manifest_path):
        return {}

    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)

    series_map = {}
    for item in manifest.get('series', []):
        series_tag = item.get('series')
        for nodule_idx in item.get('nodule_indices', []):
            series_map[nodule_idx] = series_tag
    return series_map


def resolve_nifti_path_from_manifest(subject_id: str,
                                     series_tag,
                                     nifti_root: str) -> str:
    if series_tag is None:
        return os.path.join(nifti_root, subject_id, 'ct_norm.nii.gz')
    return os.path.join(nifti_root, subject_id, series_tag, 'ct_norm.nii.gz')


def series_idx_from_tag(series_tag, fallback_idx: int) -> int:
    if isinstance(series_tag, str) and series_tag.startswith('series_'):
        try:
            return int(series_tag.split('_', 1)[1])
        except ValueError:
            return fallback_idx
    return fallback_idx


# ══════════════════════════════════════════════════════════════════════════════
# 섹션 4. 메인 처리 함수
# ══════════════════════════════════════════════════════════════════════════════

def make_labels(
    clean_json_path: str = CLEAN_JSON_PATH,
    output_csv_path: str = LABELS_CSV_PATH,
) -> dict:
    """
    nodule_info_clean.json → labels.csv 생성.

    처리 순서:
        [1] JSON 로드
        [2] 결절별 레이블 + series_idx + nifti_path 계산
        [3] CSV 저장
        [4] 통계 출력
    """

    # ── [1] JSON 로드 ────────────────────────────────────────────────────────
    if not os.path.exists(clean_json_path):
        raise FileNotFoundError(f'파일 없음: {clean_json_path}')

    with open(clean_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    nifti_root = os.path.join(PROCESSED_ROOT, 'nifti')
    print(f'[1/4] JSON 로드: {len(data)}명 환자')

    rows = []
    total_nodules = 0
    scored_nodules = 0
    no_score_count = 0
    uncertain_count = 0
    multi_series_count = 0   # 다중 시리즈 환자 수 (검증용)
    # seg 캐시: 직전 1개만 유지 (같은 환자 결절은 연속 처리되므로 충분)
    # 전체 캐시는 875명 × ~39MB = ~34GB → OOM 위험
    _cached_seg_path: str | None = None
    _cached_seg: dict | None = None

    for subject_id, patient in data.items():

        ct_meta_list = patient.get('ct_meta', [])
        valid_metas  = [m for m in ct_meta_list if 'error' not in m]
        n_series     = len(valid_metas)
        manifest_series_map = load_seg_manifest_series_map(subject_id, nifti_root)

        if n_series > 1:
            multi_series_count += 1

        for nodule in patient.get('nodules', []):
            total_nodules += 1

            scores = nodule.get('malignancy', [])
            if not scores:
                no_score_count += 1
                continue
            scored_nodules += 1

            # 악성도 통계
            mean_score = float(np.mean(scores))
            std_score  = float(np.std(scores))
            label      = get_label(scores)
            if label is None:           # mean == 3.0 → 불확실, CSV에서는 제외
                uncertain_count += 1
                continue

            # 위치/크기 정보
            derived      = nodule.get('derived', {})
            center_z     = derived.get('center_z')
            num_slices   = derived.get('num_slices')
            diameter_max = derived.get('diameter_max_mm')

            # series_idx 먼저 결정 (pixel_spacing 읽기에 필요)
            nodule_idx = nodule.get('nodule_idx', -1)
            nodule_series_uid = nodule.get('series_uid', '')
            fallback_series_idx = resolve_series_idx(nodule_series_uid, ct_meta_list)
            series_tag = manifest_series_map.get(nodule_idx)
            series_idx = series_idx_from_tag(series_tag, fallback_series_idx)

            center_x_px = derived.get('center_x')
            center_y_px = derived.get('center_y')

            # 해당 시리즈의 pixel_spacing 읽기
            ps_x, ps_y = 1.0, 1.0
            if center_x_px is not None and series_idx < len(valid_metas):
                meta = valid_metas[series_idx]
                ps_x = meta.get('pixel_spacing_x') or 1.0
                ps_y = meta.get('pixel_spacing_y') or 1.0

            center_x = round(center_x_px * ps_x, 4) if center_x_px is not None else None
            center_y = round(center_y_px * ps_y, 4) if center_y_px is not None else None

            if nodule_idx in manifest_series_map:
                nifti_path = resolve_nifti_path_from_manifest(
                    subject_id, series_tag, nifti_root
                )
            else:
                nifti_path = resolve_nifti_path(
                    subject_id, series_idx, ct_meta_list, nifti_root
                )

            # volume_mm3: seg mask voxel 카운팅 (1mm iso → mm³)
            # 직전 seg만 캐싱 → 같은 환자 결절은 반복 로드 없음, 메모리 안전
            _seg_path = str(Path(nifti_path).parent / 'seg.nii.gz')
            if _seg_path != _cached_seg_path:
                _manifest_p = Path(nifti_path).parent / 'seg_manifest.json'
                if Path(_seg_path).exists() and _manifest_p.exists():
                    _cached_seg = {
                        'array'   : nib.load(_seg_path).get_fdata(dtype=np.float32),
                        'manifest': json.load(open(_manifest_p, 'r', encoding='utf-8')),
                    }
                else:
                    _cached_seg = None
                _cached_seg_path = _seg_path

            if _cached_seg is None:
                volume_mm3 = None
            else:
                try:
                    _label = None
                    for _item in _cached_seg['manifest'].get('series', []):
                        for _nidx, _lbl in zip(_item.get('nodule_indices', []), _item.get('labels', [])):
                            if _nidx == nodule_idx:
                                _label = _lbl
                                break
                        if _label is not None:
                            break
                    volume_mm3 = float(int(np.sum(_cached_seg['array'] == _label))) if _label is not None else None
                except Exception:
                    volume_mm3 = None

            rows.append({
                'subject_id'       : subject_id,
                'nodule_idx'       : nodule_idx,
                'series_uid'       : nodule_series_uid,
                'series_idx'       : series_idx,
                'nifti_path'       : nifti_path,   # export_patches가 사용
                'malignancy_scores': ','.join(str(s) for s in scores),
                'num_readers'      : len(scores),
                'mean_score'       : round(mean_score, 4),
                'std_score'        : round(std_score,  4),
                'label'            : label,
                'center_x'         : round(center_x, 4)     if center_x     is not None else '',
                'center_y'         : round(center_y, 4)     if center_y     is not None else '',
                'center_z'         : round(center_z, 4)     if center_z     is not None else '',
                # ── [추가] pixel 좌표 원본 ─────────────────────────────────
                # center_x/y: mm 단위. NIfTI가 1mm isotropic이므로 mm = voxel index.
                # export_patches.py에서 int(round(center_x)) 로 voxel index 변환.
                # center_slice_idx는 export_patches.py에서 NIfTI affine으로 계산.
                'center_x_px'      : round(center_x_px, 4)  if center_x_px  is not None else '',
                'center_y_px'      : round(center_y_px, 4)  if center_y_px  is not None else '',
                'num_slices'       : num_slices              if num_slices   is not None else '',
                'diameter_max_mm'  : round(diameter_max, 4) if diameter_max is not None else '',
                # ── [추가] volume_mm3 ──────────────────────────────────────
                # seg mask voxel 카운팅. 1mm isotropic → voxel 수 = mm³.
                # Fleischner Society subgroup 분석 기준으로 사용.
                # seg.nii.gz 없으면 None.
                'volume_mm3'       : round(volume_mm3, 2)    if volume_mm3   is not None else '',
            })

    print(f'[2/4] 레이블 계산 완료: {total_nodules}개 결절')
    print(f'      CSV 대상(양성+악성): {len(rows)}개')
    print(f'      불확실(mean=3.0): {uncertain_count}개')
    if no_score_count:
        print(f'      malignancy 점수 없음: {no_score_count}개')
    print(f'      다중 시리즈 환자: {multi_series_count}명')

    # ── [3] CSV 저장 ─────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    fieldnames = [
        'subject_id', 'nodule_idx', 'series_uid', 'series_idx', 'nifti_path',
        'malignancy_scores', 'num_readers', 'mean_score', 'std_score',
        'label',
        'center_x', 'center_y', 'center_z',
        'center_x_px', 'center_y_px',   # [추가] pixel 좌표 원본 (export_patches 사용)
        'num_slices', 'diameter_max_mm',
        'volume_mm3',                    # [추가] seg mask 기반 부피 (mm³)
    ]

    with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f'[3/4] CSV 저장: {output_csv_path}')

    # ── [4] 통계 ─────────────────────────────────────────────────────────────
    total    = len(rows)                               # CSV에 저장된 결절 수 (불확실 제외)
    benign   = sum(1 for r in rows if r['label'] == 0) # 양성
    malign   = sum(1 for r in rows if r['label'] == 1) # 악성
    excluded = uncertain_count                         # mean == 3.0

    # nifti_path 존재 여부 검사 (labels.csv 생성 시점에 미리 경고)
    missing_nifti = [r for r in rows if not os.path.exists(r['nifti_path'])]
    if missing_nifti:
        missing_subjects = sorted(set(r['subject_id'] for r in missing_nifti))
        print(f'\n  [경고] nifti_path가 존재하지 않는 결절: {len(missing_nifti)}개')
        print(f'  해당 subject_id ({len(missing_subjects)}명):')
        for sid in missing_subjects:
            cnt = sum(1 for r in missing_nifti if r['subject_id'] == sid)
            print(f'    {sid}: {cnt}개 결절 → export_patches 시 스킵됨')

    total_original = total_nodules
    denom = total_original if total_original else 1

    print(f'\n[4/4] 통계')
    print(f'      원본 결절 수   : {total_original}개')
    print(f'      양성 (label=0) : {benign}개 ({benign/denom*100:.1f}%)')
    print(f'      악성 (label=1) : {malign}개 ({malign/denom*100:.1f}%)')
    print(f'      불확실 (제외)  : {excluded}개 ({excluded/denom*100:.1f}%) ← mean=3.0')
    if no_score_count:
        print(f'      점수 없음       : {no_score_count}개')
    print(f'      CSV 저장       : {total}개 (양성+악성)')
    if malign > 0:
        print(f'      양성:악성      : {benign}:{malign} = {benign/malign:.2f}:1')
    return {
        'total_original': total_original,
        'benign'        : benign,
        'malignant'     : malign,
        'excluded'      : excluded,
        'csv_saved'     : total,
        'ratio'         : round(benign / malign, 4) if malign else 0,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 섹션 5. 실행 진입점
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    make_labels()
