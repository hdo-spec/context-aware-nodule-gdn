# src/preprocessing/export_nifti.py
#
# ─── 역할 ────────────────────────────────────────────────────────────────────
#   nodule_info_clean.json + DICOM 원본 → NIfTI 변환 + segmentation mask 생성
#
# ─── 파이프라인에서의 위치 ────────────────────────────────────────────────────
#   [1] parse_lidc_annotations.py  XML + metadata.csv → nodule_info.json
#   [2] match_dicom.py             nodule_info.json + DICOM 헤더 → nodule_info_clean.json
#   [3] export_nifti.py  ← 현재   nodule_info_clean.json + DICOM → NIfTI + seg mask
#
# ─── 출력 파일 구조 (환자별) ─────────────────────────────────────────────────
#   data/processed/nifti/{subject_id}/
#     ct.nii.gz      ← 1mm 등방성 리샘플 + HU 원본 보존
#     ct_norm.nii.gz ← 1mm 등방성 리샘플 + [0,1] 정규화 (ML 입력용)
#     centroid.csv   ← 결절 centroid mm 좌표
#
# ─── seg mask 좌표 변환 흐름 ─────────────────────────────────────────────────
#   XML polygon: pixel 좌표 (원본 DICOM 기준)
#   → 리샘플 후 voxel 좌표:
#       x_vox = round(x_px × orig_ps_x / 1.0)
#       y_vox = round(y_px × orig_ps_y / 1.0)
#       z_vox = round((z_mm - resampled_origin_z) / resampled_spacing_z)
#
# ─── z 좌표 정합 원칙 ────────────────────────────────────────────────────────
#   LIDC-IDRI 데이터의 일부 케이스에서 XML z_position과 DICOM z 범위 사이에
#   세 가지 단위 불일치가 존재한다:
#     케이스 A — XML z가 deci-mm: parse 단계에서 보정 완료, 정상 매칭 가능
#     케이스 B — DICOM z가 deci-mm: SimpleITK GetOrigin이 raw값 반환, z_range를 /10 보정
#     케이스 C — XML z 부호 반전: 좌표계 방향이 DICOM과 반대, z_position에 -1 적용
#   align_center_z()가 세 케이스를 모두 감지하여 자동 보정한다.
#
# ─── 실행 ────────────────────────────────────────────────────────────────────
#   테스트: python -m src.preprocessing.export_nifti --subjects LIDC-IDRI-0085
#   전체:   python -m src.preprocessing.export_nifti

import os
import json
import argparse
import csv
from typing import Tuple, List, Optional, Dict

import numpy as np
import nibabel as nib
import SimpleITK as sitk
from skimage.draw import polygon as sk_polygon
from tqdm import tqdm

from src.configs.config import PROCESSED_ROOT

CLEAN_JSON_PATH = os.path.join(PROCESSED_ROOT, 'nodule_info_clean.json')
NIFTI_SAVE_DIR  = os.path.join(PROCESSED_ROOT, 'nifti')
VIOLATION_PATH  = os.path.join(PROCESSED_ROOT, 'coord_violations.json')
SEG_EMPTY_PATH  = os.path.join(PROCESSED_ROOT, 'seg_empty_subjects.json')

HU_MIN = -1000   # 공기 HU (클리핑 하한)
HU_MAX =  400    # 연조직/뼈 경계 (클리핑 상한)


# ══════════════════════════════════════════════════════════════════════════════
# 섹션 1. JSON 로드
# ══════════════════════════════════════════════════════════════════════════════

def load_clean_json(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"[1/6] JSON 로드: {len(data)}명")
    return data


# ══════════════════════════════════════════════════════════════════════════════
# 섹션 2. z 좌표 정합
# ══════════════════════════════════════════════════════════════════════════════

def _z_overlap_ratio(z_min_a: float, z_max_a: float,
                     z_min_b: float, z_max_b: float) -> float:
    """
    두 z 범위의 겹침 비율 계산.

    반환: overlap / min(range_a, range_b)
      1.0에 가까울수록 한쪽 범위가 다른 쪽에 거의 포함됨.
      0.9 이상이면 재촬영(같은 부위를 다시 스캔한 것)으로 판단.
      재촬영 중복 시리즈 제거부분에서 사용
    """
    overlap = max(0.0, min(z_max_a, z_max_b) - max(z_min_a, z_min_b))
    range_a = z_max_a - z_min_a
    range_b = z_max_b - z_min_b
    if range_a <= 0 or range_b <= 0:
        return 0.0
    return overlap / min(range_a, range_b) # 짧은 시리즈가 긴 시리즈에 완전히 포함 =


def get_z_range_from_image(image: sitk.Image) -> Tuple[float, float]:
    """
    SimpleITK Image에서 z_min, z_max 계산.
    
    CT볼륨이 어떤 방향으로 저장됐든 실제 물리적 Z범위를 계산하는 함수
    이 로직을 안하면 center_z가 범위 밖으로 배정되어 결절이 잘못된 series로 배정될 수 있음

    GetDirection()에 dicom의 z축 정보 저장
    GetDirection()[8]이 -1이면 z축 반전(Superior→Inferior) 케이스.
    반전 케이스에서는 origin_z가 z_max가 되므로 min/max를 올바르게 계산.
    """
    oz    = image.GetOrigin()[2]
    sp    = image.GetSpacing()[2]
    nz    = image.GetSize()[2]
    z_dir = image.GetDirection()[8]   # 정방향: +1.0, 반전: -1.0 
    z_last = oz + z_dir * sp * (nz - 1)
    return min(oz, z_last), max(oz, z_last)


def align_center_z(center_z: float, z_min: float,
                   z_max: float) -> Tuple[float, float, float, bool]:
    """
    center_z와 z_range 사이의 단위 불일치를 감지하여 보정.

    반환: (보정된 center_z, 보정된 z_min, 보정된 z_max, z_position_flip)
    케이스 C(부호 반전)이면 flip=True, 케이스 B(DICOM deci-mm)이면 z_range /10 보정.
    z_range 중심까지 가장 가까운 후보를 선택. margin = max(range*10%, 5mm).
    """
    margin   = max(abs(z_max - z_min) * 0.1, 5.0)
    z_center = (z_min + z_max) / 2.0

    # 1. 이미 범위 내 → 보정 불필요
    if z_min - margin <= center_z <= z_max + margin:
        return center_z, z_min, z_max, False

    cz_neg   = -center_z
    cz_fixed = center_z / 10.0
    zm_fixed = z_min / 10.0
    zx_fixed = z_max / 10.0

    neg_in   = z_min   - margin <= cz_neg   <= z_max   + margin
    fixed_in = z_min   - margin <= cz_fixed <= z_max   + margin
    dicom_in = zm_fixed - margin <= center_z <= zx_fixed + margin

    # 범위 내 후보 중 z_range 중심과 가장 가까운 것 선택
    candidates = []
    if neg_in:
        candidates.append((abs(cz_neg   - z_center), 'neg'))
    if fixed_in:
        candidates.append((abs(cz_fixed - z_center), 'fixed'))
    if dicom_in:
        candidates.append((abs(center_z - z_center), 'dicom'))

    if not candidates:
        return center_z, z_min, z_max, False   # 해결 불가 → WARN 출력됨

    best = min(candidates, key=lambda x: x[0])[1]

    if best == 'neg':
        return cz_neg, z_min, z_max, True    # 케이스 C: flip=True
    elif best == 'fixed':
        return cz_fixed, z_min, z_max, False  # 케이스 A
    else:
        return center_z, zm_fixed, zx_fixed, False  # 케이스 B


def _in_z_range(z: float, z_min: float, z_max: float, margin: float) -> bool:
    return z_min - margin <= z <= z_max + margin


def _z_range_distance(z: float, z_min: float, z_max: float) -> float:
    if z_min <= z <= z_max:
        return 0.0
    return min(abs(z - z_min), abs(z - z_max))


def assign_nodule_to_series(center_z: float,
                            series_info: List[dict]) -> Tuple[Optional[int], bool, str]:
    """
    Return (series_index, z_position_flip, match_reason).
    직접 매칭(보정 없이 범위 내) 우선, 그 다음 align 보정 후 매칭, 최후 fallback.
    """
    direct_matches = []
    for i, si in enumerate(series_info):
        margin = si['meta']['z_spacing']
        if _in_z_range(center_z, si['z_min'], si['z_max'], margin):
            center = (si['z_min'] + si['z_max']) / 2.0
            direct_matches.append((abs(center_z - center), i))

    if direct_matches:
        return min(direct_matches, key=lambda x: x[0])[1], False, 'direct'

    aligned_matches = []
    for i, si in enumerate(series_info):
        margin = si['meta']['z_spacing']
        cz, zm, zx, flip = align_center_z(center_z, si['z_min'], si['z_max'])
        if _in_z_range(cz, zm, zx, margin):
            center = (zm + zx) / 2.0
            aligned_matches.append((abs(cz - center), i, flip))

    if aligned_matches:
        _, i, flip = min(aligned_matches, key=lambda x: x[0])
        return i, flip, 'aligned'

    if not series_info:
        return None, False, 'none'

    closest = min(
        range(len(series_info)),
        key=lambda i: _z_range_distance(center_z,
                                        series_info[i]['z_min'],
                                        series_info[i]['z_max'])
    )
    return closest, False, 'fallback'


# ── 섹션 3. DICOM 로드 ──────────────────────────────────────────────────────────

def load_dicom_series(dicom_dir: str,
                      subject_id: str = '') -> Optional[sitk.Image]:
    """
    DICOM 폴더를 SimpleITK로 로드하여 3D 볼륨 반환.

    슬라이스 불균일 감지:
      nonuniformity = |실제 z 범위 - 예상 z 범위|
      예상 범위 = step[0→1] × (n_slices - 1)
      > 1.0mm → 심각 경고 (누락 슬라이스 의심)
      > 0.1mm → 경고 (부동소수점 오차 수준 초과)
    """
    try:
        reader     = sitk.ImageSeriesReader()
        file_names = reader.GetGDCMSeriesFileNames(dicom_dir)
        if not file_names:
            return None

        if len(file_names) > 1:
            fr = sitk.ImageFileReader()

            fr.SetFileName(file_names[0])
            fr.ReadImageInformation()
            z0 = float(fr.GetMetaData('0020|0032').split('\\')[-1])

            fr.SetFileName(file_names[1])
            fr.ReadImageInformation()
            z1 = float(fr.GetMetaData('0020|0032').split('\\')[-1])

            fr.SetFileName(file_names[-1])
            fr.ReadImageInformation()
            zn = float(fr.GetMetaData('0020|0032').split('\\')[-1])

            expected_step  = abs(z1 - z0)
            actual_range   = abs(zn - z0)
            expected_range = expected_step * (len(file_names) - 1)
            nonuniformity  = abs(actual_range - expected_range)

            if nonuniformity > 1.0:
                print(f"    [WARN] {subject_id}: 슬라이스 불균일 심각 "
                      f"nonuniformity={nonuniformity:.3f}mm (누락 슬라이스 의심)")
            elif nonuniformity > 0.1:
                print(f"    [WARN] {subject_id}: 슬라이스 불균일 "
                      f"nonuniformity={nonuniformity:.3f}mm")

        reader.SetFileNames(file_names)
        return reader.Execute()

    except Exception as e:
        print(f"    [ERROR] DICOM 로드 실패: {dicom_dir}\n      {e}")
        return None


# ── 섹션 4. CT 전처리 ───────────────────────────────────────────────────────────

def resample_image(image: sitk.Image,
                   new_spacing: tuple = (1.0, 1.0, 1.0),
                   interpolator=sitk.sitkLinear) -> sitk.Image:
    """
    CT 볼륨을 (1mm, 1mm, 1mm) 등방성으로 리샘플.

    HU 원본 상태에서 보간해야 물리적으로 의미있는 값이 유지됨.
    정규화 후 보간하면 경계 패딩값(HU_MIN)이 섞여 왜곡 발생.

    [새 볼륨 크기 공식]
      new_size[i] = round(orig_size[i] × orig_spacing[i] / new_spacing[i])
    """
    orig_spacing = image.GetSpacing()
    orig_size    = image.GetSize()

    if all(abs(orig_spacing[i] - new_spacing[i]) < 0.01 for i in range(3)):
        return image   # 이미 목표 spacing이면 불필요한 연산 방지

    new_size = [
        int(round(orig_size[i] * orig_spacing[i] / new_spacing[i]))
        for i in range(3)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(HU_MIN)   # 경계 밖 패딩 = 공기
    resampler.SetInterpolator(interpolator)

    return resampler.Execute(image)


def normalize_hu(image: sitk.Image,
                 hu_min: float = HU_MIN,
                 hu_max: float = HU_MAX) -> sitk.Image:
    """
    HU 클리핑 후 [0.0, 1.0] 정규화.

    GetImageFromArray()는 spacing/origin을 초기화하므로
    CopyInformation()으로 리샘플된 볼륨의 좌표계를 복사해야
    seg mask와 좌표계가 일치함.
    """
    arr  = sitk.GetArrayFromImage(image).astype(np.float32)
    arr  = np.clip(arr, hu_min, hu_max)
    arr  = (arr - hu_min) / (hu_max - hu_min)
    norm = sitk.GetImageFromArray(arr)
    norm.CopyInformation(image)   # ★ spacing/origin/direction 복사 필수
    return norm

# ══════════════════════════════════════════════════════════════════════
# SEGMENTATION MASK 생성
# ══════════════════════════════════════════════════════════════════════

def create_segmentation_mask(
    resampled_image: sitk.Image,
    nodules: list,
    ps_x: float,
    ps_y: float,
    z_flip: bool = False,
):
    """
    polygon ROI → 3D label mask

    label:
        0 = background
        1 = nodule 0
        2 = nodule 1
        ...
    """

    seg = np.zeros(
        (
            resampled_image.GetSize()[2],
            resampled_image.GetSize()[1],
            resampled_image.GetSize()[0],
        ),
        dtype=np.uint16,
    )

    origin = resampled_image.GetOrigin()
    spacing = resampled_image.GetSpacing()

    manifest = []

    for nodule_idx, nodule in enumerate(nodules):

        label = nodule_idx + 1
        drawn = False

        for _, rois in nodule.get("rois", {}).items():

            for roi in rois:

                poly = roi.get("polygon", [])

                if len(poly) < 3:
                    continue

                z_mm = roi["z_position"]

                if z_flip:
                    z_mm = -z_mm

                z_vox = int(
                    round(
                        (z_mm - origin[2]) / spacing[2]
                    )
                )

                if z_vox < 0 or z_vox >= seg.shape[0]:
                    continue

                xs = [
                    int(round(p["x"] * ps_x / spacing[0]))
                    for p in poly
                ]

                ys = [
                    int(round(p["y"] * ps_y / spacing[1]))
                    for p in poly
                ]

                rr, cc = sk_polygon(
                    ys,
                    xs,
                    shape=seg[z_vox].shape,
                )

                seg[z_vox, rr, cc] = label
                drawn = True

        if drawn:
            manifest.append(
                {
                    "nodule_idx": nodule.get("nodule_idx"),
                    "label": label,
                }
            )

    seg_img = sitk.GetImageFromArray(seg)
    seg_img.CopyInformation(resampled_image)

    return seg_img, manifest

# ══════════════════════════════════════════════════════════════════════════════
# ── 섹션 6. 좌표 변환 유틸 + 검증 ──────────────────────────────────────────────

def compute_mm_coords(nodule: dict, ps_x: float, ps_y: float) -> dict:
    """결절 polygon 좌표를 pixel → mm 변환하여 반환 (nodule_info_clean.json 업데이트용)."""
    mm_coords = {}
    for reader_key, rois in nodule.get('rois', {}).items():
        mm_rois = []
        for roi in rois:
            polygon = roi.get('polygon', [])
            if len(polygon) < 3:
                continue
            poly_mm = [{'x': round(pt['x'] * ps_x, 4), 'y': round(pt['y'] * ps_y, 4)}
                       for pt in polygon]
            mm_rois.append({
                'z_position_mm': roi['z_position'],
                'centroid_x_mm': round(float(np.mean([p['x'] for p in poly_mm])), 4),
                'centroid_y_mm': round(float(np.mean([p['y'] for p in poly_mm])), 4),
                'polygon_mm'   : poly_mm,
            })
        mm_coords[reader_key] = mm_rois
    return mm_coords


def check_coord_bounds(nodule: dict, ct_meta: dict, subject_id: str) -> list:
    """
    polygon 좌표가 원본 CT 해상도(rows × cols) 범위를 벗어나는지 검사.
    데이터 품질 검증용 — 원본 pixel 기준.
    """
    rows = ct_meta.get('rows')
    cols = ct_meta.get('cols')
    if rows is None or cols is None:
        return []

    violations = []
    for reader_key, rois in nodule.get('rois', {}).items():
        for roi in rois:
            for pt in roi.get('polygon', []):
                x, y = pt['x'], pt['y']
                x_ok = 0 <= x <= cols - 1
                y_ok = 0 <= y <= rows - 1
                if not x_ok or not y_ok:
                    violations.append({
                        'subject_id': subject_id,
                        'nodule_idx': nodule.get('nodule_idx'),
                        'reader'    : reader_key,
                        'z_position': roi['z_position'],
                        'x_px': x, 'y_px': y,
                        'ct_cols': cols, 'ct_rows': rows,
                        'x_out': not x_ok, 'y_out': not y_ok,
                    })
    return violations


# ── 섹션 7. centroid.csv 생성 ───────────────────────────────────────────────────

def build_centroid_csv(nodules: list, ps_x: float, ps_y: float,
                       save_path: str) -> None:
    """결절 centroid를 mm 좌표로 변환하여 CSV 저장. nodule 없으면 미생성."""
    centroid_list = []
    for nodule in nodules:
        derived  = nodule.get('derived', {})
        cx_px    = derived.get('center_x')
        cy_px    = derived.get('center_y')
        cz_mm    = derived.get('center_z')
        if cx_px is None or cy_px is None or cz_mm is None:
            continue
        mal_list = nodule.get('malignancy', [])
        centroid_list.append({
            'nodule_idx'       : nodule.get('nodule_idx'),
            'centroid_x_mm'    : round(cx_px * ps_x, 4),
            'centroid_y_mm'    : round(cy_px * ps_y, 4),
            'centroid_z_mm'    : round(float(cz_mm), 4),
            'malignancy_median': float(np.median(mal_list)) if mal_list else -1.0,
            'num_readers'      : len(nodule.get('rois', {})),
        })
    if centroid_list:
        fieldnames = ['nodule_idx', 'centroid_x_mm', 'centroid_y_mm',
                      'centroid_z_mm', 'malignancy_median', 'num_readers']
        with open(save_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(centroid_list)


# ══════════════════════════════════════════════════════════════════════════════
# 섹션 8. 환자 단위 처리
# ══════════════════════════════════════════════════════════════════════════════

def process_patient(subject_id: str, patient: dict,
                    save_dir: str) -> Tuple[list, dict]:
    """
    환자 1명 전체 처리.

    [A] valid_metas(error 없는 시리즈) 추출
    [B] DICOM 로드 → z 범위 계산 → [B-1] 재촬영 중복 제거 → [B-2] annotation 없는 시리즈 제거
    [C] assign_nodule_to_series()로 nodule → series 배정
    [D] 시리즈별 ct/ct_norm/centroid 생성.
    """
    all_violations = []
    all_mm_updates = {}

    # [A] valid_metas
    ct_meta_list = patient.get('ct_meta', [])
    valid_metas  = [m for m in ct_meta_list if 'error' not in m]
    if not valid_metas:
        return [], {}

    subj_dir = os.path.join(save_dir, subject_id)
    os.makedirs(subj_dir, exist_ok=True)
    nodules = patient.get('nodules', [])

    # [B] series_info 구성
    series_info = []
    for s_idx, meta in enumerate(valid_metas):
        image = load_dicom_series(meta['dicom_dir'], subject_id)
        if image is None:
            continue

        z_min, z_max = get_z_range_from_image(image)
        series_info.append({
            'orig_idx': s_idx,
            'meta' : meta,
            'image': image,
            'z_min': z_min,
            'z_max': z_max,
        })

    if not series_info:
        return [], {}

    # [B-1] 재촬영 중복 시리즈 제거 (z_overlap >= 0.9)
    filtered_series = [series_info[0]] if series_info else []
    for si in series_info[1:]:
        prev = filtered_series[-1]
        ratio = _z_overlap_ratio(prev['z_min'], prev['z_max'],
                                  si['z_min'], si['z_max'])
        if ratio >= 0.9:
            print(f"    [INFO] {subject_id}: 재촬영 시리즈 감지 "
                  f"(z_overlap={ratio:.2f}) → 중복 시리즈 제외")
        else:
            filtered_series.append(si)
    series_info = filtered_series
    # [C] series_nodule_map: 직접 매칭 우선 → align 보정 → fallback
    series_nodule_map = {i: [] for i in range(len(series_info))}
    series_flip_map = {i: False for i in range(len(series_info))}

    for nodule in nodules:
        center_z = nodule.get('derived', {}).get('center_z')
        if center_z is None:
            continue

        matched_idx, flip, reason = assign_nodule_to_series(center_z, series_info)
        if matched_idx is None:
            continue

        series_nodule_map[matched_idx].append(nodule)
        series_flip_map[matched_idx] = series_flip_map[matched_idx] or flip

        if reason == 'fallback':
            si = series_info[matched_idx]
            print(f"    [WARN] {subject_id} nodule {nodule.get('nodule_idx')}: "
                  f"center_z={center_z} 범위 밖 → series {matched_idx}에 배정 "
                  f"(z_min={si['z_min']:.1f}, z_max={si['z_max']:.1f})")

    # [B-2] annotation 없는 시리즈 제거
    annotated_series = []
    annotated_map = {}
    annotated_flip_map = {}
    for old_idx, si in enumerate(series_info):
        assigned = series_nodule_map.get(old_idx, [])
        if not assigned and nodules:
            print(f"    [INFO] {subject_id}: annotation 없는 series_{si['orig_idx']} 제외 "
                  f"(z_range=[{si['z_min']:.1f}, {si['z_max']:.1f}])")
            continue
        new_idx = len(annotated_series)
        annotated_series.append(si)
        annotated_map[new_idx] = assigned
        annotated_flip_map[new_idx] = series_flip_map.get(old_idx, False)

    series_info = annotated_series
    series_nodule_map = annotated_map
    series_flip_map = annotated_flip_map

    if not series_info:
        return [], {}

    # [D] series별 출력 생성
    for s_idx, si in enumerate(series_info):
        image     = si['image']
        meta      = si['meta']
        ps_x      = meta['pixel_spacing_x']
        ps_y      = meta['pixel_spacing_y']
        nodules_s = series_nodule_map[s_idx]

        out_dir = subj_dir if len(valid_metas) == 1 \
                  else os.path.join(subj_dir, f"series_{si['orig_idx']}")
        os.makedirs(out_dir, exist_ok=True)

        # STEP 1. 리샘플링
        try:
            resampled = resample_image(image, new_spacing=(1.0, 1.0, 1.0),
                                       interpolator=sitk.sitkLinear)
        except Exception as e:
            print(f"    [WARN] {subject_id} series_{s_idx}: 리샘플 실패, 원본 사용\n      {e}")
            resampled = image

        # STEP 2. ct.nii.gz (HU 원본 보관용)
        sitk.WriteImage(resampled,
                        os.path.join(out_dir, 'ct.nii.gz'),
                        useCompression=True)

        # STEP 3. ct_norm.nii.gz
        try:
            norm = normalize_hu(resampled, hu_min=HU_MIN, hu_max=HU_MAX)
            sitk.WriteImage(norm,
                            os.path.join(out_dir, 'ct_norm.nii.gz'),
                            useCompression=True)
        except Exception as e:
            print(f"    [WARN] {subject_id} series_{s_idx}: 정규화 실패\n      {e}")

        for nodule in nodules_s:
            all_violations.extend(check_coord_bounds(nodule, meta, subject_id))
            nidx = nodule.get('nodule_idx', 0)
            all_mm_updates[nidx] = compute_mm_coords(nodule, ps_x, ps_y)

        # STEP 4. seg.nii.gz 생성

        seg_img, manifest = create_segmentation_mask(
            resampled_image=resampled,
            nodules=nodules_s,
            ps_x=ps_x,
            ps_y=ps_y,
            z_flip=series_flip_map[s_idx],
        )

        sitk.WriteImage(
            seg_img,
            os.path.join(out_dir, "seg.nii.gz"),
            useCompression=True,
        )

        with open(
            os.path.join(out_dir, "seg_manifest.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                {
                    "series": [
                        {
                            "series": None,
                            "labels": [
                                m["label"]
                                for m in manifest
                            ],
                            "nodule_indices": [
                                m["nodule_idx"]
                                for m in manifest
                            ],
                        }
                    ]
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        # STEP 5. centroid.csv
        build_centroid_csv(
            nodules_s,
            ps_x,
            ps_y,
            os.path.join(out_dir, "centroid.csv"),
        )

    return all_violations, all_mm_updates


# ── 섹션 9. JSON 업데이트 + 보고서 저장 ─────────────────────────────────────────

def patch_json_with_mm_coords(all_mm_updates: dict, save_path: str) -> None:
    """
    nodule_info_clean.json 각 결절에 mm_coords 필드 추가.
    --subjects로 일부만 재처리해도 나머지 데이터가 보존됨.
    """
    with open(save_path, 'r', encoding='utf-8') as f:
        full_data = json.load(f)

    for subject_id, nodule_updates in all_mm_updates.items():
        for nodule in full_data.get(subject_id, {}).get('nodules', []):
            idx = nodule.get('nodule_idx')
            if idx in nodule_updates:
                nodule['mm_coords'] = nodule_updates[idx]

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(full_data, f, indent=2, ensure_ascii=False)
    print(f"[6/6] JSON 업데이트 완료: {save_path}")


def save_violation_report(all_violations: list, path: str) -> None:
    """polygon이 원본 CT 해상도 밖으로 나간 좌표 보고서 저장."""
    affected = {v['subject_id'] for v in all_violations}
    report = {
        'summary': {
            'total_violations'    : len(all_violations),
            'affected_subjects'   : len(affected),
            'affected_subject_ids': sorted(affected),
        },
        'violations': all_violations,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n{'='*50}")
    print(f"  polygon 범위 초과 총계 : {len(all_violations)}개")
    print(f"  영향받은 환자          : {len(affected)}명")
    print(f"  보고서                : {path}")
    print(f"{'='*50}")


def _collect_seg_paths(subj_dir: str) -> List[Tuple[Optional[str], str]]:
    single = os.path.join(subj_dir, 'seg.nii.gz')
    if os.path.exists(single):
        return [(None, single)]
    return [
        (d, os.path.join(subj_dir, d, 'seg.nii.gz'))
        for d in sorted(os.listdir(subj_dir))
        if d.startswith('series_')
        and os.path.exists(os.path.join(subj_dir, d, 'seg.nii.gz'))
    ]


def _load_expected_seg_labels(subj_dir: str, nodules: list) -> Dict[Optional[str], List[dict]]:
    manifest_path = os.path.join(subj_dir, 'seg_manifest.json')
    by_series = {}
    if os.path.exists(manifest_path):
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)
        for item in manifest.get('series', []):
            series_tag = item.get('series')
            labels = item.get('labels', [])
            nodule_indices = item.get('nodule_indices', [])
            by_series[series_tag] = [
                {'nodule_idx': idx, 'label': label}
                for idx, label in zip(nodule_indices, labels)
            ]
        return by_series

    # seg_manifest.json이 없는 구버전 호환: 전체 결절을 단일 그룹으로 처리
    by_series[None] = [
        {'nodule_idx': n.get('nodule_idx'), 'label': n.get('nodule_idx', 0) + 1}
        for n in nodules
    ]
    return by_series


def save_seg_empty_report(nifti_dir: str, patient_dict: dict, path: str):
    """
    seg가 완전히 비어있는 경우만 검사.

    label 번호 검증은 하지 않는다.
    (series별 label 재번호 부여 방식과 충돌)
    """
    problem = {}

    for sid in sorted(os.listdir(nifti_dir)):
        subj_dir = os.path.join(nifti_dir, sid)

        if not os.path.isdir(subj_dir):
            continue

        seg_paths = _collect_seg_paths(subj_dir)

        if not seg_paths:
            continue

        empty_series = []

        for series_tag, seg_path in seg_paths:

            seg = nib.load(seg_path).get_fdata()

            if np.count_nonzero(seg) == 0:
                empty_series.append(series_tag)

        if empty_series:
            problem[sid] = {
                "empty_series": empty_series
            }

    report = {
        "summary": {
            "total_subjects": len(problem),
            "subject_ids": sorted(problem.keys())
        },
        "subjects": problem
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 50)

    if problem:
        print(f"  [경고] seg 전체가 비어있는 환자: {len(problem)}명")

        for sid, info in problem.items():
            print(
                f"    {sid}: empty series = {info['empty_series']}"
            )

    else:
        print("  [OK] seg 전체 비어있는 환자: 0명")

    print(f"  보고서: {path}")
    print("=" * 50)
# ── 섹션 10. 메인 ───────────────────────────────────────────────────────────────

def main(subjects: Optional[List[str]] = None) -> None:
    nodule_info = load_clean_json(CLEAN_JSON_PATH)
    if subjects:
        nodule_info = {k: v for k, v in nodule_info.items() if k in subjects}
        print(f"    → 필터링 후: {len(nodule_info)}명")

    os.makedirs(NIFTI_SAVE_DIR, exist_ok=True)

    all_violations = []
    all_mm_updates = {}
    failed = 0

    print(f"[2/6] NIfTI 변환 시작 → {NIFTI_SAVE_DIR}")
    print(f"      DICOM 읽기 : 서버 원본 (복사 없음)")
    print(f"      NIfTI 저장 : {NIFTI_SAVE_DIR}")

    for subject_id, patient in tqdm(nodule_info.items(), desc="환자 처리"):
        try:
            viol, mm_upd = process_patient(subject_id, patient, NIFTI_SAVE_DIR)
            all_violations.extend(viol)
            if mm_upd:
                all_mm_updates[subject_id] = mm_upd
        except Exception as e:
            print(f"    [ERROR] {subject_id}: {e}")
            failed += 1

    print(f"\n[3/6] 완료 | 실패: {failed}명")

    print(f"[4/6] polygon 좌표 범위 초과 보고서 저장")
    save_violation_report(all_violations, VIOLATION_PATH)

    print(f"[5/6] nodule_info_clean.json mm 좌표 업데이트")
    patch_json_with_mm_coords(all_mm_updates, CLEAN_JSON_PATH)

    print(f"[6/6] seg mask 검증 (결절 있는데 seg 빈 환자 확인)")
    save_seg_empty_report(NIFTI_SAVE_DIR, nodule_info, SEG_EMPTY_PATH)

    print(f"\n{'='*60}")
    print(f"  출력 폴더 : {NIFTI_SAVE_DIR}/{{subject_id}}/")
    print(f"  파일 구성")
    print(f"    ct.nii.gz      ← 1mm 리샘플 + HU 원본 보존")
    print(f"    ct_norm.nii.gz ← 1mm 리샘플 + 정규화 [0,1] (ML 입력용)")
    print(f"    centroid.csv   ← centroid mm 좌표")
    print(f"{'='*60}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--subjects', nargs='+', default=None,
                        help='처리할 subject_id 목록. 예: LIDC-IDRI-0085')
    args = parser.parse_args()
    main(subjects=args.subjects)
