# src/preprocessing/match_dicom.py
#
# ─── 역할 ────────────────────────────────────────────────────────────────────
#   nodule_info.json의 file_location → DICOM 폴더 직접 접근
#   → DICOM 헤더 파싱으로 CT 공간 정보 수집
#   → nodule_info.json에 ct_meta 추가 후 nodule_info_clean.json 생성
#
# ─── 파이프라인에서의 위치 ────────────────────────────────────────────────────
#   [1] parse_lidc_annotations.py
#       XML + metadata.csv 파싱 → nodule_info.json
#   [2] match_dicom.py           ← 현재 파일
#       nodule_info.json + DICOM 헤더 → nodule_info_clean.json (ct_meta 추가)
#   [3] export_nifti.py
#       nodule_info_clean.json + DICOM → NIfTI + seg mask
#
# ─── 출력 ────────────────────────────────────────────────────────────────────
#   data/processed/nodule_info.json         (ct_meta 필드 추가된 원본)
#   data/processed/nodule_info_clean.json   (ct_meta 오류 없는 환자만)
#   outputs/figures/dicom_histogram.png
#
# ─── 실행 ────────────────────────────────────────────────────────────────────
#   python -m src.preprocessing.match_dicom

import os
import glob
import json
import numpy as np
import pydicom

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from collections import Counter

from src.configs.config import (
    SERVER_DICOM_ROOT,
    JSON_PATH,
    PROCESSED_ROOT,
    FIGURE_DIR,
    NODULE_Z_THR,
)


# ══════════════════════════════════════════════════════════════════════════════
# 섹션 1. file_location → DICOM 폴더 경로 변환
# ══════════════════════════════════════════════════════════════════════════════

def resolve_dicom_dir(server_root: str, file_location: str) -> str | None:
    """
    metadata.csv의 File Location을 실제 DICOM 폴더 절대경로로 변환.

    File Location 형식 예시:
        './LIDC-IDRI/LIDC-IDRI-0001/01-01-2000-NA-NA-30178/3000566.000000-NA-03192'

    변환 규칙:
        './' 또는 '.\' 제거 후 server_root와 결합 → 절대경로
    """
    relative  = file_location.lstrip('./').lstrip('.\\').replace('\\', '/')
    dicom_dir = os.path.join(server_root, relative)
    return dicom_dir if os.path.isdir(dicom_dir) else None


# ══════════════════════════════════════════════════════════════════════════════
# 섹션 2. DICOM 헤더 파싱 → CT 메타 추출
# ══════════════════════════════════════════════════════════════════════════════

def _get(ds, tag: str, cast=None):
    """DICOM 태그 안전 추출 헬퍼. 태그 없거나 변환 실패 시 None 반환."""
    val = getattr(ds, tag, None)
    if val is None:
        return None
    try:
        return cast(val) if cast else val
    except Exception:
        return None


def extract_ct_meta(dicom_dir: str) -> dict:
    """
    DICOM 폴더의 첫 번째 .dcm 헤더에서 CT 공간 정보 추출.
    stop_before_pixels=True → 픽셀 데이터 로드 없이 헤더만 읽어 속도 빠름.

    [공간 정보]
      rows, cols          : 슬라이스 해상도 (픽셀)
      num_slices          : 폴더 내 .dcm 파일 수
      pixel_spacing_x/y   : 픽셀 크기 (mm/px) — PixelSpacing[1], [0]
      z_spacing           : 슬라이스 간격 (mm) — SpacingBetweenSlices 또는 SliceThickness

    [HU 변환]
      rescale_slope       : HU = raw × slope + intercept
      rescale_intercept   : 대부분 -1024, 장비마다 다를 수 있음
      bits_stored         : 픽셀 비트 수 (보통 12 or 16)

    [윈도잉]
      window_center/width : 영상 표시 기본값 (폐 CT 표준: -600 / 1500)
    """
    dcm_files = sorted(glob.glob(os.path.join(dicom_dir, '*.dcm')))
    if not dcm_files:
        return {'error': 'no_dcm_files', 'num_slices': 0}

    num_slices = len(dcm_files)

    try:
        ds = pydicom.dcmread(dcm_files[0], stop_before_pixels=True)
    except Exception as e:
        return {'error': f'dcmread_failed: {e}', 'num_slices': num_slices}

    rows = _get(ds, 'Rows', int)
    cols = _get(ds, 'Columns', int)

    if hasattr(ds, 'PixelSpacing') and len(ds.PixelSpacing) == 2:
        pixel_spacing_y = float(ds.PixelSpacing[0])   # row 방향 = y
        pixel_spacing_x = float(ds.PixelSpacing[1])   # col 방향 = x
    else:
        pixel_spacing_y = pixel_spacing_x = None

    if hasattr(ds, 'SpacingBetweenSlices'):
        z_spacing = float(ds.SpacingBetweenSlices)
    elif hasattr(ds, 'SliceThickness'):
        z_spacing = float(ds.SliceThickness)
    else:
        z_spacing = None

    rescale_slope     = _get(ds, 'RescaleSlope',     float)
    rescale_intercept = _get(ds, 'RescaleIntercept', float)
    bits_stored       = _get(ds, 'BitsStored',       int)

    # wc = getattr(ds, 'WindowCenter', None)
    # ww = getattr(ds, 'WindowWidth',  None)
    # if isinstance(wc, pydicom.multival.MultiValue): wc = wc[0]
    # if isinstance(ww, pydicom.multival.MultiValue): ww = ww[0]
    # window_center = float(wc) if wc is not None else None
    # window_width  = float(ww) if ww is not None else None

    return {
        'dicom_dir'        : dicom_dir,
        'rows'             : rows,
        'cols'             : cols,
        'num_slices'       : num_slices,
        'pixel_spacing_x'  : pixel_spacing_x,
        'pixel_spacing_y'  : pixel_spacing_y,
        'z_spacing'        : z_spacing,
        'rescale_slope'    : rescale_slope,
        'rescale_intercept': rescale_intercept,
        'bits_stored'      : bits_stored,
        # 'window_center'    : window_center,
        # 'window_width'     : window_width,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 섹션 3. 결절 직경 계산
# ══════════════════════════════════════════════════════════════════════════════

def compute_nodule_diameter_3d(nodule: dict,
                                pixel_spacing_x: float,
                                pixel_spacing_y: float) -> dict:
    """
    polygon vertex를 mm 공간으로 변환하여 3D physical diameter 계산.

    반환:
      diameter_x/y/z_mm    : 각 축 범위 (mm)
      diameter_max_mm      : 최대 축 직경
      diameter_3d_diag_mm  : 3D 대각선 길이
    """
    xs_mm, ys_mm, zs_mm = [], [], []

    for rater_rois in nodule.get('rois', {}).values():
        for roi in rater_rois:
            z_mm = roi.get('z_position')
            if z_mm is None:
                continue
            for pt in roi.get('polygon', []):
                xs_mm.append(pt['x'] * pixel_spacing_x)
                ys_mm.append(pt['y'] * pixel_spacing_y)
                zs_mm.append(z_mm)

    if not xs_mm:
        return {}

    dx   = max(xs_mm) - min(xs_mm)
    dy   = max(ys_mm) - min(ys_mm)
    dz   = max(zs_mm) - min(zs_mm)
    diag = np.sqrt(dx**2 + dy**2 + dz**2)

    return {
        'diameter_x_mm'      : round(dx, 2),
        'diameter_y_mm'      : round(dy, 2),
        'diameter_z_mm'      : round(dz, 2),
        'diameter_max_mm'    : round(max(dx, dy, dz), 2), # crop 크기 참고
        'diameter_3d_diag_mm': round(diag, 2),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 섹션 4. series_uid → file_location 수집
# ══════════════════════════════════════════════════════════════════════════════

def collect_series_locations(nodule_info: dict) -> dict:
    """
    nodule_info.json에서 series_uid → file_location 매핑 수집.
    series uid별 file_location 모으기
    이후 nodule 순회로 모든 시리즈 경로를 복원할 수 있음.
    """
    series_map = {}
    for patient in nodule_info.values():
        for nodule in patient.get('nodules', []):
            uid = nodule.get('series_uid')
            loc = nodule.get('file_location')
            if uid and loc and uid not in series_map:
                series_map[uid] = loc
    return series_map


# ══════════════════════════════════════════════════════════════════════════════
# 섹션 5. DICOM 수집 결과 분석 출력
# ══════════════════════════════════════════════════════════════════════════════

def _print_analysis(dicom_stats: dict) -> None:
    """
    DICOM 수집 결과 분석 요약 출력.

    항목:
      [1] pixel_spacing 등방성 확인 (x == y 여부)
      [2] z_spacing 분포 + NODULE_Z_THR 안전성 검토
      [3] 결절 크기별 최소 픽셀 수 (다운샘플 안전성 판단)
      [4] RescaleSlope / RescaleIntercept 이상치 확인
      [5] BitStored 분포
    """
    valid = [v for v in dicom_stats.values() if 'error' not in v]
    if not valid:
        print("  [WARN] 유효한 series 없음")
        return

    psx = np.array([v['pixel_spacing_x'] for v in valid if v.get('pixel_spacing_x')])
    psy = np.array([v['pixel_spacing_y'] for v in valid if v.get('pixel_spacing_y')])
    psz = np.array([v['z_spacing']       for v in valid if v.get('z_spacing')])

    print()
    print("=" * 60)
    print("  DICOM Analysis Summary")
    print("=" * 60)

    # [1] pixel_spacing 등방성
    print("\n  [1] Pixel Spacing Isotropy (xy)")
    if len(psx) == len(psy):
        non_iso = int((np.abs(psx - psy) > 1e-6).sum())
        if non_iso == 0:
            print(f"  All {len(psx)} series: isotropic (x == y)")
        else:
            print(f"  [WARN] {non_iso} series: x != y")
        print(f"  xy spacing range : {psx.min():.4f} ~ {psx.max():.4f} mm/px")
        print(f"  xy spacing median: {np.median(psx):.4f} mm/px")

    # [2] z_spacing 분포
    print(f"\n  [2] Z Spacing Distribution  (NODULE_Z_THR = {NODULE_Z_THR} mm)")
    for z_val, cnt in sorted(Counter(round(z, 3) for z in psz).items()):
        bar    = "█" * int(cnt / len(psz) * 40)
        status = "OK" if NODULE_Z_THR >= z_val * 1.5 else "WARN: THR too small"
        print(f"  {z_val:5.3f} mm: {cnt:4d}개  {bar}  [{status}]")
    exceed = int((psz > NODULE_Z_THR).sum())
    if exceed == 0:
        print(f"  → NODULE_Z_THR is safe for all series.")
    else:
        print(f"  → [WARN] {exceed} series have z_spacing > NODULE_Z_THR.")

    # [3] 결절 크기별 최소 픽셀 수
    print(f"\n  [3] Nodule Size in Pixels  (worst case: {psx.max():.4f} mm/px)")
    for nodule_mm in [3, 5, 10, 20, 30]:
        px_worst = nodule_mm / psx.max()
        px_best  = nodule_mm / psx.min()
        flag = "  [WARN: < 4px]" if px_worst < 4 else ""
        print(f"  {nodule_mm:3d}mm nodule: {px_worst:.1f}px ~ {px_best:.1f}px{flag}")

    # [4] RescaleSlope / RescaleIntercept
    print(f"\n  [4] RescaleSlope / RescaleIntercept")
    slopes     = [v['rescale_slope']     for v in valid if v.get('rescale_slope')     is not None]
    intercepts = [v['rescale_intercept'] for v in valid if v.get('rescale_intercept') is not None]
    non_unit_slope    = sum(1 for s  in slopes     if abs(s  - 1.0)      > 1e-6)
    non_std_intercept = sum(1 for ic in intercepts if abs(ic - (-1024.0)) > 1.0)
    print(f"  slope != 1.0    : {non_unit_slope}개")
    print(f"  intercept != -1024 : {non_std_intercept}개")
    if non_unit_slope > 0:
        print(f"  [WARN] slope != 1.0 → HU 변환 시 slope 반드시 적용")

    # [5] BitStored 분포
    print(f"\n  [5] BitStored Distribution")
    for b, cnt in Counter(v.get('bits_stored') or 'UNKNOWN' for v in valid).most_common():
        print(f"  {cnt:4d} series  {b} bit")

    print("=" * 60)


# ══════════════════════════════════════════════════════════════════════════════
# 섹션 6. DICOM 히스토그램 생성
# ══════════════════════════════════════════════════════════════════════════════

def plot_dicom_histograms(dicom_stats: dict, figure_path: str) -> None:
    """
    dicom_stats → 공간 분포 히스토그램 4종 저장.
    rows, cols, num_slices, pixel/z spacing 분포를 시각화.
    """
    valid = [v for v in dicom_stats.values() if 'error' not in v]
    if not valid:
        print("  [WARN] 히스토그램 생략: 유효 series 없음")
        return

    def collect(key):
        return [v[key] for v in valid if v.get(key) is not None]

    rows_data   = collect('rows')
    cols_data   = collect('cols')
    slices_data = collect('num_slices')
    psx_data    = collect('pixel_spacing_x')
    psz_data    = collect('z_spacing')

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('LIDC-IDRI DICOM Spatial Distribution', fontsize=14, fontweight='bold')

    ax = axes[0, 0]
    unique_rows, counts_rows = np.unique(np.array(rows_data), return_counts=True)
    ax.bar(unique_rows.astype(str), counts_rows, color='steelblue', edgecolor='white')
    ax.set_title('Rows (height pixels)')
    ax.set_xlabel('Pixels')
    ax.set_ylabel('Series count')
    for x, cnt in enumerate(counts_rows):
        ax.text(x, cnt + 1, str(cnt), ha='center', va='bottom', fontsize=9)

    ax = axes[0, 1]
    unique_cols, counts_cols = np.unique(np.array(cols_data), return_counts=True)
    ax.bar(unique_cols.astype(str), counts_cols, color='darkorange', edgecolor='white')
    ax.set_title('Cols (width pixels)')
    ax.set_xlabel('Pixels')
    ax.set_ylabel('Series count')
    for x, cnt in enumerate(counts_cols):
        ax.text(x, cnt + 1, str(cnt), ha='center', va='bottom', fontsize=9)

    ax = axes[1, 0]
    ax.hist(np.array(slices_data), bins=30, color='seagreen', edgecolor='white')
    ax.set_title('Num Slices (depth)')
    ax.set_xlabel('Slice count')
    ax.set_ylabel('Series count')
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    ax = axes[1, 1]
    bins = np.linspace(0, max(max(psx_data), max(psz_data)) + 0.5, 40)
    ax.hist(psx_data, bins=bins, color='mediumpurple', alpha=0.7,
            label='xy spacing (mm/px)', edgecolor='white')
    ax.hist(psz_data, bins=bins, color='tomato', alpha=0.6,
            label='z spacing (mm/slice)', edgecolor='white')
    ax.set_title('Pixel / Slice Spacing (mm)')
    ax.set_xlabel('Spacing (mm)')
    ax.set_ylabel('Series count')
    ax.legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(figure_path), exist_ok=True)
    plt.savefig(figure_path, dpi=150, bbox_inches='tight')
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# 섹션 7. 메인: DICOM 탐색 → ct_meta 추가 → JSON 저장
# ══════════════════════════════════════════════════════════════════════════════

def build_dicom_stats(json_path: str, server_root: str) -> None:
    """
    nodule_info.json의 file_location → DICOM 폴더 직접 접근
    → CT 해상도 수집 → ct_meta 추가 → nodule_info_clean.json 생성.

    ct_meta 구조 (환자 1명):
      [
          {
              'series_uid'      : str,
              'dicom_dir'       : str,
              'rows'            : int,
              'cols'            : int,
              'num_slices'      : int,
              'pixel_spacing_x' : float,
              'pixel_spacing_y' : float,
              'z_spacing'       : float,
              'rescale_slope'   : float,
              'rescale_intercept': float,
              'bits_stored'     : int,
            #   'window_center'   : float,
            #   'window_width'    : float,
          },
          ...
      ]

    nodule_info_clean.json:
      ct_meta에 error 없는 환자만 포함.
      이후 export_nifti.py의 입력으로 사용됨.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        nodule_info = json.load(f)
    print(f"[1/5] nodule_info.json 로드: {len(nodule_info)}명")

    series_map = collect_series_locations(nodule_info)
    print(f"[2/5] 고유 series 수집: {len(series_map)}개")

    # DICOM 폴더 접근 + 헤더 파싱
    dicom_stats = {}
    found = not_found = error = 0

    for i, (series_uid, file_location) in enumerate(series_map.items()):
        if (i + 1) % 100 == 0:
            print(f"      처리 중: {i+1}/{len(series_map)}")

        dicom_dir = resolve_dicom_dir(server_root, file_location)
        if dicom_dir is None:
            dicom_stats[series_uid] = {'error': 'dicom_dir_not_found', 'file_location': file_location}
            not_found += 1
            continue

        meta = extract_ct_meta(dicom_dir)
        if 'error' in meta:
            error += 1
        else:
            found += 1
        dicom_stats[series_uid] = meta

    print(f"[3/5] 탐색 완료 | 성공: {found} | 폴더 없음: {not_found} | 읽기 실패: {error}")

    _print_analysis(dicom_stats)

    figure_path = os.path.join(FIGURE_DIR, 'dicom_histogram.png')
    plot_dicom_histograms(dicom_stats, figure_path)
    print(f"[4/5] 히스토그램 저장: {figure_path}")

    # nodule_info.json에 ct_meta + 결절 직경 추가
    for subject_id, patient in nodule_info.items():
        ct_meta_list = []

        for series_uid in patient.get('series_uids', []):
            meta = dicom_stats.get(series_uid, {})
            if 'error' in meta:
                ct_meta_list.append({'series_uid': series_uid, 'error': meta['error']})
            elif meta:
                ct_meta_list.append({
                    'series_uid'       : series_uid,
                    'dicom_dir'        : meta.get('dicom_dir'),
                    'rows'             : meta.get('rows'),
                    'cols'             : meta.get('cols'),
                    'num_slices'       : meta.get('num_slices'),
                    'pixel_spacing_x'  : meta.get('pixel_spacing_x'),
                    'pixel_spacing_y'  : meta.get('pixel_spacing_y'),
                    'z_spacing'        : meta.get('z_spacing'),
                    'rescale_slope'    : meta.get('rescale_slope'),
                    'rescale_intercept': meta.get('rescale_intercept'),
                    'bits_stored'      : meta.get('bits_stored'),
                    # 'window_center'    : meta.get('window_center'),
                    # 'window_width'     : meta.get('window_width'),
                })
            else:
                ct_meta_list.append({'series_uid': series_uid, 'error': 'no_file_location_in_nodules'})

        patient['ct_meta'] = ct_meta_list

        # 결절 직경 계산 (pixel_spacing이 있는 시리즈 기준)
        for nodule in patient.get('nodules', []):
            series_uid  = nodule.get('series_uid')
            matched_meta = next(
                (m for m in ct_meta_list if m.get('series_uid') == series_uid and 'error' not in m),
                None
            )
            if matched_meta is None:
                continue
            ps_x = matched_meta.get('pixel_spacing_x')
            ps_y = matched_meta.get('pixel_spacing_y')
            if ps_x is None or ps_y is None:
                continue
            diameter_info = compute_nodule_diameter_3d(nodule, ps_x, ps_y)
            if diameter_info:
                nodule.setdefault('derived', {}).update(diameter_info)

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(nodule_info, f, indent=2, ensure_ascii=False)

    # ct_meta에 error 없는 환자만 clean JSON으로 저장
    clean_dict = {
        k: v for k, v in nodule_info.items()
        if any('error' not in m for m in v.get('ct_meta', []))
    }
    clean_path = os.path.join(os.path.dirname(json_path), 'nodule_info_clean.json')
    with open(clean_path, 'w', encoding='utf-8') as f:
        json.dump(clean_dict, f, indent=2, ensure_ascii=False)

    print(f"[5/5] JSON 저장 완료")
    print(f"      원본 : {len(nodule_info)}명 → {json_path}")
    print(f"      정제 : {len(clean_dict)}명  → {clean_path}")


if __name__ == '__main__':
    build_dicom_stats(json_path=JSON_PATH, server_root=SERVER_DICOM_ROOT)
