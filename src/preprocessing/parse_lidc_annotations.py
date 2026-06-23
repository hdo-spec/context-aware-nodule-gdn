# src/preprocessing/parse_lidc_annotations.py
#
# ─── 역할 ────────────────────────────────────────────────────────────────────
#   LIDC-IDRI XML 어노테이션을 파싱하여 환자별 결절 딕셔너리를 생성하고 JSON으로 저장.
#
# ─── 파이프라인에서의 위치 ────────────────────────────────────────────────────
#   [1] parse_lidc_annotations.py  ← 현재 파일
#       XML + metadata.csv 파싱 → nodule_info.json
#   [2] match_dicom.py
#       nodule_info.json + DICOM 헤더 → nodule_info_clean.json (ct_meta 추가)
#   [3] export_nifti.py
#       nodule_info_clean.json + DICOM → NIfTI + seg mask
#
# ─── 출력 ────────────────────────────────────────────────────────────────────
#   data/processed/nodule_info.json
#
# ─── 실행 ────────────────────────────────────────────────────────────────────
#   python -m src.preprocessing.parse_lidc_annotations

import os
import glob
import json
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np

from src.configs.config import (
    XML_DIR, METADATA_CSV, JSON_PATH,
    NODULE_XY_THR, NODULE_Z_THR, MIN_POLY_PTS,
)

# LIDC XML 네임스페이스 — 없으면 태그 탐색 불가
NS = {'ns': 'http://www.nih.gov'}

# ─── deci-mm 보정 기준 ────────────────────────────────────────────────────────
# LIDC-IDRI XML의 일부 케이스에서 z_position이 deci-mm(0.1mm 단위)로 잘못 기록됨.
# *do* src/checks/check_deci_mm.py에서 확인 가능
# 인체 CT z 범위는 실측 최대 ±600mm이므로, |z| > 800이면 deci-mm로 판단.
# DICOM ImagePositionPatient는 항상 표준 mm이므로 이 보정은 XML 파싱에서만 적용.
Z_DECI_MM_THR    = 800.0    # 이 값 초과 시 deci-mm 의심
Z_PHYSICAL_MAX   = 600.0    # 인체 CT z 범위 상한 (mm)


# ══════════════════════════════════════════════════════════════════════════════
# 섹션 1. metadata.csv → UID 매핑
# ══════════════════════════════════════════════════════════════════════════════

def build_uid_map(metadata_csv: str) -> dict:
    """
    metadata.csv 로드 → {series_uid: {subject_id, file_location}} 딕셔너리 반환.
    루프마다 CSV 탐색 대신 O(1) 조회를 위해 딕셔너리로 변환.
    *do* O(1) 딕셔너리 조회 :
       리스트/DF의 경우 원하는 값을 찾을 때까지 처음부터 순서대로 읽지만,
       CSV를 딕셔너리로 변환하면 이후 조회는 크기에 관계없이 1번의 연산으로 값이 저장된 위치를 바로 계산 가능
    """
    df = pd.read_csv(metadata_csv)
    return {
        str(row['Series UID']).strip(): {
            'subject_id'   : str(row['Subject ID']).strip(),
            'file_location': str(row['File Location']).strip(),
        }
        for _, row in df.iterrows()
    }


# ══════════════════════════════════════════════════════════════════════════════
# 섹션 2. XML → 판독자별 결절 후보 수집
# ══════════════════════════════════════════════════════════════════════════════

def _fix_deci_mm(z_raw: float) -> float:
    """
    XML z_position이 deci-mm 단위로 잘못 기록된 경우 mm로 보정.
    *do* src/checks/check_deci_mm.py에서 z_position 분포 파악 가능
        z 분포를 보면 대부분 ±500mm 이내에 분포, 800 넘는 값은 deci-mm 오기록일 가능성이 높음

    판단 기준:
      |z| > 800 이고, /10 결과가 ±600 이내이면 deci-mm로 판단하여 보정.
    """
    # if abs(z_raw) > Z_DECI_MM_THR: # *do* 800 초과이면 deci-mm 의심
    #     candidate = z_raw / 10.0
    #     if abs(candidate) <= Z_PHYSICAL_MAX:    # *do* /10 결과가 ±600 이내이면
    #         return candidate                    #       -> mm로 보정
    # return z_raw
    """
    일부 XML에서 z가 deci-mm(10배)로 저장된 경우 보정.
    """
    # if abs(z_raw) > Z_DECI_MM_THR:
    #     candidate = z_raw / 10.0

    #     if abs(candidate) <= Z_PHYSICAL_MAX:
    #         return candidate

    return z_raw


def collect_raw_nodules(root: ET.Element) -> list:
    """
    XML 루트에서 판독자별 결절 후보를 flat 리스트로 수집.

    제외 조건:
      - characteristics 또는 malignancy 태그 없음
      - malignancy 값이 1~5 범위 밖
      - polygon 꼭짓점 수 < MIN_POLY_PTS
      - ROI 없는 결절

    반환 구조 (결절 1개):
      {
          'reader_id' : int,        # 판독자 인덱스 (매칭 시 같은 판독자 중복 방지용)
          'malignancy': int,        # 악성도 점수 (1~5)
          'rois'      : [           # 슬라이스별 ROI 목록
              {
                  'z_position': float,   # mm (deci-mm 보정 완료)
                  'polygon'   : [{'x': float, 'y': float}, ...],
              },
              ...
          ],
          'centroid_x': float,      # 매칭용 — JSON에 저장 안 됨
          'centroid_y': float,
          'centroid_z': float,
      }
    """
    raw_nodules = []

    for reader_id, session in enumerate(root.findall('.//ns:readingSession', NS)):
        for nodule in session.findall('ns:unblindedReadNodule', NS):

            # malignancy 추출
            char = nodule.find('ns:characteristics', NS)
            if char is None:
                continue
            mal_elem = char.find('ns:malignancy', NS)
            if mal_elem is None or not mal_elem.text:
                continue
            malignancy = int(mal_elem.text)
            if malignancy not in range(1, 6):
                continue

            # ROI(슬라이스별 polygon) 수집
            rois  = []
            all_x = []
            all_y = []
            all_z = []

            for roi in nodule.findall('ns:roi', NS):
                z_elem = roi.find('ns:imageZposition', NS)
                if z_elem is None:
                    continue

                z_pos = _fix_deci_mm(float(z_elem.text))
                
                polygon = []
                for edge in roi.findall('ns:edgeMap', NS):
                    x = float(edge.find('ns:xCoord', NS).text)
                    y = float(edge.find('ns:yCoord', NS).text)
                    polygon.append({'x': x, 'y': y})
                    all_x.append(x)
                    all_y.append(y)

                if len(polygon) < MIN_POLY_PTS:
                    continue

                rois.append({'z_position': z_pos, 'polygon': polygon})
                all_z.append(z_pos)

            if not rois:
                continue

            raw_nodules.append({
                'reader_id' : reader_id,
                'malignancy': malignancy,
                'rois'      : rois,
                'centroid_x': float(np.mean(all_x)),
                'centroid_y': float(np.mean(all_y)),
                'centroid_z': float(np.median(all_z)),
            })

    return raw_nodules


# ══════════════════════════════════════════════════════════════════════════════
# 섹션 3. centroid 거리 기반 결절 매칭
# ══════════════════════════════════════════════════════════════════════════════

def match_nodules(raw_nodules: list) -> list:
    """
    판독자 간 centroid 거리가 임계값 이내이면 같은 결절로 묶음.
    O(n²) greedy 매칭 — 결절 수가 수십 개 수준이므로 충분히 빠름.

    매칭 조건:
      xy 유클리드 거리 ≤ NODULE_XY_THR (픽셀)
      z 거리          ≤ NODULE_Z_THR  (mm)
      같은 판독자끼리는 병합 금지 (판독자 5명 버그 방지)

    반환 구조 (결절 1개):
      {
          'nodule_idx': int,
          'malignancy': [int, ...],   # 판독자별 점수 리스트
          'rois'      : {
              'reader_0': [{'z_position': float, 'polygon': [...]}, ...],
              'reader_1': [...],
              ...
          },
      }
    """
    n    = len(raw_nodules)
    used = [False] * n
    groups = []

    for i in range(n):
        if used[i]:
            continue

        group   = [raw_nodules[i]]
        used[i] = True

        for j in range(i + 1, n):
            if used[j]:
                continue

            # 같은 판독자가 이미 그룹에 있으면 병합 금지
            existing_readers = {m['reader_id'] for m in group}
            if raw_nodules[j]['reader_id'] in existing_readers:
                continue

            dx = raw_nodules[i]['centroid_x'] - raw_nodules[j]['centroid_x']
            dy = raw_nodules[i]['centroid_y'] - raw_nodules[j]['centroid_y']
            xy_dist = np.sqrt(dx**2 + dy**2)
            dz      = abs(raw_nodules[i]['centroid_z'] - raw_nodules[j]['centroid_z'])

            if xy_dist <= NODULE_XY_THR and dz <= NODULE_Z_THR:
                group.append(raw_nodules[j])
                used[j] = True

        groups.append(group)

    nodule_list = []
    for idx, group in enumerate(groups):
        malignancy_list = [m['malignancy'] for m in group]
        rois_by_rater   = {
            f"reader_{rater_idx}": [
                {'z_position': roi['z_position'], 'polygon': roi['polygon']}
                for roi in m['rois']
            ]
            for rater_idx, m in enumerate(group)
        }
        nodule_list.append({
            'nodule_idx': idx,
            'series_uid'   : None,   # build_patient_dict에서 채워짐
            'file_location': None,   # build_patient_dict에서 채워짐
            'malignancy'   : malignancy_list,
            'rois'         : rois_by_rater,
        })

    return nodule_list


# ══════════════════════════════════════════════════════════════════════════════
# 섹션 4. annotations → derived 계산
# ══════════════════════════════════════════════════════════════════════════════

def compute_derived(malignancy: list, rois: dict) -> dict:
    """
    rois 원본에서 파생값 계산.

    계산 항목:
      all_z_positions  : 결절이 걸친 모든 z좌표 (중복 제거, 정렬)
      num_slices       : 걸친 슬라이스 수
      center_z         : 판독자들이 가장 많이 표시한 z (동률이면 중앙값)
      center_slice_idx : all_z_positions 내 center_z 인덱스
      rep_polygon      : center_z 슬라이스에서 꼭짓점 수 최다인 polygon
      center_x         : rep_polygon centroid x
      center_y         : rep_polygon centroid y

    center_z를 단순 median 대신 "가장 많이 표시된 z"로 선택하는 이유:
      판독자 합의가 가장 높은 슬라이스가 결절의 실질적 중심이기 때문.
      median은 all_z 목록 길이에 따라 실제 슬라이스가 아닌 중간값이 나올 수 있음.
    """
    # 모든 z좌표 수집
    z_set = set()
    for rater_rois in rois.values():
        for roi in rater_rois:
            z_set.add(roi['z_position'])
    all_z = sorted(z_set)

    if not all_z:
        return {}

    # 가장 많이 등장한 z값 선택 (동률이면 중앙값)
    count_by_z = {}
    for rater_rois in rois.values():
        for roi in rater_rois:
            z = roi['z_position']
            count_by_z[z] = count_by_z.get(z, 0) + 1

    max_count  = max(count_by_z.values())
    candidates = [z for z, c in count_by_z.items() if c == max_count]
    center_z   = float(np.median(candidates))

    # all_z에서 center_z와 가장 가까운 실제 z 찾기
    closest_z  = min(all_z, key=lambda z: abs(z - center_z))
    center_idx = all_z.index(closest_z)

    # center_z 슬라이스에서 꼭짓점 수 최다 polygon 선정
    best_polygon = []
    best_count   = -1
    for rater_rois in rois.values():
        for roi in rater_rois:
            if roi['z_position'] != closest_z:
                continue
            if len(roi['polygon']) > best_count:
                best_count   = len(roi['polygon'])
                best_polygon = roi['polygon']

    if best_polygon:
        center_x = float(np.mean([pt['x'] for pt in best_polygon]))
        center_y = float(np.mean([pt['y'] for pt in best_polygon]))
    else:
        center_x, center_y = 0.0, 0.0

    return {
        'all_z_positions' : all_z,
        'num_slices'      : len(all_z),
        'center_z'        : closest_z,
        'center_slice_idx': center_idx,
        'center_x'        : center_x,
        'center_y'        : center_y,
        'rep_polygon'     : best_polygon,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 섹션 5. 전체 XML 순회 → patient_dict 구성
# ══════════════════════════════════════════════════════════════════════════════

def build_patient_dict() -> dict:
    """
    전체 XML 파일 순회 → 환자별 결절 딕셔너리 반환.

    처리 순서 (XML 1개 기준):
      SeriesInstanceUid 추출 → metadata.csv 매핑 → subject_id 확인
      → collect_raw_nodules (판독자별 결절 후보 수집)
      → match_nodules (판독자 간 결절 매칭)
      → compute_derived (중심 좌표 등 파생값 계산)
      → patient_dict에 결절 추가

    다중 스캔 환자 처리:
      같은 환자의 시리즈 A, B를 분리하지 않고 nodules를 합침.
      이유: 시리즈별로 분리하면 Train/Test split 시 같은 환자 폐 구조 유출(data leakage) 위험.
      dataset.py에서 subject_id 기준으로 Train/Val/Test 배정할 것.
    """
    uid_map = build_uid_map(METADATA_CSV)
    print(f"[1/4] metadata 로드: {len(uid_map)}개 UID")

    xml_paths = sorted(glob.glob(os.path.join(XML_DIR, '**', '*.xml'), recursive=True))
    print(f"[2/4] XML 파일 수집: {len(xml_paths)}개")

    patient_dict   = {}
    processed_uids = set()
    skip_no_uid = skip_no_match = skip_no_nodule = skip_duplicate = 0

    for xml_path in xml_paths:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        uid_elem = root.find('.//ns:SeriesInstanceUid', NS)
        if uid_elem is None:
            skip_no_uid += 1
            continue
        series_uid = uid_elem.text.strip()

        if series_uid in processed_uids:
            skip_duplicate += 1
            continue

        if series_uid not in uid_map:
            skip_no_match += 1
            continue

        processed_uids.add(series_uid)

        subject_id    = uid_map[series_uid]['subject_id']
        file_location = uid_map[series_uid]['file_location']

        raw_nodules = collect_raw_nodules(root)
        if not raw_nodules:
            skip_no_nodule += 1

        nodule_list = match_nodules(raw_nodules) if raw_nodules else []

        for nodule in nodule_list:
            nodule['series_uid']    = series_uid
            nodule['file_location'] = file_location
            nodule['derived']       = compute_derived(
                malignancy=nodule['malignancy'],
                rois=nodule['rois'],
            )

        if subject_id not in patient_dict:
            patient_dict[subject_id] = {
                'subject_id' : subject_id,
                'series_uids': [],
                'nodules'    : [],
            }

        if series_uid not in patient_dict[subject_id]['series_uids']:
            patient_dict[subject_id]['series_uids'].append(series_uid)

        patient_dict[subject_id]['nodules'].extend(nodule_list)

    # nodule_idx 재부여 (다중 시리즈 환자의 결절이 합쳐진 후 0부터 순서대로)
    for patient in patient_dict.values():
        for i, nodule in enumerate(patient['nodules']):
            nodule['nodule_idx'] = i

    print(f"[3/4] 처리 완료")
    print(f"      XML 총계         : {len(xml_paths)}개")
    print(f"      UID 없음         : {skip_no_uid}개")
    print(f"      중복 XML 스킵    : {skip_duplicate}개")
    print(f"      매핑 실패        : {skip_no_match}개")
    print(f"      결절 없는 시리즈 : {skip_no_nodule}개")
    print(f"      환자 수          : {len(patient_dict)}명")
    print(f"      총 결절 수       : {sum(len(p['nodules']) for p in patient_dict.values())}개")

    return patient_dict


def save_json(patient_dict: dict, json_path: str) -> None:
    """patient_dict를 subject_id 기준 정렬 후 JSON으로 저장."""
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    sorted_dict = dict(sorted(patient_dict.items()))
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(sorted_dict, f, indent=2, ensure_ascii=False)
    print(f"[4/4] JSON 저장: {json_path}")


if __name__ == '__main__':
    patient_dict = build_patient_dict()
    save_json(patient_dict, JSON_PATH)
