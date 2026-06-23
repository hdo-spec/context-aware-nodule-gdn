# src/configs/config.py

from pathlib import Path

# ── PROJECT_ROOT 확정 ────────────────────────────────────────
# 현재 파일: PROJECT_ROOT/src/configs/config.py
# parents[0] = src/configs / parents[1] = src / parents[2] = PROJECT_ROOT
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ── 서버 원본 경로 (읽기 전용) ──────────────────────────────
SERVER_ROOT         = Path('/data1/wellness_data/project/lidc-idri')
SERVER_DICOM_ROOT   = SERVER_ROOT / 'manifest-1600709154662'
SERVER_XML_DIR      = SERVER_ROOT / 'LIDC-XML-only' / 'tcia-lidc-xml'
SERVER_METADATA_CSV = SERVER_DICOM_ROOT / 'metadata.csv'

# ── 개인 작업 폴더 ───────────────────────────────────────────
RAW_ROOT       = PROJECT_ROOT / 'data' / 'raw'
PROCESSED_ROOT = PROJECT_ROOT / 'data' / 'processed'
OUTPUT_ROOT    = PROJECT_ROOT / 'outputs'

# ── 로컬 원본 경로 (복사본) ──────────────────────────────────
LOCAL_XML_DIR      = RAW_ROOT / 'xml'
LOCAL_METADATA_CSV = RAW_ROOT / 'metadata' / 'metadata.csv'

# ── 실제 사용 경로 (서버 원본 우선) ─────────────────────────
# XML_DIR = SERVER_XML_DIR if SERVER_XML_DIR.exists() else LOCAL_XML_DIR
# METADATA_CSV = SERVER_METADATA_CSV if SERVER_METADATA_CSV.exists() else LOCAL_METADATA_CSV
XML_DIR = LOCAL_XML_DIR
METADATA_CSV = LOCAL_METADATA_CSV

# ── 전처리 결과물 경로 ───────────────────────────────────────
JSON_PATH      = PROCESSED_ROOT / 'nodule_info.json'
NIFTI_SAVE_DIR = PROCESSED_ROOT / 'nifti'
NPY_CACHE_DIR  = PROCESSED_ROOT / 'npy_cache'
SPLIT_JSON     = PROCESSED_ROOT / 'split.json'

# ── Patch 저장 경로 (crop size / n_slices 조합별 디렉토리) ───
# 구조 예시:
#   data/processed/patches_2d/64x64/    ← 2D, crop=64
#   data/processed/patches_2d/96x96/    ← 2D, crop=96
#   data/processed/patches_25d/64x64_s1/ ← 2.5D, crop=64, n_slices=1
#   data/processed/patches_25d/64x64_s2/ ← 2.5D, crop=64, n_slices=2
#
# 디렉토리 이름에 설정을 포함하는 이유:
#   crop_size나 n_slices를 바꿀 때 기존 데이터를 덮어쓰지 않음.
#   실험 설정과 데이터가 1:1 대응되어 재현성 보장.
PATCHES_2D_ROOT  = PROCESSED_ROOT / 'patches_2d'
PATCHES_25D_ROOT = PROCESSED_ROOT / 'patches_25d'

def get_patch_dir(crop_size: int, n_slices: int = 0) -> Path:
    """
    crop_size와 n_slices로 patch 디렉토리 경로 반환.

    Args:
        crop_size: 크롭 크기 (예: 64 → '64x64')
        n_slices:  0이면 2D, 1 이상이면 2.5D

    Returns:
        해당 설정의 patch 저장 디렉토리 Path
    """
    if n_slices == 0:
        return PATCHES_2D_ROOT / f'{crop_size}x{crop_size}'
    else:
        return PATCHES_25D_ROOT / f'{crop_size}x{crop_size}_s{n_slices}'

# ── 출력 경로 ───────────────────────────────────────────────
CKPT_DIR    = OUTPUT_ROOT / 'checkpoints'
LOG_DIR     = OUTPUT_ROOT / 'logs'
FIGURE_DIR  = OUTPUT_ROOT / 'figures'
GRADCAM_DIR = OUTPUT_ROOT / 'gradcam'
PRED_DIR    = OUTPUT_ROOT / 'predictions'

# ── 전처리 결과 CSV / JSON 경로 ─────────────────────────────
LABELS_CSV = PROCESSED_ROOT / 'labels.csv'
SPLIT_JSON = PROCESSED_ROOT / 'split.json'

# ── 디렉토리 자동 생성 ───────────────────────────────────────
DIRS_TO_CREATE = [
    RAW_ROOT, PROCESSED_ROOT, OUTPUT_ROOT,
    NIFTI_SAVE_DIR,
    CKPT_DIR, LOG_DIR, FIGURE_DIR, GRADCAM_DIR, PRED_DIR,
]
for path in DIRS_TO_CREATE:
    path.mkdir(parents=True, exist_ok=True)

# ── 전처리 파라미터 ──────────────────────────────────────────
NODULE_XY_THR = 5.0   # 같은 결절로 판단할 x,y 거리 임계값 (pixel)
NODULE_Z_THR  = 8.0   # 같은 결절로 판단할 z 거리 임계값 (mm)
MIN_POLY_PTS  = 3     # 유효 polygon 최소 꼭짓점 수

# ── 학습 기본값 ──────────────────────────────────────────────
SEED        = 42
BATCH_SIZE  = 32
NUM_WORKERS = 4
NUM_CLASSES = 2
