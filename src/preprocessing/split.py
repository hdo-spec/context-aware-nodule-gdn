# src/preprocessing/split.py
#
# ─── 역할 ────────────────────────────────────────────────────────────────────
#   labels_2d_{crop}.csv 또는 labels.csv → train/val/test split.json 생성
#
# ─── 핵심: subject 단위 split ────────────────────────────────────────────────
#   [왜 subject 단위인가?]
#     한 환자에 결절이 여러 개일 수 있음.
#     결절 단위로 split → 같은 환자가 train/test 양쪽에 등장 → data leakage.
#     subject 단위로 split해야 실제 일반화 성능을 측정할 수 있음.
#
#   [왜 StratifiedShuffleSplit인가?]
#     StratifiedGroupKFold(n_splits=5)를 쓰고 next()만 쓰면 사실상 80/20.
#     비율을 명시적으로 지정할 수 없음.
#     StratifiedShuffleSplit은 val_ratio/test_ratio를 직접 지정 가능.
#     → 70/15/15 비율을 정확하게 구현.
#
#   [split.json vs 3개 CSV]
#     JSON 하나에 split 메타(seed, 비율, 날짜)를 같이 보존.
#     CSV 3개는 메타가 사라져 나중에 어떤 설정으로 만들었는지 알 수 없음.
#
# ─── 실행 ────────────────────────────────────────────────────────────────────
#   python -m src.preprocessing.split
#   python -m src.preprocessing.split --labels_csv data/processed/labels_2d_64.csv

import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from src.configs.config import PROCESSED_ROOT, SPLIT_JSON, SEED

VAL_RATIO  = 0.15
TEST_RATIO = 0.15


# ══════════════════════════════════════════════════════════════════════════════
# 섹션 1. subject 레벨 레이블 생성
# ══════════════════════════════════════════════════════════════════════════════

def build_subject_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    결절 단위 DataFrame → subject 단위 레이블.

    [subject 레이블 정의]
        악성 결절이 하나라도 있으면 1 (malignant patient)
        모두 양성이면 0 (benign patient)

    stratify 기준을 "환자 레벨 악성 여부"로 설정하는 이유:
        결절 레이블로 stratify하면 같은 환자가 여러 번 카운트됨.
        환자 레벨로 stratify해야 각 fold에서 악성 환자 비율이 고르게 유지됨.
    """
    return (
        df.groupby('subject_id')['label']
          .max()
          .reset_index()
          .rename(columns={'label': 'subject_label'})
    )


# ══════════════════════════════════════════════════════════════════════════════
# 섹션 2. 2단계 stratified split
# ══════════════════════════════════════════════════════════════════════════════

def stratified_split(
    subject_df : pd.DataFrame,
    val_ratio  : float = VAL_RATIO,
    test_ratio : float = TEST_RATIO,
    seed       : int   = SEED,
) -> tuple[list, list, list]:
    """
    subject 레벨 2단계 stratified split.

    [2단계 방법]
        1단계: 전체 → (train+val) / test
        2단계: (train+val) → train / val

    val의 실제 비율 계산:
        val_ratio_adj = val_ratio / (1 - test_ratio)
        예) val=0.15, test=0.15 → val_adj = 0.15/0.85 ≈ 0.176

    Returns:
        (train_ids, val_ids, test_ids): subject_id 리스트
    """
    subjects = subject_df['subject_id'].values
    labels   = subject_df['subject_label'].values

    # 1단계: test 분리
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
    trainval_idx, test_idx = next(sss1.split(subjects, labels))

    subjects_tv = subjects[trainval_idx]
    labels_tv   = labels[trainval_idx]
    test_ids    = subjects[test_idx].tolist()

    # 2단계: val 분리
    val_ratio_adj = val_ratio / (1.0 - test_ratio)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio_adj, random_state=seed)
    train_idx, val_idx = next(sss2.split(subjects_tv, labels_tv))

    train_ids = subjects_tv[train_idx].tolist()
    val_ids   = subjects_tv[val_idx].tolist()

    return train_ids, val_ids, test_ids


# ══════════════════════════════════════════════════════════════════════════════
# 섹션 3. 통계 출력 및 leakage 검증
# ══════════════════════════════════════════════════════════════════════════════

def validate_and_print(
    df         : pd.DataFrame,
    subject_df : pd.DataFrame,
    train_ids  : list,
    val_ids    : list,
    test_ids   : list,
) -> None:
    """split 결과 검증 + 분포 출력."""

    # leakage 검사: 중복 subject 없어야 함
    tv = set(train_ids) & set(val_ids)
    tt = set(train_ids) & set(test_ids)
    vt = set(val_ids)   & set(test_ids)
    assert len(tv) == 0, f'train∩val 중복: {tv}'
    assert len(tt) == 0, f'train∩test 중복: {tt}'
    assert len(vt) == 0, f'val∩test 중복: {vt}'
    print('✅ Data leakage 없음')

    # 분포 출력
    print('\n── Split 분포 ──────────────────────────────────')
    for name, ids in [('train', train_ids), ('val', val_ids), ('test', test_ids)]:
        fold_df  = df[df['subject_id'].isin(ids)]
        fold_sub = subject_df[subject_df['subject_id'].isin(ids)]
        n_nod  = len(fold_df)
        n_ben  = (fold_df['label'] == 0).sum()
        n_mal  = (fold_df['label'] == 1).sum()
        s_ben  = (fold_sub['subject_label'] == 0).sum()
        s_mal  = (fold_sub['subject_label'] == 1).sum()
        ratio  = f'{n_ben/n_mal:.2f}:1' if n_mal > 0 else 'N/A'
        print(f'  {name:5s} | 환자 {len(ids):4d}명 (양성{s_ben}/악성{s_mal})'
              f' | 결절 {n_nod:4d}개 (양성{n_ben}/악성{n_mal}) | 비율 {ratio}')
    print('────────────────────────────────────────────────')


# ══════════════════════════════════════════════════════════════════════════════
# 섹션 4. 메인 함수
# ══════════════════════════════════════════════════════════════════════════════

def make_split(
    labels_csv  : Path  = PROCESSED_ROOT / 'labels.csv',
    output_json : Path  = SPLIT_JSON,
    val_ratio   : float = VAL_RATIO,
    test_ratio  : float = TEST_RATIO,
    seed        : int   = SEED,
) -> dict:
    """labels.csv → split.json 생성."""

    if not labels_csv.exists():
        raise FileNotFoundError(f'labels.csv 없음: {labels_csv}')

    df = pd.read_csv(labels_csv)
    print(f'[split] 결절 {len(df)}개, 환자 {df["subject_id"].nunique()}명')

    subject_df = build_subject_label(df)
    train_ids, val_ids, test_ids = stratified_split(
        subject_df, val_ratio, test_ratio, seed
    )

    validate_and_print(df, subject_df, train_ids, val_ids, test_ids)

    split_dict = {
        'train': sorted(train_ids),
        'val'  : sorted(val_ids),
        'test' : sorted(test_ids),
        'meta' : {
            'total_subjects': len(subject_df),
            'val_ratio'     : val_ratio,
            'test_ratio'    : test_ratio,
            'seed'          : seed,
            'train_n'       : len(train_ids),
            'val_n'         : len(val_ids),
            'test_n'        : len(test_ids),
        },
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(split_dict, f, indent=2, ensure_ascii=False)

    print(f'\n✅ split.json 저장: {output_json}')
    return split_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--labels_csv', type=str,
                        default=str(PROCESSED_ROOT / 'labels.csv'))
    parser.add_argument('--val_ratio',  type=float, default=VAL_RATIO)
    parser.add_argument('--test_ratio', type=float, default=TEST_RATIO)
    parser.add_argument('--seed',       type=int,   default=SEED)
    args = parser.parse_args()

    make_split(
        labels_csv  = Path(args.labels_csv),
        val_ratio   = args.val_ratio,
        test_ratio  = args.test_ratio,
        seed        = args.seed,
    )
