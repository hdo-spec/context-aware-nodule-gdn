# 주변 맥락 기반 폐결절 이진 분류 모델
### GDN(Gated-Dilated Network)에 Dilation 확장과 CBAM을 결합한 접근

> 융합 메디컬 AI 스마트 웰니스 팀 프로젝트
> 홍채은(팀장) · 허다온(부팀장) · 김종호 · 이상욱 · 이유준 · 이준용

---

## 1. Overview

폐결절(pulmonary nodule)은 폐암의 초기 형태로, 조기 발견이 생존율에 직결되지만 **결절이 작을수록 판독이 어렵다**는 임상적 난제가 있다.

본 프로젝트는 *"결절 내부 특징과 함께 주변 맥락(context)을 학습하면, 판독이 어려운 작은 결절(small nodule)의 분류 성능도 향상될 것"* 이라는 가설을 세우고, **LIDC-IDRI** 데이터셋으로 GDN + CBAM 구조를 통해 정량적으로 검증했다.

| 항목 | 내용 |
|---|---|
| 데이터셋 | LIDC-IDRI (흉부 CT, DICOM, 1mm isotropic 리샘플) |
| 핵심 검증 지표 | Fleischner Society volume 기준 subgroup AUC (small / intermediate / large) |
| 최종 모델 | GDN + dilation d2=3 + CBAM(after gd4) + CBAM(after gd5) |
| 최종 성능 | **Test AUC 0.8939** · Small AUC **0.8421** · Sensitivity **0.8404** |

---

## 2. Background & Problem

- 임상에서는 결절 자체보다 **결절과 주변 폐조직·혈관·흉막의 관계**(lobulation, spiculation, vascular convergence, pleural retraction)를 함께 보고 악성도를 판단한다.
- 기존 딥러닝 연구는
  1. 결절 내부 특징에만 집중하거나 (Shen et al., 2017 — Multi-Crop CNN, AUC 0.93)
  2. 주변 맥락의 중요성은 입증했지만 (Al-Shabi et al., 2019a; Liu et al., 2024 — 주변 포함 시 Sensitivity 50%→65.46%) **그 안에서 무엇이 중요한지 선별하는 구조가 없었고**
  3. 결절 크기별(특히 small) 정량적 비교가 미흡했다.

## 3. Hypothesis

> 결절을 학습함과 동시에 결절 주변 맥락을 함께 학습해 분류 성능을 높이면,
> 크기가 작아 판독이 어렵던 소결절(small nodule) 또한 예측 성능이 향상될 것이다.

## 4. Proposed Method — GDN + CBAM

```
Input ─▶ GD layer1 ─▶ GD2 ─▶ GD3 ─▶ GD4 ─▶[CBAM]─▶ GD5 ─▶[CBAM]─▶ GlobalMaxPool ─▶ FC
```

| 구성 요소 | 역할 |
|---|---|
| **GDN (Gated-Dilated Network)** | 병렬 dilation conv(d1=지역 텍스처, d2=확장된 receptive field)로 결절 + 주변 맥락을 **수집** |
| **Alpha Gating** | Context-aware sub-network가 입력을 foreground/background 스트림으로 동적 분기, 결절 크기에 따라 local/global 경로 비율을 자동 결정 |
| **CBAM (Channel + Spatial Attention)** | GDN이 수집한 feature 중 진단에 유의미한 신호만 **선별** (gd4, gd5 출력 직후 2회 적용) |

한 문장 요약: **"GDN이 모으고, CBAM이 거른다."**

GDN의 alpha gating이 multi-scale context 선택을 내재적으로 처리하기 때문에, crop을 키우는 것(scale=6)보다 **GDN 구조 자체로 충분한 context를 확보하는 것(scale=4)** 이 더 원칙적인 접근이라는 점을 ablation으로 확인했다.

## 5. Dataset

| 항목 | 값 |
|---|---|
| 총 환자 수 | 1,010명 (다중 촬영 8명 포함) |
| 총 결절 수 | 약 2,696개 |
| 양/악성 분리 기준 | malignancy 평균 < 3.0 양성 / > 3.0 악성 (=3.0 불확실 제외, -646개) |
| 재촬영 중복 제거 | z-range overlap ≥ 90% 시리즈 제외 (-5개) |
| **최종 사용 결절** | **2,045개** |
| **최종 사용 환자** | **800명** (train 559 / val 121 / test 120, subject 단위 stratified split) |
| Size group (volume 기준) | small < 100㎣ · intermediate 100~250㎣ · large > 250㎣ |
| 입력 형식 | 2.5D 패치 (k-1, k, k+1 슬라이스), crop=32×32 |

## 6. Experiments & Results

### 6.1 베이스라인 모델 비교 (GDN vs ConvNeXt vs Dual-branch ConvNeXt)

| 모델 | Test AUC | Small AUC | Large AUC | Sensitivity | Specificity | 파라미터 수 |
|---|---|---|---|---|---|---|
| ConvNeXt | 0.8785 | 0.7286 | 0.8507 | 0.7979 | 0.7876 | 8,531,841 |
| Dual-ConvNeXt | 0.8705 | 0.6234 | 0.8489 | 0.7128 | 0.9016 | ~12,000,000+ |
| **GDN** | **0.8809** | **0.7599** | 0.8821 | 0.7021 | 0.9326 | **104,459** |

→ GDN이 **파라미터 약 82배 적음**에도 더 높은 전체 AUC, 특히 small subgroup에서 우위 → 백본으로 채택.

### 6.2 단계별 개선 (Ablation)

| 실험 | AUC | Small AUC | Sensitivity | FN |
|---|---|---|---|---|
| Baseline GDN (d2=2) | 0.8809 | 0.7599 | 0.7021 | 28 |
| + Dilation d2=3 | 0.8888 | 0.7944 | 0.8191 | 17 |
| **+ CBAM (gd4, gd5)** | **0.8939** | **0.8421** | **0.8404** | **15** |

- **FN 28 → 15**: 악성 결절 놓침 13건 감소 → 임상적 안전성 관점의 직접적 임팩트
- crop=48: val loss 발산 + threshold collapse + small subgroup AUC 저하 → **기각**
- epoch=100: 전체 AUC는 소폭 상승하나 small subgroup AUC가 하락해 핵심 가설과 충돌 → **기각**

### 6.3 정성적 분석 — Grad-CAM / Attention 해석

- Baseline: feature가 약해 노이즈성 신호에 의존하거나 결절 중심부에만 attention 집중
- CBAM 적용 후:
  - **TN 케이스**: 결절 자체 특징이 약할 때 주변 맥락(혈관, 흉막 구조)으로 attention을 확장해 판단
  - **TP 케이스**: attention이 중심점에서 경계 구조물로 재분배되어 spiculation(침상 돌기), lobulation(분엽) 등 악성 형태 특징을 추가로 포착

> Grad-CAM 비교 시, baseline GDN과 CBAM 모델의 target layer가 달라 비교가 무효해지는 문제를 발견 → CBAM 모델은 spatial attention map을 직접 시각화하는 방식으로 전환해 공정한 비교를 확보.

## 7. Conclusion

| 지표 | 결과 |
|---|---|
| Small AUC 향상 | 0.7599 → **0.8421** (+0.082) |
| Test AUC | **0.8939** (목표 0.90 근접) |
| Sensitivity | **0.8404** (FN 최소화) |

가설 검증: 주변 맥락을 함께 학습하는 구조(GDN+CBAM)가 소결절 분류 성능을 의미 있게 향상시켰다.

---

## 8. Project Structure

```
src/
├── configs/
│   └── config.py              # 경로, 전처리/학습 기본 파라미터
├── preprocessing/
│   ├── parse_lidc_annotations.py  # XML + metadata.csv → nodule_info.json
│   ├── match_dicom.py             # + DICOM 헤더 → nodule_info_clean.json
│   ├── export_nifti.py            # + DICOM → NIfTI(1mm) + seg mask
│   ├── labels.py                   # → labels.csv (malignancy 라벨링, volume_mm3)
│   ├── split.py                    # subject 단위 stratified train/val/test split
│   └── export_patches.py          # → 결절별 2.5D patch .npy 캐시
├── datasets/
│   └── dataset.py              # NoduleDataset / DualNoduleDataset, online augmentation
├── models/
│   └── models.py               # GDN, ConvNeXt, DualConvNeXt
├── training/
│   ├── engine.py                # train/validate loop (결절 단위 평균 집계)
│   └── train.py                 # 실험 실행 + config/history 기록
├── evaluation/
│   └── evaluate.py              # subgroup AUC, Grad-CAM 샘플 선정, 최종 평가
└── utils/
    ├── utils.py                  # device, seed
    └── visualize.py              # ROC, confusion matrix, learning curve, Grad-CAM
```

## 9. Pipeline (How to Run)

```bash
# 1. XML 어노테이션 파싱
python -m src.preprocessing.parse_lidc_annotations

# 2. DICOM 헤더 매칭
python -m src.preprocessing.match_dicom

# 3. NIfTI 변환 + segmentation mask 생성
python -m src.preprocessing.export_nifti

# 4. 라벨 CSV 생성
python -m src.preprocessing.labels

# 5. train/val/test split (subject 단위)
python -m src.preprocessing.split

# 6. 2.5D patch 캐시 생성
python -m src.preprocessing.export_patches --crop_sizes 32 64 96

# 7. 학습
python -m src.training.train --model gdn --crop_size 32 --augment hflip rot90

# 8. 평가
python -m src.evaluation.evaluate --exp_dir outputs/experiments/{exp_name}
```

## 10. Key Engineering Decisions

- **결절 단위 평가**: 슬라이스 단위 AUC는 같은 결절의 슬라이스 간 상관관계로 성능을 과대평가할 수 있어, validation/test에서 `(subject_id, nodule_idx)` 기준으로 슬라이스 확률을 평균해 결절 단위 AUC를 계산.
- **subject 단위 split**: 결절 단위로 split하면 같은 환자가 train/test에 동시 등장해 data leakage 발생 → subject 단위 StratifiedShuffleSplit으로 해결.
- **z 좌표 정합**: LIDC-IDRI 일부 케이스에서 XML/DICOM 간 z 좌표 단위(deci-mm)·부호 불일치가 존재 → 세 가지 케이스를 모두 감지해 자동 보정하는 `align_center_z()` 구현.
- **Grad-CAM 공정 비교**: 모델별 target layer가 다르면 비교가 무효함을 발견하고, CBAM 모델은 spatial attention map 직접 시각화로 전환. `nn.ReLU(inplace=True)`가 backward hook과 충돌하는 이슈도 확인 및 처리.

## 11. Tech Stack

`PyTorch` · `SimpleITK` · `nibabel` · `numpy` · `scikit-learn` · `pydicom`

## 12. References

- Al-Shabi, M., Lee, H. K., & Tan, M. (2019). Gated-dilated networks for lung nodule classification in CT scans. *IEEE Access*, 7, 178827-178838.
- Woo, S., Park, J., Lee, J. Y., & Kweon, I. S. (2018). CBAM: Convolutional block attention module. *ECCV*.
- Shen, W., et al. (2017). Multi-crop convolutional neural networks for lung nodule malignancy suspiciousness classification. *Pattern Recognition*.
- Liu, et al. (2024). 3D Attention Gated CNN for pulmonary nodule classification.
- Qin, Y., et al. (2021). Relationship between pulmonary nodule malignancy and surrounding pleurae, airways and vessels. *arXiv:2106.12991*.
- Snoeckx, A., et al. (2018). Evaluation of the solitary pulmonary nodule. *Insights into Imaging*, 9(1), 73-86.

---

## Team

| 역할 | 이름 |
|---|---|
| 팀장 | 홍채은 |
| 부팀장 | 허다온 |
| 팀원 | 김종호 · 이상욱 · 이유준 · 이준용 |
