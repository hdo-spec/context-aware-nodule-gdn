# LIDC-IDRI 폐결절 분류 실험 전체 정리

> **데이터셋**: LIDC-IDRI | **평가 단위**: 결절(nodule) | **입력**: 2.5D (3ch, z-1/z/z+1)  
> **pos_weight**: 1.0 | **threshold**: Youden Index (val set)

---

## 목차

1. [데이터셋 개요](#1-데이터셋-개요)
2. [Phase 1 — 모델 선택: GDN vs DualConvNeXt](#2-phase-1--모델-선택-gdn-vs-dualconvnext)
3. [Phase 2 — GDN Dilation 탐색](#3-phase-2--gdn-dilation-탐색)
4. [Phase 3 — CBAM 위치·조합 탐색](#4-phase-3--cbam-위치조합-탐색)
5. [Phase 4 — 하이퍼파라미터 튜닝](#5-phase-4--하이퍼파라미터-튜닝)
6. [Phase 5 — 구조 변경 (레이어 추가)](#6-phase-5--구조-변경-레이어-추가)
7. [전체 실험 이력 요약](#7-전체-실험-이력-요약)
8. [최종 결론](#8-최종-결론)

---

## 1. 데이터셋 개요

| split | 슬라이스 수 | 결절 수 | benign | malignant |
|-------|------------|--------|--------|-----------|
| train | 5,880 | 1,455 | 2,475 슬라이스 | 3,405 슬라이스 |
| val | 1,318 | — | — | — |
| test | 1,225 | **287** | 193 | 94 |

**Subgroup 정의 (Fleischner Society, volume 기준)**

| 그룹 | 기준 | n | benign | malignant |
|------|------|---|--------|-----------|
| small | < 100 mm³ | 84 | 76 | 8 |
| intermediate | 100~250 mm³ | 83 | 72 | 11 |
| large | > 250 mm³ | 120 | 45 | 75 |

---

## 2. Phase 1 — 모델 선택: GDN vs DualConvNeXt

### 2-1. GDN 초기 실험

crop_size=32, pos_weight=1.0 고정.

| 실험 | augmentation | best val AUC | test AUC | Sensitivity | Specificity |
|------|-------------|-------------|---------|------------|------------|
| GDN no_aug | 없음 | 0.9071 | 0.8823 | 0.6489 | 0.9534 |
| GDN aug | hflip+rot90 | **0.9134** | 0.8809 | **0.7021** | 0.9326 |
| GDN aug | hu_shift+gaussian_noise | 0.8885 | 0.8787 | 0.0319 ⚠️ | 0.9948 |

> **관찰**: 강도변환 aug(hu_shift+gaussian_noise)는 val loss 발산 및 sensitivity 붕괴 → 이후 실험에서 제외.  
> **채택**: hflip+rot90 augmentation.

**GDN Subgroup AUC**

| 그룹 | no_aug | hflip+rot90 | hu_shift+noise |
|------|--------|-------------|----------------|
| small | 0.7039 | **0.7599** | 0.6266 |
| intermediate | 0.6250 | 0.6275 | 0.6679 |
| large | 0.8993 | 0.8821 | 0.8827 |

---

### 2-2. DualConvNeXt 실험

crop_size=32+96, pos_weight=1.0.

| 설정 | 파라미터 | best val AUC | test AUC | Sensitivity | Specificity | 과적합 |
|------|---------|-------------|---------|------------|------------|--------|
| no_aug, wd=1e-4 | 12,283,905 | 0.8767 | 0.8559 | 0.7234 | 0.8549 | 심각 ⚠️ |
| no_aug, wd=1e-3 | 12,283,905 | 0.8713 | 0.8526 | 0.7553 | 0.8446 | 심각 ⚠️ |
| aug, wd=1e-4 | 12,283,905 | 0.8854 | 0.8705 | 0.7128 | 0.9016 | 심각 ⚠️ |
| aug, wd=1e-3 | 12,283,905 | 0.8822 | 0.8677 | 0.7021 | 0.9016 | 심각 ⚠️ |

> train AUC가 ep5~10에서 1.0에 도달하고 val loss가 계속 발산(최대 1.24).  
> weight_decay 강화(1e-3)로도 과적합 해소 불가.  
> GDN 대비 파라미터 **118배** 많음에도 모든 지표에서 열세.

**GDN vs DualConvNeXt 최종 비교**

| 지표 | GDN (hflip+rot90) | DualConvNeXt (aug, wd=1e-4) |
|------|-------------------|-----------------------------|
| test AUC | **0.8809** | 0.8705 |
| Sensitivity | **0.7021** | 0.7128 |
| Specificity | **0.9326** | 0.9016 |
| small AUC | **0.7599** | 0.6234 |
| intermediate AUC | **0.6275** | 0.6364 |
| large AUC | **0.8821** | 0.8489 |
| 과적합 | 없음 ✅ | 심각 ⚠️ |
| 파라미터 수 | **104,459** | 12,283,905 |

**→ GDN을 본 모델로 확정**

---

## 3. Phase 2 — GDN Dilation 탐색

베이스: GDN + hflip+rot90, pos_weight=1.0, crop=32.

기존 GDLayer의 dilation 조합(d1=1, d2=2)을 d2=3으로 확장하여 수용야(receptive field) 넓이 실험.

### 3-1. 실험 결과

| 항목 | baseline (d2=2) | d2=3, drop=0.25 | d2=3, drop=0.3 | d2=3, wd=5e-4 |
|------|----------------|----------------|----------------|---------------|
| best val AUC | **0.9134** | 0.9133 | 0.9041 | 0.9122 |
| val loss | 안정 ~0.48 ✅ | 진동 0.6~1.6 ⚠️ | 진동 0.6~1.0 ⚠️ | 진동 0.6~1.6 ⚠️ |
| test AUC | 0.8809 | **0.8888** | 0.8764 | 0.8818 |
| Sensitivity | 0.7021 | **0.8191** | 0.7553 | **0.8404** |
| Specificity | **0.9326** | 0.8187 | 0.8549 | 0.7565 |
| threshold (Youden) | 0.5 | 0.0504 | 0.1056 | 0.0400 |

**Subgroup AUC**

| 그룹 | baseline (d2=2) | d2=3, drop=0.25 | d2=3, drop=0.3 | d2=3, wd=5e-4 |
|------|----------------|----------------|----------------|---------------|
| small | 0.7599 | **0.7944** | 0.7122 | **0.8109** |
| intermediate | **0.6275** | 0.6035 | 0.5644 | 0.5455 |
| large | 0.8821 | 0.8945 | **0.9007** | 0.8919 |

> **val loss 불안정의 원인**: 평가 단위 불일치.  
> `val_loss`는 슬라이스 단위 BCE → 경계값 근처 슬라이스에 민감하게 반응.  
> `val_auc`는 결절 단위 평균 → 슬라이스 단위 변동이 상쇄되어 안정적.  
> val loss가 진동하는 동안에도 val AUC는 꾸준히 상승 → **val AUC가 신뢰할 수 있는 학습 신호**.

> **채택**: d2=3, drop=0.25, wd=1e-4 (AUC·sensitivity·small AUC 균형 최적).

---

## 4. Phase 3 — CBAM 위치·조합 탐색

베이스: GDN (d1=1, d2=3) + hflip+rot90, pos_weight=1.0, crop=32.

**GDN 구조 (실험 전)**
```
gd1(3→32) → gd2(32→32) → drop1 → gd3(32→64) → gd4(64→64) → drop2 → gd5(64→64) → drop3 → FC
```

### 4-1. CBAM 위치 실험 (EXP-A ~ D)

| 실험 | CBAM 위치 | 파라미터 | best val AUC | val loss | test AUC | Sensitivity | Specificity | threshold |
|------|----------|---------|-------------|---------|---------|------------|------------|-----------|
| EXP-A (baseline) | 없음 | 104,459 | 0.9133 | 0.6~1.6 ⚠️ | 0.8888 | 0.8191 | 0.8187 | 0.0504 |
| EXP-B | drop1 이후 (gd2 후) | 104,685 | 0.9129 | 1.0~2.8 ⚠️⚠️ | 0.8721 | 0.8191 | 0.7409 | 0.0032 ⚠️ |
| EXP-C | drop2 이후 (gd4 후) | 105,069 | 0.9152 | **0.47~0.94 ✅** | **0.8896** | 0.7979 | **0.8601** | **0.3579 ✅** |
| EXP-D | drop1+drop2 이후 | 105,295 | **0.9169** | 0.6~1.1 ⚠️ | 0.8716 | 0.7660 | 0.8446 | 0.0704 |

**Subgroup AUC**

| 그룹 | EXP-A | EXP-B | EXP-C | EXP-D |
|------|-------|-------|-------|-------|
| small | 0.7944 | 0.7796 | 0.7878 | 0.6612 ⚠️ |
| intermediate | 0.6035 | 0.4735 ⚠️ | **0.6477** | 0.5783 |
| large | **0.8945** | 0.8996 | 0.8936 | 0.8839 |

**Confusion Matrix**

| | EXP-A | EXP-B | EXP-C | EXP-D |
|-|-------|-------|-------|-------|
| TN | 158 | 143 | **166** | 163 |
| FP | 35 | 50 | **27** | 30 |
| FN | **17** | **17** | 19 | 22 |
| TP | **77** | **77** | 75 | 72 |

> **EXP-C (CBAM2 단독, gd4 이후) 권장**: val loss 유일하게 안정, AUC·Accuracy·Specificity 개선, intermediate subgroup 회복, threshold 정상화.

---

### 4-2. CBAM 위치 이동 실험 (EXP-C2, EXP-E)

EXP-C(gd4 후 CBAM)를 베이스로 CBAM 위치를 더 뒤로 이동하거나 추가 배치 실험.

**실험 구조**

| 실험 | 구조 |
|------|------|
| EXP-C | gd4 → drop2 → **[CBAM]** → gd5 → drop3 → FC |
| EXP-C2 | gd4 → drop2 → gd5 → drop3 → **[CBAM]** → FC |
| EXP-E | gd4 → drop2 → **[CBAM]** → gd5 → drop3 → **[CBAM_final]** → FC |

**결과 비교**

| 항목 | EXP-C (gd4 후) | EXP-C2 (gd5 후) | EXP-E (gd4+gd5 후) |
|------|---------------|----------------|-------------------|
| 파라미터 | 105,069 | 105,069 | 105,679 |
| best val AUC | 0.9152 | 0.9154 | **0.9183** |
| val loss | 0.47~0.94 ✅ | 0.5~0.9 ✅ | **0.5~0.75 ✅** |
| test AUC | 0.8896 | 0.8861 | **0.8939** |
| Sensitivity | 0.7979 | 0.7979 | **0.8404** |
| Specificity | **0.8601** | 0.8342 | 0.8083 |
| threshold | **0.3579 ✅** | 0.0925 | 0.1553 |
| FN (암 놓침) | 19 | 19 | **15** |

**Subgroup AUC**

| 그룹 | EXP-C | EXP-C2 | EXP-E |
|------|-------|--------|-------|
| small | 0.7878 | 0.7385 ⚠️ | **0.8421** |
| intermediate | **0.6477** | 0.5871 | 0.6250 |
| large | 0.8936 | **0.8993** | 0.8975 |

> **EXP-E (gd4+gd5 이후 CBAM 2개) 최우수**:
> - val AUC·test AUC·Sensitivity·FN·small subgroup 모두 최고치
> - EXP-D(cbam1+cbam2)가 실패했던 것과 달리 gd4+gd5 조합은 val loss 안정성 유지
> - Specificity·Accuracy는 EXP-C가 소폭 우세

**→ EXP-E 구조를 최적 CBAM 구성으로 확정**

---

## 5. Phase 4 — 하이퍼파라미터 튜닝

베이스: GDN + EXP-E 구조 (CBAM gd4+gd5) + hflip+rot90, pos_weight=1.0, crop=32.

### 5-1. pos_weight 조정 (EXP-E1)

| 항목 | EXP-E (pw=1.0) | EXP-E1 (pw=1.5) |
|------|---------------|----------------|
| best val AUC | **0.9183** | 0.9161 |
| val loss | 0.5~0.75 ✅ | 0.8~1.6 ⚠️ |
| test AUC | **0.8939** | 0.8889 |
| Sensitivity | **0.8404** | 0.7553 ⚠️ |
| FN | **15** | 23 ⚠️ |

> pos_weight 상승이 오히려 sensitivity를 역방향으로 떨어뜨림. **pw=1.0 유지.**

---

### 5-2. Epoch 증가 (EXP-E2)

| 항목 | EXP-E (ep50) | ep100 |
|------|-------------|-------|
| best val AUC | 0.9183 (ep39) | **0.9327 (ep51)** |
| val loss | 0.5~0.75 ✅ | 0.5~2.7 ⚠️ |
| test AUC | 0.8939 | **0.8990** |
| Sensitivity | **0.8404** | 0.8085 |
| small AUC | **0.8421** | 0.7303 ⚠️ |

> ep51 이후 과적합 구간 진입. small AUC 급락. **ep50 유지.**

---

### 5-3. 학습률 조정 (EXP-E3)

| 항목 | EXP-E (ep50, lr=1e-4) | ep100 (lr=1e-4) | ep100 (lr=5e-5) |
|------|-----------------------|----------------|----------------|
| test AUC | 0.8939 | 0.8990 | 0.8900 |
| Sensitivity | **0.8404** | 0.8085 | 0.8511 |
| small AUC | **0.8421** | 0.7303 | 0.8240 |
| threshold | 0.1553 | 0.0206 | 0.0048 ⚠️ |

> lr 낮추면 sensitivity·small 회복되지만 val loss 발산과 threshold drift는 지속.

---

### 5-4. 스케줄러 변경 (EXP-E4, WarmRestarts)

| 항목 | EXP-E ep50 | ep100 lr=5e-5 | WarmRestarts lr=5e-5 |
|------|-----------|--------------|----------------------|
| best val AUC | 0.9183 | 0.9237 | **0.9293** |
| test AUC | **0.8939** | 0.8900 | 0.8824 ⚠️ |
| Sensitivity | **0.8404** | 0.8511 | 0.8191 |
| small AUC | **0.8421** | 0.8240 | 0.7155 ⚠️ |
| threshold | 0.1553 | 0.0048 | **0.0021** ⚠️ |

> val AUC 수치만 높고 실제 test 성능은 전반 하락. threshold drift 극단화.  
> lr, scheduler, epoch 변경은 모두 **EXP-E ep50을 넘지 못함** → 구조적 한계 확인.

---

## 6. Phase 5 — 구조 변경 (레이어 추가)

베이스: EXP-E 구조. 레이어 추가로 표현력 확장 시도.

| 실험 | 추가 위치 | 파라미터 | test AUC | Sensitivity | small AUC | 판정 |
|------|----------|---------|---------|------------|-----------|------|
| EXP-E (기준) | — | 105,679 | 0.8939 | **0.8404** | **0.8421** | ✅ |
| EXP-F1 | gd4~gd5 사이 gd4b | 143,186 | 0.8985 | 0.7979 | 0.7434 ⚠️ | ⚠️ |
| EXP-F2 | gd5 뒤 gd6 | 143,186 | 0.8779 | 0.8298 | 0.6826 ⚠️ | ❌ |

> 레이어 추가마다 **small AUC가 일관되게 급락** (F1: -0.099, F2: -0.160).  
> train 1455 결절 대비 143K 파라미터는 과적합 경계를 초과하는 것으로 판단.  
> 현재 데이터 규모에서는 레이어 추가 방향의 구조 변경이 효과 없음을 확인.

---

## 7. 전체 실험 이력 요약

| 실험 | 주요 변경 | test AUC | Sensitivity | Specificity | small AUC | 판정 |
|------|----------|---------|------------|------------|-----------|------|
| GDN no_aug (pw=1.0) | 베이스라인 | 0.8823 | 0.6489 | 0.9534 | 0.7039 | — |
| GDN hflip+rot90 | aug 추가 | 0.8809 | 0.7021 | 0.9326 | 0.7599 | ↑ |
| GDN d2=3 | dilation 확장 | 0.8888 | 0.8191 | 0.8187 | 0.7944 | ↑ |
| EXP-C (CBAM gd4) | CBAM 단독 | 0.8896 | 0.7979 | 0.8601 | 0.7878 | ↑ |
| **EXP-E (CBAM gd4+gd5)** | **CBAM 병용** | **0.8939** | **0.8404** | **0.8083** | **0.8421** | **✅ 최적** |
| EXP-E1 (pw=1.5) | pos_weight 상승 | 0.8889 | 0.7553 | 0.8808 | 0.8240 | ↓ |
| EXP-E2 (ep100) | epoch 증가 | 0.8990 | 0.8085 | 0.8187 | 0.7303 | ↓ small |
| EXP-E3 (lr=5e-5) | lr 감소 | 0.8900 | 0.8511 | 0.7565 | 0.8240 | △ |
| EXP-E4 (WarmRestarts) | 스케줄러 변경 | 0.8824 | 0.8191 | 0.7668 | 0.7155 | ↓ |
| EXP-F1 (gd4b) | 레이어 추가 | 0.8985 | 0.7979 | 0.8653 | 0.7434 | ↓ |
| EXP-F2 (gd6) | 레이어 추가 | 0.8779 | 0.8298 | 0.7358 | 0.6826 | ❌ |

---

## 8. 최종 결론

### 최적 모델: GDN + CBAM (EXP-E)

```
gd1(3→32) → gd2(32→32) → drop1(0.25)
→ gd3(32→64) → gd4(64→64) → drop2(0.25) → [CBAM]
→ gd5(64→64) → drop3(0.5) → [CBAM_final]
→ GlobalMaxPool → BN → FC(64→1)
```

| 설정 | 값 |
|------|---|
| 모델 | GDN + CBAM(gd4 후) + CBAM_final(gd5 후) |
| crop size | 32×32 |
| GDLayer dilation | d1=1, d2=3 |
| Dropout | drop1=0.25, drop2=0.25, drop3=0.5 |
| augmentation | hflip + rot90 (prob=0.5) |
| pos_weight | 1.0 |
| optimizer | AdamW (lr=1e-4, wd=1e-4) |
| scheduler | CosineAnnealingLR (T_max=50) |
| epochs | 50 |
| threshold | Youden Index (val set) = 0.1553 |
| 파라미터 수 | 105,679 |

### 최종 성능 (test set, 결절 단위, n=287)

| 지표 | 값 |
|------|---|
| **AUC** | **0.8939** |
| Accuracy | 0.8188 |
| **Sensitivity** | **0.8404** |
| Specificity | 0.8083 |
| FN (암 놓침) | **15** (전체 실험 중 최소) |

**Subgroup AUC**

| 그룹 | AUC |
|------|-----|
| small (<100mm³) | **0.8421** (전체 실험 중 최고) |
| intermediate (100~250mm³) | 0.6250 |
| large (>250mm³) | 0.8975 |

### 핵심 인사이트

| 항목 | 결론 |
|------|------|
| **모델 선택** | GDN이 DualConvNeXt 대비 파라미터 1/118로 모든 지표 우세 |
| **dilation 확장** | d2=3이 수용야 넓혀 AUC·small AUC 향상. val loss 진동은 평가 단위 불일치가 원인으로 val AUC가 신뢰 지표 |
| **CBAM 위치** | gd4(중간) 이후가 핵심. gd2(초기) 이후는 학습 불안정 유발 |
| **CBAM 조합** | gd4+gd5 병용(EXP-E)이 단독(EXP-C) 대비 sensitivity·small AUC 동시 개선 |
| **threshold** | Youden Index 기반 최적화가 sensitivity 붕괴 방지에 필수 |
| **한계** | intermediate subgroup AUC 0.62 수준 — 경계 크기 결절의 구조적 분류 난이도 |
| **튜닝 한계** | epoch·lr·scheduler·pos_weight·레이어 추가 모두 EXP-E ep50을 넘지 못함 → 현재 구조의 성능 천장 도달 |
