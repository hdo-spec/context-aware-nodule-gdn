# src/models/models.py
#
# ─── 모델 구성 ────────────────────────────────────────────────────────────────
#   GDN        : Gated Dilated Network — 소형 결절 특화, crop_size=32
#   ConvNeXt   : ConvNeXt-Tiny 변형 — 단일 브랜치, crop_size=64
#   DualConvNeXt : ConvNeXt 두 브랜치 (small=32, large=96) → feature concat
#
# ─── 공통 설계 원칙 ───────────────────────────────────────────────────────────
#   - 입력: (B, 3, H, W)  ← 2.5D: z-1 / z / z+1 슬라이스 스택
#   - 출력: (B, 1)        ← 이진 분류 logit (BCEWithLogitsLoss 용)
#   - num_classes=1 고정  ← pos_weight로 클래스 불균형 처리하는 기존 설계와 일관성 유지
#   - global max pool     ← 결절처럼 국소 병변에서 배경에 희석되지 않게

import torch
import torch.nn as nn


# ══════════════════════════════════════════════════════════════════════════════
# CBAM (Convolutional Block Attention Module)
# ══════════════════════════════════════════════════════════════════════════════

class ChannelAttention(nn.Module):
    """
    채널 어텐션: 어떤 채널이 중요한지 학습.
    GlobalAvgPool + GlobalMaxPool → 공유 MLP → sigmoid → 합산

    GDLayer alpha와의 역할 분리:
      alpha      : foreground/background 스트림 분기 (dilation 선택)
      CBAM 채널  : concat 이후 채널 중요도 재보정
    """
    def __init__(self, in_ch: int, reduction: int = 16):
        super().__init__()
        mid_ch = max(1, in_ch // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # 두 풀링 결과에 동일한 MLP 공유 적용 (파라미터 절감)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, in_ch, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg  = self.mlp(self.avg_pool(x))   # (B, C, 1, 1)
        max_ = self.mlp(self.max_pool(x))   # (B, C, 1, 1)
        return self.sigmoid(avg + max_)      # (B, C, 1, 1) 브로드캐스트용


class SpatialAttention(nn.Module):
    """
    공간 어텐션: 어떤 위치가 중요한지 학습.
    채널 방향 AvgPool + MaxPool → conv(7×7) → sigmoid
    """
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv    = nn.Conv2d(2, 1, kernel_size=kernel_size,
                                 padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg  = x.mean(dim=1, keepdim=True)          # (B, 1, H, W)
        max_ = x.max(dim=1, keepdim=True).values    # (B, 1, H, W)
        return self.sigmoid(self.conv(torch.cat([avg, max_], dim=1)))  # (B, 1, H, W)


class CBAM(nn.Module):
    """
    CBAM: Channel Attention → Spatial Attention 순서로 적용.

    GDLayer alpha와의 역할 분리:
      alpha  : foreground/background 스트림 분기 (dilation 선택)
      CBAM   : 채널 중요도 + 공간 위치 중요도 보정

    배치 위치: drop 직후 → dropout으로 노이즈를 줄인 뒤 어텐션이
               더 안정적인 feature를 보고 판단.

    Args:
        in_ch    : 입력 채널 수
        reduction: 채널 어텐션 압축 비율 (기본값 16)
    """
    def __init__(self, in_ch: int, reduction: int = 16):
        super().__init__()
        self.channel = ChannelAttention(in_ch, reduction)
        self.spatial  = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.channel(x)   # 채널 어텐션
        x = x * self.spatial(x)   # 공간 어텐션
        return x


# ══════════════════════════════════════════════════════════════════════════════
# GDN (Gated Dilated Network)
# ══════════════════════════════════════════════════════════════════════════════

class GDLayer(nn.Module):
    """
    Gated Dilated Layer.

    구조:
      1) Context-Aware sub-network → alpha 스칼라 계산
           conv(1ch) → ReLU → GlobalAvgPool → conv(1ch) → Sigmoid
      2) alpha로 입력을 두 스트림으로 분기
           x1 = alpha * x          (foreground 집중)
           x2 = (1 - alpha) * x    (background 집중)
      3) 각 스트림에 서로 다른 dilation conv 적용
           d1: dilation=1 (지역 텍스처)
           d2: dilation=3 (넓은 수용야)
      4) channel-wise concat → out_ch 유지

    Args:
        in_ch  : 입력 채널 수
        out_ch : 출력 채널 수 (반드시 짝수 — d1/d2가 절반씩 담당)
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        assert out_ch % 2 == 0, "out_ch must be even (split equally between d1/d2)"

        branch_ch = out_ch // 2

        # ── Context-Aware sub-network ─────────────────────────────────────────
        # 입력 전체를 1채널로 압축 → GlobalAvgPool → sigmoid로 alpha [0,1] 생성
        # alpha가 1에 가까울수록 x1(foreground) 강조, 0에 가까울수록 x2(background) 강조
        self.context_conv = nn.Conv2d(in_ch, 1, kernel_size=3, padding=1, bias=True)
        self.context_relu = nn.ReLU(inplace=True)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)       # (B, 1, 1, 1) 스칼라
        self.context_fc = nn.Conv2d(1, 1, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

        # ── Dilated Conv 두 브랜치 ────────────────────────────────────────────
        # dilation=1: 지역 텍스처 (결절 경계, 내부 밀도)
        self.conv_d1 = nn.Conv2d(
            in_ch, branch_ch,
            kernel_size=3, stride=1, padding=1, dilation=1, bias=True
        )
        # dilation=3: 더 넓은 수용야 (결절 주변 맥락)
        self.conv_d2 = nn.Conv2d(
            in_ch, branch_ch,
            kernel_size=3, stride=1, padding=3, dilation=3, bias=True
        )
        self.relu = nn.ReLU(inplace=True)

        # forward 전 안전 초기화 (engine.py의 last_alpha is not None 체크를 위해)
        self.last_alpha: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # alpha 계산: (B, 1, 1, 1) 브로드캐스트 가능한 스칼라
        alpha = self.context_conv(x)
        alpha = self.context_relu(alpha)
        alpha = self.global_avg_pool(alpha)
        alpha = self.context_fc(alpha)
        alpha = self.sigmoid(alpha)
        self.last_alpha = alpha.detach().cpu()

        # 입력 분기
        x1 = alpha * x           # foreground 스트림
        x2 = (1.0 - alpha) * x  # background 스트림

        # 각 스트림에 dilation conv
        d1 = self.relu(self.conv_d1(x1))   # (B, branch_ch, H, W)
        d2 = self.relu(self.conv_d2(x2))   # (B, branch_ch, H, W)

        return torch.cat([d1, d2], dim=1)  # (B, out_ch, H, W)


class GDN(nn.Module):
    """
    Gated Dilated Network for nodule malignancy classification.

    입력: (B, 3, H, W)  — 2.5D (z-1/z/z+1 슬라이스), crop_size=32
    출력: (B, 1)        — 이진 분류 logit

    구성:
      GDLayer × 5 + Dropout × 3
      → GlobalMaxPool  (국소 결절 신호 보존, 배경 희석 방지)
      → BN → FC

    채널 구성: 3 → 32 → 32 → 64 → 64 → 64 → pool → fc
    
    CBAM 위치: drop 직후
      dropout으로 노이즈를 줄인 뒤 어텐션이 안정적인 feature를 보고 판단.
      gd1 → gd2 → drop1 → [cbam1] → gd3 → gd4 → drop2 → [cbam2] → gd5 → drop3

      # 변경 후
      gd1 → gd2 → drop1 → [cbam1] → gd3 → gd4 → drop2 → gd5 → drop3 → [cbam2]
    
    Args:
        in_ch      : 입력 채널 수 (기본값 3, 2.5D)
        num_classes: 출력 클래스 수 (기본값 1, 이진 분류)
        use_cbam1  : drop1 이후 CBAM 적용 여부 (EXP-B, EXP-D)
        use_cbam2  : drop2 이후 CBAM 적용 여부 (EXP-C, EXP-D)

    실험 설계:
      EXP-A: use_cbam1=False, use_cbam2=False  (CBAM 없음, 기존 베이스라인)
      EXP-B: use_cbam1=True,  use_cbam2=False  (CBAM #1만, gd1+2 이후)
      EXP-C: use_cbam1=False, use_cbam2=True   (CBAM #2만, gd3+4 이후)
      EXP-D: use_cbam1=True,  use_cbam2=True   (CBAM 둘 다)
      EXP-E: use_cbam1=False, use_cbam2=True, use_cbam_final=True
    """
    def __init__(self, in_ch: int = 3, num_classes: int = 1,
                 use_cbam1: bool = False, use_cbam2: bool = False, use_cbam_final: bool = False,
                 use_gd4b=False, use_gd6=False):
        super().__init__()

        # ── Feature Extraction ────────────────────────────────────────────────
        self.gd1 = GDLayer(in_ch, 32)
        self.gd2 = GDLayer(32, 32)
        self.drop1 = nn.Dropout(p=0.25)
        # cbam1: None이면 forward에서 skip (파라미터 추가 없음)
        self.cbam1 = CBAM(32) if use_cbam1 else None

        self.gd3 = GDLayer(32, 64)
        self.gd4 = GDLayer(64, 64)
        self.gd4b = GDLayer(64, 64) if use_gd4b else None
        self.drop2 = nn.Dropout(p=0.25)
        self.cbam2 = CBAM(64) if use_cbam2 else None

        self.gd5 = GDLayer(64, 64)
        self.gd6 = GDLayer(64, 64) if use_gd6 else None
        self.drop3 = nn.Dropout(p=0.5)
        self.cbam_final = CBAM(64) if use_cbam_final else None   # ← 추가

        # ── Classifier ────────────────────────────────────────────────────────
        # GlobalMaxPool: 결절이 어디에 있든 가장 강한 반응을 살려냄
        # AvgPool은 소형 결절 신호가 배경에 희석되는 단점이 있음
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)

        # BN: GlobalMaxPool 이후 feature 스케일 안정화
        self.bn = nn.BatchNorm1d(64)
        self.fc = nn.Linear(64, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.gd1(x)        # (B, 32, H, W)
        out = self.gd2(out)      # (B, 32, H, W)
        out = self.drop1(out)
        if self.cbam1 is not None:
            out = self.cbam1(out)   # 채널+공간 어텐션

        out = self.gd3(out)      # (B, 64, H, W)
        out = self.gd4(out)      # (B, 64, H, W)
        if self.gd4b is not None:
            out = self.gd4b(out)  # ← 추가
        out = self.drop2(out)
        if self.cbam2 is not None:
            out = self.cbam2(out)   # 채널+공간 어텐션

        out = self.gd5(out)      # (B, 64, H, W)
        if self.gd6 is not None:
            out = self.gd6(out)
        out = self.drop3(out)
        if self.cbam_final is not None:
            out = self.cbam_final(out) # gd5 이후 ← 추가

        out = self.global_max_pool(out)   # (B, 64, 1, 1)
        out = out.view(out.size(0), -1)   # (B, 64)
        out = self.bn(out)
        out = self.fc(out)                # (B, 1)

        return out


# ══════════════════════════════════════════════════════════════════════════════
# ConvNeXt (단일 브랜치)
# ══════════════════════════════════════════════════════════════════════════════

class LayerNorm2d(nn.Module):
    """
    (B, C, H, W) 입력에 LayerNorm 적용하는 래퍼.
    ConvNeXt는 BatchNorm 대신 LayerNorm을 사용하므로 필요.
    채널 축을 마지막으로 permute → norm → 원복.
    """
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)   # (B, H, W, C)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)   # (B, C, H, W)
        return x


class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt 기본 블록 (Liu et al. 2022).

    구조:
      DWConv(7×7) → LayerNorm → PWLinear(dim→4*dim) → GELU
      → PWLinear(4*dim→dim) → gamma scaling → residual add

    핵심 설계:
      - Depthwise conv: 채널별 독립 공간 특징 추출 (파라미터 효율적)
      - Pointwise Linear: 채널 간 정보 혼합 (MLP처럼 작동)
      - gamma: 학습 가능한 스케일 파라미터 (초기값 1e-6 → 학습 초기 안정성)
      - residual: 기울기 소실 방지

    Args:
        dim: 채널 수 (입출력 동일)
    """
    def __init__(self, dim: int):
        super().__init__()

        # Depthwise conv: 각 채널 독립적으로 7×7 공간 특징 추출
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)

        self.norm = nn.LayerNorm(dim)

        # Inverted bottleneck: dim → 4*dim → dim (채널 확장 후 압축)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

        # 학습 가능한 스케일: 초기값 1e-6으로 residual 초기 기여를 최소화
        self.gamma = nn.Parameter(1e-6 * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.dwconv(x)                # (B, C, H, W) 공간 특징
        out = out.permute(0, 2, 3, 1)       # (B, H, W, C) — Linear 적용 위해 permute
        out = self.norm(out)
        out = self.pwconv1(out)             # (B, H, W, 4C)
        out = self.act(out)
        out = self.pwconv2(out)             # (B, H, W, C)
        out = self.gamma * out              # 스케일 조정
        out = out.permute(0, 3, 1, 2)       # (B, C, H, W)

        return identity + out               # residual


class ConvNeXt(nn.Module):
    """
    ConvNeXt-Tiny 변형 — 단일 브랜치, 의료 소형 패치용.

    원 논문(ImageNet)과의 차이:
      - stem stride=2  (원 논문 stride=4 → 소형 패치에선 과도한 다운샘플)
      - depths [2,2,6,2] (Tiny 기준, 원 논문 Tiny=[3,3,9,3]보다 가벼움)
      - GlobalMaxPool  (원 논문 AvgPool → 국소 결절 신호 보존)
      - num_classes=1  (이진 분류, BCEWithLogitsLoss)

    입력: (B, 3, H, W)  — 2.5D, crop_size=64
    출력: (B, 1)        — logit

    공간 해상도 흐름 (입력 64×64 기준):
      stem(stride=2) → 32×32
      down1(stride=2) → 16×16
      down2(stride=2) → 8×8
      down3(stride=2) → 4×4
      GlobalMaxPool  → 1×1
    """
    def __init__(self, in_ch: int = 3, num_classes: int = 1):
        super().__init__()

        dims   = [64, 128, 256, 512]
        depths = [2, 2, 6, 2]           # ConvNeXt-Tiny 기준

        # ── Stem ──────────────────────────────────────────────────────────────
        # stride=2: 소형 패치에서 stride=4(원 논문)는 정보 손실이 너무 큼
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, dims[0], kernel_size=3, stride=2, padding=1),
            LayerNorm2d(dims[0])
        )

        # ── Stage + Downsample ────────────────────────────────────────────────
        self.stage1 = self._make_stage(dims[0], depths[0])
        self.down1  = self._make_downsample(dims[0], dims[1])

        self.stage2 = self._make_stage(dims[1], depths[1])
        self.down2  = self._make_downsample(dims[1], dims[2])

        self.stage3 = self._make_stage(dims[2], depths[2])
        self.down3  = self._make_downsample(dims[2], dims[3])

        self.stage4 = self._make_stage(dims[3], depths[3])

        # ── Classifier ────────────────────────────────────────────────────────
        # GlobalMaxPool: 결절 국소 신호 보존
        self.pool = nn.AdaptiveMaxPool2d(1)     # self.global_max_pool로 통일
        self.norm = nn.LayerNorm(dims[-1])
        self.fc   = nn.Linear(dims[-1], num_classes)

        self._init_weights()

    def _init_weights(self):
        """
        ConvNeXt 가중치 초기화.
        ConvNeXtBlock의 gamma는 이미 1e-6으로 초기화됨.
        Conv2d, Linear: trunc_normal (std=0.02)
        LayerNorm, BatchNorm2d: weight=1, bias=0 (표준 초기화)
        DualConvNeXt에서 fc를 Identity로 교체 후 load_state_dict 시 shape 불일치 방지.
        """
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
                nn.init.zeros_(m.bias)
                nn.init.ones_(m.weight)

    def _make_stage(self, dim: int, depth: int) -> nn.Sequential:
        return nn.Sequential(*[ConvNeXtBlock(dim) for _ in range(depth)])

    def _make_downsample(self, in_dim: int, out_dim: int) -> nn.Sequential:
        # LayerNorm → stride=2 conv: 해상도 절반, 채널 2배
        return nn.Sequential(
            LayerNorm2d(in_dim),
            nn.Conv2d(in_dim, out_dim, kernel_size=2, stride=2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.stem(x)         # (B, 64,  32, 32)

        out = self.stage1(out)     # (B, 64,  32, 32)
        out = self.down1(out)      # (B, 128, 16, 16)

        out = self.stage2(out)     # (B, 128, 16, 16)
        out = self.down2(out)      # (B, 256,  8,  8)

        out = self.stage3(out)     # (B, 256,  8,  8)
        out = self.down3(out)      # (B, 512,  4,  4)

        out = self.stage4(out)     # (B, 512,  4,  4)

        out = self.pool(out)       # (B, 512, 1, 1)
        out = out.flatten(1)       # (B, 512)
        out = self.norm(out)
        out = self.fc(out)         # (B, 1)

        return out


class ConvNeXtSmall(nn.Module):
    """
    DualConvNeXt small branch 전용 (입력 32×32).
    down3 제거 → stage4 출력 4×4 유지.
    공간 해상도: 32→stem(16)→d1(8)→d2(4)→pool(1)
    """
    def __init__(self, in_ch: int = 3):
        super().__init__()
        dims   = [64, 128, 256, 512]
        depths = [2, 2, 6, 2]

        self.stem   = nn.Sequential(
            nn.Conv2d(in_ch, dims[0], kernel_size=3, stride=2, padding=1),
            LayerNorm2d(dims[0])
        )
        self.stage1 = self._make_stage(dims[0], depths[0])
        self.down1  = self._make_downsample(dims[0], dims[1])
        self.stage2 = self._make_stage(dims[1], depths[1])
        self.down2  = self._make_downsample(dims[1], dims[2])
        self.stage3 = self._make_stage(dims[2], depths[2])
        # down3 없음
        self.pool   = nn.AdaptiveMaxPool2d(1)
        self.norm   = nn.LayerNorm(dims[2])   # 256
        self.fc     = nn.Identity()           # DualConvNeXt에서 교체용

        self._init_weights()

    def _make_stage(self, dim, depth):
        return nn.Sequential(*[ConvNeXtBlock(dim) for _ in range(depth)])

    def _make_downsample(self, in_dim, out_dim):
        return nn.Sequential(
            LayerNorm2d(in_dim),
            nn.Conv2d(in_dim, out_dim, kernel_size=2, stride=2)
        )

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
                nn.init.zeros_(m.bias)
                nn.init.ones_(m.weight)

    def forward(self, x):
        out = self.stem(x)
        out = self.stage1(out)
        out = self.down1(out)
        out = self.stage2(out)
        out = self.down2(out)
        out = self.stage3(out)   # (B, 256, 4, 4)
        out = self.pool(out)     # (B, 256, 1, 1)
        out = out.flatten(1)     # (B, 256)
        out = self.norm(out)
        return self.fc(out)
    
# ══════════════════════════════════════════════════════════════════════════════
# DualConvNeXt (이중 브랜치)
# ══════════════════════════════════════════════════════════════════════════════

class DualConvNeXt(nn.Module):
    """
    Dual-branch ConvNeXt — LIDCDualDataset과 쌍으로 사용.

    두 브랜치가 각각 다른 해상도 패치를 처리:
      small_branch : crop_size=32  → 결절 내부 세부 텍스처
      large_branch : crop_size=96  → 결절 + 주변 맥락 (spiculation 등)

    두 브랜치의 feature를 concat 후 fc로 최종 분류.
    파라미터 공유 없음 — 두 스케일이 서로 다른 패턴을 독립적으로 학습.

    입력:
      x_small: (B, 3, 32, 32)
      x_large: (B, 3, 96, 96)
    출력:
      (B, 1) logit

    공간 해상도 흐름:
      small: 32 → stem(16) → d1(8) → d2(4) → d3(2) → pool(1)
      large: 96 → stem(48) → d1(24) → d2(12) → d3(6) → pool(1)
    """
    def __init__(self, num_classes: int = 1):
        super().__init__()

        # 두 브랜치 독립 인스턴스 — 파라미터 공유 없음
        self.small_branch = ConvNeXtSmall(in_ch=3)  # 출력 256, down3 없음
        # small branch에서 입력이 (b, 3, 32, 32)일 때 convNeXt stage 4 출력이 (b, 512, 2, 2) -> GAP -> (b, 512, 1, 1) 하면
        # 공간 정보가 4픽셀밖에 없는 상태에서 최대값 1개만 남게 된다.
        # 세부 텍스처 정보가 너무 일찍 소멸될 수도 있어서 stage 3까지만 쓰거나, down3를 제거해야 될 수도 있음
        self.large_branch = ConvNeXt(in_ch=3, num_classes=num_classes)
        # 각 브랜치의 fc를 제거하고 feature를 concat한 뒤 새 fc로 분류
        # ConvNeXt.fc 이전 feature 차원: 512

        # 브랜치별 fc를 Identity로 교체 → forward에서 norm까지만 통과
        self.large_branch.fc = nn.Identity()         # large만 fc 교체

        feat_dim = 256 + 512                         # 768

        # concat된 feature → 분류
        # *수정 (확인 후) - 삭제
        # self.fusion_norm = nn.LayerNorm(feat_dim)
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, x_small: torch.Tensor, x_large: torch.Tensor) -> (torch.Tensor):
        feat_small = self.small_branch(x_small)   # (B, 256)
        feat_large  = self.large_branch(x_large)   # (B, 512)

        feat = torch.cat([feat_small, feat_large], dim=1)   # (B, 768)
        # 각 브랜치 내부에서 self.norm을 통과하고 나왔기 떄문에 concat 후 다시 LN을 적용하면 효과가 크지 않음
        out  = self.fc(feat)                                 # (B, 1) # fusion_norm 없이 바로 fc

        return out


# ══════════════════════════════════════════════════════════════════════════════
# 빠른 shape 검증
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── GDN EXP-A~D 파라미터 수 비교 ────────────────────────────────────────
    print("=== GDN CBAM 실험 설계 ===")
    configs = [
        ('EXP-A (baseline)',    False, False),
        ('EXP-B (cbam1 only)',  True,  False),
        ('EXP-C (cbam2 only)',  False, True),
        ('EXP-D (cbam1+cbam2)', True,  True),
    ]
    x = torch.randn(4, 3, 32, 32).to(device)
    for name, c1, c2 in configs:
        m   = GDN(in_ch=3, num_classes=1, use_cbam1=c1, use_cbam2=c2).to(device)
        out = m(x)
        n   = sum(p.numel() for p in m.parameters() if p.requires_grad)
        print(f"  {name:<28s} | params={n:>7,} | output={tuple(out.shape)}")
    print(f"  output: {tuple(out.shape)}")   # (4, 1)

    print("\n=== ConvNeXt ===")
    model = ConvNeXt(in_ch=3, num_classes=1).to(device)
    x = torch.randn(4, 3, 64, 64).to(device)
    out = model(x)
    print(f"  input : {tuple(x.shape)}")
    print(f"  output: {tuple(out.shape)}")   # (4, 1)

    print("\n=== DualConvNeXt ===")
    model = DualConvNeXt(num_classes=1).to(device)
    xs = torch.randn(4, 3, 32, 32).to(device)
    xl = torch.randn(4, 3, 96, 96).to(device)
    out = model(xs, xl)
    print(f"  x_small: {tuple(xs.shape)}")
    print(f"  x_large: {tuple(xl.shape)}")
    print(f"  output : {tuple(out.shape)}")  # (4, 1)
