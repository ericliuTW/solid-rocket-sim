"""
固態火箭推進概念模擬平台 — 常數與預設參數
Solid Rocket Propulsion Conceptual Simulation Platform — Constants & Defaults

╔══════════════════════════════════════════════════════════════════════╗
║  本工具僅供教學、概念分析與趨勢比較使用，                            ║
║  不能直接用於真實推進器設計、製造或安全驗證。                         ║
║  This tool is for EDUCATIONAL and CONCEPTUAL use only.              ║
║  NOT suitable for real engine design or safety certification.       ║
╚══════════════════════════════════════════════════════════════════════╝
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ── 幾何類型 ──────────────────────────────────────────────────────────
class GrainType(Enum):
    """藥柱幾何類型"""
    CYLINDRICAL = "cylindrical"          # 圓柱中心穿孔
    BATES = "bates"                      # BATES 多段式
    END_BURNER = "end_burner"            # 端面燃燒
    STAR = "star"                        # 星形（簡化）
    MOON_BURNER = "moon_burner"          # 偏心孔（簡化）


# ── 模擬時間步進 ──────────────────────────────────────────────────────
DEFAULT_TIME_STEPS: int = 500
BURN_REGRESSION_STEPS: int = 500          # 燃燒回歸離散步數

# ── 示意用推進劑特性（概念級，非真實值） ──────────────────────────────
# 這些數值僅用於「示意估算模式」，幫助使用者建立直覺，
# 絕對不可作為真實設計依據。
PROPELLANT_DENSITY_KG_M3: float = 1750.0         # 示意密度 kg/m³
BURN_RATE_COEFFICIENT_A: float = 1.007e-4          # a in r = a·P^n (m/s, P in Pa) — 示意值 (≈ KNSB 等級)
BURN_RATE_EXPONENT_N: float = 0.319               # 壓力指數 — 示意值
CHARACTERISTIC_VELOCITY_M_S: float = 1550.0       # C* 特徵速度 (m/s) — 示意值
THRUST_COEFFICIENT_CF: float = 1.45               # 推力係數 — 示意值
NOZZLE_EFFICIENCY: float = 0.90                   # 噴嘴效率 — 示意值
COMBUSTION_EFFICIENCY: float = 0.92               # 燃燒效率 — 示意值

# ── 示意噴嘴幾何 ─────────────────────────────────────────────────────
DEFAULT_NOZZLE_THROAT_DIAMETER_MM: float = 12.0   # 喉部直徑 mm — 示意值

# ── 風險閾值（教學概念用） ────────────────────────────────────────────
KN_HIGH_THRESHOLD: float = 350.0        # Kn 高值警戒
KN_LOW_THRESHOLD: float = 80.0          # Kn 低值警戒（可能熄火）
PRESSURE_SPIKE_RATIO: float = 1.5       # 壓力尖峰比（峰值/穩態 > 此值即標記）
SENSITIVITY_HIGH_THRESHOLD: float = 0.15  # 敏感度 > 15% 視為高敏感


# ── 藥柱幾何預設參數 ──────────────────────────────────────────────────
@dataclass
class GrainConfig:
    """藥柱幾何設定

    Attributes:
        grain_type: 幾何類型
        outer_diameter_mm: 外徑 (mm)
        length_mm: 單段長度 (mm)
        core_diameter_mm: 中心孔直徑 (mm)
        num_segments: 段數
        inhibited_ends: 每段被抑制的端面數 (0, 1, 或 2)
        star_points: 星形點數（僅 STAR 類型）
        star_inner_ratio: 星形內徑比（僅 STAR 類型）
        nozzle_throat_diameter_mm: 噴嘴喉部直徑 (mm)
        label: 顯示標籤
    """
    grain_type: GrainType = GrainType.BATES
    outer_diameter_mm: float = 50.0
    length_mm: float = 70.0
    core_diameter_mm: float = 18.0
    num_segments: int = 4
    inhibited_ends: int = 0               # 0 = 兩端皆燃燒, 1 = 一端抑制, 2 = 兩端抑制
    star_points: int = 5
    star_inner_ratio: float = 0.4
    nozzle_throat_diameter_mm: float = DEFAULT_NOZZLE_THROAT_DIAMETER_MM
    label: str = ""

    def __post_init__(self) -> None:
        if not self.label:
            self.label = (
                f"{self.grain_type.value} "
                f"OD{self.outer_diameter_mm:.0f} "
                f"L{self.length_mm:.0f} "
                f"CD{self.core_diameter_mm:.0f} "
                f"×{self.num_segments}"
            )


# ── 預設範例組態 ──────────────────────────────────────────────────────
EXAMPLE_CONFIGS: dict[str, GrainConfig] = {
    "BATES 標準": GrainConfig(
        grain_type=GrainType.BATES,
        outer_diameter_mm=50.0,
        length_mm=70.0,
        core_diameter_mm=18.0,
        num_segments=4,
        inhibited_ends=0,
        label="BATES 標準 (接近中性)",
    ),
    "細長藥柱": GrainConfig(
        grain_type=GrainType.CYLINDRICAL,
        outer_diameter_mm=38.0,
        length_mm=150.0,
        core_diameter_mm=12.0,
        num_segments=1,
        inhibited_ends=2,
        label="細長單段 (漸進式)",
    ),
    "短粗藥柱": GrainConfig(
        grain_type=GrainType.CYLINDRICAL,
        outer_diameter_mm=75.0,
        length_mm=40.0,
        core_diameter_mm=25.0,
        num_segments=2,
        inhibited_ends=0,
        label="短粗雙段 (漸退式)",
    ),
    "端面燃燒": GrainConfig(
        grain_type=GrainType.END_BURNER,
        outer_diameter_mm=50.0,
        length_mm=100.0,
        core_diameter_mm=0.0,
        num_segments=1,
        inhibited_ends=0,
        label="端面燃燒 (恆定面積)",
    ),
}


# ── 聲明文字 ──────────────────────────────────────────────────────────
DISCLAIMER_ZH = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                        ⚠ 重要聲明 — 教學與概念用途 ⚠                        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                            ║
║  本工具僅供教學、概念分析與趨勢比較使用。                                     ║
║  不能直接用於真實推進器設計、製造或安全驗證。                                  ║
║                                                                            ║
║  本模型忽略了許多真實世界效應，包括但不限於：                                  ║
║  • 侵蝕燃燒 (erosive burning)                                               ║
║  • 溫度對燃速的影響                                                         ║
║  • 殼體熱傳與結構應力                                                       ║
║  • 噴嘴侵蝕與喉部面積變化                                                   ║
║  • 多相流效應                                                               ║
║  • 點火暫態                                                                ║
║  • 製造公差與材料批次差異                                                    ║
║                                                                            ║
║  若需真實用途，必須依賴專業實驗、材料測試與合格工程審查。                       ║
║                                                                            ║
║  本平台同時提供「工程設計參考表」，其數值來自簡化模型推算，                     ║
║  僅可作為初步方向參考，實際製造前務必經過專業驗證。                             ║
║                                                                            ║
╚══════════════════════════════════════════════════════════════════════════════╝
""".strip()

DISCLAIMER_EN = """
╔══════════════════════════════════════════════════════════════════════════════╗
║               ⚠ DISCLAIMER — Educational & Conceptual Use Only ⚠           ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  This tool is for EDUCATIONAL, CONCEPTUAL ANALYSIS and TREND COMPARISON    ║
║  purposes ONLY.  It must NOT be used for real propulsion system design,     ║
║  manufacturing decisions, or safety certification.                         ║
║                                                                            ║
║  Many real-world effects are intentionally omitted.                        ║
║  For real applications, rely on professional testing & engineering review.  ║
╚══════════════════════════════════════════════════════════════════════════════╝
""".strip()
