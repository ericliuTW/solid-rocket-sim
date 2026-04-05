"""
固態火箭推進概念模擬平台 — 藥柱幾何模組
Grain Geometry Module

提供各種藥柱幾何的燃燒面積計算（隨燃燒回歸量 web 變化）。
所有計算均為簡化概念模型。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from constants import GrainConfig, GrainType, BURN_REGRESSION_STEPS


# ── 幾何計算結果 ──────────────────────────────────────────────────────
@dataclass
class BurnProfile:
    """一組燃燒面積隨回歸量變化的結果

    Attributes:
        web_fraction: 歸一化燃燒回歸量 (0→1)
        burn_area_mm2: 燃燒表面積 (mm²)
        volume_burned_mm3: 累積燃燒體積 (mm³)
        port_area_mm2: 通道截面積 (mm²)
        web_thickness_mm: 最大可用肉厚 (mm)
        config: 原始幾何設定
    """
    web_fraction: NDArray[np.float64]
    burn_area_mm2: NDArray[np.float64]
    volume_burned_mm3: NDArray[np.float64]
    port_area_mm2: NDArray[np.float64]
    web_thickness_mm: float
    config: GrainConfig


class GrainGeometry:
    """藥柱幾何計算器

    根據 GrainConfig 計算燃燒面積隨燃燒回歸量的變化曲線。
    支援 Cylindrical、BATES、End-burner、Star（簡化）等類型。
    """

    def __init__(self, config: GrainConfig) -> None:
        self.config = config
        self._validate()

    def _validate(self) -> None:
        """基本輸入驗證"""
        c = self.config
        if c.outer_diameter_mm <= 0:
            raise ValueError(f"外徑必須 > 0, 得到 {c.outer_diameter_mm}")
        if c.length_mm <= 0:
            raise ValueError(f"長度必須 > 0, 得到 {c.length_mm}")
        if c.grain_type != GrainType.END_BURNER and c.core_diameter_mm <= 0:
            raise ValueError(f"中心孔直徑必須 > 0 (端面燃燒除外), 得到 {c.core_diameter_mm}")
        if c.grain_type != GrainType.END_BURNER and c.core_diameter_mm >= c.outer_diameter_mm:
            raise ValueError("中心孔直徑必須小於外徑")
        if c.num_segments < 1:
            raise ValueError(f"段數必須 >= 1, 得到 {c.num_segments}")
        if c.inhibited_ends not in (0, 1, 2):
            raise ValueError(f"inhibited_ends 必須為 0, 1, 或 2, 得到 {c.inhibited_ends}")
        if c.nozzle_throat_diameter_mm <= 0:
            raise ValueError(f"噴嘴喉部直徑必須 > 0, 得到 {c.nozzle_throat_diameter_mm}")

    def compute_burn_profile(self, n_steps: int = BURN_REGRESSION_STEPS) -> BurnProfile:
        """計算燃燒面積隨回歸量變化

        Args:
            n_steps: 離散步數

        Returns:
            BurnProfile 資料物件
        """
        dispatch = {
            GrainType.CYLINDRICAL: self._cylindrical_profile,
            GrainType.BATES: self._bates_profile,
            GrainType.END_BURNER: self._end_burner_profile,
            GrainType.STAR: self._star_profile,
            GrainType.MOON_BURNER: self._moon_burner_profile,
        }
        fn = dispatch.get(self.config.grain_type)
        if fn is None:
            raise NotImplementedError(f"不支援的幾何類型: {self.config.grain_type}")
        return fn(n_steps)

    # ── Cylindrical / BATES ───────────────────────────────────────────

    def _bates_profile(self, n_steps: int) -> BurnProfile:
        """BATES 多段式藥柱

        每段為中心穿孔圓柱，端面可參與燃燒。
        燃燒面積 = 圓柱內壁面積 + 暴露端面面積（環形）。
        """
        c = self.config
        R_o = c.outer_diameter_mm / 2.0        # 外半徑
        R_i = c.core_diameter_mm / 2.0         # 初始內半徑
        L = c.length_mm                         # 單段長度
        N = c.num_segments
        web = R_o - R_i                          # 肉厚

        # 每段暴露的端面數
        exposed_ends_per_seg = 2 - c.inhibited_ends
        # 段間接觸的端面：相鄰段之間的面都暴露
        # 對 N 段：首尾各有 1 個外端面，段間有 (N-1) 個接合面 × 2 面
        # 總暴露端面數 = N × exposed_ends_per_seg  (簡化：每段獨立計算)
        total_exposed_ends = N * exposed_ends_per_seg

        fracs = np.linspace(0, 1, n_steps)
        areas = np.zeros(n_steps)
        volumes = np.zeros(n_steps)
        port_areas = np.zeros(n_steps)

        for i, f in enumerate(fracs):
            x = f * web                          # 已回歸厚度
            r_inner = R_i + x                    # 當前內半徑
            length_remaining = L - 2 * x * (exposed_ends_per_seg / 2.0)

            if r_inner >= R_o or length_remaining <= 0:
                # 燃燒完畢
                areas[i] = 0.0
                port_areas[i] = math.pi * R_o ** 2
            else:
                # 圓柱內壁
                A_core = 2 * math.pi * r_inner * length_remaining * N
                # 端面（環形）
                A_ends = total_exposed_ends * math.pi * (R_o ** 2 - r_inner ** 2)
                areas[i] = A_core + A_ends
                port_areas[i] = math.pi * r_inner ** 2

            # 累積燃燒體積（簡化積分）
            vol_total = N * math.pi * (R_o ** 2 * L)
            vol_remaining = N * math.pi * ((R_o ** 2 - r_inner ** 2) * length_remaining
                                           if r_inner < R_o and length_remaining > 0 else 0)
            # 修正：remaining grain volume
            if r_inner < R_o and length_remaining > 0:
                vol_grain_remaining = N * math.pi * (R_o ** 2 - r_inner ** 2) * length_remaining
            else:
                vol_grain_remaining = 0.0
            vol_initial_grain = N * math.pi * (R_o ** 2 - R_i ** 2) * L
            volumes[i] = vol_initial_grain - vol_grain_remaining

        return BurnProfile(
            web_fraction=fracs,
            burn_area_mm2=areas,
            volume_burned_mm3=volumes,
            port_area_mm2=port_areas,
            web_thickness_mm=web,
            config=c,
        )

    def _cylindrical_profile(self, n_steps: int) -> BurnProfile:
        """單段圓柱穿孔 — 與 BATES 相同邏輯但預設單段"""
        return self._bates_profile(n_steps)

    def _end_burner_profile(self, n_steps: int) -> BurnProfile:
        """端面燃燒器

        僅一端暴露，燃燒面積在整個過程中保持恆定（理想中性燃燒）。
        """
        c = self.config
        R_o = c.outer_diameter_mm / 2.0
        L = c.length_mm
        web = L  # 肉厚 = 長度

        fracs = np.linspace(0, 1, n_steps)
        A_const = math.pi * R_o ** 2
        areas = np.where(fracs < 1.0, A_const, 0.0)
        volumes = fracs * A_const * L
        port_areas = np.full(n_steps, 0.0)  # 無中心通道

        return BurnProfile(
            web_fraction=fracs,
            burn_area_mm2=areas,
            volume_burned_mm3=volumes,
            port_area_mm2=port_areas,
            web_thickness_mm=web,
            config=c,
        )

    def _star_profile(self, n_steps: int) -> BurnProfile:
        """星形藥柱（簡化模型）

        使用星形周長的解析近似。星形在初期有較大面積（progressive 趨勢），
        隨後當星尖被燒蝕後逐漸趨近圓形。
        """
        c = self.config
        R_o = c.outer_diameter_mm / 2.0
        n_points = c.star_points
        R_inner_base = R_o * c.star_inner_ratio
        web = R_o - R_inner_base

        fracs = np.linspace(0, 1, n_steps)
        areas = np.zeros(n_steps)
        volumes = np.zeros(n_steps)
        port_areas = np.zeros(n_steps)

        for i, f in enumerate(fracs):
            x = f * web
            r_eff = R_inner_base + x

            if r_eff >= R_o:
                areas[i] = 0.0
                port_areas[i] = math.pi * R_o ** 2
            else:
                # 星形周長隨回歸趨近圓形
                # 初期：星形周長 ≈ 2π·r·(1 + 凹凸比)
                # 簡化：star factor 隨回歸量線性衰減
                star_factor = max(0, 1.0 - f * 2.0)  # 前半程保持星形，後半趨近圓
                perimeter_multiplier = 1.0 + star_factor * 0.5 * n_points / math.pi
                perimeter = 2 * math.pi * r_eff * perimeter_multiplier

                A_lateral = perimeter * c.length_mm * c.num_segments
                exposed_ends = c.num_segments * (2 - c.inhibited_ends)
                A_ends = exposed_ends * math.pi * (R_o ** 2 - r_eff ** 2)
                areas[i] = A_lateral + A_ends
                port_areas[i] = math.pi * r_eff ** 2

            vol_initial = c.num_segments * math.pi * (R_o ** 2 - R_inner_base ** 2) * c.length_mm
            if r_eff < R_o:
                vol_remaining = c.num_segments * math.pi * (R_o ** 2 - r_eff ** 2) * c.length_mm
            else:
                vol_remaining = 0.0
            volumes[i] = vol_initial - vol_remaining

        return BurnProfile(
            web_fraction=fracs,
            burn_area_mm2=areas,
            volume_burned_mm3=volumes,
            port_area_mm2=port_areas,
            web_thickness_mm=web,
            config=c,
        )

    def _moon_burner_profile(self, n_steps: int) -> BurnProfile:
        """偏心孔（Moon burner）簡化模型

        以等效偏心圓柱近似。薄壁側先燒穿，造成面積變化不對稱。
        """
        c = self.config
        R_o = c.outer_diameter_mm / 2.0
        R_i = c.core_diameter_mm / 2.0
        # 偏心量：孔心偏向一側
        offset = (R_o - R_i) * 0.3  # 30% 偏心
        web_thin = R_o - R_i - offset   # 薄壁側肉厚
        web_thick = R_o - R_i + offset  # 厚壁側肉厚
        web = web_thin  # 以薄壁側為基準

        fracs = np.linspace(0, 1, n_steps)
        areas = np.zeros(n_steps)
        volumes = np.zeros(n_steps)
        port_areas = np.zeros(n_steps)

        for i, f in enumerate(fracs):
            x = f * web_thick  # 回歸以厚壁為全程
            r_eff = R_i + x

            if r_eff >= R_o:
                areas[i] = 0.0
                port_areas[i] = math.pi * R_o ** 2
            else:
                # 偏心效果：薄壁側先燒穿後面積驟降
                if x < web_thin:
                    # 薄壁尚未燒穿
                    perimeter = 2 * math.pi * r_eff * 1.1  # 偏心增加約 10%
                else:
                    # 薄壁已燒穿，面積下降
                    remaining_frac = 1.0 - (x - web_thin) / (web_thick - web_thin + 1e-9)
                    perimeter = 2 * math.pi * r_eff * max(0.3, remaining_frac)

                A_lateral = perimeter * c.length_mm * c.num_segments
                exposed_ends = c.num_segments * (2 - c.inhibited_ends)
                A_ends = exposed_ends * math.pi * (R_o ** 2 - r_eff ** 2)
                areas[i] = A_lateral + A_ends
                port_areas[i] = math.pi * r_eff ** 2

            vol_initial = c.num_segments * math.pi * (R_o ** 2 - R_i ** 2) * c.length_mm
            if r_eff < R_o:
                vol_remaining = c.num_segments * math.pi * (R_o ** 2 - r_eff ** 2) * c.length_mm
            else:
                vol_remaining = 0.0
            volumes[i] = vol_initial - vol_remaining

        return BurnProfile(
            web_fraction=fracs,
            burn_area_mm2=areas,
            volume_burned_mm3=volumes,
            port_area_mm2=port_areas,
            web_thickness_mm=web_thick,
            config=c,
        )

    # ── 工程參考計算 ──────────────────────────────────────────────────

    @staticmethod
    def compute_propellant_mass(config: GrainConfig) -> float:
        """計算推進劑質量（示意估算）

        Returns:
            質量 (kg) — 示意值
        """
        from constants import PROPELLANT_DENSITY_KG_M3
        R_o = config.outer_diameter_mm / 2.0 / 1000.0  # m
        R_i = config.core_diameter_mm / 2.0 / 1000.0   # m
        L = config.length_mm / 1000.0                    # m

        if config.grain_type == GrainType.END_BURNER:
            vol = math.pi * R_o ** 2 * L * config.num_segments
        else:
            vol = math.pi * (R_o ** 2 - R_i ** 2) * L * config.num_segments

        return vol * PROPELLANT_DENSITY_KG_M3

    @staticmethod
    def compute_nozzle_throat_area_mm2(config: GrainConfig) -> float:
        """噴嘴喉部面積 (mm²)"""
        r = config.nozzle_throat_diameter_mm / 2.0
        return math.pi * r ** 2
