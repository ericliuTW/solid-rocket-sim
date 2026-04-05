"""
固態火箭推進概念模擬平台 — 引擎幾何繪圖模組
Engine Geometry Drawing Module

繪製引擎縱剖面圖、藥柱橫截面圖、燃燒退化序列圖。
用於教學展示與工程設計初步參考。
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import (
    Rectangle, FancyArrowPatch, Circle, Wedge, Arc,
    Polygon, FancyBboxPatch, PathPatch,
)
from matplotlib.path import Path as MplPath
from matplotlib.collections import PatchCollection
from matplotlib.axes import Axes

from constants import GrainConfig, GrainType

# ── 重用 plotting.py 的 CJK 字型與全域樣式 ──────────────────────────
from plotting import _cjk_font

_font_kw = {"fontfamily": _cjk_font} if _cjk_font else {}

# ── 顏色常數 ─────────────────────────────────────────────────────────
COLOR_CASING = "#4B5563"       # 殼體 — 深灰
COLOR_CASING_FILL = "#D1D5DB"  # 殼體填充 — 淺灰
COLOR_PROPELLANT = "#F59E0B"   # 推進劑 — 琥珀色
COLOR_PROPELLANT_DARK = "#B45309"
COLOR_NOZZLE = "#6B7280"       # 噴嘴 — 中灰
COLOR_NOZZLE_FILL = "#9CA3AF"
COLOR_BULKHEAD = "#374151"     # 前蓋
COLOR_CORE = "#FFFFFF"         # 中心孔 — 白
COLOR_FLAME = "#EF4444"        # 火焰
COLOR_EXHAUST = "#FCA5A5"      # 排氣
COLOR_INHIBITOR = "#1E40AF"    # 抑制層 — 深藍
COLOR_DIM_LINE = "#2563EB"     # 標註線 — 藍
COLOR_BURNED = "#FEF3C7"       # 已燃燒區域 — 淡黃
COLOR_SEGMENT_GAP = "#E5E7EB"  # 段間間隙

CASING_THICKNESS_RATIO = 0.08  # 殼體壁厚佔外徑比例（繪圖用）
NOZZLE_LENGTH_RATIO = 0.20     # 噴嘴長度佔總長度比例
BULKHEAD_THICKNESS = 0.04      # 前蓋厚度佔總長度比例
SEGMENT_GAP_RATIO = 0.01       # 段間間隙佔總長度比例


class EngineDrawing:
    """引擎幾何繪圖器

    繪製固態火箭引擎的工程示意圖，包括：
    - 縱剖面圖（側視）
    - 橫截面圖（端面視圖）
    - 燃燒退化序列
    - 帶標註尺寸的工程參考圖
    """

    def __init__(self, config: GrainConfig) -> None:
        self.config = config
        self.R_o = config.outer_diameter_mm / 2.0
        self.R_i = config.core_diameter_mm / 2.0
        self.L = config.length_mm
        self.N = config.num_segments

    # ══════════════════════════════════════════════════════════════════
    #  完整引擎剖面圖（縱剖面 + 橫截面 + 尺寸標註）
    # ══════════════════════════════════════════════════════════════════

    def draw_engine_assembly(self, show: bool = False) -> Figure:
        """繪製完整引擎組裝剖面圖

        上方：縱剖面圖（側視，含尺寸標註）
        下方左：橫截面圖（端面）
        下方右：關鍵尺寸表

        Returns:
            Matplotlib Figure
        """
        fig = plt.figure(figsize=(16, 14))
        fig.suptitle(
            f"引擎幾何剖面圖 — {self.config.label}\n"
            "[教學與設計參考用]",
            fontsize=14, fontweight="bold", **_font_kw,
        )

        # 佈局：上方佔 60%，下方左右各 50%
        ax_longi = fig.add_axes([0.05, 0.42, 0.90, 0.50])   # 縱剖面
        ax_cross = fig.add_axes([0.05, 0.03, 0.40, 0.35])   # 橫截面
        ax_table = fig.add_axes([0.52, 0.03, 0.43, 0.35])   # 尺寸表

        self._draw_longitudinal_section(ax_longi)
        self._draw_cross_section(ax_cross, burn_fraction=0.0)
        self._draw_dimension_table(ax_table)

        if show:
            plt.show()
        return fig

    # ══════════════════════════════════════════════════════════════════
    #  燃燒退化序列圖
    # ══════════════════════════════════════════════════════════════════

    def draw_burn_sequence(self, n_frames: int = 6, show: bool = False) -> Figure:
        """繪製藥柱燃燒退化序列

        顯示從未燃燒到完全燃燒的橫截面變化。

        Args:
            n_frames: 序列幀數
            show: 是否顯示

        Returns:
            Matplotlib Figure
        """
        fracs = np.linspace(0.0, 0.95, n_frames)
        cols = min(n_frames, 4)
        rows = math.ceil(n_frames / cols)

        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows + 0.8))
        fig.suptitle(
            f"藥柱燃燒退化序列 — {self.config.label}\n"
            "[橫截面隨燃燒進程變化]",
            fontsize=13, fontweight="bold", **_font_kw,
        )

        if n_frames == 1:
            axes = np.array([axes])
        axes_flat = axes.flatten()

        for i, frac in enumerate(fracs):
            ax = axes_flat[i]
            self._draw_cross_section(ax, burn_fraction=frac)
            ax.set_title(f"燃燒進程 {frac * 100:.0f}%", fontsize=10, **_font_kw)

        # 隱藏多餘的子圖
        for j in range(n_frames, len(axes_flat)):
            axes_flat[j].set_visible(False)

        fig.tight_layout(rect=[0, 0, 1, 0.93])
        if show:
            plt.show()
        return fig

    # ══════════════════════════════════════════════════════════════════
    #  多類型橫截面比較
    # ══════════════════════════════════════════════════════════════════

    def draw_cross_section_only(self, burn_fraction: float = 0.0,
                                 show: bool = False) -> Figure:
        """繪製單一橫截面"""
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        self._draw_cross_section(ax, burn_fraction=burn_fraction)
        ax.set_title(
            f"藥柱橫截面 — {self.config.label}\n燃燒進程 {burn_fraction*100:.0f}%",
            fontsize=12, **_font_kw,
        )
        fig.tight_layout()
        if show:
            plt.show()
        return fig

    # ══════════════════════════════════════════════════════════════════
    #  內部繪圖方法
    # ══════════════════════════════════════════════════════════════════

    def _draw_longitudinal_section(self, ax: Axes) -> None:
        """繪製縱剖面圖（側視對稱剖面）

        繪製上半部剖面（y >= 0），以中心軸為對稱線。
        包含：殼體、前蓋、藥柱段、段間間隙、噴嘴、中心通道。
        """
        c = self.config
        R_o = self.R_o
        R_i = self.R_i
        L_seg = self.L
        N = self.N

        # 計算整體尺寸
        casing_wall = R_o * CASING_THICKNESS_RATIO * 2
        if casing_wall < 1.5:
            casing_wall = 1.5  # 最小壁厚顯示
        casing_R = R_o + casing_wall

        total_grain_length = L_seg * N
        gap = L_seg * SEGMENT_GAP_RATIO
        total_grain_with_gaps = total_grain_length + gap * (N - 1)

        bulkhead_thick = max(total_grain_length * BULKHEAD_THICKNESS, 3.0)
        nozzle_len = max(total_grain_length * NOZZLE_LENGTH_RATIO, 10.0)

        total_length = bulkhead_thick + total_grain_with_gaps + nozzle_len

        # 噴嘴尺寸
        throat_r = c.nozzle_throat_diameter_mm / 2.0
        exit_r = throat_r * 2.0  # 示意膨脹比
        nozzle_conv_len = nozzle_len * 0.4
        nozzle_div_len = nozzle_len * 0.6

        # ── 座標系：x = 軸向（左→右），y = 徑向 ─────────────

        # 繪製完整剖面（上半 + 下半）
        y_max = casing_R * 1.6
        x_min = -bulkhead_thick * 0.3
        x_max = total_length * 1.15

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(-y_max, y_max)
        ax.set_aspect("equal")
        ax.axhline(y=0, color="#9CA3AF", linewidth=0.5, linestyle="-.")  # 中心軸

        # ── 殼體（上下對稱） ──────────────────────────────────
        for sign in [1, -1]:
            # 外壁
            ax.plot(
                [0, bulkhead_thick + total_grain_with_gaps],
                [sign * casing_R, sign * casing_R],
                color=COLOR_CASING, linewidth=2.5, solid_capstyle="butt",
            )
            # 內壁
            ax.plot(
                [bulkhead_thick, bulkhead_thick + total_grain_with_gaps],
                [sign * R_o, sign * R_o],
                color=COLOR_CASING, linewidth=0.8, linestyle="--", alpha=0.4,
            )

        # ── 前蓋 ──────────────────────────────────────────────
        for sign in [1, -1]:
            bulkhead = Rectangle(
                (0, sign * (-casing_R if sign < 0 else 0)),
                bulkhead_thick,
                casing_R if sign > 0 else casing_R,
                facecolor=COLOR_CASING_FILL, edgecolor=COLOR_BULKHEAD,
                linewidth=1.5, zorder=3,
            )
            ax.add_patch(bulkhead)

        # ── 藥柱段 ───────────────────────────────────────────
        x_cursor = bulkhead_thick
        for seg_idx in range(N):
            for sign in [1, -1]:
                # 推進劑
                propellant = Rectangle(
                    (x_cursor, sign * (R_i if sign > 0 else -R_o)),
                    L_seg,
                    (R_o - R_i) if sign > 0 else (R_o - R_i),
                    facecolor=COLOR_PROPELLANT,
                    edgecolor=COLOR_PROPELLANT_DARK,
                    linewidth=1.0, zorder=2, alpha=0.85,
                )
                ax.add_patch(propellant)

                # 端面燃燒 — 無中心孔
                if c.grain_type == GrainType.END_BURNER:
                    core_block = Rectangle(
                        (x_cursor, sign * (0 if sign > 0 else -R_i)),
                        L_seg, R_i if sign > 0 else R_i,
                        facecolor=COLOR_PROPELLANT,
                        edgecolor=COLOR_PROPELLANT_DARK,
                        linewidth=0.5, zorder=2, alpha=0.85,
                    )
                    ax.add_patch(core_block)

                # 抑制端面標記
                if c.inhibited_ends >= 1:
                    # 左端面抑制
                    ax.plot(
                        [x_cursor, x_cursor],
                        [sign * R_i, sign * R_o],
                        color=COLOR_INHIBITOR, linewidth=2.5, zorder=4,
                    )
                if c.inhibited_ends >= 2:
                    # 右端面抑制
                    ax.plot(
                        [x_cursor + L_seg, x_cursor + L_seg],
                        [sign * R_i, sign * R_o],
                        color=COLOR_INHIBITOR, linewidth=2.5, zorder=4,
                    )

            # 段間間隙
            x_cursor += L_seg
            if seg_idx < N - 1:
                for sign in [1, -1]:
                    gap_rect = Rectangle(
                        (x_cursor, sign * (-casing_R if sign < 0 else -casing_R)),
                        gap, 2 * casing_R,
                        facecolor=COLOR_SEGMENT_GAP, edgecolor="none",
                        alpha=0.3, zorder=1,
                    )
                    ax.add_patch(gap_rect)
                x_cursor += gap

        # ── 中心通道 ──────────────────────────────────────────
        if c.grain_type != GrainType.END_BURNER:
            core_channel = Rectangle(
                (bulkhead_thick, -R_i),
                total_grain_with_gaps, 2 * R_i,
                facecolor=COLOR_CORE, edgecolor="none", zorder=3, alpha=0.9,
            )
            ax.add_patch(core_channel)

        # ── 噴嘴 ─────────────────────────────────────────────
        nozzle_start_x = bulkhead_thick + total_grain_with_gaps

        for sign in [1, -1]:
            # 收斂段：從殼體內徑收斂到喉部
            conv_x = [nozzle_start_x, nozzle_start_x + nozzle_conv_len]
            conv_y_outer = [sign * casing_R, sign * casing_R]
            conv_y_inner = [sign * R_o, sign * throat_r]
            # 發散段：從喉部擴張到出口
            div_x = [nozzle_start_x + nozzle_conv_len,
                      nozzle_start_x + nozzle_conv_len + nozzle_div_len]
            div_y_inner = [sign * throat_r, sign * exit_r]
            div_y_outer = [sign * casing_R, sign * exit_r * 1.3]

            # 噴嘴外壁
            nozzle_outline_x = conv_x + div_x
            nozzle_outline_y = conv_y_outer + div_y_outer
            ax.plot(nozzle_outline_x, nozzle_outline_y,
                    color=COLOR_NOZZLE, linewidth=2.0, zorder=5)

            # 噴嘴內壁（流道）
            nozzle_inner_x = conv_x + div_x
            nozzle_inner_y = conv_y_inner + div_y_inner
            ax.plot(nozzle_inner_x, nozzle_inner_y,
                    color=COLOR_NOZZLE, linewidth=1.5, zorder=5)

            # 填充噴嘴壁
            fill_x = nozzle_outline_x + nozzle_inner_x[::-1]
            fill_y = [a for a in nozzle_outline_y] + [a for a in nozzle_inner_y[::-1]]
            ax.fill(fill_x, fill_y, color=COLOR_NOZZLE_FILL, alpha=0.6, zorder=4)

        # ── 火焰/排氣示意 ─────────────────────────────────────
        exhaust_x = nozzle_start_x + nozzle_len
        for sign in [1, -1]:
            # 小三角形排氣
            tri_x = [exhaust_x, exhaust_x + nozzle_len * 0.5, exhaust_x + nozzle_len * 0.5]
            tri_y = [0, sign * exit_r * 1.8, 0]
            ax.fill(tri_x, tri_y, color=COLOR_EXHAUST, alpha=0.25, zorder=1)

        # 推力方向箭頭
        ax.annotate(
            "", xy=(exhaust_x + nozzle_len * 0.6, 0),
            xytext=(exhaust_x + nozzle_len * 0.1, 0),
            arrowprops=dict(arrowstyle="->", color=COLOR_FLAME, lw=2),
            zorder=6,
        )
        ax.text(
            exhaust_x + nozzle_len * 0.35, -y_max * 0.12,
            "排氣方向", fontsize=8, ha="center", color=COLOR_FLAME, **_font_kw,
        )

        # ── 尺寸標註 ──────────────────────────────────────────
        dim_y = casing_R + y_max * 0.15

        # 總長度
        self._dim_line(ax, 0, nozzle_start_x + nozzle_len, dim_y + y_max * 0.15,
                       f"總長 ~ {bulkhead_thick + total_grain_with_gaps + nozzle_len:.1f} mm")

        # 藥柱長度
        self._dim_line(ax, bulkhead_thick, bulkhead_thick + total_grain_with_gaps, dim_y,
                       f"藥柱總長 {total_grain_with_gaps:.1f} mm")

        # 單段標註（只標第一段）
        if N > 1:
            self._dim_line(ax, bulkhead_thick, bulkhead_thick + L_seg,
                           dim_y - y_max * 0.12,
                           f"單段 {L_seg:.1f} mm")

        # 外徑標註（垂直）
        dim_x = -bulkhead_thick * 0.15
        self._dim_line_v(ax, dim_x, -R_o, R_o,
                         f"OD {c.outer_diameter_mm:.1f}")

        # 內徑標註
        if c.grain_type != GrainType.END_BURNER:
            mid_x = bulkhead_thick + total_grain_with_gaps / 2
            self._dim_line_v(ax, mid_x, -R_i, R_i,
                             f"ID {c.core_diameter_mm:.1f}")

        # 喉部標註
        throat_x = nozzle_start_x + nozzle_conv_len
        self._dim_line_v(ax, throat_x + nozzle_div_len * 0.1, -throat_r, throat_r,
                         f"Dt {c.nozzle_throat_diameter_mm:.1f}")

        # ── 圖例 ─────────────────────────────────────────────
        legend_items = [
            (COLOR_PROPELLANT, "推進劑"),
            (COLOR_CASING_FILL, "殼體"),
            (COLOR_NOZZLE_FILL, "噴嘴"),
        ]
        if c.inhibited_ends > 0:
            legend_items.append((COLOR_INHIBITOR, "抑制層"))

        for i, (color, label) in enumerate(legend_items):
            ax.fill_between([], [], color=color, alpha=0.7, label=label)
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

        ax.set_xlabel("軸向位置 (mm)", **_font_kw)
        ax.set_ylabel("徑向位置 (mm)", **_font_kw)
        ax.set_title("縱剖面圖 (側視)", fontsize=12, fontweight="bold", **_font_kw)
        ax.grid(True, alpha=0.15)

    def _draw_cross_section(self, ax: Axes, burn_fraction: float = 0.0) -> None:
        """繪製藥柱橫截面圖

        Args:
            ax: Matplotlib Axes
            burn_fraction: 燃燒進程 (0.0 ~ 1.0)
        """
        c = self.config
        R_o = self.R_o
        R_i = self.R_i
        web = R_o - R_i if c.grain_type != GrainType.END_BURNER else R_o

        # 當前內半徑（燃燒後）
        regression = burn_fraction * web
        R_current = R_i + regression if c.grain_type != GrainType.END_BURNER else R_o

        # 是否已燒穿
        burned_through = R_current >= R_o

        ax.set_aspect("equal")
        margin = R_o * 1.5
        ax.set_xlim(-margin, margin)
        ax.set_ylim(-margin, margin)

        # ── 殼體 ─────────────────────────────────────────────
        casing_r = R_o * (1 + CASING_THICKNESS_RATIO * 2)
        casing_outer = Circle((0, 0), casing_r, facecolor=COLOR_CASING_FILL,
                               edgecolor=COLOR_CASING, linewidth=2.0, zorder=1)
        ax.add_patch(casing_outer)

        if not burned_through:
            # ── 推進劑 ───────────────────────────────────────
            if c.grain_type in (GrainType.CYLINDRICAL, GrainType.BATES):
                # 圓環形推進劑
                propellant_outer = Circle((0, 0), R_o,
                                          facecolor=COLOR_PROPELLANT,
                                          edgecolor=COLOR_PROPELLANT_DARK,
                                          linewidth=1.0, zorder=2)
                ax.add_patch(propellant_outer)
                # 已燃燒區域（內徑到當前回歸位置）
                if burn_fraction > 0:
                    burned_ring = Circle((0, 0), R_current,
                                         facecolor=COLOR_BURNED,
                                         edgecolor="#D97706",
                                         linewidth=0.5, linestyle="--", zorder=3)
                    ax.add_patch(burned_ring)
                # 中心孔
                core_hole = Circle((0, 0), R_i,
                                   facecolor=COLOR_CORE,
                                   edgecolor="#9CA3AF",
                                   linewidth=0.8, zorder=4)
                ax.add_patch(core_hole)
                # 當前燃燒面（紅色圓）
                burn_circle = Circle((0, 0), R_current,
                                      facecolor="none",
                                      edgecolor=COLOR_FLAME,
                                      linewidth=2.0, linestyle="-",
                                      zorder=5)
                ax.add_patch(burn_circle)

            elif c.grain_type == GrainType.END_BURNER:
                # 實心圓
                propellant = Circle((0, 0), R_o,
                                    facecolor=COLOR_PROPELLANT,
                                    edgecolor=COLOR_PROPELLANT_DARK,
                                    linewidth=1.0, zorder=2)
                ax.add_patch(propellant)
                ax.text(0, 0, "實心\n(端面燃燒)", ha="center", va="center",
                        fontsize=9, color="white", fontweight="bold", zorder=3, **_font_kw)

            elif c.grain_type == GrainType.STAR:
                # 星形截面
                self._draw_star_cross_section(ax, burn_fraction)

            elif c.grain_type == GrainType.MOON_BURNER:
                # 偏心孔截面
                self._draw_moon_cross_section(ax, burn_fraction)

        else:
            # 完全燒穿 — 只剩殼體
            empty = Circle((0, 0), R_o, facecolor=COLOR_BURNED,
                            edgecolor="#D97706", linewidth=1.0,
                            linestyle="--", zorder=2)
            ax.add_patch(empty)
            ax.text(0, 0, "已燃盡", ha="center", va="center",
                    fontsize=11, color="#DC2626", fontweight="bold", zorder=5, **_font_kw)

        # ── 尺寸標線 ─────────────────────────────────────────
        # 外徑
        ax.annotate(
            "", xy=(R_o, -R_o * 1.25), xytext=(-R_o, -R_o * 1.25),
            arrowprops=dict(arrowstyle="<->", color=COLOR_DIM_LINE, lw=1.2),
        )
        ax.text(0, -R_o * 1.35, f"OD {c.outer_diameter_mm:.1f} mm",
                ha="center", fontsize=8, color=COLOR_DIM_LINE, **_font_kw)

        if c.grain_type != GrainType.END_BURNER and not burned_through:
            # 內徑
            ax.annotate(
                "", xy=(R_current, R_o * 1.15), xytext=(-R_current, R_o * 1.15),
                arrowprops=dict(arrowstyle="<->", color=COLOR_FLAME, lw=1.0),
            )
            label = f"ID {R_current * 2:.1f} mm" if burn_fraction > 0 else f"ID {c.core_diameter_mm:.1f} mm"
            ax.text(0, R_o * 1.25, label,
                    ha="center", fontsize=8, color=COLOR_FLAME, **_font_kw)

        ax.set_xlabel("(mm)", fontsize=8)
        ax.grid(True, alpha=0.1)

    def _draw_star_cross_section(self, ax: Axes, burn_fraction: float) -> None:
        """繪製星形藥柱截面"""
        c = self.config
        R_o = self.R_o
        n_pts = c.star_points
        R_inner = R_o * c.star_inner_ratio
        web = R_o - R_inner
        regression = burn_fraction * web

        # 推進劑外圓
        prop_outer = Circle((0, 0), R_o,
                             facecolor=COLOR_PROPELLANT,
                             edgecolor=COLOR_PROPELLANT_DARK,
                             linewidth=1.0, zorder=2)
        ax.add_patch(prop_outer)

        # 星形中心孔
        # 星形頂點半徑隨燃燒增大（趨近圓形）
        r_star_peak = R_inner + regression
        r_star_valley = R_inner * 0.5 + regression
        if r_star_valley > r_star_peak:
            r_star_valley = r_star_peak  # 已趨近圓形

        # 產生星形多邊形
        angles = []
        radii = []
        for i in range(n_pts):
            # 頂點
            a_peak = 2 * math.pi * i / n_pts - math.pi / 2
            angles.append(a_peak)
            radii.append(min(r_star_peak, R_o - 0.5))
            # 谷底
            a_valley = a_peak + math.pi / n_pts
            angles.append(a_valley)
            radii.append(min(r_star_valley, R_o - 0.5))

        star_x = [r * math.cos(a) for r, a in zip(radii, angles)]
        star_y = [r * math.sin(a) for r, a in zip(radii, angles)]
        star_x.append(star_x[0])
        star_y.append(star_y[0])

        star_poly = Polygon(
            list(zip(star_x, star_y)),
            facecolor=COLOR_CORE if burn_fraction == 0 else COLOR_BURNED,
            edgecolor=COLOR_FLAME if burn_fraction > 0 else "#9CA3AF",
            linewidth=1.5, zorder=4,
        )
        ax.add_patch(star_poly)

        if burn_fraction == 0:
            ax.text(0, 0, f"{n_pts}-star", ha="center", va="center",
                    fontsize=8, color="#666", zorder=5)

    def _draw_moon_cross_section(self, ax: Axes, burn_fraction: float) -> None:
        """繪製偏心孔（Moon burner）截面"""
        c = self.config
        R_o = self.R_o
        R_i = self.R_i
        web = R_o - R_i
        offset = web * 0.3  # 偏心量
        regression = burn_fraction * (web + offset)

        # 推進劑外圓
        prop_outer = Circle((0, 0), R_o,
                             facecolor=COLOR_PROPELLANT,
                             edgecolor=COLOR_PROPELLANT_DARK,
                             linewidth=1.0, zorder=2)
        ax.add_patch(prop_outer)

        # 偏心中心孔
        r_current = R_i + regression
        hole_center_x = -offset  # 偏向左側

        core_hole = Circle((hole_center_x, 0), min(r_current, R_o * 0.95),
                            facecolor=COLOR_CORE if burn_fraction == 0 else COLOR_BURNED,
                            edgecolor=COLOR_FLAME if burn_fraction > 0 else "#9CA3AF",
                            linewidth=1.5, zorder=4)
        ax.add_patch(core_hole)

        if burn_fraction == 0:
            # 標記偏心
            ax.annotate(
                "", xy=(hole_center_x, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle="<->", color="#666", lw=0.8),
                zorder=5,
            )
            ax.text(hole_center_x / 2, R_i * 0.4, f"e={offset:.1f}",
                    fontsize=7, ha="center", color="#666", **_font_kw)

    def _draw_dimension_table(self, ax: Axes) -> None:
        """繪製尺寸表格"""
        c = self.config
        ax.axis("off")

        web = self.R_o - self.R_i if c.grain_type != GrainType.END_BURNER else self.L
        total_grain_len = self.L * self.N
        throat_area = math.pi * (c.nozzle_throat_diameter_mm / 2) ** 2

        rows = [
            ("項目", "數值", "單位"),
            ("─" * 18, "─" * 12, "─" * 6),
            ("幾何類型", c.grain_type.value, ""),
            ("外徑 (OD)", f"{c.outer_diameter_mm:.2f}", "mm"),
            ("內徑 (ID)", f"{c.core_diameter_mm:.2f}", "mm"),
            ("單段長度", f"{c.length_mm:.2f}", "mm"),
            ("段數", f"{c.num_segments}", ""),
            ("總藥柱長度", f"{total_grain_len:.2f}", "mm"),
            ("肉厚 (web)", f"{web:.2f}", "mm"),
            ("長徑比 (L/D)", f"{c.length_mm / c.outer_diameter_mm:.2f}", ""),
            ("", "", ""),
            ("喉部直徑 (Dt)", f"{c.nozzle_throat_diameter_mm:.2f}", "mm"),
            ("喉部面積 (At)", f"{throat_area:.2f}", "mm2"),
            ("", "", ""),
            ("抑制端面", f"{c.inhibited_ends}/段", ""),
        ]

        if c.core_diameter_mm > 0:
            rows.insert(9, ("孔徑比 (ID/OD)", f"{c.core_diameter_mm / c.outer_diameter_mm:.3f}", ""))

        text = ""
        for row in rows:
            text += f"  {row[0]:<18s}  {row[1]:>12s}  {row[2]}\n"

        ax.text(
            0.05, 0.95, text,
            transform=ax.transAxes,
            fontsize=9, verticalalignment="top",
            fontfamily=_cjk_font or "monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#F9FAFB", edgecolor="#D1D5DB"),
        )
        ax.set_title("關鍵尺寸表", fontsize=11, fontweight="bold", **_font_kw)

    # ── 標註輔助 ──────────────────────────────────────────────────

    @staticmethod
    def _dim_line(ax: Axes, x1: float, x2: float, y: float, label: str) -> None:
        """水平尺寸標註線"""
        ax.annotate(
            "", xy=(x2, y), xytext=(x1, y),
            arrowprops=dict(arrowstyle="<->", color=COLOR_DIM_LINE, lw=1.0),
        )
        ax.text((x1 + x2) / 2, y + abs(y) * 0.08, label,
                ha="center", fontsize=8, color=COLOR_DIM_LINE, **_font_kw)
        # 引線
        ax.plot([x1, x1], [0, y], color=COLOR_DIM_LINE, linewidth=0.4, alpha=0.4)
        ax.plot([x2, x2], [0, y], color=COLOR_DIM_LINE, linewidth=0.4, alpha=0.4)

    @staticmethod
    def _dim_line_v(ax: Axes, x: float, y1: float, y2: float, label: str) -> None:
        """垂直尺寸標註線"""
        ax.annotate(
            "", xy=(x, y2), xytext=(x, y1),
            arrowprops=dict(arrowstyle="<->", color=COLOR_DIM_LINE, lw=1.0),
        )
        ax.text(x - abs(x) * 0.3 - 2, (y1 + y2) / 2, label,
                ha="right", va="center", fontsize=8, color=COLOR_DIM_LINE,
                rotation=90, **_font_kw)
