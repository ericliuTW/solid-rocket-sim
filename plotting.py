"""
固態火箭推進概念模擬平台 — 視覺化模組
Plotting / Visualization Module

使用 Matplotlib 繪製高品質圖表。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import matplotlib
matplotlib.use("Agg")  # 非互動後端，兼容無 GUI 環境
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.ticker as ticker

from simulation import SimulationResult
from sensitivity import SensitivityReport
from risk_warnings import RiskFlag, RiskLevel

# ── CJK 字型設定 ─────────────────────────────────────────────────────
import platform
import glob
from matplotlib import font_manager as fm

_system = platform.system()
_cjk_font = None
_cjk_font_path = None


def _find_cjk_font() -> tuple[str | None, str | None]:
    """跨平台尋找 CJK 字型，回傳 (字型名稱, 字型檔路徑)"""

    # ── 方法 1：直接掃描常見字型檔路徑（最可靠） ───────────────
    search_paths: list[str] = []
    if _system == "Windows":
        search_paths = [
            "C:/Windows/Fonts/msjh*.ttc",      # 微軟正黑體
            "C:/Windows/Fonts/msyh*.ttc",      # 微軟雅黑
            "C:/Windows/Fonts/simhei.ttf",     # 黑體
        ]
    elif _system == "Darwin":
        search_paths = [
            "/System/Library/Fonts/PingFang.ttc",
            "/Library/Fonts/Arial Unicode*",
        ]
    else:
        # Linux — Streamlit Cloud / Docker / 一般 Linux
        search_paths = [
            "/usr/share/fonts/opentype/noto/NotoSansCJK*.ttc",
            "/usr/share/fonts/noto-cjk/NotoSansCJK*.ttc",
            "/usr/share/fonts/truetype/noto/NotoSansCJK*.ttf",
            "/usr/share/fonts/opentype/noto/*CJK*.otf",
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
            "/usr/share/fonts/**/NotoSansCJK*",
            "/usr/share/fonts/**/*CJK*",
        ]

    for pattern in search_paths:
        matches = glob.glob(pattern, recursive=True)
        if matches:
            font_path = matches[0]
            # 註冊到 matplotlib
            fm.fontManager.addfont(font_path)
            prop = fm.FontProperties(fname=font_path)
            font_name = prop.get_name()
            return font_name, font_path

    # ── 方法 2：透過 matplotlib 字型管理器查詢 ────────────────
    candidate_names = {
        "Windows": ["Microsoft JhengHei", "Microsoft YaHei", "SimHei"],
        "Darwin": ["PingFang TC", "PingFang SC", "Heiti TC"],
        "Linux": ["Noto Sans CJK TC", "Noto Sans CJK SC",
                   "Noto Sans CJK JP", "WenQuanYi Micro Hei"],
    }.get(_system, ["Noto Sans CJK SC"])

    default_font = fm.findfont(fm.FontProperties())
    for name in candidate_names:
        found = fm.findfont(fm.FontProperties(family=name))
        if found != default_font:
            return name, found

    return None, None


_cjk_font, _cjk_font_path = _find_cjk_font()

# ── 全域樣式 ──────────────────────────────────────────────────────────
_rc: dict = {
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "figure.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.unicode_minus": False,
}
if _cjk_font:
    _rc["font.family"] = "sans-serif"
    _rc["font.sans-serif"] = [_cjk_font]
plt.rcParams.update(_rc)

# 色盤
COLORS = [
    "#2563EB",  # 藍
    "#DC2626",  # 紅
    "#16A34A",  # 綠
    "#D97706",  # 橙
    "#7C3AED",  # 紫
    "#0891B2",  # 青
    "#BE185D",  # 粉
    "#4B5563",  # 灰
]

RISK_ZONE_COLOR = "#FCA5A5"
RISK_ZONE_ALPHA = 0.15


class PlotManager:
    """繪圖管理器

    提供各類模擬結果的視覺化繪圖功能。
    """

    def __init__(self, save_dir: str | Path | None = None) -> None:
        self.save_dir = Path(save_dir) if save_dir else None
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)

    # ── 單組結果完整面板 ──────────────────────────────────────────

    def plot_single_result(
        self,
        result: SimulationResult,
        mode: str = "normalized",
        risk_flags: list[RiskFlag] | None = None,
        show: bool = False,
    ) -> Figure:
        """繪製單組模擬結果的完整面板

        Args:
            result: 模擬結果
            mode: "normalized" 或 "estimated" (示意估算值)
            risk_flags: 風險標記（用於標注高風險區段）
            show: 是否顯示

        Returns:
            Matplotlib Figure
        """
        fig, axes = plt.subplots(3, 2, figsize=(14, 12))
        fig.suptitle(
            f"固態火箭概念模擬 — {result.config.label}\n"
            f"({'歸一化相對值' if mode == 'normalized' else '[!] 示意估算值 — 僅供參考'})",
            fontsize=14, fontweight="bold",
        )

        if mode == "normalized":
            x = result.time_normalized
            x_label = "歸一化燃燒進程"
            data_sets = [
                (axes[0, 0], result.burn_area_normalized, "相對燃燒面積", "Ab / Ab_0"),
                (axes[0, 1], result.kn_normalized, "相對 Kn 值", "Kn / Kn_0"),
                (axes[1, 0], result.pressure_normalized, "相對燃燒室壓力", "P / P_0"),
                (axes[1, 1], result.thrust_normalized, "相對推力", "F / F_0"),
                (axes[2, 0], result.impulse_normalized, "歸一化累積衝量", "I / I_total"),
            ]
        else:
            x = result.time_s
            x_label = "時間 (秒) [[!] 示意估算]"
            data_sets = [
                (axes[0, 0], result.burn_area_mm2, "燃燒面積 [mm²]", "Ab (mm²)"),
                (axes[0, 1], result.kn, "Kn 值 (Ab/At)", "Kn"),
                (axes[1, 0], result.pressure_mpa, "燃燒室壓力 [[!] 示意]", "P (MPa) [示意]"),
                (axes[1, 1], result.thrust_n, "推力 [[!] 示意]", "F (N) [示意]"),
                (axes[2, 0], result.impulse_ns, "累積衝量 [[!] 示意]", "I (N·s) [示意]"),
            ]

        for ax, y, title, ylabel in data_sets:
            ax.plot(x, y, color=COLORS[0], linewidth=1.8)
            ax.set_title(title)
            ax.set_xlabel(x_label)
            ax.set_ylabel(ylabel)
            ax.set_xlim(x[0], x[-1])

        # 右下角：文字摘要
        ax_text = axes[2, 1]
        ax_text.axis("off")
        summary_lines = self._build_summary_text(result, mode)
        ax_text.text(
            0.05, 0.95, "\n".join(summary_lines),
            transform=ax_text.transAxes,
            fontsize=9, verticalalignment="top", fontproperties=fm.FontProperties(fname=_cjk_font_path) if _cjk_font_path else None,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#F3F4F6", alpha=0.8),
        )

        # 高風險區段陰影
        if risk_flags:
            self._add_risk_zones(axes.flatten()[:5], result, risk_flags)

        fig.tight_layout(rect=[0, 0, 1, 0.94])

        if self.save_dir:
            fig.savefig(self.save_dir / f"single_{result.config.label[:20]}.png", bbox_inches="tight")
        if show:
            plt.show()
        return fig

    # ── 多組比較 ──────────────────────────────────────────────────

    def plot_comparison(
        self,
        results: Sequence[SimulationResult],
        mode: str = "normalized",
        show: bool = False,
    ) -> Figure:
        """多組幾何結果疊圖比較

        Args:
            results: 多組模擬結果
            mode: "normalized" 或 "estimated"
            show: 是否顯示

        Returns:
            Matplotlib Figure
        """
        fig, axes = plt.subplots(3, 2, figsize=(14, 12))
        fig.suptitle(
            f"固態火箭概念模擬 — 多組幾何趨勢比較\n"
            f"({'歸一化相對值' if mode == 'normalized' else '[!] 示意估算值 — 僅供參考'})",
            fontsize=14, fontweight="bold",
        )

        for idx, result in enumerate(results):
            color = COLORS[idx % len(COLORS)]
            label = result.config.label

            if mode == "normalized":
                x = result.time_normalized
                ys = [
                    result.burn_area_normalized,
                    result.kn_normalized,
                    result.pressure_normalized,
                    result.thrust_normalized,
                    result.impulse_normalized,
                ]
            else:
                x = result.time_s
                ys = [
                    result.burn_area_mm2,
                    result.kn,
                    result.pressure_mpa,
                    result.thrust_n,
                    result.impulse_ns,
                ]

            titles = ["燃燒面積", "Kn 值", "燃燒室壓力", "推力", "累積衝量"]
            for i, (ax, y, title) in enumerate(zip(axes.flatten()[:5], ys, titles)):
                ax.plot(x, y, color=color, linewidth=1.5, label=label, alpha=0.85)
                ax.set_title(title)

        # 圖例
        for ax in axes.flatten()[:5]:
            ax.legend(loc="best", fontsize=7)
            ax.set_xlabel("歸一化燃燒進程" if mode == "normalized" else "時間 (秒) [[!] 示意]")

        # 右下角：比較摘要
        ax_text = axes[2, 1]
        ax_text.axis("off")
        comparison_text = self._build_comparison_text(results)
        ax_text.text(
            0.05, 0.95, "\n".join(comparison_text),
            transform=ax_text.transAxes,
            fontsize=8, verticalalignment="top", fontproperties=fm.FontProperties(fname=_cjk_font_path) if _cjk_font_path else None,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#F3F4F6", alpha=0.8),
        )

        fig.tight_layout(rect=[0, 0, 1, 0.94])

        if self.save_dir:
            fig.savefig(self.save_dir / "comparison.png", bbox_inches="tight")
        if show:
            plt.show()
        return fig

    # ── 敏感度分析圖 ──────────────────────────────────────────────

    def plot_sensitivity(
        self,
        report: SensitivityReport,
        show: bool = False,
    ) -> Figure:
        """繪製敏感度分析對照圖

        Args:
            report: 敏感度分析報告
            show: 是否顯示

        Returns:
            Matplotlib Figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            f"敏感度分析 — {report.config.label}\n"
            "[!] 概念級分析，僅供趨勢比較",
            fontsize=14, fontweight="bold",
        )

        # 推力曲線對照
        ax_thrust = axes[0, 0]
        ax_thrust.set_title("推力曲線對照 (歸一化)")
        x_base = report.baseline.time_normalized
        ax_thrust.plot(x_base, report.baseline.thrust_normalized,
                       color="black", linewidth=2, label="基準", zorder=10)

        for idx, p in enumerate(report.perturbations):
            color = COLORS[(idx + 1) % len(COLORS)]
            style = "--" if p.perturbation_pct < 0 else "-."
            ax_thrust.plot(
                p.perturbed_result.time_normalized,
                p.perturbed_result.thrust_normalized,
                color=color, linewidth=1.2, linestyle=style,
                label=p.perturbed_result.config.label, alpha=0.7,
            )
        ax_thrust.legend(fontsize=7, loc="best")
        ax_thrust.set_xlabel("歸一化燃燒進程")
        ax_thrust.set_ylabel("相對推力 F/F_0")

        # 壓力曲線對照
        ax_press = axes[0, 1]
        ax_press.set_title("壓力曲線對照 (歸一化)")
        ax_press.plot(x_base, report.baseline.pressure_normalized,
                      color="black", linewidth=2, label="基準", zorder=10)
        for idx, p in enumerate(report.perturbations):
            color = COLORS[(idx + 1) % len(COLORS)]
            style = "--" if p.perturbation_pct < 0 else "-."
            ax_press.plot(
                p.perturbed_result.time_normalized,
                p.perturbed_result.pressure_normalized,
                color=color, linewidth=1.2, linestyle=style,
                label=p.perturbed_result.config.label, alpha=0.7,
            )
        ax_press.legend(fontsize=7, loc="best")
        ax_press.set_xlabel("歸一化燃燒進程")
        ax_press.set_ylabel("相對壓力 P/P_0")

        # 長條圖：峰值推力變化
        ax_bar_thrust = axes[1, 0]
        ax_bar_thrust.set_title("峰值推力變化 (%)")
        labels = [f"{p.parameter_name}\n{p.perturbation_pct:+.0f}%" for p in report.perturbations]
        values = [p.thrust_change_pct for p in report.perturbations]
        bar_colors = ["#DC2626" if abs(v) > 15 else "#2563EB" for v in values]
        bars = ax_bar_thrust.barh(range(len(labels)), values, color=bar_colors, alpha=0.8)
        ax_bar_thrust.set_yticks(range(len(labels)))
        ax_bar_thrust.set_yticklabels(labels, fontsize=8)
        ax_bar_thrust.set_xlabel("峰值推力變化 (%)")
        ax_bar_thrust.axvline(x=0, color="black", linewidth=0.5)
        # 標注高敏感度閾值
        ax_bar_thrust.axvline(x=15, color="red", linewidth=0.8, linestyle="--", alpha=0.5)
        ax_bar_thrust.axvline(x=-15, color="red", linewidth=0.8, linestyle="--", alpha=0.5)

        # 長條圖：峰值壓力變化
        ax_bar_press = axes[1, 1]
        ax_bar_press.set_title("峰值壓力變化 (%)")
        p_values = [p.pressure_change_pct for p in report.perturbations]
        p_colors = ["#DC2626" if abs(v) > 15 else "#16A34A" for v in p_values]
        ax_bar_press.barh(range(len(labels)), p_values, color=p_colors, alpha=0.8)
        ax_bar_press.set_yticks(range(len(labels)))
        ax_bar_press.set_yticklabels(labels, fontsize=8)
        ax_bar_press.set_xlabel("峰值壓力變化 (%)")
        ax_bar_press.axvline(x=0, color="black", linewidth=0.5)
        ax_bar_press.axvline(x=15, color="red", linewidth=0.8, linestyle="--", alpha=0.5)
        ax_bar_press.axvline(x=-15, color="red", linewidth=0.8, linestyle="--", alpha=0.5)

        fig.tight_layout(rect=[0, 0, 1, 0.93])

        if self.save_dir:
            fig.savefig(self.save_dir / "sensitivity.png", bbox_inches="tight")
        if show:
            plt.show()
        return fig

    # ── 輔助方法 ──────────────────────────────────────────────────

    def _build_summary_text(self, result: SimulationResult, mode: str) -> list[str]:
        """建立文字摘要"""
        lines = [
            "━━━ 模擬摘要 ━━━",
            f"幾何: {result.config.label}",
            f"段數: {result.config.num_segments}",
            f"推進劑質量: {result.propellant_mass_kg * 1000:.1f} g [示意]",
            "",
        ]
        if mode == "estimated":
            lines.extend([
                "[!] 以下為示意估算值 [!]",
                f"估算燃燒時間: {result.burn_time_s:.2f} s",
                f"估算峰值推力: {result.peak_thrust_n:.1f} N",
                f"估算平均推力: {result.average_thrust_n:.1f} N",
                f"估算總衝量: {result.total_impulse_ns:.1f} N·s",
                f"估算比衝: {result.specific_impulse_s:.0f} s",
                f"估算峰值壓力: {result.peak_pressure_mpa:.2f} MPa",
            ])
        else:
            # 趨勢特徵
            area_trend = self._classify_trend(result.burn_area_normalized)
            lines.extend([
                f"燃燒面積趨勢: {area_trend}",
                f"Kn 峰值/初始比: {np.max(result.kn_normalized):.2f}×",
                f"推力變異係數: {self._compute_cv(result.thrust_normalized):.3f}",
            ])

        lines.extend([
            "",
            "[!] 概念模型，僅供教學參考",
        ])
        return lines

    def _build_comparison_text(self, results: Sequence[SimulationResult]) -> list[str]:
        """建立比較摘要"""
        lines = ["━━━ 比較摘要 ━━━", ""]
        for r in results:
            trend = self._classify_trend(r.burn_area_normalized)
            cv = self._compute_cv(r.thrust_normalized)
            lines.append(f"• {r.config.label}")
            lines.append(f"  趨勢: {trend}, 推力 CV={cv:.3f}")
            lines.append("")
        lines.append("[!] 概念模型，僅供教學參考")
        return lines

    @staticmethod
    def _classify_trend(area_norm: np.ndarray) -> str:
        """分類燃燒趨勢"""
        valid = area_norm[area_norm > 0]
        if len(valid) < 10:
            return "資料不足"
        third = len(valid) // 3
        early = np.mean(valid[:third])
        late = np.mean(valid[2 * third:])
        if early > 0:
            ratio = late / early
            if ratio > 1.15:
                return "漸進式 (progressive)"
            elif ratio < 0.85:
                return "漸退式 (regressive)"
            else:
                return "接近中性 (neutral)"
        return "未定"

    @staticmethod
    def _compute_cv(arr: np.ndarray) -> float:
        """計算變異係數"""
        valid = arr[arr > 0]
        if len(valid) < 2:
            return 0.0
        return float(np.std(valid) / np.mean(valid))

    def _add_risk_zones(
        self,
        axes: Sequence[Axes],
        result: SimulationResult,
        flags: list[RiskFlag],
    ) -> None:
        """在圖表上標注高風險區段陰影"""
        for flag in flags:
            if flag.level in (RiskLevel.WARNING, RiskLevel.CRITICAL) and flag.affected_region:
                # 解析區段百分比
                import re
                match = re.search(r"(\d+)%.*?(\d+)%", flag.affected_region)
                if match:
                    start = float(match.group(1)) / 100.0
                    end = float(match.group(2)) / 100.0
                    for ax in axes:
                        ax.axvspan(start, end, color=RISK_ZONE_COLOR, alpha=RISK_ZONE_ALPHA)
