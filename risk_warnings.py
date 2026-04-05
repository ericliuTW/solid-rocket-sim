"""
固態火箭推進概念模擬平台 — 風險提示模組
Risk Warning / Flagging Module

教學用風險標記系統，提供概念級風險提醒。

⚠ 此系統僅做「概念風險提醒」，不夠格作為真實安全判定工具。
  實際安全評估需要專業工程審查與實驗驗證。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from constants import (
    KN_HIGH_THRESHOLD,
    KN_LOW_THRESHOLD,
    PRESSURE_SPIKE_RATIO,
    SENSITIVITY_HIGH_THRESHOLD,
)
from simulation import SimulationResult
from sensitivity import SensitivityReport


class RiskLevel(Enum):
    """風險等級"""
    INFO = "ℹ️ 資訊"
    CAUTION = "⚠️ 注意"
    WARNING = "🔶 警告"
    CRITICAL = "🔴 嚴重警告"


@dataclass
class RiskFlag:
    """單一風險標記"""
    level: RiskLevel
    category: str
    title: str
    description: str
    recommendation: str
    affected_region: str = ""  # 例如 "燃燒進程 0-20%"

    def format_text(self) -> str:
        """格式化為文字輸出"""
        lines = [
            f"  {self.level.value} [{self.category}] {self.title}",
            f"    說明: {self.description}",
        ]
        if self.affected_region:
            lines.append(f"    影響區段: {self.affected_region}")
        lines.append(f"    建議: {self.recommendation}")
        return "\n".join(lines)


class RiskFlagger:
    """教學用風險標記器

    分析模擬結果與敏感度報告，產生概念級風險提醒。

    ⚠ 此標記器僅供教學用途，不可替代真實安全評估。
    """

    def analyze(
        self,
        sim_result: SimulationResult,
        sensitivity_report: SensitivityReport | None = None,
    ) -> list[RiskFlag]:
        """分析並產生風險標記

        Args:
            sim_result: 模擬結果
            sensitivity_report: 敏感度報告（可選）

        Returns:
            風險標記列表
        """
        flags: list[RiskFlag] = []

        flags.extend(self._check_kn_range(sim_result))
        flags.extend(self._check_pressure_spike(sim_result))
        flags.extend(self._check_thrust_stability(sim_result))
        flags.extend(self._check_progressive_tendency(sim_result))
        flags.extend(self._check_model_confidence(sim_result))

        if sensitivity_report:
            flags.extend(self._check_sensitivity(sensitivity_report))

        # 永遠加上模型限制提醒
        flags.append(RiskFlag(
            level=RiskLevel.INFO,
            category="模型限制",
            title="概念模型固有限制",
            description=(
                "本模型忽略侵蝕燃燒、溫度效應、點火暫態、噴嘴侵蝕、"
                "多相流、殼體熱傳、製造公差等真實世界效應。"
            ),
            recommendation="所有結果僅供趨勢比較與教學，不可直接用於真實設計。",
        ))

        return flags

    def _check_kn_range(self, result: SimulationResult) -> list[RiskFlag]:
        """檢查 Kn 值範圍"""
        flags = []
        kn = result.kn
        valid_kn = kn[kn > 0]
        if len(valid_kn) == 0:
            return flags

        max_kn = np.max(valid_kn)
        min_kn = np.min(valid_kn)

        if max_kn > KN_HIGH_THRESHOLD:
            # 找到高 Kn 區段
            high_idx = np.where(kn > KN_HIGH_THRESHOLD)[0]
            start_pct = high_idx[0] / len(kn) * 100
            end_pct = high_idx[-1] / len(kn) * 100
            flags.append(RiskFlag(
                level=RiskLevel.WARNING,
                category="Kn 值",
                title=f"Kn 值偏高 (峰值 {max_kn:.0f})",
                description=(
                    f"Kn 值超過 {KN_HIGH_THRESHOLD:.0f} 的教學警戒線。"
                    "高 Kn 值意味著燃燒面積相對喉部面積過大，"
                    "在真實系統中可能導致壓力過高。"
                ),
                affected_region=f"燃燒進程 {start_pct:.0f}%–{end_pct:.0f}%",
                recommendation="考慮增大噴嘴喉部直徑或減少燃燒面積。",
            ))

        if min_kn < KN_LOW_THRESHOLD:
            flags.append(RiskFlag(
                level=RiskLevel.CAUTION,
                category="Kn 值",
                title=f"Kn 值偏低 (最低 {min_kn:.0f})",
                description=(
                    f"Kn 值低於 {KN_LOW_THRESHOLD:.0f}，"
                    "在真實系統中可能導致壓力不足以維持穩定燃燒。"
                ),
                recommendation="考慮減小噴嘴喉部直徑或增加燃燒面積。",
            ))

        # Kn 變化幅度
        kn_range_ratio = max_kn / min_kn if min_kn > 0 else float('inf')
        if kn_range_ratio > 2.0:
            flags.append(RiskFlag(
                level=RiskLevel.CAUTION,
                category="Kn 值",
                title=f"Kn 變化幅度大 (比值 {kn_range_ratio:.1f}×)",
                description="Kn 值在燃燒過程中變化劇烈，推力曲線將不平穩。",
                recommendation="選擇更接近中性燃燒的幾何設計以減少 Kn 變化。",
            ))

        return flags

    def _check_pressure_spike(self, result: SimulationResult) -> list[RiskFlag]:
        """檢查壓力尖峰"""
        flags = []
        pressure = result.pressure_pa
        valid_p = pressure[pressure > 0]
        if len(valid_p) < 10:
            return flags

        peak_p = np.max(valid_p)
        # 取中段作為穩態參考
        mid_start = len(valid_p) // 4
        mid_end = 3 * len(valid_p) // 4
        steady_p = np.mean(valid_p[mid_start:mid_end])

        if steady_p > 0 and peak_p / steady_p > PRESSURE_SPIKE_RATIO:
            peak_idx = np.argmax(pressure)
            peak_pct = peak_idx / len(pressure) * 100
            ratio = peak_p / steady_p
            flags.append(RiskFlag(
                level=RiskLevel.WARNING,
                category="壓力",
                title=f"壓力尖峰 (峰值/穩態 = {ratio:.2f}×)",
                description=(
                    "燃燒室壓力出現明顯尖峰，峰值顯著超過穩態值。"
                    "在真實系統中，壓力尖峰可能導致結構安全裕度不足。"
                ),
                affected_region=f"燃燒進程約 {peak_pct:.0f}%",
                recommendation="調整幾何使燃燒面積變化更平緩，或加強殼體設計安全係數。",
            ))

        # 檢查初期壓力快速上升
        early_segment = valid_p[:len(valid_p) // 5]
        if len(early_segment) > 2:
            rise_rate = (np.max(early_segment) - early_segment[0]) / early_segment[0] if early_segment[0] > 0 else 0
            if rise_rate > 0.5:
                flags.append(RiskFlag(
                    level=RiskLevel.CAUTION,
                    category="壓力",
                    title="初期壓力快速上升",
                    description="燃燒前 20% 階段壓力上升較快，可能反映初期面積增長偏大。",
                    affected_region="燃燒進程 0%–20%",
                    recommendation="考慮使用漸進點火或調整初始幾何以緩和啟動暫態。",
                ))

        return flags

    def _check_thrust_stability(self, result: SimulationResult) -> list[RiskFlag]:
        """檢查推力曲線平穩性"""
        flags = []
        thrust = result.thrust_n
        valid_t = thrust[thrust > 0]
        if len(valid_t) < 10:
            return flags

        avg = np.mean(valid_t)
        std = np.std(valid_t)
        cv = std / avg if avg > 0 else 0  # 變異係數

        if cv > 0.3:
            flags.append(RiskFlag(
                level=RiskLevel.WARNING,
                category="推力",
                title=f"推力曲線不穩定 (CV = {cv:.2f})",
                description="推力變異係數偏高，曲線波動明顯，不接近中性燃燒。",
                recommendation="選擇 BATES 或端面燃燒等更中性的幾何以改善平穩性。",
            ))
        elif cv > 0.15:
            flags.append(RiskFlag(
                level=RiskLevel.CAUTION,
                category="推力",
                title=f"推力曲線有中等波動 (CV = {cv:.2f})",
                description="推力曲線有一定波動，偏離理想中性燃燒。",
                recommendation="微調幾何比例可能改善曲線形狀。",
            ))

        return flags

    def _check_progressive_tendency(self, result: SimulationResult) -> list[RiskFlag]:
        """檢查漸進/漸退趨勢"""
        flags = []
        area_norm = result.burn_area_normalized
        valid = area_norm[area_norm > 0]
        if len(valid) < 10:
            return flags

        # 比較前 1/3 與後 1/3 的平均面積
        third = len(valid) // 3
        early_avg = np.mean(valid[:third])
        late_avg = np.mean(valid[2 * third:])

        if early_avg > 0:
            ratio = late_avg / early_avg
            if ratio > 1.3:
                flags.append(RiskFlag(
                    level=RiskLevel.CAUTION,
                    category="燃燒趨勢",
                    title=f"漸進式燃燒趨勢 (後/前比 = {ratio:.2f}×)",
                    description=(
                        "燃燒面積在後期增加，屬於 progressive 特性。"
                        "壓力與推力會在後段上升，需注意殼體能否承受後期峰值。"
                    ),
                    recommendation="確認結構在最大壓力下仍有足夠安全裕度。",
                ))
            elif ratio < 0.7:
                flags.append(RiskFlag(
                    level=RiskLevel.INFO,
                    category="燃燒趨勢",
                    title=f"漸退式燃燒趨勢 (後/前比 = {ratio:.2f}×)",
                    description="燃燒面積在後期減少，屬於 regressive 特性。推力隨時間下降。",
                    recommendation="此特性在某些應用中是期望的，無需額外動作。",
                ))

        return flags

    def _check_model_confidence(self, result: SimulationResult) -> list[RiskFlag]:
        """檢查模型可信度降低的區域"""
        flags = []
        kn = result.kn
        valid_kn = kn[kn > 0]

        if len(valid_kn) > 0 and np.max(valid_kn) > 250:
            flags.append(RiskFlag(
                level=RiskLevel.CAUTION,
                category="模型可信度",
                title="高 Kn 區域模型可信度下降",
                description=(
                    "在高 Kn 值區域，真實系統的侵蝕燃燒效應會變得顯著，"
                    "但本概念模型未納入此效應。實際燃速可能比預測值更高。"
                ),
                recommendation="在高 Kn 區域解讀結果時需特別謹慎。",
            ))

        # 端面燃燒特殊提醒
        from constants import GrainType
        if result.config.grain_type == GrainType.END_BURNER:
            flags.append(RiskFlag(
                level=RiskLevel.INFO,
                category="模型可信度",
                title="端面燃燒器特殊注意",
                description=(
                    "端面燃燒器在真實系統中可能出現非均勻燃燒、"
                    "熱浸透（heat soak-back）等問題，概念模型假設理想均勻燃燒。"
                ),
                recommendation="端面燃燒器的實際性能可能偏離理想模型較多。",
            ))

        return flags

    def _check_sensitivity(self, report: SensitivityReport) -> list[RiskFlag]:
        """根據敏感度分析結果產生風險標記"""
        flags = []

        for param in report.high_sensitivity_params:
            param_label = {
                "core_diameter_mm": "中心孔直徑",
                "length_mm": "藥柱長度",
                "outer_diameter_mm": "藥柱外徑",
            }.get(param, param)

            max_change = max(
                abs(p.thrust_change_pct) for p in report.perturbations
                if p.parameter_name == param
            )

            flags.append(RiskFlag(
                level=RiskLevel.WARNING,
                category="製造敏感度",
                title=f"{param_label} 對推力高度敏感 (±5% → {max_change:.1f}% 變化)",
                description=(
                    f"{param_label}的微小變動即可導致推力或壓力峰值產生較大變化。"
                    "在實際製造中，此參數的公差控制特別重要。"
                ),
                recommendation=f"建議將{param_label}的製造公差控制在 ±2% 以內，並進行實物測試驗證。",
            ))

        if report.max_pressure_sensitivity_pct > 30:
            flags.append(RiskFlag(
                level=RiskLevel.CRITICAL,
                category="製造敏感度",
                title=f"壓力對尺寸變動極度敏感 (最大 {report.max_pressure_sensitivity_pct:.1f}%)",
                description="壓力峰值對幾何尺寸變化極為敏感，製造誤差可能導致壓力大幅超出預期。",
                recommendation="此設計在真實製造中風險較高，建議增大安全裕度或選擇更穩健的幾何。",
            ))

        return flags

    def format_report(self, flags: list[RiskFlag]) -> str:
        """格式化風險報告為文字"""
        if not flags:
            return "  未發現需要提示的風險項目。"

        lines = []
        # 按嚴重程度排序
        priority = {
            RiskLevel.CRITICAL: 0,
            RiskLevel.WARNING: 1,
            RiskLevel.CAUTION: 2,
            RiskLevel.INFO: 3,
        }
        sorted_flags = sorted(flags, key=lambda f: priority.get(f.level, 99))

        for flag in sorted_flags:
            lines.append(flag.format_text())
            lines.append("")

        return "\n".join(lines)
