"""
固態火箭推進概念模擬平台 — 報告產生模組
Report Generation Module

產生文字摘要報告與工程設計參考表。
"""

from __future__ import annotations

import math
from datetime import datetime

import numpy as np

from constants import (
    GrainConfig,
    GrainType,
    DISCLAIMER_ZH,
    PROPELLANT_DENSITY_KG_M3,
    BURN_RATE_COEFFICIENT_A,
    BURN_RATE_EXPONENT_N,
    CHARACTERISTIC_VELOCITY_M_S,
    THRUST_COEFFICIENT_CF,
)
from simulation import SimulationResult
from sensitivity import SensitivityReport
from risk_warnings import RiskFlag, RiskFlagger


def generate_text_report(
    result: SimulationResult,
    sensitivity_report: SensitivityReport | None = None,
    risk_flags: list[RiskFlag] | None = None,
) -> str:
    """產生完整文字報告

    Args:
        result: 模擬結果
        sensitivity_report: 敏感度報告
        risk_flags: 風險標記

    Returns:
        格式化文字報告
    """
    c = result.config
    sections: list[str] = []

    # ── 標頭與聲明 ─────────────────────────────────────────────────
    sections.append("=" * 78)
    sections.append("    固態火箭推進概念模擬平台 — 分析報告")
    sections.append(f"    產生時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    sections.append("=" * 78)
    sections.append("")
    sections.append(DISCLAIMER_ZH)
    sections.append("")

    # ── 幾何輸入 ───────────────────────────────────────────────────
    sections.append("─" * 78)
    sections.append("  1. 藥柱幾何輸入參數")
    sections.append("─" * 78)
    sections.append(f"  幾何類型:       {c.grain_type.value}")
    sections.append(f"  外徑:           {c.outer_diameter_mm:.2f} mm")
    sections.append(f"  單段長度:       {c.length_mm:.2f} mm")
    sections.append(f"  中心孔直徑:     {c.core_diameter_mm:.2f} mm")
    sections.append(f"  段數:           {c.num_segments}")
    sections.append(f"  抑制端面數/段:  {c.inhibited_ends}")
    sections.append(f"  噴嘴喉部直徑:  {c.nozzle_throat_diameter_mm:.2f} mm")
    sections.append(f"  肉厚:           {result.burn_profile.web_thickness_mm:.2f} mm")
    sections.append("")

    # ── 工程設計參考表 ─────────────────────────────────────────────
    sections.append("─" * 78)
    sections.append("  2. 工程設計參考表（⚠ 示意估算 — 須經專業驗證）")
    sections.append("─" * 78)
    sections.append("  ┌─────────────────────────────────────────────────────────┐")
    sections.append("  │  ⚠ 以下數值來自簡化模型，僅供初步方向參考               │")
    sections.append("  │  實際製造前務必經過專業工程審查與實驗驗證               │")
    sections.append("  └─────────────────────────────────────────────────────────┘")
    sections.append("")

    # 幾何尺寸
    sections.append("  【藥柱製造尺寸】")
    sections.append(f"    外徑 (OD):             {c.outer_diameter_mm:.2f} mm")
    sections.append(f"    內徑 (ID / 中心孔):    {c.core_diameter_mm:.2f} mm")
    sections.append(f"    單段長度:              {c.length_mm:.2f} mm")
    sections.append(f"    段數:                  {c.num_segments}")
    sections.append(f"    總藥柱長度:            {c.length_mm * c.num_segments:.2f} mm")
    sections.append(f"    肉厚 (web):            {result.burn_profile.web_thickness_mm:.2f} mm")
    sections.append(f"    長徑比 (L/D) 每段:     {c.length_mm / c.outer_diameter_mm:.2f}")
    if c.core_diameter_mm > 0:
        sections.append(f"    孔徑比 (ID/OD):        {c.core_diameter_mm / c.outer_diameter_mm:.3f}")
    sections.append("")

    # 噴嘴
    r_throat = c.nozzle_throat_diameter_mm / 2.0
    At = math.pi * r_throat ** 2
    sections.append("  【噴嘴參考尺寸】")
    sections.append(f"    喉部直徑:              {c.nozzle_throat_diameter_mm:.2f} mm")
    sections.append(f"    喉部面積:              {At:.2f} mm²")
    # 出口面積估算（膨脹比 ~4 作為示意）
    expansion_ratio = 4.0
    Ae = At * expansion_ratio
    De = math.sqrt(4 * Ae / math.pi)
    sections.append(f"    示意出口直徑 (ε≈{expansion_ratio:.0f}): {De:.2f} mm [⚠ 示意]")
    sections.append(f"    示意出口面積:          {Ae:.2f} mm² [⚠ 示意]")
    sections.append("")

    # 殼體
    sections.append("  【燃燒室殼體參考】")
    # 內徑 = 藥柱外徑 + 間隙
    casing_clearance = 1.0  # mm 概念間隙
    casing_id = c.outer_diameter_mm + casing_clearance
    # 殼體長度 = 總藥柱長度 + 前後空間
    headspace = 10.0  # mm 概念值
    casing_length = c.length_mm * c.num_segments + 2 * headspace
    sections.append(f"    建議殼體內徑:          ≥ {casing_id:.1f} mm [⚠ 示意]")
    sections.append(f"    建議殼體內長:          ≥ {casing_length:.1f} mm [⚠ 示意]")
    sections.append(f"    峰值壓力 (示意):       {result.peak_pressure_mpa:.2f} MPa [⚠ 示意]")

    # 安全係數提醒
    safety_factor = 2.5
    design_pressure = result.peak_pressure_mpa * safety_factor
    sections.append(f"    建議設計壓力 (SF={safety_factor}):  {design_pressure:.2f} MPa [⚠ 示意]")
    sections.append("    ※ 安全係數與材料選擇需由合格工程師決定")
    sections.append("")

    # 性能估算
    sections.append("  【性能估算值】（⚠ 示意估算 — 不可用於真實預測）")
    sections.append(f"    推進劑質量:            {result.propellant_mass_kg * 1000:.1f} g")
    sections.append(f"    估算燃燒時間:          {result.burn_time_s:.2f} s")
    sections.append(f"    估算峰值推力:          {result.peak_thrust_n:.1f} N")
    sections.append(f"    估算平均推力:          {result.average_thrust_n:.1f} N")
    sections.append(f"    估算總衝量:            {result.total_impulse_ns:.1f} N·s")
    sections.append(f"    估算比衝:              {result.specific_impulse_s:.0f} s")

    # 推力分級（NAR 分類參考）
    total_impulse = result.total_impulse_ns
    motor_class = _classify_motor(total_impulse)
    sections.append(f"    概念推力分級:          {motor_class} [⚠ 示意]")
    sections.append("")

    # ── 趨勢分析 ───────────────────────────────────────────────────
    sections.append("─" * 78)
    sections.append("  3. 燃燒趨勢分析（概念級）")
    sections.append("─" * 78)

    # 面積趨勢
    area_norm = result.burn_area_normalized
    valid_area = area_norm[area_norm > 0]
    if len(valid_area) >= 10:
        third = len(valid_area) // 3
        early = np.mean(valid_area[:third])
        late = np.mean(valid_area[2 * third:])
        ratio = late / early if early > 0 else 0

        if ratio > 1.15:
            trend = "漸進式 (Progressive)"
            desc = "燃燒面積隨時間增加，推力與壓力逐漸上升。"
        elif ratio < 0.85:
            trend = "漸退式 (Regressive)"
            desc = "燃燒面積隨時間減少，推力逐漸下降。"
        else:
            trend = "接近中性 (Near-Neutral)"
            desc = "燃燒面積在整個過程中相對穩定，推力曲線較平穩。"

        sections.append(f"  面積趨勢:     {trend}")
        sections.append(f"  後/前面積比:  {ratio:.3f}")
        sections.append(f"  說明:         {desc}")
    else:
        sections.append("  面積趨勢:     資料不足")

    # 推力平穩性
    thrust_norm = result.thrust_normalized
    valid_thrust = thrust_norm[thrust_norm > 0]
    if len(valid_thrust) >= 2:
        cv = float(np.std(valid_thrust) / np.mean(valid_thrust))
        if cv < 0.1:
            stability = "優良 — 接近理想中性燃燒"
        elif cv < 0.2:
            stability = "尚可 — 有一定波動但整體可接受"
        else:
            stability = "欠佳 — 推力波動明顯"
        sections.append(f"  推力穩定性:   {stability} (CV = {cv:.3f})")
    sections.append("")

    # Kn 統計
    kn = result.kn
    valid_kn = kn[kn > 0]
    if len(valid_kn) > 0:
        sections.append(f"  Kn 範圍:      {np.min(valid_kn):.0f} – {np.max(valid_kn):.0f}")
        sections.append(f"  Kn 平均:      {np.mean(valid_kn):.0f}")
        sections.append(f"  Kn 變化比:    {np.max(valid_kn) / np.min(valid_kn):.2f}×")
    sections.append("")

    # ── 敏感度分析 ─────────────────────────────────────────────────
    if sensitivity_report:
        sections.append("─" * 78)
        sections.append("  4. 敏感度分析結果")
        sections.append("─" * 78)

        for p in sensitivity_report.perturbations:
            marker = " ⚠" if p.is_high_sensitivity else ""
            sections.append(
                f"  {p.parameter_name:25s} {p.perturbation_pct:+6.1f}%  →  "
                f"推力 {p.thrust_change_pct:+6.1f}%  壓力 {p.pressure_change_pct:+6.1f}%  "
                f"衝量 {p.impulse_change_pct:+6.1f}%{marker}"
            )

        if sensitivity_report.high_sensitivity_params:
            sections.append("")
            sections.append("  ⚠ 高敏感度參數: " + ", ".join(sensitivity_report.high_sensitivity_params))
            sections.append("    這些參數的製造公差需要特別控制。")
        sections.append("")

    # ── 風險提示 ───────────────────────────────────────────────────
    if risk_flags:
        sections.append("─" * 78)
        sections.append("  5. 教學用風險提示（⚠ 僅供概念提醒，非安全判定）")
        sections.append("─" * 78)
        flagger = RiskFlagger()
        sections.append(flagger.format_report(risk_flags))
        sections.append("")

    # ── 模型假設與限制 ─────────────────────────────────────────────
    sections.append("─" * 78)
    sections.append("  6. 模型假設與限制")
    sections.append("─" * 78)
    sections.append("  本概念模型基於以下簡化假設：")
    sections.append("  • 穩態燃燒假設：每個時間步驟均視為穩態平衡")
    sections.append("  • 均勻燃速：整個燃燒面上燃速一致")
    sections.append("  • 聖維南燃速方程：r = a·P^n，係數為示意值")
    sections.append("  • 理想氣體假設")
    sections.append("  • 忽略侵蝕燃燒效應")
    sections.append("  • 忽略點火暫態過程")
    sections.append("  • 忽略噴嘴喉部侵蝕（喉部面積恆定）")
    sections.append("  • 忽略熱損失與殼體熱傳")
    sections.append("  • 忽略多相流效應")
    sections.append("  • 忽略製造公差與材料批次變異")
    sections.append("  • 忽略藥柱結構完整性（不計裂紋、脫粘等）")
    sections.append("")
    sections.append("  需要小心解讀的區域：")
    sections.append("  • 燃燒初期（點火暫態被省略）")
    sections.append("  • 燃燒末期（滅火暫態被省略）")
    sections.append("  • 高 Kn 區域（侵蝕燃燒效應可能顯著）")
    sections.append("  • 壓力峰值區域（真實值可能因暫態效應更高）")
    sections.append("")

    # ── 示意推進劑參數 ─────────────────────────────────────────────
    sections.append("─" * 78)
    sections.append("  7. 使用的示意推進劑參數")
    sections.append("─" * 78)
    sections.append(f"  密度:         {PROPELLANT_DENSITY_KG_M3:.0f} kg/m³")
    sections.append(f"  燃速係數 a:   {BURN_RATE_COEFFICIENT_A:.4e} m/s (P in Pa)")
    sections.append(f"  壓力指數 n:   {BURN_RATE_EXPONENT_N:.2f}")
    sections.append(f"  C* 特徵速度:  {CHARACTERISTIC_VELOCITY_M_S:.0f} m/s")
    sections.append(f"  推力係數 Cf:  {THRUST_COEFFICIENT_CF:.2f}")
    sections.append("  ⚠ 以上均為概念示意值，不代表任何特定推進劑。")
    sections.append("")
    sections.append("=" * 78)
    sections.append("  報告結束 — 本工具僅供教學與概念分析")
    sections.append("=" * 78)

    return "\n".join(sections)


def _classify_motor(total_impulse_ns: float) -> str:
    """根據總衝量分類（NAR 分類參考）"""
    classes = [
        (0.3125, "1/8A"), (0.625, "1/4A"), (1.25, "1/2A"),
        (2.5, "A"), (5.0, "B"), (10.0, "C"),
        (20.0, "D"), (40.0, "E"), (80.0, "F"),
        (160.0, "G"), (320.0, "H"), (640.0, "I"),
        (1280.0, "J"), (2560.0, "K"), (5120.0, "L"),
        (10240.0, "M"), (20480.0, "N"), (40960.0, "O"),
    ]
    for threshold, name in classes:
        if total_impulse_ns <= threshold:
            return name
    return "O+"
