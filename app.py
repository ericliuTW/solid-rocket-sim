"""
固態火箭推進概念模擬平台 — Streamlit 互動介面
Solid Rocket Propulsion Conceptual Simulation Platform — Streamlit UI

啟動方式: streamlit run app.py

╔══════════════════════════════════════════════════════════════════════╗
║  本工具僅供教學、概念分析與趨勢比較使用，                            ║
║  不能直接用於真實推進器設計、製造或安全驗證。                         ║
╚══════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import math

import streamlit as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from constants import (
    GrainConfig,
    GrainType,
    EXAMPLE_CONFIGS,
    DISCLAIMER_ZH,
    KN_HIGH_THRESHOLD,
    KN_LOW_THRESHOLD,
)
from geometry import GrainGeometry
from simulation import ConceptSimulator, SimulationResult
from sensitivity import SensitivityAnalyzer, SensitivityReport
from risk_warnings import RiskFlagger, RiskFlag, RiskLevel
from plotting import PlotManager, COLORS
from report import generate_text_report
from engine_drawing import EngineDrawing


# ── 頁面設定 ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="固態火箭推進概念模擬平台",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 標題與聲明 ────────────────────────────────────────────────────────
st.title("🚀 固態火箭推進概念模擬平台")
st.markdown("**Solid Rocket Propulsion Conceptual Simulation Platform**")

with st.expander("⚠️ 重要聲明 — 教學與概念用途（點擊展開）", expanded=True):
    st.warning(
        "本工具僅供**教學、概念分析與趨勢比較**使用。\n\n"
        "**不能直接用於真實推進器設計、製造或安全驗證。**\n\n"
        "本模型忽略了許多真實世界效應（侵蝕燃燒、溫度效應、點火暫態、"
        "噴嘴侵蝕、多相流、殼體熱傳、製造公差等）。\n\n"
        "本平台提供的「工程設計參考表」數值來自簡化模型推算，"
        "僅可作為初步方向參考，實際製造前**務必經過專業驗證**。\n\n"
        "若需真實用途，必須依賴專業實驗、材料測試與合格工程審查。"
    )

# ── 側邊欄：參數控制 ──────────────────────────────────────────────────
st.sidebar.header("📐 藥柱幾何參數")

# 預設選擇
preset = st.sidebar.selectbox(
    "預設範例",
    ["自訂"] + list(EXAMPLE_CONFIGS.keys()),
    index=0,
)

if preset != "自訂":
    default_cfg = EXAMPLE_CONFIGS[preset]
else:
    default_cfg = GrainConfig()

grain_type_str = st.sidebar.selectbox(
    "幾何類型",
    ["bates", "cylindrical", "end_burner", "star", "moon_burner"],
    index=["bates", "cylindrical", "end_burner", "star", "moon_burner"].index(
        default_cfg.grain_type.value
    ),
    format_func=lambda x: {
        "bates": "BATES 多段式",
        "cylindrical": "圓柱穿孔",
        "end_burner": "端面燃燒",
        "star": "星形（簡化）",
        "moon_burner": "偏心孔（簡化）",
    }[x],
)

grain_type_map = {
    "bates": GrainType.BATES,
    "cylindrical": GrainType.CYLINDRICAL,
    "end_burner": GrainType.END_BURNER,
    "star": GrainType.STAR,
    "moon_burner": GrainType.MOON_BURNER,
}

od = st.sidebar.number_input("外徑 OD (mm)", min_value=10.0, max_value=500.0,
                              value=default_cfg.outer_diameter_mm, step=1.0)
length = st.sidebar.number_input("單段長度 L (mm)", min_value=5.0, max_value=1000.0,
                                  value=default_cfg.length_mm, step=1.0)

is_end_burner = grain_type_str == "end_burner"
cd = st.sidebar.number_input(
    "中心孔直徑 CD (mm)",
    min_value=0.0,
    max_value=od - 1.0 if not is_end_burner else 0.0,
    value=default_cfg.core_diameter_mm if not is_end_burner else 0.0,
    step=0.5,
    disabled=is_end_burner,
)

segments = st.sidebar.number_input("段數", min_value=1, max_value=20,
                                    value=default_cfg.num_segments, step=1)
inhibited = st.sidebar.selectbox("抑制端面數/段", [0, 1, 2],
                                  index=default_cfg.inhibited_ends)
throat = st.sidebar.number_input("噴嘴喉部直徑 (mm)", min_value=1.0, max_value=100.0,
                                  value=default_cfg.nozzle_throat_diameter_mm, step=0.5)

st.sidebar.divider()
st.sidebar.header("⚙️ 顯示設定")
display_mode = st.sidebar.radio(
    "輸出模式",
    ["歸一化 (Normalized)", "示意估算 (Estimated)", "兩者並列"],
    index=0,
)
run_sensitivity = st.sidebar.checkbox("執行敏感度分析", value=False)

st.sidebar.divider()
st.sidebar.header("📊 比較模式")
enable_compare = st.sidebar.checkbox("啟用多組比較", value=False)

compare_presets: list[str] = []
if enable_compare:
    compare_presets = st.sidebar.multiselect(
        "選擇比較組態",
        list(EXAMPLE_CONFIGS.keys()),
        default=list(EXAMPLE_CONFIGS.keys())[:2],
    )

# ── 建立設定 ──────────────────────────────────────────────────────────
config = GrainConfig(
    grain_type=grain_type_map[grain_type_str],
    outer_diameter_mm=od,
    length_mm=length,
    core_diameter_mm=cd if not is_end_burner else 0.0,
    num_segments=int(segments),
    inhibited_ends=int(inhibited),
    nozzle_throat_diameter_mm=throat,
)

# ── 執行模擬 ──────────────────────────────────────────────────────────
try:
    simulator = ConceptSimulator(config)
    result = simulator.run()
except (ValueError, ZeroDivisionError) as e:
    st.error(f"模擬錯誤: {e}")
    st.stop()

# 風險分析
flagger = RiskFlagger()
sens_report: SensitivityReport | None = None

if run_sensitivity:
    analyzer = SensitivityAnalyzer(config)
    sens_report = analyzer.run_analysis()

risk_flags = flagger.analyze(result, sens_report)

# ── 主頁面內容 ────────────────────────────────────────────────────────

# 幾何摘要卡片
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("推進劑質量", f"{result.propellant_mass_kg * 1000:.1f} g",
              help="⚠ 示意估算值")
with col2:
    st.metric("肉厚", f"{result.burn_profile.web_thickness_mm:.1f} mm")
with col3:
    kn_max = np.max(result.kn[result.kn > 0]) if np.any(result.kn > 0) else 0
    st.metric("Kn 峰值", f"{kn_max:.0f}")
with col4:
    from report import _classify_motor
    motor_class = _classify_motor(result.total_impulse_ns)
    st.metric("概念分級", motor_class, help="⚠ 示意估算")

# 示意性能卡片
st.subheader("⚠ 示意性能估算（僅供參考，不可用於真實預測）")
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.metric("燃燒時間", f"{result.burn_time_s:.2f} s")
with c2:
    st.metric("峰值推力", f"{result.peak_thrust_n:.1f} N")
with c3:
    st.metric("平均推力", f"{result.average_thrust_n:.1f} N")
with c4:
    st.metric("總衝量", f"{result.total_impulse_ns:.1f} N·s")
with c5:
    st.metric("峰值壓力", f"{result.peak_pressure_mpa:.2f} MPa")

# ── 引擎幾何剖面圖 ───────────────────────────────────────────────────
st.divider()
st.subheader("🔧 引擎幾何剖面圖")

drawer = EngineDrawing(config)

engine_tab1, engine_tab2, engine_tab3 = st.tabs([
    "引擎組裝剖面", "藥柱橫截面", "燃燒退化序列"
])

with engine_tab1:
    fig_asm = drawer.draw_engine_assembly()
    st.pyplot(fig_asm)
    plt.close(fig_asm)

with engine_tab2:
    burn_pct = st.slider("燃燒進程 (%)", 0, 95, 0, step=5, key="cross_burn")
    fig_cs = drawer.draw_cross_section_only(burn_fraction=burn_pct / 100.0)
    st.pyplot(fig_cs)
    plt.close(fig_cs)

with engine_tab3:
    n_frames = st.selectbox("序列幀數", [4, 6, 8], index=1, key="burn_frames")
    fig_seq = drawer.draw_burn_sequence(n_frames=n_frames)
    st.pyplot(fig_seq)
    plt.close(fig_seq)

# ── 圖表 ─────────────────────────────────────────────────────────────
st.divider()

plotter = PlotManager()


def _plot_panel(res: SimulationResult, mode: str, flags: list[RiskFlag]) -> None:
    """繪製一組完整面板"""
    fig = plotter.plot_single_result(res, mode=mode, risk_flags=flags)
    st.pyplot(fig)
    plt.close(fig)


if display_mode == "歸一化 (Normalized)":
    st.subheader("📈 歸一化趨勢圖")
    _plot_panel(result, "normalized", risk_flags)
elif display_mode == "示意估算 (Estimated)":
    st.subheader("📈 示意估算圖（⚠ 僅供參考）")
    _plot_panel(result, "estimated", risk_flags)
else:
    tab1, tab2 = st.tabs(["歸一化 (Normalized)", "示意估算 (Estimated)"])
    with tab1:
        _plot_panel(result, "normalized", risk_flags)
    with tab2:
        st.warning("以下為示意估算值，僅供建立直覺，不可用於真實設計。")
        _plot_panel(result, "estimated", risk_flags)

# ── 多組比較 ──────────────────────────────────────────────────────────
if enable_compare and compare_presets:
    st.divider()
    st.subheader("📊 多組幾何趨勢比較")

    compare_results = [result]  # 包含當前設定
    for name in compare_presets:
        cfg = EXAMPLE_CONFIGS[name]
        try:
            sim = ConceptSimulator(cfg)
            compare_results.append(sim.run())
        except (ValueError, ZeroDivisionError):
            st.warning(f"跳過無法模擬的設定: {name}")

    if len(compare_results) > 1:
        fig = plotter.plot_comparison(compare_results, mode="normalized")
        st.pyplot(fig)
        plt.close(fig)

# ── 敏感度分析 ────────────────────────────────────────────────────────
if sens_report:
    st.divider()
    st.subheader("🔍 敏感度分析")

    fig = plotter.plot_sensitivity(sens_report)
    st.pyplot(fig)
    plt.close(fig)

    # 敏感度數據表
    st.markdown("**擾動結果明細:**")
    rows = []
    for p in sens_report.perturbations:
        rows.append({
            "參數": p.parameter_name,
            "擾動": f"{p.perturbation_pct:+.1f}%",
            "推力變化": f"{p.thrust_change_pct:+.1f}%",
            "壓力變化": f"{p.pressure_change_pct:+.1f}%",
            "衝量變化": f"{p.impulse_change_pct:+.1f}%",
            "高敏感": "⚠" if p.is_high_sensitivity else "",
        })
    st.dataframe(rows, use_container_width=True)

    if sens_report.high_sensitivity_params:
        st.warning(
            f"⚠ 高敏感度參數: {', '.join(sens_report.high_sensitivity_params)}\n\n"
            "這些參數的微小變動可能導致性能顯著變化，製造公差需要特別控制。"
        )

# ── 風險提示 ──────────────────────────────────────────────────────────
st.divider()
st.subheader("⚠ 教學用風險提示")
st.caption("此系統僅做「概念風險提醒」，不夠格作為真實安全判定工具。")

if risk_flags:
    for flag in sorted(risk_flags, key=lambda f: {
        RiskLevel.CRITICAL: 0, RiskLevel.WARNING: 1,
        RiskLevel.CAUTION: 2, RiskLevel.INFO: 3,
    }.get(f.level, 99)):
        if flag.level == RiskLevel.CRITICAL:
            st.error(f"**{flag.title}**\n\n{flag.description}\n\n*建議: {flag.recommendation}*")
        elif flag.level == RiskLevel.WARNING:
            st.warning(f"**{flag.title}**\n\n{flag.description}\n\n*建議: {flag.recommendation}*")
        elif flag.level == RiskLevel.CAUTION:
            st.info(f"**{flag.title}**\n\n{flag.description}\n\n*建議: {flag.recommendation}*")
        else:
            st.caption(f"ℹ️ **{flag.title}** — {flag.description}")

# ── 工程設計參考表 ────────────────────────────────────────────────────
st.divider()
st.subheader("📋 工程設計參考表")
st.warning("以下數值來自簡化模型推算，僅供初步方向參考，實際製造前務必經過專業驗證。")

col_a, col_b = st.columns(2)

with col_a:
    st.markdown("**藥柱製造尺寸**")
    web = result.burn_profile.web_thickness_mm
    st.markdown(f"""
    | 項目 | 數值 |
    |------|------|
    | 外徑 (OD) | {config.outer_diameter_mm:.2f} mm |
    | 內徑 (ID) | {config.core_diameter_mm:.2f} mm |
    | 單段長度 | {config.length_mm:.2f} mm |
    | 段數 | {config.num_segments} |
    | 總藥柱長度 | {config.length_mm * config.num_segments:.2f} mm |
    | 肉厚 (web) | {web:.2f} mm |
    | 長徑比 (L/D) | {config.length_mm / config.outer_diameter_mm:.2f} |
    """)

with col_b:
    st.markdown("**噴嘴與殼體參考 [⚠ 示意]**")
    r_t = config.nozzle_throat_diameter_mm / 2.0
    At = math.pi * r_t ** 2
    Ae = At * 4.0
    De = math.sqrt(4 * Ae / math.pi)
    casing_id = config.outer_diameter_mm + 1.0
    casing_len = config.length_mm * config.num_segments + 20.0
    design_p = result.peak_pressure_mpa * 2.5
    st.markdown(f"""
    | 項目 | 數值 |
    |------|------|
    | 喉部直徑 | {config.nozzle_throat_diameter_mm:.2f} mm |
    | 喉部面積 | {At:.2f} mm² |
    | 示意出口直徑 (ε≈4) | {De:.2f} mm |
    | 建議殼體內徑 | ≥ {casing_id:.1f} mm |
    | 建議殼體內長 | ≥ {casing_len:.1f} mm |
    | 峰值壓力 (示意) | {result.peak_pressure_mpa:.2f} MPa |
    | 設計壓力 (SF=2.5) | {design_p:.2f} MPa |
    """)

# ── 完整報告下載 ──────────────────────────────────────────────────────
st.divider()
st.subheader("📄 完整報告")

report_text = generate_text_report(result, sens_report, risk_flags)

with st.expander("檢視完整文字報告"):
    st.code(report_text, language=None)

st.download_button(
    label="📥 下載完整報告 (.txt)",
    data=report_text.encode("utf-8"),
    file_name="solid_rocket_concept_report.txt",
    mime="text/plain",
)

# ── 模型假設面板 ──────────────────────────────────────────────────────
st.divider()
with st.expander("📖 模型假設與限制"):
    st.markdown("""
    **本概念模型基於以下簡化假設：**

    - **穩態燃燒假設**：每個時間步驟均視為穩態平衡
    - **均勻燃速**：整個燃燒面上燃速一致
    - **聖維南燃速方程**：r = a·P^n，係數為示意值
    - **理想氣體假設**
    - 忽略侵蝕燃燒效應
    - 忽略點火暫態過程
    - 忽略噴嘴喉部侵蝕（喉部面積恆定）
    - 忽略熱損失與殼體熱傳
    - 忽略多相流效應
    - 忽略製造公差與材料批次變異
    - 忽略藥柱結構完整性（不計裂紋、脫粘等）

    **未納入的真實世界效應：**

    - 侵蝕燃燒（高流速區域燃速增加）
    - 溫度對燃速的影響
    - 殼體熱傳與結構應力
    - 噴嘴侵蝕與喉部面積變化
    - 多相流效應（金屬氧化物顆粒等）
    - 點火暫態與滅火暫態
    - 製造公差與材料批次差異
    - 藥柱裂紋與脫粘
    - 重力與加速度效應
    """)

# ── 頁尾 ─────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "🚀 固態火箭推進概念模擬平台 | "
    "本工具僅供教學、概念分析與趨勢比較使用 | "
    "不能直接用於真實推進器設計、製造或安全驗證"
)
