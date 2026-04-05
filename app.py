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
from trajectory import TrajectoryEstimator, RocketConfig


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
                              value=min(max(10.0, default_cfg.outer_diameter_mm), 500.0), step=1.0)
length = st.sidebar.number_input("單段長度 L (mm)", min_value=5.0, max_value=1000.0,
                                  value=min(max(5.0, default_cfg.length_mm), 1000.0), step=1.0)

is_end_burner = grain_type_str == "end_burner"
_cd_max = od - 1.0 if not is_end_burner else 0.0
_cd_default = min(default_cfg.core_diameter_mm, _cd_max) if not is_end_burner else 0.0
cd = st.sidebar.number_input(
    "中心孔直徑 CD (mm)",
    min_value=0.0,
    max_value=_cd_max,
    value=max(0.0, _cd_default),
    step=0.5,
    disabled=is_end_burner,
)

segments = st.sidebar.number_input("段數", min_value=1, max_value=20,
                                    value=min(max(1, default_cfg.num_segments), 20), step=1)
inhibited = st.sidebar.selectbox("抑制端面數/段", [0, 1, 2],
                                  index=min(default_cfg.inhibited_ends, 2))
throat = st.sidebar.number_input("噴嘴喉部直徑 (mm)", min_value=1.0, max_value=100.0,
                                  value=min(max(1.0, default_cfg.nozzle_throat_diameter_mm), 100.0), step=0.5)

st.sidebar.divider()
st.sidebar.header("⚙️ 顯示設定")
display_mode = st.sidebar.radio(
    "輸出模式",
    ["歸一化 (Normalized)", "示意估算 (Estimated)", "兩者並列"],
    index=0,
)
run_sensitivity = st.sidebar.checkbox("執行敏感度分析", value=False)

st.sidebar.divider()
st.sidebar.header("🚀 飛行軌跡估算")
run_trajectory = st.sidebar.checkbox("啟用飛行軌跡估算", value=True)
total_rocket_mass_g = st.sidebar.number_input(
    "火箭總質量 (g)（含推進劑）", min_value=100.0, max_value=50000.0,
    value=2500.0, step=100.0,
    help="整支火箭的起飛重量，包含推進劑、殼體、機身、鰭、頭錐、回收系統、酬載等一切質量",
)
rocket_body_dia = st.sidebar.number_input(
    "火箭體直徑 (mm)", min_value=10.0, max_value=300.0,
    value=54.0, step=1.0, help="火箭外管直徑，影響空氣阻力",
)
drag_cd = st.sidebar.number_input(
    "阻力係數 Cd", min_value=0.1, max_value=1.5,
    value=0.45, step=0.05, help="典型模型火箭 0.3~0.6",
)

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

# ── 飛行軌跡估算 ──────────────────────────────────────────────────────
if run_trajectory:
    st.divider()
    st.subheader("🚀 飛行軌跡概念估算")
    st.warning(
        "以下為高度簡化的 1D 垂直飛行估算，忽略風、姿態、地球曲率等。"
        "僅供建立直覺，不可用於真實飛行預測。"
    )

    total_mass_kg = total_rocket_mass_g / 1000.0
    propellant_mass_kg = result.propellant_mass_kg
    # 結構+酬載 = 總質量 - 推進劑
    dry_mass_kg = max(total_mass_kg - propellant_mass_kg, 0.1)

    rocket_cfg = RocketConfig(
        body_diameter_mm=rocket_body_dia,
        drag_coefficient=drag_cd,
        structural_mass_kg=dry_mass_kg,
        payload_mass_kg=0.0,  # 已包含在總質量中
    )
    traj_estimator = TrajectoryEstimator(result, rocket_cfg)

    try:
        traj = traj_estimator.estimate()

        # 數值卡片
        tc1, tc2, tc3, tc4 = st.columns(4)
        with tc1:
            st.metric("最大高度", f"{traj.max_altitude_m:.0f} m")
        with tc2:
            st.metric("最大速度", f"{traj.max_velocity_m_s:.0f} m/s")
        with tc3:
            st.metric("最大加速度", f"{traj.max_acceleration_g:.1f} G")
        with tc4:
            st.metric("推重比", f"{traj.thrust_to_weight_ratio:.1f}")

        tc5, tc6, tc7, tc8 = st.columns(4)
        with tc5:
            st.metric("起飛總質量", f"{traj.total_mass_kg * 1000:.0f} g")
        with tc6:
            st.metric("乾重（不含推進劑）", f"{dry_mass_kg * 1000:.0f} g")
        with tc7:
            st.metric("燃盡速度", f"{traj.burnout_velocity_m_s:.0f} m/s")
        with tc8:
            st.metric("燃盡高度", f"{traj.burnout_altitude_m:.0f} m")

        if traj.thrust_to_weight_ratio < 1.0:
            st.error(
                f"推重比 {traj.thrust_to_weight_ratio:.2f} < 1.0 — "
                "火箭無法離開發射架！需要減輕質量或增大推力。"
            )

        # ── 飛行軌跡圖 ───────────────────────────────────────
        fig_traj, axes_t = plt.subplots(1, 3, figsize=(15, 5))
        fig_traj.suptitle(
            "飛行軌跡概念估算 [!] 僅供參考",
            fontsize=13, fontweight="bold",
        )

        # 高度 vs 時間
        ax_h = axes_t[0]
        ax_h.plot(traj.time_s, traj.altitude_m, color="#2563EB", linewidth=2)
        ax_h.axvline(x=traj.burnout_time_s, color="#DC2626", linestyle="--",
                     linewidth=0.8, alpha=0.7, label="燃盡")
        ax_h.set_title("高度 vs 時間")
        ax_h.set_xlabel("時間 (s)")
        ax_h.set_ylabel("高度 (m)")
        ax_h.legend(fontsize=8)

        # 速度 vs 時間
        ax_v = axes_t[1]
        ax_v.plot(traj.time_s, traj.velocity_m_s, color="#16A34A", linewidth=2)
        ax_v.axvline(x=traj.burnout_time_s, color="#DC2626", linestyle="--",
                     linewidth=0.8, alpha=0.7, label="燃盡")
        ax_v.axhline(y=0, color="gray", linewidth=0.5)
        ax_v.set_title("速度 vs 時間")
        ax_v.set_xlabel("時間 (s)")
        ax_v.set_ylabel("速度 (m/s)")
        ax_v.legend(fontsize=8)

        # 加速度 vs 時間
        ax_a = axes_t[2]
        ax_a.plot(traj.time_s, traj.acceleration_m_s2 / 9.81, color="#D97706", linewidth=2)
        ax_a.axvline(x=traj.burnout_time_s, color="#DC2626", linestyle="--",
                     linewidth=0.8, alpha=0.7, label="燃盡")
        ax_a.axhline(y=0, color="gray", linewidth=0.5)
        ax_a.set_title("加速度 vs 時間")
        ax_a.set_xlabel("時間 (s)")
        ax_a.set_ylabel("加速度 (G)")
        ax_a.legend(fontsize=8)

        fig_traj.tight_layout(rect=[0, 0, 1, 0.93])
        st.pyplot(fig_traj)
        plt.close(fig_traj)

        # ── 總質量-高度曲線 ────────────────────────────────────
        st.markdown("**火箭總質量 vs 最大高度**")
        st.caption("不同總質量下，火箭能達到的概念估算高度。")

        # 掃描總質量範圍
        min_total_g = propellant_mass_kg * 1000 + 100  # 至少推進劑+100g
        max_total_g = max(total_rocket_mass_g * 4, min_total_g + 1000)
        sweep_total_g = np.linspace(min_total_g, max_total_g, 10)
        sweep_altitudes = np.zeros(len(sweep_total_g))

        for i, tg in enumerate(sweep_total_g):
            sweep_dry = max(tg / 1000.0 - propellant_mass_kg, 0.1)
            traj_estimator.rocket.structural_mass_kg = sweep_dry
            traj_estimator.rocket.payload_mass_kg = 0.0
            try:
                r = traj_estimator.estimate(dt=0.02)
                alt = r.max_altitude_m if r.thrust_to_weight_ratio >= 1.0 else 0.0
                sweep_altitudes[i] = max(0.0, alt)
            except (ValueError, ZeroDivisionError):
                sweep_altitudes[i] = 0.0

        # 恢復原設定
        traj_estimator.rocket.structural_mass_kg = dry_mass_kg
        traj_estimator.rocket.payload_mass_kg = 0.0

        fig_pa, ax_pa = plt.subplots(1, 1, figsize=(8, 5))
        ax_pa.plot(sweep_total_g, sweep_altitudes, color="#7C3AED", linewidth=2.5)
        ax_pa.fill_between(sweep_total_g, sweep_altitudes, alpha=0.1, color="#7C3AED")

        # 標記當前總質量
        current_alt = traj.max_altitude_m
        ax_pa.plot(total_rocket_mass_g, current_alt, "ro", markersize=10, zorder=5)
        ax_pa.annotate(
            f"  {total_rocket_mass_g:.0f}g -> {current_alt:.0f}m",
            xy=(total_rocket_mass_g, current_alt),
            fontsize=10, color="#DC2626", fontweight="bold",
        )

        # 找出推重比=1的臨界質量
        for i in range(len(sweep_altitudes)):
            if sweep_altitudes[i] <= 0 and i > 0:
                ax_pa.axvline(x=sweep_total_g[i], color="#DC2626", linestyle=":",
                             linewidth=1, alpha=0.7)
                ax_pa.text(sweep_total_g[i], max(sweep_altitudes) * 0.5,
                          f" TWR<1\n No lift-off",
                          fontsize=8, color="#DC2626")
                break

        ax_pa.set_title("火箭總質量 vs 最大高度 [!] 概念估算", fontsize=12, fontweight="bold")
        ax_pa.set_xlabel("火箭總質量 (g)")
        ax_pa.set_ylabel("最大高度 (m)")
        ax_pa.set_xlim(sweep_total_g[0], sweep_total_g[-1])
        ax_pa.set_ylim(bottom=0)
        ax_pa.grid(True, alpha=0.3)

        fig_pa.tight_layout()
        st.pyplot(fig_pa)
        plt.close(fig_pa)

    except Exception as e:
        st.error(f"軌跡估算錯誤: {e}")

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
