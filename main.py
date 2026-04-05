"""
固態火箭推進概念模擬平台 — CLI 主程式
Solid Rocket Propulsion Conceptual Simulation Platform — CLI Entry Point

╔══════════════════════════════════════════════════════════════════════╗
║  本工具僅供教學、概念分析與趨勢比較使用，                            ║
║  不能直接用於真實推進器設計、製造或安全驗證。                         ║
╚══════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from constants import (
    GrainConfig,
    GrainType,
    EXAMPLE_CONFIGS,
    DISCLAIMER_ZH,
    DEFAULT_NOZZLE_THROAT_DIAMETER_MM,
)
from geometry import GrainGeometry
from simulation import ConceptSimulator
from sensitivity import SensitivityAnalyzer
from risk_warnings import RiskFlagger
from plotting import PlotManager
from report import generate_text_report
from engine_drawing import EngineDrawing


def main() -> None:
    """CLI 主入口"""
    print(DISCLAIMER_ZH)
    print()

    parser = argparse.ArgumentParser(
        description="固態火箭推進概念模擬平台 (教學用)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", help="子命令")

    # ── simulate 命令 ─────────────────────────────────────────────
    sim_parser = sub.add_parser("simulate", help="執行單組概念模擬")
    _add_geometry_args(sim_parser)
    sim_parser.add_argument("--mode", choices=["normalized", "estimated", "both"],
                            default="both", help="輸出模式")
    sim_parser.add_argument("--sensitivity", action="store_true", help="執行敏感度分析")
    sim_parser.add_argument("--save-dir", type=str, default="./output", help="輸出目錄")
    sim_parser.add_argument("--show", action="store_true", help="顯示圖表")

    # ── compare 命令 ──────────────────────────────────────────────
    cmp_parser = sub.add_parser("compare", help="比較預設範例幾何")
    cmp_parser.add_argument("--mode", choices=["normalized", "estimated"],
                            default="normalized", help="輸出模式")
    cmp_parser.add_argument("--save-dir", type=str, default="./output", help="輸出目錄")
    cmp_parser.add_argument("--show", action="store_true", help="顯示圖表")

    # ── examples 命令 ─────────────────────────────────────────────
    sub.add_parser("examples", help="顯示範例設定")

    args = parser.parse_args()

    if args.command == "simulate":
        _run_simulate(args)
    elif args.command == "compare":
        _run_compare(args)
    elif args.command == "examples":
        _show_examples()
    else:
        parser.print_help()


def _add_geometry_args(parser: argparse.ArgumentParser) -> None:
    """添加幾何參數 CLI 引數"""
    parser.add_argument("--type", type=str, default="bates",
                        choices=["cylindrical", "bates", "end_burner", "star", "moon_burner"],
                        help="藥柱幾何類型")
    parser.add_argument("--od", type=float, default=50.0, help="外徑 (mm)")
    parser.add_argument("--length", type=float, default=70.0, help="單段長度 (mm)")
    parser.add_argument("--cd", type=float, default=18.0, help="中心孔直徑 (mm)")
    parser.add_argument("--segments", type=int, default=4, help="段數")
    parser.add_argument("--inhibited-ends", type=int, default=0, choices=[0, 1, 2],
                        help="每段被抑制的端面數")
    parser.add_argument("--throat", type=float, default=DEFAULT_NOZZLE_THROAT_DIAMETER_MM,
                        help="噴嘴喉部直徑 (mm)")
    parser.add_argument("--label", type=str, default="", help="顯示標籤")


def _build_config(args: argparse.Namespace) -> GrainConfig:
    """從 CLI 引數建立 GrainConfig"""
    type_map = {
        "cylindrical": GrainType.CYLINDRICAL,
        "bates": GrainType.BATES,
        "end_burner": GrainType.END_BURNER,
        "star": GrainType.STAR,
        "moon_burner": GrainType.MOON_BURNER,
    }
    return GrainConfig(
        grain_type=type_map[args.type],
        outer_diameter_mm=args.od,
        length_mm=args.length,
        core_diameter_mm=args.cd,
        num_segments=args.segments,
        inhibited_ends=args.inhibited_ends,
        nozzle_throat_diameter_mm=args.throat,
        label=args.label,
    )


def _run_simulate(args: argparse.Namespace) -> None:
    """執行單組模擬"""
    config = _build_config(args)
    print(f"\n▶ 模擬幾何: {config.label}")
    print(f"  類型={config.grain_type.value}, OD={config.outer_diameter_mm}mm, "
          f"L={config.length_mm}mm, CD={config.core_diameter_mm}mm, "
          f"段數={config.num_segments}, 喉徑={config.nozzle_throat_diameter_mm}mm")

    # 模擬
    simulator = ConceptSimulator(config)
    result = simulator.run()

    # 敏感度分析
    sens_report = None
    if args.sensitivity:
        print("\n▶ 執行敏感度分析...")
        analyzer = SensitivityAnalyzer(config)
        sens_report = analyzer.run_analysis()

    # 風險分析
    flagger = RiskFlagger()
    risk_flags = flagger.analyze(result, sens_report)

    # 報告
    report_text = generate_text_report(result, sens_report, risk_flags)
    print("\n" + report_text)

    # 繪圖
    save_dir = Path(args.save_dir)
    plotter = PlotManager(save_dir=save_dir)

    if args.mode in ("normalized", "both"):
        fig = plotter.plot_single_result(result, mode="normalized",
                                         risk_flags=risk_flags, show=args.show)
        print(f"  → 歸一化圖表已儲存至 {save_dir}")
        import matplotlib.pyplot as plt
        plt.close(fig)

    if args.mode in ("estimated", "both"):
        fig = plotter.plot_single_result(result, mode="estimated",
                                         risk_flags=risk_flags, show=args.show)
        print(f"  → 示意估算圖表已儲存至 {save_dir}")
        import matplotlib.pyplot as plt
        plt.close(fig)

    if sens_report:
        fig = plotter.plot_sensitivity(sens_report, show=args.show)
        print(f"  → 敏感度分析圖表已儲存至 {save_dir}")
        import matplotlib.pyplot as plt
        plt.close(fig)

    # 引擎幾何圖
    import matplotlib.pyplot as plt
    drawer = EngineDrawing(config)

    fig = drawer.draw_engine_assembly(show=args.show)
    fig.savefig(save_dir / "engine_assembly.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  → 引擎剖面圖已儲存至 {save_dir / 'engine_assembly.png'}")

    fig = drawer.draw_burn_sequence(n_frames=6, show=args.show)
    fig.savefig(save_dir / "burn_sequence.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  → 燃燒退化序列已儲存至 {save_dir / 'burn_sequence.png'}")

    # 儲存報告
    report_path = save_dir / "report.txt"
    report_path.write_text(report_text, encoding="utf-8")
    print(f"  → 文字報告已儲存至 {report_path}")


def _run_compare(args: argparse.Namespace) -> None:
    """比較預設範例"""
    print("\n▶ 比較預設範例幾何...")
    results = []
    for name, config in EXAMPLE_CONFIGS.items():
        print(f"  模擬: {name}")
        simulator = ConceptSimulator(config)
        results.append(simulator.run())

    save_dir = Path(args.save_dir)
    plotter = PlotManager(save_dir=save_dir)
    fig = plotter.plot_comparison(results, mode=args.mode, show=args.show)
    print(f"\n  → 比較圖表已儲存至 {save_dir}")

    import matplotlib.pyplot as plt
    plt.close(fig)

    # 為每個結果產生簡要摘要
    for result in results:
        flagger = RiskFlagger()
        flags = flagger.analyze(result)
        report = generate_text_report(result, risk_flags=flags)
        rpt_path = save_dir / f"report_{result.config.grain_type.value}.txt"
        rpt_path.write_text(report, encoding="utf-8")
    print(f"  → 個別報告已儲存至 {save_dir}")


def _show_examples() -> None:
    """顯示範例設定"""
    print("\n可用的預設範例:")
    print("─" * 60)
    for name, config in EXAMPLE_CONFIGS.items():
        print(f"  {name}:")
        print(f"    類型={config.grain_type.value}, OD={config.outer_diameter_mm}mm, "
              f"L={config.length_mm}mm, CD={config.core_diameter_mm}mm, "
              f"段數={config.num_segments}")
    print()
    print("範例命令:")
    print("  python main.py simulate --type bates --od 50 --length 70 --cd 18 --segments 4 --sensitivity")
    print("  python main.py compare")
    print("  python main.py simulate --type end_burner --od 50 --length 100 --cd 0 --segments 1")


if __name__ == "__main__":
    main()
