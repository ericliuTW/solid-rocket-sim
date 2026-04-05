"""
固態火箭推進概念模擬平台 — 敏感度分析模組
Sensitivity Analysis Module

提供參數微擾分析，評估幾何參數變動對推力曲線與壓力峰值的影響。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from copy import deepcopy

import numpy as np
from numpy.typing import NDArray

from constants import GrainConfig, SENSITIVITY_HIGH_THRESHOLD
from simulation import ConceptSimulator, SimulationResult


@dataclass
class PerturbationResult:
    """單一擾動的結果"""
    parameter_name: str
    perturbation_pct: float           # +5 or -5
    baseline_peak_thrust: float
    perturbed_peak_thrust: float
    thrust_change_pct: float
    baseline_peak_pressure: float
    perturbed_peak_pressure: float
    pressure_change_pct: float
    baseline_total_impulse: float
    perturbed_total_impulse: float
    impulse_change_pct: float
    perturbed_result: SimulationResult
    is_high_sensitivity: bool


@dataclass
class SensitivityReport:
    """完整敏感度分析報告"""
    baseline: SimulationResult
    perturbations: list[PerturbationResult]
    config: GrainConfig
    high_sensitivity_params: list[str]

    @property
    def max_thrust_sensitivity_pct(self) -> float:
        if not self.perturbations:
            return 0.0
        return max(abs(p.thrust_change_pct) for p in self.perturbations)

    @property
    def max_pressure_sensitivity_pct(self) -> float:
        if not self.perturbations:
            return 0.0
        return max(abs(p.pressure_change_pct) for p in self.perturbations)


class SensitivityAnalyzer:
    """敏感度分析器

    對指定幾何設定進行參數微擾，分析推力、壓力、衝量的變化。
    """

    DEFAULT_PERTURBATIONS: dict[str, list[float]] = {
        "core_diameter_mm": [-5.0, +5.0],
        "length_mm": [-5.0, +5.0],
        "outer_diameter_mm": [-5.0, +5.0],
    }

    def __init__(self, config: GrainConfig) -> None:
        self.config = config

    def run_analysis(
        self,
        perturbations: dict[str, list[float]] | None = None,
        include_segment_change: bool = True,
    ) -> SensitivityReport:
        """執行敏感度分析

        Args:
            perturbations: {參數名: [擾動百分比列表]}，預設 ±5%
            include_segment_change: 是否包含段數 ±1 的分析

        Returns:
            SensitivityReport
        """
        if perturbations is None:
            perturbations = self.DEFAULT_PERTURBATIONS

        # 基準模擬
        baseline_sim = ConceptSimulator(self.config)
        baseline = baseline_sim.run()

        results: list[PerturbationResult] = []
        high_sens_params: list[str] = []

        # 連續參數擾動
        for param_name, pct_list in perturbations.items():
            for pct in pct_list:
                perturbed_config = deepcopy(self.config)
                base_val = getattr(perturbed_config, param_name)
                new_val = base_val * (1.0 + pct / 100.0)

                # 確保值合理
                if new_val <= 0:
                    continue
                # 確保 core < outer
                if param_name == "core_diameter_mm" and new_val >= perturbed_config.outer_diameter_mm:
                    continue
                if param_name == "outer_diameter_mm" and new_val <= perturbed_config.core_diameter_mm:
                    continue

                setattr(perturbed_config, param_name, new_val)
                perturbed_config.label = f"{param_name} {pct:+.0f}%"

                try:
                    perturbed_sim = ConceptSimulator(perturbed_config)
                    perturbed = perturbed_sim.run()
                except (ValueError, ZeroDivisionError):
                    continue

                thrust_chg = _pct_change(baseline.peak_thrust_n, perturbed.peak_thrust_n)
                press_chg = _pct_change(baseline.peak_pressure_mpa, perturbed.peak_pressure_mpa)
                impulse_chg = _pct_change(baseline.total_impulse_ns, perturbed.total_impulse_ns)

                is_high = (
                    abs(thrust_chg) > SENSITIVITY_HIGH_THRESHOLD * 100
                    or abs(press_chg) > SENSITIVITY_HIGH_THRESHOLD * 100
                )

                if is_high and param_name not in high_sens_params:
                    high_sens_params.append(param_name)

                results.append(PerturbationResult(
                    parameter_name=param_name,
                    perturbation_pct=pct,
                    baseline_peak_thrust=baseline.peak_thrust_n,
                    perturbed_peak_thrust=perturbed.peak_thrust_n,
                    thrust_change_pct=thrust_chg,
                    baseline_peak_pressure=baseline.peak_pressure_mpa,
                    perturbed_peak_pressure=perturbed.peak_pressure_mpa,
                    pressure_change_pct=press_chg,
                    baseline_total_impulse=baseline.total_impulse_ns,
                    perturbed_total_impulse=perturbed.total_impulse_ns,
                    impulse_change_pct=impulse_chg,
                    perturbed_result=perturbed,
                    is_high_sensitivity=is_high,
                ))

        # 段數變動
        if include_segment_change and self.config.num_segments > 1:
            for delta in [-1, +1]:
                perturbed_config = deepcopy(self.config)
                new_seg = perturbed_config.num_segments + delta
                if new_seg < 1:
                    continue
                perturbed_config.num_segments = new_seg
                perturbed_config.label = f"segments {delta:+d}"

                try:
                    perturbed_sim = ConceptSimulator(perturbed_config)
                    perturbed = perturbed_sim.run()
                except (ValueError, ZeroDivisionError):
                    continue

                thrust_chg = _pct_change(baseline.peak_thrust_n, perturbed.peak_thrust_n)
                press_chg = _pct_change(baseline.peak_pressure_mpa, perturbed.peak_pressure_mpa)
                impulse_chg = _pct_change(baseline.total_impulse_ns, perturbed.total_impulse_ns)
                is_high = abs(thrust_chg) > SENSITIVITY_HIGH_THRESHOLD * 100

                results.append(PerturbationResult(
                    parameter_name="num_segments",
                    perturbation_pct=delta / self.config.num_segments * 100,
                    baseline_peak_thrust=baseline.peak_thrust_n,
                    perturbed_peak_thrust=perturbed.peak_thrust_n,
                    thrust_change_pct=thrust_chg,
                    baseline_peak_pressure=baseline.peak_pressure_mpa,
                    perturbed_peak_pressure=perturbed.peak_pressure_mpa,
                    pressure_change_pct=press_chg,
                    baseline_total_impulse=baseline.total_impulse_ns,
                    perturbed_total_impulse=perturbed.total_impulse_ns,
                    impulse_change_pct=impulse_chg,
                    perturbed_result=perturbed,
                    is_high_sensitivity=is_high,
                ))

        return SensitivityReport(
            baseline=baseline,
            perturbations=results,
            config=self.config,
            high_sensitivity_params=high_sens_params,
        )


def _pct_change(baseline: float, perturbed: float) -> float:
    """計算百分比變化"""
    if abs(baseline) < 1e-12:
        return 0.0
    return (perturbed - baseline) / baseline * 100.0
