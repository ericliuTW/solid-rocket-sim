"""
固態火箭推進概念模擬平台 — 模擬引擎模組
Simulation Engine Module

提供概念級時間步進模擬，計算 Kn、壓力、推力等隨時間變化趨勢。
同時提供「相對值模式」與「示意估算模式」。

⚠ 所有「示意估算值」均為極度簡化計算，僅供建立直覺，不可用於真實設計。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from constants import (
    GrainConfig,
    PROPELLANT_DENSITY_KG_M3,
    BURN_RATE_COEFFICIENT_A,
    BURN_RATE_EXPONENT_N,
    CHARACTERISTIC_VELOCITY_M_S,
    THRUST_COEFFICIENT_CF,
    NOZZLE_EFFICIENCY,
    COMBUSTION_EFFICIENCY,
    DEFAULT_TIME_STEPS,
)
from geometry import GrainGeometry, BurnProfile


@dataclass
class SimulationResult:
    """模擬結果

    包含歸一化（相對值）與示意估算兩種輸出。

    Attributes:
        time_normalized: 歸一化時間 (0→1)
        burn_area_normalized: 歸一化燃燒面積 (相對初始值)
        kn_normalized: 歸一化 Kn 值 (相對初始值)
        pressure_normalized: 歸一化壓力 (相對初始值)
        thrust_normalized: 歸一化推力 (相對初始值)
        impulse_normalized: 歸一化累積衝量 (0→1)

        time_s: 示意時間 (秒) — ⚠ 示意估算值
        burn_area_mm2: 燃燒面積 (mm²)
        kn: Kn 值 (Ab/At)
        pressure_pa: 燃燒室壓力 (Pa) — ⚠ 示意估算值
        pressure_mpa: 燃燒室壓力 (MPa) — ⚠ 示意估算值
        thrust_n: 推力 (N) — ⚠ 示意估算值
        impulse_ns: 累積衝量 (N·s) — ⚠ 示意估算值
        total_impulse_ns: 總衝量 (N·s) — ⚠ 示意估算值
        burn_time_s: 燃燒時間 (秒) — ⚠ 示意估算值
        average_thrust_n: 平均推力 (N) — ⚠ 示意估算值
        peak_thrust_n: 峰值推力 (N) — ⚠ 示意估算值
        peak_pressure_mpa: 峰值壓力 (MPa) — ⚠ 示意估算值
        specific_impulse_s: 比衝 (秒) — ⚠ 示意估算值
        propellant_mass_kg: 推進劑質量 (kg) — ⚠ 示意估算值

        burn_profile: 原始燃燒面積曲線
        config: 幾何設定
    """
    # 歸一化值
    time_normalized: NDArray[np.float64]
    burn_area_normalized: NDArray[np.float64]
    kn_normalized: NDArray[np.float64]
    pressure_normalized: NDArray[np.float64]
    thrust_normalized: NDArray[np.float64]
    impulse_normalized: NDArray[np.float64]

    # 示意估算值
    time_s: NDArray[np.float64]
    burn_area_mm2: NDArray[np.float64]
    kn: NDArray[np.float64]
    pressure_pa: NDArray[np.float64]
    pressure_mpa: NDArray[np.float64]
    thrust_n: NDArray[np.float64]
    impulse_ns: NDArray[np.float64]
    total_impulse_ns: float
    burn_time_s: float
    average_thrust_n: float
    peak_thrust_n: float
    peak_pressure_mpa: float
    specific_impulse_s: float
    propellant_mass_kg: float

    # 參考
    burn_profile: BurnProfile
    config: GrainConfig


class ConceptSimulator:
    """概念級固態火箭模擬器

    使用簡化穩態燃燒模型：
        r = a · P^n                          (聖維南燃速方程)
        P = (ρ · a · C* · Ab / At)^(1/(1-n))  (穩態壓力方程)
        F = Cf · P · At                       (推力方程)

    ⚠ 這是高度簡化的穩態假設，忽略暫態效應、侵蝕燃燒等。
    """

    def __init__(
        self,
        config: GrainConfig,
        propellant_density: float = PROPELLANT_DENSITY_KG_M3,
        burn_rate_a: float = BURN_RATE_COEFFICIENT_A,
        burn_rate_n: float = BURN_RATE_EXPONENT_N,
        c_star: float = CHARACTERISTIC_VELOCITY_M_S,
        cf: float = THRUST_COEFFICIENT_CF,
        nozzle_eff: float = NOZZLE_EFFICIENCY,
        combustion_eff: float = COMBUSTION_EFFICIENCY,
    ) -> None:
        self.config = config
        self.rho = propellant_density
        self.a = burn_rate_a
        self.n = burn_rate_n
        self.c_star = c_star
        self.cf = cf
        self.nozzle_eff = nozzle_eff
        self.combustion_eff = combustion_eff

    def run(self, n_steps: int = DEFAULT_TIME_STEPS) -> SimulationResult:
        """執行概念模擬

        Args:
            n_steps: 時間步數

        Returns:
            SimulationResult
        """
        # 計算燃燒面積曲線
        geom = GrainGeometry(self.config)
        profile = geom.compute_burn_profile(n_steps)

        # 噴嘴喉部面積
        At_mm2 = GrainGeometry.compute_nozzle_throat_area_mm2(self.config)
        At_m2 = At_mm2 * 1e-6

        # 推進劑質量
        mass_kg = GrainGeometry.compute_propellant_mass(self.config)

        # ── 計算各步的壓力與推力 ─────────────────────────────────
        Ab_mm2 = profile.burn_area_mm2.copy()
        Ab_m2 = Ab_mm2 * 1e-6

        # Kn = Ab / At
        kn = np.where(At_mm2 > 0, Ab_mm2 / At_mm2, 0.0)

        # 穩態壓力方程: P = (ρ·a·C*·Ab/At)^(1/(1-n))
        # 這裡 Ab, At 要用 m² 單位
        ratio = self.rho * self.a * self.c_star * self.combustion_eff
        pressure_pa = np.zeros(n_steps)
        for i in range(n_steps):
            if Ab_m2[i] > 0 and At_m2 > 0:
                base = ratio * Ab_m2[i] / At_m2
                if base > 0:
                    pressure_pa[i] = base ** (1.0 / (1.0 - self.n))
                else:
                    pressure_pa[i] = 0.0
            else:
                pressure_pa[i] = 0.0

        pressure_mpa = pressure_pa / 1e6

        # 推力: F = Cf · η_nozzle · P · At
        thrust_n = self.cf * self.nozzle_eff * pressure_pa * At_m2

        # 燃速: r = a · P^n (m/s)
        burn_rate = self.a * np.power(np.maximum(pressure_pa, 0), self.n)

        # 估算燃燒時間：用平均燃速積分肉厚
        web_m = profile.web_thickness_mm / 1000.0
        avg_burn_rate = np.mean(burn_rate[burn_rate > 0]) if np.any(burn_rate > 0) else 1e-3
        burn_time_s = web_m / avg_burn_rate if avg_burn_rate > 0 else 1.0

        # 時間軸
        time_s = np.linspace(0, burn_time_s, n_steps)
        dt = burn_time_s / n_steps

        # 累積衝量
        impulse_ns = np.cumsum(thrust_n) * dt
        total_impulse = impulse_ns[-1] if len(impulse_ns) > 0 else 0.0

        # 平均推力
        valid_thrust = thrust_n[thrust_n > 0]
        avg_thrust = np.mean(valid_thrust) if len(valid_thrust) > 0 else 0.0
        peak_thrust = np.max(thrust_n) if len(thrust_n) > 0 else 0.0
        peak_pressure = np.max(pressure_mpa) if len(pressure_mpa) > 0 else 0.0

        # 比衝
        g0 = 9.80665
        isp = total_impulse / (mass_kg * g0) if mass_kg > 0 else 0.0

        # ── 歸一化 ───────────────────────────────────────────────
        time_norm = profile.web_fraction.copy()

        def _normalize(arr: NDArray) -> NDArray:
            ref = arr[0] if arr[0] > 0 else np.max(arr)
            if ref > 0:
                return arr / ref
            return np.zeros_like(arr)

        area_norm = _normalize(Ab_mm2)
        kn_norm = _normalize(kn)
        press_norm = _normalize(pressure_pa)
        thrust_norm = _normalize(thrust_n)

        imp_max = impulse_ns[-1] if impulse_ns[-1] > 0 else 1.0
        impulse_norm = impulse_ns / imp_max

        return SimulationResult(
            time_normalized=time_norm,
            burn_area_normalized=area_norm,
            kn_normalized=kn_norm,
            pressure_normalized=press_norm,
            thrust_normalized=thrust_norm,
            impulse_normalized=impulse_norm,
            time_s=time_s,
            burn_area_mm2=Ab_mm2,
            kn=kn,
            pressure_pa=pressure_pa,
            pressure_mpa=pressure_mpa,
            thrust_n=thrust_n,
            impulse_ns=impulse_ns,
            total_impulse_ns=total_impulse,
            burn_time_s=burn_time_s,
            average_thrust_n=avg_thrust,
            peak_thrust_n=peak_thrust,
            peak_pressure_mpa=peak_pressure,
            specific_impulse_s=isp,
            propellant_mass_kg=mass_kg,
            burn_profile=profile,
            config=self.config,
        )
