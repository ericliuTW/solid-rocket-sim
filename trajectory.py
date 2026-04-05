"""
固態火箭推進概念模擬平台 — 飛行軌跡概念估算模組
Trajectory / Altitude Estimation Module

根據推力曲線估算火箭的飛行高度與酬載能力。
使用簡化 1D 垂直飛行模型（含空氣阻力）。

⚠ 所有結果均為高度簡化的概念估算，不可用於真實飛行預測。
   忽略：風、地球曲率、大氣密度變化、姿態動力學、
   結構質量精確值、氣動加熱、穩定性等。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from simulation import SimulationResult


# ── 大氣與物理常數 ────────────────────────────────────────────────────
G0: float = 9.80665                    # 重力加速度 m/s²
RHO_AIR_SEA_LEVEL: float = 1.225      # 海平面空氣密度 kg/m³
SCALE_HEIGHT: float = 8500.0           # 大氣標高 m（指數模型）

# ── 示意用火箭結構參數 ────────────────────────────────────────────────
DEFAULT_DRAG_COEFFICIENT: float = 0.45  # 典型模型火箭 Cd
DEFAULT_STRUCTURAL_MASS_RATIO: float = 1.8   # 結構質量/推進劑質量（含殼體+機身+鰭+頭錐+回收）


@dataclass
class RocketConfig:
    """火箭整體設定（示意用）

    Attributes:
        body_diameter_mm: 火箭體直徑 (mm)，預設 = 殼體外徑
        drag_coefficient: 阻力係數 Cd
        structural_mass_kg: 結構質量 (kg)，不含推進劑與酬載
        payload_mass_kg: 酬載質量 (kg)
    """
    body_diameter_mm: float = 54.0
    drag_coefficient: float = DEFAULT_DRAG_COEFFICIENT
    structural_mass_kg: float = 0.0    # 0 = 自動估算
    payload_mass_kg: float = 0.0


@dataclass
class TrajectoryResult:
    """軌跡估算結果

    ⚠ 所有數值均為概念估算，不可用於真實飛行預測。

    Attributes:
        time_s: 時間陣列 (秒)
        altitude_m: 高度陣列 (米)
        velocity_m_s: 速度陣列 (m/s)
        acceleration_m_s2: 加速度陣列 (m/s²)
        max_altitude_m: 最大高度 (米)
        max_velocity_m_s: 最大速度 (m/s)
        max_acceleration_g: 最大加速度 (G)
        burnout_time_s: 燃盡時間 (秒)
        burnout_altitude_m: 燃盡高度 (米)
        burnout_velocity_m_s: 燃盡速度 (m/s)
        apogee_time_s: 到達最高點的時間 (秒)
        total_mass_kg: 起飛總質量 (kg)
        propellant_mass_kg: 推進劑質量 (kg)
        structural_mass_kg: 結構質量 (kg)
        payload_mass_kg: 酬載質量 (kg)
        thrust_to_weight_ratio: 起飛推重比
    """
    time_s: NDArray[np.float64]
    altitude_m: NDArray[np.float64]
    velocity_m_s: NDArray[np.float64]
    acceleration_m_s2: NDArray[np.float64]
    max_altitude_m: float
    max_velocity_m_s: float
    max_acceleration_g: float
    burnout_time_s: float
    burnout_altitude_m: float
    burnout_velocity_m_s: float
    apogee_time_s: float
    total_mass_kg: float
    propellant_mass_kg: float
    structural_mass_kg: float
    payload_mass_kg: float
    thrust_to_weight_ratio: float


class TrajectoryEstimator:
    """概念級 1D 垂直飛行軌跡估算器

    模型假設：
    - 純垂直飛行（1D）
    - 空氣阻力：D = 0.5 · Cd · A · ρ(h) · v²
    - 大氣密度隨高度指數衰減：ρ(h) = ρ₀ · exp(-h / H)
    - 恆定重力（忽略高度變化）
    - 推力曲線直接來自模擬結果
    - 燃盡後進入慣性滑行直到速度歸零

    ⚠ 這是高度簡化的概念模型。
    """

    def __init__(
        self,
        sim_result: SimulationResult,
        rocket_config: RocketConfig | None = None,
    ) -> None:
        self.sim = sim_result
        self.rocket = rocket_config or RocketConfig()

        # 自動設定火箭體直徑 = 殼體外徑 + 間隙
        if self.rocket.body_diameter_mm <= 0:
            self.rocket.body_diameter_mm = sim_result.config.outer_diameter_mm + 2.0

        # 自動估算結構質量
        if self.rocket.structural_mass_kg <= 0:
            self.rocket.structural_mass_kg = (
                sim_result.propellant_mass_kg * DEFAULT_STRUCTURAL_MASS_RATIO
            )

    def estimate(self, dt: float = 0.005) -> TrajectoryResult:
        """執行軌跡估算

        Args:
            dt: 時間步長 (秒)

        Returns:
            TrajectoryResult
        """
        sim = self.sim
        rocket = self.rocket

        # 質量
        m_prop = sim.propellant_mass_kg
        m_struct = rocket.structural_mass_kg
        m_payload = rocket.payload_mass_kg
        m_total_initial = m_prop + m_struct + m_payload
        m_empty = m_struct + m_payload  # 燃盡後質量

        # 參考面積
        body_r_m = rocket.body_diameter_mm / 2.0 / 1000.0
        A_ref = math.pi * body_r_m ** 2  # m²
        Cd = rocket.drag_coefficient

        # 推力曲線插值（從模擬結果取）
        burn_time = sim.burn_time_s
        n_burn_pts = len(sim.thrust_n)
        thrust_times = np.linspace(0, burn_time, n_burn_pts)
        thrust_values = sim.thrust_n.copy()

        # 模擬時間：燃燒 + 滑行（估算足夠長）
        t_max = burn_time * 50  # 足夠到達頂點再下降
        n_steps = int(t_max / dt) + 1

        # 陣列
        t_arr = np.zeros(n_steps)
        h_arr = np.zeros(n_steps)     # 高度 m
        v_arr = np.zeros(n_steps)     # 速度 m/s（正 = 上）
        a_arr = np.zeros(n_steps)     # 加速度 m/s²

        # 推重比
        F_initial = thrust_values[0] if len(thrust_values) > 0 else 0
        twr = F_initial / (m_total_initial * G0) if m_total_initial > 0 else 0

        # ── 時間步進積分 ─────────────────────────────────────────
        apogee_idx = 0

        for i in range(1, n_steps):
            t = i * dt
            t_arr[i] = t
            h = h_arr[i - 1]
            v = v_arr[i - 1]

            # 當前質量（推進劑線性消耗近似）
            if t < burn_time:
                frac_burned = t / burn_time
                m_current = m_total_initial - m_prop * frac_burned
            else:
                m_current = m_empty

            if m_current <= 0:
                m_current = m_empty

            # 推力
            if t < burn_time:
                # 插值推力
                idx_f = t / burn_time * (n_burn_pts - 1)
                idx_lo = int(idx_f)
                idx_hi = min(idx_lo + 1, n_burn_pts - 1)
                frac = idx_f - idx_lo
                F = thrust_values[idx_lo] * (1 - frac) + thrust_values[idx_hi] * frac
            else:
                F = 0.0

            # 大氣密度（指數模型）
            h_clamp = max(h, 0)
            rho = RHO_AIR_SEA_LEVEL * math.exp(-h_clamp / SCALE_HEIGHT)

            # 空氣阻力（方向與速度相反）
            D = 0.5 * Cd * A_ref * rho * v * abs(v)  # 保留方向

            # 加速度: a = (F - D) / m - g
            a = (F - D) / m_current - G0

            # 歐拉積分
            v_new = v + a * dt
            h_new = h + v * dt + 0.5 * a * dt * dt

            # 地面限制
            if h_new < 0 and t > burn_time:
                h_new = 0
                v_new = 0
                a = 0

            h_arr[i] = h_new
            v_arr[i] = v_new
            a_arr[i] = a

            # 偵測最高點（速度由正轉負）
            if v > 0 and v_new <= 0 and t > burn_time * 0.5:
                apogee_idx = i
                break  # 到達頂點即可停止

        # 截斷到有效範圍
        if apogee_idx > 0:
            end = apogee_idx + 1
        else:
            # 沒偵測到頂點，找最大高度
            end = np.argmax(h_arr) + 1
            if end < 10:
                end = n_steps
        end = min(end, n_steps)

        t_arr = t_arr[:end]
        h_arr = h_arr[:end]
        v_arr = v_arr[:end]
        a_arr = a_arr[:end]

        # 統計
        max_alt = float(np.max(h_arr))
        max_vel = float(np.max(v_arr))
        max_acc_g = float(np.max(np.abs(a_arr)) / G0)

        # 燃盡點
        burnout_idx = min(int(burn_time / dt), end - 1)
        burnout_alt = float(h_arr[burnout_idx])
        burnout_vel = float(v_arr[burnout_idx])

        apogee_time = float(t_arr[np.argmax(h_arr)])

        return TrajectoryResult(
            time_s=t_arr,
            altitude_m=h_arr,
            velocity_m_s=v_arr,
            acceleration_m_s2=a_arr,
            max_altitude_m=max_alt,
            max_velocity_m_s=max_vel,
            max_acceleration_g=max_acc_g,
            burnout_time_s=burn_time,
            burnout_altitude_m=burnout_alt,
            burnout_velocity_m_s=burnout_vel,
            apogee_time_s=apogee_time,
            total_mass_kg=m_total_initial,
            propellant_mass_kg=m_prop,
            structural_mass_kg=m_struct,
            payload_mass_kg=m_payload,
            thrust_to_weight_ratio=twr,
        )

    def payload_altitude_curve(
        self,
        payload_range_kg: NDArray[np.float64] | None = None,
        n_points: int = 15,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """計算酬載-高度曲線

        Args:
            payload_range_kg: 酬載質量陣列，None = 自動範圍
            n_points: 資料點數

        Returns:
            (payload_masses_kg, max_altitudes_m)
        """
        if payload_range_kg is None:
            # 從 0 到推進劑質量的 3 倍（通常已無法飛行）
            max_payload = self.sim.propellant_mass_kg * 3.0
            payload_range_kg = np.linspace(0, max_payload, n_points)

        altitudes = np.zeros(len(payload_range_kg))

        for i, payload in enumerate(payload_range_kg):
            self.rocket.payload_mass_kg = payload
            try:
                result = self.estimate()
                alt = result.max_altitude_m
                # 如果推重比 < 1，火箭無法起飛
                if result.thrust_to_weight_ratio < 1.0:
                    alt = 0.0
                altitudes[i] = max(0.0, alt)
            except (ValueError, ZeroDivisionError):
                altitudes[i] = 0.0

        return payload_range_kg, altitudes
