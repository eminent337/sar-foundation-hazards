"""
Temporal Kinematic Creep Physics & Early Warning System for InSAR / SAR.
Implements:
1. Physics-based temporal deformation tracking:
   d(t) = d0 + v * t + 0.5 * a * t^2 + seasonal + residual
2. Fukuzono (1985) & Voight (1988) inverse velocity failure criterion:
   dv/dt = A * v^alpha, with 1/v -> 0 as t -> tf for tertiary accelerating creep.
3. Geotechnical Trigger Action Response Plan (TARP) multi-level early warning alerts.
4. Spatio-temporal kinematic inversion over 1D time series and 3D radar tensor stacks.
5. Realistic empirical benchmarks: Chirano Gold Mine (open-pit high-wall) and
   Canadian Rocky Mountains (mountain highway slope).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn

class CreepPhase(str, Enum):
    """Geotechnical slope creep deformation phases."""
    STABLE = "Stable"
    PRIMARY = "Primary (Transient)"
    SECONDARY = "Secondary (Steady-State)"
    TERTIARY = "Tertiary (Accelerating)"


class TARPLevel(str, Enum):
    """Trigger Action Response Plan (TARP) alert levels."""
    GREEN = "GREEN"      # Normal background creep; operational
    YELLOW = "YELLOW"    # Watch: anomalous velocity or acceleration onset
    ORANGE = "ORANGE"    # Warning: confirmed tertiary creep, failure window weeks
    RED = "RED"          # Evacuation / Alarm: critical acceleration, failure imminent


@dataclass
class FukuzonoResult:
    """Result of Fukuzono / Voight inverse velocity failure prediction."""
    time_of_failure: Optional[float]           # Predicted failure epoch tf
    time_to_failure: Optional[float]           # Remaining time tf - t_current
    r_squared: float                           # Goodness of fit (R^2) of 1/v regression
    slope: float                               # Regression slope m = -A
    intercept: float                           # Regression intercept c = A * tf
    is_accelerating: bool                      # Whether tertiary acceleration is detected
    is_valid_failure_prediction: bool          # Valid future failure with positive confidence
    alpha: float = 2.0                         # Voight alpha exponent (2.0 = standard Fukuzono)
    tf_confidence_lower: Optional[float] = None # 95% lower confidence bound on tf
    tf_confidence_upper: Optional[float] = None # 95% upper confidence bound on tf
    smoothed_times: np.ndarray = field(default_factory=lambda: np.array([]))
    smoothed_velocities: np.ndarray = field(default_factory=lambda: np.array([]))
    inverse_velocities: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class KinematicState:
    """Complete kinematic state decomposition for a monitoring point or pixel."""
    velocity: float                            # Linear velocity (mm/year or mm/day)
    acceleration: float                        # Linear acceleration (mm/year^2 or mm/day^2)
    seasonal_amplitude: float                  # Annual sinusoidal amplitude (mm)
    seasonal_phase: float                      # Seasonal phase offset (radians)
    residual_rmse: float                       # Noise / unmodeled residual RMSE (mm)
    r_squared: float                           # Coefficient of determination R^2
    creep_phase: CreepPhase                    # Inferred geotechnical creep phase
    tarp_level: TARPLevel                      # TARP early warning alert level
    d0: float = 0.0                            # Reference datum offset (mm)
    semi_annual_amplitude: float = 0.0         # Semi-annual harmonic amplitude (mm)
    inverse_velocity: Optional[float] = None   # Current 1/v (year/mm or day/mm)
    time_of_failure: Optional[float] = None    # Predicted failure epoch tf
    time_to_failure: Optional[float] = None    # Remaining time tf - t_current
    failure_confidence_r2: Optional[float] = None # R^2 of Fukuzono regression
    alpha_exponent: float = 2.0                # Voight alpha exponent
    fitted_coefficients: Dict[str, float] = field(default_factory=dict)
    displacement_pred: Optional[np.ndarray] = None
    residuals: Optional[np.ndarray] = None


def decompose_kinematics(
    time: Union[np.ndarray, torch.Tensor],
    displacement: Union[np.ndarray, torch.Tensor],
    coherence: Optional[Union[np.ndarray, torch.Tensor]] = None,
    annual_period: float = 365.25,
    harmonics: int = 1,
    include_acceleration: bool = True,
    regularization: float = 1e-6,
    robust: bool = False,
    time_unit: str = "days",
    center_time: bool = True,
) -> Dict[str, Any]:
    """
    Decompose temporal displacement time-series into physical components:
    d(t) = d0 + v * t + 0.5 * a * t^2 + sum_{k=1}^K [S_{k,1} sin(2*pi*k*t/T) + S_{k,2} cos(2*pi*k*t/T)] + epsilon(t)

    Args:
        time: Temporal baseline epochs (1D array or tensor).
        displacement: Observed cumulative displacement series in mm (1D array or tensor).
        coherence: Interferometric coherence gamma in [0, 1] used for optimal weighting.
        annual_period: Period of primary seasonal cycle (default 365.25 for days, 1.0 for years).
        harmonics: Number of seasonal harmonics (1 = annual, 2 = annual + semi-annual).
        include_acceleration: Whether to estimate quadratic acceleration term 0.5 * a * t^2.
        regularization: Ridge parameter lambda for numerical stability.
        robust: Whether to apply Iteratively Reweighted Least Squares (Huber loss).
        time_unit: "days" or "years".

    Returns:
        Dictionary containing:
            - 'd0': datum displacement offset (mm)
            - 'velocity': linear steady-state velocity v (mm/unit_time)
            - 'velocity_mm_yr': velocity normalized to mm/year
            - 'acceleration': acceleration a (mm/unit_time^2)
            - 'acceleration_mm_yr2': acceleration normalized to mm/year^2
            - 'seasonal_amplitude': primary seasonal amplitude (mm)
            - 'seasonal_phase': primary seasonal phase (radians)
            - 'semi_annual_amplitude': semi-annual amplitude (mm, if harmonics >= 2)
            - 'fitted': modeled displacement series d_hat(t)
            - 'residuals': residual error epsilon(t) = d(t) - d_hat(t)
            - 'rmse': residual root-mean-squared error (mm)
            - 'r_squared': coefficient of determination R^2
            - 'coefficients': vector of fitted model coefficients
    """
    # Convert to NumPy float64
    if isinstance(time, torch.Tensor):
        t_arr = time.detach().cpu().numpy().astype(np.float64).ravel()
    else:
        t_arr = np.asarray(time, dtype=np.float64).ravel()

    if isinstance(displacement, torch.Tensor):
        d_arr = displacement.detach().cpu().numpy().astype(np.float64).ravel()
    else:
        d_arr = np.asarray(displacement, dtype=np.float64).ravel()

    n = len(t_arr)
    if n != len(d_arr):
        raise ValueError(f"Length mismatch: time ({n}) != displacement ({len(d_arr)})")
    if n < 3:
        raise ValueError(f"At least 3 temporal epochs required, got {n}")

    # Temporal centering for numerical stability and mean velocity decoupling
    t_ref = float(np.mean(t_arr)) if center_time else 0.0
    tau = t_arr - t_ref

    # Build design matrix G
    # Column 0: Constant offset (1.0)
    # Column 1: Centered linear time (tau)
    cols = [np.ones(n, dtype=np.float64), tau]

    # Column 2: Acceleration term 0.5 * tau^2
    if include_acceleration:
        cols.append(0.5 * (tau**2))

    # Seasonal Fourier harmonics using absolute time t_arr
    for k in range(1, harmonics + 1):
        omega_k = (2.0 * math.pi * k) / annual_period
        cols.append(np.sin(omega_k * t_arr))
        cols.append(np.cos(omega_k * t_arr))

    G = np.column_stack(cols)
    p = G.shape[1]

    # Weights from coherence: Touzi et al. InSAR phase variance weighting w = gamma^2 / (1 - gamma^2)
    if coherence is not None:
        if isinstance(coherence, torch.Tensor):
            coh_arr = coherence.detach().cpu().numpy().astype(np.float64).ravel()
        else:
            coh_arr = np.asarray(coherence, dtype=np.float64).ravel()
        coh_clamped = np.clip(coh_arr, 0.05, 0.999)
        # Optimal maximum likelihood InSAR weights
        w = (coh_clamped**2) / (1.0 - (coh_clamped**2) + 1e-4)
        # Normalize weights so sum equals n
        w = w * (n / np.sum(w))
    else:
        w = np.ones(n, dtype=np.float64)

    W = np.diag(w)

    # Inversion: Normal equations with Ridge regularization
    # (G^T W G + lambda I) m = G^T W d
    GtW = G.T @ W
    GtWG = GtW @ G + regularization * np.eye(p, dtype=np.float64)
    GtWd = GtW @ d_arr

    try:
        m = np.linalg.solve(GtWG, GtWd)
    except np.linalg.LinAlgError:
        m = np.linalg.lstsq(GtWG, GtWd, rcond=None)[0]

    # Optional Iteratively Reweighted Least Squares (IRLS) with Huber loss
    if robust:
        huber_k = 1.345
        for _ in range(5):
            res = d_arr - G @ m
            scale = 1.4826 * np.median(np.abs(res - np.median(res))) + 1e-6
            u = np.abs(res) / scale
            huber_w = np.where(u <= huber_k, 1.0, huber_k / np.maximum(u, 1e-6))
            eff_w = w * huber_w
            eff_W = np.diag(eff_w)
            GtW = G.T @ eff_W
            GtWG = GtW @ G + regularization * np.eye(p, dtype=np.float64)
            m = np.linalg.solve(GtWG, GtW @ d_arr)

    # Predictions & residuals
    fitted = G @ m
    residuals = d_arr - fitted
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((d_arr - np.mean(d_arr))**2)
    r2 = float(1.0 - ss_res / (ss_tot + 1e-12))
    rmse = float(np.sqrt(np.mean(residuals**2)))

    acc_idx = 2 if include_acceleration else None
    acceleration = float(m[acc_idx]) if acc_idx is not None else 0.0
    v_mid = float(m[1])

    if center_time and include_acceleration:
        v_0 = v_mid - acceleration * t_ref
        d0 = float(m[0]) - v_0 * t_ref - 0.5 * acceleration * (t_ref**2)
    elif center_time:
        v_0 = v_mid
        d0 = float(m[0]) - v_0 * t_ref
    else:
        v_0 = v_mid
        d0 = float(m[0])

    # Conversions to annual metrics
    yr_mult = 365.25 if time_unit.lower() == "days" else 1.0
    vel_mm_yr = v_mid * yr_mult
    acc_mm_yr2 = acceleration * (yr_mult**2)
    # Extract seasonal harmonics
    s_idx = 3 if include_acceleration else 2
    if harmonics >= 1 and s_idx + 1 < p:
        s1 = float(m[s_idx])
        c1 = float(m[s_idx + 1])
        amp1 = float(np.hypot(s1, c1))
        phase1 = float(np.arctan2(c1, s1))
    else:
        amp1, phase1 = 0.0, 0.0

    amp2 = 0.0
    if harmonics >= 2 and s_idx + 3 < p:
        s2 = float(m[s_idx + 2])
        c2 = float(m[s_idx + 3])
        amp2 = float(np.hypot(s2, c2))

    return {
        "d0": d0,
        "velocity": v_0,
        "velocity_mean": v_mid,
        "velocity_mm_yr": vel_mm_yr,
        "velocity_initial_mm_yr": v_0 * yr_mult,
        "acceleration": acceleration,
        "acceleration_mm_yr2": acc_mm_yr2,
        "seasonal_amplitude": amp1,
        "seasonal_phase": phase1,
        "semi_annual_amplitude": amp2,
        "fitted": fitted,
        "residuals": residuals,
        "rmse": rmse,
        "r_squared": r2,
        "coefficients": m,
    }


def smooth_velocity_series(
    time: np.ndarray,
    displacement: np.ndarray,
    window_length: int = 5,
    polyorder: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute smooth instantaneous velocities and accelerations from discrete displacements.
    Uses rolling local linear/polynomial regression to overcome InSAR phase noise.

    Args:
        time: 1D temporal array (days or years).
        displacement: 1D cumulative displacement array (mm).
        window_length: Local moving window length in epochs (must be >= 3).
        polyorder: Polynomial order (1 for linear velocity, 2 for quadratic).

    Returns:
        Tuple of (eval_times, velocities, accelerations).
    """
    n = len(time)
    if n < 3:
        raise ValueError(f"Need at least 3 epochs for velocity estimation, got {n}")

    # Ensure odd window length
    w_len = min(window_length, n)
    if w_len % 2 == 0:
        w_len -= 1
    w_len = max(w_len, 3)

    half_w = w_len // 2
    velocities = np.zeros(n, dtype=np.float64)
    accelerations = np.zeros(n, dtype=np.float64)

    for i in range(n):
        idx_start = max(0, i - half_w)
        idx_end = min(n, i + half_w + 1)

        t_sub = time[idx_start:idx_end]
        d_sub = displacement[idx_start:idx_end]

        if len(t_sub) >= 3 and polyorder >= 2:
            # Quadratic fit: d(t) = c0 + c1*t + c2*t^2 -> v = c1 + 2*c2*t, a = 2*c2
            coeffs = np.polyfit(t_sub - time[i], d_sub, deg=2)
            # coeffs[0] * dt^2 + coeffs[1] * dt + coeffs[2]
            velocities[i] = coeffs[1]
            accelerations[i] = 2.0 * coeffs[0]
        else:
            # Linear fit: d(t) = m*t + c -> v = m
            coeffs = np.polyfit(t_sub, d_sub, deg=1)
            velocities[i] = coeffs[0]
            # Finite difference for acceleration if available
            if i > 0:
                dt = time[i] - time[i - 1]
                accelerations[i] = (velocities[i] - velocities[i - 1]) / (dt + 1e-9)

    return time, velocities, accelerations


def fit_fukuzono_voight(
    time: Union[np.ndarray, torch.Tensor],
    displacement: Optional[Union[np.ndarray, torch.Tensor]] = None,
    velocity: Optional[Union[np.ndarray, torch.Tensor]] = None,
    alpha: float = 2.0,
    min_velocity: float = 0.05,
    min_tertiary_points: int = 4,
    smoothing_window: int = 5,
    confidence_level: float = 0.95,
) -> FukuzonoResult:
    """
    Fit the Fukuzono (1985) / Voight (1988) inverse velocity failure model:
    dv/dt = A * v^alpha

    When alpha = 2 (Fukuzono's empirical law for rock / soil slope failure):
        d(1/v)/dt = -A  ==>  1/v(t) = A * (tf - t) = -A * t + A * tf
    The linear extrapolation of inverse velocity (1/v) to 0 yields the time-of-failure (tf).

    For general alpha != 1, 2:
        (1/v)^(alpha - 1) = (alpha - 1) * A * (tf - t)

    Args:
        time: Temporal baseline epochs (days or years).
        displacement: Cumulative displacement series (mm). If provided, smooth velocities
                      are computed.
        velocity: Pre-computed velocities. Used if displacement is None.
        alpha: Voight acceleration exponent (default: 2.0).
        min_velocity: Minimum positive velocity threshold to consider for accelerating tertiary creep.
        min_tertiary_points: Minimum number of accelerating epochs required to compute tf.
        smoothing_window: Window size for velocity smoothing.
        confidence_level: Confidence level for tf bounds (default: 0.95).

    Returns:
        FukuzonoResult containing tf, time_to_failure, R^2, slope, intercept, and confidence bounds.
    """
    if isinstance(time, torch.Tensor):
        t_arr = time.detach().cpu().numpy().astype(np.float64).ravel()
    else:
        t_arr = np.asarray(time, dtype=np.float64).ravel()

    n = len(t_arr)
    if displacement is not None:
        if isinstance(displacement, torch.Tensor):
            d_arr = displacement.detach().cpu().numpy().astype(np.float64).ravel()
        else:
            d_arr = np.asarray(displacement, dtype=np.float64).ravel()
        _, v_arr, a_arr = smooth_velocity_series(t_arr, d_arr, window_length=smoothing_window)
    elif velocity is not None:
        if isinstance(velocity, torch.Tensor):
            v_arr = velocity.detach().cpu().numpy().astype(np.float64).ravel()
        else:
            v_arr = np.asarray(velocity, dtype=np.float64).ravel()
        a_arr = np.gradient(v_arr, t_arr)
    else:
        raise ValueError("Either displacement or velocity must be provided.")

    t_now = float(t_arr[-1])

    # Filter for tertiary accelerating segment
    # Conditions: positive velocity > min_velocity and positive acceleration
    # We examine the trailing window leading up to t_now
    accelerating_mask = (v_arr > min_velocity) & (a_arr > -1e-6)

    # Require that recent epochs are accelerating
    if not np.any(accelerating_mask) or np.sum(accelerating_mask) < min_tertiary_points:
        return FukuzonoResult(
            time_of_failure=None,
            time_to_failure=None,
            r_squared=0.0,
            slope=0.0,
            intercept=0.0,
            is_accelerating=False,
            is_valid_failure_prediction=False,
            alpha=alpha,
            smoothed_times=t_arr,
            smoothed_velocities=v_arr,
            inverse_velocities=np.where(v_arr > 1e-4, 1.0 / np.maximum(v_arr, 1e-6), np.nan),
        )

    # Extract the accelerating sequence towards current time
    # Scan backward from end to find continuous accelerating window
    run_len = 0
    for idx in range(n - 1, -1, -1):
        if accelerating_mask[idx]:
            run_len += 1
        else:
            # Allow at most 1 transient noise fluctuation if run_len >= min_tertiary_points
            if run_len >= min_tertiary_points:
                break
            run_len = 0

    if run_len < min_tertiary_points:
        # Fall back to all points where v > min_velocity and a >= 0 in the second half of series
        half_n = max(0, n - max(min_tertiary_points + 2, n // 2))
        sub_mask = np.zeros(n, dtype=bool)
        sub_mask[half_n:] = accelerating_mask[half_n:]
        if np.sum(sub_mask) < min_tertiary_points:
            # Just take the last min_tertiary_points points if velocities are monotonically rising
            sub_mask = np.zeros(n, dtype=bool)
            sub_mask[-min_tertiary_points:] = True
    else:
        sub_mask = np.zeros(n, dtype=bool)
        sub_mask[(n - run_len):n] = True

    t_fit = t_arr[sub_mask]
    v_fit = v_arr[sub_mask]

    # Compute generalized inverse velocity:
    # If alpha == 2: y = 1 / v
    # If alpha != 1, 2: y = (1 / v)^(alpha - 1)
    if math.isclose(alpha, 2.0, abs_tol=1e-3):
        y_fit = 1.0 / np.maximum(v_fit, 1e-8)
    else:
        exponent = alpha - 1.0
        y_fit = (1.0 / np.maximum(v_fit, 1e-8)) ** exponent

    # Linear regression: y = m * t + c
    poly, cov = np.polyfit(t_fit, y_fit, deg=1, cov=True)
    slope = float(poly[0])      # m
    intercept = float(poly[1])  # c

    # Goodness of fit (R^2)
    y_pred = slope * t_fit + intercept
    ss_res = float(np.sum((y_fit - y_pred)**2))
    ss_tot = float(np.sum((y_fit - np.mean(y_fit))**2))
    r2 = 1.0 - (ss_res / (ss_tot + 1e-12)) if ss_tot > 1e-12 else 0.0

    # Failure criterion: 1/v -> 0 as t -> tf
    # m * tf + c = 0 ==> tf = -c / m
    is_accelerating = (slope < -1e-8)
    tf = None
    time_to_failure = None
    tf_lower = None
    tf_upper = None
    is_valid = False

    if is_accelerating:
        tf_val = -intercept / slope
        # Only valid if failure is predicted in the future or within small margin
        if tf_val >= t_now - 0.5:
            tf = float(tf_val)
            time_to_failure = float(max(0.0, tf - t_now))
            is_valid = (r2 >= 0.50)

            # Confidence interval for tf via Delta method:
            # var(tf) = (1/m^2) * var(c) + (c^2 / m^4) * var(m) - 2*(c / m^3) * cov(m, c)
            try:
                var_m = cov[0, 0]
                var_c = cov[1, 1]
                cov_mc = cov[0, 1]
                var_tf = (1.0 / (slope**2)) * var_c + ((intercept**2) / (slope**4)) * var_m - 2.0 * (intercept / (slope**3)) * cov_mc
                if var_tf > 0:
                    se_tf = math.sqrt(var_tf)
                    crit = stats.t.ppf((1.0 + confidence_level) / 2.0, df=max(1, len(t_fit) - 2))
                    tf_lower = float(max(t_now, tf - crit * se_tf))
                    tf_upper = float(tf + crit * se_tf)
            except Exception:
                pass

    inv_v_all = np.where(v_arr > 1e-6, 1.0 / np.maximum(v_arr, 1e-6), np.nan)

    return FukuzonoResult(
        time_of_failure=tf,
        time_to_failure=time_to_failure,
        r_squared=float(np.clip(r2, 0.0, 1.0)),
        slope=slope,
        intercept=intercept,
        is_accelerating=is_accelerating,
        is_valid_failure_prediction=is_valid,
        alpha=alpha,
        tf_confidence_lower=tf_lower,
        tf_confidence_upper=tf_upper,
        smoothed_times=t_arr,
        smoothed_velocities=v_arr,
        inverse_velocities=inv_v_all,
    )


def evaluate_tarp_alert(
    velocity: float,
    acceleration: float,
    creep_phase: CreepPhase,
    time_to_failure: Optional[float] = None,
    failure_r2: Optional[float] = None,
    thresholds: Optional[Dict[str, float]] = None,
) -> TARPLevel:
    """
    Evaluate Geotechnical Trigger Action Response Plan (TARP) alert level:
    - GREEN: Normal background operations. Stable or steady creep below threshold.
    - YELLOW: Watch status. Elevated velocity, initiation of acceleration.
    - ORANGE: Warning status. Confirmed tertiary creep with intermediate failure window.
    - RED: Evacuation / Alarm status. Imminent catastrophic collapse predicted.

    Args:
        velocity: InSAR line-of-sight or true velocity (mm/year).
        acceleration: InSAR acceleration (mm/year^2).
        creep_phase: CreepPhase classification enum.
        time_to_failure: Remaining time to predicted failure (days).
        failure_r2: Goodness of fit (R^2) for Fukuzono inverse velocity model.
        thresholds: Custom TARP operational thresholds dictionary.

    Returns:
        TARPLevel (GREEN, YELLOW, ORANGE, or RED).
    """
    default_thresh = {
        "v_yellow": 15.0,        # mm/yr
        "v_orange": 50.0,        # mm/yr
        "v_red": 120.0,          # mm/yr
        "a_yellow": 10.0,        # mm/yr^2
        "a_orange": 30.0,        # mm/yr^2
        "a_red": 80.0,           # mm/yr^2
        "ttf_red_days": 14.0,    # Failure within 14 days
        "ttf_orange_days": 60.0, # Failure within 60 days
        "min_r2_alert": 0.70,    # Minimum R^2 for inverse velocity alerts
    }
    if thresholds:
        default_thresh.update(thresholds)
    th = default_thresh

    # Check for RED alert:
    # 1. Valid tertiary creep with failure within critical lead time (e.g. <= 14 days) and high R^2
    if (
        creep_phase == CreepPhase.TERTIARY
        and time_to_failure is not None
        and time_to_failure <= th["ttf_red_days"]
        and (failure_r2 is None or failure_r2 >= th["min_r2_alert"])
    ):
        return TARPLevel.RED

    # 2. Extreme velocity and critical acceleration
    if abs(velocity) >= th["v_red"] and acceleration >= th["a_red"]:
        return TARPLevel.RED

    # Check for ORANGE alert:
    # 1. Tertiary creep with failure predicted within intermediate window (e.g. <= 60 days)
    if (
        creep_phase == CreepPhase.TERTIARY
        and time_to_failure is not None
        and time_to_failure <= th["ttf_orange_days"]
        and (failure_r2 is None or failure_r2 >= 0.60)
    ):
        return TARPLevel.ORANGE

    # 2. High velocity or high acceleration
    if abs(velocity) >= th["v_orange"] and acceleration >= th["a_orange"]:
        return TARPLevel.ORANGE

    # Check for YELLOW alert:
    # 1. Onset of acceleration or primary transient creep
    if (
        abs(velocity) >= th["v_yellow"]
        or acceleration >= th["a_yellow"]
        or creep_phase == CreepPhase.PRIMARY
        or (creep_phase == CreepPhase.TERTIARY and (time_to_failure is None or time_to_failure > th["ttf_orange_days"]))
    ):
        return TARPLevel.YELLOW

    # Otherwise GREEN
    return TARPLevel.GREEN


class SpatioTemporalKinematicModel(nn.Module):
    """
    Spatio-Temporal Kinematic Deformation & Early Warning Model.
    Supports both:
    1. Deep sequence Transformer / Bi-LSTM forward inference on (B, T, C) tensors.
    2. Analytical physics least-squares decomposition and Fukuzono-Voight forecasting.
    """

    def __init__(
        self,
        in_features: int = 4,
        hidden_dim: int = 64,
        num_layers: int = 1,
        num_heads: int = 2,
        num_classes: int = 3,
        dropout: float = 0.1,
        annual_period: float = 365.25,
        time_unit: str = "days",
        harmonics: int = 1,
        alpha_voight: float = 2.0,
        tarp_thresholds: Optional[Dict[str, float]] = None,
        **kwargs: Any,
    ):
        super().__init__()
        self.annual_period = annual_period
        self.time_unit = time_unit
        self.harmonics = harmonics
        self.alpha_voight = alpha_voight
        self.tarp_thresholds = tarp_thresholds

        # Neural sequence modeling parameters
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.input_proj = nn.Linear(in_features, hidden_dim)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.kinematic_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )
        self.classifier_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, num_classes),
        )
        self.inv_velocity_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Softplus(),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass for deep sequence modeling on displacement/coherence sequences.
        Shape: (B, T, in_features) -> kinematics & failure logits.
        """
        B, T, C = x.shape
        h = self.input_proj(x)
        lstm_out, _ = self.lstm(h)
        attn_out, _ = self.temporal_attn(lstm_out, lstm_out, lstm_out)
        feat = self.norm(lstm_out + attn_out)
        last_feat = feat[:, -1, :]
        kin = self.kinematic_head(last_feat)
        pred_velocity = kin[:, 0]
        pred_acceleration = kin[:, 1]
        creep_logits = self.classifier_head(last_feat)
        inv_v = self.inv_velocity_head(last_feat).squeeze(-1)
        return {
            "velocity": pred_velocity,
            "acceleration": pred_acceleration,
            "creep_logits": creep_logits,
            "inv_velocity": inv_v,
            "sequence_embeddings": feat,
        }
    def fit_point(
        self,
        time: Union[np.ndarray, torch.Tensor],
        displacement: Union[np.ndarray, torch.Tensor],
        coherence: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> KinematicState:
        """
        Perform complete kinematic decomposition and failure forecasting on a single point/prism.
        Args:
            time: 1D time epochs (days).
            displacement: 1D displacement series (mm).
            coherence: Optional 1D interferometric coherence gamma.

        Returns:
            KinematicState object.
        """
        if isinstance(time, torch.Tensor):
            t_arr = time.detach().cpu().numpy().astype(np.float64)
        else:
            t_arr = np.asarray(time, dtype=np.float64)

        if isinstance(displacement, torch.Tensor):
            d_arr = displacement.detach().cpu().numpy().astype(np.float64)
        else:
            d_arr = np.asarray(displacement, dtype=np.float64)

        # Guard against seasonal collinearity if time span is shorter than seasonal period
        t_span = float(t_arr[-1] - t_arr[0])
        effective_harmonics = self.harmonics if t_span >= 0.90 * self.annual_period else 0

        # 1. Physics-based linear decomposition for uncorrupted velocity
        decomp = decompose_kinematics(
            time=t_arr,
            displacement=d_arr,
            coherence=coherence,
            annual_period=self.annual_period,
            harmonics=effective_harmonics,
            time_unit=self.time_unit,
            include_acceleration=False,
        )
        v_yr = decomp["velocity_mm_yr"]
        yr_mult = 365.25 if self.time_unit.lower() == "days" else 1.0
        chord_vel = float((d_arr[-1] - d_arr[0]) / (t_span / yr_mult)) if t_span > 0 else v_yr
        if abs(chord_vel) > 20.0 or abs(v_yr) > 20.0:
            v_yr = chord_vel
        decomp_acc = decompose_kinematics(
            time=t_arr,
            displacement=d_arr,
            coherence=coherence,
            annual_period=self.annual_period,
            harmonics=effective_harmonics,
            time_unit=self.time_unit,
            include_acceleration=True,
        )
        a_yr2 = decomp_acc["acceleration_mm_yr2"]
        # 2. Fukuzono / Voight failure criterion fit
        fukuzono = fit_fukuzono_voight(
            time=t_arr,
            displacement=d_arr,
            alpha=self.alpha_voight,
            smoothing_window=5,
        )

        # 3. Classify geotechnical creep phase
        # Primary: decelerating after load disturbance (a < -10)
        # Secondary: steady state (|a| < 10)
        # Tertiary: sustained positive acceleration (a > 10, v accelerating)
        if fukuzono.is_accelerating and fukuzono.is_valid_failure_prediction:
            creep_phase = CreepPhase.TERTIARY
        elif a_yr2 > 15.0 and abs(v_yr) > 10.0:
            creep_phase = CreepPhase.TERTIARY
        elif a_yr2 < -15.0 and abs(v_yr) > 8.0:
            creep_phase = CreepPhase.PRIMARY
        elif abs(v_yr) > 5.0:
            creep_phase = CreepPhase.SECONDARY
        else:
            creep_phase = CreepPhase.STABLE

        # 4. Evaluate TARP Alert Level
        tarp = evaluate_tarp_alert(
            velocity=v_yr,
            acceleration=a_yr2,
            creep_phase=creep_phase,
            time_to_failure=fukuzono.time_to_failure,
            failure_r2=fukuzono.r_squared,
            thresholds=self.tarp_thresholds,
        )

        curr_v = float(fukuzono.smoothed_velocities[-1]) if len(fukuzono.smoothed_velocities) > 0 else decomp["velocity"]
        inv_v = (1.0 / curr_v) if abs(curr_v) > 1e-6 else None

        return KinematicState(
            velocity=v_yr,
            acceleration=a_yr2,
            seasonal_amplitude=decomp["seasonal_amplitude"],
            seasonal_phase=decomp["seasonal_phase"],
            residual_rmse=decomp["rmse"],
            r_squared=decomp["r_squared"],
            creep_phase=creep_phase,
            tarp_level=tarp,
            d0=decomp["d0"],
            semi_annual_amplitude=decomp["semi_annual_amplitude"],
            inverse_velocity=inv_v,
            time_of_failure=fukuzono.time_of_failure,
            time_to_failure=fukuzono.time_to_failure,
            failure_confidence_r2=fukuzono.r_squared,
            alpha_exponent=self.alpha_voight,
            fitted_coefficients={"d0": decomp["d0"], "v": decomp["velocity"], "a": decomp["acceleration"]},
            displacement_pred=decomp["fitted"],
            residuals=decomp["residuals"],
        )

    def fit_tensor_stack(
        self,
        time: Union[np.ndarray, torch.Tensor],
        displacement_stack: Union[np.ndarray, torch.Tensor],
        coherence_stack: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
        """
        Vectorized spatio-temporal kinematic decomposition over full (T, H, W) radar tensor stack.

        Args:
            time: Temporal baselines (T,) in days.
            displacement_stack: Displacement tensor of shape (T, H, W) in mm.
            coherence_stack: Spatial coherence tensor of shape (T, H, W) in [0, 1].

        Returns:
            Dictionary containing spatial maps:
                - 'velocity_map': (H, W) in mm/year
                - 'acceleration_map': (H, W) in mm/year^2
                - 'seasonal_amplitude_map': (H, W) in mm
                - 'residual_rmse_map': (H, W) in mm
                - 'r_squared_map': (H, W)
                - 'creep_phase_map': (H, W) string array
                - 'time_to_failure_map': (H, W) float array (nan for stable pixels)
                - 'tarp_risk_map': (H, W) string array (GREEN, YELLOW, ORANGE, RED)
        """
        # Convert to Torch tensor for GPU/CPU vectorized least squares
        if isinstance(time, np.ndarray):
            t_ten = torch.from_numpy(time).float()
        else:
            t_ten = time.float()

        if isinstance(displacement_stack, np.ndarray):
            disp_ten = torch.from_numpy(displacement_stack).float()
        else:
            disp_ten = displacement_stack.float()

        num_t, h, w = disp_ten.shape
        t_arr = t_ten.cpu().numpy().astype(np.float64)

        # Build design matrix G of shape (T, P) with centered time for numerical stability
        t_ref = t_ten.mean()
        tau = t_ten - t_ref
        omega = (2.0 * math.pi) / self.annual_period
        g_cols = [
            torch.ones(num_t, device=disp_ten.device),
            tau,
            0.5 * (tau**2),
            torch.sin(omega * t_ten),
            torch.cos(omega * t_ten),
        ]
        G = torch.stack(g_cols, dim=1)  # (T, 5)

        # Reshape displacement to (T, H*W)
        d_flat = disp_ten.reshape(num_t, h * w)

        # Solve linear system via torch.linalg.lstsq
        # G * M = d_flat ==> M has shape (5, H*W)
        sol = torch.linalg.lstsq(G, d_flat).solution

        yr_mult = 365.25 if self.time_unit == "days" else 1.0
        v_map = (sol[1] * yr_mult).reshape(h, w).cpu().numpy()
        a_map = (sol[2] * (yr_mult**2)).reshape(h, w).cpu().numpy()
        s_sin = sol[3].reshape(h, w).cpu().numpy()
        s_cos = sol[4].reshape(h, w).cpu().numpy()
        amp_map = np.hypot(s_sin, s_cos)

        # Residuals
        pred_flat = G @ sol
        res_flat = d_flat - pred_flat
        rmse_map = torch.sqrt(torch.mean(res_flat**2, dim=0)).reshape(h, w).cpu().numpy()

        ss_tot = torch.sum((d_flat - d_flat.mean(dim=0, keepdim=True))**2, dim=0)
        ss_res = torch.sum(res_flat**2, dim=0)
        r2_flat = 1.0 - (ss_res / (ss_tot + 1e-7))
        r2_map = r2_flat.reshape(h, w).cpu().numpy()

        # Spatial failure forecasting & TARP risk mapping
        creep_phase_map = np.full((h, w), CreepPhase.STABLE.value, dtype=object)
        ttf_map = np.full((h, w), np.nan, dtype=np.float64)
        tarp_map = np.full((h, w), TARPLevel.GREEN.value, dtype=object)

        disp_np = disp_ten.cpu().numpy()

        # Perform Fukuzono evaluation on high-risk pixels (v > 10 mm/yr and a > 10 mm/yr^2)
        candidate_y, candidate_x = np.where((v_map > 10.0) & (a_map > 10.0))
        for y_idx, x_idx in zip(candidate_y, candidate_x):
            d_series = disp_np[:, y_idx, x_idx]
            fuk = fit_fukuzono_voight(
                time=t_arr,
                displacement=d_series,
                alpha=self.alpha_voight,
            )
            phase = CreepPhase.TERTIARY if (fuk.is_accelerating and fuk.is_valid_failure_prediction) else CreepPhase.SECONDARY
            creep_phase_map[y_idx, x_idx] = phase.value
            if fuk.time_to_failure is not None:
                ttf_map[y_idx, x_idx] = fuk.time_to_failure

            tarp = evaluate_tarp_alert(
                velocity=v_map[y_idx, x_idx],
                acceleration=a_map[y_idx, x_idx],
                creep_phase=phase,
                time_to_failure=fuk.time_to_failure,
                failure_r2=fuk.r_squared,
                thresholds=self.tarp_thresholds,
            )
            tarp_map[y_idx, x_idx] = tarp.value

        # Update remaining non-candidate pixels
        for y in range(h):
            for x in range(w):
                if creep_phase_map[y, x] == CreepPhase.STABLE.value:
                    v_val = v_map[y, x]
                    a_val = a_map[y, x]
                    if a_val < -15.0 and abs(v_val) > 8.0:
                        phase = CreepPhase.PRIMARY
                    elif abs(v_val) > 5.0:
                        phase = CreepPhase.SECONDARY
                    else:
                        phase = CreepPhase.STABLE
                    creep_phase_map[y, x] = phase.value

                    tarp = evaluate_tarp_alert(
                        velocity=v_val,
                        acceleration=a_val,
                        creep_phase=phase,
                        thresholds=self.tarp_thresholds,
                    )
                    tarp_map[y, x] = tarp.value

        return {
            "velocity_map": v_map,
            "acceleration_map": a_map,
            "seasonal_amplitude_map": amp_map,
            "residual_rmse_map": rmse_map,
            "r_squared_map": r2_map,
            "creep_phase_map": creep_phase_map,
            "time_to_failure_map": ttf_map,
            "tarp_risk_map": tarp_map,
        }
