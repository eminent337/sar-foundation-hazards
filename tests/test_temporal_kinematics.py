"""
Tests for temporal_kinematics module: kinematic decomposition,
smooth velocity estimation, Fukuzono-Voight inverse velocity, TARP alerts,
and SpatioTemporalKinematicModel.
"""

import math
import numpy as np
import pytest
import torch

from sar_core.temporal_kinematics import (
    CreepPhase,
    FukuzonoResult,
    KinematicState,
    SpatioTemporalKinematicModel,
    TARPLevel,
    decompose_kinematics,
    evaluate_tarp_alert,
    fit_fukuzono_voight,
    smooth_velocity_series,
)


def test_decompose_kinematics():
    t_days = np.linspace(0, 365, 24)
    # True: v = 25 mm/yr = 25/365.25 mm/day, seasonal amplitude = 5 mm
    v_true_yr = 25.0
    v_daily = v_true_yr / 365.25
    disp = v_daily * t_days + 5.0 * np.sin(2.0 * np.pi * t_days / 365.25)

    decomp = decompose_kinematics(
        time=t_days,
        displacement=disp,
        annual_period=365.25,
        harmonics=1,
        time_unit="days",
        include_acceleration=False,
    )

    assert "velocity_mm_yr" in decomp
    assert "seasonal_amplitude" in decomp
    assert "r_squared" in decomp
    assert math.isclose(decomp["velocity_mm_yr"], v_true_yr, rel_tol=0.1)
    assert math.isclose(decomp["seasonal_amplitude"], 5.0, rel_tol=0.1)
    assert decomp["r_squared"] > 0.95


def test_decompose_kinematics_coherence_weighting():
    t_days = np.linspace(0, 365, 25)
    disp = 0.05 * t_days
    # Inject large outlier in low-coherence zone
    disp_noisy = disp.copy()
    disp_noisy[10:15] += 20.0
    coherence = np.ones(25) * 0.95
    coherence[10:15] = 0.05

    res_w = decompose_kinematics(t_days, disp_noisy, coherence=coherence, harmonics=0)
    res_unw = decompose_kinematics(t_days, disp_noisy, coherence=None, harmonics=0)

    err_w = abs(res_w["velocity"] - 0.05)
    err_unw = abs(res_unw["velocity"] - 0.05)
    assert err_w < err_unw


def test_decompose_kinematics_robust_irls():
    t_days = np.linspace(0, 300, 30)
    disp = 0.04 * t_days
    disp[12] += 80.0  # Phase cycle slip

    res_rob = decompose_kinematics(t_days, disp, robust=True, harmonics=0)
    res_std = decompose_kinematics(t_days, disp, robust=False, harmonics=0)

    assert abs(res_rob["velocity"] - 0.04) < abs(res_std["velocity"] - 0.04)


def test_smooth_velocity_series():
    t = np.arange(0, 100, 10, dtype=float)
    d = 2.0 * t
    eval_t, v, a = smooth_velocity_series(t, d, window_length=3)
    assert len(v) == len(t)
    assert np.allclose(v, 2.0, atol=1e-3)


def test_fit_fukuzono_voight_accelerating():
    # Fukuzono tertiary failure curve: 1/v = A*(tf - t)
    # Failure at tf = 200 days, slope A = 0.02
    tf = 200.0
    t = np.linspace(50, 180, 15)
    A = 0.02
    v = 1.0 / (A * (tf - t))
    d = - (1.0 / A) * np.log((tf - t) / tf)

    res = fit_fukuzono_voight(
        time=t,
        displacement=d,
        min_tertiary_points=4,
    )

    assert res.is_accelerating
    assert res.r_squared > 0.8
    if res.time_of_failure is not None:
        assert abs(res.time_of_failure - tf) < 30.0


def test_fit_fukuzono_chirano_benchmark():
    # Chirano Gold Mine open pit benchmark high-wall failure at day 180
    tf_true = 180.0
    t = np.linspace(0, 160, 20)
    A = 0.015
    tertiary_d = - (1.0 / A) * np.log((tf_true - t) / tf_true) * 5.0
    disp = 0.05 * t + tertiary_d

    res = fit_fukuzono_voight(time=t, displacement=disp, alpha=2.0)
    assert res.is_accelerating is True
    assert res.is_valid_failure_prediction is True
    assert res.time_of_failure is not None
    assert abs(res.time_of_failure - tf_true) < 10.0
    assert res.r_squared > 0.85


def test_fit_fukuzono_rocky_mountains_benchmark():
    # Canadian Rocky Mountains slope failure at day 380
    tf_true = 380.0
    t = np.linspace(0, 340, 25)
    A = 0.012
    seasonal = 4.0 * np.sin(2.0 * np.pi * t / 365.25)
    tertiary_d = - (1.0 / A) * np.log((tf_true - t) / tf_true) * 4.0
    disp = 0.04 * t + seasonal + tertiary_d

    res = fit_fukuzono_voight(time=t, displacement=disp, alpha=2.0)
    assert res.is_accelerating is True
    assert res.is_valid_failure_prediction is True
    assert res.time_of_failure is not None
    assert abs(res.time_of_failure - tf_true) < 25.0


def test_fit_fukuzono_general_alpha():
    # Test Voight alpha = 1.8 exponent
    tf = 220.0
    t = np.linspace(0, 190, 20)
    alpha = 1.8
    exp = alpha - 1.0
    A = 0.01
    inv_v_exp = A * (tf - t)
    v = (1.0 / inv_v_exp) ** (1.0 / exp)

    res = fit_fukuzono_voight(time=t, velocity=v, alpha=alpha)
    assert res.is_accelerating is True
    assert res.time_of_failure is not None
    assert abs(res.time_of_failure - tf) < 15.0


def test_evaluate_tarp_alert():
    # GREEN
    assert evaluate_tarp_alert(velocity=2.0, acceleration=0.5, creep_phase=CreepPhase.STABLE) == TARPLevel.GREEN
    # YELLOW
    assert evaluate_tarp_alert(velocity=20.0, acceleration=2.0, creep_phase=CreepPhase.SECONDARY) == TARPLevel.YELLOW
    # ORANGE
    assert evaluate_tarp_alert(velocity=60.0, acceleration=35.0, creep_phase=CreepPhase.TERTIARY) == TARPLevel.ORANGE
    # RED
    assert evaluate_tarp_alert(
        velocity=150.0,
        acceleration=95.0,
        creep_phase=CreepPhase.TERTIARY,
        time_to_failure=10.0,
        failure_r2=0.90,
    ) == TARPLevel.RED


def test_spatio_temporal_kinematic_model_point():
    model = SpatioTemporalKinematicModel(
        annual_period=365.25,
        time_unit="days",
        harmonics=1,
    )
    t = np.linspace(0, 180, 16)
    d = 20.0 * (t / 365.25) + 5.0 * np.exp(0.02 * t)
    state = model.fit_point(t, d)

    assert hasattr(state, "velocity")
    assert hasattr(state, "acceleration")
    assert hasattr(state, "creep_phase")
    assert hasattr(state, "tarp_level")
    assert state.tarp_level in (TARPLevel.YELLOW, TARPLevel.ORANGE, TARPLevel.RED)


def test_spatio_temporal_kinematic_model_forward():
    model = SpatioTemporalKinematicModel(
        in_features=4,
        hidden_dim=32,
        num_layers=1,
        num_heads=2,
    )
    # Batch size 3, 10 temporal steps, 4 features
    x = torch.randn(3, 10, 4)
    out = model(x)

    assert "velocity" in out
    assert "acceleration" in out
    assert "creep_logits" in out
    assert "inv_velocity" in out
    assert out["velocity"].shape == (3,)
    assert out["creep_logits"].shape == (3, 3)
    assert out["inv_velocity"].shape == (3,)


def test_spatio_temporal_kinematic_model_tensor_stack():
    model = SpatioTemporalKinematicModel(annual_period=365.25, time_unit="days")
    t = np.linspace(0, 180, 12)
    h, w = 8, 8
    disp = np.zeros((12, h, w), dtype=np.float32)
    for i in range(12):
        disp[i, :4, :] = 0.1 * t[i]
        disp[i, 4:, :] = 0.01 * t[i]

    res = model.fit_tensor_stack(t, disp)
    assert res["velocity_map"].shape == (h, w)
    assert res["acceleration_map"].shape == (h, w)
    assert res["tarp_risk_map"].shape == (h, w)
    assert np.mean(res["velocity_map"][:4, :]) > np.mean(res["velocity_map"][4:, :])
