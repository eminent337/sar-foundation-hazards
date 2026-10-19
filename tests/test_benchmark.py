"""
Tests for benchmark_evaluator module: ground-truth GPS prisms,
phase unwrapping RMSE, velocity RMSE, and Matthews Correlation Coefficient.
"""

import math
import os
import tempfile
import numpy as np
import pytest
import torch

from sar_core.benchmark_evaluator import (
    GPSPrism,
    BenchmarkMetrics,
    BenchmarkEvaluator,
    generate_benchmark_prisms,
)
from sar_core.report_generator import (
    generate_geotechnical_report,
    export_kinematic_summary_csv,
)
from sar_core.temporal_kinematics import (
    CreepPhase,
    SpatioTemporalKinematicModel,
    TARPLevel,
)


def test_gps_prism_properties():
    t_days = np.array([0, 12, 24, 36])
    disp = np.array([0.0, 1.2, 2.8, 5.0])
    p = GPSPrism(
        prism_id="P-01",
        x=25.4,
        y=40.8,
        displacement_mm=disp,
        temporal_days=t_days,
        true_velocity_mm_yr=50.0,
        true_acceleration_mm_yr2=15.0,
        elevation_m=320.0,
        is_unstable=True,
        true_creep_phase=CreepPhase.TERTIARY,
        benchmark_site="Chirano Gold Mine",
    )

    assert p.grid_x == 25
    assert p.grid_y == 40
    assert p.true_cumulative_disp_mm == 5.0
    assert p.insar_cumulative_disp_mm == 5.0
    assert p.is_failing_tertiary is True
    assert p.tarp_level == TARPLevel.ORANGE


def test_compute_phase_unwrapping_rmse():
    true_phase = np.linspace(-3.0, 3.0, 100)
    # Add constant datum shift 2.5 rad and small noise
    pred_phase = true_phase + 2.5 + np.random.normal(0, 0.05, 100)

    direct_rmse, datum_rmse, mae = BenchmarkEvaluator.compute_phase_unwrapping_rmse(
        true_phase, pred_phase, calibrate_datum=True
    )
    assert direct_rmse > 2.0
    assert datum_rmse < 0.15  # Corrected for arbitrary InSAR constant offset
    assert mae > 2.0


def test_compute_velocity_rmse():
    true_vel = np.array([10.0, 20.0, 50.0, 80.0])
    pred_vel = np.array([12.0, 19.0, 52.0, 78.0])

    rmse, mae, r2 = BenchmarkEvaluator.compute_velocity_rmse(true_vel, pred_vel)
    assert rmse < 2.5
    assert mae < 2.0
    assert r2 > 0.95


def test_compute_matthews_correlation_coefficient():
    # 4 stations: [unstable, unstable, stable, stable]
    y_true = [True, True, False, False]
    y_pred = [True, True, False, False]

    mcc, cm, prec, rec, f1, acc = BenchmarkEvaluator.compute_matthews_correlation_coefficient(y_true, y_pred)
    assert mcc == 1.0
    assert cm["TP"] == 2
    assert cm["TN"] == 2
    assert cm["FP"] == 0
    assert cm["FN"] == 0
    assert prec == 1.0
    assert rec == 1.0
    assert f1 == 1.0
    assert acc == 1.0


def test_generate_benchmark_prisms():
    prisms_chirano = generate_benchmark_prisms(site="chirano", num_prisms=12)
    assert len(prisms_chirano) == 12
    assert all(isinstance(p, GPSPrism) for p in prisms_chirano)
    assert any(p.is_unstable for p in prisms_chirano)
    assert any(not p.is_unstable for p in prisms_chirano)

    prisms_rocky = generate_benchmark_prisms(site="rocky_mountains", num_prisms=10)
    assert len(prisms_rocky) == 10
    assert any("Rocky" in p.benchmark_site for p in prisms_rocky)


def test_benchmark_evaluator_evaluate():
    prisms = generate_benchmark_prisms(site="chirano", num_prisms=8)
    metrics = BenchmarkEvaluator.evaluate_prisms(
        prisms=prisms,
        phase_metrics={"phase_rmse_rad": 0.25, "coh_weighted_phase_rmse_rad": 0.18, "phase_mae_rad": 0.12},
        site_name="Chirano Gold Mine",
    )
    assert isinstance(metrics, BenchmarkMetrics)
    assert metrics.num_prisms == 8
    assert metrics.phase_unwrapping_rmse == 0.25
    assert metrics.coherence_weighted_phase_rmse_rad == 0.18
    assert metrics.velocity_rmse >= 0.0
    assert -1.0 <= metrics.mcc <= 1.0
    table = metrics.summary_table()
    assert "GEOTECHNICAL BENCHMARK EVALUATION" in table


def test_end_to_end_chirano_evaluation():
    prisms = generate_benchmark_prisms(site="chirano", num_prisms=15, seed=42)
    model = SpatioTemporalKinematicModel(annual_period=365.25, time_unit="days", harmonics=0)

    pred_vels = []
    for p in prisms:
        state = model.fit_point(p.temporal_days, p.displacement_mm)
        pred_vels.append(state.velocity)

    metrics = BenchmarkEvaluator.evaluate_prisms(
        prisms=prisms,
        insar_velocities=pred_vels,
        site_name="Chirano Gold Mine",
    )

    assert metrics.velocity_rmse < 3.5
    assert metrics.mcc > 0.80
    assert metrics.accuracy > 0.85


def test_end_to_end_rocky_mountains_evaluation():
    prisms = generate_benchmark_prisms(site="rocky_mountains", num_prisms=15, seed=101)
    model = SpatioTemporalKinematicModel(annual_period=365.25, time_unit="days", harmonics=1)

    pred_vels = []
    for p in prisms:
        state = model.fit_point(p.temporal_days, p.displacement_mm)
        pred_vels.append(state.velocity)

    metrics = BenchmarkEvaluator.evaluate_prisms(
        prisms=prisms,
        insar_velocities=pred_vels,
        site_name="Canadian Rocky Mountains",
    )

    assert metrics.velocity_rmse < 3.5
    assert metrics.mcc > 0.70


def test_geotechnical_report_and_csv_export():
    prisms = generate_benchmark_prisms(site="chirano", num_prisms=6, seed=42)
    metrics = BenchmarkEvaluator.evaluate_prisms(prisms=prisms, site_name="Chirano Gold Mine")

    with tempfile.TemporaryDirectory() as tmpdir:
        report_path = os.path.join(tmpdir, "report.md")
        csv_path = os.path.join(tmpdir, "summary.csv")

        report = generate_geotechnical_report(metrics, prisms=prisms, output_path=report_path)
        assert "# 🛡️ GEOTECHNICAL SLOPE STABILITY CERTIFICATE" in report
        assert os.path.exists(report_path)

        export_kinematic_summary_csv(prisms, filepath=csv_path)
        assert os.path.exists(csv_path)
