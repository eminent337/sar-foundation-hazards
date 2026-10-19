"""
Tests for visualizer and report_generator modules.
"""

import os
import tempfile
from pathlib import Path
import pytest
import numpy as np
import torch
from PIL import Image

from sar_core.benchmark_evaluator import BenchmarkEvaluator, generate_benchmark_prisms
from sar_core.report_generator import export_kinematic_summary_csv, generate_geotechnical_report
from sar_core.sar_tensor_pipeline import SensorType, generate_highwall_simulation
from sar_core.visualizer import plot_sar_diagnostics


def test_plot_sar_diagnostics():
    with tempfile.TemporaryDirectory() as tmpdir:
        stack = generate_highwall_simulation(
            sensor=SensorType.SENTINEL_1,
            num_timesteps=4,
            grid_size=(32, 32),
            seed=42,
        )
        prisms = generate_benchmark_prisms(site="chirano", num_prisms=4)
        out_png = os.path.join(tmpdir, "test_diagnostic.png")

        fig = plot_sar_diagnostics(
            tensor_stack=stack,
            unwrapped_phase=stack.wrapped_phase,
            prisms=prisms,
            output_path=out_png,
            dpi=150,  # Fast for test
        )

        assert os.path.exists(out_png)
        assert os.path.getsize(out_png) > 10000

        with Image.open(out_png) as img:
            assert img.format == "PNG"
            assert img.size[0] > 1000
            assert img.size[1] > 800


def test_generate_geotechnical_report():
    with tempfile.TemporaryDirectory() as tmpdir:
        prisms = generate_benchmark_prisms(site="chirano", num_prisms=6)
        metrics = BenchmarkEvaluator.evaluate_prisms(prisms=prisms)
        out_md = os.path.join(tmpdir, "report.md")

        report = generate_geotechnical_report(
            metrics=metrics,
            prisms=prisms,
            site_name="Test Mine High-Wall",
            output_path=out_md,
        )

        assert os.path.exists(out_md)
        assert "GEOTECHNICAL SLOPE STABILITY CERTIFICATE" in report
        assert "Test Mine High-Wall" in report
        assert "Phase Unwrapping RMSE" in report
        assert "Trigger Action Response Plan" in report


def test_export_kinematic_summary_csv():
    with tempfile.TemporaryDirectory() as tmpdir:
        prisms = generate_benchmark_prisms(site="chirano", num_prisms=5)
        out_csv = os.path.join(tmpdir, "summary.csv")

        export_kinematic_summary_csv(
            prisms=prisms,
            output_path=out_csv,
        )

        assert os.path.exists(out_csv)
        with open(out_csv, "r", encoding="utf-8") as f:
            lines = f.readlines()
            assert len(lines) == 6  # Header + 5 prisms
            assert "Prism_ID" in lines[0]
            assert "True_Velocity_mm_yr" in lines[0]
