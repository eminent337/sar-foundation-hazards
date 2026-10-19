"""
Tests for end-to-end CLI driver run_pipeline.py.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path
import pytest


def test_cli_pipeline_execution():
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_root = Path(__file__).resolve().parent.parent
        script = repo_root / "run_pipeline.py"

        cmd = [
            sys.executable,
            str(script),
            "--timesteps", "4",
            "--grid-size", "32",
            "--train-epochs", "1",
            "--output-dir", tmpdir,
            "--dpi", "100",
            "--seed", "42",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(repo_root))
        assert result.returncode == 0, f"run_pipeline.py failed: {result.stderr}"

        # Verify artifacts
        assert (Path(tmpdir) / "sar_publication_diagnostics.png").exists()
        assert (Path(tmpdir) / "kinematic_prism_summary.csv").exists()
        assert (Path(tmpdir) / "geotechnical_slope_stability_report.md").exists()
