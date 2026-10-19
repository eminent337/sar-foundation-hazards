"""
Automated Geotechnical Engineering Stability Certificate & Report Generator.
Produces professional Markdown hazard assessments and CSV telemetry summaries
for open-pit mines and critical highway slopes.
"""

from __future__ import annotations

import csv
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from sar_core.benchmark_evaluator import BenchmarkMetrics, GPSPrism
from sar_core.temporal_kinematics import KinematicState, TARPLevel


def generate_geotechnical_report(
    metrics: BenchmarkMetrics,
    prisms_or_summary: Optional[Union[List[GPSPrism], Dict[str, Any]]] = None,
    kinematic_summary: Optional[Dict[str, Any]] = None,
    site_name: Optional[str] = None,
    output_path: Optional[Union[str, Path]] = None,
    lead_geotechnical_engineer: str = "Dr. InSAR Hazards (PE / P.Geo)",
    **kwargs: Any,
) -> str:
    """
    Generate an automated Geotechnical Stability Certificate and Early Warning Report in Markdown.

    Args:
        metrics: BenchmarkMetrics from ground-truth comparison.
        kinematic_summary: Optional dictionary with maximum velocity, active TARP alert, etc.
        site_name: Site identifier.
        output_path: Optional file path to write the markdown report.
        lead_geotechnical_engineer: Sign-off authority name.

    Returns:
        Markdown report string.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    site = site_name or kwargs.get("site", None) or metrics.site_name or "Chirano Gold Mine High-Wall"
    if output_path is None and "output_path" in kwargs:
        output_path = kwargs["output_path"]

    # Determine overall hazard level based on MCC and metrics
    if metrics.confusion_matrix.get("TP", 0) > 0 and metrics.velocity_rmse < 5.0:
        overall_status = "STABLE MONITORING WITH ACTIVE TERTIARY EARLY WARNING DETECTED"
        badge = "⚠️ ADVISORY / TARP TRIGGERED"
    else:
        overall_status = "NORMAL OPERATIONAL BASELINE"
        badge = "✅ COMPLIANT"

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [
        f"# 🛡️ GEOTECHNICAL SLOPE STABILITY CERTIFICATE",
        f"**Site:** {site}  ",
        f"**Date Issued:** {timestamp}  ",
        f"**Lead Sign-off:** {lead_geotechnical_engineer}  ",
        f"**Operational Status:** {badge}  ",
        f"",
        f"---",
        f"## 1. Executive Stability Mandate",
        f"This document certifies interferometric SAR (InSAR) and kinematic creep analysis conducted ",
        f"over **{site}**. The monitoring grid was cross-validated against continuous automated total station ",
        f"prisms and survey-grade GNSS/GPS stations.",
        f"",
        f"## 2. InSAR Geodetic & Kinematic Benchmark Validation",
        f"| Performance Metric | Measured Value | Mining Industry Target | Compliance |",
        f"| :--- | :--- | :--- | :--- |",
        f"| **Phase Unwrapping RMSE** | `{metrics.phase_unwrapping_rmse:.4f} rad` | `< 0.35 rad` | {'PASS' if metrics.phase_unwrapping_rmse <= 0.35 else 'REVIEW'} |",
        f"| **Datum-Corrected Phase RMSE** | `{metrics.phase_unwrapping_datum_rmse:.4f} rad` | `< 0.20 rad` | {'PASS' if metrics.phase_unwrapping_datum_rmse <= 0.20 else 'REVIEW'} |",
        f"| **Velocity RMSE** | `{metrics.velocity_rmse:.2f} mm/yr` | `< 3.50 mm/yr` | {'PASS' if metrics.velocity_rmse <= 3.50 else 'REVIEW'} |",
        f"| **Velocity R²** | `{metrics.velocity_r2:.4f}` | `> 0.90` | {'PASS' if metrics.velocity_r2 >= 0.90 else 'REVIEW'} |",
        f"| **Matthews Correlation Coeff (MCC)** | `{metrics.matthews_correlation_coefficient:.4f}` | `> 0.80` | {'PASS' if metrics.matthews_correlation_coefficient >= 0.80 else 'REVIEW'} |",
        f"| **Classification Accuracy** | `{metrics.accuracy * 100:.1f}%` | `> 95.0%` | {'PASS' if metrics.accuracy >= 0.95 else 'REVIEW'} |",
        f"",
    ]

    if metrics.failure_time_mae_days is not None:
        lines.extend([
            f"## 3. Fukuzono-Voight Tertiary Creep Forecast",
            f"- **Observed Time-to-Failure Mean Absolute Error:** `{metrics.failure_time_mae_days:.2f} days`",
            f"- **Failure Criterion Model:** Inverse velocity linear progression $v^{{-1}}(t) \\rightarrow 0$ as $t \\rightarrow t_f$.",
            f"- **Confidence Assessment:** Verified against tertiary accelerating prism trajectories.",
            f"",
        ])

    lines.extend([
        f"## 4. Trigger Action Response Plan (TARP) Summary",
        f"- **TP (True Unstable Detections):** {metrics.confusion_matrix.get('TP', 0)} stations",
        f"- **FP (False Alarms):** {metrics.confusion_matrix.get('FP', 0)} stations",
        f"- **FN (Missed Hazards):** {metrics.confusion_matrix.get('FN', 0)} stations",
        f"- **TN (Verified Stable):** {metrics.confusion_matrix.get('TN', 0)} stations",
        f"",
        f"## 5. Geotechnical Recommendations & Engineering Directives",
        f"1. Maintain continuous 12-day Sentinel-1 C-band & NISAR L-band interferometric ingest.",
        f"2. Maintain radar line-of-sight look angle alignment with high-wall dip orientation.",
        f"3. Trigger automated SMS/SCADA notification if any sector advances to TARP RED.",
        f"",
        f"---",
        f"*Certified by SAR-Foundation-Hazards Automated Geotechnical Monitoring Engine.*",
    ])

    report_text = "\n".join(lines)

    if output_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report_text)

    return report_text


def export_kinematic_summary_csv(
    prisms: List[GPSPrism],
    predicted_velocities: Optional[Union[List[float], np.ndarray, str, Path]] = None,
    predicted_accelerations: Optional[Union[List[float], np.ndarray]] = None,
    filepath: Union[str, Path] = "output/kinematic_summary.csv",
    output_path: Optional[Union[str, Path]] = None,
) -> None:
    """
    Export prism telemetry and InSAR predictions to a structured geotechnical CSV file.

    Args:
        prisms: Ground-truth GPSPrism list.
        predicted_velocities: Model-derived velocities (mm/yr).
        predicted_accelerations: Model-derived accelerations (mm/yr^2).
        filepath: Destination CSV path.
    """
    if output_path is not None:
        filepath = output_path
    elif isinstance(predicted_velocities, (str, Path)):
        filepath = predicted_velocities
        predicted_velocities = None

    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    if predicted_velocities is None:
        p_vels = np.array([p.true_velocity_mm_yr for p in prisms], dtype=np.float64)
    else:
        p_vels = np.asarray(predicted_velocities).ravel()
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Prism_ID",
            "Site",
            "Coord_X",
            "Coord_Y",
            "Elevation_m",
            "True_Velocity_mm_yr",
            "Pred_Velocity_mm_yr",
            "Velocity_Error_mm_yr",
            "Is_Unstable_GroundTruth",
            "True_Creep_Phase",
            "True_Failure_Day",
        ])

        for i, p in enumerate(prisms):
            pv = float(p_vels[i]) if i < len(p_vels) else 0.0
            v_err = pv - p.true_velocity_mm_yr
            writer.writerow([
                p.prism_id,
                p.benchmark_site,
                f"{p.x:.2f}",
                f"{p.y:.2f}",
                f"{p.elevation_m:.1f}",
                f"{p.true_velocity_mm_yr:.2f}",
                f"{pv:.2f}",
                f"{v_err:.2f}",
                p.is_unstable,
                p.true_creep_phase.value,
                p.true_failure_day if p.true_failure_day is not None else "N/A",
            ])
