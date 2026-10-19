"""
SAR-Foundation-Hazards: Multi-Sensor Self-Supervised SAR/InSAR Foundation Architecture
& Kinematic Deformation Transformer for Mining High-Walls and Critical Infrastructure.
"""

from sar_core.sar_tensor_pipeline import (
    SensorType,
    SensorConfig,
    RadarTensorStack,
    DualPolComplexTensor,
    construct_dual_pol_complex_tensor,
    compute_complex_interferogram,
    phase_to_displacement_mm,
    displacement_mm_to_phase,
    compute_span,
    compute_cross_ratio,
    compute_rvi,
    wrap_phase,
    estimate_spatial_coherence,
    generate_highwall_simulation,
    SARTensorDataset,
)
from sar_core.vit_mae_foundation import (
    PatchEmbed,
    ViTMAEFoundation,
    DeepPhaseUnwrappingHead,
    ViTMAELoss,
)
from sar_core.temporal_kinematics import (
    KinematicState,
    CreepPhase,
    TARPLevel,
    SpatioTemporalKinematicModel,
    fit_fukuzono_voight,
    decompose_kinematics,
    evaluate_tarp_alert,
)
from sar_core.benchmark_evaluator import (
    GPSPrism,
    BenchmarkMetrics,
    BenchmarkEvaluator,
    generate_benchmark_prisms,
)
from sar_core.visualizer import (
    plot_sar_diagnostics,
)
from sar_core.report_generator import (
    generate_geotechnical_report,
    export_kinematic_summary_csv,
)

__version__ = "1.0.0"

__all__ = [
    "SensorType",
    "SensorConfig",
    "RadarTensorStack",
    "DualPolComplexTensor",
    "construct_dual_pol_complex_tensor",
    "compute_complex_interferogram",
    "phase_to_displacement_mm",
    "displacement_mm_to_phase",
    "compute_span",
    "compute_cross_ratio",
    "compute_rvi",
    "wrap_phase",
    "estimate_spatial_coherence",
    "generate_highwall_simulation",
    "SARTensorDataset",
    "PatchEmbed",
    "ViTMAEFoundation",
    "DeepPhaseUnwrappingHead",
    "ViTMAELoss",
    "KinematicState",
    "CreepPhase",
    "TARPLevel",
    "SpatioTemporalKinematicModel",
    "fit_fukuzono_voight",
    "decompose_kinematics",
    "evaluate_tarp_alert",
    "GPSPrism",
    "BenchmarkMetrics",
    "BenchmarkEvaluator",
    "generate_benchmark_prisms",
    "plot_sar_diagnostics",
    "generate_geotechnical_report",
    "export_kinematic_summary_csv",
]
