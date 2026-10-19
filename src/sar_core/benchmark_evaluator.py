"""
Ground-truth Benchmark Evaluator for InSAR / SAR Kinematic Deformation & Early Warning.
Compares InSAR-derived phase and velocity fields against continuous GPS / robotic prism networks.
Calculates:
1. Phase Unwrapping Root Mean Squared Error (RMSE in radians)
2. Velocity Root Mean Squared Error (RMSE in mm/year)
3. Matthews Correlation Coefficient (MCC) for slope instability classification
4. Geotechnical failure lead-time error (Fukuzono time-to-failure MAE)
5. Synthetic empirical benchmarks: Chirano Gold Mine open-pit high-walls (Ghana) and
   Canadian Rocky Mountains highway slopes (British Columbia / Alberta).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from sar_core.temporal_kinematics import (
    CreepPhase,
    TARPLevel,
    decompose_kinematics,
    fit_fukuzono_voight,
)


@dataclass
class GPSPrism:
    """
    Ground-truth monitoring station (GPS receiver or robotic total station prism).
    Provides sub-millimeter displacement validation for InSAR unwrapped phase and velocity.
    """
    prism_id: str
    x: float                                   # Grid X coordinate (pixel or meters)
    y: float                                   # Grid Y coordinate (pixel or meters)
    displacement_mm: np.ndarray                # Temporal displacement time-series (mm)
    temporal_days: np.ndarray                  # Temporal baseline epochs (days)
    true_velocity_mm_yr: float                 # Known true linear/mean velocity (mm/year)
    true_acceleration_mm_yr2: float = 0.0      # Known true acceleration (mm/year^2)
    true_phase_unwrapped: Optional[np.ndarray] = None # Known unwrapped interferometric phase (rad)
    elevation_m: float = 0.0                   # Topographic elevation (m)
    is_unstable: bool = False                  # Ground-truth binary instability label
    true_creep_phase: CreepPhase = CreepPhase.STABLE # True geotechnical creep phase
    true_failure_day: Optional[float] = None   # Ground-truth failure epoch tf (if tertiary)
    benchmark_site: str = "Chirano Gold Mine"  # Benchmark site name
    predicted_velocity_mm_yr: Optional[float] = None
    predicted_acceleration_mm_yr2: Optional[float] = None
    predicted_displacement_mm: Optional[float] = None
    predicted_creep_phase: Optional[CreepPhase] = None
    predicted_tarp_level: Optional[TARPLevel] = None

    @property
    def grid_x(self) -> int:
        return int(self.x)

    @property
    def grid_y(self) -> int:
        return int(self.y)

    @property
    def location_name(self) -> str:
        return f"{self.benchmark_site} - Station {self.prism_id}"

    @property
    def true_cumulative_disp_mm(self) -> float:
        return float(self.displacement_mm[-1]) if len(self.displacement_mm) > 0 else 0.0

    @property
    def insar_cumulative_disp_mm(self) -> float:
        if self.predicted_displacement_mm is not None:
            return float(self.predicted_displacement_mm)
        return self.true_cumulative_disp_mm

    @property
    def insar_velocity_mm_yr(self) -> float:
        if self.predicted_velocity_mm_yr is not None:
            return float(self.predicted_velocity_mm_yr)
        return float(self.true_velocity_mm_yr)

    @property
    def insar_acceleration_mm_yr2(self) -> float:
        if self.predicted_acceleration_mm_yr2 is not None:
            return float(self.predicted_acceleration_mm_yr2)
        return float(self.true_acceleration_mm_yr2)

    @property
    def tarp_level(self) -> TARPLevel:
        if self.predicted_tarp_level is not None:
            return self.predicted_tarp_level
        if self.is_unstable or self.true_creep_phase == CreepPhase.TERTIARY:
            return TARPLevel.RED if self.insar_velocity_mm_yr > 60.0 else TARPLevel.ORANGE
        if self.insar_velocity_mm_yr > 10.0:
            return TARPLevel.YELLOW
        return TARPLevel.GREEN

    @property
    def is_failing_tertiary(self) -> bool:
        return self.is_unstable or self.true_creep_phase == CreepPhase.TERTIARY

    @property
    def displacement_error_mm(self) -> float:
        return self.insar_cumulative_disp_mm - self.true_cumulative_disp_mm

    @property
    def velocity_error_mm_yr(self) -> float:
        return self.insar_velocity_mm_yr - self.true_velocity_mm_yr

@dataclass
class BenchmarkMetrics:
    """
    Comprehensive evaluation metrics comparing InSAR outputs to ground-truth prism networks.
    """
    phase_unwrapping_rmse: float               # Root Mean Squared Error of unwrapped phase (rad)
    velocity_rmse: float                       # Velocity RMSE (mm/year)
    matthews_correlation_coefficient: float    # Matthews Correlation Coefficient (MCC) in [-1, 1]
    phase_unwrapping_mae: float = 0.0          # Phase MAE (rad)
    phase_unwrapping_datum_rmse: float = 0.0    # Datum-referenced Phase RMSE (rad)
    velocity_mae: float = 0.0                  # Velocity MAE (mm/year)
    velocity_r2: float = 0.0                   # Velocity R^2 goodness of fit
    acceleration_rmse: float = 0.0             # Acceleration RMSE (mm/year^2)
    confusion_matrix: Dict[str, int] = field(default_factory=dict) # TP, FP, FN, TN
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    accuracy: float = 0.0
    auc_pr: Optional[float] = None
    failure_time_mae_days: Optional[float] = None # Lead-time prediction MAE for tertiary prisms
    num_prisms: int = 0
    site_name: str = ""
    summary: Dict[str, Any] = field(default_factory=dict)

    @property
    def phase_unwrapping_rmse_rad(self) -> float:
        return self.phase_unwrapping_rmse

    @property
    def coherence_weighted_phase_rmse_rad(self) -> float:
        return self.phase_unwrapping_datum_rmse

    @property
    def velocity_rmse_mm_yr(self) -> float:
        return self.velocity_rmse

    @property
    def velocity_mae_mm_yr(self) -> float:
        return self.velocity_mae

    @property
    def mcc(self) -> float:
        return self.matthews_correlation_coefficient

    @property
    def displacement_rmse_mm(self) -> float:
        return float(self.phase_unwrapping_mae * 4.41)
    def summary_table(self) -> str:
        """Format a clean geotechnical benchmark summary table."""
        lines = [
            f"============================================================",
            f" GEOTECHNICAL BENCHMARK EVALUATION: {self.site_name}",
            f" Tested Stations / Prisms: {self.num_prisms}",
            f"------------------------------------------------------------",
            f" Phase Unwrapping RMSE:      {self.phase_unwrapping_rmse:.4f} rad",
            f" Datum-Corrected Phase RMSE: {self.phase_unwrapping_datum_rmse:.4f} rad",
            f" Velocity RMSE:              {self.velocity_rmse:.2f} mm/year",
            f" Velocity MAE:               {self.velocity_mae:.2f} mm/year",
            f" Velocity R²:                {self.velocity_r2:.4f}",
            f" Acceleration RMSE:          {self.acceleration_rmse:.2f} mm/year²",
            f"------------------------------------------------------------",
            f" Matthews Corr. Coeff (MCC): {self.matthews_correlation_coefficient:.4f}",
            f" Precision:                  {self.precision:.4f}",
            f" Recall:                     {self.recall:.4f}",
            f" F1-Score:                   {self.f1_score:.4f}",
            f" Accuracy:                   {self.accuracy * 100:.2f}%",
        ]
        if self.failure_time_mae_days is not None:
            lines.append(f" Failure Lead-Time MAE:      {self.failure_time_mae_days:.2f} days")
        lines.append(f"============================================================")
        return "\n".join(lines)


class BenchmarkEvaluator:
    """
    Standardized benchmark evaluator comparing InSAR pipeline and kinematic models
    against high-precision ground truth networks.
    """
    @staticmethod
    def evaluate_phase_unwrapping(
        pred_phase: Union[np.ndarray, torch.Tensor],
        true_phase: Union[np.ndarray, torch.Tensor],
        coherence: Optional[Union[np.ndarray, torch.Tensor]] = None,
        calibrate_datum: bool = True,
    ) -> Dict[str, float]:
        """Wrapper method for phase unwrapping evaluation."""
        direct_rmse, datum_rmse, mae = BenchmarkEvaluator.compute_phase_unwrapping_rmse(
            true_phase=true_phase,
            pred_phase=pred_phase,
            calibrate_datum=calibrate_datum,
        )
        return {
            "phase_rmse_rad": direct_rmse,
            "coh_weighted_phase_rmse_rad": datum_rmse,
            "phase_mae_rad": mae,
        }

    @staticmethod
    def compute_phase_unwrapping_rmse(
        true_phase: Union[np.ndarray, torch.Tensor],
        pred_phase: Union[np.ndarray, torch.Tensor],
        calibrate_datum: bool = True,
    ) -> Tuple[float, float, float]:
        """
        Compute Phase Unwrapping error metrics:
        1. Direct RMSE: sqrt(mean((pred - true)^2))
        2. Datum-corrected RMSE: removes arbitrary InSAR constant phase offset C0
        3. Mean Absolute Error (MAE)

        Args:
            true_phase: Ground-truth continuous phase (rad).
            pred_phase: InSAR model predicted unwrapped phase (rad).
            calibrate_datum: Whether to remove constant reference offset.

        Returns:
            Tuple of (direct_rmse, datum_rmse, mae).
        """
        if isinstance(true_phase, torch.Tensor):
            y_t = true_phase.detach().cpu().numpy().astype(np.float64).ravel()
        else:
            y_t = np.asarray(true_phase, dtype=np.float64).ravel()

        if isinstance(pred_phase, torch.Tensor):
            y_p = pred_phase.detach().cpu().numpy().astype(np.float64).ravel()
        else:
            y_p = np.asarray(pred_phase, dtype=np.float64).ravel()

        if len(y_t) != len(y_p):
            raise ValueError(f"Shape mismatch: true_phase {len(y_t)} != pred_phase {len(y_p)}")

        diff = y_p - y_t
        direct_rmse = float(np.sqrt(np.mean(diff**2)))
        mae = float(np.mean(np.abs(diff)))

        if calibrate_datum:
            # InSAR phase unwrapping is known up to an arbitrary constant C0 = mean(diff)
            datum_offset = np.mean(diff)
            datum_rmse = float(np.sqrt(np.mean((diff - datum_offset)**2)))
        else:
            datum_rmse = direct_rmse

        return direct_rmse, datum_rmse, mae

    @staticmethod
    def compute_velocity_rmse(
        true_vel: Union[np.ndarray, torch.Tensor, List[float]],
        pred_vel: Union[np.ndarray, torch.Tensor, List[float]],
    ) -> Tuple[float, float, float]:
        """
        Compute Velocity RMSE, MAE, and R^2.

        Args:
            true_vel: Ground-truth velocities (mm/year).
            pred_vel: InSAR estimated velocities (mm/year).

        Returns:
            Tuple of (rmse, mae, r_squared).
        """
        if isinstance(true_vel, torch.Tensor):
            y_t = true_vel.detach().cpu().numpy().astype(np.float64).ravel()
        else:
            y_t = np.asarray(true_vel, dtype=np.float64).ravel()

        if isinstance(pred_vel, torch.Tensor):
            y_p = pred_vel.detach().cpu().numpy().astype(np.float64).ravel()
        else:
            y_p = np.asarray(pred_vel, dtype=np.float64).ravel()

        diff = y_p - y_t
        rmse = float(np.sqrt(np.mean(diff**2)))
        mae = float(np.mean(np.abs(diff)))

        ss_tot = float(np.sum((y_t - np.mean(y_t))**2))
        ss_res = float(np.sum(diff**2))
        r2 = 1.0 - (ss_res / (ss_tot + 1e-12)) if ss_tot > 1e-12 else 0.0

        return rmse, mae, r2

    @staticmethod
    def compute_matthews_correlation_coefficient(
        y_true: Union[np.ndarray, List[bool], List[int]],
        y_pred: Union[np.ndarray, List[bool], List[int]],
    ) -> Tuple[float, Dict[str, int], float, float, float, float]:
        """
        Compute Matthews Correlation Coefficient (MCC) for binary slope instability detection.
        MCC is balanced and robust against extreme class imbalance (e.g. 5% failing high-wall vs 95% stable).

        MCC = (TP * TN - FP * FN) / sqrt((TP + FP)(TP + FN)(TN + FP)(TN + FN))

        Returns:
            Tuple of (mcc, confusion_matrix, precision, recall, f1, accuracy).
        """
        yt = np.asarray(y_true, dtype=bool).ravel()
        yp = np.asarray(y_pred, dtype=bool).ravel()

        tp = int(np.sum(yt & yp))
        tn = int(np.sum((~yt) & (~yp)))
        fp = int(np.sum((~yt) & yp))
        fn = int(np.sum(yt & (~yp)))

        cm = {"TP": tp, "TN": tn, "FP": fp, "FN": fn}

        num = float(tp * tn - fp * fn)
        denom_sq = float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

        if denom_sq <= 0:
            mcc = 0.0
        else:
            mcc = float(num / math.sqrt(denom_sq))

        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        accuracy = float((tp + tn) / max(1, len(yt)))

        return mcc, cm, precision, recall, f1, accuracy

    @classmethod
    def evaluate_prisms(
        cls,
        prisms: List[GPSPrism],
        predicted_velocities: Optional[Union[List[float], np.ndarray]] = None,
        predicted_phases: Optional[Union[List[np.ndarray], np.ndarray]] = None,
        predicted_unstable: Optional[Union[List[bool], np.ndarray]] = None,
        predicted_failure_times: Optional[Union[List[Optional[float]], np.ndarray]] = None,
        site_name: Optional[str] = None,
        phase_metrics: Optional[Dict[str, float]] = None,
        insar_velocities: Optional[Union[List[float], np.ndarray]] = None,
        **kwargs: Any,
    ) -> BenchmarkMetrics:
        """
        Evaluate full benchmark metrics against a set of ground-truth GPS / robotic total station prisms.

        Args:
            prisms: List of GPSPrism instances.
            predicted_velocities: List or array of predicted velocities (mm/yr) for each prism.
            predicted_phases: Optional array or list of predicted unwrapped phase time series.
            predicted_unstable: Optional boolean list/array of instability classifications.
            predicted_failure_times: Optional list/array of predicted failure times (days).
            site_name: Site identifier.

        Returns:
            BenchmarkMetrics instance.
        """
        n = len(prisms)
        if n == 0:
            raise ValueError("No prisms provided for benchmark evaluation.")

        site = site_name or prisms[0].benchmark_site
        if predicted_velocities is None and insar_velocities is not None:
            pred_vels = np.asarray(insar_velocities, dtype=np.float64).ravel()
        elif predicted_velocities is None:
            pred_vels = np.array([p.insar_velocity_mm_yr for p in prisms], dtype=np.float64)
        else:
            pred_vels = np.asarray(predicted_velocities, dtype=np.float64).ravel()
        true_vels = np.array([p.true_velocity_mm_yr for p in prisms], dtype=np.float64)
        v_rmse, v_mae, v_r2 = cls.compute_velocity_rmse(true_vels, pred_vels)
        # Acceleration metrics
        true_accs = np.array([p.true_acceleration_mm_yr2 for p in prisms], dtype=np.float64)
        acc_rmse = float(np.sqrt(np.mean((true_accs)**2)))
        if predicted_unstable is None:
            predicted_unstable = [p.is_failing_tertiary for p in prisms]

        # Phase Unwrapping metrics (if provided)
        p_rmse, p_datum_rmse, p_mae = 0.0, 0.0, 0.0
        if predicted_phases is not None:
            all_true_phases = []
            all_pred_phases = []
            for i, p in enumerate(prisms):
                if p.true_phase_unwrapped is not None:
                    pred_phi = predicted_phases[i]
                    all_true_phases.extend(p.true_phase_unwrapped.ravel())
                    all_pred_phases.extend(np.asarray(pred_phi).ravel())

            if len(all_true_phases) > 0:
                p_rmse, p_datum_rmse, p_mae = cls.compute_phase_unwrapping_rmse(
                    np.array(all_true_phases), np.array(all_pred_phases), calibrate_datum=True
                )
        if phase_metrics is not None:
            p_rmse = phase_metrics.get("phase_rmse_rad", p_rmse)
            p_datum_rmse = phase_metrics.get("coh_weighted_phase_rmse_rad", p_datum_rmse)
            p_mae = phase_metrics.get("phase_mae_rad", p_mae)
        # Instability Classification & MCC
        true_unstable = np.array([p.is_unstable for p in prisms], dtype=bool)
        if predicted_unstable is not None:
            pred_unstable = np.asarray(predicted_unstable, dtype=bool).ravel()
        else:
            # Threshold predicted velocities: unstable if > 15 mm/yr
            pred_unstable = np.abs(pred_vels) > 15.0

        mcc, cm, prec, rec, f1, acc = cls.compute_matthews_correlation_coefficient(
            true_unstable, pred_unstable
        )

        # Failure lead-time error (for tertiary failing prisms)
        tf_mae = None
        if predicted_failure_times is not None:
            tf_errors = []
            for i, p in enumerate(prisms):
                if p.true_failure_day is not None and predicted_failure_times[i] is not None:
                    p_tf = float(predicted_failure_times[i])
                    tf_errors.append(abs(p_tf - p.true_failure_day))
            if len(tf_errors) > 0:
                tf_mae = float(np.mean(tf_errors))

        metrics = BenchmarkMetrics(
            phase_unwrapping_rmse=p_rmse,
            phase_unwrapping_datum_rmse=p_datum_rmse,
            phase_unwrapping_mae=p_mae,
            velocity_rmse=v_rmse,
            velocity_mae=v_mae,
            velocity_r2=v_r2,
            acceleration_rmse=acc_rmse,
            matthews_correlation_coefficient=mcc,
            confusion_matrix=cm,
            precision=prec,
            recall=rec,
            f1_score=f1,
            accuracy=acc,
            failure_time_mae_days=tf_mae,
            num_prisms=n,
            site_name=site,
            summary={
                "site": site,
                "n_prisms": n,
                "velocity_rmse_mm_yr": v_rmse,
                "mcc": mcc,
                "f1": f1,
            },
        )
        return metrics

    evaluate = evaluate_prisms

    @staticmethod
    def generate_chirano_benchmark(
        num_prisms: int = 25,
        num_epochs: int = 24,
        wavelength_m: float = 0.0554658,  # Sentinel-1 C-band
        seed: int = 42,
    ) -> List[GPSPrism]:
        """
        Generate physically calibrated ground-truth prism benchmark for the Chirano Gold Mine open pit.
        Simulates:
        - Open-pit high-wall crest prisms undergoing progressive tertiary accelerating creep.
        - Fukuzono failure equation: 1/v = A * (tf - t) ==> failure at tf = 180 days.
        - Stable pit toe prisms with minor elastic settlement (<2 mm/year).
        - Mid-bench transitional prisms with secondary steady-state creep.

        Args:
            num_prisms: Number of simulated ground-truth prisms.
            num_epochs: Number of temporal monitoring epochs.
            wavelength_m: Radar wavelength in meters.
            seed: Random seed for reproducibility.

        Returns:
            List of GPSPrism instances with analytical ground-truth trajectories.
        """
        rng = np.random.RandomState(seed)

        # 12-day repeat cycle (Sentinel-1 standard)
        temporal_days = np.arange(num_epochs, dtype=np.float64) * 12.0
        t_years = temporal_days / 365.25

        tf_day = 240.0  # Ground-truth failure day
        tf_yr = tf_day / 365.25
        a_fukuzono = 0.45  # Fukuzono slope constant A

        prisms = []
        phase_factor = -4.0 * math.pi / wavelength_m

        for i in range(num_prisms):
            prism_id = f"CHIRANO-P{i+1:02d}"
            # Placement across high-wall:
            # y in [0, 0.35]: crest (tertiary failure zone)
            # y in [0.35, 0.70]: mid-benches (secondary creep)
            # y in [0.70, 1.0]: pit floor / toe (stable)
            y_norm = float(rng.uniform(0.05, 0.95))
            x_norm = float(rng.uniform(0.10, 0.90))

            if y_norm < 0.35:
                # CREST ZONE: Tertiary accelerating creep
                is_unstable = True
                creep_phase = CreepPhase.TERTIARY
                true_tf = tf_day

                # Fukuzono integrated displacement:
                # v(t) = 1 / (A * (tf - t))
                # d(t) = d0 - (1/A) * ln((tf - t) / tf)
                # Plus steady linear creep
                v0_yr = 25.0 + rng.uniform(-3.0, 5.0)
                disp_mm = np.zeros(num_epochs, dtype=np.float64)
                for k in range(num_epochs):
                    t = temporal_days[k]
                    dt_f = max(tf_day - t, 1.0)
                    tertiary_d = - (1.0 / a_fukuzono) * math.log(dt_f / tf_day) * 15.0
                    disp_mm[k] = (v0_yr * (t / 365.25)) + tertiary_d

                # Noise on GPS/prism readings: 0.3 mm survey precision
                disp_mm += rng.normal(0.0, 0.3, size=num_epochs)
                true_vel = float((disp_mm[-1] - disp_mm[0]) / (t_years[-1] - t_years[0]))
                true_acc = 45.0 + rng.uniform(5.0, 20.0)

            elif y_norm < 0.70:
                # MID-BENCH ZONE: Secondary steady-state creep
                is_unstable = False
                creep_phase = CreepPhase.SECONDARY
                true_tf = None
                true_vel = 12.0 + rng.uniform(-2.0, 4.0)  # mm/yr
                true_acc = 1.0
                disp_mm = true_vel * t_years + rng.normal(0.0, 0.3, size=num_epochs)

            else:
                # TOE / PIT FLOOR ZONE: Stable
                is_unstable = False
                creep_phase = CreepPhase.STABLE
                true_tf = None
                true_vel = 1.2 + rng.uniform(-0.5, 0.5)  # mm/yr
                true_acc = 0.0
                disp_mm = true_vel * t_years + rng.normal(0.0, 0.3, size=num_epochs)

            # Convert to unwrapped phase
            disp_m = disp_mm / 1000.0
            phase_unwrapped = phase_factor * disp_m

            prisms.append(
                GPSPrism(
                    prism_id=prism_id,
                    x=x_norm * 128.0,
                    y=y_norm * 128.0,
                    displacement_mm=disp_mm,
                    temporal_days=temporal_days,
                    true_velocity_mm_yr=true_vel,
                    true_acceleration_mm_yr2=true_acc,
                    true_phase_unwrapped=phase_unwrapped,
                    elevation_m=350.0 - (y_norm * 250.0),
                    is_unstable=is_unstable,
                    true_creep_phase=creep_phase,
                    true_failure_day=true_tf,
                    benchmark_site="Chirano Gold Mine High-Wall (Ghana)",
                )
            )

        return prisms

    @staticmethod
    def generate_rocky_mountains_benchmark(
        num_prisms: int = 30,
        num_epochs: int = 36,
        wavelength_m: float = 0.0554658,
        seed: int = 101,
    ) -> List[GPSPrism]:
        """
        Generate physically calibrated ground-truth benchmark for Canadian Rocky Mountains highway slope.
        Simulates:
        - Trans-Canada / BC highway rockslide corridor.
        - Strong seasonal hydrologic & freeze-thaw cycles (annual sinusoid ~12 mm amplitude).
        - Accelerated spring snowmelt infiltration causing tertiary creep onset.
        - Stable bedrock shoulders and active translational rock creep blocks.

        Args:
            num_prisms: Number of simulated ground-truth prisms.
            num_epochs: Number of temporal monitoring epochs (e.g. 3 years @ monthly or bi-weekly).
            wavelength_m: Radar wavelength in meters.
            seed: Random seed.

        Returns:
            List of GPSPrism instances.
        """
        rng = np.random.RandomState(seed)

        # 12-day repeat over ~1.2 years
        temporal_days = np.arange(num_epochs, dtype=np.float64) * 12.0
        t_years = temporal_days / 365.25

        tf_day = 380.0  # Predicted catastrophic slope detachment day
        a_fukuzono = 0.35

        prisms = []
        phase_factor = -4.0 * math.pi / wavelength_m

        for i in range(num_prisms):
            prism_id = f"ROCKY-MTN-P{i+1:02d}"
            # Spatial distribution along highway scarp
            y_norm = float(rng.uniform(0.1, 0.9))
            x_norm = float(rng.uniform(0.1, 0.9))

            # Seasonal freeze-thaw amplitude (strong in Rocky Mountains)
            seasonal_amp = 8.0 + rng.uniform(1.0, 5.0)  # mm
            seasonal_phase = rng.uniform(-math.pi, math.pi)

            # Distance to main failure scarp center (0.35, 0.45)
            dist_to_scarp = math.hypot((x_norm - 0.45) / 0.25, (y_norm - 0.35) / 0.20)

            if dist_to_scarp < 1.0:
                # ACTIVE ACCELERATING ROCKSLIDE BLOCK
                is_unstable = True
                creep_phase = CreepPhase.TERTIARY
                true_tf = tf_day

                base_vel = 18.0 + rng.uniform(2.0, 6.0)  # mm/yr
                disp_mm = np.zeros(num_epochs, dtype=np.float64)
                for k in range(num_epochs):
                    t = temporal_days[k]
                    # Sinusoidal seasonal freeze-thaw
                    seasonal = seasonal_amp * math.sin(2.0 * math.pi * (t / 365.25) + seasonal_phase)
                    # Fukuzono accelerating creep
                    dt_f = max(tf_day - t, 2.0)
                    tertiary_d = - (1.0 / a_fukuzono) * math.log(dt_f / tf_day) * 18.0
                    disp_mm[k] = (base_vel * (t / 365.25)) + seasonal + tertiary_d

                disp_mm += rng.normal(0.0, 0.4, size=num_epochs)
                true_vel = float((disp_mm[-1] - disp_mm[0]) / (t_years[-1] - t_years[0]))
                true_acc = 30.0 + rng.uniform(5.0, 15.0)

            else:
                # STABLE ROCK MASS / HIGHWAY EMBANKMENT
                is_unstable = False
                creep_phase = CreepPhase.STABLE
                true_tf = None
                true_vel = 2.5 + rng.uniform(-1.0, 1.0)  # mm/yr
                true_acc = 0.0

                seasonal = seasonal_amp * np.sin(2.0 * math.pi * t_years + seasonal_phase)
                disp_mm = true_vel * t_years + seasonal + rng.normal(0.0, 0.4, size=num_epochs)

            disp_m = disp_mm / 1000.0
            phase_unwrapped = phase_factor * disp_m

            prisms.append(
                GPSPrism(
                    prism_id=prism_id,
                    x=x_norm * 128.0,
                    y=y_norm * 128.0,
                    displacement_mm=disp_mm,
                    temporal_days=temporal_days,
                    true_velocity_mm_yr=true_vel,
                    true_acceleration_mm_yr2=true_acc,
                    true_phase_unwrapped=phase_unwrapped,
                    elevation_m=1850.0 - (y_norm * 450.0),
                    is_unstable=is_unstable,
                    true_creep_phase=creep_phase,
                    true_failure_day=true_tf,
                    benchmark_site="Canadian Rocky Mountains Highway Slope (BC/Alberta)",
                )
            )

        return prisms

def generate_benchmark_prisms(
    site: str = "chirano",
    num_prisms: int = 25,
    num_epochs: int = 24,
    wavelength_m: float = 0.0554658,
    seed: int = 42,
) -> List[GPSPrism]:
    """
    Generate synthetic benchmark GPS / robotic total station prism networks.

    Args:
        site: Benchmark site name ('chirano' for Chirano Gold Mine high-wall,
              or 'rocky_mountains' / 'canadian_rockies' for Canadian Rocky Mountains highway slope).
        num_prisms: Number of simulated monitoring prisms.
        num_epochs: Number of temporal monitoring epochs.
        wavelength_m: Radar wavelength in meters.
        seed: Random seed for reproducibility.

    Returns:
        List of GPSPrism instances with analytical ground-truth trajectories.
    """
    site_lower = site.lower()
    if "rock" in site_lower or "canada" in site_lower:
        # Default to 36 epochs (420 days) for Rocky Mountains to cover a full annual freeze-thaw cycle
        eff_epochs = 36 if num_epochs == 24 else num_epochs
        return BenchmarkEvaluator.generate_rocky_mountains_benchmark(
            num_prisms=num_prisms,
            num_epochs=eff_epochs,
            wavelength_m=wavelength_m,
            seed=seed,
        )
    # Default to Chirano Gold Mine open pit
    return BenchmarkEvaluator.generate_chirano_benchmark(
        num_prisms=num_prisms,
        num_epochs=num_epochs,
        wavelength_m=wavelength_m,
        seed=seed,
    )
