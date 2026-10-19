"""
Multi-sensor radar tensor pipeline for complex SAR / InSAR processing.
Supports Sentinel-1 (C-band) and NISAR (L-band) ingest, polarimetric decomposition,
interferometric phase wrapping, and spatial-temporal coherence estimation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class SensorType(str, Enum):
    """Supported SAR sensor types."""
    SENTINEL_1 = "Sentinel-1"
    NISAR = "NISAR"


@dataclass(frozen=True)
class SensorConfig:
    """Radar sensor configuration and physical constants."""
    name: SensorType
    band: str
    frequency_ghz: float
    wavelength_m: float
    polarizations: Tuple[str, ...]
    range_resolution_m: float
    azimuth_resolution_m: float
    repeat_cycle_days: float

    @classmethod
    def sentinel_1_c_band(cls) -> "SensorConfig":
        """Sentinel-1 C-band operational mode (5.405 GHz, ~5.55 cm wavelength)."""
        freq = 5.405e9
        c = 299792458.0
        return cls(
            name=SensorType.SENTINEL_1,
            band="C",
            frequency_ghz=5.405,
            wavelength_m=c / freq,  # ~0.0554658 m
            polarizations=("VV", "VH"),
            range_resolution_m=5.0,
            azimuth_resolution_m=20.0,
            repeat_cycle_days=12.0,
        )

    @classmethod
    def nisar_l_band(cls) -> "SensorConfig":
        """NASA-ISRO SAR (NISAR) L-band mode (1.257 GHz, ~23.85 cm wavelength)."""
        freq = 1.257e9
        c = 299792458.0
        return cls(
            name=SensorType.NISAR,
            band="L",
            frequency_ghz=1.257,
            wavelength_m=c / freq,  # ~0.238498 m
            polarizations=("VV", "VH"),
            range_resolution_m=6.0,
            azimuth_resolution_m=6.0,
            repeat_cycle_days=12.0,
        )

    @classmethod
    def nisar_s_band(cls) -> "SensorConfig":
        """NASA-ISRO SAR (NISAR) S-band mode (3.200 GHz, ~9.37 cm wavelength)."""
        freq = 3.200e9
        c = 299792458.0
        return cls(
            name=SensorType.NISAR,
            band="S",
            frequency_ghz=3.200,
            wavelength_m=c / freq,  # ~0.093685 m
            polarizations=("VV", "VH"),
            range_resolution_m=4.0,
            azimuth_resolution_m=4.0,
            repeat_cycle_days=12.0,
        )

    @property
    def fringe_ambiguity_mm(self) -> float:
        """Displacement in millimeters corresponding to one 2pi interferometric fringe."""
        return (self.wavelength_m * 1000.0) / 2.0

    @property
    def phase_per_mm(self) -> float:
        """Phase change in radians corresponding to 1 mm of displacement."""
        return (4.0 * math.pi) / (self.wavelength_m * 1000.0)

    def phase_to_displacement_mm(self, phase: torch.Tensor) -> torch.Tensor:
        """Convert interferometric phase to line-of-sight displacement in mm with sub-millimeter precision."""
        return phase_to_displacement_mm(phase, self.wavelength_m)

    def displacement_mm_to_phase(self, disp_mm: torch.Tensor) -> torch.Tensor:
        """Convert line-of-sight displacement in mm to interferometric phase in radians."""
        return displacement_mm_to_phase(disp_mm, self.wavelength_m)


def phase_to_displacement_mm(
    phase: torch.Tensor,
    wavelength_m: float,
) -> torch.Tensor:
    """
    Convert interferometric phase (radians) to line-of-sight displacement (mm).
    Guarantees sub-millimeter precision:
    disp_mm = - (wavelength_m * 1000.0) / (4.0 * pi) * phase

    Args:
        phase: Phase tensor in radians.
        wavelength_m: Radar carrier wavelength in meters.

    Returns:
        Displacement in millimeters.
    """
    scale = -(wavelength_m * 1000.0) / (4.0 * math.pi)
    return phase * scale


def displacement_mm_to_phase(
    displacement_mm: torch.Tensor,
    wavelength_m: float,
) -> torch.Tensor:
    """
    Convert line-of-sight displacement (mm) to interferometric phase (radians).
    Guarantees sub-millimeter precision:
    phase = - (4.0 * pi) / (wavelength_m * 1000.0) * displacement_mm

    Args:
        displacement_mm: Displacement tensor in millimeters.
        wavelength_m: Radar carrier wavelength in meters.

    Returns:
        Phase in radians.
    """
    scale = -(4.0 * math.pi) / (wavelength_m * 1000.0)
    return displacement_mm * scale


@dataclass
class DualPolComplexTensor:
    """
    Container for dual-polarization (VV/VH) complex radar tensors (SLC / Single Look Complex).
    Supports Cartesian (real/imag) and Polar (amplitude/phase) representations,
    polarimetric decomposition, and complex interferogram construction.
    """
    vv: torch.Tensor  # Complex tensor (..., H, W), dtype torch.complex64 or torch.complex128
    vh: torch.Tensor  # Complex tensor (..., H, W), dtype torch.complex64 or torch.complex128
    sensor_config: Optional[SensorConfig] = None

    def __post_init__(self):
        if self.vv.shape != self.vh.shape:
            raise ValueError(
                f"Shape mismatch between VV {self.vv.shape} and VH {self.vh.shape}"
            )
        if not torch.is_complex(self.vv):
            raise TypeError(f"Expected complex tensor for VV, got {self.vv.dtype}")
        if not torch.is_complex(self.vh):
            raise TypeError(f"Expected complex tensor for VH, got {self.vh.dtype}")

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.vv.shape

    @property
    def intensity_vv(self) -> torch.Tensor:
        """VV linear intensity sigma_0 = |S_VV|^2 (strictly non-negative)."""
        return torch.clamp(self.vv.abs() ** 2, min=0.0)

    @property
    def intensity_vh(self) -> torch.Tensor:
        """VH linear intensity sigma_0 = |S_VH|^2 (strictly non-negative)."""
        return torch.clamp(self.vh.abs() ** 2, min=0.0)

    def span(self, factor: float = 2.0) -> torch.Tensor:
        """Compute polarimetric Span: sigma_VV + factor * sigma_VH (strictly non-negative)."""
        return compute_span(self.intensity_vv, self.intensity_vh, factor=factor)

    def cross_ratio(self, eps: float = 1e-7, in_db: bool = False) -> torch.Tensor:
        """Compute Cross-Ratio (depolarization ratio) CR = sigma_VH / sigma_VV."""
        return compute_cross_ratio(self.intensity_vv, self.intensity_vh, eps=eps, in_db=in_db)

    def rvi(self, eps: float = 1e-7) -> torch.Tensor:
        """Compute dual-polarization Radar Vegetation Index (RVI) in [0, 1]."""
        return compute_rvi(self.intensity_vv, self.intensity_vh, eps=eps)

    def compute_interferogram(
        self, other: "DualPolComplexTensor", pol: str = "VV"
    ) -> torch.Tensor:
        """Form complex interferogram: I = S_1 * conj(S_2)."""
        s1 = getattr(self, pol.lower())
        s2 = getattr(other, pol.lower())
        return compute_complex_interferogram(s1, s2)

    def compute_coherence(
        self,
        other: "DualPolComplexTensor",
        window_size: int = 5,
        pol: str = "VV",
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """Estimate spatial complex coherence in [0, 1]."""
        s1 = getattr(self, pol.lower())
        s2 = getattr(other, pol.lower())
        return estimate_spatial_coherence(s1, s2, window_size=window_size, eps=eps)

    def to(self, device: Union[str, torch.device]) -> "DualPolComplexTensor":
        """Move underlying complex tensors to target device."""
        return DualPolComplexTensor(
            vv=self.vv.to(device),
            vh=self.vh.to(device),
            sensor_config=self.sensor_config,
        )


def construct_dual_pol_complex_tensor(
    vv_component1: torch.Tensor,
    vv_component2: torch.Tensor,
    vh_component1: torch.Tensor,
    vh_component2: torch.Tensor,
    representation: str = "cartesian",
    sensor_config: Optional[SensorConfig] = None,
) -> DualPolComplexTensor:
    """
    Construct dual-pol (VV/VH) complex SLC tensors.

    Args:
        vv_component1: Real part (if 'cartesian') or Amplitude (if 'polar') for VV.
        vv_component2: Imaginary part (if 'cartesian') or Phase in radians (if 'polar') for VV.
        vh_component1: Real part (if 'cartesian') or Amplitude (if 'polar') for VH.
        vh_component2: Imaginary part (if 'cartesian') or Phase in radians (if 'polar') for VH.
        representation: 'cartesian' (real, imag) or 'polar' (amplitude, phase).
        sensor_config: Optional SensorConfig (Sentinel-1 or NISAR).

    Returns:
        DualPolComplexTensor containing complex VV and VH tensors.
    """
    if representation == "cartesian":
        vv = torch.complex(vv_component1, vv_component2)
        vh = torch.complex(vh_component1, vh_component2)
    elif representation == "polar":
        amp_vv = torch.clamp(vv_component1, min=0.0)
        phi_vv = vv_component2
        amp_vh = torch.clamp(vh_component1, min=0.0)
        phi_vh = vh_component2
        vv = torch.complex(amp_vv * torch.cos(phi_vv), amp_vv * torch.sin(phi_vv))
        vh = torch.complex(amp_vh * torch.cos(phi_vh), amp_vh * torch.sin(phi_vh))
    else:
        raise ValueError(
            f"Unsupported representation '{representation}'. Choose 'cartesian' or 'polar'."
        )

    return DualPolComplexTensor(vv=vv, vh=vh, sensor_config=sensor_config)


def compute_complex_interferogram(
    slc1: torch.Tensor,
    slc2: torch.Tensor,
) -> torch.Tensor:
    """
    Compute complex interferogram from two coregistered Single Look Complex (SLC) SAR images:
    I = S_1 * S_2^*

    Args:
        slc1: Complex master SLC tensor of shape (..., H, W).
        slc2: Complex slave SLC tensor of shape (..., H, W).

    Returns:
        Complex interferogram tensor of shape (..., H, W).
    """
    if slc1.shape != slc2.shape:
        raise ValueError(f"Shape mismatch: {slc1.shape} != {slc2.shape}")
    return slc1 * torch.conj(slc2)

def compute_span(
    vv_intensity: torch.Tensor,
    vh_intensity: torch.Tensor,
    factor: float = 2.0,
) -> torch.Tensor:
    """
    Compute total scattered power (Span) for dual-polarization SAR.
    Span = sigma_VV + 2 * sigma_VH (assuming HV reciprocity).

    Args:
        vv_intensity: Tensor of VV backscatter intensity (linear power).
        vh_intensity: Tensor of VH backscatter intensity (linear power).
        factor: Reciprocity multiplier for cross-pol (default: 2.0).

    Returns:
        Tensor of polarimetric Span in linear intensity.
    """
    if vv_intensity.shape != vh_intensity.shape:
        raise ValueError(
            f"Shape mismatch: VV shape {vv_intensity.shape} != VH shape {vh_intensity.shape}"
        )
    span = vv_intensity + factor * vh_intensity
    return torch.clamp(span, min=0.0)


def compute_cross_ratio(
    vv_intensity: torch.Tensor,
    vh_intensity: torch.Tensor,
    eps: float = 1e-7,
    in_db: bool = False,
) -> torch.Tensor:
    """
    Compute Cross-Ratio (depolarization ratio) CR = sigma_VH / sigma_VV.

    Args:
        vv_intensity: Linear power VV intensity.
        vh_intensity: Linear power VH intensity.
        eps: Regularization constant preventing division by zero.
        in_db: Whether to convert the cross-ratio to decibels (10*log10).

    Returns:
        Tensor of cross-ratio.
    """
    ratio = (vh_intensity + eps) / (vv_intensity + eps)
    if in_db:
        return 10.0 * torch.log10(torch.clamp(ratio, min=1e-12))
    return ratio


def compute_rvi(
    vv_intensity: torch.Tensor,
    vh_intensity: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Compute dual-polarization Radar Vegetation Index (RVI):
    RVI = (4 * sigma_VH) / (sigma_VV + sigma_VH + eps)
    Ranges from 0.0 (bare soil / specular) to 1.0 (dense volume scattering).

    Args:
        vv_intensity: Linear power VV intensity.
        vh_intensity: Linear power VH intensity.
        eps: Regularization constant.

    Returns:
        Tensor of RVI normalized in [0, 1].
    """
    denominator = vv_intensity + vh_intensity + eps
    rvi = (4.0 * vh_intensity) / denominator
    return torch.clamp(rvi, 0.0, 1.0)


def wrap_phase(continuous_phase: torch.Tensor) -> torch.Tensor:
    """
    Wrap continuous phase psi to interval [-pi, +pi].
    Uses the robust atan2(sin, cos) formulation.

    Args:
        continuous_phase: Tensor of continuous phase (radians).

    Returns:
        Wrapped phase tensor in [-pi, +pi].
    """
    return torch.atan2(torch.sin(continuous_phase), torch.cos(continuous_phase))


def estimate_spatial_coherence(
    slc1: torch.Tensor,
    slc2: torch.Tensor,
    window_size: int = 5,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Estimate spatial complex interferometric coherence gamma:
    gamma = |E[S1 * S2*]| / sqrt(E[|S1|^2] * E[|S2|^2])
    where E[] is evaluated via a 2D moving spatial multi-look window.

    Args:
        slc1: Complex SLC tensor of shape (..., H, W).
        slc2: Complex SLC tensor of shape (..., H, W).
        window_size: Moving box-car multi-look window width (must be odd).
        eps: Small constant preventing division by zero.

    Returns:
        Tensor of coherence magnitude gamma in [0, 1].
    """
    if window_size % 2 == 0:
        raise ValueError(f"window_size must be odd, got {window_size}")

    # Determine batch dimensions
    orig_shape = slc1.shape
    h, w = orig_shape[-2], orig_shape[-1]
    slc1_flat = slc1.reshape(-1, 1, h, w)
    slc2_flat = slc2.reshape(-1, 1, h, w)

    # Complex cross product and powers
    # S1 * conj(S2)
    s1_conj_s2 = slc1_flat * torch.conj(slc2_flat)
    cross_real = s1_conj_s2.real
    cross_imag = s1_conj_s2.imag
    p1 = (slc1_flat.abs()) ** 2
    p2 = (slc2_flat.abs()) ** 2

    # Box-car filter via avg_pool2d
    pad = window_size // 2
    cross_real_avg = F.avg_pool2d(cross_real, kernel_size=window_size, stride=1, padding=pad)
    cross_imag_avg = F.avg_pool2d(cross_imag, kernel_size=window_size, stride=1, padding=pad)
    p1_avg = F.avg_pool2d(p1, kernel_size=window_size, stride=1, padding=pad)
    p2_avg = F.avg_pool2d(p2, kernel_size=window_size, stride=1, padding=pad)

    cross_mag = torch.sqrt(cross_real_avg**2 + cross_imag_avg**2)
    denom = torch.sqrt(torch.clamp(p1_avg * p2_avg, min=eps)) + eps
    coherence = cross_mag / denom
    coherence = torch.clamp(coherence, 0.0, 1.0)

    return coherence.reshape(orig_shape)


@dataclass
class RadarTensorStack:
    """
    InSAR / PolSAR spatio-temporal tensor stack container.
    Stores multi-temporal backscatter, polarimetry, phase, and coherence.
    """
    sensor_config: SensorConfig
    temporal_baseline_days: torch.Tensor  # Shape: (T,)
    vv_intensity: torch.Tensor            # Shape: (T, H, W)
    vh_intensity: torch.Tensor            # Shape: (T, H, W)
    span: torch.Tensor                    # Shape: (T, H, W)
    cross_ratio: torch.Tensor             # Shape: (T, H, W)
    rvi: torch.Tensor                     # Shape: (T, H, W)
    wrapped_phase: torch.Tensor           # Shape: (T, H, W) in [-pi, pi]
    coherence: torch.Tensor               # Shape: (T, H, W) in [0, 1]
    continuous_phase: Optional[torch.Tensor] = None  # Shape: (T, H, W)
    displacement_mm: Optional[torch.Tensor] = None   # Shape: (T, H, W)
    dem: Optional[torch.Tensor] = None               # Shape: (H, W)
    metadata: Dict[str, Union[float, str, np.ndarray]] = field(default_factory=dict)

    @property
    def shape(self) -> Tuple[int, int, int]:
        """Return (T, H, W) dimensions."""
        return tuple(self.wrapped_phase.shape)  # type: ignore

    @property
    def fringe_ambiguity_mm(self) -> float:
        """Displacement in mm corresponding to one 2pi interferometric fringe."""
        return self.sensor_config.fringe_ambiguity_mm

    def phase_to_displacement(self, phase: torch.Tensor) -> torch.Tensor:
        """Convert interferometric phase to line-of-sight displacement in mm with sub-millimeter precision."""
        return self.sensor_config.phase_to_displacement_mm(phase)

    def displacement_to_phase(self, disp_mm: torch.Tensor) -> torch.Tensor:
        """Convert line-of-sight displacement in mm to interferometric phase in radians."""
        return self.sensor_config.displacement_mm_to_phase(disp_mm)

    def to(self, device: Union[str, torch.device]) -> "RadarTensorStack":
        """Move all underlying tensors to target device."""
        return RadarTensorStack(
            sensor_config=self.sensor_config,
            temporal_baseline_days=self.temporal_baseline_days.to(device),
            vv_intensity=self.vv_intensity.to(device),
            vh_intensity=self.vh_intensity.to(device),
            span=self.span.to(device),
            cross_ratio=self.cross_ratio.to(device),
            rvi=self.rvi.to(device),
            wrapped_phase=self.wrapped_phase.to(device),
            coherence=self.coherence.to(device),
            continuous_phase=(
                self.continuous_phase.to(device)
                if self.continuous_phase is not None
                else None
            ),
            displacement_mm=(
                self.displacement_mm.to(device)
                if self.displacement_mm is not None
                else None
            ),
            dem=self.dem.to(device) if self.dem is not None else None,
            metadata=self.metadata,
        )


def generate_highwall_simulation(
    sensor: SensorType = SensorType.SENTINEL_1,
    num_timesteps: int = 12,
    grid_size: Tuple[int, int] = (128, 128),
    seed: int = 42,
    noise_level: float = 0.15,
) -> RadarTensorStack:
    """
    Generate synthetic, physically grounded open-pit mining high-wall InSAR dataset.
    Simulates high-wall bench topography, localized tertiary accelerating creep on the
    pit crest, stable toe, atmospheric phase screen (APS), and vegetation decorrelation.

    Args:
        sensor: SensorType (Sentinel-1 C-band or NISAR L-band).
        num_timesteps: Number of sequential temporal acquisitions (epochs).
        grid_size: Spatial grid dimensions (H, W).
        seed: Random seed for reproducibility.
        noise_level: Magnitude of phase speckle noise and APS turbulence.

    Returns:
        RadarTensorStack populated with polarimetry, phase, and coherence.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    config = (
        SensorConfig.sentinel_1_c_band()
        if sensor == SensorType.SENTINEL_1
        else SensorConfig.nisar_l_band()
    )
    wavelength = config.wavelength_m
    h, w = grid_size

    # 1. Physical Grid & DEM (Open-pit high-wall slope profile)
    # y-axis represents elevation descent: y=0 (pit crest: ~350m), y=H (pit toe: ~100m)
    y_coords = torch.linspace(0, 1, h)
    x_coords = torch.linspace(0, 1, w)
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")

    # Bench terraces: smooth slope with 4 stepped mining benches
    bench_steps = 4.0
    bench_profile = yy - 0.04 * torch.sin(2.0 * math.pi * bench_steps * yy)
    dem = 350.0 - 250.0 * bench_profile  # Elevation in meters

    # 2. Temporal acquisition timeline (days)
    repeat_days = config.repeat_cycle_days
    t_days = torch.arange(0, num_timesteps, dtype=torch.float32) * repeat_days
    t_years = t_days / 365.25

    # 3. True Kinematic Displacement Field d(t, y, x) [mm]
    # Localized instability centered on the pit crest (y=0.25, x=0.50)
    crest_cy, crest_cx = 0.25, 0.50
    dist_sq = ((yy - crest_cy) / 0.18) ** 2 + ((xx - crest_cx) / 0.25) ** 2
    instability_spatial_mask = torch.exp(-dist_sq)  # Peak at crest failure zone

    # Fukuzono-Voight tertiary accelerating creep:
    # d(t) = v0*t + 0.5*a*t^2 + tertiary_onset * exp(k * t)
    # Secondary creep: steady 15 mm/yr
    # Tertiary creep accelerates rapidly towards collapse around t = 160 days (~0.44 yr)
    v0 = 18.0  # mm/yr base velocity in unstable crest
    a0 = 35.0  # mm/yr^2 acceleration
    k_tertiary = 4.5  # Acceleration exponent

    disp_stack = torch.zeros(num_timesteps, h, w, dtype=torch.float32)
    for t_idx in range(num_timesteps):
        t = t_years[t_idx].item()
        # Non-linear accelerating crest displacement
        d_crest = (v0 * t + 0.5 * a0 * (t**2) + 2.5 * (math.exp(k_tertiary * t) - 1.0))
        # Stable / toe displacement: minimal linear settlement (<2 mm/yr)
        d_stable = 1.2 * t + 0.5 * math.sin(2.0 * math.pi * t)  # Slight seasonal thermal drift

        disp = (d_crest * instability_spatial_mask) + (d_stable * (1.0 - instability_spatial_mask))
        disp_stack[t_idx] = disp

    # 4. Phase Conversion: psi = -4*pi / lambda * (disp / 1000) [radians]
    phase_factor = -4.0 * math.pi / wavelength
    disp_meters = disp_stack / 1000.0
    continuous_phase = phase_factor * disp_meters

    # Add topographic phase component (flattened / residual DEM contribution)
    topo_residual = 0.05 * (dem - dem.mean()) / dem.std()
    continuous_phase += topo_residual.unsqueeze(0)

    # 5. Atmospheric Phase Screen (APS) & Spatial Noise
    # Atmospheric turbulence: smooth spatial 2D Gaussian filter on white noise
    kernel_size = 15
    aps_kernel = torch.ones(1, 1, kernel_size, kernel_size) / (kernel_size * kernel_size)
    aps_noise = torch.randn(num_timesteps, 1, h, w) * noise_level * 3.5
    aps = F.conv2d(aps_noise, aps_kernel, padding=kernel_size // 2).squeeze(1)
    continuous_phase += aps

    # Add localized speckle noise
    phase_noise = torch.randn_like(continuous_phase) * (noise_level * 0.4)
    continuous_phase += phase_noise

    # 6. Wrapped Phase [-pi, +pi]
    wrapped_phase = wrap_phase(continuous_phase)

    # 7. Polarimetric Backscatter & Vegetation Simulation
    # Bare rock on high-wall benches (low RVI ~0.1-0.25), vegetated surrounding plateau (RVI ~0.6-0.8)
    vegetation_mask = torch.sigmoid((yy - 0.7) * 8.0) + 0.15 * torch.sigmoid(-(yy - 0.1) * 8.0)
    vegetation_mask = torch.clamp(vegetation_mask, 0.0, 1.0)

    # C-band has higher backscatter variation; L-band penetrates deeper
    if sensor == SensorType.SENTINEL_1:
        base_vv = 0.12 - 0.06 * vegetation_mask + 0.03 * torch.randn(num_timesteps, h, w).abs()
        base_vh = 0.015 + 0.035 * vegetation_mask + 0.008 * torch.randn(num_timesteps, h, w).abs()
    else:  # NISAR L-band
        base_vv = 0.18 - 0.04 * vegetation_mask + 0.02 * torch.randn(num_timesteps, h, w).abs()
        base_vh = 0.025 + 0.055 * vegetation_mask + 0.012 * torch.randn(num_timesteps, h, w).abs()

    vv_intensity = torch.clamp(base_vv, min=1e-4)
    vh_intensity = torch.clamp(base_vh, min=1e-5)

    span = compute_span(vv_intensity, vh_intensity)
    cross_ratio = compute_cross_ratio(vv_intensity, vh_intensity)
    rvi = compute_rvi(vv_intensity, vh_intensity)

    # 8. Physical Coherence Simulation:
    # High coherence on exposed high-wall rock, moderate in dry areas, decorrelation over vegetation
    # NISAR (L-band) maintains higher coherence in vegetated zones than Sentinel-1 (C-band)
    veg_decorrelation = 0.40 if sensor == SensorType.SENTINEL_1 else 0.15
    temporal_decorrelation = 0.02 * t_days.unsqueeze(-1).unsqueeze(-1) / 30.0

    coherence_base = 0.92 - (veg_decorrelation * rvi) - (0.15 * (disp_stack.abs() / 25.0).clamp(0, 1))
    coherence_base = coherence_base - temporal_decorrelation
    coherence = torch.clamp(coherence_base, 0.10, 0.99)

    return RadarTensorStack(
        sensor_config=config,
        temporal_baseline_days=t_days,
        vv_intensity=vv_intensity,
        vh_intensity=vh_intensity,
        span=span,
        cross_ratio=cross_ratio,
        rvi=rvi,
        wrapped_phase=wrapped_phase,
        coherence=coherence,
        continuous_phase=continuous_phase,
        displacement_mm=disp_stack,
        dem=dem,
        metadata={
            "site": "Chirano Gold Mine High-Wall Open Pit (Ghana)",
            "sensor": config.name.value,
            "band": config.band,
            "wavelength_m": config.wavelength_m,
            "temporal_epochs": num_timesteps,
            "grid_resolution_m": config.range_resolution_m,
        },
    )


class SARTensorDataset(Dataset):
    """
    PyTorch Dataset yielding multi-channel radar tensors for ViT-MAE pretraining
    and Deep Phase Unwrapping regression.
    Channels: [cos(phi), sin(phi), coherence, span_norm, rvi] (5 channels).
    """

    def __init__(
        self,
        tensor_stack: RadarTensorStack,
        normalize_span: bool = True,
    ):
        super().__init__()
        self.stack = tensor_stack
        self.t_len = tensor_stack.wrapped_phase.shape[0]

        # Prepare 5-channel feature tensors for each temporal epoch:
        # Channel 0: cos(wrapped_phase)
        # Channel 1: sin(wrapped_phase)
        # Channel 2: coherence gamma in [0, 1]
        # Channel 3: normalized span
        # Channel 4: RVI in [0, 1]

        phi = tensor_stack.wrapped_phase
        coh = tensor_stack.coherence
        span = tensor_stack.span
        rvi = tensor_stack.rvi

        if normalize_span:
            s_min = span.min()
            s_max = span.max()
            span_norm = (span - s_min) / (s_max - s_min + 1e-7)
        else:
            span_norm = span

        cos_phi = torch.cos(phi)
        sin_phi = torch.sin(phi)

        # Feature shape: (T, 5, H, W)
        self.features = torch.stack(
            [cos_phi, sin_phi, coh, span_norm, rvi], dim=1
        ).float()

        self.continuous_phase = (
            tensor_stack.continuous_phase.float()
            if tensor_stack.continuous_phase is not None
            else None
        )
        self.coherence = coh.float()

    def __len__(self) -> int:
        return self.t_len

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = {
            "features": self.features[idx],         # Shape: (5, H, W)
            "coherence": self.coherence[idx],       # Shape: (H, W)
        }
        if self.continuous_phase is not None:
            sample["continuous_phase"] = self.continuous_phase[idx]  # Shape: (H, W)
        return sample
