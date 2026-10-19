"""
Tests for sar_tensor_pipeline module: sensor configurations, polarimetry,
phase wrapping, spatial coherence, and synthetic simulation.
"""

import math
import pytest
import torch
import numpy as np

from sar_core.sar_tensor_pipeline import (
    SensorType,
    SensorConfig,
    DualPolComplexTensor,
    RadarTensorStack,
    construct_dual_pol_complex_tensor,
    compute_complex_interferogram,
    compute_span,
    compute_cross_ratio,
    compute_rvi,
    wrap_phase,
    estimate_spatial_coherence,
    generate_highwall_simulation,
    SARTensorDataset,
    phase_to_displacement_mm,
    displacement_mm_to_phase,
)


def test_sensor_config_sentinel_1():
    s1 = SensorConfig.sentinel_1_c_band()
    assert s1.name == SensorType.SENTINEL_1
    assert s1.band == "C"
    assert math.isclose(s1.frequency_ghz, 5.405, rel_tol=1e-3)
    assert math.isclose(s1.wavelength_m, 0.0554658, rel_tol=1e-3)
    assert s1.polarizations == ("VV", "VH")
    assert s1.repeat_cycle_days == 12.0
    assert s1.range_resolution_m == 5.0


def test_sensor_config_nisar():
    nisar = SensorConfig.nisar_l_band()
    assert nisar.name == SensorType.NISAR
    assert nisar.band == "L"
    assert math.isclose(nisar.frequency_ghz, 1.257, rel_tol=1e-3)
    assert math.isclose(nisar.wavelength_m, 0.238498, rel_tol=1e-3)
    assert nisar.repeat_cycle_days == 12.0


def test_compute_span():
    vv = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    vh = torch.tensor([[0.5, 0.25], [1.0, 1.5]])
    span = compute_span(vv, vh, factor=2.0)
    expected = vv + 2.0 * vh
    assert torch.allclose(span, expected)


def test_compute_cross_ratio():
    vv = torch.tensor([[2.0, 4.0]])
    vh = torch.tensor([[1.0, 1.0]])
    cr = compute_cross_ratio(vv, vh, in_db=False)
    assert torch.allclose(cr, torch.tensor([[0.5, 0.25]]), atol=1e-4)

    cr_db = compute_cross_ratio(vv, vh, in_db=True)
    assert torch.all(cr_db < 0.0)  # Since ratio < 1


def test_compute_rvi():
    vv = torch.tensor([[1.0, 2.0]])
    vh = torch.tensor([[0.5, 0.5]])
    rvi = compute_rvi(vv, vh)
    assert torch.all((rvi >= 0.0) & (rvi <= 1.0))
    # For vv=1, vh=0.5 -> 4*0.5 / (1.5) = 2.0/1.5 = 1.333 clamped to 1.0
    assert math.isclose(rvi[0, 0].item(), 1.0, abs_tol=1e-3)


def test_wrap_phase():
    raw_phase = torch.tensor([-4 * math.pi, -math.pi, 0.0, math.pi / 2, math.pi, 3 * math.pi])
    wrapped = wrap_phase(raw_phase)
    assert torch.all(wrapped >= -math.pi)
    assert torch.all(wrapped <= math.pi)
    assert math.isclose(wrapped[2].item(), 0.0, abs_tol=1e-5)
    assert math.isclose(wrapped[3].item(), math.pi / 2, abs_tol=1e-5)


def test_estimate_spatial_coherence():
    h, w = 32, 32
    # Perfectly correlated SLCs
    slc1 = torch.complex(torch.ones(1, h, w), torch.zeros(1, h, w))
    slc2 = torch.complex(torch.ones(1, h, w), torch.zeros(1, h, w))
    coh = estimate_spatial_coherence(slc1, slc2, window_size=5)
    assert coh.shape == (1, h, w)
    assert torch.all(coh > 0.95)

    # Decorrelated random SLCs
    torch.manual_seed(42)
    slc_rand1 = torch.complex(torch.randn(1, h, w), torch.randn(1, h, w))
    slc_rand2 = torch.complex(torch.randn(1, h, w), torch.randn(1, h, w))
    coh_rand = estimate_spatial_coherence(slc_rand1, slc_rand2, window_size=5)
    assert torch.all((coh_rand >= 0.0) & (coh_rand <= 1.0))
    assert coh_rand.mean().item() < 0.8


def test_generate_highwall_simulation():
    stack = generate_highwall_simulation(
        sensor=SensorType.SENTINEL_1,
        num_timesteps=6,
        grid_size=(64, 64),
        seed=123,
    )
    assert isinstance(stack, RadarTensorStack)
    assert stack.shape == (6, 64, 64)
    assert stack.wrapped_phase.shape == (6, 64, 64)
    assert stack.coherence.shape == (6, 64, 64)
    assert stack.displacement_mm.shape == (6, 64, 64)
    assert stack.dem.shape == (64, 64)
    assert torch.all(stack.coherence >= 0.0) and torch.all(stack.coherence <= 1.0)
    assert torch.all(stack.wrapped_phase >= -math.pi) and torch.all(stack.wrapped_phase <= math.pi)
    assert "site" in stack.metadata


def test_sar_tensor_dataset():
    stack = generate_highwall_simulation(
        sensor=SensorType.NISAR,
        num_timesteps=4,
        grid_size=(32, 32),
        seed=999,
    )
    dataset = SARTensorDataset(stack)
    assert len(dataset) == 4
    sample = dataset[0]
    assert "features" in sample
    assert "coherence" in sample
    assert sample["features"].shape == (5, 32, 32)
    assert sample["coherence"].shape == (32, 32)

def test_sensor_config_submillimeter_precision():
    """Verify sub-millimeter precision in phase and displacement round-trips."""
    cfg_c = SensorConfig.sentinel_1_c_band()
    cfg_l = SensorConfig.nisar_l_band()
    cfg_s = SensorConfig.nisar_s_band()

    assert cfg_s.band == "S"
    assert cfg_s.fringe_ambiguity_mm > 40.0

    for cfg in [cfg_c, cfg_l, cfg_s]:
        for d_val in [0.01, 0.05, 0.1, 0.5, 1.0, 5.0]:
            disp = torch.tensor([d_val], dtype=torch.float64)
            phase = cfg.displacement_mm_to_phase(disp)
            rec_disp = cfg.phase_to_displacement_mm(phase)
            assert torch.allclose(disp, rec_disp, atol=1e-9)


def test_phase_conversion_autograd():
    """Verify autograd gradient propagation through phase conversion."""
    wavelength = 0.0554658
    phase = torch.tensor([0.5, -1.2, 2.0], dtype=torch.float32, requires_grad=True)
    disp = phase_to_displacement_mm(phase, wavelength)
    loss = disp.sum()
    loss.backward()

    expected_scale = -(wavelength * 1000.0) / (4.0 * math.pi)
    assert torch.allclose(phase.grad, torch.tensor([expected_scale] * 3, dtype=torch.float32))


def test_dual_pol_complex_tensor_cartesian_and_polar():
    """Test constructing real dual-pol complex tensor in Cartesian and Polar forms."""
    # Cartesian
    r_vv = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    i_vv = torch.tensor([[0.5, -0.5], [1.0, 0.0]])
    r_vh = torch.tensor([[0.2, 0.4], [0.6, 0.8]])
    i_vh = torch.tensor([[0.1, -0.1], [0.2, 0.0]])

    slc_cart = construct_dual_pol_complex_tensor(r_vv, i_vv, r_vh, i_vh, representation="cartesian")
    assert slc_cart.shape == (2, 2)
    assert torch.is_complex(slc_cart.vv)
    assert torch.is_complex(slc_cart.vh)
    assert torch.all(slc_cart.intensity_vv >= 0.0)
    assert torch.all(slc_cart.intensity_vh >= 0.0)
    assert torch.all(slc_cart.span() >= 0.0)
    assert torch.all(slc_cart.cross_ratio() >= 0.0)
    assert torch.all((slc_cart.rvi() >= 0.0) & (slc_cart.rvi() <= 1.0))

    # Polar
    amp_vv = torch.tensor([[2.0, 1.5], [3.0, 0.5]])
    phi_vv = torch.tensor([[0.0, math.pi / 2], [-math.pi / 4, math.pi]])
    amp_vh = torch.tensor([[0.5, 0.3], [0.8, 0.1]])
    phi_vh = torch.tensor([[0.1, 0.2], [-0.3, 0.4]])

    slc_polar = construct_dual_pol_complex_tensor(amp_vv, phi_vv, amp_vh, phi_vh, representation="polar")
    assert torch.allclose(slc_polar.vv.abs(), amp_vv, atol=1e-5)
    assert torch.allclose(slc_polar.vh.abs(), amp_vh, atol=1e-5)


def test_compute_complex_interferogram():
    """Test interferogram formation and complex multilooking coherence."""
    s1 = torch.complex(torch.tensor([math.cos(0.5)]), torch.tensor([math.sin(0.5)]))
    s2 = torch.complex(2.0 * torch.tensor([math.cos(0.2)]), 2.0 * torch.tensor([math.sin(0.2)]))
    ifg = compute_complex_interferogram(s1, s2)

    assert torch.angle(ifg).item() == pytest.approx(0.3, abs=1e-5)
    assert ifg.abs().item() == pytest.approx(2.0, abs=1e-5)


def test_span_non_negativity_and_autograd():
    """Test polarimetric Span non-negativity and gradient backpropagation."""
    vv = torch.tensor([2.0, 4.0], dtype=torch.float32, requires_grad=True)
    vh = torch.tensor([1.0, 2.0], dtype=torch.float32, requires_grad=True)
    span = compute_span(vv, vh, factor=2.0)

    assert torch.all(span >= 0.0)
    loss = span.sum()
    loss.backward()
    assert torch.allclose(vv.grad, torch.tensor([1.0, 1.0]))
    assert torch.allclose(vh.grad, torch.tensor([2.0, 2.0]))


def test_cross_ratio_db_and_bounds():
    """Test cross-ratio linear and decibel conversion."""
    vv = torch.tensor([1.0, 4.0], dtype=torch.float32)
    vh = torch.tensor([0.25, 1.0], dtype=torch.float32)
    cr = compute_cross_ratio(vv, vh, in_db=False)
    assert torch.all(cr >= 0.0)
    assert cr[0].item() == pytest.approx(0.25, abs=1e-4)

    cr_db = compute_cross_ratio(vv, vh, in_db=True)
    assert cr_db[0].item() == pytest.approx(-6.0206, rel=1e-3)
