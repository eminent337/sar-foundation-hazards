"""
Tests for ViT-MAE Foundation architecture and Deep Phase Unwrapping heads.
"""

import math
import pytest
import torch

from sar_core.vit_mae_foundation import (
    PatchEmbed,
    ViTMAEFoundation,
    DeepPhaseUnwrappingHead,
    ViTMAELoss,
    get_2d_sincos_pos_embed,
)


def test_pos_embed_shape():
    grid_size = (16, 16)
    embed_dim = 128
    pos_embed = get_2d_sincos_pos_embed(embed_dim, grid_size)
    assert pos_embed.shape == (256, 128)
    assert not torch.isnan(pos_embed).any()


def test_patch_embed():
    pe = PatchEmbed(img_size=(64, 64), patch_size=8, in_chans=5, embed_dim=96)
    assert pe.num_patches == 64
    x = torch.randn(2, 5, 64, 64)
    tokens = pe(x)
    assert tokens.shape == (2, 64, 96)


def test_vit_mae_foundation_forward():
    model = ViTMAEFoundation(
        img_size=(64, 64),
        patch_size=8,
        in_chans=5,
        encoder_embed_dim=96,
        encoder_depth=2,
        encoder_num_heads=3,
        decoder_embed_dim=48,
        decoder_depth=2,
        decoder_num_heads=3,
        mask_ratio=0.75,
    )

    x = torch.randn(2, 5, 64, 64)
    out = model(x)
    assert "recon_loss" in out
    assert "pred_patches" in out
    assert "mask" in out
    assert "unwrapped_phase" in out

    assert out["recon_loss"].ndim == 0
    assert out["unwrapped_phase"].shape == (2, 1, 64, 64)
    assert out["mask"].shape == (2, 64)
    # Check 75% masking: 64 * 0.75 = 48 masked
    assert int(out["mask"].sum().item()) == 48 * 2


def test_forward_unwrap_inference():
    model = ViTMAEFoundation(
        img_size=(32, 32),
        patch_size=8,
        in_chans=5,
        encoder_embed_dim=64,
        encoder_depth=2,
        encoder_num_heads=2,
        decoder_embed_dim=32,
        decoder_depth=1,
        decoder_num_heads=2,
    )
    x = torch.randn(3, 5, 32, 32)
    unw = model.forward_unwrap(x)
    assert unw.shape == (3, 1, 32, 32)


def test_vit_mae_loss():
    criterion = ViTMAELoss(
        weight_recon=0.5,
        weight_phase=1.0,
        weight_wrap=0.5,
        weight_grad=0.2,
    )

    pred_dict = {
        "recon_loss": torch.tensor(0.42),
        "unwrapped_phase": torch.randn(2, 1, 32, 32),
    }
    target_phase = torch.randn(2, 32, 32)
    wrapped_phase = torch.atan2(torch.sin(target_phase), torch.cos(target_phase))
    coherence = torch.rand(2, 32, 32)

    loss_dict = criterion(
        pred_dict=pred_dict,
        target_phase=target_phase,
        wrapped_phase=wrapped_phase,
        coherence=coherence,
    )

    assert "loss_total" in loss_dict
    assert "loss_recon" in loss_dict
    assert "loss_phase" in loss_dict
    assert "loss_wrap" in loss_dict
    assert "loss_grad" in loss_dict
    assert loss_dict["loss_total"].item() > 0.0

def test_head_integer_wrap_count_2pi_ambiguity():
    """Verify resolving psi = phi + 2*pi*k to determine integer wrap ambiguity k."""
    two_pi = 2.0 * math.pi
    phi = torch.tensor([0.2, -1.5, 2.1, -0.8], dtype=torch.float32)
    true_k = torch.tensor([0, 1, -2, 5], dtype=torch.int64)
    psi = phi + two_pi * true_k.float()

    pred_k = DeepPhaseUnwrappingHead.compute_integer_wrap_count(psi, phi)
    assert torch.equal(pred_k, true_k)


def test_head_verify_cyclic_consistency():
    """Verify cyclic consistency check 1 - cos(psi - phi) < tol."""
    two_pi = 2.0 * math.pi
    phi = torch.tensor([0.5, -2.0, 1.2], dtype=torch.float32)
    psi_exact = phi + two_pi * torch.tensor([3.0, -1.0, 2.0])

    res, is_cons = DeepPhaseUnwrappingHead.verify_cyclic_consistency(psi_exact, phi)
    assert torch.all(is_cons)
    assert torch.allclose(res, torch.zeros_like(res), atol=1e-5)


def test_unwrap_to_displacement_submillimeter():
    """Test end-to-end unwrap_to_displacement method."""
    model = ViTMAEFoundation(
        img_size=(32, 32),
        patch_size=8,
        in_chans=5,
        encoder_embed_dim=48,
        encoder_depth=2,
        encoder_num_heads=3,
        decoder_embed_dim=24,
        decoder_depth=1,
        decoder_num_heads=3,
    )
    x = torch.randn(2, 5, 32, 32)
    wavelength_c_band = 0.0554658

    disp_mm = model.unwrap_to_displacement(x, wavelength_c_band)
    assert disp_mm.shape == (2, 1, 32, 32)
    assert not torch.isnan(disp_mm).any()


def test_cyclic_consistency_loss_zero_at_multiples_of_2pi():
    """Verify cyclic loss is zero when unwrapped phase is an exact integer multiple of 2pi away."""
    loss_fn = ViTMAELoss(weight_recon=0.0, weight_phase=0.0, weight_wrap=1.0, weight_grad=0.0)
    two_pi = 2.0 * math.pi
    wrapped = torch.tensor([[[[0.5, -1.2], [2.0, -0.4]]]])
    continuous = wrapped + two_pi * 4.0

    pred_dict = {
        "unwrapped_phase": continuous,
        "recon_loss": torch.tensor(0.0),
    }
    losses = loss_fn(pred_dict=pred_dict, wrapped_phase=wrapped)
    assert losses["loss_wrap"].item() == pytest.approx(0.0, abs=1e-5)


def test_end_to_end_phase_unwrapping_optimization():
    """Verify model optimization on synthetic fringe pattern without NaNs or instability."""
    torch.manual_seed(42)
    model = ViTMAEFoundation(
        img_size=(32, 32),
        patch_size=8,
        in_chans=5,
        encoder_embed_dim=48,
        encoder_depth=2,
        encoder_num_heads=3,
        decoder_embed_dim=24,
        decoder_depth=1,
        decoder_num_heads=3,
        mask_ratio=0.75,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = ViTMAELoss(weight_recon=0.2, weight_phase=1.0, weight_wrap=0.5, weight_grad=0.1)

    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, 32), torch.linspace(-1, 1, 32), indexing="ij"
    )
    true_phase = 3.5 * grid_x
    wrapped = torch.atan2(torch.sin(true_phase), torch.cos(true_phase))
    coh = torch.ones(32, 32) * 0.9

    feats = torch.stack(
        [torch.cos(wrapped), torch.sin(wrapped), coh, torch.ones(32, 32), torch.zeros(32, 32)],
        dim=0,
    ).unsqueeze(0)

    losses = []
    for _ in range(3):
        optimizer.zero_grad()
        out = model(feats)
        loss_dict = loss_fn(
            pred_dict=out,
            target_phase=true_phase.unsqueeze(0),
            wrapped_phase=wrapped.unsqueeze(0),
            coherence=coh.unsqueeze(0),
        )
        total = loss_dict["loss_total"]
        total.backward()
        optimizer.step()
        losses.append(total.item())

    for l in losses:
        assert not math.isnan(l)
        assert not math.isinf(l)
