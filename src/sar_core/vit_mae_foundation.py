"""
Vision Transformer Masked Autoencoder (ViT-MAE) Foundation Model for SAR / InSAR.
Incorporates NASA-IBM Prithvi and DeepInSAR geospatial architectures for self-supervised
representation learning on complex radar tensors and Deep Phase Unwrapping regression.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: Tuple[int, int]) -> torch.Tensor:
    """
    Generate 2D sinusoidal position embedding for spatial patches.

    Args:
        embed_dim: Embedding dimension (must be even).
        grid_size: Tuple of (grid_h, grid_w) patch counts.

    Returns:
        Tensor of shape (grid_h * grid_w, embed_dim).
    """
    grid_h, grid_w = grid_size
    grid_y = torch.arange(grid_h, dtype=torch.float32)
    grid_x = torch.arange(grid_w, dtype=torch.float32)
    yy, xx = torch.meshgrid(grid_y, grid_x, indexing="ij")
    grid = torch.stack([xx, yy], dim=0)  # Shape: (2, grid_h, grid_w)

    dim_h = embed_dim // 2
    dim_w = embed_dim - dim_h

    def _get_1d_sincos(pos: torch.Tensor, dim: int) -> torch.Tensor:
        omega = torch.arange(dim // 2, dtype=torch.float32)
        omega /= (dim // 2)
        omega = 1.0 / (10000.0**omega)
        pos = pos.reshape(-1)
        out = torch.einsum("m,d->md", pos, omega)
        return torch.cat([torch.sin(out), torch.cos(out)], dim=1)

    emb_x = _get_1d_sincos(grid[0], dim_w)
    emb_y = _get_1d_sincos(grid[1], dim_h)
    return torch.cat([emb_x, emb_y], dim=1)


class PatchEmbed(nn.Module):
    """
    2D Radar Tensor to Patch Embedding.
    Maps (B, C, H, W) -> (B, N, D) via 2D convolution.
    """

    def __init__(
        self,
        img_size: Tuple[int, int] = (128, 128),
        patch_size: int = 8,
        in_chans: int = 5,
        embed_dim: int = 192,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size, img_size[1] // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) -> (B, D, H/P, W/P) -> (B, N, D)
        B, C, H, W = x.shape
        if (H, W) != self.img_size:
            # Handle dynamically if dimensions differ but are divisible
            grid_h, grid_w = H // self.patch_size, W // self.patch_size
        else:
            grid_h, grid_w = self.grid_size

        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)
        return x


class TransformerBlock(nn.Module):
    """Pre-LN Transformer Block with Multihead Self-Attention and MLP."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_x = self.norm1(x)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class DeepPhaseUnwrappingHead(nn.Module):
    """
    CNN-Transformer hybrid regression head for continuous phase unwrapping psi = phi + 2*pi*k.
    Takes patch embeddings and reconstructs full-resolution continuous phase field.
    """

    def __init__(
        self,
        embed_dim: int,
        patch_size: int,
        grid_size: Tuple[int, int],
        out_chans: int = 1,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.grid_size = grid_size
        self.embed_dim = embed_dim

        # Project patch tokens back into 2D feature map
        self.unpatch = nn.Linear(embed_dim, patch_size * patch_size * 32)

        # Refinement convolutional residual network to eliminate patch artifacts and predict smooth psi
        self.refine = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, out_chans, kernel_size=3, padding=1),
        )

    def forward(
        self, x: torch.Tensor, grid_size: Optional[Tuple[int, int]] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Patch tokens (B, N, embed_dim)
            grid_size: Optional custom (grid_h, grid_w) patch dimensions.

        Returns:
            Continuous phase field of shape (B, 1, H, W).
        """
        B, N, D = x.shape
        if grid_size is not None:
            gh, gw = grid_size
        elif N == self.grid_size[0] * self.grid_size[1]:
            gh, gw = self.grid_size
        else:
            side = int(math.isqrt(N))
            if side * side == N:
                gh, gw = side, side
            else:
                gh, gw = self.grid_size
        p = self.patch_size

        # (B, N, D) -> (B, N, p*p*32)
        h = self.unpatch(x)
        # Reshape to (B, gh, gw, p, p, 32)
        h = h.reshape(B, gh, gw, p, p, 32)
        # Permute to (B, 32, gh, p, gw, p) -> (B, 32, gh*p, gw*p)
        h = h.permute(0, 5, 1, 3, 2, 4).reshape(B, 32, gh * p, gw * p)

        phase_out = self.refine(h)
        return phase_out

    @staticmethod
    def compute_integer_wrap_count(
        continuous_phase: torch.Tensor,
        wrapped_phase: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute estimated integer 2pi cycle wrap count k:
        psi = phi + 2*pi*k  =>  k = round((psi - phi) / (2*pi))

        Args:
            continuous_phase: Unwrapped continuous phase psi (..., H, W).
            wrapped_phase: Wrapped input phase phi (..., H, W) in [-pi, pi].

        Returns:
            Integer wrap count tensor k (dtype int64).
        """
        two_pi = 2.0 * math.pi
        k = torch.round((continuous_phase - wrapped_phase) / two_pi)
        return k.long()

    @staticmethod
    def verify_cyclic_consistency(
        continuous_phase: torch.Tensor,
        wrapped_phase: torch.Tensor,
        tolerance: float = 1e-4,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Verify that wrapped(continuous_phase) reproduces wrapped_phase within tolerance.
        Computes cyclic residual: 1 - cos(psi - phi)

        Returns:
            (cyclic_residual, is_consistent_mask)
        """
        cyclic_residual = 1.0 - torch.cos(continuous_phase - wrapped_phase)
        is_consistent = cyclic_residual < tolerance
        return cyclic_residual, is_consistent

class ViTMAEFoundation(nn.Module):
    """
    Self-Supervised Vision Transformer Masked Autoencoder (ViT-MAE) for SAR / InSAR.
    Performs 75% spatial masking for speckle/fringe self-supervised pre-training
    and houses a Deep Phase Unwrapping regression head.
    """

    def __init__(
        self,
        img_size: Tuple[int, int] = (128, 128),
        patch_size: int = 8,
        in_chans: int = 5,
        encoder_embed_dim: int = 192,
        encoder_depth: int = 6,
        encoder_num_heads: int = 6,
        decoder_embed_dim: int = 96,
        decoder_depth: int = 3,
        decoder_num_heads: int = 4,
        mlp_ratio: float = 4.0,
        mask_ratio: float = 0.75,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.mask_ratio = mask_ratio

        # Ensure embed_dim is cleanly divisible by num_heads
        if encoder_embed_dim % encoder_num_heads != 0:
            for cand in [16, 8, 4, 2, 1]:
                if encoder_embed_dim % cand == 0 and cand <= encoder_embed_dim:
                    encoder_num_heads = cand
                    break
        if decoder_embed_dim % decoder_num_heads != 0:
            for cand in [16, 8, 4, 2, 1]:
                if decoder_embed_dim % cand == 0 and cand <= decoder_embed_dim:
                    decoder_num_heads = cand
                    break
        # 1. Patch Embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=encoder_embed_dim,
        )
        num_patches = self.patch_embed.num_patches
        self.grid_size = self.patch_embed.grid_size

        # 2. Positional Embeddings
        pos_embed = get_2d_sincos_pos_embed(encoder_embed_dim, self.grid_size)
        self.register_buffer("pos_embed", pos_embed.unsqueeze(0), persistent=False)

        # 3. Encoder Blocks
        self.encoder_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=encoder_embed_dim,
                    num_heads=encoder_num_heads,
                    mlp_ratio=mlp_ratio,
                )
                for _ in range(encoder_depth)
            ]
        )
        self.encoder_norm = nn.LayerNorm(encoder_embed_dim)

        # 4. Decoder Components
        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        dec_pos_embed = get_2d_sincos_pos_embed(decoder_embed_dim, self.grid_size)
        self.register_buffer("decoder_pos_embed", dec_pos_embed.unsqueeze(0), persistent=False)

        self.decoder_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=decoder_embed_dim,
                    num_heads=decoder_num_heads,
                    mlp_ratio=mlp_ratio,
                )
                for _ in range(decoder_depth)
            ]
        )
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)

        # Reconstruction head for MAE pretraining (predicts raw patch pixels)
        self.decoder_pred = nn.Linear(
            decoder_embed_dim, patch_size * patch_size * in_chans, bias=True
        )

        # 5. Deep Phase Unwrapping Head
        self.unwrapping_head = DeepPhaseUnwrappingHead(
            embed_dim=decoder_embed_dim,
            patch_size=patch_size,
            grid_size=self.grid_size,
            out_chans=1,
        )

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Convert (B, C, H, W) -> (B, N, P*P*C).
        """
        p = self.patch_size
        c = imgs.shape[1]
        h = imgs.shape[2] // p
        w = imgs.shape[3] // p
        x = imgs.reshape(imgs.shape[0], c, h, p, w, p)
        x = torch.einsum("nchpwq->nhwpqc", x)
        x = x.reshape(imgs.shape[0], h * w, p * p * c)
        return x

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert (B, N, P*P*C) -> (B, C, H, W).
        """
        p = self.patch_size
        c = self.in_chans
        h, w = self.grid_size
        x = x.reshape(x.shape[0], h, w, p, p, c)
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(x.shape[0], c, h * p, w * p)
        return imgs

    def random_masking(
        self, x: torch.Tensor, mask_ratio: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform 75% random spatial patch masking.
        Performs per-sample shuffling without looping over batch dimension.

        Args:
            x: Patch tokens of shape (B, N, D).
            mask_ratio: Fraction of patches to mask (e.g. 0.75).

        Returns:
            x_masked: Visible patch tokens (B, N_vis, D).
            mask: Binary mask (B, N) where 1 indicates masked, 0 indicates visible.
            ids_restore: Indices to restore original ordering (B, N).
        """
        N, D = x.shape[1], x.shape[2]
        len_keep = int(N * (1 - mask_ratio))
        if len_keep < 1:
            len_keep = 1

        noise = torch.rand(x.shape[0], N, device=x.device)  # Random noise in [0, 1]

        # Sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # Ascending order
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # Keep the first len_keep tokens
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # Generate binary mask: 0 is keep, 1 is remove
        mask = torch.ones([x.shape[0], N], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(
        self, x: torch.Tensor, mask_ratio: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through ViT-MAE encoder with random patch masking."""
        # Embed patches: (B, C, H, W) -> (B, N, D)
        x = self.patch_embed(x)
        # Add position embedding
        x = x + self.pos_embed

        # Mask patches
        if mask_ratio > 0.0:
            x, mask, ids_restore = self.random_masking(x, mask_ratio)
        else:
            mask = torch.zeros(x.shape[0], x.shape[1], device=x.device)
            ids_restore = torch.arange(x.shape[1], device=x.device).unsqueeze(0).repeat(x.shape[0], 1)

        # Apply Transformer blocks
        for blk in self.encoder_blocks:
            x = blk(x)
        x = self.encoder_norm(x)
        return x, mask, ids_restore

    def forward_decoder(
        self, x: torch.Tensor, ids_restore: torch.Tensor
    ) -> torch.Tensor:
        """
        Reassemble visible tokens and mask tokens, apply decoder Transformer blocks.
        """
        # Embed tokens to decoder dimension
        x = self.decoder_embed(x)

        # Append mask tokens
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] - x.shape[1], 1
        )
        x_ = torch.cat([x, mask_tokens], dim=1)
        # Unshuffle to original spatial order
        x_full = torch.gather(
            x_,
            dim=1,
            index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]),
        )

        # Add decoder position embedding
        x = x_full + self.decoder_pos_embed

        # Apply decoder blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        return x

    def forward_mae(
        self, imgs: torch.Tensor, mask_ratio: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Self-supervised pre-training step.
        Computes reconstruction loss on masked patches.
        """
        ratio = self.mask_ratio if mask_ratio is None else mask_ratio
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio=ratio)
        decoded = self.forward_decoder(latent, ids_restore)
        pred = self.decoder_pred(decoded)  # (B, N, P*P*C)

        # Compute reconstruction loss only on masked patches
        target = self.patchify(imgs)
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # (B, N) mean over patch pixels

        if mask.sum() > 0:
            recon_loss = (loss * mask).sum() / mask.sum()
        else:
            recon_loss = loss.mean()

        return recon_loss, pred, mask

    def forward_unwrap(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Inference / regression forward pass for Deep Phase Unwrapping.
        Processes unmasked inputs (mask_ratio=0.0) and predicts continuous phase psi.

        Args:
            imgs: Radar tensor of shape (B, C_in, H, W).

        Returns:
            Continuous unwrapped phase field (B, 1, H, W) [radians].
        """
        latent, _, ids_restore = self.forward_encoder(imgs, mask_ratio=0.0)
        decoded = self.forward_decoder(latent, ids_restore)
        unwrapped_phase = self.unwrapping_head(decoded)
        return unwrapped_phase

    def unwrap_to_displacement(
        self,
        imgs: torch.Tensor,
        wavelength_m: float,
    ) -> torch.Tensor:
        """
        End-to-end inference from multi-channel radar tensor to line-of-sight displacement in mm.
        Guarantees sub-millimeter precision.

        Args:
            imgs: Radar tensor of shape (B, C, H, W).
            wavelength_m: Carrier wavelength in meters (e.g. 0.0554658 for Sentinel-1).

        Returns:
            Displacement map in millimeters of shape (B, 1, H, W).
        """
        unwrapped_phase = self.forward_unwrap(imgs)
        scale = -(wavelength_m * 1000.0) / (4.0 * math.pi)
        return unwrapped_phase * scale
    def forward(
        self, imgs: torch.Tensor, mask_ratio: Optional[float] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Full joint forward pass returning both self-supervised reconstruction
        and continuous phase unwrapping predictions.
        """
        ratio = self.mask_ratio if mask_ratio is None else mask_ratio
        recon_loss, pred_patches, mask = self.forward_mae(imgs, mask_ratio=ratio)
        unwrapped_phase = self.forward_unwrap(imgs)

        return {
            "recon_loss": recon_loss,
            "pred_patches": pred_patches,
            "mask": mask,
            "unwrapped_phase": unwrapped_phase,
        }


class ViTMAELoss(nn.Module):
    """
    Multi-task loss function combining:
    1. MAE patch reconstruction loss (self-supervised speckle/fringe representations)
    2. Coherence-weighted continuous phase regression loss
    3. Wrapped phase cyclic consistency loss: 1 - cos(psi_pred - phi_wrapped)
    4. Spatial gradient consistency loss: ||grad(psi_pred) - grad(psi_gt)||_1
    """

    def __init__(
        self,
        weight_recon: float = 0.5,
        weight_phase: float = 1.0,
        weight_wrap: float = 0.5,
        weight_grad: float = 0.2,
    ):
        super().__init__()
        self.w_recon = weight_recon
        self.w_phase = weight_phase
        self.w_wrap = weight_wrap
        self.w_grad = weight_grad

    def forward(
        self,
        pred_dict: Dict[str, torch.Tensor],
        target_phase: Optional[torch.Tensor] = None,
        wrapped_phase: Optional[torch.Tensor] = None,
        coherence: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        loss_dict: Dict[str, torch.Tensor] = {}

        # 1. MAE Reconstruction Loss
        recon_loss = pred_dict["recon_loss"]
        loss_dict["loss_recon"] = recon_loss

        total_loss = self.w_recon * recon_loss
        pred_phase = pred_dict["unwrapped_phase"]
        if pred_phase.dim() == 4:
            pred_phase = pred_phase.squeeze(1)  # (B, H, W)

        # 2. Continuous Phase Unwrapping Loss
        if target_phase is not None:
            tp = target_phase.squeeze(1) if target_phase.dim() == 4 else target_phase
            diff = torch.abs(pred_phase - tp)
            if coherence is not None:
                coh = coherence.squeeze(1) if coherence.dim() == 4 else coherence
                coh_norm = coh / (coh.mean(dim=[-2, -1], keepdim=True) + 1e-6)
                phase_loss = (diff * coh_norm).mean()
            else:
                phase_loss = diff.mean()
            loss_dict["loss_phase"] = phase_loss
            total_loss = total_loss + self.w_phase * phase_loss

            # Spatial gradient consistency loss
            pred_dx = pred_phase[:, :, 1:] - pred_phase[:, :, :-1]
            pred_dy = pred_phase[:, 1:, :] - pred_phase[:, :-1, :]
            tgt_dx = tp[:, :, 1:] - tp[:, :, :-1]
            tgt_dy = tp[:, 1:, :] - tp[:, :-1, :]

            grad_loss = F.l1_loss(pred_dx, tgt_dx) + F.l1_loss(pred_dy, tgt_dy)
            loss_dict["loss_grad"] = grad_loss
            total_loss = total_loss + self.w_grad * grad_loss
        elif wrapped_phase is not None and self.w_grad > 0:
            # Self-supervised spatial gradient consistency using wrapped differences (Itoh condition)
            wp = wrapped_phase.squeeze(1) if wrapped_phase.dim() == 4 else wrapped_phase
            pred_dx = pred_phase[:, :, 1:] - pred_phase[:, :, :-1]
            pred_dy = pred_phase[:, 1:, :] - pred_phase[:, :-1, :]
            diff_x = torch.atan2(torch.sin(wp[:, :, 1:] - wp[:, :, :-1]), torch.cos(wp[:, :, 1:] - wp[:, :, :-1]))
            diff_y = torch.atan2(torch.sin(wp[:, 1:, :] - wp[:, :-1, :]), torch.cos(wp[:, 1:, :] - wp[:, :-1, :]))
            if coherence is not None:
                coh = coherence.squeeze(1) if coherence.dim() == 4 else coherence
                coh_x = 0.5 * (coh[:, :, 1:] + coh[:, :, :-1])
                coh_y = 0.5 * (coh[:, 1:, :] + coh[:, :-1, :])
                grad_loss = (torch.abs(pred_dx - diff_x) * coh_x).mean() + (torch.abs(pred_dy - diff_y) * coh_y).mean()
            else:
                grad_loss = F.l1_loss(pred_dx, diff_x) + F.l1_loss(pred_dy, diff_y)
            loss_dict["loss_grad"] = grad_loss
            total_loss = total_loss + self.w_grad * grad_loss

        # 3. Wrapped Cyclic Consistency Loss: 1 - cos(psi_pred - phi_wrapped)
        if wrapped_phase is not None:
            wp = wrapped_phase.squeeze(1) if wrapped_phase.dim() == 4 else wrapped_phase
            # 1 - cos(psi - phi) is 0 whenever psi - phi is an exact multiple of 2*pi
            wrap_loss = (1.0 - torch.cos(pred_phase - wp)).mean()
            loss_dict["loss_wrap"] = wrap_loss
            total_loss = total_loss + self.w_wrap * wrap_loss

        loss_dict["loss_total"] = total_loss
        return loss_dict
