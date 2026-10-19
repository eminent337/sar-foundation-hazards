# 🛰️ SAR-Foundation-Hazards (v1.0.0 Specification)
### Multi-Sensor Self-Supervised SAR/InSAR Foundation Architecture & Kinematic Deformation Transformer for Mining High-Walls and Critical Infrastructure

## 1. Technical & Commercial Mandate
Build a commercial-grade, production-quality PyTorch library that implements a Self-Supervised Masked Autoencoder (MAE) and Temporal Deformation Transformer for complex Synthetic Aperture Radar (SAR) and interferometric (InSAR) datasets.
- Incorporates the architectural lessons of **NASA-IBM Prithvi Geospatial Foundation Model**, **TorchGeo**, and **DeepInSAR**.
- Sub-millimeter kinematic displacement precision with temporal coherence weighting ($\gamma$).
- Multi-Sensor / Multi-Frequency support: Dual-polarization Sentinel-1 (VV/VH, C-band) + NASA-ISRO NISAR (L-band / S-band) ingest tensor pipeline.
- Deep InSAR Phase Unwrapping & Coherence Reconstruction via Vision Transformer (ViT) blocks.
- Real empirical benchmarks: Chirano Gold Mine open-pit slopes (Ghana) & Western Canadian transportation corridor slopes (BC/Alberta).
- 100% automated test coverage with `pytest`.
- Standalone CLI execution (`run_pipeline.py`).

## 2. Core Modules in `src/sar_core/`:
1. `sar_tensor_pipeline.py`:
   - Dual-pol (VV, VH) C-band and L-band multi-temporal complex SLC/interferogram tensor loader.
   - Polarimetric feature engineering: Span, Cross-Ratio ($VH/VV$), Radar Vegetation Index ($RVI$).
   - Normalized phase wrapped inputs $[-\pi, +\pi]$ and temporal coherence maps $\gamma \in [0, 1]$.
2. `vit_mae_foundation.py`:
   - Vision Transformer Masked Autoencoder (ViT-MAE) tailored for complex SAR phase and amplitude:
     * Patch embedding for multi-channel radar tensors: $(B, C, H, W) \rightarrow (B, N, D)$.
     * 75% random spatial masking to learn self-supervised physical representations of decorrelated speckle and fringe patterns.
     * Transformer encoder-decoder architecture with sinusoidal positional embeddings.
   - Deep Phase Unwrapping Head: Direct regression of continuous phase $\psi = \phi + 2\pi k$ overcoming $2\pi$ phase ambiguities.
3. `temporal_kinematics.py`:
   - Spatio-temporal Transformer / LSTM velocity model tracking displacement series:
     $$d(t) = v \cdot t + \frac{1}{2} a \cdot t^2 + \sum A_i \sin(\omega_i t + \phi_i) + \epsilon$$
   - Early-warning kinematic anomaly detector: Detects non-linear tertiary accelerating creep on open-pit slopes prior to catastrophic high-wall collapse (Fukuzono / Voight failure criterion).
4. `benchmark_evaluator.py`:
   - High-precision validation against ground-truth GPS prisms and simulated slope failures.
   - Metric suite: Phase Unwrapping Root Mean Squared Error (RMSE in radians), Velocity Error (mm/year), Matthews Correlation Coefficient (MCC), and Area Under PR Curve (AUC-PR).
5. `visualizer.py`:
   - 300 DPI 4-panel publication diagnostic plot:
     * Panel A: Wrapped interferogram vs Transformer unwrapped phase field.
     * Panel B: Coherence map and reconstruction error residual.
     * Panel C: Time-series displacement curves across pit crest vs toe.
     * Panel D: Kinematic early-warning acceleration risk map.
6. `report_generator.py`:
   - Automated engineering geotechnical stability certificate (Markdown and CSV).

## 3. Package Structure:
```
sar-foundation-hazards/
├── pyproject.toml
├── requirements.txt
├── README.md
├── LICENSE
├── run_pipeline.py
├── SPECIFICATION.md
├── src/
│   └── sar_core/
│       ├── __init__.py
│       ├── sar_tensor_pipeline.py
│       ├── vit_mae_foundation.py
│       ├── temporal_kinematics.py
│       ├── benchmark_evaluator.py
│       ├── visualizer.py
│       └── report_generator.py
├── tests/
│   ├── __init__.py
│   ├── test_tensor_pipeline.py
│   ├── test_vit_mae.py
│   ├── test_temporal_kinematics.py
│   └── test_benchmark.py
├── notebooks/
│   └── 01_sar_foundation_walkthrough.ipynb
└── output/
```
