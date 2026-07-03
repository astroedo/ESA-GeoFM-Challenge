# ESA Phi-Lab GeoFM Challenge — Embed2Heights

Multi-modal fusion of geospatial foundation model (GFM) embeddings for joint **land cover segmentation** (buildings, vegetation, water) and **nDSM height regression**, developed for the ESA Phi-Lab GeoFM Challenge.

**Final leaderboard score: 0.4130** (`notebook/ESA_v5_final.ipynb`, FusionNetV4).

The pipeline was trained entirely on free-tier Colab GPUs — every design decision below (caching strategy, checkpoint/resume, time budgeting, FP16 stability) exists because of that constraint.

---

## 1. Task

Given pre-computed embeddings from four foundation models over a 256×256 patch (10 m/px), predict a 4-channel output:

| Channel | Target |
|---|---|
| 0 | Building area fraction |
| 1 | Tree/vegetation area fraction |
| 2 | Water area fraction |
| 3 | nDSM height (m) |

Scoring is a fixed weighted combination:

```
Score = 0.25·mIoU_build + 0.15·mIoU_veg + 0.15·mIoU_water
      + 0.25·f(RMSE_height|build) + 0.20·f(RMSE_height|veg)
```

The weighting matters: buildings dominate (0.50 of the score comes from building IoU + building-height RMSE), which drove the class-balanced sampling and the decision to give buildings their own height decoder.

## 2. Input modalities

Two streams with fundamentally different geometry, fused inside the network:

**Spatial stream — dense 256×256 grids**
- **AlphaEarth** (64 ch): native-resolution features, sharp object boundaries. Backbone input.
- **Tessera** (128 ch): additional spectral/textural context at full resolution.

**Token stream — 16×16 ViT tokens**
- **TerraMind S2** (768 ch): scene-level semantics, long-range context.
- **THOR** (768 ch): complementary features that helped height estimation, but with unstable dynamic range (see §5).

## 3. Model iterations

Each notebook is a self-contained experiment; the progression was driven by observed failure modes, not architecture shopping.

| # | Notebook | Idea | Outcome |
|---|---|---|---|
| 1 | `train_alphaearth_only.ipynb` | Spatial stream only, plain UNet | Crisp building edges, but height RMSE poor — no scene context (TerraMind-only variant scored 0.124; building IoU 0.05, confirming tokens alone can't localize) |
| 2 | `train_dual_branch.ipynb` | Early fusion: TerraMind tokens 1×1-conv'd, bilinearly upsampled to 256², concatenated with AlphaEarth | Val mIoU 0.575, partial leaderboard **0.4112** |
| 3 | `train_attention_fusion.ipynb` | Replace static upsampling with cross-attention (pixels query tokens); gradient loss on height for sharper edges; checkpoint/auto-resume system | Sharper structural transitions; resume system became standard for all later runs |
| 4 | `train_fpn_head.ipynb` | Opposite hypothesis: treat GFMs as frozen backbone, train only a ~300k-param FPN head | Useful ablation — confirmed a larger fused decoder was worth the capacity on this dataset |
| 5 | `ESA_v5_final.ipynb` | **FusionNetV4** (below) + FP16 stability fixes | Leaderboard **0.4130** |

Note on v4→v5: the v4 run of the final architecture **never trained**. Every batch produced a non-finite loss, the NaN-skip guard silently dropped all of them, and the model evaluated at initialization (`trn=0.000, val=nan`). v5 is the same architecture with the numerical root cause fixed — see §5.

## 4. Final architecture — FusionNetV4

```
AlphaEarth (64) ─┐
Tessera   (128) ─┴─ concat (192ch, 256²)
        │
   SE-UNet encoder  (DoubleConv + Squeeze-Excitation, 3 downsamples)
        │
   Bottleneck 32² ── ASPP (dilation 1/6/12/18)
        │
        ├── + residual token injection  (TM‖THOR 1536ch → 1×1 conv → 256ch)
        └── cross-attention: 32² spatial queries attend over 256 tokens (4 heads)
        │
   UNet decoder with skip connections
        │
        ├── seg head (3ch logits)
        ├── height head — buildings ──┐  64-bin classification over log1p(h),
        └── height head — vegetation ─┘  soft-argmax → continuous height
```

Design choices worth explaining:

- **Dual height heads with per-pixel routing.** Building and vegetation height have different statistics and different error weights in the score. Each head sees the shared decoder features *plus its own class logit*, and the final height is routed per pixel by whichever class probability is higher. Each branch is supervised only where its class exists, weighted by class coverage.
- **Height as bin classification + soft-argmax**, not direct regression. Regressing height in log space through 64 bins (Huber on the soft-argmax value + cross-entropy on the bin index + a gradient-matching loss) was more stable than plain L2 and produces sharper height discontinuities at building edges.
- **ASPP dilation rates 1/6/12/18** at the 32² bottleneck cover effective receptive fields from ~8 m to ~144 m on the ground — roughly the range from single buildings to city blocks.
- **Both token-fusion mechanisms are kept**: residual injection gives every bottleneck location cheap access to scene context; cross-attention lets individual locations selectively pull from specific tokens. They are complementary, not redundant.

**Loss**: per-class focal BCE (γ=2, w=8 for buildings; γ=1 for veg/water) + weighted soft-IoU + the dual height losses above, combined with height weighted 3×.

## 5. FP16 numerical stability — the v4→v5 fix

The most instructive bug in the project.

**Symptom**: with `torch.amp.autocast`, every training batch produced `loss = NaN`. The skip-bad-batch guard masked this — training "ran" for a full session with zero weight updates.

**Root cause**: the PREP stage normalizes embeddings per-channel and stores them as fp16 clipped to ±60000. Embedding dimensions with near-zero variance (std ≈ 0) explode under `(x − μ)/σ` into ±60000 spikes. Those values are representable in fp16 (max ≈ 65504) — but the *first convolution's* accumulated products overflow, producing Inf, which BatchNorm turns into NaN across the whole activation map. Everything downstream is NaN.

**Fix and process**:
- `FEAT_CLAMP = 15.0`: clamp normalized features to ±15 at dataset load *and* at inference. Real signal after per-channel standardization lives well within ±15; only the degenerate-dimension spikes are removed. Applied at load time so the existing preprocessed zips didn't need rebuilding.
- A **diagnostic cell** that runs a single batch through the model and every individual loss term, reporting `finite / min / max` for each tensor — proving the loss is finite *before* committing GPU-hours. This turned a silent multi-hour failure into a 30-second check.
- Defense in depth for residual edge cases: NaN sanitization inside the height loss, gradient clipping, and scaler-aware `OneCycleLR` stepping (the scheduler only steps when the GradScaler didn't skip the optimizer step — otherwise LR and optimizer state desynchronize).

## 6. Training under Colab constraints

Free-tier Colab means preemptible runtimes, a slow network-mounted Drive filesystem, and a hard session clock. The pipeline treats all three as first-class requirements:

- **Two-tier data caching.** Token embeddings (TerraMind, THOR) and labels are small enough at fp16 to pin entirely in host RAM. The heavy spatial zips (AlphaEarth, Tessera) are staged once from Drive to the local SSD and read through a persistent per-worker zip handle with corruption-retry — avoiding the Drive I/O path entirely during training.
- **Crash-safe resume.** Full state (model, optimizer, scheduler, GradScaler, EMA shadow, best score, history) is checkpointed to Drive every epoch. After a disconnect, "Run all" resumes from the last epoch with no manual intervention. Drive staging itself retries with remount on the transport errors Colab's FUSE mount throws under load.
- **Time budgeting.** Training estimates the next epoch's duration from recent epochs and stops cleanly before the session limit rather than being killed mid-epoch.

**Regularization**: geometric augmentation (flips/rotations applied consistently across all modalities and labels), embedding-space Gaussian noise (σ=0.03) and 5% channel dropout on the pre-trained features — augmenting in latent space since raw imagery is not available — plus a `WeightedRandomSampler` giving 3× weight to patches with meaningful building coverage. **Optimization**: AdamW, OneCycleLR, EMA of weights (decay 0.999); the EMA model is evaluated at the end and kept if it beats the best raw checkpoint.

## 7. Inference

- **Per-class threshold sweep** on validation (IoU-optimal threshold per class), then a piecewise-linear probability remap so submitted probabilities are calibrated around 0.5 at the chosen threshold.
- **6-view TTA** (identity, 3 flips, 2 rotations) with height routed per view.
- **Multi-seed ensembling** (optional cell): discovers all trained seed directories, averages predictions across models × 6 views × 3 scales (0.75/1.0/1.25).

## 8. Repository structure

```
notebook/
├── ESA_v5_final.ipynb          # production pipeline: FusionNetV4, FP16 fixes,
│                               #   caching, TTA, thresholding, ensemble
├── train_attention_fusion.ipynb  # cross-attention fusion + gradient loss + resume system
├── train_dual_branch.ipynb       # early convolutional fusion baseline (LB 0.4112)
├── train_fpn_head.ipynb          # frozen-backbone ablation (~300k-param FPN head)
└── train_alphaearth_only.ipynb   # single-modality spatial baseline
```

## 9. Reproducing

The final notebook is organized as lettered sections (A: one-time preprocessing → K: submission sanity check) and is designed to be re-run top-to-bottom after any interruption. Set `SEED` in section B and re-run end-to-end per ensemble member; section L builds the ensembled submission from all trained seeds.
