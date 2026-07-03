# ESA GeoFM Challenge — Embed2Heights

Predicting building/vegetation/water segmentation and nDSM height from geospatial
foundation model (GFM) embeddings.

**Current best: v5 — leaderboard score 0.413** (`notebooks/ESA_v5_final.ipynb`)

## Task

Given precomputed embeddings from multiple GFMs, predict for each 256×256 patch:

| Band | Content |
|------|---------|
| 0 | Building fraction (%) |
| 1 | Vegetation fraction (%) |
| 2 | Water fraction (%) |
| 3 | nDSM height (metres) |

**Scoring:** mIoU_buildings (25%) · mIoU_trees (15%) · mIoU_water (15%) · RMSE_H_buildings (25%) · RMSE_H_veg (20%)

## Input embeddings

| Model | Channels | Resolution | Notes |
|-------|----------|------------|-------|
| AlphaEarth | 64 | 256×256 | spatial branch |
| Tessera | 128 | 256×256 | spatial branch |
| TerraMind S1/S2 | 768 | 16×16 | token branch |
| THOR S1/S2 | 768 | 16×16 | S2 range ±18000, needs normalisation |

---

## v5 — current best model

Notebook: `notebooks/ESA_v5_final.ipynb` · **Score 0.413**

### Architecture — `FusionNetV4` (SE-UNet + ASPP + cross-attention, dual height heads)

Two input branches:

- **Spatial branch** (256×256): AlphaEarth (64ch) + Tessera (128ch) concatenated → 192ch,
  encoded by a U-Net encoder with `DoubleConv` blocks + **SE (squeeze-excitation)** channel
  attention at every stage.
- **Token branch** (16×16): TerraMind + THOR concatenated → 1536ch treated as 256 tokens.

Fusion & decoding:

- **ASPP** at the bottleneck (dilation rates 1/6/12/18 ≈ 8–144 m context at 10 m/px).
- **Cross-attention**: bottleneck spatial queries attend over the 256 TM+THOR tokens
  (multi-head, 4 heads, dim 256).
- U-Net decoder with skip connections → **sigmoid segmentation head** (3ch) +
  **dual height heads** (separate building / vegetation branches, each predicting height
  bins with soft-argmax + a combined fallback head). Height trained in `log1p` space.

### Loss

- Segmentation: per-class **focal BCE** (building γ=2 w=8, veg/water γ=1 w=3) +
  weighted **soft-IoU** (2.0 build / 1.5 veg / 0.5 water).
- Height: per-branch **Huber + bin cross-entropy + gradient loss**, weighted by class
  coverage; combined `1.5·L_hb + 1.5·L_hv + 0.5·L_hc`, total height weight ×3.

### Training

- Colab T4, AMP (fp16 autocast), batch 16, 50 epochs, ~11 h budget.
- AdamW (wd 1e-2), **OneCycleLR max_lr 1.5e-4** (3e-4 diverged at ~epoch 31 under AMP),
  grad clip 0.5, **EMA** (decay 0.999) — best of raw/EMA kept per epoch.
- Augmentation: flips + 90° rotations (all tensors, labels aligned) + embedding-space
  gaussian noise (σ=0.03) + channel dropout (p=0.05).
- Checkpoints saved to Drive every epoch → **auto-resumes after Colab disconnects**
  ("just Run all").

### Inference

- Per-class **threshold sweep** on val (saved to `final_thresholds.json`).
- **6-view TTA** at submission time; optional **multi-seed ensemble** cell
  (train seeds 42/123/2024, averages predictions + thresholds).
- `expm1` on height output; sanity-check cell validates 946 files of shape (4, 256, 256).

### The v4→v5 bug (why v4 never learned)

v4 had the same architecture but scored as if untrained: `trn=0.000, val=nan`. Root
cause: PREP normalises embeddings and clips to ±60000 fp16; **near-constant embedding
dims (std≈0) blow up to ±60000 spikes**, the first conv overflows fp16 under autocast
(max ≈ 65504) → Inf → BatchNorm → NaN loss on every batch → the NaN-skip guard silently
dropped *all* batches. v5 fixes:

- `FEAT_CLAMP = 15` — clamp normalised features to ±15 at load and inference
  (works on existing zips, no re-PREP needed).
- Diagnostic cell (§D2) runs one batch and verifies every loss term is finite
  **before** spending GPU hours.
- NaN-safe height loss, `nan_to_num` on all inputs.

---

## Experiment log

| Version | Notebook | Approach | Score | Outcome |
|---------|----------|----------|-------|---------|
| v0 | `ESA_final_fusion_kaggle-2` | SimpleFusionHead: TerraMind S2 only → 1×1 convs → bilinear upsample | 0.1241 | Baseline. Building IoU only 0.053 — 16×16 input too coarse for buildings |
| v1 | `train_fpn_head` | FPN-style decoder head | — | Superseded |
| v2 | `train_attention_fusion` | Attention-based multi-embedding fusion | — | Superseded |
| v3 | `train_unet_attn_fusion` / `-3` | U-Net decoder + attention fusion | — | Superseded |
| v4 | (same arch as v5) | SE-UNet + ASPP + cross-attn, dual height heads | 0.4112* | **Never trained** — fp16 overflow → NaN on every batch, weights ~stuck at init (see bug post-mortem above) |
| **v5** | `ESA_v5_final` | v4 + FEAT_CLAMP, NaN-safe losses, diagnostics, embedding augmentation, TTA/ensemble | **0.413** | **Current best** |

\* partial/init-equivalent weights.

---

## Environment & data

Google Colab (T4 GPU), data on Google Drive at `MyDrive/ESA_Challenge/`.
Dataset via `eotdl datasets get embed2heights -v 1 -f` (re-run each session — auth token
at `/root/.cache/eotdl/creds.json`, key `id_token`).

v5 pre-packs everything into npy zips (`kaggle_transfer_v5/`) and stages them to local
SSD (`/content/tmp`) before training — Drive I/O is the #1 crash source.

### Gotchas / hard-won fixes

- **fp16 overflow** from normalised-embedding spikes → clamp features to ±15 (see v4 bug)
- `num_workers=0` fallback if DataLoader workers OOM (v5 runs with 2)
- Embeddings contain NaN pixels → `np.nansum` in normalisation stats, `nan_to_num` at load
- One rogue 255×255 file exists → always `F.interpolate` guard to 256×256
- Patch IDs: extract 4-digit ID from filename with `re.search(r'_(\d{4})_', stem)`
- Catalog saved permanently at `catalog.v1.parquet` — never read from `/root/.cache/`
- eotdl test download is two-step: presigned URL from API, then GET the URL
- Copy test files to local SSD before inference to avoid Drive I/O crashes
- Height trained in `log1p` space → `expm1` before writing predictions

## Submission format

`submission.zip` containing `predictions/{ID}_MM_2022.npy`, each of shape `(4, 256, 256)`:
build%, veg%, water%, height (metres).

## Next steps

- Train remaining ensemble seeds (123, 2024) and submit the multi-seed ensemble
- Water val→test gap: embedding augmentation added in v5, monitor whether it closes
