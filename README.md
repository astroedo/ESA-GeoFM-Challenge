# ESA GeoFM Challenge — Embed2Heights

Predicting building/vegetation/water segmentation and nDSM height from geospatial
foundation model (GFM) embeddings.

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

| Model | Channels | Resolution |
|-------|----------|------------|
| AlphaEarth | 64 | 256×256 |
| Tessera | 128 | 256×256 |
| TerraMind S1/S2 | 768 | 16×16 |
| THOR S1/S2 | 768 | 16×16 (S2 range ±18000, needs normalisation) |

## Notebooks

- `notebooks/train_fpn_head.ipynb` — FPN-style decoder head
- `notebooks/train_attention_fusion.ipynb` — attention-based multi-embedding fusion
- `notebooks/train_unet_attn_fusion.ipynb` / `-3` — U-Net decoder with attention fusion
- `notebooks/building_height-2.ipynb` — height regression experiments
- `notebooks/ESA_final_fusion_kaggle-2.ipynb` — final fusion pipeline (Kaggle)

## Environment

Google Colab (T4 GPU), data on Google Drive at `MyDrive/ESA_Challenge/`.
Dataset fetched via `eotdl datasets get embed2heights -v 1 -f`.

### Gotchas / hard-won fixes

- `num_workers=0` in all DataLoaders — worker processes OOM otherwise
- Embeddings contain NaN pixels → use `np.nansum` in normalisation stats
- One rogue 255×255 file exists → always `F.interpolate` guard to 256×256
- Copy test files to local SSD (`/content/test_s2/`) before inference to avoid Drive I/O crashes
- Height target trained in `log1p` space → `expm1` before writing predictions

## Submission format

`submission.zip` containing `predictions/{ID}_MM_2022.npy`, each of shape `(4, 256, 256)`:
build%, veg%, water%, height (metres).

## Current status

Baseline `SimpleFusionHead` (TerraMind S2 only → 1×1 conv stack → bilinear upsample):
score **0.1241** (val_loss 0.5547). Next step: two-branch fusion with AlphaEarth to
improve building IoU.
