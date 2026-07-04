#!/usr/bin/env python
"""
Windows/VSCode optimized trainer for AlphaEarth + TerraMind S2 fusion.

Expected data layout:

  D:/ESA_Challenge/
    train/alphaearth_emb/*.tif
    train/terramind_s2_emb/*.tif
    train/labels/*.tif
    test/alphaearth_emb/*.tif
    test/terramind_test_s2_emb/*.tif
    norm_stats.npy

Example:

  python train_fusion_windows_vscode.py ^
    --data-root D:/ESA_Challenge ^
    --work-dir D:/ESA_Challenge/_fusion_work ^
    --mode all ^
    --epochs 50 ^
    --batch-size 8 ^
    --num-workers 4 ^
    --tta
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import re
import time
import warnings
import zipfile
from contextlib import nullcontext
from dataclasses import dataclass
from multiprocessing import freeze_support
from pathlib import Path
from typing import Iterable

import numpy as np
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)


@dataclass(frozen=True)
class NormStats:
    ae_mean: np.ndarray
    ae_std: np.ndarray
    tm_mean: np.ndarray
    tm_std: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train AlphaEarth + TerraMind fusion efficiently on Windows."
    )
    parser.add_argument("--data-root", type=Path, default=Path("D:/ESA_Challenge"))
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--mode", choices=["prepare", "train", "infer", "all"], default="all")

    parser.add_argument("--ae-train", type=Path)
    parser.add_argument("--tm-train", type=Path)
    parser.add_argument("--label-dir", type=Path)
    parser.add_argument("--ae-test", type=Path)
    parser.add_argument("--tm-test", type=Path)
    parser.add_argument("--norm-stats", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--submission", type=Path)

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=max(2, min(6, (os.cpu_count() or 4) // 2)))
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--max-train", type=int, default=0, help="0 uses all matched training patches.")
    parser.add_argument(
        "--stats-sample",
        type=int,
        default=128,
        help="Files per embedding used when norm_stats.npy is missing. 0 computes exact stats.",
    )

    parser.add_argument("--fuse-ch", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--height-lambda", type=float, default=1.5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--min-lr-ratio", type=float, default=0.05)

    parser.add_argument("--cache-ae", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cache-tm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument("--input-dtype", choices=["auto", "float16", "float32"], default="auto")
    parser.add_argument("--mmap-cache", action="store_true")

    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--channels-last", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> argparse.Namespace:
    root = args.data_root
    stage = root / "_eotdl_stage" / "embed2heights" / "data"
    args.work_dir = args.work_dir or root / "_fusion_work"
    args.ae_train = args.ae_train or first_existing(
        root / "train" / "alphaearth_emb",
        stage / "train" / "alphaearth_emb",
    )
    args.tm_train = args.tm_train or first_existing(
        root / "train" / "terramind_s2_emb",
        stage / "train" / "terramind_s2_emb",
    )
    args.label_dir = args.label_dir or first_existing(
        root / "train" / "labels",
        stage / "train" / "labels",
    )
    args.ae_test = args.ae_test or first_existing(
        root / "test" / "alphaearth_emb",
        root / "test" / "alphaearth_test_emb",
        stage / "test" / "alphaearth_emb",
        stage / "test" / "alphaearth_test_emb",
    )
    if not args.ae_test.exists():
        args.ae_test = root / "test" / "alphaearth_test_emb"
    args.tm_test = args.tm_test or first_existing(
        root / "test" / "terramind_test_s2_emb",
        stage / "test" / "terramind_test_s2_emb",
    )
    args.norm_stats = args.norm_stats or first_existing(
        root / "norm_stats.npy",
        stage / "norm_stats.npy",
        args.work_dir / "norm_stats_fusion.npy",
    )
    args.checkpoint = args.checkpoint or args.work_dir / "best_model_fusion_windows.pth"
    args.submission = args.submission or args.work_dir / "submission_fusion_windows.zip"
    return args


def first_existing(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")


def require_dir(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")


def patch_id(stem: str) -> str | None:
    match = re.search(r"_(\d{4})_", stem)
    if match:
        return match.group(1)
    numbers = re.findall(r"\d{4}", stem)
    return numbers[0] if numbers else None


def clean_submission_stem(stem: str) -> str:
    clean = re.sub(r"^s2_", "", stem)
    clean = re.sub(r"_embeddings$", "", clean)
    return clean


def list_tifs(path: Path) -> list[Path]:
    return sorted(path.glob("*.tif"))


def list_nonempty_tifs(path: Path) -> list[Path]:
    return [p for p in list_tifs(path) if p.stat().st_size > 1024]


def load_tif(path: Path, target_hw: int | None = None) -> np.ndarray:
    with rasterio.open(path) as src:
        arr = src.read().astype(np.float32)
    if target_hw and (arr.shape[-2] != target_hw or arr.shape[-1] != target_hw):
        tensor = torch.from_numpy(arr).unsqueeze(0)
        tensor = F.interpolate(
            tensor, size=(target_hw, target_hw), mode="bilinear", align_corners=False
        )
        arr = tensor.squeeze(0).numpy()
    return arr


def load_norm_stats(path: Path) -> NormStats:
    if not path.exists():
        raise FileNotFoundError(f"Missing norm stats: {path}")
    stats = np.load(path, allow_pickle=True).item()
    for key in ("alphaearth_emb", "terramind_s2_emb"):
        if key not in stats:
            raise KeyError(f"Missing '{key}' in {path}")
    return NormStats(
        ae_mean=stats["alphaearth_emb"]["mean"].reshape(-1, 1, 1).astype(np.float32),
        ae_std=stats["alphaearth_emb"]["std"].reshape(-1, 1, 1).astype(np.float32),
        tm_mean=stats["terramind_s2_emb"]["mean"].reshape(-1, 1, 1).astype(np.float32),
        tm_std=stats["terramind_s2_emb"]["std"].reshape(-1, 1, 1).astype(np.float32),
    )


def compute_channel_stats(folder: Path, target_hw: int | None, max_files: int) -> tuple[np.ndarray, np.ndarray]:
    files = list_nonempty_tifs(folder)
    if not files:
        raise RuntimeError(f"No tif files found for norm stats: {folder}")

    total = None
    total_sq = None
    count = None
    used = 0
    skipped = 0
    for i, path in enumerate(files, 1):
        if max_files > 0 and used >= max_files:
            break
        try:
            arr = load_tif(path, target_hw=target_hw).astype(np.float32)
        except Exception as exc:
            skipped += 1
            print(f"  skip stats file {path.name}: {exc}")
            continue
        channels = arr.shape[0]
        flat = arr.reshape(channels, -1).astype(np.float64)
        valid = np.isfinite(flat)
        if total is None:
            total = np.zeros(channels, dtype=np.float64)
            total_sq = np.zeros(channels, dtype=np.float64)
            count = np.zeros(channels, dtype=np.float64)
        total += np.where(valid, flat, 0.0).sum(axis=1)
        total_sq += np.where(valid, flat * flat, 0.0).sum(axis=1)
        count += valid.sum(axis=1)
        used += 1
        if used % 50 == 0 or i == len(files) or (max_files > 0 and used == max_files):
            print(f"  stats {folder.name}: used={used} scanned={i}/{len(files)} skipped={skipped}")

    assert total is not None and total_sq is not None and count is not None
    safe_count = np.maximum(count, 1)
    mean = total / safe_count
    var = np.maximum(total_sq / safe_count - mean**2, 1e-12)
    return mean.astype(np.float32), np.sqrt(var).astype(np.float32)


def load_or_compute_norm_stats(args: argparse.Namespace) -> NormStats:
    if args.norm_stats.exists():
        return load_norm_stats(args.norm_stats)

    sample = args.stats_sample
    print(f"Missing norm stats; computing sampled stats -> {args.norm_stats}")
    print("Use --stats-sample 0 for exact stats, but sampled stats are faster for quick fusion.")
    ae_mean, ae_std = compute_channel_stats(args.ae_train, target_hw=256, max_files=sample)
    tm_mean, tm_std = compute_channel_stats(args.tm_train, target_hw=None, max_files=sample)
    stats = {
        "alphaearth_emb": {"mean": ae_mean, "std": ae_std},
        "terramind_s2_emb": {"mean": tm_mean, "std": tm_std},
    }
    args.norm_stats.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.norm_stats, stats)
    return load_norm_stats(args.norm_stats)


def normalize(arr: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    arr = (arr - mean) / (std + 1e-6)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def choose_input_dtype(args: argparse.Namespace, device: torch.device) -> np.dtype:
    if args.input_dtype == "float32":
        return np.float32
    if args.input_dtype == "float16":
        if device.type != "cuda" or not args.amp:
            print("input-dtype=float16 requested but CUDA AMP is off; using float32.")
            return np.float32
        return np.float16
    return np.float16 if device.type == "cuda" and args.amp else np.float32


def prepare_cache(
    source_dir: Path,
    cache_dir: Path,
    mean: np.ndarray,
    std: np.ndarray,
    target_hw: int | None,
    label: str,
    force: bool = False,
) -> Path:
    require_dir(source_dir, label)
    cache_dir.mkdir(parents=True, exist_ok=True)
    files = list_tifs(source_dir)
    if not files:
        raise RuntimeError(f"No tif files found in {source_dir}")

    todo = []
    for path in files:
        out = cache_dir / f"{path.stem}.npy"
        if force or not out.exists():
            todo.append((path, out))

    print(f"{label} cache: {len(todo)} missing / {len(files)} total -> {cache_dir}")
    start = time.time()
    for i, (src, dst) in enumerate(todo, 1):
        arr = load_tif(src, target_hw=target_hw)
        arr = normalize(arr, mean, std).astype(np.float16)
        np.save(dst, arr)
        if i % 100 == 0 or i == len(todo):
            elapsed = max(time.time() - start, 1e-6)
            print(f"  {label}: {i}/{len(todo)} ({i / elapsed:.1f} files/s)")
    return cache_dir


class FusionDataset(Dataset):
    def __init__(
        self,
        ae_files: list[Path],
        tm_files: list[Path],
        label_files: list[Path],
        stats: NormStats,
        input_dtype: np.dtype,
        augment: bool,
        mmap_cache: bool,
    ) -> None:
        self.ae_files = ae_files
        self.tm_files = tm_files
        self.label_files = label_files
        self.stats = stats
        self.input_dtype = input_dtype
        self.augment = augment
        self.mmap_cache = mmap_cache

    def __len__(self) -> int:
        return len(self.ae_files)

    def _load_array(
        self,
        path: Path,
        mean: np.ndarray,
        std: np.ndarray,
        target_hw: int | None,
    ) -> np.ndarray:
        if path.suffix.lower() == ".npy":
            mmap = "r" if self.mmap_cache else None
            arr = np.load(path, mmap_mode=mmap)
            return np.asarray(arr, dtype=self.input_dtype)
        arr = normalize(load_tif(path, target_hw=target_hw), mean, std)
        return arr.astype(self.input_dtype, copy=False)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        try:
            ae = self._load_array(self.ae_files[idx], self.stats.ae_mean, self.stats.ae_std, 256)
            tm = self._load_array(self.tm_files[idx], self.stats.tm_mean, self.stats.tm_std, None)
            label = load_tif(self.label_files[idx], target_hw=256)
        except Exception as exc:
            print(f"Bad sample index={idx}: {exc}")
            ae = np.zeros((64, 256, 256), dtype=self.input_dtype)
            tm = np.zeros((768, 16, 16), dtype=self.input_dtype)
            label = np.zeros((4, 256, 256), dtype=np.float32)

        seg = label[:3].clip(0, 1).astype(np.float32, copy=False)
        height = np.log1p(label[3:4].clip(0, None)).astype(np.float32, copy=False)

        ae_t = torch.from_numpy(np.asarray(ae))
        tm_t = torch.from_numpy(np.asarray(tm))
        seg_t = torch.from_numpy(seg)
        height_t = torch.from_numpy(height)

        if self.augment:
            if random.random() > 0.5:
                ae_t = torch.flip(ae_t, [-1])
                tm_t = torch.flip(tm_t, [-1])
                seg_t = torch.flip(seg_t, [-1])
                height_t = torch.flip(height_t, [-1])
            if random.random() > 0.5:
                ae_t = torch.flip(ae_t, [-2])
                tm_t = torch.flip(tm_t, [-2])
                seg_t = torch.flip(seg_t, [-2])
                height_t = torch.flip(height_t, [-2])
            k = random.randint(0, 3)
            if k:
                ae_t = torch.rot90(ae_t, k, dims=[-2, -1])
                tm_t = torch.rot90(tm_t, k, dims=[-2, -1])
                seg_t = torch.rot90(seg_t, k, dims=[-2, -1])
                height_t = torch.rot90(height_t, k, dims=[-2, -1])

        return ae_t, tm_t, seg_t, height_t


def make_group_norm(channels: int) -> nn.GroupNorm:
    groups = min(8, channels)
    while channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class ConvGnSilu(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dropout: float = 0.0) -> None:
        super().__init__()
        pad = kernel_size // 2
        layers: list[nn.Module] = [
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=pad, bias=False),
            make_group_norm(out_ch),
            nn.SiLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            ConvGnSilu(channels, channels, dropout=dropout),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            make_group_norm(channels),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class ChannelSE(nn.Module):
    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.net(x)


class TerraAlphaGatedFusion(nn.Module):
    def __init__(self, ae_ch: int = 64, tm_ch: int = 768, fuse_ch: int = 96, dropout: float = 0.05) -> None:
        super().__init__()
        self.ae_stem = nn.Sequential(
            ConvGnSilu(ae_ch, fuse_ch, dropout=dropout),
            ResidualBlock(fuse_ch, dropout=dropout),
            ResidualBlock(fuse_ch, dropout=dropout),
        )
        self.tm_stem = nn.Sequential(
            nn.Conv2d(tm_ch, 256, 1, bias=False),
            make_group_norm(256),
            nn.SiLU(inplace=True),
            ResidualBlock(256, dropout=dropout),
            nn.Conv2d(256, fuse_ch, 1, bias=False),
            make_group_norm(fuse_ch),
            nn.SiLU(inplace=True),
        )
        self.gate = nn.Sequential(
            ConvGnSilu(fuse_ch * 2, fuse_ch, dropout=dropout),
            nn.Conv2d(fuse_ch, fuse_ch, 1),
            nn.Sigmoid(),
        )
        self.decoder = nn.Sequential(
            ConvGnSilu(fuse_ch * 2, fuse_ch, dropout=dropout),
            ResidualBlock(fuse_ch, dropout=dropout),
            ChannelSE(fuse_ch),
            ResidualBlock(fuse_ch, dropout=dropout),
        )
        self.seg_head = nn.Sequential(
            ConvGnSilu(fuse_ch, fuse_ch // 2, dropout=dropout),
            nn.Conv2d(fuse_ch // 2, 3, 1),
        )
        self.height_head = nn.Sequential(
            ConvGnSilu(fuse_ch, fuse_ch // 2, dropout=dropout),
            nn.Conv2d(fuse_ch // 2, 1, 1),
        )

    def forward(self, ae: torch.Tensor, tm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ae_feat = self.ae_stem(ae)
        tm_feat = self.tm_stem(tm)
        tm_feat = F.interpolate(tm_feat, size=ae_feat.shape[-2:], mode="bilinear", align_corners=False)
        gate = self.gate(torch.cat([ae_feat, tm_feat], dim=1))
        fused = torch.cat([ae_feat * (1.0 + gate), tm_feat * gate], dim=1)
        feat = self.decoder(fused)
        seg_logits = self.seg_head(feat)
        height_log1p = F.softplus(self.height_head(feat))
        return seg_logits, height_log1p


def seg_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    weights = torch.tensor([4.0, 1.0, 1.0], device=logits.device).view(1, 3, 1, 1)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (bce * weights).mean()


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    dims = (0, 2, 3)
    inter = (prob * target).sum(dim=dims)
    denom = prob.sum(dim=dims) + target.sum(dim=dims)
    dice = (2 * inter + eps) / (denom + eps)
    weights = torch.tensor([2.0, 1.0, 1.0], device=logits.device)
    return (weights * (1.0 - dice)).sum() / weights.sum()


def height_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    weight = 1.0 + (target > 0).float() * 2.0
    loss = F.huber_loss(pred, target, delta=1.0, reduction="none")
    return (weight * loss).mean()


def total_loss(
    seg_logits: torch.Tensor,
    height_log1p: torch.Tensor,
    seg_target: torch.Tensor,
    height_target: torch.Tensor,
    height_lambda: float,
) -> torch.Tensor:
    return (
        seg_loss(seg_logits, seg_target)
        + dice_loss(seg_logits, seg_target)
        + height_lambda * height_loss(height_log1p, height_target)
    )


class MetricAccumulator:
    def __init__(self) -> None:
        self.inter = torch.zeros(3, dtype=torch.float64)
        self.union = torch.zeros(3, dtype=torch.float64)
        self.height_sse = 0.0
        self.height_n = 0

    @torch.no_grad()
    def update(
        self,
        seg_logits: torch.Tensor,
        seg_target: torch.Tensor,
        height_log1p: torch.Tensor,
        height_target: torch.Tensor,
    ) -> None:
        pred = (torch.sigmoid(seg_logits.float()) > 0.5).cpu()
        target = (seg_target.float() > 0.5).cpu()
        self.inter += (pred & target).sum(dim=(0, 2, 3)).to(torch.float64)
        self.union += (pred | target).sum(dim=(0, 2, 3)).to(torch.float64)

        height_m = torch.expm1(height_log1p.float()).clamp(0).cpu()
        target_m = torch.expm1(height_target.float()).clamp(0).cpu()
        diff = height_m - target_m
        self.height_sse += float((diff * diff).sum().item())
        self.height_n += diff.numel()

    def compute(self) -> tuple[float, float, float, float, float]:
        iou = self.inter / (self.union + 1e-6)
        rmse = math.sqrt(self.height_sse / max(self.height_n, 1))
        return iou.mean().item(), iou[0].item(), iou[1].item(), iou[2].item(), rmse


def amp_context(device: torch.device, enabled: bool):
    if not enabled or device.type != "cuda":
        return nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type="cuda", enabled=True)
    return torch.cuda.amp.autocast(enabled=True)


def make_scaler(device: torch.device, enabled: bool):
    enabled = enabled and device.type == "cuda"
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def unwrap_model(model: nn.Module) -> nn.Module:
    return model._orig_mod if hasattr(model, "_orig_mod") else model


def index_by_patch(files: Iterable[Path]) -> dict[str, Path]:
    indexed = {}
    for path in files:
        pid = patch_id(path.stem)
        if pid:
            indexed[pid] = path
    return indexed


def split_ids(ids: list[str], val_frac: float, seed: int) -> tuple[list[str], list[str]]:
    rng = random.Random(seed)
    ids = list(ids)
    rng.shuffle(ids)
    n_val = max(1, int(len(ids) * val_frac)) if len(ids) > 1 else 0
    return ids[n_val:], ids[:n_val]


def move_image_tensor(tensor: torch.Tensor, device: torch.device, channels_last: bool) -> torch.Tensor:
    tensor = tensor.to(device, non_blocking=True)
    if channels_last and device.type == "cuda":
        tensor = tensor.contiguous(memory_format=torch.channels_last)
    return tensor


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    workers: int,
    device: torch.device,
) -> DataLoader:
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
    }
    if workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
        if os.name == "nt":
            kwargs["multiprocessing_context"] = "spawn"
    return DataLoader(dataset, **kwargs)


def save_history(path: Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def checkpoint_load(model: nn.Module, checkpoint: Path, device: torch.device) -> dict:
    try:
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    unwrap_model(model).load_state_dict(state)
    return ckpt if isinstance(ckpt, dict) else {"model": state}


def build_train_sources(args: argparse.Namespace, stats: NormStats) -> tuple[Path, Path]:
    ae_source = args.ae_train
    tm_source = args.tm_train
    if args.cache_ae:
        ae_source = prepare_cache(
            args.ae_train,
            args.work_dir / "cache" / "train_ae_npy",
            stats.ae_mean,
            stats.ae_std,
            target_hw=256,
            label="AlphaEarth train",
            force=args.force_cache,
        )
    if args.cache_tm:
        tm_source = prepare_cache(
            args.tm_train,
            args.work_dir / "cache" / "train_tm_npy",
            stats.tm_mean,
            stats.tm_std,
            target_hw=None,
            label="TerraMind train",
            force=args.force_cache,
        )
    return ae_source, tm_source


def build_datasets(
    args: argparse.Namespace,
    stats: NormStats,
    input_dtype: np.dtype,
) -> tuple[FusionDataset, FusionDataset]:
    ae_source, tm_source = build_train_sources(args, stats)
    ae_files = sorted(ae_source.glob("*.npy")) if args.cache_ae else list_nonempty_tifs(ae_source)
    tm_files = sorted(tm_source.glob("*.npy")) if args.cache_tm else list_nonempty_tifs(tm_source)
    ae_map = index_by_patch(ae_files)
    tm_map = index_by_patch(tm_files)
    label_map = index_by_patch(list_nonempty_tifs(args.label_dir))
    common = sorted(ae_map.keys() & tm_map.keys() & label_map.keys())
    if args.max_train > 0:
        common = common[: args.max_train]
    if not common:
        raise RuntimeError("No matched training patches across AlphaEarth, TerraMind, and labels.")

    train_ids, val_ids = split_ids(common, args.val_frac, args.seed)
    train_ds = FusionDataset(
        [ae_map[i] for i in train_ids],
        [tm_map[i] for i in train_ids],
        [label_map[i] for i in train_ids],
        stats,
        input_dtype=input_dtype,
        augment=True,
        mmap_cache=args.mmap_cache,
    )
    val_ds = FusionDataset(
        [ae_map[i] for i in val_ids],
        [tm_map[i] for i in val_ids],
        [label_map[i] for i in val_ids],
        stats,
        input_dtype=input_dtype,
        augment=False,
        mmap_cache=args.mmap_cache,
    )
    print(f"Matched patches: {len(common)} | train={len(train_ds)} val={len(val_ds)}")
    return train_ds, val_ds


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * args.min_lr_ratio
    )
    scaler = make_scaler(device, args.amp)
    start_epoch = 1
    best_val = math.inf

    if args.resume and args.checkpoint.exists():
        ckpt = checkpoint_load(model, args.checkpoint, device)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_val = float(ckpt.get("val_loss", best_val))
        print(f"Resumed epoch {start_epoch - 1}, best val={best_val:.4f}")

    history: list[dict[str, float]] = []
    no_improve = 0

    for epoch in range(start_epoch, args.epochs + 1):
        start = time.time()
        model.train()
        loss_sum = 0.0
        sample_count = 0

        for ae, tm, seg_t, height_t in train_loader:
            bs = ae.size(0)
            ae = move_image_tensor(ae, device, args.channels_last)
            tm = move_image_tensor(tm, device, args.channels_last)
            seg_t = seg_t.to(device, non_blocking=True)
            height_t = height_t.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with amp_context(device, args.amp):
                seg_p, height_p = model(ae, tm)
                loss = total_loss(seg_p, height_p, seg_t, height_t, args.height_lambda)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            loss_sum += float(loss.item()) * bs
            sample_count += bs

        train_loss = loss_sum / max(sample_count, 1)

        model.eval()
        val_loss_sum = 0.0
        val_count = 0
        metrics = MetricAccumulator()

        with torch.inference_mode():
            for ae, tm, seg_t, height_t in val_loader:
                bs = ae.size(0)
                ae = move_image_tensor(ae, device, args.channels_last)
                tm = move_image_tensor(tm, device, args.channels_last)
                seg_t = seg_t.to(device, non_blocking=True)
                height_t = height_t.to(device, non_blocking=True)

                with amp_context(device, args.amp):
                    seg_p, height_p = model(ae, tm)
                    loss = total_loss(seg_p, height_p, seg_t, height_t, args.height_lambda)

                val_loss_sum += float(loss.item()) * bs
                val_count += bs
                metrics.update(seg_p, seg_t, height_p, height_t)

        val_loss = val_loss_sum / max(val_count, 1)
        miou, iou_b, iou_v, iou_w, rmse_h = metrics.compute()
        scheduler.step()

        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            no_improve = 0
            args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "model": unwrap_model(model).state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "val_loss": best_val,
                    "args": vars(args),
                },
                args.checkpoint,
            )
        else:
            no_improve += 1

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "miou": miou,
            "iou_building": iou_b,
            "iou_vegetation": iou_v,
            "iou_water": iou_w,
            "height_rmse": rmse_h,
            "lr": scheduler.get_last_lr()[0],
        }
        history.append(row)
        save_history(args.work_dir / "history_fusion_windows.csv", history)

        tag = " *" if improved else ""
        print(
            f"Ep {epoch:03d}/{args.epochs} | {time.time() - start:.1f}s | "
            f"train={train_loss:.4f} val={val_loss:.4f} | "
            f"mIoU={miou:.3f} (b={iou_b:.3f} v={iou_v:.3f} w={iou_w:.3f}) | "
            f"RMSE_H={rmse_h:.2f}m{tag}"
        )

        if args.patience > 0 and no_improve >= args.patience:
            print(f"Early stopping: no improvement for {args.patience} epochs.")
            break

    print(f"Best val_loss={best_val:.4f} -> {args.checkpoint}")


@torch.no_grad()
def predict_one(
    model: nn.Module,
    ae: torch.Tensor,
    tm: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    ae = move_image_tensor(ae, device, args.channels_last)
    tm = move_image_tensor(tm, device, args.channels_last)

    def run(ae_in: torch.Tensor, tm_in: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with amp_context(device, args.amp):
            seg, height = model(ae_in, tm_in)
        return seg.float(), height.float()

    if not args.tta:
        return run(ae, tm)

    outputs = []
    for dims in (None, [-1], [-2], [-2, -1]):
        if dims is None:
            seg, height = run(ae, tm)
        else:
            seg, height = run(torch.flip(ae, dims), torch.flip(tm, dims))
            seg = torch.flip(seg, dims)
            height = torch.flip(height, dims)
        outputs.append((seg, height))

    seg_mean = torch.stack([item[0] for item in outputs]).mean(0)
    height_mean = torch.stack([item[1] for item in outputs]).mean(0)
    return seg_mean, height_mean


def infer(
    model: nn.Module,
    args: argparse.Namespace,
    stats: NormStats,
    input_dtype: np.dtype,
    device: torch.device,
) -> None:
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    ckpt = checkpoint_load(model, args.checkpoint, device)
    model.eval()
    print(f"Loaded checkpoint: epoch={ckpt.get('epoch')} val_loss={ckpt.get('val_loss')}")

    ae_map = index_by_patch(list_tifs(args.ae_test))
    tm_map = index_by_patch(list_tifs(args.tm_test))
    test_ids = sorted(ae_map.keys() & tm_map.keys())
    if not test_ids:
        raise RuntimeError("No matched test patches across AlphaEarth and TerraMind.")

    pred_dir = args.work_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    print(f"Test patches: {len(test_ids)}")

    with torch.inference_mode():
        for i, pid in enumerate(test_ids, 1):
            ae_path = ae_map[pid]
            tm_path = tm_map[pid]
            ae = normalize(load_tif(ae_path, target_hw=256), stats.ae_mean, stats.ae_std).astype(input_dtype)
            tm = normalize(load_tif(tm_path), stats.tm_mean, stats.tm_std).astype(input_dtype)
            ae_t = torch.from_numpy(ae).unsqueeze(0)
            tm_t = torch.from_numpy(tm).unsqueeze(0)

            seg_logits, height_log1p = predict_one(model, ae_t, tm_t, args, device)
            seg = torch.sigmoid(seg_logits).squeeze(0).cpu().numpy().clip(0, 1)
            height = torch.expm1(height_log1p).clamp(0).squeeze(0).cpu().numpy()
            pred = np.concatenate([seg, height], axis=0).astype(np.float32)

            out_name = f"{clean_submission_stem(tm_path.stem)}.npy"
            np.save(pred_dir / out_name, pred)
            if i % 100 == 0 or i == len(test_ids):
                print(f"  inference: {i}/{len(test_ids)}")

    args.submission.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.submission, "w", zipfile.ZIP_DEFLATED) as zf:
        for npy in sorted(pred_dir.glob("*.npy")):
            zf.write(npy, f"predictions/{npy.name}")
    print(f"Submission saved: {args.submission} ({args.submission.stat().st_size / 1e6:.1f} MB)")


def validate_inputs(args: argparse.Namespace) -> None:
    if args.mode in ("prepare", "train", "all"):
        require_dir(args.ae_train, "AlphaEarth train")
        require_dir(args.tm_train, "TerraMind train")
        require_dir(args.label_dir, "label dir")
    if args.mode in ("infer", "all"):
        require_dir(args.ae_test, "AlphaEarth test")
        require_dir(args.tm_test, "TerraMind test")


def build_model(args: argparse.Namespace, device: torch.device) -> nn.Module:
    model = TerraAlphaGatedFusion(fuse_ch=args.fuse_ch, dropout=args.dropout).to(device)
    if args.channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    if args.compile:
        try:
            model = torch.compile(model)
            print("torch.compile enabled")
        except Exception as exc:
            print(f"torch.compile failed, continuing without compile: {exc}")
    return model


def main() -> None:
    args = resolve_paths(parse_args())
    args.work_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    validate_inputs(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"AlphaEarth train: {args.ae_train}")
    print(f"TerraMind train : {args.tm_train}")
    print(f"Labels          : {args.label_dir}")
    print(f"AlphaEarth test : {args.ae_test}")
    print(f"TerraMind test  : {args.tm_test}")

    stats = load_or_compute_norm_stats(args)
    input_dtype = choose_input_dtype(args, device)
    print(f"Input dtype: {np.dtype(input_dtype).name}")
    print(f"Norm channels: AlphaEarth={stats.ae_mean.shape[0]} TerraMind={stats.tm_mean.shape[0]}")
    print(f"Work dir: {args.work_dir}")

    if args.mode == "prepare":
        build_train_sources(args, stats)
        print("Cache preparation complete.")
        return

    model = build_model(args, device)

    if args.mode in ("train", "all"):
        train_ds, val_ds = build_datasets(args, stats, input_dtype)
        train_loader = make_loader(train_ds, args.batch_size, True, args.num_workers, device)
        val_loader = make_loader(val_ds, args.batch_size, False, args.num_workers, device)
        train(model, train_loader, val_loader, args, device)

    if args.mode in ("infer", "all"):
        infer(model, args, stats, input_dtype, device)


if __name__ == "__main__":
    freeze_support()
    main()
