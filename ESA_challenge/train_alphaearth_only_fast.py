#!/usr/bin/env python
"""
Fast local AlphaEarth-only baseline.

This needs only:
  train/alphaearth_emb
  train/labels
  test/alphaearth_test_emb or test/alphaearth_emb

It avoids the much larger multi-embedding/fusion download path and writes a
competition-style submission zip.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import re
import warnings
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)


@dataclass(frozen=True)
class AeStats:
    mean: np.ndarray
    std: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and infer an AlphaEarth-only baseline.")
    parser.add_argument("--data-root", type=Path, default=Path("D:/ESA_Challenge"))
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--mode", choices=["train", "infer", "all"], default="all")
    parser.add_argument("--ae-train", type=Path)
    parser.add_argument("--label-dir", type=Path)
    parser.add_argument("--ae-test", type=Path)
    parser.add_argument("--norm-stats", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--submission", type=Path)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train", type=int, default=0, help="0 uses all matched training patches.")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--tta", action="store_true")
    return parser.parse_args()


def first_existing(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def resolve_paths(args: argparse.Namespace) -> argparse.Namespace:
    root = args.data_root
    stage = root / "_eotdl_stage" / "embed2heights" / "data"
    args.work_dir = args.work_dir or root / "_ae_fast_work"
    args.ae_train = args.ae_train or first_existing(
        root / "train" / "alphaearth_emb",
        stage / "train" / "alphaearth_emb",
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
    args.norm_stats = args.norm_stats or first_existing(
        root / "norm_stats.npy",
        stage / "norm_stats.npy",
    )
    args.checkpoint = args.checkpoint or args.work_dir / "best_model_ae_only.pth"
    args.submission = args.submission or args.work_dir / "submission_ae_only.zip"
    return args


def patch_id(stem: str) -> str | None:
    match = re.search(r"_(\d{4})_", stem)
    if match:
        return match.group(1)
    numbers = re.findall(r"\d{4}", stem)
    return numbers[0] if numbers else None


def clean_submission_stem(stem: str) -> str:
    clean = re.sub(r"^emb_", "", stem)
    clean = re.sub(r"_quantized$", "", clean)
    clean = re.sub(r"_embeddings$", "", clean)
    return clean


def list_tifs(path: Path) -> list[Path]:
    return sorted(path.glob("*.tif"))


def load_tif(path: Path, target_hw: int = 256) -> np.ndarray:
    with rasterio.open(path) as src:
        arr = src.read().astype(np.float32)
    if arr.shape[-2] != target_hw or arr.shape[-1] != target_hw:
        tensor = torch.from_numpy(arr).unsqueeze(0)
        tensor = F.interpolate(tensor, size=(target_hw, target_hw), mode="bilinear", align_corners=False)
        arr = tensor.squeeze(0).numpy()
    return arr


def compute_ae_stats(folder: Path, max_files: int = 0) -> AeStats:
    files = list_tifs(folder)
    if max_files > 0:
        files = files[:max_files]
    if not files:
        raise RuntimeError(f"No AlphaEarth training files found: {folder}")
    total = None
    total_sq = None
    count = 0
    for i, path in enumerate(files, 1):
        arr = load_tif(path)
        channels = arr.shape[0]
        flat = arr.reshape(channels, -1).astype(np.float64)
        if total is None:
            total = np.zeros(channels, dtype=np.float64)
            total_sq = np.zeros(channels, dtype=np.float64)
        total += flat.sum(axis=1)
        total_sq += (flat * flat).sum(axis=1)
        count += flat.shape[1]
        if i % 100 == 0 or i == len(files):
            print(f"  stats: {i}/{len(files)}")
    assert total is not None and total_sq is not None
    mean = total / count
    var = np.maximum(total_sq / count - mean**2, 1e-12)
    return AeStats(mean.astype(np.float32).reshape(-1, 1, 1), np.sqrt(var).astype(np.float32).reshape(-1, 1, 1))


def load_or_compute_stats(args: argparse.Namespace) -> AeStats:
    if args.norm_stats.exists():
        stats = np.load(args.norm_stats, allow_pickle=True).item()
        if "alphaearth_emb" in stats:
            mean = stats["alphaearth_emb"]["mean"].astype(np.float32).reshape(-1, 1, 1)
            std = stats["alphaearth_emb"]["std"].astype(np.float32).reshape(-1, 1, 1)
            return AeStats(mean, std)
    print("norm_stats.npy missing; computing AlphaEarth stats locally.")
    stats = compute_ae_stats(args.ae_train, args.max_train)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.work_dir / "norm_stats_ae_only.npy", {"alphaearth_emb": {"mean": stats.mean, "std": stats.std}})
    return stats


def normalize(arr: np.ndarray, stats: AeStats) -> np.ndarray:
    arr = (arr - stats.mean) / (stats.std + 1e-6)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


class AeDataset(Dataset):
    def __init__(self, ae_files: list[Path], label_files: list[Path], stats: AeStats, augment: bool):
        self.ae_files = ae_files
        self.label_files = label_files
        self.stats = stats
        self.augment = augment

    def __len__(self) -> int:
        return len(self.ae_files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ae = normalize(load_tif(self.ae_files[idx]), self.stats)
        label = load_tif(self.label_files[idx])
        seg = label[:3].clip(0, 1)
        height = np.log1p(label[3:4].clip(0, None))
        ae_t = torch.from_numpy(ae.astype(np.float32))
        seg_t = torch.from_numpy(seg.astype(np.float32))
        h_t = torch.from_numpy(height.astype(np.float32))
        if self.augment:
            if random.random() > 0.5:
                ae_t = torch.flip(ae_t, [-1]); seg_t = torch.flip(seg_t, [-1]); h_t = torch.flip(h_t, [-1])
            if random.random() > 0.5:
                ae_t = torch.flip(ae_t, [-2]); seg_t = torch.flip(seg_t, [-2]); h_t = torch.flip(h_t, [-2])
        return ae_t, seg_t, h_t


class ConvBnRelu(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class AlphaEarthHead(nn.Module):
    def __init__(self, in_ch: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBnRelu(in_ch, 128),
            ConvBnRelu(128, 128),
            ConvBnRelu(128, 64),
            ConvBnRelu(64, 64),
        )
        self.seg_head = nn.Conv2d(64, 3, 1)
        self.height_head = nn.Conv2d(64, 1, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.encoder(x)
        return self.seg_head(feat), F.softplus(self.height_head(feat))


def seg_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    weights = torch.tensor([2.0, 1.0, 1.0], device=logits.device).view(1, 3, 1, 1)
    return (F.binary_cross_entropy_with_logits(logits, target, reduction="none") * weights).mean()


def total_loss(logits: torch.Tensor, height: torch.Tensor, seg_t: torch.Tensor, height_t: torch.Tensor) -> torch.Tensor:
    return seg_loss(logits, seg_t) + 1.5 * F.huber_loss(height, height_t, delta=1.0)


def build_file_lists(args: argparse.Namespace) -> tuple[list[Path], list[Path]]:
    ae_map = {patch_id(p.stem): p for p in list_tifs(args.ae_train) if patch_id(p.stem)}
    label_map = {patch_id(p.stem): p for p in list_tifs(args.label_dir) if patch_id(p.stem)}
    ids = sorted(ae_map.keys() & label_map.keys())
    if args.max_train > 0:
        ids = ids[: args.max_train]
    if len(ids) < 2:
        raise RuntimeError("Not enough matched AlphaEarth/label training patches.")
    return [ae_map[i] for i in ids], [label_map[i] for i in ids]


def train_model(model: nn.Module, args: argparse.Namespace, stats: AeStats, device: torch.device) -> None:
    ae_files, label_files = build_file_lists(args)
    ids = list(range(len(ae_files)))
    random.shuffle(ids)
    n_val = max(1, int(len(ids) * args.val_frac))
    val_idx = ids[:n_val]
    train_idx = ids[n_val:]
    train_ds = AeDataset([ae_files[i] for i in train_idx], [label_files[i] for i in train_idx], stats, True)
    val_ds = AeDataset([ae_files[i] for i in val_idx], [label_files[i] for i in val_idx], stats, False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    print(f"Train patches: {len(train_ds)}  Val patches: {len(val_ds)}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr / 20)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and args.amp)
    best = math.inf
    rows = []
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for ae, seg_t, h_t in train_loader:
            ae, seg_t, h_t = ae.to(device), seg_t.to(device), h_t.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda" and args.amp):
                seg_p, h_p = model(ae)
                loss = total_loss(seg_p, h_p, seg_t, h_t)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
        train_loss /= max(1, len(train_loader))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for ae, seg_t, h_t in val_loader:
                ae, seg_t, h_t = ae.to(device), seg_t.to(device), h_t.to(device)
                with torch.amp.autocast("cuda", enabled=device.type == "cuda" and args.amp):
                    seg_p, h_p = model(ae)
                    val_loss += total_loss(seg_p, h_p, seg_t, h_t).item()
        val_loss /= max(1, len(val_loader))
        scheduler.step()
        if val_loss < best:
            best = val_loss
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_loss": val_loss}, args.checkpoint)
        rows.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"Epoch {epoch:03d}/{args.epochs} train={train_loss:.4f} val={val_loss:.4f} best={best:.4f}")

    with (args.work_dir / "history_ae_only.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved checkpoint: {args.checkpoint}")


def predict_one(model: nn.Module, ae: torch.Tensor, args: argparse.Namespace, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    ae = ae.to(device)
    with torch.amp.autocast("cuda", enabled=device.type == "cuda" and args.amp):
        if not args.tta:
            return model(ae)
        outputs = []
        for dims in (None, [-1], [-2], [-2, -1]):
            if dims is None:
                seg, height = model(ae)
            else:
                seg, height = model(torch.flip(ae, dims))
                seg = torch.flip(seg, dims)
                height = torch.flip(height, dims)
            outputs.append((seg.float(), height.float()))
        return torch.stack([x[0] for x in outputs]).mean(0), torch.stack([x[1] for x in outputs]).mean(0)


def infer(model: nn.Module, args: argparse.Namespace, stats: AeStats, device: torch.device) -> None:
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt)
    model.eval()
    pred_dir = args.work_dir / "predictions_ae_only"
    pred_dir.mkdir(parents=True, exist_ok=True)
    files = list_tifs(args.ae_test)
    if not files:
        raise RuntimeError(f"No AlphaEarth test files found: {args.ae_test}")
    print(f"Inference patches: {len(files)}")
    with torch.inference_mode():
        for i, path in enumerate(files, 1):
            ae = normalize(load_tif(path), stats).astype(np.float32)
            seg_logits, height_log1p = predict_one(model, torch.from_numpy(ae).unsqueeze(0), args, device)
            seg = torch.sigmoid(seg_logits).squeeze(0).cpu().numpy().clip(0, 1)
            height = torch.expm1(height_log1p).clamp(0).squeeze(0).cpu().numpy()
            pred = np.concatenate([seg, height], axis=0).astype(np.float32)
            np.save(pred_dir / f"{clean_submission_stem(path.stem)}.npy", pred)
            if i % 100 == 0 or i == len(files):
                print(f"  inference: {i}/{len(files)}")
    args.submission.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.submission, "w", zipfile.ZIP_DEFLATED) as zf:
        for npy in sorted(pred_dir.glob("*.npy")):
            zf.write(npy, f"predictions/{npy.name}")
    print(f"Submission saved: {args.submission} ({args.submission.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    args = resolve_paths(parse_args())
    args.work_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"AlphaEarth train: {args.ae_train}")
    print(f"Labels          : {args.label_dir}")
    print(f"AlphaEarth test : {args.ae_test}")
    stats = load_or_compute_stats(args)
    model = AlphaEarthHead(in_ch=stats.mean.shape[0]).to(device)
    if args.compile:
        model = torch.compile(model)
    if args.mode in ("train", "all"):
        train_model(model, args, stats, device)
    if args.mode in ("infer", "all"):
        infer(model, args, stats, device)


if __name__ == "__main__":
    main()
