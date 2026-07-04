#!/usr/bin/env python
"""
Download the complete official ESA GeoAI embed2heights dataset to a local folder.

This script wraps the official EOTDL CLI:

  eotdl datasets get embed2heights --version 1 --assets --path <stage_dir>

and then prepares a local folder layout compatible with train_fusion_windows_vscode.py:

  <dest>/
    train/
    test/
    norm_stats.npy              # copied if present in the downloaded package

Official pages:
  Challenge forum: https://platform-challenges.philab.esa.int/geoai/forum
  Dataset page   : https://www.eotdl.com/datasets/embed2heights
  EOTDL docs     : https://www.eotdl.com/docs/datasets/stage

Notes:
  - EOTDL currently requires Python >= 3.12 for installation.
  - Run `eotdl auth login` once before downloading if the CLI asks you to log in.
  - The official dataset is large, about 147.8 GB.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DATASET_NAME = "embed2heights"
DATASET_VERSION = "1"
DATASET_ID = "69d501bfc216b89fad07bfb9"
API_DATASET_URL = f"https://api.eotdl.com/datasets?name={DATASET_NAME}"


FUSION_DIRS = (
    Path("train/alphaearth_emb"),
    Path("train/terramind_s2_emb"),
    Path("train/labels"),
    Path("test/alphaearth_test_emb"),
    Path("test/terramind_test_s2_emb"),
)

FULL_DIRS = (Path("train"), Path("test"))

REQUIRED_NONEMPTY_DIRS = FUSION_DIRS


@dataclass(frozen=True)
class CopyPlan:
    source: Path
    target: Path
    files: int
    bytes: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and prepare the official embed2heights dataset locally."
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path("D:/ESA_Challenge"),
        help="Final local dataset folder used by the training script.",
    )
    parser.add_argument(
        "--stage-dir",
        type=Path,
        default=None,
        help="Raw EOTDL download/staging folder. Defaults to <dest>/_eotdl_stage.",
    )
    parser.add_argument("--dataset", default=DATASET_NAME)
    parser.add_argument("--version", default=DATASET_VERSION)
    parser.add_argument(
        "--workers",
        type=int,
        default=max(4, min(16, os.cpu_count() or 8)),
        help="EOTDL_STAGE_WORKERS value used by the official CLI.",
    )
    parser.add_argument(
        "--full",
        dest="full",
        action="store_true",
        default=True,
        help="Organize the complete downloaded train/test folders into --dest. This is the default.",
    )
    parser.add_argument(
        "--fusion-only",
        dest="full",
        action="store_false",
        help="Organize only the folders used by train_fusion_windows_vscode.py.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Do not call EOTDL; only organize/check files already present in --stage-dir.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Pass --force to EOTDL and overwrite the staged dataset.",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Physically copy files from stage-dir to dest. Default uses directory junctions/symlinks where possible.",
    )
    parser.add_argument(
        "--compute-norm-stats",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If norm_stats.npy is missing, compute AlphaEarth and TerraMind S2 mean/std after organizing.",
    )
    parser.add_argument(
        "--stats-max-files",
        type=int,
        default=0,
        help="Optional limit for norm-stat files per embedding type. 0 means exact stats over all files.",
    )
    parser.add_argument(
        "--install-eotdl",
        action="store_true",
        help="Install/upgrade eotdl with pip before downloading. Requires Python >= 3.12.",
    )
    parser.add_argument(
        "--login",
        action="store_true",
        help="Run `eotdl auth login` before downloading.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the download and organize plan without changing files.",
    )
    parser.add_argument(
        "--no-verify",
        dest="verify",
        action="store_false",
        default=True,
        help="Skip post-download completeness checks.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Do not download or reorganize; only check an existing --dest folder.",
    )
    return parser.parse_args()


def run(cmd: list[str], env: dict[str, str] | None = None, dry_run: bool = False) -> None:
    print("+ " + " ".join(str(part) for part in cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True, env=env)


def check_python_for_eotdl() -> None:
    if sys.version_info < (3, 12):
        print(
            "WARNING: Official EOTDL docs state that Python >= 3.12 is required.\n"
            f"Current interpreter: {sys.version.split()[0]}\n"
            "If `pip install eotdl` fails, create a Python 3.12 venv for the downloader."
        )


def install_eotdl(dry_run: bool) -> None:
    check_python_for_eotdl()
    run([sys.executable, "-m", "pip", "install", "--upgrade", "eotdl"], dry_run=dry_run)


def find_eotdl_executable() -> str | None:
    local_names = ["eotdl.exe", "eotdl.cmd", "eotdl"]
    python_dir = Path(sys.executable).resolve().parent
    for name in local_names:
        candidate = python_dir / name
        if candidate.exists():
            return str(candidate)

    exe = shutil.which("eotdl")
    if exe:
        return exe
    return None


def ensure_eotdl_available() -> str:
    exe = find_eotdl_executable()
    if exe:
        print(f"EOTDL CLI: {exe}")
        return exe
    raise FileNotFoundError(
        "Could not find the EOTDL CLI in this Python environment.\n"
        "Install it into the repo environment first:\n"
        "  .\\.venv-download\\Scripts\\python.exe -m pip install -r requirements_downloader_windows.txt\n"
        "or rerun this script with --install-eotdl."
    )


def login_if_requested(args: argparse.Namespace, eotdl_exe: str) -> None:
    if args.login:
        run([eotdl_exe, "auth", "login"], dry_run=args.dry_run)


def download_with_eotdl(args: argparse.Namespace, eotdl_exe: str) -> None:
    if args.skip_download:
        print("Skipping download because --skip-download was set.")
        return

    env = os.environ.copy()
    env["EOTDL_STAGE_WORKERS"] = str(args.workers)
    env["EOTDL_DOWNLOAD_PATH"] = str(args.stage_dir)

    cmd = [
        eotdl_exe,
        "datasets",
        "get",
        args.dataset,
        "--version",
        str(args.version),
        "--assets",
        "--path",
        str(args.stage_dir),
    ]
    if args.force_download:
        cmd.append("--force")
    run(cmd, env=env, dry_run=args.dry_run)


def iter_candidate_roots(stage_dir: Path) -> Iterable[Path]:
    names = [
        DATASET_NAME,
        "data",
        "embed2heights",
        "Embed2Heights",
    ]
    yield stage_dir
    for name in names:
        yield stage_dir / name
    if not stage_dir.exists():
        return
    for child in stage_dir.glob("*"):
        if child.is_dir():
            yield child
            yield child / "data"
    for child in stage_dir.rglob("train"):
        if child.is_dir():
            yield child.parent


def find_data_root(stage_dir: Path) -> Path:
    candidates = []
    for root in iter_candidate_roots(stage_dir):
        score = sum((root / rel).exists() for rel in FUSION_DIRS[:3])
        score += sum((root / rel).exists() for rel in FUSION_DIRS[3:])
        if score:
            candidates.append((score, root))
    if candidates:
        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1]

    raise FileNotFoundError(
        f"Could not locate the dataset folders under {stage_dir}.\n"
        "Expected folders like train/alphaearth_emb and train/labels.\n"
        "Run the EOTDL download first, or pass the correct --stage-dir."
    )


def folder_stats(path: Path) -> tuple[int, int]:
    files = 0
    size = 0
    if not path.exists():
        return files, size
    for item in path.rglob("*"):
        if item.is_file():
            files += 1
            size += item.stat().st_size
    return files, size


def make_copy_plan(data_root: Path, dest: Path, full: bool) -> list[CopyPlan]:
    rels = list(FULL_DIRS if full else FUSION_DIRS)

    plans: list[CopyPlan] = []
    for rel in rels:
        src = data_root / rel
        if not src.exists():
            print(f"WARNING: missing source folder: {src}")
            continue
        files, size = folder_stats(src)
        plans.append(CopyPlan(source=src, target=dest / rel, files=files, bytes=size))
    return plans


def patch_id(stem: str) -> str | None:
    match = re.search(r"_(\d{4})_", stem)
    if match:
        return match.group(1)
    numbers = re.findall(r"\d{4}", stem)
    return numbers[0] if numbers else None


def tif_patch_ids(folder: Path) -> set[str]:
    ids = set()
    for path in folder.glob("*.tif"):
        pid = patch_id(path.stem)
        if pid:
            ids.add(pid)
    return ids


def verify_dataset_root(root: Path) -> dict[str, dict[str, int]]:
    print("Verifying downloaded dataset...")
    if not root.exists():
        raise FileNotFoundError(f"Dataset folder does not exist: {root}")

    report: dict[str, dict[str, int]] = {}
    total_tifs = 0
    missing = []
    empty = []
    for rel in REQUIRED_NONEMPTY_DIRS:
        folder = root / rel
        files, size = folder_stats(folder)
        tifs = len(list(folder.glob("*.tif"))) if folder.exists() else 0
        report[str(rel)] = {"files": files, "tif_files": tifs, "bytes": size}
        total_tifs += tifs
        if not folder.exists():
            missing.append(str(rel))
        elif tifs == 0:
            empty.append(str(rel))
        print(f"  {rel}: {tifs:,} tif files, {size / 1e9:.2f} GB")

    if missing:
        raise RuntimeError(
            "The download is incomplete; these required folders are missing:\n"
            + "\n".join(f"  - {item}" for item in missing)
        )
    if total_tifs == 0:
        raise RuntimeError(
            "No .tif assets were found. This usually means only metadata was staged. "
            "The official EOTDL command must include --assets."
        )
    if empty:
        raise RuntimeError(
            "The download is incomplete; these required folders contain no .tif files:\n"
            + "\n".join(f"  - {item}" for item in empty)
        )

    ae_train = tif_patch_ids(root / "train" / "alphaearth_emb")
    tm_train = tif_patch_ids(root / "train" / "terramind_s2_emb")
    labels = tif_patch_ids(root / "train" / "labels")
    train_common = ae_train & tm_train & labels
    if not train_common:
        raise RuntimeError("No matched training patch IDs across AlphaEarth, TerraMind S2, and labels.")
    if len(train_common) < max(len(ae_train), len(tm_train), len(labels)):
        print(
            "WARNING: Some training patch IDs are not present in all three training folders "
            f"(matched {len(train_common):,}; alphaearth {len(ae_train):,}; "
            f"terramind {len(tm_train):,}; labels {len(labels):,})."
        )

    ae_test_dir = root / "test" / "alphaearth_emb"
    if not ae_test_dir.exists():
        ae_test_dir = root / "test" / "alphaearth_test_emb"
    ae_test = tif_patch_ids(ae_test_dir)
    tm_test = tif_patch_ids(root / "test" / "terramind_test_s2_emb")
    test_common = ae_test & tm_test
    if not test_common:
        raise RuntimeError("No matched test patch IDs across AlphaEarth and TerraMind S2.")
    if len(test_common) < max(len(ae_test), len(tm_test)):
        print(
            "WARNING: Some test patch IDs are not present in both test folders "
            f"(matched {len(test_common):,}; alphaearth {len(ae_test):,}; terramind {len(tm_test):,})."
        )

    report["_matched_patches"] = {
        "train": len(train_common),
        "test": len(test_common),
        "total_tif_files": total_tifs,
    }
    print(f"Matched train patches: {len(train_common):,}")
    print(f"Matched test patches : {len(test_common):,}")
    return report


def remove_existing_link(path: Path) -> None:
    if path.exists() or path.is_symlink():
        is_windows_reparse = False
        if os.name == "nt" and path.exists():
            attrs = getattr(path.stat(), "st_file_attributes", 0)
            is_windows_reparse = bool(attrs & getattr(__import__("stat"), "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
        if path.is_symlink() or path.is_file():
            path.unlink()
        elif is_windows_reparse:
            path.rmdir()
        elif path.is_dir() and not any(path.iterdir()):
            path.rmdir()
        else:
            raise FileExistsError(
                f"Target already exists and is not empty: {path}\n"
                "Move/delete it first, or rerun without --copy to reuse the existing folder."
            )


def link_or_copy_dir(src: Path, dst: Path, copy: bool, dry_run: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        action = "copy" if copy else "link"
        print(f"{action}: {src} -> {dst}")
        return

    if copy:
        if dst.exists():
            raise FileExistsError(f"Copy target already exists: {dst}")
        shutil.copytree(src, dst)
        return

    if dst.exists() and dst.is_dir() and not dst.is_symlink() and any(dst.iterdir()):
        print(f"reuse existing folder: {dst}")
        return

    remove_existing_link(dst)
    try:
        if os.name == "nt":
            os.symlink(src, dst, target_is_directory=True)
        else:
            os.symlink(src, dst)
    except OSError:
        if os.name == "nt":
            subprocess.run(["cmd", "/c", "mklink", "/J", str(dst), str(src)], check=True)
        else:
            raise


def copy_optional_files(data_root: Path, dest: Path, dry_run: bool) -> None:
    for name in ("norm_stats.npy", "catalog.parquet"):
        src = data_root / name
        if not src.exists():
            continue
        dst = dest / name
        print(f"file: {src} -> {dst}")
        if not dry_run:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


def compute_channel_stats(folder: Path, max_files: int = 0) -> tuple[list[float], list[float]]:
    try:
        import numpy as np
        import rasterio
    except ImportError as exc:
        raise ImportError(
            "Computing norm_stats.npy requires numpy and rasterio. Install them with:\n"
            "  pip install numpy rasterio"
        ) from exc

    files = sorted(folder.glob("*.tif"))
    if max_files > 0:
        files = files[:max_files]
    if not files:
        raise RuntimeError(f"No tif files found for norm stats: {folder}")

    total = None
    total_sq = None
    count = 0
    for i, path in enumerate(files, 1):
        with rasterio.open(path) as src:
            arr = src.read().astype("float32")
        c = arr.shape[0]
        flat = arr.reshape(c, -1)
        if total is None:
            total = np.zeros(c, dtype=np.float64)
            total_sq = np.zeros(c, dtype=np.float64)
        total += flat.sum(axis=1, dtype=np.float64)
        total_sq += (flat.astype(np.float64) ** 2).sum(axis=1)
        count += flat.shape[1]
        if i % 100 == 0 or i == len(files):
            print(f"  stats {folder.name}: {i}/{len(files)}")

    assert total is not None and total_sq is not None
    mean = total / count
    var = np.maximum(total_sq / count - mean**2, 1e-12)
    std = np.sqrt(var)
    return mean.astype(np.float32).tolist(), std.astype(np.float32).tolist()


def maybe_compute_norm_stats(args: argparse.Namespace) -> None:
    stats_path = args.dest / "norm_stats.npy"
    if stats_path.exists():
        print(f"norm_stats.npy already exists: {stats_path}")
        return
    if not args.compute_norm_stats:
        print("norm_stats.npy missing; skipped because --no-compute-norm-stats was set.")
        return
    if args.dry_run:
        print(f"would compute norm_stats.npy -> {stats_path}")
        return

    import numpy as np

    ae_dir = args.dest / "train" / "alphaearth_emb"
    tm_dir = args.dest / "train" / "terramind_s2_emb"
    print("Computing norm_stats.npy for training script...")
    ae_mean, ae_std = compute_channel_stats(ae_dir, args.stats_max_files)
    tm_mean, tm_std = compute_channel_stats(tm_dir, args.stats_max_files)
    stats = {
        "alphaearth_emb": {
            "mean": np.asarray(ae_mean, dtype=np.float32),
            "std": np.asarray(ae_std, dtype=np.float32),
        },
        "terramind_s2_emb": {
            "mean": np.asarray(tm_mean, dtype=np.float32),
            "std": np.asarray(tm_std, dtype=np.float32),
        },
    }
    np.save(stats_path, stats)
    print(f"Saved {stats_path}")


def organize_dataset(args: argparse.Namespace) -> None:
    data_root = find_data_root(args.stage_dir)
    print(f"Detected downloaded data root: {data_root}")
    plans = make_copy_plan(data_root, args.dest, full=args.full)
    if not plans:
        raise RuntimeError("No folders to organize.")

    total_files = sum(plan.files for plan in plans)
    total_bytes = sum(plan.bytes for plan in plans)
    print(f"Organizing {len(plans)} folders, {total_files:,} files, {total_bytes / 1e9:.2f} GB")

    for plan in plans:
        print(f"  {plan.source} -> {plan.target} ({plan.files:,} files, {plan.bytes / 1e9:.2f} GB)")
        link_or_copy_dir(plan.source, plan.target, copy=args.copy, dry_run=args.dry_run)
    copy_optional_files(data_root, args.dest, dry_run=args.dry_run)
    maybe_compute_norm_stats(args)
    if args.verify and not args.dry_run:
        args.verify_report = verify_dataset_root(args.dest)


def write_manifest(args: argparse.Namespace) -> None:
    manifest = {
        "dataset": args.dataset,
        "dataset_id": DATASET_ID if args.dataset == DATASET_NAME else None,
        "version": args.version,
        "dest": str(args.dest),
        "stage_dir": str(args.stage_dir),
        "official_dataset_page": f"https://www.eotdl.com/datasets/{args.dataset}",
        "official_challenge_forum": "https://platform-challenges.philab.esa.int/geoai/forum",
        "organized_layout": "full train/test" if args.full else "fusion folders only",
        "verification": getattr(args, "verify_report", None),
    }
    path = args.dest / "download_manifest.json"
    print(f"manifest: {path}")
    if not args.dry_run:
        args.dest.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.dest = args.dest.resolve()
    args.stage_dir = (args.stage_dir or (args.dest / "_eotdl_stage")).resolve()

    print(f"Destination : {args.dest}")
    print(f"Stage dir   : {args.stage_dir}")
    print(f"Dataset     : {args.dataset} version {args.version}")
    print(f"Workers     : {args.workers}")

    if args.verify_only:
        args.verify_report = verify_dataset_root(args.dest)
        write_manifest(args)
        print("Done.")
        return

    if args.install_eotdl:
        install_eotdl(args.dry_run)

    if not args.skip_download:
        check_python_for_eotdl()
        eotdl_exe = ensure_eotdl_available()
        login_if_requested(args, eotdl_exe)
        download_with_eotdl(args, eotdl_exe)

    organize_dataset(args)
    write_manifest(args)
    print("Done.")


if __name__ == "__main__":
    main()
