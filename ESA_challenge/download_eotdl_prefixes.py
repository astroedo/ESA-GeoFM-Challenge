#!/usr/bin/env python
"""
Selective EOTDL asset downloader for embed2heights.

Use this when the full EOTDL assets download is too large. It reads the local
catalog parquet and downloads only rows whose id starts with selected prefixes.

Default prefix:
  data/train/terramind_s2_emb
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd
import requests


DEFAULT_STAGE = Path("D:/ESA_Challenge/_eotdl_stage/embed2heights")
DEFAULT_PREFIXES = ["data/train/terramind_s2_emb"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download selected embed2heights assets by catalog prefix.")
    parser.add_argument("--stage-root", type=Path, default=DEFAULT_STAGE)
    parser.add_argument("--catalog", type=Path, default=None)
    parser.add_argument("--prefix", action="append", default=None, help="Catalog id prefix to download. Can repeat.")
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Download at most this many matching assets after skipping already complete files. 0 means no limit.",
    )
    parser.add_argument("--workers", type=int, default=1, help="Reserved for compatibility; downloads sequentially.")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_token() -> str:
    creds_path = Path.home() / ".cache" / "eotdl" / "creds.json"
    if not creds_path.exists():
        raise FileNotFoundError(
            f"EOTDL credentials not found: {creds_path}\n"
            "Run .\\.venv-download\\Scripts\\eotdl.exe auth login first."
        )
    creds = json.loads(creds_path.read_text(encoding="utf-8"))
    token = creds.get("id_token") or creds.get("access_token")
    if not token:
        raise KeyError(f"No id_token/access_token in {creds_path}")
    return token


def expected_size(row) -> int:
    try:
        return int(row["assets"]["asset"].get("size") or 0)
    except Exception:
        return 0


def asset_href(row) -> str:
    return row["assets"]["asset"]["href"]


def download_one(row, out_path: Path, headers: dict[str, str], timeout: int, force: bool) -> bool:
    size = expected_size(row)
    if out_path.exists() and not force:
        if size <= 0 or out_path.stat().st_size >= max(1000, int(size * 0.99)):
            return False

    out_path.parent.mkdir(parents=True, exist_ok=True)
    href = asset_href(row)
    meta = requests.get(href, headers=headers, timeout=timeout)
    meta.raise_for_status()
    data = meta.json()
    url = data.get("presigned_url") or data.get("url") or data.get("href")
    if not url:
        raise KeyError(f"No presigned URL returned for {row['id']}")

    tmp = out_path.with_suffix(out_path.suffix + ".part")
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with tmp.open("wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    os.replace(tmp, out_path)
    return True


def main() -> None:
    args = parse_args()
    args.stage_root = args.stage_root.resolve()
    catalog = (args.catalog or (args.stage_root / "catalog.v1.parquet")).resolve()
    prefixes = args.prefix or DEFAULT_PREFIXES

    if not catalog.exists():
        raise FileNotFoundError(
            f"Catalog not found: {catalog}\n"
            "Run the normal EOTDL command once without assets, or use the existing staged catalog."
        )

    df = pd.read_parquet(catalog)
    ids = df["id"].astype(str)
    mask = False
    for prefix in prefixes:
        mask = mask | ids.str.startswith(prefix)
    selected = df[mask].copy()
    if selected.empty:
        raise RuntimeError(f"No catalog rows matched prefixes: {prefixes}")

    pending_rows = []
    complete_rows = []
    for _, row in selected.iterrows():
        out = args.stage_root / str(row["id"])
        size = expected_size(row)
        if out.exists() and (size <= 0 or out.stat().st_size >= max(1000, int(size * 0.99))):
            complete_rows.append(row)
        else:
            pending_rows.append(row)
    if args.max_files > 0:
        pending_rows = pending_rows[: args.max_files]
    selected_to_download = pd.DataFrame(pending_rows)

    total_bytes = sum(expected_size(row) for _, row in selected_to_download.iterrows())
    print(f"Catalog : {catalog}")
    print(f"Stage   : {args.stage_root}")
    print(f"Prefixes: {', '.join(prefixes)}")
    print(f"Matched : {len(selected):,} files")
    print(f"Already complete: {len(complete_rows):,}/{len(selected):,}")
    print(f"To download: {len(selected_to_download):,} files, {total_bytes / 1e9:.2f} GB expected")

    if args.dry_run:
        for _, row in selected_to_download.head(10).iterrows():
            print(f"  {row['id']} -> {args.stage_root / str(row['id'])}")
        print("Dry run only.")
        return

    headers = {"Authorization": f"Bearer {load_token()}"}
    downloaded = 0
    skipped = 0
    failed: list[tuple[str, str]] = []
    start = time.time()

    for i, (_, row) in enumerate(selected_to_download.iterrows(), 1):
        out = args.stage_root / str(row["id"])
        try:
            changed = download_one(row, out, headers, args.timeout, args.force)
            downloaded += int(changed)
            skipped += int(not changed)
        except Exception as exc:
            failed.append((str(row["id"]), str(exc)))
        if i % 50 == 0 or i == len(selected_to_download):
            elapsed = max(time.time() - start, 1)
            print(
                f"  {i:,}/{len(selected_to_download):,} done | downloaded={downloaded:,} "
                f"skipped={skipped:,} failed={len(failed):,} | {i / elapsed:.2f} files/s"
            )

    if failed:
        print("Failures:")
        for item, err in failed[:10]:
            print(f"  {item}: {err}")
        raise RuntimeError(f"{len(failed)} downloads failed.")

    print("Done.")


if __name__ == "__main__":
    main()
