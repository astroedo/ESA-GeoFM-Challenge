# Windows VSCode Run Guide

This folder contains a Windows-local version of the AlphaEarth + TerraMind S2
fusion trainer:

- `download_embed2heights_local.py`
- `train_fusion_windows_vscode.py`
- `requirements_downloader_windows.txt`
- `requirements_windows.txt`
- `.vscode/launch.json`

## 1. Suggested folder layout

Put the challenge data on a fast SSD/NVMe drive:

```text
D:/ESA_Challenge/
  train/
    alphaearth_emb/
    terramind_s2_emb/
    labels/
  test/
    alphaearth_emb/
    terramind_test_s2_emb/
  norm_stats.npy
```

The script writes cache, checkpoint, history, predictions, and submission zip to:

```text
D:/ESA_Challenge/_fusion_work/
```

You can change this with `--work-dir`.

## 2. Environment

For downloading the official EOTDL dataset, use Python 3.12 because EOTDL's
official docs require Python >= 3.12:

```powershell
Set-Location D:\Documents\ESA_challenge
py -3.13 -m venv .venv-download
.\.venv-download\Scripts\python.exe -m pip install --upgrade pip
.\.venv-download\Scripts\python.exe -m pip install -r requirements_downloader_windows.txt
.\.venv-download\Scripts\eotdl.exe auth login
```

If your machine has Python 3.12 instead of 3.13, replace `py -3.13` with
`py -3.12`. Keep the `.venv-download` path inside this repository so packages
are installed locally instead of into the C drive user Python environment.

Download and prepare the complete local folder:

```powershell
.\.venv-download\Scripts\python.exe download_embed2heights_local.py `
  --dest D:/ESA_Challenge `
  --workers 8
```

The official dataset is about 147.8 GB. The script calls the official EOTDL
CLI with `--assets`, then organizes the complete `train/` and `test/` folders
expected by the official package while keeping the training script layout
working. By default it links the staged folders instead of copying them. Add
`--copy` only if you really want a second physical copy.

After organizing, the script verifies that the required `.tif` asset folders are
present and non-empty, and that train/test patch IDs match across the required
inputs. If you only want to check an existing download:

```bat
python download_embed2heights_local.py --dest D:/ESA_Challenge --verify-only
```

If you intentionally want to keep only the subset used by
`train_fusion_windows_vscode.py`, add `--fusion-only`.

### Lightweight fusion path

If the full official download is too large, do not continue downloading every
asset. The fusion trainer only needs:

```text
train/alphaearth_emb
train/terramind_s2_emb
train/labels
test/alphaearth_test_emb
test/terramind_test_s2_emb
```

If the staged catalog already exists, selectively download only the missing
TerraMind S2 training folder:

```powershell
.\.venv-download\Scripts\python.exe download_eotdl_prefixes.py `
  --stage-root D:/ESA_Challenge/_eotdl_stage/embed2heights `
  --prefix data/train/terramind_s2_emb
```

Then run a quick fusion smoke test:

```powershell
.\.venv-download\Scripts\python.exe -m pip install torch torchvision
.\.venv-download\Scripts\python.exe train_fusion_windows_vscode.py `
  --data-root D:/ESA_Challenge `
  --epochs 3 `
  --max-train 128 `
  --batch-size 2 `
  --num-workers 0 `
  --no-cache-ae `
  --no-cache-tm
```

For a stronger run, remove `--max-train 128`, increase `--epochs`, and enable
`--tta` for inference.

If `norm_stats.npy` is not included in the downloaded package, the script will
compute the AlphaEarth and TerraMind S2 normalization stats after download. This
does one full read of those two training folders. To skip that step:

```bat
python download_embed2heights_local.py --dest D:/ESA_Challenge --no-compute-norm-stats
```

For training, create and activate your PyTorch venv:

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
```

Install PyTorch using the command that matches your CUDA version from:

```text
https://pytorch.org/get-started/locally/
```

Then install the remaining packages:

```bat
pip install -r requirements_windows.txt
```

## 3. Fast local training

```bat
python train_fusion_windows_vscode.py ^
  --data-root D:/ESA_Challenge ^
  --work-dir D:/ESA_Challenge/_fusion_work ^
  --mode all ^
  --epochs 50 ^
  --batch-size 8 ^
  --num-workers 4 ^
  --tta
```

For a smaller GPU, try:

```bat
python train_fusion_windows_vscode.py --data-root D:/ESA_Challenge --batch-size 4 --num-workers 2
```

## 4. Useful modes

Prepare cache only:

```bat
python train_fusion_windows_vscode.py --data-root D:/ESA_Challenge --mode prepare
```

Train only:

```bat
python train_fusion_windows_vscode.py --data-root D:/ESA_Challenge --mode train
```

Inference only:

```bat
python train_fusion_windows_vscode.py --data-root D:/ESA_Challenge --mode infer --tta
```

Resume training:

```bat
python train_fusion_windows_vscode.py --data-root D:/ESA_Challenge --mode train --resume
```

## 5. VSCode

Open this folder in VSCode, select the Python interpreter from `.venv`, then use
Run and Debug:

- `Fusion Train Windows`
- `Fusion Infer Windows`

Edit `.vscode/launch.json` if your dataset is not under `D:/ESA_Challenge`.
