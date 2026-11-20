# BoneCast — Bone Age From Hand X-Rays

CNN regression: a child's hand X-ray + sex → bone age in months (MAE).
Compares every model and training algo used, on GPU when available.

## Setup (project-local env)

```powershell
uv venv
uv pip install -r requirements.txt
# or: python -m venv --system-site-packages .venv; .\.venv\Scripts\Activate.ps1
```

GPU is automatic (`cuda` if `torch.cuda.is_available()`, AMP on CUDA).
Verified here on NVIDIA GeForce RTX 3050 4GB Laptop GPU.

## Data (required before real training)

```powershell
kaggle datasets download kmader/rsna-bone-age -p data --unzip
```

Expected: `data/boneage-training-dataset.csv` (`id,boneage,male`) + PNG folder
(12,611 train / 1,425 val / 200 test, PNGs up to ~2080×1600).
Targets: mean ~127mo, std ~41mo; boys 54% / girls 46%; 72% of cases are 8–15y.

## Models benchmarked (`python -m src.benchmark`)

| Model | Params | Forward/batch@256 (RTX 3050) |
|---|---|---|
| ResNet-50 + sex (main, `src/model.py`) | 24.0M | 48.8 ms |
| ResNet-50 no-sex (ablation) | 24.0M | 48.1 ms |
| ResNet-18 + sex (light) | 11.2M | 15.2 ms |

Baselines (synthetic N(127,41), n=512): mean MAE 31.65mo, sex-stratified 31.64mo
(theory ≈ 0.8×41 = 32.8mo). Real-CSV baselines print automatically when data exists.
Target bars: challenge winner ~4.2mo, radiologists ~4.8–7.0mo.

Algos (one synthetic z-score step): MSE-on-z 1.1648, L1-on-z 0.7252.
Default training: z-targets, MSE-on-z, AdamW, ReduceLROnPlateau, early stop on val MAE-months.

## Train / evaluate

```powershell
.\.venv\Scripts\python.exe -m src.train --size 256 --epochs 2 --batch 8   # smoke (4GB GPU)
python -m src.train --size 512 --epochs 20 --batch 16                      # full
python -m src.evaluate   # MAE overall + boys/girls vs mean baseline
python -m src.eda        # target distribution by sex -> figs/
```

Key rules the code enforces: pad-to-square before resize (never stretch),
sex fused with image features, normalized targets (denormalize for MAE),
train-only augmentation (no horizontal flip — left-hand chirality).
