# Data (not committed)

Download the RSNA Pediatric Bone Age set from Kaggle:

```powershell
kaggle datasets download kmader/rsna-bone-age -p data --unzip
```

Expected layout:

```text
data/
  boneage-training-dataset.csv   # id, boneage (months), male
  boneage-training-dataset/      # 12,611 PNGs
  boneage-validation-dataset.csv # 1,425 rows (if present in repack)
  README.md
```

Compute train `MEAN`/`STD` at build time — do not hardcode.
See `src/eda.py` for the sex-split distribution check.
