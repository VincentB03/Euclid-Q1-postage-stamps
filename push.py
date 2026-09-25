"""Concatenate the datasets saved in $SCRATCH/datasets/batch*, drop duplicate obj_id and push to the Hub."""

import glob, os, sys
sys.path.insert(0, "src")
from datasets import load_from_disk, concatenate_datasets
from dataset_builder import (
    drop_duplicate_obj_ids, push_dataset, update_dataset_card,
)

REPO = "VincentB03/Euclid-Q1-postage-stamps"

dirs = sorted(glob.glob(f"{os.environ['SCRATCH']}/datasets/batch*"))
if not dirs:
    sys.exit("no batch* directory found in $SCRATCH/datasets")

parts = [load_from_disk(d) for d in dirs]
for d, p in zip(dirs, parts):
    print(f"{len(p):>8}  {d}")

ds = concatenate_datasets(parts)
print(f"{len(ds):>8}  concatenated")
ds = drop_duplicate_obj_ids(ds)

print(f"pushing to {REPO} ...")
push_dataset(ds, repo_id=REPO)
print(f"{len(ds)} rows pushed")

try:
    update_dataset_card(
        REPO, ds,
        {
            "obs_selection": "explicit --obs-ids, built in batches on Jean Zay",
            "processes": 4,
            "psf_residual": True,
            "reference_psf": "src/euclid_vis_isotropic_min_psf.fits",
            "drop_duplicates": True,
            "zero_flagged_pixels": True,
            "min_snr": "none",
            "drop_truncated": True,
            "output_mode": "fresh push (concatenated local batches)",
            "visibility": "private",
        },
        command="srun python -u src/main.py --obs-ids <ids> --skip-acquire "
                "--processes 4 --zero-flagged-pixels --drop-duplicates "
                "--drop-truncated --reference-psf src/euclid_vis_isotropic_min_psf.fits "
                "--save-to $SCRATCH/datasets/batchNN",
    )
    print("dataset card updated")
except Exception as exc:
    print(f"WARNING: dataset card not updated: {exc}")
