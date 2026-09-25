# Euclid Q1 VIS postage stamps

Builds a machine-learning dataset of **64 × 64 postage stamps** of isolated
galaxies from Euclid Q1 VIS calibrated exposures. Each stamp comes with its
noise map, bad-pixel mask and the PSF interpolated at the source position. The
result is a Hugging Face [`datasets.Dataset`](https://huggingface.co/docs/datasets),
saved locally or pushed to the Hub.

Pipeline: **acquire** (Euclid archive downloads) → **extract** (per-quadrant
FITS) → **build** (stamps) → **output** (disk or Hub).

## Dataset

One row per source:

| Field | Type | Description |
|---|---|---|
| `obs_id`, `quadrant` | `string` | Euclid observation ID, VIS quadrant (e.g. `3-4.E`) |
| `ra`, `dec` | `float32` | Source position (deg, ICRS) |
| `obj_id` | `int64` | MER catalogue `object_id` |
| `flux` | `float32` | `FLUX_VIS_UNIF` from the PHZ catalogue |
| `snr` | `float32` | Isophotal S/N on the 3σ segmentation map ([see below](#sn-and-truncation)) |
| `truncated` | `bool` | Segmentation map touches the stamp edge |
| `sci_subtracted` | `float32 [64, 64]` | Science − background |
| `noise_map` | `float32 [64, 64]` | RMS |
| `binary_mask` | `int32 [64, 64]` | 1 = valid pixel, 0 = flagged pixel |
| `psf_stamp` | `float32 [21, 21]` | PSF at the source position, normalised to sum 1 |
| `psf_residual` | `float32 [21, 21]` | *(optional)* kernel `k` such that `reference_psf ⊛ k = psf_stamp` |

```python
from datasets import load_dataset, load_from_disk
ds = load_from_disk("<DATA_DIR>/dataset")        # local build
ds = load_dataset("<repo_id>", split="train")    # from the Hub
```

## Installation

Python ≥ 3.9:

```bash
pip install -r requirements.txt
```

- **Data directory**: all files are read from and written to `$EUCLID_DATA_DIR`.
  If it is unset: `/content/drive/MyDrive/Q1_VIS_CALIBRATED_DB` on Colab,
  otherwise `data/Q1_VIS_CALIBRATED_DB/` in the repository.
- **Euclid archive**: anonymous access is enough for Q1. If a product needs
  authentication, call `Euclid.login()` (`astroquery.esa.euclid`) first.
- **Hugging Face**: pushing needs a write token (`--hf-token`, `HF_TOKEN` or
  `huggingface-cli login`).

## Usage

```bash
# Full run on the first 17 observations, 4 workers, saved to <DATA_DIR>/dataset
python src/main.py --limit 17 --processes 4

# Data already downloaded and sliced: build and push a new Hub dataset
python src/main.py --limit 17 --skip-acquire --skip-extract --push

# Append new observations to the existing Hub dataset, one row per obj_id
python src/main.py --obs-ids 2698 2699 --skip-acquire --skip-extract --merge --drop-duplicates
```

Nothing is uploaded without `--push` or `--merge`. Each push also writes a
*Build info* block (command, parameters, observation IDs) at the end of the Hub
dataset card, leaving the rest of the card as is.

| Option | Effect |
|---|---|
| `--obs-ids ID ...`, `--obs-ids-file PATH`, `--limit N` | Observations to process (default: all optimal ones) |
| `--skip-acquire`, `--skip-extract`, `--skip-build` | Skip a stage |
| `--processes N` | Build workers (default: all cores; `1` = sequential) |
| `--min-snr X` | Drop sources with `snr ≤ X` |
| `--drop-truncated` | Drop truncated sources |
| `--zero-flagged-pixels` | Set flagged pixels of `sci_subtracted` to 0 ([see below](#flagged-pixels)) |
| `--drop-duplicates` | One row per `obj_id`; with `--merge`, existing Hub rows win |
| `--no-residual` | Do not add the `psf_residual` column |
| `--reference-psf PATH` | Reference PSF (default: `src/euclid_vis_isotropic_min_psf.fits`) |
| `--push` / `--merge` | Push a new dataset / append to the existing Hub dataset |
| `--repo-id ID`, `--public`, `--hf-token TOKEN` | Hub target (private by default) |
| `--save-to DIR` | Local output (default when not pushing: `<DATA_DIR>/dataset`) |
| `--quiet` | Less logging |

## How it works

1. **Observations**: one per sky tile, i.e. the first dither (`00-1`) of each
   VIS observation in `q1.calibrated_frame`, de-duplicated on RA/Dec rounded
   to 0.1°.
2. **Acquire**: download the science (DET) and background (BKG) frames, the
   global PSF model and a MER ⋈ PHZ ⋈ morphology catalogue covering each
   observation. Files already on disk are not downloaded again; BKG frames and
   catalogues are only fetched when the DET frame is present.
3. **Extract**: slice each frame into its 144 quadrants (6 × 6 CCDs × 4) under
   `quadrant-data/`.
4. **Select**: keep sources with `point_like_prob ≤ 0.5`,
   `0.575 ≤ FLUX_VIS_UNIF ≤ 575.4` µJy (VIS 24.5 to 17 AB mag),
   `spurious_prob ≤ 0.2`, `det_quality_flag == 0`, `deblended_flag == 0`, and
   no catalogue neighbour within `(46 + a // 2) × 0.1″`, `a` being the
   neighbour's `semimajor_axis`.
5. **Cut**: 64 × 64 cutout (sources too close to a quadrant edge are skipped),
   background subtracted, dropped if ≥ 8 % of its pixels are flagged
   (`FLG & FLAG_BITMASK`). The PSF is bilinearly interpolated on the
   quadrant's 9 × 9 PSF grid.
6. **Post-process** (optional): de-duplicate `obj_id` (a source can appear in
   several quadrants or observations); add `psf_residual` by dividing
   `psf_stamp` by the isotropic reference PSF in Fourier space.

All thresholds are in [`src/config.py`](src/config.py).

### S/N and truncation

Unflagged pixels above `3 × noise_map` are grouped into 8-connected regions.
The source is the union of the regions that reach the central 5 × 5 px box
(which tolerates a flagged or slightly offset centre). Then:

```
snr       = Σ sci_subtracted / sqrt(Σ noise_map²)   over the source pixels
truncated = the source touches the stamp edge
```

`snr = 0` when no region reaches the box. The stamps come from single,
non-resampled exposures, so pixel noise is uncorrelated and `noise_map` alone
gives the right error.

Both columns are always stored, so other cuts can be applied later with
`dataset.filter`. The principle (S/N ≤ 10 cut, 3σ segmentation map, edge test)
follows [Csizi et al. 2025, A&A 695, A283](https://arxiv.org/abs/2409.07528),
Sect. 4.2; the exact formula and the central box are choices of this pipeline.
**This definition is provisional.** After changing it, rebuild the dataset
(`--push`) instead of `--merge`-ing into one built with the old definition.

### Flagged pixels

By default, flagged pixels (hot pixels, cosmic rays, saturation) keep their
value in `sci_subtracted`: nothing is lost, but a few extreme values can
dominate the stamp's range unless `binary_mask` is applied downstream.
`--zero-flagged-pixels` sets them to 0 instead. `binary_mask` records them in
both cases; choose according to how the dataset will be used.

## Repository layout

```
src/
  main.py                CLI: acquire → extract → build → output
  config.py              constants, data directory, quadrant list
  dataset_builder.py     source selection, stamp cutting, dataset assembly, Hub push
  psf_model.py           PSF grid interpolation, residual PSF kernels
  euclid_vis_isotropic_min_psf.fits    21 × 21 isotropic reference PSF
utils/
  db_utils.py            Euclid archive queries, downloads, quadrant slicing
notebooks/
  Clipping_study.ipynb   pixel-value histograms of sci_subtracted
push.py                  concatenate the batches saved in $SCRATCH/datasets/batch* and push them
```

After a run, the data directory holds the full frames (`EUC_VIS_SWL-DET-*`,
`EUC_VIS_SWL-BKG-*`), the PSF model (`EUC_VIS_GRD-PSF-*`), the catalogues
(`catalogue_obs_<obs_id>.fits`), the per-quadrant files (`quadrant-data/`) and
the saved dataset (`dataset/`).

## Caveats

- Only the first dither (`00-1`) of each observation is used.
- PSF quadrant tiles must be 189 × 189 px (9 × 9 stamps of 21 × 21).
- `--merge` needs the Hub dataset to have the same columns as the new rows.
- On macOS, code that calls the build stage needs an
  `if __name__ == "__main__":` guard (multiprocessing uses spawn).
