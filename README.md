# Euclid Q1 VIS postage stamps

A pipeline that turns Euclid Q1 VIS calibrated imaging into a machine-learning
ready dataset of **64 × 64 postage stamps** of isolated sources, each paired with
its noise map, a bad-pixel mask, and a locally interpolated PSF (optionally plus
a residual PSF kernel).

The pipeline has three parts:

1. **Acquisition** — query the Euclid Science Archive, download the calibrated
   science frames, their background frames, the global VIS PSF model and the
   cross-matched source catalogues, then slice everything into per-quadrant FITS
   files.
2. **Build** — for every selected, isolated catalogue source, cut a
   background-subtracted stamp with its noise and mask, interpolate the PSF at
   the source position, and collect the records into a
   [`datasets.Dataset`](https://huggingface.co/docs/datasets). Optionally
   de-duplicated down to one row per `obj_id`.
3. **Output** — save the dataset locally and/or push it to the Hugging Face Hub.

---

## Dataset schema

One row per source (`src/dataset_builder.py::HF_FEATURES`):

| Field            | Type                | Description |
|------------------|---------------------|-------------|
| `obs_id`         | `string`            | Euclid observation ID |
| `quadrant`       | `string`            | VIS quadrant, e.g. `3-4.E` |
| `ra`, `dec`      | `float32`           | Source coordinates (deg, ICRS) |
| `obj_id`         | `int64`             | MER catalogue `object_id` |
| `flux`           | `float32`           | `FLUX_VIS_UNIF` from the PHZ catalogue |
| `sci_subtracted` | `float32 [64, 64]`  | Science stamp minus background (flagged pixels keep their real value by default, or are zeroed with `--zero-flagged-pixels`; see `binary_mask`) |
| `noise_map`      | `float32 [64, 64]`  | RMS stamp |
| `binary_mask`    | `int32 [64, 64]`    | 1 = valid pixel, 0 = flagged pixel |
| `psf_stamp`      | `float32 [21, 21]`  | PSF interpolated at the source position, normalised to sum 1 |
| `psf_residual`   | `float32 [21, 21]`  | *(optional)* kernel `k` such that `reference_psf (*) k = psf_stamp` |

Stamp size, PSF size and every selection threshold live in `src/config.py`.

---

## Requirements

* Python 3.9+
* Packages in `requirements.txt`:

  ```bash
  pip install -r requirements.txt
  ```

  (`astroquery`, `astropy`, `numpy`, `pandas`, `datasets`, `huggingface_hub`)

* Anonymous access is enough for Q1 archive queries and downloads. If a product
  requires authentication, log in first in a Python shell:

  ```python
  from astroquery.esa.euclid import Euclid
  Euclid.login()  # prompts for your Euclid SAS credentials
  ```

* Pushing to the Hub needs a token, via `--hf-token` or the `HF_TOKEN`
  environment variable (or `huggingface-cli login`).

---

## Configuration

### Data directory

All products are read from / written to a single directory, resolved in this
order:

1. the `EUCLID_DATA_DIR` environment variable (any path — a Google Drive mount
   or a plain local folder);
2. otherwise `/content/drive/MyDrive/Q1_VIS_CALIBRATED_DB` when a Colab Drive
   mount is detected;
3. otherwise `<repo>/data/Q1_VIS_CALIBRATED_DB`.

```bash
export EUCLID_DATA_DIR=/data/euclid/q1        # local run
# or, on Colab
export EUCLID_DATA_DIR=/content/drive/MyDrive/Q1_VIS_CALIBRATED_DB
```

### Key constants (`src/config.py`)

| Constant | Default | Meaning |
|---|---|---|
| `STAMP_SIZE` | `64` | Science / noise / mask stamp side (px) |
| `PSF_SIZE` | `21` | PSF stamp side (px) |
| `POINT_PROB` | `0.5` | Max `point_like_prob` kept (more point-like → dropped) |
| `DISTANCE` | `46` | Stamp "diagonal" used by the isolation cut (px) |
| `PIXEL_SIZE` | `0.1` | VIS pixel scale (arcsec/px) |
| `FLUX_MIN`, `FLUX_MAX` | `0.57544`, `575.44` | `FLUX_VIS_UNIF` window |
| `MAX_SPURIOUS_PROB` | `0.2` | Max `spurious_prob` kept |
| `FLAG_BITMASK` | `1 \| 262144` | VIS `FLG` bits treated as bad pixels (bit 0 + bit 18) |
| `MAX_BAD_PIXEL_FRACTION` | `0.10` | Drop a stamp at/above this fraction of flagged pixels |
| `HF_REPO_ID` | `VincentB03/euclid-Q1-VF` | Default Hub dataset repo |
| `QUADRANTS` | 144 entries | `i-j.L` for `i,j ∈ 1..6`, `L ∈ {E,F,G,H}` |

---

## Quick start

Run the whole pipeline through the orchestrator:

```bash
# Full run on the first 17 "optimal" observations, 4 build workers, save locally
python src/main.py --limit 17 --processes 4

# Data already downloaded & sliced: only (re)build and push a fresh dataset
python src/main.py --limit 17 --skip-acquire --skip-extract --push

# Append newly built stamps to the existing Hub dataset
python src/main.py --obs-ids 2698 2699 --skip-acquire --skip-extract --merge

# Same, but drop duplicate obj_id rows within the new batch and against the
# existing Hub dataset (existing rows win over colliding new ones)
python src/main.py --obs-ids 2698 2699 --skip-acquire --skip-extract \
  --merge --drop-duplicates
```

By default nothing is uploaded: the dataset is written to
`<DATA_DIR>/dataset` with `datasets.save_to_disk`. Uploading happens only with
`--push` (fresh) or `--merge` (concatenate with the existing Hub dataset, then
push).

### `src/main.py` options

| Group | Option | Effect |
|---|---|---|
| Observations | `--obs-ids ID [ID ...]` | Process exactly these observation IDs |
| | `--obs-ids-file PATH` | One observation ID per line |
| | `--limit N` | Keep the first `N` optimal observations |
| Stages | `--skip-acquire` | Frames / backgrounds / PSF / catalogues already downloaded |
| | `--skip-extract` | Per-quadrant FITS already sliced |
| | `--skip-build` | Stop after acquisition + extraction |
| Build | `--processes N` | Build workers (default: all cores; `1` = sequential) |
| | `--no-residual` | Do not add the `psf_residual` column |
| | `--drop-duplicates` | Enforce at most one row per `obj_id` in the final dataset |
| | `--zero-flagged-pixels` | Zero out flagged pixels in `sci_subtracted` instead of keeping their real value |
| | `--reference-psf PATH` | Isotropic reference PSF FITS (default: `src/euclid_vis_isotropic_min_psf.fits`) |
| Output | `--push` | Push a fresh dataset to the Hub |
| | `--merge` | Concatenate with the existing Hub dataset, then push |
| | `--save-to DIR` | `save_to_disk` target (default when not pushing: `<DATA_DIR>/dataset`) |
| | `--repo-id ID` | Hub dataset repo id |
| | `--public` | Push as a public dataset (default: private) |
| | `--hf-token TOKEN` | Overrides `HF_TOKEN` |
| | `--quiet` | Less logging |

> `src/main.py` uses `multiprocessing` in the build stage. On macOS (spawn
> start method) any script that calls into the build must guard its entry point
> with `if __name__ == "__main__":` — `src/main.py` already does.

---

## How the pipeline works

### 1. Resolve observation IDs

`get_optimal_observation_ids()` runs

```sql
SELECT DISTINCT observation_id, ra, dec
FROM q1.calibrated_frame
WHERE instrument_name = 'VIS'
```

keeps the first dither of each observation, rounds `ra`/`dec` (1 decimal by
default) and de-duplicates spatially, so the result is a minimal set of
observations that tile the Q1 VIS footprint with little overlap.

### 2. Acquire

| Function | Archive source | Output |
|---|---|---|
| `sync_calibrated_frames` | `q1.calibrated_frame`, VIS, dither `00-1` | `EUC_VIS_SWL-DET-*.fits` |
| `sync_background_frames` | `q1.aux_calibrated`, `stype='BKG'` | `EUC_VIS_SWL-BKG-*.fits` |
| `sync_psf_model` | `q1.aux_calibrated`, `stype='PSF MODEL'` | `EUC_VIS_GRD-PSF-*.fits` |
| `sync_observation_catalogs` | `catalogue.mer_catalogue` ⋈ `catalogue.phz_photo_z` ⋈ `catalogue.mer_morphology` | `catalogue_obs_<obs_id>.fits` |

Each function checks the data directory first and only downloads what is
missing. `sync_observation_catalogs` derives the sky footprint of an
observation from the WCS of its `*.SCI` extensions, then pulls a cross-matched
catalogue (photometry, morphology, star/galaxy flags, quality flags) inside
that RA/Dec box.

### 3. Extract quadrants

The VIS focal plane is 6 × 6 CCDs, each split into 4 quadrants (E/F/G/H) — 144
quadrants of roughly 2048 × 2066 px. The `extract_quadrants_from_*` functions
split each full-frame FITS into standalone per-quadrant files under
`quadrant-data/`:

* science → primary header + `{q}.SCI` + `{q}.RMS` + `{q}.FLG`
* background → primary header + `{q}`
* PSF model → primary header + `{q}` (saved as `PSF_{q}.fits`)

### 4. Build stamps

Per observation (`process_obs_id`), then per quadrant:

1. **Select sources** (`select_sources`) — keep catalogue rows with
   `point_like_prob ≤ POINT_PROB`, `FLUX_MIN ≤ FLUX_VIS_UNIF ≤ FLUX_MAX`,
   `det_quality_flag == 0`, `deblended_flag == 0`,
   `spurious_prob ≤ MAX_SPURIOUS_PROB`.
2. **Isolation cut** (`apply_isolation_cut`) — using
   `match_to_catalog_sky(nthneighbor=2)` against the *full* catalogue, keep only
   sources whose nearest neighbour is farther than
   `(DISTANCE + neighbour_semimajor_axis // 2) × PIXEL_SIZE` arcsec.
3. **Cut the stamp** (`_extract_stamp`) — `Cutout2D(..., mode='strict')`, so
   sources too close to a quadrant edge are skipped. From the same pixel slice:
   subtract the background and build the bad-pixel mask from
   `FLG & FLAG_BITMASK`. Flagged pixels are counted only to drop the stamp when
   they reach `≥ MAX_BAD_PIXEL_FRACTION` of it; by default they otherwise keep
   their real `sci_subtracted` value and are recorded in `binary_mask` (see
   `--zero-flagged-pixels` below for the alternative).
4. **Interpolate the PSF** — `EuclidPSFModel.interpolate_at(x, y)` at the source
   pixel position gives a normalised 21 × 21 PSF stamp.

`build_dataset(obs_ids, processes=...)` wraps `iter_records` in
`Dataset.from_generator`. `iter_records` runs one worker per observation with
`multiprocessing.Pool.imap_unordered`; pass `processes=1` for a single-process
run.

#### Flagged pixels: real value vs. zeroed out (`--zero-flagged-pixels`)

Flagged pixels (hot pixels, cosmic rays, saturation — anything caught by
`FLAG_BITMASK`) are a small minority of a stamp, but their **value** can be
extreme. By default (`--zero-flagged-pixels` off) `sci_subtracted` keeps
their real, unmodified value — nothing is thrown away, but a handful of
outlier pixels can then dominate the value range of the whole stamp (a raw
`imshow`, `min`/`max`, or any statistic that isn't `binary_mask`-aware will
be skewed by them), even though a generative model's loss can be told to
ignore them via `binary_mask` at training time.

With `--zero-flagged-pixels`, those pixels are set to `0.0` directly in
`sci_subtracted` (matching the exploratory notebook this pipeline was
originally ported from) — the stamp's value range is no longer skewed by
defects, at the cost of discarding their real value (which some tasks, e.g.
inpainting-style training, may actually want).

Neither is strictly better — **it's up to whoever builds the dataset to pick
based on what will consume it.** `binary_mask` records which pixels were
flagged either way, so the choice is always recoverable/reproducible from the
dataset itself.

### 5. De-duplicate (optional, `--drop-duplicates`)

`drop_duplicate_obj_ids(dataset)` reduces the dataset to at most one row per
`obj_id` (first occurrence wins) — useful because a source near a quadrant
boundary can be selected from more than one quadrant, and an observation can
overlap its neighbours. When it actually removes rows, it prints how many:

```
[drop-duplicates] removed 2 duplicate obj_id row(s); 4 unique row(s) kept
```

Applied twice when relevant:

* right after the build, on the freshly built dataset alone;
* again in `merge_and_push` when both `--merge` and `--drop-duplicates` are
  set, on the concatenated dataset — existing Hub rows come first, so they win
  over colliding new ones.

### 6. Residual PSF (optional)

`add_psf_residual(dataset)` adds a `psf_residual` column. Each `psf_stamp` is
modelled as `reference_psf (*) kernel`, and the kernel is recovered by dividing
the two in Fourier space (`compute_psf_residual`). The reference PSF is the
isotropic 21 × 21 stamp in `src/euclid_vis_isotropic_min_psf.fits`.
`reconvolve_psf` inverts the operation and is useful to check that a kernel
round-trips back to its stamp.

### 7. Output

* `dataset.save_to_disk(dir)` — local Arrow dataset.
* `push_dataset(dataset)` — `push_to_hub`, token from `HF_TOKEN`.
* `merge_and_push(dataset, drop_duplicates=...)` — `load_dataset(repo_id)` +
  `concatenate_datasets` + optional dedup + `push_to_hub`, for incrementally
  growing the Hub dataset across runs.

---

## Repository layout

```
src/
├── config.py             constants + data-directory resolution + QUADRANTS
├── psf_model.py           EuclidPSFModel + reference/residual PSF helpers
├── dataset_builder.py     source selection, stamp extraction, Dataset assembly
├── main.py                CLI orchestrator (acquire → extract → build → output)
└── euclid_vis_isotropic_min_psf.fits   21×21 isotropic reference PSF
utils/
└── db_utils.py            Euclid archive queries, downloads, quadrant slicing
requirements.txt
```

### Data directory layout (after a run)

```
$EUCLID_DATA_DIR/
├── EUC_VIS_SWL-DET-<obs>-00-1-*.fits     full science frames
├── EUC_VIS_SWL-BKG-<obs>-00-1-*.fits     full background frames
├── EUC_VIS_GRD-PSF-*.fits                global VIS PSF model
├── catalogue_obs_<obs>.fits              per-observation cross-matched catalogue
├── quadrant-data/
│   ├── ...DET-...*_<q>.fits              per-quadrant SCI + RMS + FLG
│   ├── ...BKG-...*_<q>.fits              per-quadrant background
│   └── PSF_<q>.fits                      per-quadrant PSF grid
└── dataset/                              saved datasets.Dataset (default output)
```

---

## Programmatic use

Run stage by stage from Python (make sure `src/` and the repo root are on
`sys.path`, as `src/main.py` does):

```python
from utils.db_utils import (
    get_optimal_observation_ids, sync_calibrated_frames, sync_background_frames,
    sync_psf_model, sync_observation_catalogs,
    extract_quadrants_from_frames, extract_quadrants_from_backgrounds,
    extract_quadrants_from_psf,
)
from config import DATA_DIR, QUADRANT_DIR, QUADRANTS
from dataset_builder import build_dataset, add_psf_residual, push_dataset

obs_ids = get_optimal_observation_ids()[:17]

frame_files = sync_calibrated_frames(obs_ids, DATA_DIR)
sync_background_frames(frame_files, obs_ids, DATA_DIR)
psf_path = sync_psf_model(DATA_DIR)
sync_observation_catalogs(obs_ids, DATA_DIR)

extract_quadrants_from_frames(DATA_DIR, QUADRANT_DIR, QUADRANTS)
extract_quadrants_from_backgrounds(DATA_DIR, QUADRANT_DIR, QUADRANTS)
extract_quadrants_from_psf(psf_path, QUADRANT_DIR, QUADRANTS)

ds = build_dataset(obs_ids, processes=4)
ds = add_psf_residual(ds)
ds.save_to_disk(f"{DATA_DIR}/dataset")
# push_dataset(ds)   # needs HF_TOKEN
```

Load it back:

```python
from datasets import load_from_disk
ds = load_from_disk(f"{DATA_DIR}/dataset")
row = ds[0]
row["sci_subtracted"]   # 64 x 64
row["psf_stamp"]        # 21 x 21
```

---

## Notes and caveats

* **PSF tile shape.** `EuclidPSFModel` expects each per-quadrant PSF tile to be
  `189 × 189` (a 9 × 9 grid of 21 × 21 stamps). A different layout raises a
  reshape error.