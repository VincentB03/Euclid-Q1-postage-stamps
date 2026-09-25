"""Build the Euclid Q1 VIS postage-stamp dataset, one observation per process.

On macOS (spawn start method), scripts calling ``build_dataset`` /
``iter_records`` need an ``if __name__ == "__main__":`` guard.
"""

import functools
import glob
import multiprocessing
import os
import re
import warnings
from datetime import datetime, timezone

import numpy as np
from scipy import ndimage
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.table import Table
from astropy.wcs import WCS
from datasets import Array2D, Dataset, Features, Value

from config import (
    DATA_DIR,
    DISTANCE,
    FLAG_BITMASK,
    FLUX_MAX,
    FLUX_MIN,
    HF_REPO_ID,
    MAX_BAD_PIXEL_FRACTION,
    MAX_SPURIOUS_PROB,
    PIXEL_SIZE,
    POINT_PROB,
    PSF_SIZE,
    QUADRANT_DIR,
    QUADRANTS,
    SEGMENTATION_CENTER_BOX,
    SEGMENTATION_NSIGMA,
    STAMP_SIZE,
)
from psf_model import EuclidPSFModel

# Extracted science quadrant file name -> (core id, quadrant)
_FRAME_RE = re.compile(r'-DET-(\d{6}-\d{2}-\d+)-.*_([1-6]-[1-6]-[E-H])\.fits')

# Schema of one dataset row.
HF_FEATURES = Features({
    "obs_id": Value("string"),
    "quadrant": Value("string"),
    "ra": Value("float32"),
    "dec": Value("float32"),
    "obj_id": Value("int64"),
    "flux": Value("float32"),
    "snr": Value("float32"),
    "truncated": Value("bool"),
    "sci_subtracted": Array2D(shape=(STAMP_SIZE, STAMP_SIZE), dtype="float32"),
    "noise_map": Array2D(shape=(STAMP_SIZE, STAMP_SIZE), dtype="float32"),
    "binary_mask": Array2D(shape=(STAMP_SIZE, STAMP_SIZE), dtype="int32"),
    "psf_stamp": Array2D(shape=(PSF_SIZE, PSF_SIZE), dtype="float32"),
})


# ---------------------------------------------------------------------------
# Catalogue selection
# ---------------------------------------------------------------------------
def catalogue_path(obs_id, data_dir=DATA_DIR):
    """Local path of the catalogue FITS written by ``sync_observation_catalogs``."""
    return os.path.join(data_dir, f'catalogue_obs_{str(obs_id).zfill(6)}.fits')


def select_sources(catalogue):
    """Keep catalogue rows that pass the MER quality cuts."""
    mask = (
        (catalogue['point_like_prob'] <= POINT_PROB)
        & (catalogue['FLUX_VIS_UNIF'] >= FLUX_MIN)
        & (catalogue['FLUX_VIS_UNIF'] <= FLUX_MAX)
        & (catalogue['det_quality_flag'] == 0)
        & (catalogue['deblended_flag'] == 0)
        & (catalogue['spurious_prob'] <= MAX_SPURIOUS_PROB)
    )
    return catalogue[mask]


def apply_isolation_cut(sources, catalogue, distance=DISTANCE, pixel_size=PIXEL_SIZE):
    """Keep sources whose nearest catalogue neighbour is farther than
    ``(distance + neighbour_semimajor_axis // 2) * pixel_size`` arcsec."""
    if len(sources) == 0:
        return sources

    all_coords = SkyCoord(catalogue['right_ascension'], catalogue['declination'],
                          unit="deg", frame="icrs")
    src_coords = SkyCoord(sources['right_ascension'], sources['declination'],
                          unit="deg", frame="icrs")

    idx, d2d, _ = src_coords.match_to_catalog_sky(all_coords, nthneighbor=2)
    neighbour_size = catalogue['semimajor_axis'][idx]
    min_arcsec = (distance + neighbour_size // 2) * pixel_size
    return sources[d2d.arcsec > min_arcsec]


# ---------------------------------------------------------------------------
# Stamp extraction
# ---------------------------------------------------------------------------
def _resolve_files(obs_id, quadrant, quadrant_dir):
    """``(sci_path, bkg_path, psf_path)`` for one obs_id/quadrant, or None."""
    q_str = quadrant.replace(".", "-")
    # Padded and dash-delimited, or 2682 would also match '...T045100.762682Z'
    sci_files = glob.glob(os.path.join(quadrant_dir, f'*-DET-{str(obs_id).zfill(6)}-*_{q_str}.fits'))
    if not sci_files:
        return None

    sci_path = sci_files[0]
    m = _FRAME_RE.search(os.path.basename(sci_path))
    if not m:
        return None

    core_id, quadrant_str = m.group(1), m.group(2)
    bkg_files = glob.glob(os.path.join(quadrant_dir, f"*-BKG-{core_id}-*_{quadrant_str}.fits"))
    psf_files = glob.glob(os.path.join(quadrant_dir, f"*PSF*_{quadrant_str}.fits"))
    if not bkg_files or not psf_files:
        return None

    return sci_path, bkg_files[0], psf_files[0]


def _segment_source(sci_sub, rms_stamp, bad_pixels, center, nsigma=SEGMENTATION_NSIGMA,
                    center_box=SEGMENTATION_CENTER_BOX):
    """``(snr, truncated)`` of the source at ``center`` ``(row, col)``.

    The source is the union of the 8-connected regions of unflagged pixels
    above ``nsigma * rms`` that reach the central ``center_box`` box.
    ``snr = sum(signal) / sqrt(sum(rms**2))`` over it; ``truncated`` if it
    touches the stamp edge. No region in the box gives ``(0.0, False)``.
    """
    above = (sci_sub > nsigma * rms_stamp) & ~bad_pixels
    labels, _ = ndimage.label(above, structure=np.ones((3, 3)))
    row, col = center
    half = center_box // 2
    ids = np.unique(labels[row - half:row + half + 1, col - half:col + half + 1])
    ids = ids[ids > 0]
    if ids.size == 0:
        return 0.0, False

    seg = np.isin(labels, ids)
    snr = sci_sub[seg].sum() / np.sqrt((rms_stamp[seg] ** 2).sum())
    truncated = bool(seg[0].any() or seg[-1].any() or seg[:, 0].any() or seg[:, -1].any())
    return float(snr), truncated


def _extract_stamp(source, obs_id, quadrant, sci_data, bkg_data, flg_data, rms_data,
                   wcs, psf_model, stamp_size=STAMP_SIZE, zero_flagged_pixels=False,
                   min_snr=None, drop_truncated=False):
    """Build one record dict for ``source``, or None if it fails a cut."""
    position = SkyCoord(source['right_ascension'], source['declination'],
                        unit="deg", frame="icrs")
    try:
        cutout = Cutout2D(sci_data, position, (stamp_size, stamp_size),
                          wcs=wcs, mode='strict')

        y_slice, x_slice = cutout.slices_original
        bkg_stamp = bkg_data[y_slice, x_slice]
        flg_stamp = flg_data[y_slice, x_slice]
        # Native byte order: pyarrow rejects big-endian FITS slices
        rms_stamp = rms_data[y_slice, x_slice].astype(np.float32)

        bad_pixels = (flg_stamp & FLAG_BITMASK) != 0
        if bad_pixels.sum() / bad_pixels.size >= MAX_BAD_PIXEL_FRACTION:
            return None

        sci_sub = cutout.data.astype(float) - bkg_stamp
        if zero_flagged_pixels:
            sci_sub[bad_pixels] = 0.0
        binary_mask = np.where(bad_pixels, 0, 1).astype(np.int32)

        x_cut, y_cut = cutout.position_cutout
        snr, truncated = _segment_source(sci_sub, rms_stamp, bad_pixels,
                                         (int(round(y_cut)), int(round(x_cut))))
        if min_snr is not None and snr <= min_snr:
            return None
        if drop_truncated and truncated:
            return None

        x_pix, y_pix = cutout.position_original
        psf_stamp = psf_model.interpolate_at(float(x_pix), float(y_pix))

        return {
            "obs_id": str(obs_id),
            "quadrant": str(quadrant),
            "ra": float(source['right_ascension']),
            "dec": float(source['declination']),
            "obj_id": int(source['object_id']),
            "flux": float(source['FLUX_VIS_UNIF']),
            "snr": snr,
            "truncated": truncated,
            "sci_subtracted": sci_sub,
            "noise_map": rms_stamp,
            "binary_mask": binary_mask,
            "psf_stamp": psf_stamp,
        }
    except Exception:
        return None


def _process_quadrant(obs_id, quadrant, sci_path, bkg_path, psf_path, sources,
                      stamp_size=STAMP_SIZE, psf_size=PSF_SIZE, zero_flagged_pixels=False,
                      min_snr=None, drop_truncated=False):
    """Records for every source in ``sources`` that fits inside this quadrant."""
    with fits.open(psf_path) as hdul_psf:
        psf_raw = next(ext.data for ext in hdul_psf
                       if ext.data is not None and ext.data.ndim == 2)
    psf_model = EuclidPSFModel(psf_raw, stamp_size=psf_size)

    records = []
    with fits.open(sci_path, memmap=True) as hdul_sci, \
            fits.open(bkg_path, memmap=True) as hdul_bkg:
        sci_data = hdul_sci[f'{quadrant}.SCI'].data
        flg_data = hdul_sci[f'{quadrant}.FLG'].data
        rms_data = hdul_sci[f'{quadrant}.RMS'].data
        wcs = WCS(hdul_sci[f'{quadrant}.SCI'].header)
        bkg_data = hdul_bkg[1].data

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            for source in sources:
                record = _extract_stamp(source, obs_id, quadrant, sci_data, bkg_data,
                                        flg_data, rms_data, wcs, psf_model, stamp_size,
                                        zero_flagged_pixels=zero_flagged_pixels,
                                        min_snr=min_snr, drop_truncated=drop_truncated)
                if record is not None:
                    records.append(record)
    return records


# ---------------------------------------------------------------------------
# Per-observation worker + record generators
# ---------------------------------------------------------------------------
def process_obs_id(obs_id, data_dir=DATA_DIR, quadrant_dir=QUADRANT_DIR, quadrants=QUADRANTS,
                   zero_flagged_pixels=False, min_snr=None, drop_truncated=False,
                   verbose=False):
    """Every stamp record for one observation (one parallel work unit)."""
    cat_path = catalogue_path(obs_id, data_dir)
    if not os.path.exists(cat_path):
        if verbose:
            print(f"-> [obs {obs_id}] no catalogue found, skipping.")
        return []

    catalogue = Table.read(cat_path)
    sources = apply_isolation_cut(select_sources(catalogue), catalogue)
    if len(sources) == 0:
        if verbose:
            print(f"-> [obs {obs_id}] no source survives the cuts, skipping.")
        return []

    n_quadrants = len(quadrants)
    if verbose:
        print(f"-> [obs {obs_id}] {len(sources)} source(s) after cuts; "
              f"scanning {n_quadrants} quadrant(s)...")

    records = []
    for i, quadrant in enumerate(quadrants, start=1):
        resolved = _resolve_files(obs_id, quadrant, quadrant_dir)
        if resolved is None:
            if verbose:
                print(f"   [obs {obs_id}] quadrant {i}/{n_quadrants} ({quadrant}): "
                      f"missing file(s), skipped.")
            continue
        sci_path, bkg_path, psf_path = resolved
        new_records = _process_quadrant(obs_id, quadrant, sci_path, bkg_path, psf_path, sources,
                                        zero_flagged_pixels=zero_flagged_pixels,
                                        min_snr=min_snr, drop_truncated=drop_truncated)
        records.extend(new_records)
        if verbose:
            print(f"   [obs {obs_id}] quadrant {i}/{n_quadrants} ({quadrant}): "
                  f"{len(new_records)} stamp(s) -> {len(records)} total")

    if verbose:
        print(f"-> [obs {obs_id}] done: {len(records)} stamp(s).")
    return records


def iter_records(obs_ids, data_dir=DATA_DIR, quadrant_dir=QUADRANT_DIR,
                 quadrants=QUADRANTS, processes=None, zero_flagged_pixels=False,
                 min_snr=None, drop_truncated=False, verbose=False):
    """Yield one record per stamp; ``processes=1`` runs sequentially, else a pool
    (default: all cores)."""
    obs_ids = list(obs_ids)
    worker = functools.partial(process_obs_id, data_dir=data_dir,
                               quadrant_dir=quadrant_dir, quadrants=list(quadrants),
                               zero_flagged_pixels=zero_flagged_pixels, min_snr=min_snr,
                               drop_truncated=drop_truncated, verbose=verbose)

    if processes == 1:
        for j, obs_id in enumerate(obs_ids, start=1):
            if verbose:
                print(f"[build] observation {j}/{len(obs_ids)}: obs_id {obs_id}")
            yield from worker(obs_id)
        return

    if verbose:
        n_workers = processes or multiprocessing.cpu_count()
        print(f"[build] processing {len(obs_ids)} observation(s) with {n_workers} worker(s)...")
    with multiprocessing.Pool(processes=processes or multiprocessing.cpu_count()) as pool:
        for j, batch in enumerate(pool.imap_unordered(worker, obs_ids), start=1):
            if verbose:
                print(f"[build] {j}/{len(obs_ids)} observation(s) done, "
                      f"{len(batch)} stamp(s) in this batch")
            yield from batch


# ---------------------------------------------------------------------------
# Dataset assembly
# ---------------------------------------------------------------------------
def build_dataset(obs_ids, quadrants=QUADRANTS, data_dir=DATA_DIR,
                  quadrant_dir=QUADRANT_DIR, processes=None, zero_flagged_pixels=False,
                  min_snr=None, drop_truncated=False, verbose=False):
    """Build the ``datasets.Dataset`` of stamps for ``obs_ids``.

    ``zero_flagged_pixels`` sets flagged pixels of ``sci_subtracted`` to 0
    (default: keep their value; ``binary_mask`` records them either way).
    ``min_snr`` drops sources with ``snr <= min_snr``, ``drop_truncated`` those
    touching the stamp edge (see ``_segment_source``).
    """
    return Dataset.from_generator(
        iter_records,
        features=HF_FEATURES,
        gen_kwargs={
            "obs_ids": list(obs_ids),
            "data_dir": data_dir,
            "quadrant_dir": quadrant_dir,
            "quadrants": tuple(quadrants),
            "processes": processes,
            "zero_flagged_pixels": zero_flagged_pixels,
            "min_snr": min_snr,
            "drop_truncated": drop_truncated,
            "verbose": verbose,
        },
    )


def drop_duplicate_obj_ids(dataset, verbose=True):
    """Return ``dataset`` with at most one row per ``obj_id`` (first occurrence wins)."""
    seen = set()
    keep = []
    for i, obj_id in enumerate(dataset["obj_id"]):
        if obj_id in seen:
            continue
        seen.add(obj_id)
        keep.append(i)
    if verbose:
        n_removed = len(dataset) - len(keep)
        if n_removed:
            print(f"[drop-duplicates] removed {n_removed} duplicate "
                  f"obj_id row(s); {len(keep)} unique row(s) kept")
        else:
            print(f"[drop-duplicates] no duplicate obj_id found; "
                  f"{len(keep)} row(s) kept")
    return dataset.select(keep)


def push_dataset(dataset, repo_id=HF_REPO_ID, private=True, token=None):
    """Push to the Hub. Token from ``token`` or the ``HF_TOKEN`` env var."""
    dataset.push_to_hub(repo_id, private=private,
                        token=token or os.environ.get("HF_TOKEN"))


def merge_and_push(new_dataset, repo_id=HF_REPO_ID, private=True, token=None,
                   drop_duplicates=False, verbose=True):
    """Append to the existing Hub dataset and push; with ``drop_duplicates``,
    existing rows win over colliding new ones."""
    from datasets import concatenate_datasets, load_dataset

    token = token or os.environ.get("HF_TOKEN")
    existing = load_dataset(repo_id, split="train", token=token)
    merged = concatenate_datasets([existing, new_dataset])
    if drop_duplicates:
        merged = drop_duplicate_obj_ids(merged, verbose=verbose)
    push_dataset(merged, repo_id, private, token)
    return merged


# ---------------------------------------------------------------------------
# Hub dataset card
# ---------------------------------------------------------------------------
_CARD_START = "<!-- BUILD-INFO:START -->"
_CARD_END = "<!-- BUILD-INFO:END -->"


def render_build_info(dataset, params=None, command=None):
    """Markdown "Build info" block for the Hub dataset card."""
    obs_ids = sorted(set(dataset["obs_id"]))
    columns = "\n".join(
        f"| `{name}` | {feat} |" for name, feat in dataset.features.items()
    )
    param_rows = "\n".join(
        f"| `{k}` | {v} |" for k, v in (params or {}).items()
    ) or "| _(none)_ | |"
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines = [
        _CARD_START,
        "## Build info",
        "",
        f"_Automatically appended by `src/main.py` at push time ({stamp}). "
        "Everything above this line is preserved as-is; only this block is "
        "regenerated on each push._",
        "",
        f"- **Rows in this dataset:** {len(dataset)}",
        f"- **Observation IDs ({len(obs_ids)}):** {', '.join(obs_ids)}",
    ]
    if command:
        lines += [
            "",
            "### Command used to build this dataset",
            "",
            "```bash",
            command,
            "```",
        ]
    lines += [
        "",
        "### Columns",
        "",
        "| column | feature |",
        "| --- | --- |",
        columns,
        "",
        "### Run parameters",
        "",
        "| parameter | value |",
        "| --- | --- |",
        param_rows,
        _CARD_END,
    ]
    return "\n".join(lines)


def update_dataset_card(repo_id, dataset, params=None, command=None, token=None):
    """Replace the ``BUILD-INFO`` block of the Hub dataset card, keeping the rest of the text."""
    from huggingface_hub import DatasetCard

    token = token or os.environ.get("HF_TOKEN")
    card = DatasetCard.load(repo_id, token=token)

    text = re.sub(
        re.escape(_CARD_START) + r".*?" + re.escape(_CARD_END),
        "",
        card.text or "",
        flags=re.DOTALL,
    ).rstrip()

    block = render_build_info(dataset, params, command)
    card.text = f"{text}\n\n{block}\n" if text else f"{block}\n"

    card.push_to_hub(repo_id, token=token)


def add_psf_residual(dataset, reference_psf_path=None, batch_size=256):
    """Add a ``psf_residual`` column: kernel such that ``psf_ref (*) kernel = psf_stamp``."""
    from psf_model import centered_fft2, compute_psf_residual, load_reference_psf

    psf_ref = (load_reference_psf() if reference_psf_path is None
               else load_reference_psf(reference_psf_path))
    ref_fft = centered_fft2(psf_ref)

    def _batch(batch):
        stamps = np.asarray(batch["psf_stamp"], dtype=np.float64)
        kernels = compute_psf_residual(stamps, psf_ref, ref_fft=ref_fft)
        batch["psf_residual"] = [k.astype(np.float32) for k in kernels]
        return batch

    return dataset.map(_batch, batched=True, batch_size=batch_size)
