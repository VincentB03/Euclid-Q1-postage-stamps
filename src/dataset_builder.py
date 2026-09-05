"""Build the Euclid Q1 VIS postage-stamp dataset.

For every selected, isolated source in each observation catalogue, cut a
background-subtracted ``STAMP_SIZE`` science stamp with its noise map,
bad-pixel mask and locally interpolated PSF, then assemble the records into a
``datasets.Dataset``. Work is parallelised one observation per process.

Multiprocessing uses 'spawn' on macOS, so scripts that call
``build_dataset`` / ``iter_records`` must guard the entry point with
``if __name__ == "__main__":``.
"""

import functools
import glob
import multiprocessing
import os
import re
import warnings
from datetime import datetime, timezone

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.table import Table
from astropy.wcs import WCS
from datasets import Array2D, Dataset, Features, Value

from config import (
    DATA_DIR,
    DISTANCE,
    EMPTY_STAMP_CENTER_FRAC,
    EMPTY_STAMP_SNR_THRESHOLD,
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
    STAMP_SIZE,
)
from psf_model import EuclidPSFModel

# Filename pattern of an extracted science quadrant (see utils/db_utils.py).
_FRAME_RE = re.compile(r'-DET-(\d{6}-\d{2}-\d+)-.*_([1-6]-[1-6]-[E-H])\.fits')

# Schema of one dataset row.
HF_FEATURES = Features({
    "obs_id": Value("string"),
    "quadrant": Value("string"),
    "ra": Value("float32"),
    "dec": Value("float32"),
    "obj_id": Value("int64"),
    "flux": Value("float32"),
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
    """Drop sources whose 2nd-nearest catalogue neighbour is too close.

    Minimum separation (arcsec) is
    ``(distance + neighbour_semimajor_axis // 2) * pixel_size``.
    """
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
    sci_files = glob.glob(os.path.join(quadrant_dir, f'*DET*{obs_id}*_{q_str}.fits'))
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


def _extract_stamp(source, obs_id, quadrant, sci_data, bkg_data, flg_data, rms_data,
                   wcs, psf_model, stamp_size=STAMP_SIZE, zero_flagged_pixels=False):
    """Build one record dict for ``source``, or None if it fails a cut."""
    position = SkyCoord(source['right_ascension'], source['declination'],
                        unit="deg", frame="icrs")
    try:
        cutout = Cutout2D(sci_data, position, (stamp_size, stamp_size),
                          wcs=wcs, mode='strict')

        y_slice, x_slice = cutout.slices_original
        bkg_stamp = bkg_data[y_slice, x_slice]
        flg_stamp = flg_data[y_slice, x_slice]
        # .astype() forces native byte order: FITS data is big-endian, and raw
        # slices (unlike sci_sub/binary_mask below, which go through an
        # arithmetic op that already converts) keep that byte order, which
        # pyarrow's writer rejects ("Byte-swapped arrays not supported").
        rms_stamp = rms_data[y_slice, x_slice].astype(np.float32)

        bad_pixels = (flg_stamp & FLAG_BITMASK) != 0
        if bad_pixels.sum() / bad_pixels.size >= MAX_BAD_PIXEL_FRACTION:
            return None

        # By default, flagged pixels keep their real (science - background) value;
        # they are only counted for the fraction cut above and recorded in
        # binary_mask. With zero_flagged_pixels=True they are zeroed out here
        # instead (see build_dataset's docstring for the tradeoff).
        sci_sub = cutout.data.astype(float) - bkg_stamp
        if zero_flagged_pixels:
            sci_sub[bad_pixels] = 0.0
        binary_mask = np.where(bad_pixels, 0, 1).astype(np.int32)

        x_pix, y_pix = cutout.position_original
        psf_stamp = psf_model.interpolate_at(float(x_pix), float(y_pix))

        return {
            "obs_id": str(obs_id),
            "quadrant": str(quadrant),
            "ra": float(source['right_ascension']),
            "dec": float(source['declination']),
            "obj_id": int(source['object_id']),
            "flux": float(source['FLUX_VIS_UNIF']),
            "sci_subtracted": sci_sub,
            "noise_map": rms_stamp,
            "binary_mask": binary_mask,
            "psf_stamp": psf_stamp,
        }
    except Exception:
        return None


def _process_quadrant(obs_id, quadrant, sci_path, bkg_path, psf_path, sources,
                      stamp_size=STAMP_SIZE, psf_size=PSF_SIZE, zero_flagged_pixels=False):
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
                                        zero_flagged_pixels=zero_flagged_pixels)
                if record is not None:
                    records.append(record)
    return records


# ---------------------------------------------------------------------------
# Per-observation worker + record generators
# ---------------------------------------------------------------------------
def process_obs_id(obs_id, data_dir=DATA_DIR, quadrant_dir=QUADRANT_DIR, quadrants=QUADRANTS,
                   zero_flagged_pixels=False, verbose=False):
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
                                        zero_flagged_pixels=zero_flagged_pixels)
        records.extend(new_records)
        if verbose:
            print(f"   [obs {obs_id}] quadrant {i}/{n_quadrants} ({quadrant}): "
                  f"{len(new_records)} stamp(s) -> {len(records)} total")

    if verbose:
        print(f"-> [obs {obs_id}] done: {len(records)} stamp(s).")
    return records


def iter_records(obs_ids, data_dir=DATA_DIR, quadrant_dir=QUADRANT_DIR,
                 quadrants=QUADRANTS, processes=None, zero_flagged_pixels=False,
                 verbose=False):
    """Yield one record dict per stamp, one observation per worker.

    ``processes=1`` runs sequentially (handy for debugging); otherwise an
    ``imap_unordered`` pool of ``processes`` workers (default: all cores).
    """
    obs_ids = list(obs_ids)
    worker = functools.partial(process_obs_id, data_dir=data_dir,
                               quadrant_dir=quadrant_dir, quadrants=list(quadrants),
                               zero_flagged_pixels=zero_flagged_pixels, verbose=verbose)

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
                  verbose=False):
    """Materialise the postage-stamp dataset from the record generator.

    ``zero_flagged_pixels`` controls what ``sci_subtracted`` holds at pixels
    flagged bad in ``FLG`` (see ``FLAG_BITMASK``):

    - ``False`` (default): keep their real (science - background) value.
      Since a handful of defective pixels can carry extreme values (hot
      pixels, cosmic rays, saturation), they can dominate the stamp's value
      range even though they are a small minority of it -- a problem if
      something naively looks at min/max or a raw display, even though a
      generative model's loss can be told to ignore them via
      ``binary_mask``.
    - ``True``: zero them out directly in ``sci_subtracted`` (matches the
      original exploratory notebook this pipeline was ported from).

    Neither option is strictly better: zeroing loses the real pixel value
    (useful e.g. for inpainting-style tasks) but gives a display/statistics
    that isn't skewed by defects; keeping it preserves information but
    requires consistently applying ``binary_mask`` downstream. Pick based on
    what consumes the dataset.
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


def _central_peak_snr(sci, noise, center_frac):
    """Peak ``sci / noise`` in a ``center_frac``-sized box at the middle of the stamp."""
    h, w = sci.shape
    bh, bw = max(1, int(h * center_frac)), max(1, int(w * center_frac))
    cy, cx = h // 2, w // 2
    y0, y1 = cy - bh // 2, cy + bh // 2 + 1
    x0, x1 = cx - bw // 2, cx + bw // 2 + 1
    snr = sci[y0:y1, x0:x1] / np.maximum(noise[y0:y1, x0:x1], 1e-12)
    return snr.max()


def drop_empty_stamps(dataset, center_frac=EMPTY_STAMP_CENTER_FRAC,
                      snr_threshold=EMPTY_STAMP_SNR_THRESHOLD, verbose=True):
    """Drop stamps with no significant source at the center (no galaxy to model).

    Peak signal-to-noise (``sci_subtracted / noise_map``) is measured in a
    ``center_frac``-sized box at the middle of the stamp; rows below
    ``snr_threshold`` there are dropped.
    """
    keep = [
        i for i, (sci, noise) in enumerate(zip(dataset["sci_subtracted"], dataset["noise_map"]))
        if _central_peak_snr(np.asarray(sci), np.asarray(noise), center_frac) >= snr_threshold
    ]
    if verbose:
        n_removed = len(dataset) - len(keep)
        if n_removed:
            print(f"[drop-empty-stamps] removed {n_removed} stamp(s) with no significant "
                  f"source at center; {len(keep)} row(s) kept")
        else:
            print(f"[drop-empty-stamps] no empty stamp found; {len(keep)} row(s) kept")
    return dataset.select(keep)


def push_dataset(dataset, repo_id=HF_REPO_ID, private=True, token=None):
    """Push to the Hub. Token from ``token`` or the ``HF_TOKEN`` env var."""
    dataset.push_to_hub(repo_id, private=private,
                        token=token or os.environ.get("HF_TOKEN"))


def merge_and_push(new_dataset, repo_id=HF_REPO_ID, private=True, token=None,
                   drop_duplicates=False, verbose=True):
    """Concatenate with the existing Hub dataset, then push the union back.

    With ``drop_duplicates`` the merged dataset is reduced to one row per
    ``obj_id``; existing rows come first, so they win over new colliding ones.
    """
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
    """Markdown block describing how ``dataset`` was produced by ``src/main.py``."""
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
    """Append/refresh the ``BUILD-INFO`` block at the end of the Hub dataset card.

    Any hand-written presentation in the card is kept untouched: a previous
    ``BUILD-INFO`` block (wherever it sits) is stripped, then a fresh one is
    appended after the existing text.
    """
    import re as _re

    from huggingface_hub import DatasetCard

    token = token or os.environ.get("HF_TOKEN")
    card = DatasetCard.load(repo_id, token=token)

    text = _re.sub(
        _re.escape(_CARD_START) + r".*?" + _re.escape(_CARD_END),
        "",
        card.text or "",
        flags=_re.DOTALL,
    ).rstrip()

    block = render_build_info(dataset, params, command)
    card.text = f"{text}\n\n{block}\n" if text else f"{block}\n"

    card.push_to_hub(repo_id, token=token)


def add_psf_residual(dataset, reference_psf_path=None, batch_size=256):
    """Add a ``psf_residual`` column: kernel such that ``psf_ref (*) kernel = psf_stamp``.

    Needs the isotropic reference PSF FITS (see ``psf_model.DEFAULT_REFERENCE_PSF``).
    """
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
