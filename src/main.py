"""End-to-end orchestrator for the Euclid Q1 VIS postage-stamp database.

Stages, run in order (each can be skipped once done):
  1. obs       resolve the list of observation IDs
  2. acquire   download calibrated frames, backgrounds, PSF model, catalogues
  3. extract   slice per-quadrant FITS (SCI/RMS/FLG, BKG, PSF)
  4. build     cut background-subtracted stamps into a ``datasets.Dataset``
  5. dedup     enforce one row per obj_id (optional, --drop-duplicates)
  6. residual  add the ``psf_residual`` column (optional)
  7. output    save locally and/or push to the Hugging Face Hub
               (a push also refreshes the "Build info" section of the Hub
               README: columns, obs_ids, row count, run parameters)

Hub pushes read the token from ``--hf-token`` or the ``HF_TOKEN`` env var.

Examples
--------
    # full run on the first 17 optimal observations, 4 build workers, save locally
    python src/main.py --limit 17 --processes 4

    # data already downloaded & sliced: only (re)build + push a fresh dataset
    python src/main.py --limit 17 --skip-acquire --skip-extract --push

    # append newly built stamps to the existing Hub dataset
    python src/main.py --obs-ids 2698 2699 --skip-acquire --skip-extract --merge
"""

import argparse
import os
import sys

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SRC_DIR)
for _p in (_SRC_DIR, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config import DATA_DIR, HF_REPO_ID, QUADRANT_DIR, QUADRANTS  # noqa: E402
from dataset_builder import (  # noqa: E402
    add_psf_residual,
    build_dataset,
    drop_duplicate_obj_ids,
    merge_and_push,
    push_dataset,
    update_dataset_card,
)
from utils.db_utils import (  # noqa: E402
    extract_quadrants_from_backgrounds,
    extract_quadrants_from_frames,
    extract_quadrants_from_psf,
    get_optimal_observation_ids,
    sync_background_frames,
    sync_calibrated_frames,
    sync_observation_catalogs,
    sync_psf_model,
)


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------
def resolve_obs_ids(args, verbose=True):
    """Observation IDs from --obs-ids / --obs-ids-file / --limit, else all optimal ones."""
    if args.obs_ids:
        ids = [str(x) for x in args.obs_ids]
    elif args.obs_ids_file:
        with open(args.obs_ids_file) as fh:
            ids = [line.strip() for line in fh if line.strip()]
    else:
        ids = get_optimal_observation_ids(verbose=verbose)
        if args.limit is not None:
            ids = ids[: args.limit]

    if not ids:
        sys.exit("No observation IDs to process.")
    if verbose:
        preview = ", ".join(map(str, ids[:10])) + (" ..." if len(ids) > 10 else "")
        print(f"[obs] {len(ids)} observation ID(s): {preview}")
    return ids


def acquire(obs_ids, verbose=True):
    """Download every raw product the build stage needs."""
    frame_files = sync_calibrated_frames(obs_ids, DATA_DIR, verbose=verbose)
    sync_background_frames(frame_files, obs_ids, DATA_DIR, verbose=verbose)
    sync_psf_model(DATA_DIR, verbose=verbose)
    sync_observation_catalogs(obs_ids, DATA_DIR, verbose=verbose)


def extract(verbose=True):
    """Slice the downloaded full-frame FITS into per-quadrant files."""
    psf_path = sync_psf_model(DATA_DIR, verbose=verbose)  # idempotent: locate on disk
    extract_quadrants_from_frames(DATA_DIR, QUADRANT_DIR, QUADRANTS, verbose=verbose)
    extract_quadrants_from_backgrounds(DATA_DIR, QUADRANT_DIR, QUADRANTS, verbose=verbose)
    extract_quadrants_from_psf(psf_path, QUADRANT_DIR, QUADRANTS, verbose=verbose)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(
        prog="src/main.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    sel = p.add_mutually_exclusive_group()
    sel.add_argument("--obs-ids", nargs="+", metavar="ID",
                     help="explicit observation IDs to process")
    sel.add_argument("--obs-ids-file", metavar="PATH",
                     help="file with one observation ID per line")
    p.add_argument("--limit", type=int, metavar="N",
                   help="keep only the first N optimal observations")

    p.add_argument("--skip-acquire", action="store_true",
                   help="frames/backgrounds/PSF/catalogues already downloaded")
    p.add_argument("--skip-extract", action="store_true",
                   help="per-quadrant FITS already sliced")
    p.add_argument("--skip-build", action="store_true",
                   help="stop after acquisition + extraction")

    p.add_argument("--processes", type=int, default=None, metavar="N",
                   help="build workers (default: all cores; 1 = sequential)")
    p.add_argument("--no-residual", action="store_true",
                   help="do not add the psf_residual column")
    p.add_argument("--drop-duplicates", action="store_true",
                   help="enforce at most one row per obj_id in the final dataset")
    p.add_argument("--reference-psf", metavar="PATH", default=None,
                   help="isotropic reference PSF FITS "
                        "(default: src/euclid_vis_isotropic_min_psf.fits)")

    out = p.add_mutually_exclusive_group()
    out.add_argument("--push", action="store_true",
                     help="push a fresh dataset to the Hub")
    out.add_argument("--merge", action="store_true",
                     help="concatenate with the existing Hub dataset, then push")
    p.add_argument("--save-to", metavar="DIR", default=None,
                   help="save_to_disk target (default when not pushing: <DATA_DIR>/dataset)")
    p.add_argument("--repo-id", default=HF_REPO_ID, help="Hub dataset repo id")
    p.add_argument("--public", action="store_true", help="push as a public dataset")
    p.add_argument("--hf-token", default=None, help="overrides the HF_TOKEN env var")

    p.add_argument("--quiet", action="store_true")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    verbose = not args.quiet

    obs_ids = resolve_obs_ids(args, verbose=verbose)

    if args.skip_acquire:
        if verbose:
            print("[acquire] skipped")
    else:
        acquire(obs_ids, verbose=verbose)

    if args.skip_extract:
        if verbose:
            print("[extract] skipped")
    else:
        extract(verbose=verbose)

    if args.skip_build:
        if verbose:
            print("[build] skipped (--skip-build); done.")
        return

    if verbose:
        print("[build] cutting stamps ...")
    dataset = build_dataset(obs_ids, processes=args.processes)
    if verbose:
        print(f"[build] {len(dataset)} stamp(s)")
    if len(dataset) == 0:
        sys.exit("[build] empty dataset - nothing to output.")

    if args.drop_duplicates:
        if verbose:
            print("[drop-duplicates] enforcing unique obj_id ...")
        dataset = drop_duplicate_obj_ids(dataset, verbose=verbose)

    if not args.no_residual:
        if verbose:
            print("[residual] adding psf_residual column ...")
        dataset = add_psf_residual(dataset, reference_psf_path=args.reference_psf)

    run_params = {
        "command": "python " + " ".join(sys.argv[1:] if argv is None else argv),
        "obs_selection": (
            "explicit --obs-ids" if args.obs_ids
            else f"--obs-ids-file {args.obs_ids_file}" if args.obs_ids_file
            else f"optimal set (--limit {args.limit})" if args.limit is not None
            else "all optimal observations"
        ),
        "processes": args.processes if args.processes is not None else "all cores",
        "psf_residual": not args.no_residual,
        "reference_psf": args.reference_psf or "default (src/euclid_vis_isotropic_min_psf.fits)",
        "drop_duplicates": args.drop_duplicates,
    }

    private = not args.public
    pushed = None
    if args.merge:
        if verbose:
            print(f"[output] merging into {args.repo_id} and pushing ...")
        pushed = merge_and_push(dataset, repo_id=args.repo_id, private=private,
                                token=args.hf_token,
                                drop_duplicates=args.drop_duplicates)
        run_params["output_mode"] = "merge (concatenated with existing Hub dataset)"
        if verbose:
            print(f"[output] pushed {len(pushed)} row(s) to {args.repo_id}")
    elif args.push:
        if verbose:
            print(f"[output] pushing to {args.repo_id} ...")
        push_dataset(dataset, repo_id=args.repo_id, private=private, token=args.hf_token)
        pushed = dataset
        run_params["output_mode"] = "fresh push"
        if verbose:
            print(f"[output] pushed {len(dataset)} row(s) to {args.repo_id}")

    if pushed is not None:
        run_params["visibility"] = "public" if args.public else "private"
        try:
            update_dataset_card(args.repo_id, pushed, run_params, token=args.hf_token)
            if verbose:
                print(f"[output] build-info section updated in {args.repo_id} README")
        except Exception as exc:  # noqa: BLE001 - card update must not fail the run
            print(f"[output] WARNING: could not update dataset card: {exc}")

    if args.save_to or not (args.push or args.merge):
        target = args.save_to or os.path.join(DATA_DIR, "dataset")
        dataset.save_to_disk(target)
        if verbose:
            print(f"[output] saved to {target}")


if __name__ == "__main__":
    main()
