"""
download.py  -  Stage 1 of the pipeline: DATA ACQUISITION
============================================================

WHAT THIS FILE DOES (in plain English)
--------------------------------------
A telescope called Kepler stared at ~150,000 stars and measured how bright each
one was, over and over, for years. That table of (time, brightness) numbers for
one star is called a LIGHT CURVE.

NASA stores each light curve as a ".fits" file (an astronomy file format) on a
public server called MAST. The Python library `lightkurve` knows how to:
    1. search MAST for a given star,
    2. download its FITS files,
    3. hand us the raw numbers.

This script's ONE job: given a star, fetch its Kepler observations, pull out the
brightness measurements, stitch them into a single continuous array, and SAVE it
to disk so the next stages (filter -> preprocess -> search -> features) have
something to work with.

WHY PDCSAP_FLUX (not SAP_FLUX)
------------------------------
Each FITS file gives us TWO brightness columns:
    - SAP_FLUX     = raw brightness off the detector (has instrument drift)
    - PDCSAP_FLUX  = NASA already removed common instrument systematics for us
We use PDCSAP_FLUX as the primary signal. (Key Technical Decision #1.)

HOW TO RUN IT
-------------
    # one star (KIC 11442793 is the famous Kepler-90 8-planet system)
    python download.py --target "KIC 11442793"

    # a whole BATCH listed in a CSV that has a 'target' column
    python download.py --batch targets.csv

Outputs land in  ../data/raw/  as:
    <target>.npy   -> dict with time, flux, flux_err arrays  (fast reload)
    <target>.csv   -> same data as a spreadsheet (easy to eyeball)

NOTE: needs INTERNET (talks to NASA's MAST). Run on your laptop or Google Colab.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import lightkurve as lk
except ImportError:
    sys.exit(
        "lightkurve is not installed.\n"
        "Install the pipeline deps first:\n"
        "    pip install -r pipeline/requirements.txt\n"
        "or at minimum:  pip install lightkurve"
    )


# Resolves to  <repo>/data/raw  no matter what folder you run from.
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def _safe_filename(target: str) -> str:
    """'KIC 11442793' -> 'KIC_11442793' so filenames stay filesystem-safe."""
    return "".join(c if c.isalnum() else "_" for c in target).strip("_")


def download_light_curve(
    target: str,
    mission: str = "Kepler",
    flux_column: str = "pdcsap_flux",
    author: str = "Kepler",
):
    """Download and stitch together every light curve for one star.

    WHY "STITCH"? Kepler observed in ~3-month chunks called "quarters". One star
    can have 17+ separate files. `.stitch()` glues them into one continuous curve
    so later stages see the star's whole history, not 17 fragments.

    Returns a stitched lightkurve.LightCurve, or None if nothing was found.
    """
    print(f"[download] searching MAST for: {target!r} (mission={mission})")

    # Ask MAST what light-curve files exist. Returns a table, not the data yet.
    search_result = lk.search_lightcurve(target, mission=mission, author=author)

    if len(search_result) == 0:
        print(f"[download] WARNING: no results found for {target!r}. Skipping.")
        return None

    print(f"[download] found {len(search_result)} observation(s). Downloading...")

    # Actually pull every FITS file -> a LightCurveCollection (list of quarters).
    collection = search_result.download_all(flux_column=flux_column)

    if collection is None or len(collection) == 0:
        print(f"[download] WARNING: download returned nothing for {target!r}.")
        return None

    stitched = collection.stitch()  # glue all quarters into one curve
    print(f"[download] stitched {len(collection)} quarters -> {len(stitched.flux)} points")
    return stitched


def save_light_curve(lc, target: str) -> None:
    """Save to ../data/raw as both .npy (fast) and .csv (readable).

    Three parallel arrays:
        time      -> when each measurement was taken (days, "BKJD")
        flux      -> the star's brightness at that time (PDCSAP_FLUX)
        flux_err  -> the telescope's uncertainty on that brightness
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    stem = _safe_filename(target)

    # .value strips the astronomy units wrapper -> plain numpy arrays.
    time = np.asarray(lc.time.value, dtype=float)
    flux = np.asarray(lc.flux.value, dtype=float)
    try:
        flux_err = np.asarray(lc.flux_err.value, dtype=float)
    except (AttributeError, TypeError):
        flux_err = np.full_like(flux, np.nan)

    npy_path = RAW_DIR / f"{stem}.npy"
    np.save(
        npy_path,
        {"target": target, "time": time, "flux": flux, "flux_err": flux_err},
        allow_pickle=True,
    )

    csv_path = RAW_DIR / f"{stem}.csv"
    pd.DataFrame({"time": time, "flux": flux, "flux_err": flux_err}).to_csv(
        csv_path, index=False
    )
    print(f"[download] saved -> {npy_path.name} and {csv_path.name} ({len(flux)} rows)")


def process_target(target: str, mission: str = "Kepler") -> bool:
    """Download + save one target. True on success, False if skipped."""
    lc = download_light_curve(target, mission=mission)
    if lc is None:
        return False
    save_light_curve(lc, target)
    return True


def process_batch(csv_path: str, mission: str = "Kepler") -> None:
    """Download many stars from a CSV that has a 'target' column."""
    df = pd.read_csv(csv_path)
    if "target" not in df.columns:
        sys.exit(f"Batch CSV {csv_path!r} must have a 'target' column.")

    targets = df["target"].dropna().astype(str).tolist()
    print(f"[download] batch mode: {len(targets)} targets from {csv_path}")

    ok, skipped = 0, 0
    for i, target in enumerate(targets, start=1):
        print(f"\n--- [{i}/{len(targets)}] {target} ---")
        if process_target(target, mission=mission):
            ok += 1
        else:
            skipped += 1
    print(f"\n[download] batch done: {ok} downloaded, {skipped} skipped.")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stage 1: download Kepler light curves from NASA's MAST archive."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--target", help='One star, e.g. "KIC 11442793" or "Kepler-10".')
    group.add_argument("--batch", help="CSV with a 'target' column for many stars.")
    parser.add_argument(
        "--mission", default="Kepler", choices=["Kepler", "K2", "TESS"],
        help="Which mission's data to fetch (default: Kepler).",
    )
    return parser


def main(argv=None) -> None:
    args = build_arg_parser().parse_args(argv)
    if args.target:
        if not process_target(args.target, mission=args.mission):
            sys.exit(1)
    else:
        process_batch(args.batch, mission=args.mission)


if __name__ == "__main__":
    main()
