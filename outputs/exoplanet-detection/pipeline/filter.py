"""
filter.py  -  Stage 2: QUALITY ASSESSMENT (the bouncer at the door)
====================================================================

WHAT THIS DOES (plain English)
------------------------------
Raw telescope data is messy. Before any science, we throw out garbage:

  1. NaN gaps        - moments where the telescope recorded nothing
  2. Cosmic ray hits - a particle smacks the camera -> one-frame UPWARD spike
  3. Junk cadences   - non-finite values, absurd readings

CRITICAL DOMAIN RULE: we only clip UPWARD spikes. A DOWNWARD dip might be
the planet we're hunting! Never sigma-clip below the baseline here.

We also produce a QC REPORT per star (kept %, gap count, noise level) and
REJECT stars whose data is too broken to trust (>40% missing, etc.).

USAGE
-----
    python filter.py                     # filter every .npy in data/raw/
    python filter.py --target KIC_11442793
Input : data/raw/<star>.npy       (from download.py)
Output: data/cleaned/<star>_filtered.npy  + printed QC report
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np

RAW_DIR   = Path(__file__).resolve().parent.parent / "data" / "raw"
CLEAN_DIR = Path(__file__).resolve().parent.parent / "data" / "cleaned"

# ---- tunable thresholds (sensible Kepler defaults) ----
MAX_MISSING_FRAC = 0.40   # reject star if >40% of cadences are unusable
COSMIC_SIGMA     = 5.0    # clip points > 5 sigma ABOVE local median
WINDOW           = 25     # cadences in the rolling window (~12h at 30-min cadence)


def rolling_median(x: np.ndarray, w: int) -> np.ndarray:
    """Median of a sliding window - our 'local normal brightness' estimate.
    Pure numpy (no scipy needed): pad edges, then take median over strides."""
    if w % 2 == 0: w += 1
    pad = w // 2
    xp = np.pad(x, pad, mode="edge")
    try:
        from numpy.lib.stride_tricks import sliding_window_view
        return np.median(sliding_window_view(xp, w), axis=1)
    except Exception:  # tiny fallback for old numpy
        return np.array([np.median(xp[i:i+w]) for i in range(len(x))])


def filter_curve(time: np.ndarray, flux: np.ndarray, flux_err: np.ndarray):
    """Run all quality steps. Returns (time, flux, flux_err, report) or None if rejected."""
    n0 = len(flux)

    # -- step 1: drop non-finite cadences (NaN flux, NaN time, infs) --
    ok = np.isfinite(time) & np.isfinite(flux)
    n_nan = int(n0 - ok.sum())
    time, flux, flux_err = time[ok], flux[ok], flux_err[ok]

    if n0 == 0 or (n_nan / max(n0, 1)) > MAX_MISSING_FRAC:
        return None  # too much missing data -> reject star

    # -- step 2: cosmic-ray removal (UPWARD spikes only!) --
    med = rolling_median(flux, WINDOW)
    resid = flux - med
    # MAD = robust noise estimate (stddev that ignores outliers)
    mad = np.median(np.abs(resid - np.median(resid))) * 1.4826
    mad = mad if mad > 0 else np.std(resid) + 1e-12
    cosmic = resid > COSMIC_SIGMA * mad          # only ABOVE the baseline
    n_cosmic = int(cosmic.sum())
    keep = ~cosmic
    time, flux, flux_err = time[keep], flux[keep], flux_err[keep]

    # -- step 3: gap census (for the report; big gaps matter to BLS later) --
    if len(time) > 1:
        dt = np.diff(time)
        cadence = np.median(dt)
        n_gaps = int((dt > 5 * cadence).sum())   # count gaps >5x normal spacing
    else:
        cadence, n_gaps = float("nan"), 0

    report = {
        "n_input": n0, "n_nan_dropped": n_nan, "n_cosmic_clipped": n_cosmic,
        "n_kept": len(flux), "kept_frac": round(len(flux) / n0, 4),
        "cadence_days": round(float(cadence), 5), "n_big_gaps": n_gaps,
        "noise_mad_ppm": round(float(mad / np.median(flux) * 1e6), 1),
    }
    return time, flux, flux_err, report


def process_file(npy_path: Path) -> bool:
    d = np.load(npy_path, allow_pickle=True).item()
    out = filter_curve(np.asarray(d["time"]), np.asarray(d["flux"]), np.asarray(d["flux_err"]))
    if out is None:
        print(f"[filter] REJECTED {npy_path.name}: too much missing/broken data")
        return False
    time, flux, flux_err, rep = out
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CLEAN_DIR / npy_path.name.replace(".npy", "_filtered.npy")
    np.save(out_path, {"target": d.get("target", npy_path.stem), "time": time,
                       "flux": flux, "flux_err": flux_err, "qc": rep}, allow_pickle=True)
    print(f"[filter] {npy_path.name}: kept {rep['n_kept']}/{rep['n_input']} "
          f"({rep['kept_frac']*100:.1f}%) | NaN {rep['n_nan_dropped']} | "
          f"cosmic {rep['n_cosmic_clipped']} | gaps {rep['n_big_gaps']} | "
          f"noise {rep['noise_mad_ppm']} ppm -> {out_path.name}")
    return True


def main(argv=None):
    ap = argparse.ArgumentParser(description="Stage 2: quality-filter raw light curves.")
    ap.add_argument("--target", help="filter one star (filename stem in data/raw)")
    a = ap.parse_args(argv)
    files = ([RAW_DIR / f"{a.target}.npy"] if a.target
             else sorted(RAW_DIR.glob("*.npy")))
    if not files:
        sys.exit(f"No .npy files found in {RAW_DIR} - run download.py first.")
    done = sum(process_file(f) for f in files if f.exists())
    print(f"[filter] finished: {done}/{len(files)} passed QC.")


if __name__ == "__main__":
    main()
