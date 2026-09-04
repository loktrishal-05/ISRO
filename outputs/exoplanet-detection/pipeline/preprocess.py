"""
preprocess.py  -  Stage 3: DETREND + NORMALIZE (flatten the star, keep the dip)
================================================================================

WHAT THIS DOES (plain English)
------------------------------
Stars aren't steady lamps - they slowly brighten/dim (starspots, rotation),
and instruments drift. That slow "wander" hides our tiny transit dips.

DETRENDING = estimate the slow trend, then divide it out:
      cleaned_flux = flux / trend        -> flat line at 1.0, dips survive

We use a SAVITZKY-GOLAY filter: slides a window along the curve and fits a
small polynomial in each window = a smart moving average that follows slow
curves but ignores fast events.

*** THE GOLDEN RULE (Key Decision #2) ***
The window MUST be LONGER than the transit duration. A transit lasts hours;
we default the window to 2 DAYS. If the window were shorter, the filter would
"learn" the dip itself and erase the very signal we hunt. That's why
--window-hours exists and is loud about it.

Uses scipy if installed; otherwise falls back to a pure-numpy implementation
(same math: least-squares polynomial convolution).

USAGE
-----
    python preprocess.py                          # all files in data/cleaned/
    python preprocess.py --window-hours 48        # default 48h = 2 days
Input : data/cleaned/<star>_filtered.npy   (from filter.py)
Output: data/cleaned/<star>_detrended.npy  {time, flux(normalized), flux_err, trend}
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np

CLEAN_DIR = Path(__file__).resolve().parent.parent / "data" / "cleaned"


# ---------- Savitzky-Golay (scipy if present, numpy fallback) ----------
def savgol(y: np.ndarray, window: int, poly: int = 2) -> np.ndarray:
    if window % 2 == 0: window += 1                      # must be odd
    window = min(window, len(y) - (1 - len(y) % 2))      # can't exceed data
    if window < poly + 2: return np.full_like(y, np.median(y))
    try:
        from scipy.signal import savgol_filter
        return savgol_filter(y, window, poly)
    except ImportError:
        # numpy fallback: build the least-squares smoothing kernel once.
        # Fitting poly p over window w == convolving with pinv row 0.
        half = window // 2
        x = np.arange(-half, half + 1)
        A = np.vander(x, poly + 1, increasing=True)      # design matrix
        kernel = np.linalg.pinv(A)[0]                    # row 0 = smoothed value at center
        ypad = np.pad(y, half, mode="edge")
        return np.convolve(ypad, kernel[::-1], mode="valid")


def detrend(time: np.ndarray, flux: np.ndarray, window_hours: float = 48.0):
    """Divide out the slow trend. Returns (normalized_flux, trend).
    Window is converted hours -> cadences using the actual observed spacing."""
    cadence_days = float(np.median(np.diff(time))) if len(time) > 1 else 0.0204
    window = int(round((window_hours / 24.0) / max(cadence_days, 1e-9)))
    window = max(window, 11)                             # never absurdly small
    trend = savgol(flux, window)
    trend = np.where(np.abs(trend) < 1e-12, 1e-12, trend)  # no divide-by-zero
    return flux / trend, trend


def phase_fold(time: np.ndarray, period: float, t0: float = 0.0) -> np.ndarray:
    """Wrap time onto orbital phase [0,1). Used later by search.py + the CNN.
    (Reminder: the LSTM uses UNFOLDED series - folding erases timing variations.)"""
    return ((time - t0) / period) % 1.0


def process_file(path: Path, window_hours: float) -> bool:
    d = np.load(path, allow_pickle=True).item()
    time, flux = np.asarray(d["time"]), np.asarray(d["flux"])
    flux_err   = np.asarray(d.get("flux_err", np.full_like(flux, np.nan)))

    norm, trend = detrend(time, flux, window_hours)
    # normalize errors by the same trend so units stay consistent
    err_norm = flux_err / np.where(np.abs(trend) < 1e-12, 1e-12, trend)

    out = CLEAN_DIR / path.name.replace("_filtered.npy", "_detrended.npy")
    np.save(out, {"target": d.get("target", path.stem), "time": time,
                  "flux": norm, "flux_err": err_norm, "trend": trend,
                  "window_hours": window_hours, "qc": d.get("qc", {})},
            allow_pickle=True)
    scatter_ppm = np.std(norm) * 1e6
    print(f"[preprocess] {path.name}: window={window_hours}h "
          f"-> flat around {np.median(norm):.6f}, scatter {scatter_ppm:.0f} ppm "
          f"-> {out.name}")
    return True


def main(argv=None):
    ap = argparse.ArgumentParser(description="Stage 3: detrend + normalize filtered curves.")
    ap.add_argument("--target", help="one star (stem before _filtered.npy)")
    ap.add_argument("--window-hours", type=float, default=48.0,
                    help="SavGol window in HOURS. MUST exceed the longest transit "
                         "duration you care about (default 48h; Kepler transits are 1-13h).")
    a = ap.parse_args(argv)
    if a.window_hours < 24:
        print(f"[preprocess] WARNING: {a.window_hours}h window is risky - long transits "
              f"(>{a.window_hours/3:.0f}h) may get partially erased. 48h is the safe default.")
    files = ([CLEAN_DIR / f"{a.target}_filtered.npy"] if a.target
             else sorted(CLEAN_DIR.glob("*_filtered.npy")))
    if not files:
        sys.exit(f"No *_filtered.npy in {CLEAN_DIR} - run filter.py first.")
    done = sum(process_file(f, a.window_hours) for f in files if f.exists())
    print(f"[preprocess] finished: {done}/{len(files)} detrended.")


if __name__ == "__main__":
    main()
