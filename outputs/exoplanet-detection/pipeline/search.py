"""
search.py  -  Stage 4: BLS PERIOD SEARCH (find the repeating dip)
==================================================================

WHAT THIS DOES (plain English)
------------------------------
After detrending, a transit is a tiny dip that repeats every orbit.
BOX LEAST SQUARES (BLS) finds it by brute force:

  for every trial PERIOD:
      fold the curve at that period (stack all orbits on top of each other)
      slide a box-shaped "dip" template across the folded curve
      how well does a box of this width/position fit? -> score
  the period whose best box scores highest = our candidate.

Outputs per star: period (days), t0 (a transit center time), duration (hours),
depth (ppm), SNR, and SDE (how much the best period stands out vs all others).

SPEED (this is the important part for real Kepler data)
-------------------------------------------------------
A Kepler SHORT-CADENCE star has ~1.5 MILLION points. Running BLS on every raw
point is pointless (transits last hours; 1-minute sampling is overkill) AND
astropy's `autopower` picks a runaway period grid -> 30+ minutes.

So we do what every real pipeline does:
  1. BIN the light curve to a coarse cadence (~10 min) -> ~100k points, no loss
     of transit information (a transit spans many bins).
  2. Use a BOUNDED, frequency-uniform period grid we control (not autopower).
This turns a 30-minute search into a few seconds.

FAST PATH: astropy's compiled BLS on the binned curve + our fixed grid.
FALLBACK : pure-numpy implementation (what selftest.py exercises offline).

USAGE
-----
    python search.py                          # all *_detrended.npy
    python search.py --target Kepler_10
    python search.py --pmin 0.5 --pmax 40
    python search.py --bin-minutes 10         # tune the binning cadence
Input : data/cleaned/<star>_detrended.npy   (from preprocess.py)
Output: data/cleaned/<star>_bls.npy
"""
from __future__ import annotations
import argparse, sys, time as _time
from pathlib import Path
import numpy as np

CLEAN_DIR = Path(__file__).resolve().parent.parent / "data" / "cleaned"

# box widths to try, as FRACTION of the period (transits are 0.5%..5% of an orbit)
Q_VALUES = (0.005, 0.01, 0.02, 0.03, 0.05)
N_BINS   = 256      # phase bins when folding (numpy path)
OVERSAMPLE = 3      # period-grid density multiplier
MAX_POINTS_ASTROPY = 200_000   # bin down to at most this many points before BLS


# ----------------------------------------------------------------------
# Binning: collapse many raw points into coarse time bins (the speed key)
# ----------------------------------------------------------------------
def bin_curve(time, flux, bin_minutes):
    """Average flux into fixed-width time bins. ~1.5M pts -> ~100k, keeping
    the transit shape intact (a transit covers many bins)."""
    bin_days = bin_minutes / (60.0 * 24.0)
    t0 = time.min()
    idx = np.floor((time - t0) / bin_days).astype(np.int64)
    # sums per bin -> means per bin
    order = np.argsort(idx, kind="mergesort")
    idx_s, flux_s, time_s = idx[order], flux[order], time[order]
    uniq, start = np.unique(idx_s, return_index=True)
    sums_f = np.add.reduceat(flux_s, start)
    sums_t = np.add.reduceat(time_s, start)
    counts = np.diff(np.append(start, len(idx_s)))
    return sums_t / counts, sums_f / counts


MAX_GRID = 250_000   # hard cap on trial periods (protects against runaway grids)

def period_grid(T, p_min, p_max):
    """Frequency-uniform trial periods. Spacing chosen so a transit never
    drifts more than a fraction of its width across the whole baseline.
    Capped at MAX_GRID so a multi-year baseline can't create millions of
    trials (that was the 30-minute hang)."""
    df = min(Q_VALUES) / (T * OVERSAMPLE)
    f_lo, f_hi = 1.0 / p_max, 1.0 / p_min
    n = int(np.ceil((f_hi - f_lo) / df))
    if n > MAX_GRID:
        df = (f_hi - f_lo) / MAX_GRID          # coarsen uniformly to fit the cap
        n = MAX_GRID
    freqs = f_lo + df * np.arange(n)
    return 1.0 / freqs


# ----------------------------------------------------------------------
# Fast path: astropy compiled BLS on a BOUNDED grid (no autopower)
# ----------------------------------------------------------------------
def bls_astropy(time, flux, p_min, p_max):
    from astropy.timeseries import BoxLeastSquares
    T = time.max() - time.min()
    periods = period_grid(T, p_min, p_max)
    # durations to try, in days (short transits -> a few hours)
    durations = np.unique(np.clip(
        np.array([0.02, 0.05, 0.1, 0.2, 0.4]), 0, 0.9 * p_min))
    durations = durations[durations > 0]
    model = BoxLeastSquares(time, flux)
    res = model.power(periods, durations, objective="snr")
    i = int(np.argmax(res.power))
    P, t0, dur = float(res.period[i]), float(res.transit_time[i]), float(res.duration[i])
    depth = float(res.depth[i])
    pw = np.asarray(res.power, float)
    sde = float((pw.max() - pw.mean()) / (pw.std() + 1e-12))
    # PROPER transit SNR: depth vs out-of-transit scatter, boosted by how many
    # points sit inside transits. (astropy's power value is a different unit -
    # that's what printed the misleading "SNR=0.0" on real data.)
    phase = ((time - t0) / P + 0.5) % 1.0 - 0.5
    in_tr = np.abs(phase) < 0.5 * dur / P
    out_f = flux[~in_tr]
    sigma = 1.4826 * np.median(np.abs(out_f - np.median(out_f))) + 1e-12
    snr = depth / sigma * np.sqrt(max(int(in_tr.sum()), 1))
    return {"period": P, "t0": t0, "duration_hr": dur * 24, "depth_ppm": depth * 1e6,
            "snr": float(snr), "sde": sde,
            "periods": np.asarray(res.period, float), "power": pw,
            "n_grid": len(periods)}


# ----------------------------------------------------------------------
# Fallback: pure-numpy BLS (offline; used by selftest.py)
# ----------------------------------------------------------------------
def bls_numpy(time, flux, p_min, p_max):
    t = time - time.min()
    T = t.max()
    dm = flux - np.median(flux)
    sigma = 1.4826 * np.median(np.abs(dm - np.median(dm))) + 1e-12
    periods = period_grid(T, p_min, p_max)
    power = np.zeros(len(periods))
    best = {"stat": -1.0}
    for k, P in enumerate(periods):
        phase = (t / P) % 1.0
        b = np.minimum((phase * N_BINS).astype(np.int64), N_BINS - 1)
        s_b = np.bincount(b, weights=dm, minlength=N_BINS)
        n_b = np.bincount(b, minlength=N_BINS)
        Ntot, Stot = n_b.sum(), s_b.sum()
        cs = np.concatenate(([0.0], np.cumsum(np.concatenate((s_b, s_b)))))
        cn = np.concatenate(([0],   np.cumsum(np.concatenate((n_b, n_b)))))
        stat_best = 0.0
        for q in Q_VALUES:
            w = max(1, int(round(q * N_BINS)))
            s_in = cs[w:w + N_BINS] - cs[:N_BINS]
            n_in = cn[w:w + N_BINS] - cn[:N_BINS]
            ok = (n_in > 0) & (n_in < Ntot)
            if not ok.any():
                continue
            mean_in  = np.where(ok, s_in / np.maximum(n_in, 1), 0.0)
            mean_out = np.where(ok, (Stot - s_in) / np.maximum(Ntot - n_in, 1), 0.0)
            depth = mean_out - mean_in
            stat = np.where(depth > 0, depth * np.sqrt(n_in * (Ntot - n_in) / Ntot), 0.0)
            i = int(np.argmax(stat))
            if stat[i] > stat_best:
                stat_best = float(stat[i])
                if stat_best > best["stat"]:
                    best = {"stat": stat_best, "period": float(P), "bin": i, "w": w,
                            "depth": float(depth[i]), "n_in": int(n_in[i])}
        power[k] = stat_best
    P, i, w = best["period"], best["bin"], best["w"]
    duration_days = w / N_BINS * P
    phase_center  = (i + w / 2) / N_BINS
    t0 = time.min() + phase_center * P
    snr = best["stat"] / sigma
    sde = (power.max() - power.mean()) / (power.std() + 1e-12)
    return {"period": P, "t0": float(t0), "duration_hr": duration_days * 24,
            "depth_ppm": best["depth"] * 1e6, "snr": float(snr), "sde": float(sde),
            "periods": periods, "power": power, "n_grid": len(periods)}


def run_search(time, flux, p_min=0.5, p_max=None, bin_minutes=10.0, verbose=True):
    time = np.asarray(time, float)
    flux = np.asarray(flux, float)
    good = np.isfinite(time) & np.isfinite(flux)
    time, flux = time[good], flux[good]

    T = time.max() - time.min()
    if p_max is None:
        p_max = max(1.0, min(T / 2.2, 50.0))   # need >=2 transits in the baseline

    n_raw = len(time)
    # BIN only if we have many points and binning still leaves plenty
    if n_raw > MAX_POINTS_ASTROPY and bin_minutes > 0:
        tb, fb = bin_curve(time, flux, bin_minutes)
        if verbose:
            print(f"[search]   binned {n_raw:,} -> {len(tb):,} points "
                  f"({bin_minutes:g}-min cadence)")
        time, flux = tb, fb

    t_start = _time.perf_counter()
    try:
        res = bls_astropy(time, flux, p_min, p_max)
        engine = "astropy"
    except ImportError:
        res = bls_numpy(time, flux, p_min, p_max)
        engine = "numpy"
    res["engine"] = engine
    res["search_seconds"] = round(_time.perf_counter() - t_start, 2)
    if verbose:
        print(f"[search]   engine={engine}  grid={res['n_grid']:,} periods  "
              f"in {res['search_seconds']}s")
    return res


def process_file(path: Path, p_min: float, p_max, bin_minutes: float) -> bool:
    d = np.load(path, allow_pickle=True).item()
    time, flux = np.asarray(d["time"]), np.asarray(d["flux"])
    res = run_search(time, flux, p_min, p_max, bin_minutes)
    out = CLEAN_DIR / path.name.replace("_detrended.npy", "_bls.npy")
    res["target"] = d.get("target", path.stem)
    np.save(out, res, allow_pickle=True)
    print(f"[search] {res['target']}: P={res['period']:.4f} d | depth={res['depth_ppm']:.0f} ppm "
          f"| dur={res['duration_hr']:.2f} h | SNR={res['snr']:.1f} | SDE={res['sde']:.1f} -> {out.name}")
    return True


def main(argv=None):
    ap = argparse.ArgumentParser(description="Stage 4: BLS period search.")
    ap.add_argument("--target", help="one star (stem before _detrended.npy)")
    ap.add_argument("--pmin", type=float, default=0.5, help="shortest trial period, days")
    ap.add_argument("--pmax", type=float, default=None, help="longest trial period, days")
    ap.add_argument("--bin-minutes", type=float, default=10.0,
                    help="bin cadence before search (0 = no binning)")
    a = ap.parse_args(argv)
    files = ([CLEAN_DIR / f"{a.target}_detrended.npy"] if a.target
             else sorted(CLEAN_DIR.glob("*_detrended.npy")))
    if not files:
        sys.exit(f"No *_detrended.npy in {CLEAN_DIR} - run preprocess.py first.")
    done = sum(process_file(f, a.pmin, a.pmax, a.bin_minutes) for f in files if f.exists())
    print(f"[search] finished: {done}/{len(files)} searched.")


if __name__ == "__main__":
    main()
