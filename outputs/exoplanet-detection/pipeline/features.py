"""
features.py  -  Stage 5: FEATURE ENGINEERING (the false-positive catchers)
===========================================================================

WHAT THIS DOES (plain English)
------------------------------
The BLS gave us WHERE the dip is. Now we measure EVERYTHING about it that
helps separate real planets from impostors. The biggest impostor is the
ECLIPSING BINARY (two stars orbiting each other) - it also makes dips!
But it leaves fingerprints:

  * odd_even_diff : binaries often alternate deep/shallow dips (two different
                    stars blocking each other). Planets: every dip identical.
  * secondary_ppm : a binary's companion GLOWS, so half an orbit later there's
                    a second small dip. Planets (dark) leave almost none.
  * v_metric      : star-star eclipses are V-shaped; planet transits have a
                    flat bottom (U-shaped). We measure core depth vs full depth.
  * symmetry      : real transits are symmetric in time; junk often isn't.

Also produces the CNN's two inputs (the AstroNet trick):
  * GLOBAL view : whole folded orbit in 512 bins  ("context")
  * LOCAL view  : zoom on the transit, +/-2.5 durations, 128 bins ("close-up")

USAGE
-----
    python features.py                      # all stars with _bls.npy
    python features.py --target KIC_11442793
Input : data/cleaned/<star>_detrended.npy + <star>_bls.npy
Output: data/features/<star>_features.json  (tabular features)
        data/features/<star>_views.npy      {global, local}  (CNN inputs)
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

CLEAN_DIR = Path(__file__).resolve().parent.parent / "data" / "cleaned"
FEAT_DIR  = Path(__file__).resolve().parent.parent / "data" / "features"
GLOBAL_BINS, LOCAL_BINS, LOCAL_WIDTH = 512, 128, 2.5   # LOCAL_WIDTH in transit-durations


def fold(time, period, t0):
    """Phase in [-0.5, 0.5) with the transit centered at phase 0."""
    return ((time - t0) / period + 0.5) % 1.0 - 0.5


def binned_median(x, y, bins, lo, hi):
    """Median of y in equal x-bins (NaN where a bin is empty, then filled)."""
    idx = np.clip(((x - lo) / (hi - lo) * bins).astype(int), 0, bins - 1)
    out = np.full(bins, np.nan)
    for b in np.unique(idx):
        out[b] = np.median(y[idx == b])
    # fill empty bins with overall median so the CNN never sees NaN
    out[np.isnan(out)] = np.nanmedian(out)
    return out


def depth_in_window(phase, flux, center, half_width, baseline):
    """How deep is the curve inside [center +/- half_width]? (ppm, >0 = dip)"""
    m = np.abs(phase - center) < half_width
    if m.sum() < 3:
        return 0.0, int(m.sum())
    return float((baseline - np.median(flux[m])) * 1e6), int(m.sum())


def extract(time, flux, bls):
    P, t0 = bls["period"], bls["t0"]
    dur_days = bls["duration_hr"] / 24.0
    half_q = (dur_days / P) / 2.0                     # half transit width, in phase units
    ph = fold(time, P, t0)
    out_of_transit = np.abs(ph) > 2.5 * half_q
    baseline = np.median(flux[out_of_transit])
    noise_ppm = 1.4826 * np.median(np.abs(flux[out_of_transit] - baseline)) * 1e6 + 1e-9

    # ---- primary transit depth (re-measured cleanly) ----
    depth_ppm, n_in = depth_in_window(ph, flux, 0.0, half_q, baseline)

    # ---- odd vs even transits (EB fingerprint #1) ----
    ph2 = fold(time, 2 * P, t0)                       # fold at DOUBLE period:
    d_odd,  _ = depth_in_window(ph2, flux, 0.0,  half_q / 2, baseline)   # odd transits at 0
    d_even, _ = depth_in_window(ph2, flux, -0.5, half_q / 2, baseline)   # even at -0.5 (wraps)
    d_even2, _ = depth_in_window(ph2, flux, 0.5 - 1.0, half_q / 2, baseline)
    d_even = max(d_even, d_even2)
    odd_even_diff = abs(d_odd - d_even) / max(depth_ppm, 1.0)

    # ---- secondary eclipse at phase 0.5 (EB fingerprint #2) ----
    sec_ppm, _ = depth_in_window(np.abs(ph), flux, 0.5, max(half_q, 0.01), baseline)
    secondary_ratio = max(sec_ppm, 0.0) / max(depth_ppm, 1.0)

    # ---- V-shape metric (EB fingerprint #3): core depth vs full depth ----
    d_core, _ = depth_in_window(ph, flux, 0.0, half_q * 0.25, baseline)
    v_metric = d_core / max(depth_ppm, 1.0)           # U (planet) ~1.0-1.3, V (EB) higher

    # ---- ingress/egress symmetry ----
    left  = (ph > -half_q) & (ph < 0)
    right = (ph > 0) & (ph < half_q)
    if left.sum() > 2 and right.sum() > 2:
        symmetry = abs(np.median(flux[left]) - np.median(flux[right])) * 1e6 / max(depth_ppm, 1.0)
    else:
        symmetry = 0.0

    # ---- how many separate transits did we actually see? ----
    epochs = np.round((time - t0) / P)
    n_transits = int(len(np.unique(epochs[np.abs(ph) < half_q])))

    feats = {
        "period_d": round(P, 6), "t0": round(t0, 5),
        "duration_hr": round(bls["duration_hr"], 3),
        "depth_ppm": round(depth_ppm, 1), "snr_bls": round(bls["snr"], 2),
        "sde": round(bls["sde"], 2), "noise_ppm": round(noise_ppm, 1),
        "odd_even_diff": round(odd_even_diff, 4),
        "secondary_ratio": round(secondary_ratio, 4),
        "v_metric": round(v_metric, 4), "symmetry": round(symmetry, 4),
        "duty_cycle": round(dur_days / P, 5), "n_transits": n_transits,
        "n_points_in_transit": n_in,
    }

    # ---- CNN views (AstroNet-style) ----
    gview = binned_median(ph, flux, GLOBAL_BINS, -0.5, 0.5)
    span = LOCAL_WIDTH * 2 * half_q
    lview = binned_median(ph, flux, LOCAL_BINS, -span, span)
    # normalize: baseline -> 0, transit bottom -> -1  (standard AstroNet scaling)
    def norm(v):
        v = v - np.median(v)
        lo = v.min()
        return v / abs(lo) if lo < 0 else v
    return feats, norm(gview), norm(lview)


def process_target(stem: str) -> bool:
    det, bls_f = CLEAN_DIR / f"{stem}_detrended.npy", CLEAN_DIR / f"{stem}_bls.npy"
    if not det.exists() or not bls_f.exists():
        print(f"[features] SKIP {stem}: missing detrended/bls file"); return False
    d   = np.load(det,  allow_pickle=True).item()
    bls = np.load(bls_f, allow_pickle=True).item()
    feats, gview, lview = extract(np.asarray(d["time"]), np.asarray(d["flux"]), bls)
    feats["target"] = d.get("target", stem)
    FEAT_DIR.mkdir(parents=True, exist_ok=True)
    (FEAT_DIR / f"{stem}_features.json").write_text(json.dumps(feats, indent=2))
    np.save(FEAT_DIR / f"{stem}_views.npy", {"global": gview, "local": lview}, allow_pickle=True)
    print(f"[features] {stem}: depth={feats['depth_ppm']:.0f}ppm oddeven={feats['odd_even_diff']:.3f} "
          f"sec={feats['secondary_ratio']:.3f} v={feats['v_metric']:.2f} sym={feats['symmetry']:.3f} "
          f"n_tr={feats['n_transits']}")
    return True


def main(argv=None):
    ap = argparse.ArgumentParser(description="Stage 5: engineered features + CNN views.")
    ap.add_argument("--target", help="one star stem, e.g. KIC_11442793")
    a = ap.parse_args(argv)
    stems = ([a.target] if a.target else
             sorted(p.name.replace("_bls.npy", "") for p in CLEAN_DIR.glob("*_bls.npy")))
    if not stems:
        sys.exit(f"No *_bls.npy in {CLEAN_DIR} - run search.py first.")
    done = sum(process_target(s) for s in stems)
    print(f"[features] finished: {done}/{len(stems)} stars featurized.")


if __name__ == "__main__":
    main()
