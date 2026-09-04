"""
get_labels.py  -  fetch the KOI catalog -> labels.csv (the training answers)
=============================================================================

WHAT THIS DOES (plain English)
------------------------------
Models learn from ANSWERS. NASA already spent a decade vetting Kepler signals
and published the verdicts in the KOI (Kepler Objects of Interest) table:

    CONFIRMED       -> a real planet (our positive class)
    FALSE POSITIVE  -> an impostor, usually an eclipsing binary (negative class)
    CANDIDATE       -> nobody knows yet (NEVER train on these - they're the
                       "unknowns" our demo makes predictions on)

This script downloads that table from the NASA Exoplanet Archive's public
TAP API (no account needed) and writes TWO files:

    pipeline/labels.csv    target,label,koi_fpflag_ss,...  -> for export.py --labels
    pipeline/targets_training.csv   training download list  -> for download.py --batch
                                    (separate file - your demo targets.csv stays untouched)

BONUS COLUMNS: the koi_fpflag_* flags say WHY something is a false positive.
koi_fpflag_ss = "stellar eclipse" = our eclipsing-binary label - exactly what
the FP-specialist model (Stage 8c) trains on. We grab them now so we never
have to re-download.

NEEDS INTERNET - run on your laptop/Colab, not in a sandbox:
    python pipeline/get_labels.py                  # balanced 400-star list
    python pipeline/get_labels.py --n-per-class 50 # smaller quick-start list
    python pipeline/get_labels.py --stratified     # SNR-stratified v2 list
                                                   # -> targets_training_v2.csv

WHY --stratified EXISTS (honesty note)
--------------------------------------
The v1 list took the TOP-SNR stars of each class. Great for a first proof,
but it made the benchmark too easy: with only loud, clean signals, depth
alone separates planets from binaries and the model scores a meaningless
1.000. --stratified instead samples evenly across the SNR distribution
(quiet signals included), so the metric reflects the hard cases the
eclipsing-binary physics features actually exist for.
"""
from __future__ import annotations
import argparse
import csv
import json
import random
import sys
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
TAP = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"

# One row per KOI: star id, disposition, and the "why is it an FP" flags
QUERY = """
SELECT kepid, kepoi_name, koi_disposition,
       koi_fpflag_nt, koi_fpflag_ss, koi_fpflag_co, koi_fpflag_ec,
       koi_period, koi_depth, koi_model_snr
FROM cumulative
""".strip()


def fetch_koi_table():
    url = TAP + "?" + urllib.parse.urlencode(
        {"query": QUERY, "format": "json"})
    print("[labels] downloading KOI cumulative table from NASA Exoplanet Archive...")
    with urllib.request.urlopen(url, timeout=120) as r:
        rows = json.load(r)
    print(f"[labels] received {len(rows)} KOI rows")
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description="Fetch KOI labels -> labels.csv + targets.csv")
    ap.add_argument("--n-per-class", type=int, default=200,
                    help="stars per class for the training download list")
    ap.add_argument("--stratified", action="store_true",
                    help="sample evenly across SNR quintiles per class "
                         "(seed 42) instead of top-SNR; writes the v2 list")
    ap.add_argument("--out-labels", default=str(HERE / "labels.csv"))
    ap.add_argument("--out-targets", default=None,
                    help="default: targets_training.csv, or "
                         "targets_training_v2.csv with --stratified "
                         "(separate files so the curated demo targets.csv is untouched)")
    a = ap.parse_args(argv)
    if a.out_targets is None:
        a.out_targets = str(HERE / ("targets_training_v2.csv" if a.stratified
                                    else "targets_training.csv"))

    rows = fetch_koi_table()

    # one label per STAR (kepid): a star with ANY confirmed planet counts as
    # CONFIRMED; else FALSE POSITIVE if any FP KOI; else CANDIDATE. Matching
    # our star-level split - labels must live at the same level as the split.
    rank = {"CONFIRMED": 2, "FALSE POSITIVE": 1, "CANDIDATE": 0}
    per_star = {}
    for r in rows:
        kepid = r["kepid"]
        disp = (r["koi_disposition"] or "CANDIDATE").upper()
        best = per_star.get(kepid)
        if best is None or rank.get(disp, 0) > rank.get(best["label"], 0):
            per_star[kepid] = {
                "target": f"KIC {kepid}", "label": disp,
                "koi_name": r.get("kepoi_name") or "",
                "fpflag_nt": r.get("koi_fpflag_nt") or 0,
                "fpflag_ss": r.get("koi_fpflag_ss") or 0,   # eclipsing binary!
                "fpflag_co": r.get("koi_fpflag_co") or 0,
                "fpflag_ec": r.get("koi_fpflag_ec") or 0,
                "koi_period": r.get("koi_period") or "",
                "koi_depth": r.get("koi_depth") or "",
                "koi_snr": r.get("koi_model_snr") or "",
            }
    stars = list(per_star.values())
    counts = {}
    for s in stars:
        counts[s["label"]] = counts.get(s["label"], 0) + 1
    print(f"[labels] {len(stars)} unique stars: {counts}")

    with open(a.out_labels, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(stars[0].keys()))
        w.writeheader(); w.writerows(stars)
    print(f"[labels] wrote {a.out_labels}")

    # training download list, two flavors:
    #   default:      highest-SNR stars first (clean signals, easy benchmark)
    #   --stratified: even coverage of the SNR distribution (quintiles), so
    #                 quiet/hard signals are represented and the metric is
    #                 honest. Seeded shuffle -> reproducible.
    def snr(s):
        try:
            return float(s["koi_snr"])
        except (TypeError, ValueError):
            return 0.0
    picks = []
    if a.stratified:
        rng = random.Random(42)
        per_quintile = max(1, a.n_per_class // 5)
        for want in ("CONFIRMED", "FALSE POSITIVE"):
            pool = sorted([s for s in stars if s["label"] == want], key=snr)
            k, n = len(pool) // 5, len(pool)
            got = 0
            for q in range(5):
                seg = pool[q * k: (q + 1) * k if q < 4 else n]
                rng.shuffle(seg)
                take = seg[:per_quintile]
                picks += take
                got += len(take)
            print(f"[labels] {want}: {got} stars, ~{per_quintile}/SNR-quintile "
                  f"(pool {n}, SNR {snr(pool[0]):.1f}-{snr(pool[-1]):.1f})")
    else:
        for want in ("CONFIRMED", "FALSE POSITIVE"):
            pool = sorted([s for s in stars if s["label"] == want], key=snr, reverse=True)
            picks += pool[: a.n_per_class]
    with open(a.out_targets, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["target", "mission"])
        for s in picks:
            w.writerow([s["target"], "Kepler"])
    mode = "SNR-stratified quintiles, seed 42" if a.stratified else "by SNR"
    print(f"[labels] wrote {a.out_targets} ({len(picks)} stars, "
          f"{a.n_per_class}/class {mode})")
    print("[labels] NEXT: python pipeline/download.py --batch pipeline/targets_training.csv")


if __name__ == "__main__":
    try:
        main()
    except urllib.error.URLError as e:
        sys.exit(f"[labels] network error: {e}\n"
                 "        This script needs internet - run it on your laptop/Colab.")
