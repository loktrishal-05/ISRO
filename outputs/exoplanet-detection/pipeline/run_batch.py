"""
run_batch.py  -  ONE COMMAND to build the real training set
=============================================================

WHAT THIS DOES (plain English)
------------------------------
Training the model needs hundreds of stars, each pushed through the whole
pipeline. Doing that by hand (download -> filter -> preprocess -> search ->
features, per star) would be misery. This script does it all, robustly:

  for each star in pipeline/targets_training.csv:
      1. download   (LONG-cadence only + first N quarters - keeps each star
                     ~30k points instead of 1.5 million; plenty for training)
      2. filter -> preprocess -> search -> features
      3. one summary line

  - RESUME-SAFE: a star whose features already exist is skipped instantly,
    so you can stop it any time (Ctrl+C) and re-run - it continues where it
    left off.
  - FAILURE-PROOF: one bad star (MAST timeout, missing data) is logged and
    skipped; the batch keeps going.
  - BALANCED: targets are interleaved planet/false-positive using labels.csv,
    so "--limit 20" gives ~10 of each class, never 20 of one.

USAGE (on your laptop - needs internet for the download step)
-------------------------------------------------------------
    python pipeline/run_batch.py --limit 20      # quick validation batch
    python pipeline/run_batch.py                 # the full list (overnight)
    python pipeline/run_batch.py --max-quarters 4   # faster, less data/star

THEN:
    python pipeline/export.py --labels pipeline/labels.csv
    python models/xgboost_model.py --csv data/features/dataset_tabular.csv
"""
from __future__ import annotations
import argparse, csv, sys, time as clock
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import filter as flt           # process_file(raw_npy) -> bool
import preprocess as pre       # process_file(filtered_npy, window_hours) -> bool
import search as srch          # process_file(detrended_npy, p_min, p_max, bin_minutes) -> bool
import features as feat        # process_target(stem) -> bool

RAW   = HERE.parent / "data" / "raw"
CLEAN = HERE.parent / "data" / "cleaned"
FEAT  = HERE.parent / "data" / "features"


def _safe_filename(target: str) -> str:
    """'KIC 11904151' -> 'KIC_11904151' (same rule as download.py)."""
    return "".join(c if c.isalnum() else "_" for c in target).strip("_")


def failure_kind(exc: Exception) -> str:
    """Keep the resumable batch log useful when MAST or a cached FITS fails."""
    text = str(exc).lower()
    if "timeout" in text or "timed out" in text:
        return "MAST TIMEOUT"
    if "mast.stsci.edu" in text or "max retries exceeded" in text:
        return "MAST NETWORK ERROR"
    if any(s in text for s in ("corrupt", "not recognized as a supported",
                               "error in reading", "invalid argument")):
        return "CORRUPT FILE"
    return "FAILED"


def read_targets(path: Path):
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows or "target" not in rows[0]:
        sys.exit(f"[batch] {path} needs a 'target' column - run get_labels.py first.")
    return [(r["target"].strip(), (r.get("mission") or "Kepler").strip()) for r in rows]


def read_labels(path: Path):
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8") as fh:
        return {r["target"]: r["label"] for r in csv.DictReader(fh)}


def interleave_by_class(targets, labels):
    """Alternate CONFIRMED / FALSE POSITIVE so --limit N stays ~balanced."""
    pos = [t for t in targets if labels.get(t[0]) == "CONFIRMED"]
    neg = [t for t in targets if labels.get(t[0]) == "FALSE POSITIVE"]
    other = [t for t in targets if labels.get(t[0]) not in ("CONFIRMED", "FALSE POSITIVE")]
    mixed = []
    for i in range(max(len(pos), len(neg))):
        if i < len(pos): mixed.append(pos[i])
        if i < len(neg): mixed.append(neg[i])
    return mixed + other


def download_long(target: str, mission: str, max_quarters: int):
    """Long-cadence-only download, capped at N quarters.

    Long cadence = one measurement per 30 min. For finding/measuring transits
    that last hours, that's all the model needs - and it is ~30x smaller and
    faster than short cadence. (Kepler-10 full download: 1.48M points;
    8 quarters long-cadence: ~30k points. Same features, 50x less wait.)
    """
    import lightkurve as lk   # imported here so offline stages never need it
    import download as dl     # (download.py requires lightkurve at import time)
    author = "Kepler" if mission == "Kepler" else None
    sr = lk.search_lightcurve(target, mission=mission, author=author, cadence="long")
    if len(sr) == 0:
        raise RuntimeError("no MAST results")
    if max_quarters and len(sr) > max_quarters:
        sr = sr[:max_quarters]
    col = sr.download_all(flux_column="pdcsap_flux")
    if col is None or len(col) == 0:
        raise RuntimeError("download returned nothing")
    dl.save_light_curve(col.stitch(), target)


def run_star(target: str, mission: str, a) -> str:
    stem = _safe_filename(target)
    if (FEAT / f"{stem}_features.json").exists() and not a.redo:
        return "cached"
    if not (RAW / f"{stem}.npy").exists() or a.redo:
        download_long(target, mission, a.max_quarters)
    if not flt.process_file(RAW / f"{stem}.npy"):
        raise RuntimeError("filter stage failed")
    if not pre.process_file(CLEAN / f"{stem}_filtered.npy", a.window_hours):
        raise RuntimeError("preprocess stage failed")
    if not srch.process_file(CLEAN / f"{stem}_detrended.npy", a.p_min, None, a.bin_minutes):
        raise RuntimeError("search stage failed")
    if not feat.process_target(stem):
        raise RuntimeError("features stage failed")
    return "ok"


def main(argv=None):
    ap = argparse.ArgumentParser(description="Batch: download + full pipeline for the training list.")
    ap.add_argument("--targets", default=str(HERE / "targets_training.csv"))
    ap.add_argument("--labels",  default=str(HERE / "labels.csv"))
    ap.add_argument("--limit", type=int, default=0, help="only the first N stars (0 = all)")
    ap.add_argument("--max-quarters", type=int, default=8,
                    help="Kepler quarters per star (8 = ~2 years, good default)")
    ap.add_argument("--window-hours", type=float, default=48.0)
    ap.add_argument("--p-min", type=float, default=0.5)
    ap.add_argument("--bin-minutes", type=float, default=10.0)
    ap.add_argument("--resume", action="store_true",
                    help="explicitly request the default resume-safe behavior: "
                         "skip targets that already have feature files")
    ap.add_argument("--redo", action="store_true", help="reprocess even if features exist")
    a = ap.parse_args(argv)

    targets = read_targets(Path(a.targets))
    labels = read_labels(Path(a.labels))
    if labels:
        targets = interleave_by_class(targets, labels)
    else:
        print("[batch] note: no labels.csv found - cannot balance classes for --limit")
    if a.limit:
        targets = targets[: a.limit]

    n = len(targets)
    print(f"[batch] {n} stars | long cadence, max {a.max_quarters} quarters each")
    ok = cached = failed = 0
    failures = []
    times = []
    t_start = clock.time()
    for i, (target, mission) in enumerate(targets, 1):
        lab = labels.get(target, "?")
        t0 = clock.time()
        try:
            status = run_star(target, mission, a)
        except KeyboardInterrupt:
            print("\n[batch] interrupted - re-run the same command to resume.")
            break
        except Exception as e:
            failed += 1
            kind = failure_kind(e)
            failures.append((target, kind, str(e)[:120]))
            print(f"[{i}/{n}] {target} ({lab}): {kind} "
                  f"({ok} OK / {cached} cached / {failed} failed) - {str(e)[:120]}")
            continue
        dt = clock.time() - t0
        if status == "cached":
            cached += 1
            print(f"[{i}/{n}] {target} ({lab}): already done, skipped")
        else:
            ok += 1
            times.append(dt)
            avg = sum(times) / len(times)
            left = (n - i) * avg
            print(f"[{i}/{n}] {target} ({lab}): OK in {dt:.0f}s | "
                  f"{ok} OK / {cached} cached / {failed} failed | ~{left/60:.0f} min remaining")

    total_min = (clock.time() - t_start) / 60
    print(f"\n[batch] finished in {total_min:.1f} min: "
          f"{ok} processed, {cached} cached, {failed} failed")
    if failures:
        print("[batch] failures (safe to ignore a few - MAST hiccups happen):")
        for t, kind, msg in failures[:15]:
            print(f"    {t} [{kind}]: {msg}")

    n_feat = len(list(FEAT.glob("*_features.json")))
    n_labeled = sum(1 for f in FEAT.glob("*_features.json")
                    if labels.get(f.name.replace("_features.json", "").replace("_", " ", 1)))
    print(f"[batch] features on disk: {n_feat} stars")
    print("[batch] NEXT:")
    print("    python pipeline/export.py --labels pipeline/labels.csv")
    print("    python models/xgboost_model.py --csv data/features/dataset_tabular.csv")


if __name__ == "__main__":
    main()
