"""
evaluate.py  -  THE MEASURING TAPE for every model we train
============================================================

WHAT THIS FILE DOES (plain English)
-----------------------------------
Training a model is easy; knowing whether it's any GOOD is the hard part.
This file is a small library of honest measurements, used by
xgboost_model.py (and later the CNN / meta-learner):

  * AUC-PR (average precision)  -> our HEADLINE metric (Key Decision:
      accuracy is meaningless here because planets are rare; AUC-PR
      punishes false alarms the way judges expect)
  * ROC-AUC                     -> secondary ranking metric
  * Isotonic CALIBRATION        -> turns raw model scores into honest
      probabilities ("87%" should be right ~87% of the time). This is the
      step that lets us legitimately say "calibrated confidence".
  * ECE + reliability table     -> proof the calibration worked
  * Best-F1 threshold           -> where to draw the planet/FP line
  * Confusion counts, Brier score

DESIGN NOTE: pure numpy + standard library only - no pandas/sklearn/scipy -
so it runs anywhere the pipeline runs, including offline sandboxes.
"""
from __future__ import annotations
import csv
import json
from pathlib import Path
import numpy as np


# ----------------------------------------------------------------------
# Dataset loading (stdlib csv - keeps this importable with numpy alone)
# ----------------------------------------------------------------------

def load_rows(csv_path):
    """Read dataset_tabular.csv (from pipeline/export.py) into row-dicts."""
    with open(csv_path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit(f"[evaluate] {csv_path} is empty - run the pipeline first.")
    for r in rows:
        r["label"] = (r.get("label") or "").strip().upper()
        r["split"] = (r.get("split") or "").strip().lower()
    return rows


def _to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def feature_matrix(rows, features, medians=None):
    """Rows -> (X, medians). NaNs are filled with TRAIN medians (passed in
    for val/test/inference so no information leaks backwards)."""
    X = np.array([[_to_float(r.get(f)) for f in features] for r in rows], float)
    if medians is None:
        medians = np.nanmedian(X, axis=0)
        medians = np.where(np.isnan(medians), 0.0, medians)
    bad = np.where(np.isnan(X))
    X[bad] = np.take(np.asarray(medians), bad[1])
    return X, medians


# ----------------------------------------------------------------------
# Ranking metrics
# ----------------------------------------------------------------------

def average_precision(y, scores):
    """AUC-PR. Walk down the ranking; every time we hit a real planet,
    record the precision so far; average those. 1.0 = perfect ranking."""
    y = np.asarray(y, float)
    s = np.asarray(scores, float)
    n_pos = y.sum()
    if n_pos == 0 or n_pos == len(y):
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    yo = y[order]
    precision_at_k = np.cumsum(yo) / np.arange(1, len(yo) + 1)
    return float((precision_at_k * yo).sum() / n_pos)


def roc_auc(y, scores):
    """Probability a random planet outranks a random false positive
    (Mann-Whitney formulation, ties get averaged ranks)."""
    y = np.asarray(y, float)
    s = np.asarray(scores, float)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), float)
    ranks[order] = np.arange(1, len(s) + 1)
    ss = s[order]
    i = 0
    while i < len(ss):                       # average the ranks of ties
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def brier(y, p):
    """Mean squared error of the probabilities. Lower = better + honest."""
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    return float(np.mean((p - y) ** 2))


# ----------------------------------------------------------------------
# Calibration: isotonic regression (pool-adjacent-violators), pure numpy
# ----------------------------------------------------------------------

def fit_isotonic(scores, y):
    """Learn a monotonic map raw-score -> honest probability, fitted on the
    VALIDATION set (never train: the model already aced train; never test:
    that would be grading our own homework)."""
    s = np.asarray(scores, float)
    t = np.asarray(y, float)
    order = np.argsort(s, kind="mergesort")
    s_sorted, t_sorted = s[order], t[order]

    means = []
    weights = []
    for val in t_sorted:                     # classic PAV: merge any block
        means.append(float(val))             # that breaks monotonicity
        weights.append(1.0)
        while len(means) > 1 and means[-2] >= means[-1]:
            w = weights[-1] + weights[-2]
            m = (means[-1] * weights[-1] + means[-2] * weights[-2]) / w
            means.pop(); weights.pop()
            means[-1], weights[-1] = m, w

    fitted = np.repeat(np.array(means), np.array(weights).astype(int))

    # collapse duplicate scores so np.interp gets strictly usable knots
    ux, inv = np.unique(s_sorted, return_inverse=True)
    uy = np.zeros(len(ux)); cnt = np.zeros(len(ux))
    np.add.at(uy, inv, fitted)
    np.add.at(cnt, inv, 1.0)
    uy /= cnt
    return {"x": ux.tolist(), "y": uy.tolist()}


def apply_isotonic(cal, scores):
    p = np.interp(np.asarray(scores, float), np.asarray(cal["x"]), np.asarray(cal["y"]))
    return np.clip(p, 0.0, 1.0)


def ece(y, p, bins=10):
    """Expected Calibration Error: average gap between what the model SAYS
    ("80% sure") and what actually HAPPENS (fraction that are planets)."""
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    err = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            err += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(err)


def reliability_table(y, p, bins=10):
    """Per-bin (predicted vs actual) - this becomes the reliability curve
    in the frontend and the eval report."""
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    out = []
    for b in range(bins):
        m = idx == b
        out.append({
            "bin": f"{b / bins:.1f}-{(b + 1) / bins:.1f}",
            "n": int(m.sum()),
            "mean_predicted": round(float(p[m].mean()), 4) if m.any() else None,
            "fraction_actual": round(float(y[m].mean()), 4) if m.any() else None,
        })
    return out


# ----------------------------------------------------------------------
# Threshold + confusion
# ----------------------------------------------------------------------

def best_f1_threshold(y, p):
    """Scan every distinct probability as a cut line; keep the one with the
    best F1 on VALIDATION. Returns (threshold, f1)."""
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    best_t, best_f1 = 0.5, -1.0
    for t in np.unique(p):
        pred = p >= t
        tp = float((pred & (y == 1)).sum())
        fp = float((pred & (y == 0)).sum())
        fn = float((~pred & (y == 1)).sum())
        denom = 2 * tp + fp + fn
        f1 = (2 * tp / denom) if denom else 0.0
        if f1 > best_f1:
            best_t, best_f1 = float(t), float(f1)
    return best_t, best_f1


def confusion(y, p, threshold):
    y = np.asarray(y, float)
    pred = np.asarray(p, float) >= threshold
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum()); tn = int((~pred & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)}


# ----------------------------------------------------------------------
# Tiny CLI: pretty-print a saved eval report
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "models/saved/eval_report.json")
    rep = json.loads(path.read_text())
    print(json.dumps(rep, indent=2))
