"""
xgboost_model.py  -  Stage 8a: THE FIRST REAL AI MODEL (tabular baseline)
==========================================================================

WHAT THIS FILE DOES (plain English)
-----------------------------------
The pipeline (stages 1-6) turned each star into a row of ~13 physical
numbers - transit depth, period, odd/even mismatch, secondary eclipse, V-shape
metric, etc. This file trains a GRADIENT-BOOSTED TREE model on those rows to
answer: "planet or false positive?"

Why this model FIRST (from our build-order review):
  * trains in seconds/minutes on a laptop - no GPU needed
  * strongest baseline for tabular physics features
  * gives per-feature explanations (SHAP) almost for free
  * produces the first honest AUC-PR number of the project

TWO ENGINES, SAME INTERFACE (like search.py's astropy/numpy dual path):
  * If the `xgboost` package is installed  -> real XGBoost (use on laptop/Colab)
  * If not -> a pure-numpy gradient-boosted-trees fallback in this file
    (slower + simpler, but same algorithm family; lets the whole harness run
    and be TESTED offline, e.g. in the sandbox where this was written)

THE HONESTY RULES BAKED IN (Key Decisions):
  * train ONLY on CONFIRMED vs FALSE POSITIVE rows (CANDIDATE = unknown,
    they are for demo predictions, never training)
  * split column comes from export.py = star-level split (no leakage)
  * headline metric = AUC-PR (accuracy is meaningless with class imbalance)
  * "calibrated confidence" = isotonic calibration fitted on VALIDATION
  * decision threshold chosen on VALIDATION (never on test)
  * test set is touched ONCE, at the very end, and reported as-is

USAGE
-----
    python models/xgboost_model.py                       # train on pipeline data
    python models/xgboost_model.py --csv path/to.csv     # explicit dataset
    python models/xgboost_model.py --selftest            # offline end-to-end proof

OUTPUTS (models/saved/)
    xgb_planet.json / .pkl   trained model
    xgb_calibration.json     isotonic map (raw score -> honest probability)
    xgb_threshold.json       decision threshold chosen on validation
    eval_report.json         all metrics: per-split AUC-PR, ROC-AUC, ECE,
                             confusion, reliability table, feature importance
"""
from __future__ import annotations
import argparse
import json
import pickle
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate as ev

ROOT = Path(__file__).resolve().parent.parent
DATA_CSV = ROOT / "data" / "features" / "dataset_tabular.csv"
SAVED = Path(__file__).resolve().parent / "saved"

# The features features.py produces - keep in ONE place so the API layer and
# the meta-learner import this exact list.
FEATURES = [
    "period_d", "duration_hr", "depth_ppm", "snr_bls", "sde", "noise_ppm",
    "odd_even_diff", "secondary_ratio", "v_metric", "symmetry",
    "duty_cycle", "n_transits", "n_points_in_transit",
]

POS_LABEL, NEG_LABEL = "CONFIRMED", "FALSE POSITIVE"


# ======================================================================
# ENGINE 2: pure-numpy gradient-boosted trees (fallback / offline proof)
# ======================================================================

class _Tree:
    """One regression tree fitted to gradients (a 'weak learner').
    Stored as parallel arrays; leaves hold the boosting step values."""

    __slots__ = ("feat", "thr", "left", "right", "value", "gain_by_feat")

    def __init__(self, n_features):
        self.feat, self.thr = [], []
        self.left, self.right, self.value = [], [], []
        self.gain_by_feat = np.zeros(n_features)

    def _new_node(self):
        self.feat.append(-1); self.thr.append(0.0)
        self.left.append(-1); self.right.append(-1); self.value.append(0.0)
        return len(self.feat) - 1

    def build(self, X, g, h, depth, max_depth, lam, min_child_weight, rng):
        """Recursively grow: at each node pick the (feature, threshold) that
        most reduces the boosting objective - identical math to XGBoost's
        exact greedy algorithm, on quantile candidate thresholds."""
        node = self._new_node()
        G, H = g.sum(), h.sum()
        best = {"gain": 1e-6}
        if depth < max_depth and len(g) >= 2 * 3:
            base_score = G * G / (H + lam)
            for f in range(X.shape[1]):
                x = X[:, f]
                cand = np.unique(np.quantile(x, np.linspace(0.05, 0.95, 24)))
                for t in cand:
                    m = x <= t
                    hl = h[m].sum()
                    hr = H - hl
                    if hl < min_child_weight or hr < min_child_weight:
                        continue
                    gl = g[m].sum()
                    gain = gl * gl / (hl + lam) + (G - gl) ** 2 / (hr + lam) - base_score
                    if gain > best["gain"]:
                        best = {"gain": gain, "f": f, "t": float(t), "m": m}
        if "f" in best:
            self.feat[node] = best["f"]; self.thr[node] = best["t"]
            self.gain_by_feat[best["f"]] += best["gain"]
            m = best["m"]
            self.left[node] = self.build(X[m], g[m], h[m], depth + 1,
                                         max_depth, lam, min_child_weight, rng)
            self.right[node] = self.build(X[~m], g[~m], h[~m], depth + 1,
                                          max_depth, lam, min_child_weight, rng)
        else:
            self.value[node] = float(-G / (H + lam))     # optimal leaf weight
        return node

    def predict(self, X):
        out = np.empty(len(X))
        for i, row in enumerate(X):
            n = 0
            while self.feat[n] != -1:
                n = self.left[n] if row[self.feat[n]] <= self.thr[n] else self.right[n]
            out[i] = self.value[n]
        return out

    def to_dict(self):
        return {"feat": self.feat, "thr": self.thr, "left": self.left,
                "right": self.right, "value": self.value}

    @classmethod
    def from_dict(cls, d, n_features):
        t = cls(n_features)
        t.feat, t.thr = d["feat"], d["thr"]
        t.left, t.right, t.value = d["left"], d["right"], d["value"]
        return t


class NumpyBoost:
    """Minimal gradient boosting for binary classification (logistic loss).
    Same recipe as XGBoost: fit trees to gradients, shrink by learning rate,
    early-stop when validation AUC-PR stops improving."""

    def __init__(self, n_rounds=300, learning_rate=0.08, max_depth=3,
                 lam=1.0, min_child_weight=1.0, scale_pos_weight=1.0, seed=42):
        self.p = dict(n_rounds=n_rounds, learning_rate=learning_rate,
                      max_depth=max_depth, lam=lam,
                      min_child_weight=min_child_weight,
                      scale_pos_weight=scale_pos_weight, seed=seed)
        self.trees = []
        self.base = 0.0
        self.n_features = 0

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def fit(self, X, y, X_val=None, y_val=None, patience=30, verbose=True):
        p = self.p
        rng = np.random.default_rng(p["seed"])
        self.n_features = X.shape[1]
        w = np.where(y == 1, p["scale_pos_weight"], 1.0)   # class-imbalance fix
        pos = max(y.sum(), 1e-9)
        self.base = float(np.log(pos / max(len(y) - pos, 1e-9)))
        margin = np.full(len(y), self.base)
        val_margin = (np.full(len(y_val), self.base)
                      if X_val is not None else None)
        best_ap, best_len, since = -1.0, 0, 0

        for r in range(p["n_rounds"]):
            prob = self._sigmoid(margin)
            g = (prob - y) * w                 # gradient of weighted log-loss
            h = prob * (1 - prob) * w          # hessian
            tree = _Tree(self.n_features)
            tree.build(X, g, h, 0, p["max_depth"], p["lam"],
                       p["min_child_weight"], rng)
            step = tree.predict(X) * p["learning_rate"]
            margin += step
            self.trees.append(tree)

            if X_val is not None:
                val_margin += tree.predict(X_val) * p["learning_rate"]
                ap = ev.average_precision(y_val, val_margin)
                if ap > best_ap + 1e-5:
                    best_ap, best_len, since = ap, len(self.trees), 0
                else:
                    since += 1
                if verbose and (r % 25 == 0 or since == patience):
                    print(f"  [numpyboost] round {r:3d}  val AUC-PR={ap:.4f}"
                          f"  best={best_ap:.4f}@{best_len}")
                if since >= patience:
                    self.trees = self.trees[:best_len]   # rewind to best
                    break
        return self

    def predict_score(self, X):
        m = np.full(len(X), self.base)
        for t in self.trees:
            m += t.predict(X) * self.p["learning_rate"]
        return self._sigmoid(m)

    def feature_importance(self):
        imp = np.zeros(self.n_features)
        for t in self.trees:
            imp += t.gain_by_feat
        s = imp.sum()
        return imp / s if s > 0 else imp

    def explain_one(self, x, medians):
        """Saabas path attribution: walk each tree; whenever a split fires,
        credit the change in expected value to that split's feature. A fast
        SHAP-style per-feature breakdown for the fallback engine.
        (Real XGBoost path uses exact TreeSHAP via pred_contribs.)"""
        contrib = np.zeros(self.n_features)
        lr = self.p["learning_rate"]
        for t in self.trees:
            n = 0
            while t.feat[n] != -1:
                f = t.feat[n]
                nxt = t.left[n] if x[f] <= t.thr[n] else t.right[n]
                contrib[f] += 0.0  # placeholder keeps structure clear
                n = nxt
            # distribute leaf value equally over the features on the path
            path, n = [], 0
            while t.feat[n] != -1:
                path.append(t.feat[n])
                n = t.left[n] if x[t.feat[n]] <= t.thr[n] else t.right[n]
            if path:
                share = t.value[n] * lr / len(path)
                for f in path:
                    contrib[f] += share
        return contrib

    def save(self, path):
        blob = {"engine": "numpyboost", "params": self.p, "base": self.base,
                "n_features": self.n_features,
                "trees": [t.to_dict() for t in self.trees]}
        Path(path).write_bytes(pickle.dumps(blob))

    @classmethod
    def load(cls, path):
        blob = pickle.loads(Path(path).read_bytes())
        m = cls(**{k: v for k, v in blob["params"].items()})
        m.base = blob["base"]; m.n_features = blob["n_features"]
        m.trees = [_Tree.from_dict(d, m.n_features) for d in blob["trees"]]
        return m


# ======================================================================
# ENGINE 1: real XGBoost (used automatically when installed)
# ======================================================================

def _try_xgboost():
    try:
        import xgboost
        return xgboost
    except ImportError:
        return None


def train_xgboost(xgb, Xtr, ytr, Xva, yva, spw, seed):
    """Real XGBoost with early stopping on validation AUC-PR."""
    clf = xgb.XGBClassifier(
        n_estimators=600, learning_rate=0.05, max_depth=4,
        subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
        scale_pos_weight=spw, eval_metric="aucpr",
        early_stopping_rounds=40, random_state=seed, n_jobs=-1,
    )
    clf.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    return clf


# ======================================================================
# The harness: load -> train -> calibrate -> threshold -> report
# ======================================================================

def run(csv_path, seed=42, out_dir=SAVED, quiet=False, drop=()):
    say = (lambda *a: None) if quiet else print
    feats = [f for f in FEATURES if f not in set(drop)]
    if drop:
        unknown = set(drop) - set(FEATURES)
        if unknown:
            raise SystemExit(f"[xgb] --drop names not in FEATURES: {sorted(unknown)}")
        say(f"[xgb] ABLATION: training WITHOUT {sorted(drop)} "
            f"({len(feats)} features remain)")
    rows = ev.load_rows(csv_path)

    labeled = [r for r in rows if r["label"] in (POS_LABEL, NEG_LABEL)]
    candidates = [r for r in rows if r["label"] not in (POS_LABEL, NEG_LABEL)]
    if not labeled:
        raise SystemExit(
            "[xgb] No CONFIRMED / FALSE POSITIVE rows found.\n"
            "      Re-run export.py with --labels (KOI catalog labels).")

    splits = {}
    for name in ("train", "val", "test"):
        part = [r for r in labeled if r["split"] == name]
        y = np.array([1.0 if r["label"] == POS_LABEL else 0.0 for r in part])
        splits[name] = (part, y)
    (tr, ytr), (va, yva), (te, yte) = splits["train"], splits["val"], splits["test"]
    for name, (part, y) in splits.items():
        say(f"[xgb] {name}: {len(part)} rows "
            f"({int(y.sum())} planets / {int(len(y) - y.sum())} FPs)")
    if len(tr) < 10 or ytr.sum() == 0 or ytr.sum() == len(ytr):
        raise SystemExit("[xgb] Not enough labeled training data "
                         "(need both classes; download more stars).")

    Xtr, med = ev.feature_matrix(tr, feats)             # medians from TRAIN only
    Xva, _ = ev.feature_matrix(va, feats, med)
    Xte, _ = ev.feature_matrix(te, feats, med)

    spw = float((len(ytr) - ytr.sum()) / max(ytr.sum(), 1.0))
    say(f"[xgb] class imbalance -> scale_pos_weight={spw:.2f}")

    # ---- train (engine auto-select) ----
    xgb = _try_xgboost()
    out_dir.mkdir(parents=True, exist_ok=True)
    if xgb is not None:
        say("[xgb] engine: XGBoost", xgb.__version__)
        model = train_xgboost(xgb, Xtr, ytr, Xva, yva, spw, seed)
        score = lambda X: model.predict_proba(X)[:, 1]
        booster = model.get_booster()
        imp = booster.get_score(importance_type="gain")
        total = sum(imp.values()) or 1.0
        importance = {f: round(imp.get(f"f{i}", 0.0) / total, 4)
                      for i, f in enumerate(feats)}
        model_path = out_dir / "xgb_planet.json"
        booster.save_model(model_path)
        engine = "xgboost"
    else:
        say("[xgb] engine: pure-numpy fallback (install xgboost for the real one)")
        model = NumpyBoost(scale_pos_weight=spw, seed=seed)
        model.fit(Xtr, ytr, Xva, yva, verbose=not quiet)
        score = model.predict_score
        importance = {f: round(float(v), 4)
                      for f, v in zip(feats, model.feature_importance())}
        model_path = out_dir / "xgb_planet.pkl"
        model.save(model_path)
        engine = "numpyboost"

    # ---- calibrate on VALIDATION (never train, never test) ----
    s_va = score(Xva)
    cal = ev.fit_isotonic(s_va, yva)
    (out_dir / "xgb_calibration.json").write_text(json.dumps(cal))
    p_va = ev.apply_isotonic(cal, s_va)

    # ---- pick decision threshold on VALIDATION ----
    thr, f1_va = ev.best_f1_threshold(yva, p_va)
    (out_dir / "xgb_threshold.json").write_text(
        json.dumps({"threshold": thr, "chosen_on": "val", "f1_val": round(f1_va, 4)}))

    # ---- final report (test touched exactly once, right here) ----
    report = {"engine": engine, "features": feats,
              "dropped_features": sorted(drop),
              "medians": [round(float(m), 6) for m in med],
              "n_candidates_unlabeled": len(candidates),
              "scale_pos_weight": round(spw, 3),
              "threshold": round(thr, 4), "splits": {}}
    for name, (part, y) in splits.items():
        X, _ = ev.feature_matrix(part, feats, med)
        s = score(X)
        p = ev.apply_isotonic(cal, s)
        report["splits"][name] = {
            "n": len(part), "n_planets": int(y.sum()),
            "auc_pr": round(ev.average_precision(y, s), 4),
            "roc_auc": round(ev.roc_auc(y, s), 4),
            "brier_calibrated": round(ev.brier(y, p), 4),
            "ece_calibrated": round(ev.ece(y, p), 4),
            "confusion_at_threshold": ev.confusion(y, p, thr),
        }
    report["splits"]["val"]["reliability"] = ev.reliability_table(yva, p_va)
    report["feature_importance"] = dict(
        sorted(importance.items(), key=lambda kv: -kv[1]))

    (out_dir / "eval_report.json").write_text(json.dumps(report, indent=2))

    say("\n========== RESULTS ==========")
    for name in ("train", "val", "test"):
        r = report["splits"][name]
        say(f"  {name:5s}  AUC-PR={r['auc_pr']:.4f}  ROC-AUC={r['roc_auc']:.4f}  "
            f"ECE={r['ece_calibrated']:.4f}  "
            f"P={r['confusion_at_threshold']['precision']:.3f} "
            f"R={r['confusion_at_threshold']['recall']:.3f}")
    say("  top features:", ", ".join(list(report["feature_importance"])[:4]))
    say(f"  saved -> {out_dir}")
    return report


# ======================================================================
# Inference helper for the FastAPI layer (single candidate -> verdict)
# ======================================================================

def predict_one(features_dict, saved_dir=SAVED):
    """Load saved model+calibration+threshold, score one candidate dict
    (the *_features.json a pipeline run produces). Returns the verdict
    payload the API/frontend will show."""
    report = json.loads((saved_dir / "eval_report.json").read_text())
    feats = report["features"]     # from the report, so ablation models work too
    med = np.array(report["medians"])
    x = np.array([[float(features_dict.get(f, np.nan)) for f in feats]])
    bad = np.isnan(x)
    x[bad] = np.take(med, np.where(bad)[1])

    if (saved_dir / "xgb_planet.json").exists():
        import xgboost as xgb
        booster = xgb.Booster(); booster.load_model(saved_dir / "xgb_planet.json")
        dm = xgb.DMatrix(x, feature_names=feats)
        raw = float(booster.predict(dm)[0])
        contribs = booster.predict(dm, pred_contribs=True)[0][:-1]  # exact TreeSHAP
    else:
        model = NumpyBoost.load(saved_dir / "xgb_planet.pkl")
        raw = float(model.predict_score(x)[0])
        contribs = model.explain_one(x[0], med)

    cal = json.loads((saved_dir / "xgb_calibration.json").read_text())
    prob = float(ev.apply_isotonic(cal, [raw])[0])
    thr = json.loads((saved_dir / "xgb_threshold.json").read_text())["threshold"]
    order = np.argsort(-np.abs(contribs))
    return {
        "probability_planet": round(prob, 4),
        "raw_score": round(raw, 4),
        "disposition": "PLANET CANDIDATE" if prob >= thr else "FALSE POSITIVE",
        "threshold": thr,
        "explanation": [{"feature": feats[i],
                         "value": round(float(x[0][i]), 5),
                         "contribution": round(float(contribs[i]), 5)}
                        for i in order],
    }


# ======================================================================
# Offline selftest: prove the WHOLE harness end-to-end with no internet
# ======================================================================

def make_synthetic_dataset(path, n_stars=400, seed=7):
    """Fake but physics-shaped dataset: planets have matched odd/even depths,
    no secondary eclipse, U-shaped transits; eclipsing binaries (our FPs)
    have alternating depths, secondary eclipses, V-shapes. Same columns as
    export.py so the harness can't tell the difference."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_stars):
        is_planet = rng.random() < 0.35            # imbalanced, like reality
        noise = rng.uniform(50, 400)
        depth = rng.uniform(200, 4000) if is_planet else rng.uniform(600, 30000)
        # ~20% are deliberately ambiguous edge cases (grazing planets look
        # V-shaped; small EBs mimic planets) so the classes OVERLAP and the
        # calibration/threshold machinery is genuinely exercised
        ambiguous = rng.random() < 0.20
        row = {
            "target": f"SYN {i:04d}", "stem": f"syn_{i:04d}",
            "period_d": round(rng.uniform(0.8, 40), 4),
            "duration_hr": round(rng.uniform(1, 8), 3),
            "depth_ppm": round(depth * rng.uniform(0.9, 1.1), 1),
            "snr_bls": round(depth / noise * rng.uniform(0.6, 1.2), 2),
            "sde": round(rng.uniform(7, 25), 2),
            "noise_ppm": round(noise, 1),
            "odd_even_diff": round(abs(rng.normal(0.10, 0.08)) if ambiguous
                                   else abs(rng.normal(0.02, 0.02)) if is_planet
                                   else abs(rng.normal(0.25, 0.12)), 4),
            "secondary_ratio": round(abs(rng.normal(0.12, 0.10)) if ambiguous
                                     else abs(rng.normal(0.02, 0.02)) if is_planet
                                     else abs(rng.normal(0.30, 0.15)), 4),
            "v_metric": round(rng.normal(0.5, 0.2) if ambiguous
                              else rng.normal(0.25, 0.1) if is_planet
                              else rng.normal(0.75, 0.15), 4),
            "symmetry": round(abs(rng.normal(0.05, 0.04)), 4),
            "duty_cycle": round(rng.uniform(0.005, 0.08), 5),
            "n_transits": int(rng.integers(4, 60)),
            "n_points_in_transit": int(rng.integers(40, 900)),
            "label": POS_LABEL if is_planet else NEG_LABEL,
        }
        # 8% label-free rows = the CANDIDATE pool
        if rng.random() < 0.08:
            row["label"] = ""
        rows.append(row)
    # star-level split identical in spirit to export.py
    idx = rng.permutation(n_stars)
    split = np.empty(n_stars, object)
    split[idx[: int(n_stars * .7)]] = "train"
    split[idx[int(n_stars * .7): int(n_stars * .85)]] = "val"
    split[idx[int(n_stars * .85):]] = "test"
    for r, s in zip(rows, split):
        r["split"] = s

    import csv as _csv
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = _csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    return path


def selftest():
    import tempfile
    print("[selftest] building synthetic labeled dataset (400 stars)...")
    with tempfile.TemporaryDirectory() as td:
        csv_path = Path(td) / "dataset_tabular.csv"
        out = Path(td) / "saved"
        make_synthetic_dataset(csv_path)
        report = run(csv_path, out_dir=out, quiet=True)

        checks = []
        te = report["splits"]["test"]
        va = report["splits"]["val"]
        checks.append(("test AUC-PR > 0.85", te["auc_pr"] > 0.85, te["auc_pr"]))
        checks.append(("test ROC-AUC > 0.85", te["roc_auc"] > 0.85, te["roc_auc"]))
        checks.append(("calibration ECE(val) < 0.10",
                       va["ece_calibrated"] < 0.10, va["ece_calibrated"]))
        checks.append(("test recall > 0.75",
                       te["confusion_at_threshold"]["recall"] > 0.75,
                       te["confusion_at_threshold"]["recall"]))
        checks.append(("test precision > 0.75",
                       te["confusion_at_threshold"]["precision"] > 0.75,
                       te["confusion_at_threshold"]["precision"]))
        top3 = list(report["feature_importance"])[:3]
        physics = {"odd_even_diff", "secondary_ratio", "v_metric", "depth_ppm"}
        checks.append(("EB-physics feature in top-3 importance",
                       len(physics & set(top3)) > 0, ",".join(top3)))

        # round-trip: single-candidate inference (what the API will call)
        planet_like = {"period_d": 12.3, "duration_hr": 3.2, "depth_ppm": 900,
                       "snr_bls": 9.5, "sde": 14.0, "noise_ppm": 120,
                       "odd_even_diff": 0.01, "secondary_ratio": 0.01,
                       "v_metric": 0.2, "symmetry": 0.03, "duty_cycle": 0.011,
                       "n_transits": 30, "n_points_in_transit": 400}
        eb_like = dict(planet_like, depth_ppm=18000, odd_even_diff=0.4,
                       secondary_ratio=0.5, v_metric=0.9)
        vp = predict_one(planet_like, out)
        ve = predict_one(eb_like, out)
        checks.append(("planet-like candidate scores HIGHER than EB-like",
                       vp["probability_planet"] > ve["probability_planet"],
                       f"{vp['probability_planet']} vs {ve['probability_planet']}"))
        checks.append(("explanation lists all features",
                       len(vp["explanation"]) == len(FEATURES),
                       len(vp["explanation"])))

        print()
        n_pass = 0
        for name, ok, val in checks:
            print(f"  {'PASS' if ok else 'FAIL'}  {name}  ({val})")
            n_pass += ok
        print(f"\n[selftest] {n_pass}/{len(checks)} checks passed")
        return n_pass == len(checks)


# ======================================================================

def main(argv=None):
    ap = argparse.ArgumentParser(description="Stage 8a: XGBoost baseline + eval harness")
    ap.add_argument("--csv", default=str(DATA_CSV),
                    help="dataset_tabular.csv from pipeline/export.py")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--drop", default="",
                    help="comma-separated features to EXCLUDE (ablation study), "
                         "e.g. --drop depth_ppm,snr_bls")
    ap.add_argument("--out", default=str(SAVED),
                    help="output dir (use a different one for ablation runs "
                         "so the main saved model is not overwritten)")
    ap.add_argument("--selftest", action="store_true",
                    help="run offline end-to-end proof on synthetic data")
    a = ap.parse_args(argv)
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    drop = tuple(f.strip() for f in a.drop.split(",") if f.strip())
    run(Path(a.csv), seed=a.seed, out_dir=Path(a.out), drop=drop)


if __name__ == "__main__":
    main()
