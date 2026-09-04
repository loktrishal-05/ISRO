
"""
export.py  -  Stage 6: DATASET ASSEMBLY (hand-off to the model team)
=====================================================================

WHAT THIS DOES (plain English)
------------------------------
Collects every star's features + CNN views into TWO tidy files the model
team can load in one line on Colab:

  data/features/dataset_tabular.csv   -> for XGBoost / FP-specialist
  data/features/dataset_views.npz     -> for the CNN (global+local arrays)

LABELS: pass --labels labels.csv (columns: target,label) where label is
CONFIRMED / FALSE POSITIVE / CANDIDATE. Rows without labels get label="".
(Reminder: train on CONFIRMED vs FALSE POSITIVE; CANDIDATEs are the
"unknowns" our demo predicts on - never train on them.)

THE LEAKAGE RULE (Key Decision: hold out entire STARS)
------------------------------------------------------
The same star must NEVER appear in both train and test - the model would
memorize that star's noise pattern and cheat. We split BY STAR with a fixed
seed: 70% train / 15% val / 15% test, stored in a "split" column.

USAGE
-----
    python export.py
    python export.py --labels pipeline/labels.csv --seed 42
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

FEAT_DIR = Path(__file__).resolve().parent.parent / "data" / "features"


def load_rows() -> pd.DataFrame:
    rows = []
    for f in sorted(FEAT_DIR.glob("*_features.json")):
        d = json.loads(f.read_text())
        d["stem"] = f.name.replace("_features.json", "")
        rows.append(d)
    if not rows:
        sys.exit(f"No *_features.json in {FEAT_DIR} - run features.py first.")
    return pd.DataFrame(rows)


def star_level_split(df: pd.DataFrame, seed: int) -> pd.Series:
    """70/15/15 split BY STAR (not by row) so no star leaks across sets."""
    stars = df["target"].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(stars)
    n = len(stars)
    train_ids = set(stars[: int(n * .70)])
    val_ids   = set(stars[int(n * .70): int(n * .85)])
    return df["target"].map(lambda s: "train" if s in train_ids
                            else "val" if s in val_ids else "test")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Stage 6: assemble final datasets.")
    ap.add_argument("--labels", help="CSV with columns target,label")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(argv)

    df = load_rows()

    # ---- attach labels if provided ----
    if a.labels:
        lab = pd.read_csv(a.labels)
        df = df.merge(lab[["target", "label"]], on="target", how="left")
        df["label"] = df["label"].fillna("")
        n_lab = (df["label"] != "").sum()
        print(f"[export] labels: {n_lab}/{len(df)} rows labeled")
    else:
        df["label"] = ""
        print("[export] no --labels given: exporting unlabeled (fine for inference)")

    df["split"] = star_level_split(df, a.seed)

    # ---- tabular dataset ----
    csv_path = FEAT_DIR / "dataset_tabular.csv"
    df.to_csv(csv_path, index=False)

    # ---- CNN views dataset (aligned row-for-row with the CSV) ----
    gv, lv, keep = [], [], []
    for i, stem in enumerate(df["stem"]):
        p = FEAT_DIR / f"{stem}_views.npy"
        if p.exists():
            v = np.load(p, allow_pickle=True).item()
            gv.append(v["global"]); lv.append(v["local"]); keep.append(i)
    npz_path = FEAT_DIR / "dataset_views.npz"
    np.savez_compressed(
        npz_path,
        global_view=np.stack(gv) if gv else np.empty((0,)),
        local_view=np.stack(lv) if lv else np.empty((0,)),
        target=df["target"].iloc[keep].to_numpy(dtype=object),
        label=df["label"].iloc[keep].to_numpy(dtype=object),
        split=df["split"].iloc[keep].to_numpy(dtype=object),
    )

    print(f"[export] {len(df)} rows -> {csv_path.name}")
    print(f"[export] {len(gv)} view-pairs -> {npz_path.name}")
    print("[export] split counts:", df["split"].value_counts().to_dict())
    print("[export] READY: models can now load dataset_tabular.csv / dataset_views.npz")


if __name__ == "__main__":
    main()
