# 🛰️ ExoDetect — AI-Enabled Exoplanet Detection from Noisy Light Curves

**ISRO Bhartiya Antariksh Hackathon 2026** · Team R.M.K. College of Engineering and Technology

When a planet crosses its star, the star dims by ~0.01–1%. We detect those dips in
noisy Kepler telescope data using a physics-aware pipeline + an AI ensemble with
calibrated confidence, explainability (SHAP), and a novelty branch for signals
nobody has seen before. We don't replace NASA's Robovetter — **we absorb its
vetting rules as features and add learned reasoning on top.**

## Architecture

    FITS (MAST / ISRO PRADAN)
      └─ pipeline/  download → filter → preprocess → search(BLS) → features → export
           └─ models/   XGBoost · 1D-CNN (global+local) · FP-specialist · meta-learner
                          + SSL autoencoder novelty branch (parallel, not downstream)
                └─ api/       FastAPI + Supabase
                     └─ frontend/  React + Three.js (360° transit scene, bloom) + dashboards

## Status

| Piece | State |
|---|---|
| Stage 1–6 data pipeline | ✅ built + verified (`python pipeline/selftest.py` → 9/9) |
| Interactive app preview (4 pages, 3D, dashboards) | ✅ `preview/app.html` |
| Models (XGBoost → CNN → FP → meta) | ⏳ next |
| FastAPI + Supabase | ⏳ |
| React frontend | ⏳ (preview doubles as design spec + demo fallback) |

## Quickstart (pipeline)

    pip install -r pipeline/requirements.txt
    python pipeline/selftest.py                          # verify env (offline, ~6s)
    python pipeline/download.py --batch pipeline/targets.csv   # needs internet
    python pipeline/filter.py && python pipeline/preprocess.py
    python pipeline/search.py && python pipeline/features.py
    python pipeline/export.py --labels pipeline/labels.csv

Outputs: `data/features/dataset_tabular.csv` (XGBoost) + `dataset_views.npz` (CNN).

## Key technical rules (do not break)

PDCSAP_FLUX not SAP_FLUX · SavGol window ≫ transit duration (48h default) ·
clip only UPWARD outliers pre-detection · LSTM eats UNFOLDED series ·
split by STAR not by row · report AUC-PR not accuracy ·
meta-learner trains on out-of-fold predictions · novelty branch runs in PARALLEL.

## Team

| Person | Owns |
|---|---|
| P1 Data+ML | pipeline/, CNN+LSTM |
| P2 ML+XAI | XGBoost, FP-specialist, autoencoder, meta, SHAP |
| P3 Backend | FastAPI, Supabase, Docker, deploy |
| P4 Frontend | React, Three.js, dashboards, Vercel |


## Models (Stage 8) - AI ensemble

| File | What it is |
|---|---|
| `models/xgboost_model.py` | First real model: gradient-boosted trees on the 13 physics features. Auto-uses real XGBoost if installed, else a built-in pure-numpy engine. |
| `models/evaluate.py` | Shared eval harness: AUC-PR, ROC-AUC, isotonic calibration, ECE/reliability, best-F1 threshold, confusion. Pure numpy - runs anywhere. |
| `pipeline/get_labels.py` | Downloads NASA's KOI verdicts -> `labels.csv` (training answers, incl. `fpflag_ss` eclipsing-binary flags for the FP specialist) + `targets_training.csv` (balanced download list). Needs internet. |

**Train on real data (laptop/Colab):**
```bash
pip install xgboost                                  # optional but recommended
python pipeline/get_labels.py                        # 1. fetch answers from NASA
python pipeline/download.py --batch pipeline/targets_training.csv   # 2. fetch light curves
python pipeline/filter.py && python pipeline/preprocess.py          # 3. clean
python pipeline/search.py && python pipeline/features.py            # 4. detect + measure
python pipeline/export.py --labels pipeline/labels.csv              # 5. assemble dataset
python models/xgboost_model.py                       # 6. train + calibrate + report
```
Outputs land in `models/saved/`: the trained model, calibration map, decision
threshold, and `eval_report.json` (per-split AUC-PR/ROC-AUC/ECE, reliability
table, feature importance - the numbers for the judges).

**Verify offline (no internet, ~2s):** `python models/xgboost_model.py --selftest`

## Results so far (honest version)

| Benchmark | Test AUC-PR | Notes |
|---|---|---|
| High-SNR sample (v1: 400 stars, top-SNR per class) | 1.000 | **Too easy - not a headline number.** Loud signals only; depth_ppm + snr_bls separate the classes alone (51% + 19% of importance). Confusion 31/0/0/30, ECE 0.00 - the task, not the model, is saturated. |
| SNR-stratified sample (v2: 200/class, 40 per SNR quintile, seed 42) | **pending — X after retraining** | This is the honest benchmark: it includes quiet/hard signals. Report test AUC-PR, the threshold confusion matrix, ECE, reliability table, and the depth/SNR ablation before using it as a headline result. |
| Ablation: no depth_ppm/snr_bls (v1 data) | 0.913 | Physics features alone (secondary_ratio, v_metric, noise_ppm, sde) still classify - the eclipsing-binary story holds without the two "loudness" features. |

**Run the honest benchmark:**
```bash
python pipeline/run_batch.py --targets pipeline/targets_training_v2.csv
python pipeline/export.py --labels pipeline/labels.csv
python models/xgboost_model.py
python models/xgboost_model.py --drop depth_ppm,snr_bls --out models/saved_ablation
```
The v2 target list is separate from v1; completed stars are resume-safe and are skipped.

Ablation runs: `python models/xgboost_model.py --drop depth_ppm,snr_bls --out models/saved_ablation`
