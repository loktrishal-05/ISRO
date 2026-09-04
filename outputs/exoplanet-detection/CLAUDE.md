# CLAUDE.md - project context for Claude Code

## What this project is
"AI-enabled Detection of Exoplanets from Noisy Astronomical Light Curves" -
ISRO Bhartiya Antariksh Hackathon (BAH) 2026 entry. Detects transiting
exoplanets in NASA Kepler data and separates them from eclipsing-binary
impostors with an ML classifier, then characterizes each detection
(size vs Earth/Mars/Jupiter, habitability, distance, host star).

The owner is an AI/DS student with no astronomy background - explain
concepts in plain language, avoid unexplained jargon, and keep every step
verifiable with a command they can run.

## Environment (IMPORTANT)
- Windows 11, PowerShell. Python 3.14.6.
- Both `py` and `python` now work (2026-07-19: Store stub aliases deleted,
  Python\bin moved to front of user PATH). Full path if needed:
  `C:\Users\Lohith k\AppData\Local\Python\pythoncore-3.14-64\python.exe`
- Project root: `C:\Users\Lohith k\Desktop\ISRO\outputs\exoplanet-detection`
  BEWARE: a STALE duplicate exists under `C:\Users\Lohith k\OneDrive\Desktop\ISRO\...`
  - never touch the OneDrive one.
- Installed: numpy, scipy, pandas, matplotlib, astropy, lightkurve,
  xgboost 3.3.0 (trainer auto-uses it), fastapi, uvicorn.
- Git repo since 2026-07-19 (main branch). GitHub push pending one
  interactive step: `gh auth login` then
  `gh repo create exoplanet-detection --private --source . --push`
  (gh CLI 2.96 installed at "C:\Program Files\GitHub CLI\gh.exe").

## Layout
    pipeline/   7-stage data pipeline, each file has a plain-English docstring
      download.py      Stage 1: MAST -> data/raw/<stem>.npy+.csv (stem: "KIC 123"->"KIC_123")
      filter.py        Stage 2: NaN/cosmic-ray/gap cleanup -> *_filtered.npy
      preprocess.py    Stage 3: Savitzky-Golay detrend -> *_detrended.npy
      search.py        Stage 4: BLS period search (10-min binning >200k pts,
                       250k-trial capped grid, astropy engine + numpy fallback) -> *_bls.npy
      features.py      Stage 5: 13 physics features -> data/features/<stem>_features.json
      characterize.py  Stage 5b: radius/habitability/distance (12-check --selftest)
      export.py        Stage 6: dataset_tabular.csv + dataset_views.npz,
                       star-level 70/15/15 split (seed 42), --labels join on 'target'
      run_batch.py     orchestrator: download(long-cadence, 8 quarters)->stages
                       per star; resume-safe, per-star error isolation,
                       class-interleaved; --limit N for small batches
      get_labels.py    KOI verdicts -> labels.csv + targets_training.csv (400 stars)
      fetch_stellar.py NASA stellar catalog -> stellar_params.csv (teff/srad/smass/dist)
    models/
      xgboost_model.py trainer: XGBoost or pure-numpy GBT fallback; isotonic
                       calibration on val, best-F1 threshold on val, test touched
                       once; predict_one() ready for the API; --selftest (8 checks)
      evaluate.py      AUC-PR, ROC-AUC, Brier, ECE, reliability, PAV isotonic
      saved/           trained model + calibration + threshold + eval_report.json
    data/       raw/ cleaned/ features/ (gitignore candidates - large)
    preview/index.html  demo dashboard (3D scene, results page, Habitability panel)

## Non-negotiable ML honesty rules (the project's brand)
1. Train ONLY on CONFIRMED vs FALSE POSITIVE. NEVER on CANDIDATE rows.
2. Splits are BY STAR (no star in two splits). Never resplit per-row.
3. Headline metric: AUC-PR. Calibrate on val, pick threshold on val,
   touch test exactly once.
4. Habitability is an "indicator", never a claim (transits give size/orbit,
   not atmosphere).
5. UI must not show models that do not exist. The mock "LSTM/Transformer"
   bars in preview/index.html must be removed or replaced by real models.

## Verified state (2026-07-20)
- Full pipeline verified on real NASA data: rediscovered Kepler-10b
  (P=0.8375 d vs NASA 0.837; depth 187ppm; SDE 41.5) and characterized it
  (1.58 R_earth super-Earth, ~1980K, 605 ly - matches published values).
- v1 batch DONE: 400 stars (top-SNR sample) fully processed; 29 MAST/cache
  failures all recovered on resume re-run (corrupt ~/.cache/lightkurve dirs
  must be deleted first - that was the fix).
- v1 training result - REPORT HONESTLY, NEVER as the headline:
  test AUC-PR=1.000 (61 rows: 31/0/0/30 confusion, ECE=0.00). The top-SNR
  sample made the task trivially separable by depth_ppm+snr_bls (51%+19%
  importance). It's a "benchmark saturated" result, not model quality.
- Ablation WITHOUT depth_ppm/snr_bls on v1 data: test AUC-PR=0.913 -
  physics features (secondary_ratio, v_metric, noise_ppm, sde) carry the
  EB-rejection story. `--drop f1,f2` and `--out dir` flags added to trainer.
- Benchmark status (always keep both results visible):
  - **High-SNR sample (v1): 1.000** test AUC-PR — saturated, never headline.
  - **SNR-stratified sample (v2): X after retraining** — the honest benchmark.
    It must report test AUC-PR, threshold confusion, ECE, reliability table,
    and an ablation without depth_ppm/snr_bls.
- v2 SNR-stratified list built: get_labels.py --stratified ->
  targets_training_v2.csv (200/class, 40/SNR-quintile, seed 42; only 40
  stars overlap v1). Flow verified with --limit 2. Overnight v2 batch:
  `py -u pipeline\run_batch.py --targets pipeline\targets_training_v2.csv`.
  Then run export, the main trainer, and the ablation trainer; replace X only
  with the resulting held-out test metric.
- predict_one() now reads the feature list from eval_report.json, so
  ablation models serve correctly through the API.
- Known quirk: name-based targets ("Kepler-10") don't join labels.csv (keys
  are "KIC <id>") -> such rows export with label="" and are excluded. Fine.
- FastAPI backend scaffolded in api/main.py (POST /analyze/{target},
  GET /results/{target}, /health); smoke-tested against Kepler-10.

## Roadmap (agreed priorities)
P0  1. `py pipeline\run_batch.py` full 400-star batch (overnight, resume-safe)
    2. git init + .gitignore (exclude data/, models/saved/) + GitHub push
    3. Retrain on full data -> real AUC-PR + calibration report
    4. FastAPI backend wrapping pipeline + models predict_one();
       wire preview app to it (replace demo data)
    5. Make UI ensemble honest (real models only)
P1  6. CNN on dataset_views.npz (Colab), meta-learner over XGB+CNN+FP-rules
    7. Demo reel: Kepler-452b (habitable-zone hit), an EB rejection live,
       baseline-vs-ML comparison table, batch triage mode
P2  8. TESS support, PDF vetting report, limitations slide
Reframe (never claim falsely): "FPGA" in the PPT -> "edge-deployable,
ONNX-ready, FPGA port on roadmap" with measured ms/star latency.

## Cleanup owed
(done 2026-07-19: update_*.ps1, bootstrap_models.ps1, bootstrap.ps1,
.write-test all deleted)

## Commands cheat-sheet
    py pipeline\run_batch.py --limit 20      # balanced validation batch
    py pipeline\run_batch.py                 # full 400 (resume-safe)
    py pipeline\export.py --labels pipeline\labels.csv
    py models\xgboost_model.py --csv data\features\dataset_tabular.csv
    py models\xgboost_model.py --selftest    # 8 checks, offline
    py pipeline\characterize.py --selftest   # 12 checks, offline
    py pipeline\characterize.py --target <stem>
