# Pipeline — Stages 1–5 (Data)

Turns raw Kepler telescope data into model-ready arrays. Build order is strict:

    download.py -> filter.py -> preprocess.py -> search.py -> features.py -> export.py
    [DONE]         [DONE]       [DONE]           (next)       (next)         (next)

## Setup (machine WITH internet — laptop or Google Colab)

    pip install -r pipeline/requirements.txt

## Verify your environment (no internet needed)

    python pipeline/selftest.py      # builds a fake star, runs stages 2-3, checks physics

## Stage 1 — download.py [DONE]
Downloads Kepler light curves from NASA MAST via `lightkurve`, keeps PDCSAP_FLUX
(NASA-corrected), stitches quarters, saves to `data/raw/`.

    python pipeline/download.py --target "KIC 11442793"
    python pipeline/download.py --batch pipeline/targets.csv

**Official data note (ISRO BAH):** the problem statement's Kepler FITS + KOI labels
are hosted on NASA MAST (public, scripted here). ISRO's own archive (ISSDC / PRADAN
portal, https://pradan.issdc.gov.in) hosts AstroSat data — if the hackathon supplies
FITS files directly, drop them in `data/raw/` and continue from filter.py identically.

## Stage 2 — filter.py [DONE]  (quality control)
Drops NaN gaps, clips cosmic-ray hits (UPWARD spikes only — downward dips may be
planets!), reports gaps + noise, rejects hopeless stars (>40% missing).

    python pipeline/filter.py                    # all of data/raw/
    python pipeline/filter.py --target KIC_11442793

## Stage 3 — preprocess.py [DONE]  (detrend + normalize)
Savitzky–Golay detrending. **Golden rule enforced: window (default 48h) must be
longer than any transit (1–13h)** or the signal gets erased. scipy if available,
pure-numpy fallback otherwise. Includes `phase_fold()` helper for later stages.

    python pipeline/preprocess.py                # all *_filtered.npy
    python pipeline/preprocess.py --window-hours 48

Verified end-to-end on synthetic data: trend removed (r=+0.002), cosmics gone,
900 ppm transit recovered at 81%.

## Next
- search.py   — BLS/TLS period search (finds WHEN the dips repeat)
- features.py — 8+ engineered features (depth, duration, odd-even, secondary…)
- export.py   — final .npy/.csv datasets for the model team

## Stage 4 — search.py [DONE]  (BLS period search)
Folds the curve at thousands of trial periods, slides a box-dip template,
finds the period where dips align. numpy engine (verified: found 11.3d planet
at 11.302d, SDE 13.1) + astropy compiled fast-path when installed.

## Stage 5 — features.py [DONE]  (false-positive catchers)
odd_even_diff, secondary_ratio, v_metric, symmetry, SNR, SDE, duty cycle,
n_transits + AstroNet-style global(512)/local(128) CNN views per star.

## Stage 6 — export.py [DONE]  (dataset assembly)
dataset_tabular.csv + dataset_views.npz with 70/15/15 split BY STAR
(leakage-safe). Labels via --labels labels.csv (train only on
CONFIRMED vs FALSE POSITIVE; CANDIDATEs are inference-time unknowns).
