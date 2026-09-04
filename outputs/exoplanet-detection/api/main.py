"""
api/main.py  -  FastAPI backend for the exoplanet-detection pipeline
====================================================================

WHAT THIS IS (plain English)
----------------------------
A small web server that lets the preview dashboard (or curl) drive the real
pipeline instead of demo data:

  POST /analyze/{target}   start the full pipeline for one star
                           (download -> filter -> detrend -> BLS search ->
                           features -> characterize -> model verdict).
                           Analysis takes minutes (the BLS search alone can be
                           5+ min), so this returns immediately with a job
                           status you can poll.
  GET  /results/{target}   everything we know about a star: job status,
                           the 13 physics features, the characterization
                           (radius / habitability / distance), and the model's
                           calibrated verdict if a trained model exists.
  GET  /health             liveness probe + model availability.

RUN IT
------
    py -m uvicorn api.main:app --reload --port 8000
    # then e.g.:
    #   curl -X POST http://127.0.0.1:8000/analyze/Kepler-10
    #   curl http://127.0.0.1:8000/results/Kepler-10

HONESTY NOTES (same rules as CLAUDE.md)
---------------------------------------
- If no trained model exists in models/saved/, /results says so plainly
  instead of inventing a score.
- Habitability fields are indicators, never claims.
"""
from __future__ import annotations

import json
import sys
import threading
import traceback
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT / "models"))

RAW = ROOT / "data" / "raw"
CLEAN = ROOT / "data" / "cleaned"
FEAT = ROOT / "data" / "features"
SAVED = ROOT / "models" / "saved"

app = FastAPI(
    title="Exoplanet Detection API",
    description="Wraps the 7-stage Kepler pipeline and the calibrated classifier.",
    version="0.1.0",
)
app.add_middleware(  # the preview page is opened from file:// or another port
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


def _safe_filename(target: str) -> str:
    """'KIC 11904151' -> 'KIC_11904151' (same rule as download.py)."""
    return "".join(c if c.isalnum() else "_" for c in target).strip("_")


# ----------------------------------------------------------------------
# Job registry: one background thread per target, status pollable.
# In-memory on purpose - restart the server and you can simply re-POST;
# every stage is resume-safe (existing files are reused).
# ----------------------------------------------------------------------
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


def _run_pipeline(target: str, stem: str, redo: bool) -> None:
    job = _jobs[stem]

    def stage(name: str) -> None:
        job["stage"] = name

    try:
        if not (FEAT / f"{stem}_features.json").exists() or redo:
            if not (RAW / f"{stem}.npy").exists() or redo:
                stage("download")
                import lightkurve as lk
                import download as dl
                sr = lk.search_lightcurve(target, mission="Kepler",
                                          author="Kepler", cadence="long")
                if len(sr) == 0:
                    raise RuntimeError(f"no MAST results for {target!r}")
                if len(sr) > 8:
                    sr = sr[:8]
                col = sr.download_all(flux_column="pdcsap_flux")
                if col is None or len(col) == 0:
                    raise RuntimeError("download returned nothing")
                dl.save_light_curve(col.stitch(), target)

            import filter as flt
            import preprocess as pre
            import search as srch
            import features as feat

            stage("filter")
            if not flt.process_file(RAW / f"{stem}.npy"):
                raise RuntimeError("filter stage failed")
            stage("preprocess")
            if not pre.process_file(CLEAN / f"{stem}_filtered.npy", 48.0):
                raise RuntimeError("preprocess stage failed")
            stage("search")  # the slow one: minutes, not seconds
            if not srch.process_file(CLEAN / f"{stem}_detrended.npy", 0.5, None, 10.0):
                raise RuntimeError("search stage failed")
            stage("features")
            if not feat.process_target(stem):
                raise RuntimeError("features stage failed")

        stage("characterize")
        import characterize as ch
        table = ch.load_stellar_table()
        ch.process_target(FEAT / f"{stem}_features.json", table, {})

        job["stage"] = "done"
        job["status"] = "complete"
    except Exception as exc:  # keep the traceback for /results, don't kill the server
        job["status"] = "failed"
        job["error"] = f"{type(exc).__name__}: {exc}"
        job["traceback"] = traceback.format_exc(limit=5)


def _model_verdict(stem: str):
    """Calibrated model score for an analyzed star, or an honest 'no model'."""
    feats_path = FEAT / f"{stem}_features.json"
    if not feats_path.exists():
        return None
    if not (SAVED / "eval_report.json").exists():
        return {"available": False,
                "note": "No trained model in models/saved/ yet - "
                        "run models/xgboost_model.py first."}
    import xgboost_model as xm
    verdict = xm.predict_one(json.loads(feats_path.read_text()))
    verdict["available"] = True
    return verdict


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_trained": (SAVED / "eval_report.json").exists(),
        "stars_analyzed": len(list(FEAT.glob("*_features.json"))),
    }


@app.post("/analyze/{target}", status_code=202)
def analyze(target: str, redo: bool = False):
    """Kick off the full pipeline for one star; poll /results/{target}."""
    stem = _safe_filename(target)
    with _jobs_lock:
        job = _jobs.get(stem)
        if job and job["status"] == "running":
            return {"target": target, "status": "running", "stage": job["stage"],
                    "note": "analysis already in progress"}
        _jobs[stem] = {"target": target, "status": "running", "stage": "queued"}
    t = threading.Thread(target=_run_pipeline, args=(target, stem, redo), daemon=True)
    t.start()
    return {"target": target, "status": "running", "stage": "queued",
            "note": "poll GET /results/{target} - the BLS search step can take minutes"}


@app.get("/results/{target}")
def results(target: str):
    stem = _safe_filename(target)
    job = _jobs.get(stem)
    feats_path = FEAT / f"{stem}_features.json"
    char_path = FEAT / f"{stem}_characterization.json"

    if not feats_path.exists() and job is None:
        raise HTTPException(404, f"{target!r} has not been analyzed - POST /analyze/{target} first")

    out = {"target": target, "stem": stem}
    if job:
        out["job"] = {k: job[k] for k in ("status", "stage", "error") if k in job}
    if feats_path.exists():
        out["features"] = json.loads(feats_path.read_text())
    if char_path.exists():
        out["characterization"] = json.loads(char_path.read_text())
        out["habitability_note"] = ("'Potentially habitable' is an indicator from "
                                    "size and orbit only - transits cannot see an "
                                    "atmosphere, so it is never a claim of life.")
    out["model"] = _model_verdict(stem)
    return out
