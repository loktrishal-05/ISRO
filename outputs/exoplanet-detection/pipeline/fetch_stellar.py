"""
fetch_stellar.py  -  download HOST STAR properties -> stellar_params.csv
=========================================================================

WHAT THIS DOES (plain English)
------------------------------
characterize.py needs to know each planet's STAR: its temperature, radius,
mass, distance from Earth, and sky position. NASA already measured all of
this for every Kepler star. This script downloads two public tables from the
NASA Exoplanet Archive and merges them into pipeline/stellar_params.csv:

  cumulative            -> kepid, planet names, RA/Dec, T_eff, R_star, M_star
  q1_q17_dr25_stellar   -> distance from Earth (parsecs)

Then characterize.py works fully offline using this file.

NEEDS INTERNET - run on your laptop/Colab:
    python pipeline/fetch_stellar.py
"""
from __future__ import annotations
import csv, json, sys, urllib.parse, urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
TAP = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"


def tap(query):
    url = TAP + "?" + urllib.parse.urlencode({"query": query, "format": "json"})
    with urllib.request.urlopen(url, timeout=180) as r:
        return json.load(r)


def main():
    print("[stellar] downloading KOI host-star table (cumulative)...")
    rows = tap("SELECT kepid, kepler_name, kepoi_name, ra, dec, "
               "koi_steff, koi_srad, koi_smass FROM cumulative")
    print(f"[stellar] {len(rows)} KOI rows")

    # one row per STAR: prefer the row that actually has a kepler_name
    stars = {}
    for r in rows:
        k = r["kepid"]
        if k not in stars or (r.get("kepler_name") and not stars[k].get("kepler_name")):
            stars[k] = r

    print("[stellar] downloading distances...")
    dist = {}          # by kepid
    dist_name = {}     # by star name (fallback source)
    try:               # source 1: Kepler DR25 stellar catalog
        for r in tap("SELECT kepid, dist FROM q1_q17_dr25_stellar WHERE dist IS NOT NULL"):
            dist[r["kepid"]] = r["dist"]
        print(f"[stellar] DR25 distances for {len(dist)} stars")
    except Exception as e:
        print(f"[stellar] DR25 distance query failed ({e}) - trying stellarhosts...")
    if not dist:
        try:           # source 2: confirmed-planet hosts table (has Gaia distances)
            for r in tap("SELECT DISTINCT hostname, sy_dist FROM stellarhosts "
                         "WHERE sy_dist IS NOT NULL"):
                dist_name[str(r["hostname"]).strip().lower()] = r["sy_dist"]
            print(f"[stellar] stellarhosts distances for {len(dist_name)} stars")
        except Exception as e:
            print(f"[stellar] WARNING: no distance source worked ({e}) - continuing without")

    out = HERE / "stellar_params.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["target", "kepid", "kepler_name", "ra", "dec",
                    "teff", "srad", "smass", "dist_pc"])
        for k, r in stars.items():
            # kepler_name is per-planet like "Kepler-10 b" -> star name "Kepler-10"
            name = (r.get("kepler_name") or "").rsplit(" ", 1)[0]
            d = dist.get(k, "") or dist_name.get(name.strip().lower(), "")
            w.writerow([f"KIC {k}", k, name, r.get("ra"), r.get("dec"),
                        r.get("koi_steff"), r.get("koi_srad"),
                        r.get("koi_smass"), d])
    print(f"[stellar] wrote {out} ({len(stars)} stars)")
    print("[stellar] NEXT: python pipeline/characterize.py")


if __name__ == "__main__":
    try:
        main()
    except urllib.error.URLError as e:
        sys.exit(f"[stellar] network error: {e} - run this on your laptop/Colab.")
