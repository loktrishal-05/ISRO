"""
characterize.py  -  Stage 5b: PLANET CHARACTERIZATION (the "so what?" stage)
=============================================================================

WHAT THIS DOES (plain English)
------------------------------
Detection says "there's a planet." This stage answers the questions humans
actually ask next - using only (a) our measured transit numbers and
(b) the host star's properties from NASA's stellar catalog:

  HOW BIG?    Rp = R_star * sqrt(transit depth)      -> in Earth radii,
              compared to Mars / Earth / Jupiter
  HOW FAR FROM ITS STAR?  Kepler's 3rd law: a = (M_star * P_years^2)^(1/3)
  HOW HOT?    equilibrium temperature + stellar energy received (insolation)
  HABITABLE?  is it inside the star's Habitable Zone (Kopparapu limits)
              AND small enough to be rocky?  (honest indicator, not a claim -
              transits give radius, not atmosphere)
  HOW FAR FROM EARTH?  catalog distance (parsecs -> light-years)
  WHERE IN THE SKY?    RA/Dec -> constellation + galactic coordinates
                       (needs astropy; skipped gracefully if missing)

Stellar params come from pipeline/stellar_params.csv (fetch_stellar.py) or
--teff/--srad/--smass/--dist flags. No catalog match -> assumes a Sun-like
star and SAYS SO in the output (honesty flag).

USAGE
-----
    python pipeline/characterize.py                     # all *_features.json
    python pipeline/characterize.py --target Kepler_10
    python pipeline/characterize.py --target Kepler_10 --teff 5708 --srad 1.065 --smass 0.913 --dist 173
    python pipeline/characterize.py --selftest          # offline physics proof
Input : data/features/<star>_features.json  (+ pipeline/stellar_params.csv)
Output: data/features/<star>_characterization.json
"""
from __future__ import annotations
import argparse, csv, json, math, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FEAT_DIR = HERE.parent / "data" / "features"
STELLAR_CSV = HERE / "stellar_params.csv"

# ---- constants ----
RSUN_IN_REARTH = 109.1        # how many Earths fit across the Sun
TSUN = 5772.0                 # Sun's effective temperature, K
PC_TO_LY = 3.26156
R_MARS, R_EARTH, R_JUP = 0.532, 1.0, 11.209   # Earth radii
ALBEDO = 0.3                  # Earth-like reflectivity assumption for Teq

# Habitable-zone insolation limits (Kopparapu et al. 2013), in Earth units:
HZ_CONSERVATIVE = (0.356, 1.107)   # max greenhouse .. runaway greenhouse
HZ_OPTIMISTIC   = (0.320, 1.776)   # early Mars     .. recent Venus


def size_class(rp):
    if rp < 0.8:  return "sub-Earth"
    if rp < 1.25: return "Earth-sized"
    if rp < 2.0:  return "super-Earth"
    if rp < 4.0:  return "sub-Neptune"
    if rp < 7.0:  return "Neptune-like"
    return "Jupiter-like gas giant"


def characterize(period_d, depth_ppm, teff=TSUN, srad=1.0, smass=1.0,
                 dist_pc=None, ra=None, dec=None, assumed=False):
    """All the physics in one place. Returns a plain dict (JSON-able)."""
    depth = max(depth_ppm, 0.0) * 1e-6

    # --- size ---
    rp = srad * math.sqrt(depth) * RSUN_IN_REARTH          # Earth radii
    # --- orbit (Kepler's 3rd law, solar units -> AU) ---
    a_au = (smass * (period_d / 365.25) ** 2) ** (1.0 / 3.0)
    # --- energy from the star ---
    lum = (srad ** 2) * (teff / TSUN) ** 4                 # L / L_sun
    insol = lum / (a_au ** 2) if a_au > 0 else float("inf")  # S / S_earth
    rstar_au = srad * 0.00465047
    teq = teff * math.sqrt(rstar_au / (2 * a_au)) * (1 - ALBEDO) ** 0.25

    # --- habitable-zone verdict (indicator, honestly framed) ---
    in_hz_cons = HZ_CONSERVATIVE[0] <= insol <= HZ_CONSERVATIVE[1]
    in_hz_opt  = HZ_OPTIMISTIC[0]  <= insol <= HZ_OPTIMISTIC[1]
    rocky = rp < 1.8
    if insol > HZ_OPTIMISTIC[1]:
        zone = "too hot (closer to its star than the habitable zone)"
    elif insol < HZ_OPTIMISTIC[0]:
        zone = "too cold (beyond the habitable zone)"
    elif in_hz_cons:
        zone = "inside the CONSERVATIVE habitable zone"
    else:
        zone = "inside the OPTIMISTIC habitable zone"
    habitable = (in_hz_cons or in_hz_opt) and rocky
    why = []
    why.append(f"receives {insol:.2f}x Earth's stellar energy -> {zone}")
    why.append(f"radius {rp:.2f} R_earth -> " +
               ("small enough to be rocky" if rocky else
                "likely gas/volatile-rich (not rocky)"))

    out = {
        "planet_radius_rearth": round(rp, 3),
        "size_class": size_class(rp),
        "vs_mars":    round(rp / R_MARS, 2),
        "vs_earth":   round(rp / R_EARTH, 2),
        "vs_jupiter": round(rp / R_JUP, 3),
        "semi_major_axis_au": round(a_au, 5),
        "orbital_period_d": round(period_d, 5),
        "equilibrium_temp_K": round(teq, 1),
        "insolation_earths": round(insol, 3),
        "habitable_zone": zone,
        "potentially_habitable": bool(habitable),
        "habitability_reasoning": why,
        "star": {"teff_K": round(teff, 0), "radius_rsun": round(srad, 3),
                 "mass_msun": round(smass, 3),
                 "params_source": "ASSUMED SUN-LIKE (no catalog match)" if assumed
                                  else "stellar catalog"},
    }
    # --- distance & sky position (only if catalog gave them) ---
    if dist_pc:
        out["distance_pc"] = round(dist_pc, 1)
        out["distance_ly"] = round(dist_pc * PC_TO_LY, 1)
    if ra is not None and dec is not None:
        out["ra_deg"], out["dec_deg"] = round(ra, 5), round(dec, 5)
        try:                                    # astropy = bonus, not required
            from astropy.coordinates import SkyCoord, get_constellation
            import astropy.units as u
            c = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
            out["constellation"] = get_constellation(c)
            g = c.galactic
            out["galactic_coords"] = {"l_deg": round(float(g.l.deg), 3),
                                      "b_deg": round(float(g.b.deg), 3)}
        except ImportError:
            pass
    return out


# ----------------------------------------------------------------------
# stellar catalog lookup
# ----------------------------------------------------------------------
def _norm(s):
    return "".join(str(s).lower().split()).replace("_", "").replace("-", "")


def load_stellar_table():
    if not STELLAR_CSV.exists():
        return []
    with open(STELLAR_CSV, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def match_star(target, table):
    """Match 'Kepler-10' / 'KIC 11904151' etc. against the catalog rows.
    EXACT matches always win - otherwise 'Kepler-10' would grab 'Kepler-100'
    (whichever appears first). Prefix match is only a last resort."""
    tn = _norm(target)
    prefix_hit = None
    for r in table:
        cands = [r.get("target", ""), r.get("kepler_name", ""),
                 "kic" + str(r.get("kepid", ""))]
        for c in cands:
            cn = _norm(c)
            if not cn:
                continue
            if cn == tn:                      # exact -> return immediately
                return r
            if prefix_hit is None and (cn.startswith(tn) or tn.startswith(cn)):
                prefix_hit = r                # remember, but keep looking
    return prefix_hit


def _f(row, key):
    try:
        return float(row.get(key))
    except (TypeError, ValueError):
        return None


def process_target(path: Path, table, overrides) -> bool:
    feats = json.loads(path.read_text())
    target = feats.get("target", path.stem.replace("_features", ""))
    row = match_star(target, table) if table else None
    teff = overrides.get("teff") or (row and _f(row, "teff")) or TSUN
    srad = overrides.get("srad") or (row and _f(row, "srad")) or 1.0
    smass = overrides.get("smass") or (row and _f(row, "smass")) or 1.0
    dist = overrides.get("dist") or (row and _f(row, "dist_pc"))
    ra   = row and _f(row, "ra")
    dec  = row and _f(row, "dec")
    assumed = row is None and not overrides.get("teff")

    result = characterize(feats["period_d"], feats["depth_ppm"], teff, srad,
                          smass, dist, ra, dec, assumed)
    result["target"] = target
    out = path.parent / path.name.replace("_features.json", "_characterization.json")
    out.write_text(json.dumps(result, indent=2))
    hab = "POTENTIALLY HABITABLE" if result["potentially_habitable"] else "not habitable"
    d = f" | {result.get('distance_ly','?')} ly away" if "distance_ly" in result else ""
    print(f"[characterize] {target}: Rp={result['planet_radius_rearth']} R_e "
          f"({result['size_class']}) | Teq={result['equilibrium_temp_K']}K "
          f"| S={result['insolation_earths']}x Earth | {hab}{d} -> {out.name}")
    return True


# ----------------------------------------------------------------------
# offline selftest: three physics sanity cases with known answers
# ----------------------------------------------------------------------
def selftest():
    checks = []
    # 1. Earth twin around the Sun -> ~1 R_e, ~1 AU, S~1, HABITABLE
    e = characterize(365.25, (R_EARTH / RSUN_IN_REARTH) ** 2 * 1e6)
    checks.append(("Earth twin radius ~1 R_e", abs(e["planet_radius_rearth"] - 1) < 0.05, e["planet_radius_rearth"]))
    checks.append(("Earth twin orbit ~1 AU", abs(e["semi_major_axis_au"] - 1) < 0.01, e["semi_major_axis_au"]))
    checks.append(("Earth twin insolation ~1", abs(e["insolation_earths"] - 1) < 0.05, e["insolation_earths"]))
    checks.append(("Earth twin HABITABLE", e["potentially_habitable"], e["habitable_zone"]))
    checks.append(("Earth twin Teq ~255K", abs(e["equilibrium_temp_K"] - 255) < 12, e["equilibrium_temp_K"]))
    # 2. Kepler-10b (lava world): 0.837d, 187ppm, real star params
    k = characterize(0.8375, 187, teff=5708, srad=1.065, smass=0.913, dist_pc=173)
    checks.append(("Kepler-10b radius 1.4-1.7 R_e", 1.4 < k["planet_radius_rearth"] < 1.7, k["planet_radius_rearth"]))
    checks.append(("Kepler-10b orbit ~0.017 AU", abs(k["semi_major_axis_au"] - 0.0168) < 0.002, k["semi_major_axis_au"]))
    checks.append(("Kepler-10b NOT habitable (lava)", not k["potentially_habitable"] and k["equilibrium_temp_K"] > 1500, k["equilibrium_temp_K"]))
    checks.append(("Kepler-10 distance ~564 ly", abs(k["distance_ly"] - 564) < 10, k["distance_ly"]))
    # 3. Jupiter analog in the HZ -> right size class, NOT habitable (gas giant)
    j = characterize(4332.6, (R_JUP / RSUN_IN_REARTH) ** 2 * 1e6)
    checks.append(("Jupiter analog ~11.2 R_e", abs(j["planet_radius_rearth"] - 11.2) < 0.3, j["planet_radius_rearth"]))
    checks.append(("Jupiter analog class = gas giant", "Jupiter" in j["size_class"], j["size_class"]))
    checks.append(("gas giant never 'habitable'", not j["potentially_habitable"], j["vs_jupiter"]))

    n = 0
    for name, ok, val in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}  ({val})")
        n += ok
    print(f"\n[selftest] {n}/{len(checks)} checks passed")
    return n == len(checks)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Stage 5b: physical characterization + habitability")
    ap.add_argument("--target", help="one star (stem before _features.json)")
    ap.add_argument("--teff", type=float, help="star temperature K (override)")
    ap.add_argument("--srad", type=float, help="star radius in Suns (override)")
    ap.add_argument("--smass", type=float, help="star mass in Suns (override)")
    ap.add_argument("--dist", type=float, help="distance in parsecs (override)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    files = ([FEAT_DIR / f"{a.target}_features.json"] if a.target
             else sorted(FEAT_DIR.glob("*_features.json")))
    files = [f for f in files if f.exists()]
    if not files:
        sys.exit(f"No *_features.json in {FEAT_DIR} - run features.py first.")
    table = load_stellar_table()
    if not table:
        print("[characterize] note: no stellar_params.csv - run fetch_stellar.py "
              "(internet) or pass --teff/--srad/--smass/--dist for real star values.")
    ov = {"teff": a.teff, "srad": a.srad, "smass": a.smass, "dist": a.dist}
    done = sum(process_target(f, table, ov) for f in files)
    print(f"[characterize] finished: {done}/{len(files)} characterized.")


if __name__ == "__main__":
    main()
