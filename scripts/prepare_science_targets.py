"""Sourced science stops for Marswalk planning -> data/prepared/science-targets.json.

Gale (1 m area): targets Curiosity actually investigated. Position = the rover's localized PLACES
pose on the sol of the investigation (localized_interp_demv2.csv); the target itself lies within
arm reach (~2 m) of that pose. Global: landing sites from the USGS SIM 3292 landing-site layer.
"What to measure" and work minutes are planning assumptions for a crewed EVA, stated as such.
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import shapefile

ROOT = Path(__file__).resolve().parents[1]
RAW = Path(os.environ["LOCALAPPDATA"]) / "mars-pathfinder/raw"
PLACES = RAW / "pds/localized_interp_demv2.csv"
SITES = RAW / "usgs/ex/SIM3292_MarsGlobalGeologicGIS_20M/SIM3292_Shapefiles/SIM3292_Global_Landing_sites"

GALE = [
    {"id": "jake-matijevic", "name": "Jake Matijevic rock", "sol": 46,
     "why": "First rock Curiosity examined with its contact instruments; its alkaline, feldspar-rich composition was unlike any igneous rock seen before on Mars, showing that Gale's source magmas were more varied than expected.",
     "measure": "Close-up imaging and handheld X-ray fluorescence of a cleaned surface; collect a hand sample.",
     "work_min": 30,
     "sources": [{"label": "Stolper et al. 2013, Science 341, 'The Petrochemistry of Jake_M'", "url": "https://doi.org/10.1126/science.1239463"},
                 {"label": "NASA JPL: Curiosity touches its first rock", "url": "https://www.jpl.nasa.gov/news/nasa-mars-rover-preparing-to-bounce-laser-off-rock"}]},
    {"id": "bathurst-inlet", "name": "Bathurst Inlet", "sol": 54,
     "why": "Dark, fine-grained rock checked with the hand lens imager and APXS on the approach to Glenelg; a reference for the basaltic rocks around Bradbury Landing.",
     "measure": "Hand-lens imaging of grain size and layering; X-ray fluorescence for composition.",
     "work_min": 20,
     "sources": [{"label": "NASA: Curiosity Mars rover checks odd-looking rock (sol 54)", "url": "https://www.nasa.gov/mission_pages/msl/news/msl20121001.html"}]},
    {"id": "rocknest", "name": "Rocknest sand shadow", "sol": 61,
     "why": "Where Curiosity first scooped soil (sols 61-100). CheMin returned the first X-ray diffraction of Martian soil: basaltic minerals plus a large amorphous, water-bearing component.",
     "measure": "Scoop and sieve windblown sand; bag samples for mineralogy and volatile analysis.",
     "work_min": 45,
     "sources": [{"label": "Bish et al. 2013, Science 341, X-ray diffraction results from Rocknest", "url": "https://doi.org/10.1126/science.1238932"},
                 {"label": "NASA: Rover's first soil studies help fingerprint Martian minerals", "url": "https://www.jpl.nasa.gov/news/nasa-rovers-first-soil-studies-help-fingerprint-martian-minerals"}]},
]

SITE_SCIENCE = {
    "Viking 1": "First successful U.S. landing (1976); Chryse Planitia outflow-channel plains and the Viking biology experiments.",
    "Viking 2": "Utopia Planitia plains; observed seasonal surface frost.",
    "Pathfinder": "Ares Vallis flood deposits; first Mars rover (Sojourner) measured rock chemistry.",
    "MER A": "Spirit, Gusev crater: hydrothermal silica and carbonate-bearing outcrops in the Columbia Hills.",
    "MER B": "Opportunity, Meridiani Planum: hematite concretions and sulfate sandstones recording ancient water.",
    "PHX": "Phoenix: dug into ground ice and measured perchlorate in the soil at high northern latitude.",
    "MSL": "Curiosity, Gale crater: ancient lake deposits and the layered foothills of Aeolis Mons.",
}


def pose_on(sol):
    rows = [r for r in csv.DictReader(PLACES.open(encoding="utf-8-sig")) if r["frame"] == "ROVER" and r["sol"] and int(r["sol"]) <= sol]
    last = max(rows, key=lambda r: (int(r["sol"]), int(r["site"]), int(r["drive"])))
    return float(last["planetocentric_latitude"]), float(last["longitude"]), last


def main():
    out = {"source_note": "Gale positions: MSL PLACES localized rover pose on the investigation sol (target within ~2 m). "
                          "Global: USGS SIM 3292 landing-site layer. Work minutes and measurements are crewed-EVA planning assumptions.",
           "gale": [], "global": []}
    for t in GALE:
        lat, lon, row = pose_on(t["sol"])
        out["gale"].append({**t, "lat": round(lat, 7), "lon": round(lon, 7), "position_source": f"PLACES site {row['site']} drive {row['drive']} (sol {row['sol']})"})
    reader = shapefile.Reader(str(SITES))
    for rec, shape in zip(reader.records(), reader.shapes()):
        r = rec.as_dict()
        if r["NAME"] not in SITE_SCIENCE:
            continue
        lon, lat = shape.points[0]
        out["global"].append({"id": r["NAME"].lower().replace(" ", "-"), "name": r["FULL_NAME"], "lat": round(lat, 4), "lon": round(lon, 4),
                              "why": SITE_SCIENCE[r["NAME"]], "measure": "Context imaging, regolith sample and comparison with the mission's published results.",
                              "work_min": 30, "position_source": "USGS SIM 3292 landing-site layer",
                              "sources": [{"label": f"{r['FULL_NAME']} (NASA)", "url": r["WEB_LINK"]}]})
    (ROOT / "data/prepared/science-targets.json").write_text(json.dumps(out, indent=1), "utf-8")
    print(len(out["gale"]), "gale", len(out["global"]), "global")
    for t in out["gale"]:
        print(t["name"], t["lat"], t["lon"], t["position_source"])


if __name__ == "__main__":
    main()
