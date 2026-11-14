"""Extract Curiosity's actual driven path (every interpolated drive step) from MSL PLACES.

Source: localized_interp_demv2.csv of the PDS MSL PLACES bundle (already cached by prepare_pds.py).
Each ROVER row is a localized pose at one drive-counter step (~0.5 m apart while driving), with the
global NED position and the rover yaw. Between two camera acquisitions this is what the rover
really did: forward and reverse driving, arcs and turns in place. No interpolation is added here.
Run: python scripts/extract_drive_path.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "mars-pathfinder" / "raw" / "pds" / "localized_interp_demv2.csv"
SOURCE = "https://planetarydata.jpl.nasa.gov/w10n/msl/msl_places/data_localizations/localized_interp_demv2.csv"
OUT = ROOT / "data" / "prepared" / "places-drive-path.json"


def main():
    payload = CACHE.read_bytes()
    frames = json.loads((ROOT / "data" / "prepared" / "frames.json").read_text("utf-8"))
    keys = [(int(f["places"]["row"]["site"]), int(f["places"]["row"]["drive"])) for f in frames if f.get("places", {}).get("row")]
    lo, hi = min(keys), max(keys)
    rows = []
    for r in csv.DictReader(payload.decode("utf-8-sig").splitlines()):
        if r["frame"] != "ROVER":
            continue
        k = (int(r["site"]), int(r["drive"]))
        if lo <= k <= hi:
            rows.append([k[0], k[1], int(r["pose"]), round(float(r["northing"]), 3), round(float(r["easting"]), 3),
                         round(float(r["yaw"]), 3), int(r["sol"]), int(r["sclk"])])
    rows.sort(key=lambda r: (r[0], r[1], r[2]))
    OUT.write_text(json.dumps({
        "source": SOURCE, "source_sha256": hashlib.sha256(payload).hexdigest(),
        "frame": "global NED metres (same as frames.json corrected_rover_origin_global_ned_m); yaw degrees, clockwise from north",
        "columns": ["site", "drive", "pose", "northing_m", "easting_m", "yaw_deg", "sol", "sclk"],
        "range": {"from": list(lo), "to": list(hi)}, "rows": rows}, separators=(",", ":")), "utf-8")
    print(OUT, len(rows), OUT.stat().st_size)


if __name__ == "__main__":
    main()
