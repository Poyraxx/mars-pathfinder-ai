"""Rasterize the USGS Geologic Map of Mars (Tanaka et al. 2014, SIM 3292) for route queries and display.

Input: SIM3292_Global_Geology.shp (Mars 2000 sphere, planetocentric degrees), from
https://pubs.usgs.gov/sim/3292/downloads/sim3292_database.zip (cached under %LOCALAPPDATA%).
Output (data/prepared/geology/):
  units.png    8-bit unit index per 1/8 degree cell (2880 x 1440, row 0 = 90N, col 0 = 180W); 0 = no unit
  overlay.png  same grid coloured by unit age group, for map display
  legend.json  index -> unit code, description, period, colour, and source citation
The map is 1:20,000,000 scale; contacts are generalised at that scale (several km).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import shapefile
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(os.environ["LOCALAPPDATA"]) / "mars-pathfinder/raw/usgs/ex/SIM3292_MarsGlobalGeologicGIS_20M/SIM3292_Shapefiles/SIM3292_Global_Geology"
OUT = ROOT / "data/prepared/geology"
PPD = 8
W, H = 360 * PPD, 180 * PPD

# Period from the unit code prefix (Tanaka et al. 2014 naming: e/m/l = early/middle/late, A/H/N = Amazonian/Hesperian/Noachian).
PERIODS = [("lA", "Late Amazonian", (99, 165, 222)), ("mA", "Middle Amazonian", (122, 186, 230)), ("eA", "Early Amazonian", (150, 206, 236)),
           ("AH", "Amazonian–Hesperian", (156, 205, 168)), ("A", "Amazonian", (130, 190, 230)),
           ("lH", "Late Hesperian", (125, 189, 110)), ("eH", "Early Hesperian", (163, 207, 116)), ("HN", "Hesperian–Noachian", (214, 196, 120)),
           ("H", "Hesperian", (145, 199, 112)), ("lN", "Late Noachian", (222, 161, 95)), ("mN", "Middle Noachian", (204, 128, 82)),
           ("eN", "Early Noachian", (176, 104, 74)), ("N", "Noachian", (196, 130, 86))]


def period(code: str):
    for prefix, name, colour in PERIODS:
        if code.startswith(prefix):
            return name, colour
    return "Undivided", (170, 170, 170)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    reader = shapefile.Reader(str(SRC))
    units = sorted({rec["Unit"] for rec in reader.records()})
    index = {u: i + 1 for i, u in enumerate(units)}
    desc = {}
    canvas = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(canvas)
    # Larger polygons first so enclosed small units are drawn on top.
    items = sorted(zip(reader.records(), reader.shapes()), key=lambda rs: -rs[0]["SphArea_km"])
    for rec, shape in items:
        desc[rec["Unit"]] = rec["UnitDesc"]
        parts = list(shape.parts) + [len(shape.points)]
        for a, b in zip(parts[:-1], parts[1:]):
            ring = [((x + 180) * PPD, (90 - y) * PPD) for x, y in shape.points[a:b]]
            if len(ring) >= 3:
                draw.polygon(ring, fill=index[rec["Unit"]])
    grid = np.asarray(canvas)
    Image.fromarray(grid).save(OUT / "units.png", optimize=True)
    lut = np.zeros((256, 4), np.uint8)
    legend = []
    for u, i in index.items():
        name, colour = period(u)
        lut[i] = (*colour, 255)
        legend.append({"index": i, "unit": u, "description": desc[u], "period": name, "colour": "#%02x%02x%02x" % colour,
                       "area_fraction": round(float((grid == i).mean()), 5)})
    Image.fromarray(lut[grid], "RGBA").save(OUT / "overlay.png", optimize=True)
    (OUT / "legend.json").write_text(json.dumps({
        "source": "Tanaka, K.L., et al., 2014, Geologic map of Mars: USGS Scientific Investigations Map 3292, scale 1:20,000,000",
        "url": "https://pubs.usgs.gov/sim/3292/", "grid": {"ppd": PPD, "width": W, "height": H, "origin": "row 0 = 90N, col 0 = 180W"},
        "note": "Rasterised at 1/8 degree (~7.4 km). Contacts are generalised for 1:20M scale; a unit is regional context, not the rock under a single footstep.",
        "periods": [{"period": n, "colour": "#%02x%02x%02x" % c} for _, n, c in PERIODS],
        "units": legend}, indent=1), "utf-8")
    print(len(units), "units", (OUT / "units.png").stat().st_size, (OUT / "overlay.png").stat().st_size)


if __name__ == "__main__":
    main()
