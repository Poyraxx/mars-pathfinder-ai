"""Convert the USGS/IAU Mars nomenclature centre points to a compact JSON catalogue.

Source: Gazetteer of Planetary Nomenclature (IAU WGPSN / USGS Astrogeology),
MARS_nomenclature_center_pts.zip. Coordinates are kept as published
(planetocentric, east-positive) and only wrapped from 0..360 to -180..180.
"""
import io
import json
import os
import urllib.request
import zipfile
from pathlib import Path

import shapefile

URL = "https://asc-planetarynames-data.s3.us-west-2.amazonaws.com/MARS_nomenclature_center_pts.zip"
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "prepared" / "global" / "nomenclature.json"


def main():
    cache = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "mars-pathfinder" / "raw" / "global" / "nomen.zip"
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(urllib.request.urlopen(urllib.request.Request(URL, headers={"User-Agent": "MarsPathfinderResearch/1.0"}), timeout=120).read())
    import hashlib
    payload = cache.read_bytes()
    z = zipfile.ZipFile(io.BytesIO(payload))
    base = "MARS_nomenclature_center_pts"
    reader = shapefile.Reader(shp=io.BytesIO(z.read(base + ".shp")), shx=io.BytesIO(z.read(base + ".shx")),
                              dbf=io.BytesIO(z.read(base + ".dbf")), encoding="utf-8", encodingErrors="replace")
    features = []
    for rec in reader.records():
        d = rec.as_dict()
        if not str(d["approval"]).startswith("Adopted"):
            continue
        lon = float(d["center_lon"]); lon = (lon + 180) % 360 - 180
        kind = str(d["type"]).split(",")[0].strip()
        features.append({"name": d["clean_name"] or d["name"], "type": kind, "diameter_km": round(float(d["diameter"]), 2),
                         "lat": round(float(d["center_lat"]), 4), "lon": round(lon, 4), "link": d["link"]})
    features.sort(key=lambda f: -f["diameter_km"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"source": "USGS Gazetteer of Planetary Nomenclature (IAU WGPSN)", "url": URL,
                               "zip_sha256": hashlib.sha256(payload).hexdigest(),
                               "coordinates": "planetocentric latitude, east-positive longitude wrapped to [-180,180)",
                               "count": len(features), "features": features}, ensure_ascii=False, separators=(",", ":")), "utf-8")
    from collections import Counter
    print(len(features), Counter(f["type"] for f in features).most_common(12))


if __name__ == "__main__":
    main()
