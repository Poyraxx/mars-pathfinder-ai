"""Live coarse planning grids anywhere on Mars from the MOLA/HRSC blended DEM.

Source: Mars MGS MOLA - MEX HRSC Blended DEM Global 200 m v2 (NASA/USGS/ESA/DLR/FU Berlin),
served by Esri as LERC tiles in a geographic Mars 2000 sphere tiling (512 px tiles,
origin -180/90). Tiles are fetched when a mission is created, mosaicked and resampled
bilinearly to a square-metre equirectangular grid centred on the mission. Cells are
hundreds of metres wide: slopes are averaged over that baseline and say nothing about
rocks, boulders or metre-scale hazards.
"""
from __future__ import annotations

import json
import math
import os
import threading
import httpx
from collections import OrderedDict
from pathlib import Path

import numpy as np

from .navigation import terrain_analysis

R = 3396190.0
TILE = 512
LEVEL0_RES = 0.3515625
MAX_LEVEL = 7
MAX_CELLS = 384
MAX_SPAN_KM = 2000.0
MAX_ABS_LAT = 80.0
URL = "https://astro.arcgis.com/arcgis/rest/services/OnMars/MDEM200M/ImageServer/tile/{z}/{y}/{x}"
SOURCE = "Mars MGS MOLA - MEX HRSC Blended DEM Global 200m v2 (NASA/USGS/ESA/DLR/FU Berlin; Esri LERC tiles)"

_tiles: OrderedDict = OrderedDict()
_lock = threading.Lock()
# httpx ships its own CA bundle (certifi); slim containers may lack system certificates.
_http = httpx.Client(timeout=40, follow_redirects=True, headers={"User-Agent": "MarsPathfinderResearch/1.0"})
_dll = None


def _lerc():
    global _dll
    if os.name == "nt" and _dll is None:
        config = Path(__file__).resolve().parents[1] / ".local-run.json"
        dll_dir = os.environ.get("MARS_DLL_DIR") or (json.loads(config.read_text("utf-8")).get("dll_dir") if config.exists() else None)
        if dll_dir:
            _dll = os.add_dll_directory(dll_dir)
    import lerc
    return lerc


def tile(z: int, y: int, x: int) -> np.ndarray:
    key = (z, y, x)
    with _lock:
        if key in _tiles:
            _tiles.move_to_end(key)
            return _tiles[key]
    try:
        response = _http.get(URL.format(z=z, y=y, x=x)); response.raise_for_status(); payload = response.content
    except httpx.HTTPError as exc:
        raise OSError("dem_tile_unavailable") from exc
    if len(payload) > 8_000_000:
        raise ValueError("dem_tile_too_large")
    err, arr, mask = _lerc().decode(payload)[:3]
    if err != 0:
        raise ValueError("dem_tile_decode_failed")
    arr = np.asarray(arr, dtype=np.float32)[:TILE, :TILE].copy()
    if mask is not None:
        arr[~np.asarray(mask, dtype=bool)[:TILE, :TILE]] = np.nan
    arr[np.abs(arr) > 30000] = np.nan
    with _lock:
        _tiles[key] = arr
        while len(_tiles) > 48:
            _tiles.popitem(last=False)
    return arr


def sample(lat: np.ndarray, lon: np.ndarray, z: int) -> np.ndarray:
    """Bilinear DEM sample at planetocentric lat / east lon (degrees)."""
    res = LEVEL0_RES / 2 ** z
    px = (lon + 180.0) / res - 0.5
    py = (90.0 - lat) / res - 0.5
    x0 = np.floor(px).astype(int); y0 = np.floor(py).astype(int)
    fx = px - x0; fy = py - y0
    ncols = int(round(360 / res)); nrows = int(round(180 / res))
    out = np.zeros(lat.shape, dtype=np.float64)
    for dy, dx, w in ((0, 0, (1 - fx) * (1 - fy)), (0, 1, fx * (1 - fy)), (1, 0, (1 - fx) * fy), (1, 1, fx * fy)):
        gx = (x0 + dx) % ncols; gy = np.clip(y0 + dy, 0, nrows - 1)
        tx, ty = gx // TILE, gy // TILE
        vals = np.empty(lat.shape, dtype=np.float64)
        for (a, b) in set(zip(ty.ravel().tolist(), tx.ravel().tolist())):
            sel = (ty == a) & (tx == b)
            vals[sel] = tile(z, a, b)[gy[sel] - a * TILE, gx[sel] - b * TILE]
        out += w * vals
    return out


class GlobalRegion:
    def __init__(self, points: list[dict]):
        lats = [p["lat"] for p in points]; lons = [p["lon"] for p in points]
        if max(abs(v) for v in lats) > MAX_ABS_LAT:
            raise ValueError("polar_not_supported")
        if max(lons) - min(lons) > 180:
            raise ValueError("antimeridian_not_supported")
        clat = (max(lats) + min(lats)) / 2; clon = (max(lons) + min(lons)) / 2
        coslat = math.cos(math.radians(clat))
        span_n = math.radians(max(lats) - min(lats)) * R
        span_e = math.radians(max(lons) - min(lons)) * R * coslat
        span = max(span_n, span_e)
        if span > MAX_SPAN_KM * 1000:
            raise ValueError("span_too_large")
        extent = max(span * 1.5, 4000.0)
        native = [LEVEL0_RES / 2 ** z * math.pi / 180 * R for z in range(MAX_LEVEL + 1)]
        cell = max(extent / MAX_CELLS, native[MAX_LEVEL])
        z = next(z for z in range(MAX_LEVEL + 1) if native[z] <= cell * 1.01)
        n = int(math.ceil(extent / cell))
        self.meta = {"id": "global", "name": "Global coarse terrain", "radius_m": R, "standard_parallel_deg": clat,
                     "central_lon_deg": 0, "central_lat_deg": 0, "cell_m": round(cell, 3),
                     "west_m": R * coslat * math.radians(clon) - n * cell / 2,
                     "north_m": R * math.radians(clat) + n * cell / 2, "shape": [n, n],
                     "source_id": SOURCE, "source_level": z, "source_native_m": round(native[z], 1),
                     "coverage_level": "global_coarse",
                     "limitations": f"{cell:.0f} m cells from a ~200 m DEM: regional context only, not obstacle or hazard evidence."}
        rows, cols = np.mgrid[0:n, 0:n]
        y = self.meta["north_m"] - (rows + .5) * cell
        x = self.meta["west_m"] + (cols + .5) * cell
        lat = np.degrees(y / R)
        lon = (np.degrees(x / (R * coslat)) + 180) % 360 - 180
        self.elevation = sample(lat, lon, z).astype(np.float32)
        self.analysis = terrain_analysis(self.elevation, cell)
        self.meta["bounds"] = [float(lon.min()), float(lat.min()), float(lon.max()), float(lat.max())]
        self.meta["elevation_range_m"] = [float(np.nanmin(self.elevation)), float(np.nanmax(self.elevation))]
