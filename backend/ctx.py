"""Global orbital imagery windows from the Murray Lab CTX beta01 mosaic (~5 m/px).

Tiles: Esri-hosted geographic Mars 2000 sphere tiling, 512 px PNG, origin -180/90,
level 12 = 5.09 m/px. The mosaic is preliminary and uncontrolled (Caltech Murray Lab).
"""
from __future__ import annotations

import io
import math
import threading
import httpx
from collections import OrderedDict

import numpy as np
from PIL import Image

URL = "https://astro.arcgis.com/arcgis/rest/services/OnMars/CTX/MapServer/tile/{z}/{y}/{x}"
SOURCE = "Murray Lab CTX global mosaic beta01 (Caltech; NASA/JPL/MSSS), Esri tiles"
R = 3396190.0
TILE = 512
LEVEL0_RES = 0.3515625 * 2  # degrees per pixel at level 0 (2 x 1 tiles of 512 px over 360 x 180)
MAX_LEVEL = 12
_cache: OrderedDict = OrderedDict()
_lock = threading.Lock()
# httpx ships its own CA bundle (certifi); slim containers may lack system certificates.
_http = httpx.Client(timeout=40, follow_redirects=True, headers={"User-Agent": "MarsPathfinderResearch/1.0"})


def _res(z: int) -> float:
    return 180.0 / TILE / 2 ** z


def tile(z: int, y: int, x: int) -> np.ndarray | None:
    key = (z, y, x)
    with _lock:
        if key in _cache:
            _cache.move_to_end(key); return _cache[key]
    try:
        response = _http.get(URL.format(z=z, y=y, x=x)); response.raise_for_status(); payload = response.content
        im = Image.open(io.BytesIO(payload))
        arr = np.asarray(im.convert("LA"), dtype=np.uint8)
        grey = arr[..., 0].astype(np.float32)
        grey[arr[..., 1] == 0] = np.nan
    except Exception:
        grey = None
    with _lock:
        _cache[key] = grey
        while len(_cache) > 96:
            _cache.popitem(last=False)
    return grey


def window(lat: float, lon: float, size_m: float, out_px: int = 224) -> tuple[np.ndarray | None, dict]:
    """Square window centred at lat/lon, size_m on a side, resampled to out_px."""
    target = size_m / out_px
    z = MAX_LEVEL
    for level in range(MAX_LEVEL + 1):
        if _res(level) * math.pi / 180 * R <= target * 1.5:
            z = level; break
    res = _res(z)
    half_lat = math.degrees(size_m / 2 / R)
    half_lon = half_lat / max(math.cos(math.radians(lat)), 0.05)
    px0 = (lon - half_lon + 180) / res; px1 = (lon + half_lon + 180) / res
    py0 = (90 - lat - half_lat) / res; py1 = (90 - lat + half_lat) / res
    cols = np.linspace(px0, px1, out_px); rows = np.linspace(py0, py1, out_px)
    gx, gy = np.meshgrid(cols, rows)
    ncols = int(round(360 / res))
    ix = np.floor(gx).astype(int) % ncols; iy = np.clip(np.floor(gy).astype(int), 0, int(round(180 / res)) - 1)
    out = np.full(gx.shape, np.nan, dtype=np.float32)
    for ty, tx in {(a, b) for a, b in zip((iy // TILE).ravel().tolist(), (ix // TILE).ravel().tolist())}:
        t = tile(z, ty, tx)
        if t is None:
            continue
        sel = (iy // TILE == ty) & (ix // TILE == tx)
        out[sel] = t[iy[sel] - ty * TILE, ix[sel] - tx * TILE]
    coverage = float(np.isfinite(out).mean())
    info = {"level": z, "native_m_per_px": round(res * math.pi / 180 * R, 2), "coverage": round(coverage, 3)}
    if coverage < 0.9:
        return None, info
    out = np.nan_to_num(out, nan=float(np.nanmean(out)))
    lo, hi = np.percentile(out, [1, 99])
    out = np.clip((out - lo) / max(hi - lo, 1e-3), 0, 1)
    return out, info
