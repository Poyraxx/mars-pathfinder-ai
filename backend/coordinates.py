"""Mars spherical equirectangular grids, with metadata-defined reference radius.

Heights remain areoid elevations; these routines perform horizontal transforms only.
"""
import math
import numpy as np


def wrap_lon(lon: float) -> float:
    return (lon + 180) % 360 - 180


def project(lat: float, lon: float, meta: dict) -> tuple[float, float]:
    r = meta["radius_m"]
    lat_ts = math.radians(meta.get("standard_parallel_deg", 0))
    x = r * math.cos(lat_ts) * math.radians(wrap_lon(lon - meta.get("central_lon_deg", 0)))
    y = r * math.radians(lat - meta.get("central_lat_deg", 0))
    return x, y


def unproject(x: float, y: float, meta: dict) -> dict:
    r = meta["radius_m"]
    return {"lat": math.degrees(y / r) + meta.get("central_lat_deg", 0),
            "lon": wrap_lon(math.degrees(x / (r * math.cos(math.radians(meta.get("standard_parallel_deg", 0)))))
                            + meta.get("central_lon_deg", 0))}


def point_to_cell(point: dict, meta: dict) -> tuple[int, int]:
    x, y = project(point["lat"], point["lon"], meta)
    return math.floor((meta["north_m"] - y) / meta["cell_m"]), math.floor((x - meta["west_m"]) / meta["cell_m"])


def cell_to_point(row: int, col: int, meta: dict) -> dict:
    return unproject(meta["west_m"] + (col + .5) * meta["cell_m"],
                     meta["north_m"] - (row + .5) * meta["cell_m"], meta)


def planetographic_to_centric(lat_deg: float, a: float, b: float) -> float:
    return math.degrees(math.atan2(b * b * math.sin(math.radians(lat_deg)),
                                   a * a * math.cos(math.radians(lat_deg))))


def site_ned_to_enu(points: np.ndarray) -> np.ndarray:
    return np.asarray(points)[..., [1, 0, 2]] * np.array([1, 1, -1])
