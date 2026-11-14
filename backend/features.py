"""Relate routes to IAU-named Mars features (USGS Gazetteer centres and diameters).

A feature is modelled as a circle of its published diameter around its published centre.
This is a coarse footprint: real outlines (e.g. Valles Marineris) are not circular, so
'crosses' and 'avoid' use the same approximation and are labelled as such.
"""
from __future__ import annotations

import json
import math
from functools import lru_cache

import numpy as np

from .data import DATA_DIR

R = 3396190.0


@lru_cache(maxsize=1)
def catalogue() -> list[dict]:
    path = DATA_DIR / "global" / "nomenclature.json"
    return json.loads(path.read_text("utf-8"))["features"] if path.exists() else []


def _distance_m(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(lon2 - lon1)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def find(name: str) -> dict | None:
    key = name.strip().lower()
    return next((f for f in catalogue() if f["name"].lower() == key), None)


def along_route(points: list[dict], limit: int = 12) -> list[dict]:
    """Named features the route crosses or passes within half a diameter (min 2 km) of."""
    if not points:
        return []
    lat = np.array([p["lat"] for p in points]); lon = np.array([p["lon"] for p in points])
    step = _distance_m(lat[:-1], lon[:-1], lat[1:], lon[1:]) if len(lat) > 1 else np.zeros(0)
    along = np.r_[0, np.cumsum(step)]
    pad = max(5000.0, float(along[-1]) * .5)
    lat0, lat1 = lat.min() - math.degrees(pad / R) - 15, lat.max() + math.degrees(pad / R) + 15
    out = []
    for f in catalogue():
        if f["diameter_km"] < 1 or not (lat0 <= f["lat"] <= lat1):
            continue
        radius = f["diameter_km"] * 500
        d = _distance_m(lat, lon, f["lat"], f["lon"])
        i = int(d.argmin())
        if d[i] <= radius:
            relation = "crosses"
        elif d[i] <= radius + max(2000.0, radius * .5):
            relation = "passes near"
        else:
            continue
        out.append({"name": f["name"], "type": f["type"], "diameter_km": f["diameter_km"], "relation": relation,
                    "closest_km": round(float(max(d[i] - radius, 0)) / 1000, 1), "centre_distance_km": round(float(d[i]) / 1000, 1),
                    "along_km": round(float(along[i]) / 1000, 2)})
    out.sort(key=lambda x: (x["relation"] != "crosses", x["along_km"]))
    return out[:limit]


def avoid_mask(meta: dict, shape: tuple, names: list[str]) -> tuple[np.ndarray, list[dict]]:
    rows, cols = np.mgrid[0:shape[0], 0:shape[1]]
    cell = meta["cell_m"]
    x = meta["west_m"] + (cols + .5) * cell
    y = meta["north_m"] - (rows + .5) * cell
    coslat = math.cos(math.radians(meta.get("standard_parallel_deg", 0)))
    lat = np.degrees(y / meta["radius_m"]) + meta.get("central_lat_deg", 0)
    lon = (np.degrees(x / (meta["radius_m"] * coslat)) + meta.get("central_lon_deg", 0) + 180) % 360 - 180
    mask = np.zeros(shape, dtype=bool); used = []
    for name in names:
        f = find(name)
        if f is None:
            raise KeyError(name)
        inside = _distance_m(lat, lon, f["lat"], f["lon"]) <= f["diameter_km"] * 500
        mask |= inside
        used.append({"name": f["name"], "type": f["type"], "diameter_km": f["diameter_km"], "cells": int(inside.sum())})
    return mask, used
