"""Live drive replay with turn-by-turn style re-routing (GTA-like waypoint navigation).

For a scenario, the real rover poses (PLACES localizations of archived mid-drive frames)
are replayed in order. The navigator plans a route to the scenario goal (the rover's real
arrival point) on the regional DEM. At every frame:

* cross-track deviation from the active route is measured; beyond DEVIATION_M the route
  is re-planned from the rover's real position. If the new route starts behind the rover
  (heading difference > 120 degrees) the event is a U-turn.
* for registered frames, stereo-measured obstacles are added to the map; when they cut
  the remaining route the navigator re-plans ("obstacle ahead").
* the active route is projected into the camera frame through measured stereo pixels.

Everything is computed on the server from archived data; motion between frames is not
interpolated and nothing is replayed from a recording.
"""
from __future__ import annotations

import json
import math
import time

import numpy as np

from . import data
from .camera_route import project_route
from .coordinates import cell_to_point
from .navigation import cost_grid, inflate_obstacles, plan_legs
from .perception import localize_obstacles

R = 3396190.0
DEVIATION_M = 2.0
UTURN_DEG = 120.0
CLEAR_AROUND_ROVER_M = 1.5
_cache: dict = {}


def scenarios() -> list[dict]:
    path = data.DATA_DIR / "scenarios.json"
    return json.loads(path.read_text("utf-8")) if path.exists() else []


def _xy(ned, meta):
    # Gale meta uses standard parallel 0 and central lon 0: x = east metres, y = north metres.
    return np.array([ned[1], ned[0]], dtype=float)


def _cell(xy, meta):
    return (int(math.floor((meta["north_m"] - xy[1]) / meta["cell_m"])), int(math.floor((xy[0] - meta["west_m"]) / meta["cell_m"])))


def _centre(rc, meta):
    return np.array([meta["west_m"] + (rc[1] + .5) * meta["cell_m"], meta["north_m"] - (rc[0] + .5) * meta["cell_m"]])


def _track_distance(p, poly):
    """Distance from p to polyline, plus along-route position of the closest point."""
    best, along, acc = 1e18, 0.0, 0.0
    for a, b in zip(poly, poly[1:]):
        ab = b - a; L = float(np.hypot(*ab))
        t = 0.0 if L == 0 else float(np.clip(np.dot(p - a, ab) / L ** 2, 0, 1))
        d = float(np.hypot(*(p - (a + t * ab))))
        if d < best:
            best, along = d, acc + t * L
        acc += L
    return best, along, acc


def _heading(v):
    return math.degrees(math.atan2(v[0], v[1])) % 360  # clockwise from north, v = (east, north)


def compute(scenario_id: str, profile: dict | None = None, goal: dict | None = None) -> dict:
    key = (scenario_id, None if goal is None else (round(goal["lat"], 7), round(goal["lon"], 7)))
    if key in _cache:
        return _cache[key]
    sc = next((s for s in scenarios() if s["id"] == scenario_id), None)
    if sc is None:
        raise KeyError(scenario_id)
    region = data.regions[sc["region_id"]]; meta = region.meta
    catalogue = {f["id"]: f for f in data.frame_catalog()}
    frames = [catalogue[i] for i in sc["frames"] if i in catalogue]
    profile = {"max_slope_deg": 20.0, "clearance_m": 1.0, "obstacle_step_m": .15, **(profile or {})}
    poses = [np.array(f["places"]["corrected_rover_origin_global_ned_m"], dtype=float) for f in frames]
    yaws = [float(f["places"]["row"]["yaw"]) % 360 for f in frames]
    if goal is None:
        goal_xy = _xy(poses[-1], meta)
    else:  # user-chosen destination (equirectangular, standard parallel 0: x = R*lon, y = R*lat)
        goal_xy = np.array([math.radians(goal["lon"]) * R, math.radians(goal["lat"]) * R])
    goal_rc = _cell(goal_xy, meta)
    if not (0 <= goal_rc[0] < region.elevation.shape[0] and 0 <= goal_rc[1] < region.elevation.shape[1]) or not region.analysis["valid"][goal_rc]:
        raise ValueError("goal_outside_region")
    blocked = np.zeros(region.elevation.shape, dtype=bool)
    began = time.perf_counter()

    def plan(from_xy):
        b = blocked.copy()
        rr, cc = np.ogrid[:b.shape[0], :b.shape[1]]
        s = _cell(from_xy, meta)
        b[(rr - s[0]) ** 2 + (cc - s[1]) ** 2 <= (CLEAR_AROUND_ROVER_M / meta["cell_m"]) ** 2] = False  # the rover stands here
        cost = cost_grid(region.analysis, "surface", profile["max_slope_deg"], b)
        path, _ = plan_legs(cost, [s, goal_rc], meta["cell_m"])
        return path

    route_cells = plan(_xy(poses[0], meta)); version = 1
    steps, events = [], []
    for k, (f, ned, yaw) in enumerate(zip(frames, poses, yaws)):
        p = _xy(ned, meta)
        event = None
        poly = np.array([_centre(c, meta) for c in route_cells])
        dev, along, total = _track_distance(p, poly)
        # Stereo obstacles from this registered frame (measured geometry, not the learned class).
        hazards = 0
        if f["registration"].get("verified"):
            loc = localize_obstacles(f, np.zeros(tuple(f["shape"]), dtype=np.uint8), np.zeros(tuple(f["shape"])), meta, profile["obstacle_step_m"])
            if loc.get("accepted") and len(loc.get("points_rc", [])):
                hazards = int(loc.get("hazard_pixels", 0))
                update = inflate_obstacles(blocked.shape, loc["points_rc"], (profile["clearance_m"] + loc["registration_error_m"]) / meta["cell_m"])
                ahead = [c for c in route_cells if np.hypot(*(_centre(c, meta) - p)) > CLEAR_AROUND_ROVER_M]
                new = update & ~blocked
                blocked |= update
                if any(new[c[0], c[1]] for c in ahead):
                    event = "obstacle"
        if event is None and dev > DEVIATION_M:
            event = "off_route"
        if event and (k < len(frames) - 1 or goal is not None):
            try:
                new_cells = plan(p)
            except (ValueError, TimeoutError):
                new_cells = route_cells; event = "no_alternative"
            if event != "no_alternative":
                first = _centre(new_cells[min(3, len(new_cells) - 1)], meta) - p
                turn = abs((_heading(first) - yaw + 180) % 360 - 180) if np.hypot(*first) > 0 else 0.0
                if turn > UTURN_DEG and event == "off_route":
                    event = "u_turn"
                route_cells, version = new_cells, version + 1
                events.append({"step": k, "type": event, "deviation_m": round(dev, 2), "route_version": version, "turn_deg": round(turn, 0)})
            poly = np.array([_centre(c, meta) for c in route_cells])
            dev, along, total = _track_distance(p, poly)
        remaining = max(total - along, 0.0)
        idx = np.unique(np.linspace(0, len(route_cells) - 1, min(len(route_cells), 200)).astype(int))
        # Registered frames: placement verified by image correlation. Others: drawn with the archived
        # PLACES pose only, flagged unverified (display only; never used to update the map).
        verified = bool(f["registration"].get("verified"))
        tol = max(.35, min(1.0, f["registration"]["horizontal_error_bound_m"] / 2)) if verified else .5
        projection = project_route(f, meta, route_cells, tol)
        projection = {k2: projection[k2] for k2 in ("segments", "visible_route_m", "tolerance_m")} | {"placement": "image-verified" if verified else "archived PLACES pose, not independently verified"}
        lat, lon = math.degrees(ned[0] / R), (math.degrees(ned[1] / R) + 180) % 360 - 180
        steps.append({"step": k, "frame_id": f["id"], "sol": f["sol"], "time": f["acquired_at"], "rover": {"lat": lat, "lon": lon, "yaw_deg": round(yaw, 1)},
                      "registered": bool(f["registration"].get("verified")), "event": event, "deviation_m": round(dev, 2),
                      "remaining_m": round(remaining, 1), "route_version": version, "hazard_pixels": hazards,
                      "route": [cell_to_point(*route_cells[i], meta) for i in idx], "camera_route": projection})
    goal_out = {"lat": math.degrees(goal_xy[1] / R), "lon": (math.degrees(goal_xy[0] / R) + 180) % 360 - 180, "custom": goal is not None}
    result = {"scenario": {k: sc[k] for k in ("id", "title", "subtitle", "story", "references", "note", "camera", "sols")}, "goal": goal_out,
              "steps": steps, "events": events, "parameters": {"deviation_m": DEVIATION_M, "u_turn_deg": UTURN_DEG, **profile},
              "trail": [s["rover"] for s in steps], "compute_ms": round((time.perf_counter() - began) * 1000, 1),
              "method": "Real PLACES rover poses replayed in order; route to the real arrival point re-planned on deviation or new stereo obstacles."}
    _cache[key] = result
    while len(_cache) > 12:
        _cache.pop(next(iter(_cache)))
    return result
