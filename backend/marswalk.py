"""Marswalk mission layer: movement profiles, science stops, round trip with reserve, environment, geology.

Every number carries its kind: "measurement" (instrument data), "historical" (climatology from past
missions) or "model" (computed here from orbital geometry / DEM). Planning assumptions are labelled as
such; nothing here certifies suit consumables or crew safety.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from functools import lru_cache

import numpy as np
from PIL import Image

from . import data
from .navigation import cost_grid, plan_legs

# ---------------------------------------------------------------- movement profiles
PROFILES = {
    "astronaut": {"label": "Walking astronaut", "speed_m_s": 0.8, "max_slope_deg": 25, "clearance_m": 0.5, "obstacle_step_m": 0.30,
                  "slope_speed": "tobler", "budget_min": 420,
                  "basis": "Planning assumption. Apollo lunar EVA walking ran roughly 0.5-1 m/s; slope slows walking per Tobler's hiking function. Not validated for a Mars suit."},
    "crew_cart": {"label": "Crew hauling equipment", "speed_m_s": 0.5, "max_slope_deg": 15, "clearance_m": 1.0, "obstacle_step_m": 0.15,
                  "slope_speed": "tobler", "budget_min": 420,
                  "basis": "Planning assumption: two crew with a wheeled sample/equipment cart; gentler slopes, wider clearance, smaller steps than walking."},
    "rover": {"label": "Curiosity-class rover", "speed_m_s": 0.04, "max_slope_deg": 20, "clearance_m": 1.5, "obstacle_step_m": 0.25,
              "slope_speed": "constant", "budget_min": 1440,
              "basis": "Curiosity top speed is about 4 cm/s (NASA); 20 deg is a conservative drive limit; 0.25 m step matches this project's wheel-hazard threshold. Real drives average far slower."},
}


def tobler_factor(slope_rise_over_run):
    """Tobler (1993) hiking speed relative to flat ground; Earth-derived, used as a relative slope penalty only."""
    return math.exp(-3.5 * abs(slope_rise_over_run + 0.05)) / math.exp(-3.5 * 0.05)


def path_time_s(cells, elevation, cell_m, profile):
    """Travel time along grid cells with slope-dependent speed for walkers."""
    p = PROFILES.get(profile.get("kind", "astronaut"), PROFILES["astronaut"])
    speed = float(profile.get("speed_m_s", p["speed_m_s"]))
    total = 0.0
    for (r0, c0), (r1, c1) in zip(cells, cells[1:]):
        run = math.hypot(r1 - r0, c1 - c0) * cell_m
        rise = float(elevation[r1, c1] - elevation[r0, c0])
        factor = tobler_factor(rise / run) if p["slope_speed"] == "tobler" and run > 0 and np.isfinite(rise) else 1.0
        total += run / max(speed * factor, 1e-3)
    return total


# ---------------------------------------------------------------- science stops
@lru_cache(maxsize=1)
def science_targets():
    path = data.DATA_DIR / "science-targets.json"
    return json.loads(path.read_text("utf-8")) if path.exists() else {"gale": [], "global": []}


# ---------------------------------------------------------------- round trip and reserve
def eva_summary(region, mission, route_mode="terrain"):
    """Outbound (with stops) + work at stops + return to start, against a time budget; marginal cost per stop."""
    req = mission["request"]
    profile = {**PROFILES.get(req["profile"].get("kind", "astronaut"), PROFILES["astronaut"]), **req["profile"]}
    eva = req.get("eva") or {}
    budget_min = float(eva.get("budget_min") or profile["budget_min"])
    work = list(eva.get("work_min") or [])
    stops = req["stops"]
    work = [float(work[i]) if i < len(work) else 30.0 for i in range(len(stops))]
    blocked = mission["blocked"] if route_mode == "surface" else None
    avoid = mission.get("_avoid")
    if avoid is not None:
        blocked = avoid if blocked is None else (blocked | avoid)
    cost = cost_grid(region.analysis, route_mode, float(profile["max_slope_deg"]), blocked)
    from .coordinates import point_to_cell
    cell = lambda p: point_to_cell(p, region.meta)
    start, goal = cell(req["start"]), cell(req["goal"])
    stop_cells = [cell(p) for p in stops]
    cm = region.meta["cell_m"]

    def leg(points):
        path, _ = plan_legs(cost, points, cm)
        dist = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:])) * cm
        return path, dist, path_time_s(path, region.elevation, cm, profile)

    try:
        out_path, out_m, out_s = leg([start, *stop_cells, goal])
        back_path, back_m, back_s = leg([goal, start])
    except (ValueError, TimeoutError) as exc:
        return {"status": str(exc), "profile": profile["label"]}
    work_s = sum(work) * 60
    total_s = out_s + work_s + back_s
    reserve_s = budget_min * 60 - total_s
    marginal = []
    for i in range(len(stops)):
        try:
            _, m_without, s_without = leg([start, *[c for j, c in enumerate(stop_cells) if j != i], goal])
            marginal.append({"stop": i + 1, "extra_distance_m": round(out_m - m_without, 1),
                             "extra_time_min": round((out_s - s_without + work[i] * 60) / 60, 1),
                             "reserve_without_min": round((reserve_s + out_s - s_without + work[i] * 60) / 60, 1)})
        except (ValueError, TimeoutError):
            marginal.append({"stop": i + 1, "extra_distance_m": None, "extra_time_min": None, "reserve_without_min": None})
    return {"status": "ready", "profile": profile["label"], "profile_kind": req["profile"].get("kind", "astronaut"), "route_mode": route_mode,
            "speed_m_s": profile["speed_m_s"], "max_slope_deg": profile["max_slope_deg"], "basis": profile["basis"],
            "outbound": {"distance_m": round(out_m, 1), "time_min": round(out_s / 60, 1)},
            "stops": [{"stop": i + 1, "work_min": w} for i, w in enumerate(work)],
            "return": {"distance_m": round(back_m, 1), "time_min": round(back_s / 60, 1),
                       "points": [_point(region, c) for c in back_path[:: max(1, len(back_path) // 300)]]},
            "total_min": round(total_s / 60, 1), "budget_min": budget_min, "reserve_min": round(reserve_s / 60, 1),
            "reserve_fraction": round(reserve_s / (budget_min * 60), 3), "stop_marginal": marginal,
            "note": "Time budget is an adjustable planning figure, not a suit consumables model. Walking speed uses Tobler's slope penalty."}


def _point(region, rc):
    from .coordinates import cell_to_point
    return cell_to_point(int(rc[0]), int(rc[1]), region.meta)


# ---------------------------------------------------------------- environment
def solar_longitude(when: datetime) -> float:
    """Ls (deg) from Allison & McEwen (2000) without the small planetary perturbation terms (<0.05 deg)."""
    jd_ut = when.timestamp() / 86400 + 2440587.5
    dt = jd_ut + 69.184 / 86400 - 2451545.0
    m = math.radians(19.3871 + 0.52402073 * dt)
    alpha = 270.3871 + 0.524038496 * dt
    eq_centre = (10.691 + 3.0e-7 * dt) * math.sin(m) + 0.623 * math.sin(2 * m) + 0.050 * math.sin(3 * m) + 0.005 * math.sin(4 * m) + 0.0005 * math.sin(5 * m)
    return (alpha + eq_centre) % 360


def environment(lat: float, lon: float, elevation_m: float | None, when: datetime | None = None) -> dict:
    when = when or datetime.now(timezone.utc)
    ls = solar_longitude(when)
    phi, decl = math.radians(lat), math.asin(math.sin(math.radians(25.19)) * math.sin(math.radians(ls)))
    r_au = 1.5236 * (1 - 0.0934 ** 2) / (1 + 0.0934 * math.cos(math.radians(ls - 251)))
    s0 = 1361 / r_au ** 2
    x = -math.tan(phi) * math.tan(decl)
    h0 = math.pi if x <= -1 else 0.0 if x >= 1 else math.acos(x)
    daily = s0 / math.pi * (h0 * math.sin(phi) * math.sin(decl) + math.cos(phi) * math.cos(decl) * math.sin(h0))
    daylight_h = 24.6597 * h0 / math.pi
    noon_cos = max(0.0, math.cos(phi - decl))
    t_noon_max = ((1 - 0.25) * s0 * noon_cos / (0.95 * 5.670374e-8)) ** 0.25 - 273.15 if noon_cos > 0 else None
    pressure = 610 * math.exp(-(elevation_m or 0) / 11100) if elevation_m is not None else None
    gale_pressure = 610 * math.exp(4500 / 11100)
    northern = ["spring", "summer", "autumn", "winter"][int(ls // 90)]
    season = f"Northern {northern} / southern {['autumn', 'winter', 'spring', 'summer'][int(ls // 90)]}"
    dusty = 180 <= ls < 360
    items = [
        {"key": "season", "label": "Season (Ls)", "value": f"Ls {ls:.0f}° · {season}", "kind": "model",
         "source": "Allison & McEwen 2000, Mars solar longitude algorithm", "url": "https://doi.org/10.1016/S0032-0633(99)00092-6"},
        {"key": "dust_season", "label": "Dust storm season", "value": "Yes: Ls 180-360 (perihelion), regional storms likely" if dusty else "No: Ls 0-180 (aphelion), clearer skies",
         "kind": "historical", "source": "Decades of orbital dust climatology (e.g. MGS TES, Mars Climate Sounder)", "url": "https://mars.nasa.gov/resources/21606/dust-storm-season/"},
        {"key": "insolation", "label": "Daily mean sunlight (top of atmosphere)", "value": f"{daily:.0f} W/m²", "kind": "model",
         "source": "Orbital geometry (obliquity 25.19°, eccentricity 0.0934); dust opacity not included", "url": "https://nssdc.gsfc.nasa.gov/planetary/factsheet/marsfact.html"},
        {"key": "daylight", "label": "Daylight", "value": f"{daylight_h:.1f} h of a 24.66 h sol", "kind": "model",
         "source": "Sunrise hour angle at this latitude and Ls", "url": "https://nssdc.gsfc.nasa.gov/planetary/factsheet/marsfact.html"},
        {"key": "noon_temperature", "label": "Noon ground temperature, upper bound",
         "value": f"≤ {t_noon_max:.0f} °C" if t_noon_max is not None else "Polar night: no sunlight", "kind": "model",
         "source": "Radiative equilibrium with albedo 0.25, emissivity 0.95; real peaks are lower (thermal inertia, dust). Nights are typically below -70 °C.", "url": "https://nssdc.gsfc.nasa.gov/planetary/factsheet/marsfact.html"},
        {"key": "pressure", "label": "Surface pressure estimate", "value": f"{pressure:.0f} Pa" if pressure is not None else "—", "kind": "model",
         "source": "610 Pa at datum, 11.1 km scale height, route elevation from the DEM", "url": "https://nssdc.gsfc.nasa.gov/planetary/factsheet/marsfact.html"},
        {"key": "radiation", "label": "Radiation dose (surface)", "value": "≈ 0.64 mSv/day measured at Gale (2012-13)", "kind": "measurement",
         "source": "MSL RAD, Hassler et al. 2014, Science 343. Varies with solar cycle and altitude" + (
             f"; air column here is {pressure / gale_pressure:.0%} of Gale's, so shielding is {'lower' if pressure < gale_pressure else 'similar or higher'} (model)" if pressure else ""),
         "url": "https://doi.org/10.1126/science.1244797"},
    ]
    return {"lat": lat, "lon": lon, "date_utc": when.isoformat(timespec="seconds"), "ls_deg": round(ls, 1), "items": items,
            "note": "Each value says whether it is a measurement, historical climatology or a model computed here. Map layers (dust, thermal inertia, night infrared) show the measured orbital context."}


# ---------------------------------------------------------------- geology
@lru_cache(maxsize=1)
def _geology():
    base = data.DATA_DIR / "geology"
    if not (base / "units.png").exists():
        return None, None
    return np.asarray(Image.open(base / "units.png")), json.loads((base / "legend.json").read_text("utf-8"))


def geology_along(points: list[dict]) -> dict | None:
    """USGS SIM 3292 units crossed by a route, in order, with along-route distance shares."""
    grid, legend = _geology()
    if grid is None or len(points) < 2:
        return None
    ppd = legend["grid"]["ppd"]
    units = {u["index"]: u for u in legend["units"]}
    r = 3396.19
    seq, along = [], 0.0
    for i, p in enumerate(points):
        if i:
            q = points[i - 1]
            a = math.sin(math.radians(p["lat"] - q["lat"]) / 2) ** 2 + math.cos(math.radians(p["lat"])) * math.cos(math.radians(q["lat"])) * math.sin(math.radians(p["lon"] - q["lon"]) / 2) ** 2
            along += 2 * r * math.asin(min(1, math.sqrt(a)))
        row = min(grid.shape[0] - 1, max(0, int((90 - p["lat"]) * ppd)))
        col = min(grid.shape[1] - 1, max(0, int((p["lon"] + 180) * ppd)))
        idx = int(grid[row, col])
        if not seq or seq[-1]["index"] != idx:
            seq.append({"index": idx, "from_km": round(along, 2)})
    total = along
    out = []
    for i, s in enumerate(seq):
        end = seq[i + 1]["from_km"] if i + 1 < len(seq) else total
        u = units.get(s["index"], {"unit": "?", "description": "No mapped unit", "period": "—", "colour": "#888888"})
        out.append({"unit": u["unit"], "description": u["description"], "period": u["period"], "colour": u["colour"],
                    "from_km": s["from_km"], "to_km": round(end, 2)})
    return {"segments": out, "source": legend["source"], "url": legend["url"], "note": legend["note"]}
