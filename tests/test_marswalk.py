"""Marswalk layer: profiles, round trip with reserve, sourced environment, geology."""
import time
from datetime import datetime, timezone

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend import marswalk
from backend.main import app


def test_solar_longitude_matches_curiosity_landing():
    assert marswalk.solar_longitude(datetime(2012, 8, 6, 5, 17, tzinfo=timezone.utc)) == pytest.approx(150.6, abs=0.6)


def test_environment_values_are_labelled_by_kind_and_sourced():
    env = marswalk.environment(-4.59, 137.44, -4500, datetime(2012, 8, 6, tzinfo=timezone.utc))
    kinds = {i['key']: i['kind'] for i in env['items']}
    assert kinds['radiation'] == 'measurement' and kinds['dust_season'] == 'historical' and kinds['pressure'] == 'model'
    assert all(i['url'].startswith('https://') and i['source'] for i in env['items'])


def test_tobler_slows_walking_uphill_and_steep_downhill():
    assert marswalk.tobler_factor(0) == pytest.approx(1, abs=.2)
    assert marswalk.tobler_factor(.3) < .5 and marswalk.tobler_factor(-.5) < marswalk.tobler_factor(-.05)


def test_geology_along_gale_is_the_mapped_unit():
    g = marswalk.geology_along([{'lat': -4.59, 'lon': 137.44}, {'lat': -4.59, 'lon': 137.45}])
    assert g and g['segments'][0]['unit'] == 'Htu' and '3292' in g['source']


def test_round_trip_reserve_and_stop_marginals():
    t = marswalk.science_targets()['gale']
    body = {"region_id": "gale", "start": {"lat": -4.5903, "lon": 137.4447}, "goal": {"lat": -4.5890, "lon": 137.4520},
            "stops": [{"lat": t[0]['lat'], "lon": t[0]['lon']}],
            "profile": {"kind": "astronaut", "max_slope_deg": 25, "speed_m_s": .8, "clearance_m": .5, "obstacle_step_m": .3},
            "eva": {"budget_min": 120, "work_min": [30]}}
    with TestClient(app) as c:
        mid = c.post('/api/missions', json=body).json()['id']
        c.post(f'/api/missions/{mid}/plan', json={})
        for _ in range(80):
            s = c.get(f'/api/missions/{mid}/state').json()
            if s['state'] in ('ready', 'error'):
                break
            time.sleep(.25)
    e = s['eva']
    assert e['status'] == 'ready' and e['return']['distance_m'] > 0
    assert e['total_min'] == pytest.approx(e['outbound']['time_min'] + 30 + e['return']['time_min'], abs=.2)
    assert e['reserve_min'] == pytest.approx(120 - e['total_min'], abs=.2)
    assert e['stop_marginal'][0]['extra_time_min'] >= 30


def test_overlays_and_targets_are_published_with_sources():
    with TestClient(app) as c:
        cat = c.get('/api/catalog').json()
        assert {'dust', 'thermal_inertia', 'night_ir', 'geology', 'confidence'} <= {o['id'] for o in cat['overlays']}
        assert all(o['source'] and o['url'] for o in cat['overlays'])
        assert set(cat['profiles']) == {'astronaut', 'crew_cart', 'rover'}
        assert c.get('/api/overlays/unknown/0/0/0').status_code == 404
        targets = c.get('/api/science-targets').json()
        assert all(t['sources'] and t['position_source'] for t in targets['gale'] + targets['global'])


def _region(elevation, cell_m=1.0):
    from types import SimpleNamespace
    from backend.navigation import terrain_analysis
    return SimpleNamespace(elevation=elevation, analysis=terrain_analysis(elevation, cell_m), meta={"cell_m": cell_m})


def _decide(elevation, start, goal, kind="astronaut"):
    from backend import marswalk
    from backend.navigation import cost_grid
    region = _region(elevation)
    profile = marswalk.profile_of({"kind": kind, "max_slope_deg": marswalk.PROFILES[kind]["max_slope_deg"]})
    cost = cost_grid(region.analysis, "terrain", profile["max_slope_deg"])
    path, decision = marswalk.route_decision(region, [start, goal], profile, cost)
    return region, path, decision


def test_low_hill_is_climbed_when_going_around_takes_longer():
    yy, xx = np.mgrid[0:120, 0:120]
    hill = 3.0 * np.exp(-((yy - 60) ** 2 + (xx - 60) ** 2) / (2 * 25.0 ** 2))
    region, path, d = _decide(hill, (60, 5), (60, 115))
    chosen = next(o for o in d["options"] if o["name"] == d["chosen"])
    assert all(o["time_min"] >= chosen["time_min"] for o in d["options"])
    assert chosen["climb_m"] > 1
    assert any("climbing is the better choice" in r or "has to climb" in r or "Chosen" in r for r in d["reasons"])


def test_steep_wall_is_never_crossed_and_reason_says_so():
    elevation = np.zeros((120, 120))
    elevation[:, 55:65] = np.linspace(0, 20, 10)[None, :]
    elevation[:, 65:] = 20
    elevation[100:, 20:100] = np.linspace(0, 20, 80)[None, :]
    elevation[100:, 100:] = 20
    region, path, d = _decide(elevation, (20, 10), (20, 110))
    slopes = region.analysis["slope"][tuple(np.array(path).T)]
    assert slopes.max() <= d["limit_deg"] + 1e-6
    assert d["direct"]["too_steep_m"] > 0
    assert "too steep" in d["reasons"][0]


def test_big_hill_is_walked_around_when_climbing_is_slower():
    yy, xx = np.mgrid[0:160, 0:160]
    hill = 40.0 * np.exp(-((yy - 80) ** 2 + (xx - 80) ** 2) / (2 * 22.0 ** 2))
    region, path, d = _decide(hill, (80, 4), (80, 156))
    over = next(o for o in d["options"] if o["name"] == "over")
    chosen = next(o for o in d["options"] if o["name"] == d["chosen"])
    assert chosen["time_min"] <= over["time_min"]
    assert chosen["climb_m"] < 40
    assert slopes_ok(region, path, d)


def slopes_ok(region, path, d):
    return region.analysis["slope"][tuple(np.array(path).T)].max() <= d["limit_deg"] + 1e-6
