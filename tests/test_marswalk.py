"""Marswalk layer: profiles, round trip with reserve, sourced environment, geology."""
import time
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from backend import marswalk
from backend.main import app


def test_solar_longitude_matches_curiosity_landing():
    # Curiosity landed 2012-08-06 at Ls ~150.
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
