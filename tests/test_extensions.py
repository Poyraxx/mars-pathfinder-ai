import json
import math
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend import data, global_dem
from backend.main import app, missions


def _plane_tiles(monkeypatch, grade):
    # Algorithm fixture only: a uniform north-rising plane, not Mars data.
    def fake_tile(z, y, x):
        res = global_dem.LEVEL0_RES / 2 ** z
        rows = np.arange(512) + y * 512
        lat = 90 - (rows + .5) * res
        metres_north = np.radians(lat) * global_dem.R
        return np.repeat((metres_north * grade)[:, None], 512, axis=1).astype(np.float32)
    monkeypatch.setattr(global_dem, "tile", fake_tile)


def test_global_region_grid_reproduces_known_slope(monkeypatch):
    _plane_tiles(monkeypatch, grade=.05)
    region = global_dem.GlobalRegion([{"lat": 10.0, "lon": 20.0}, {"lat": 10.4, "lon": 20.3}])
    slope = region.analysis["slope"][5:-5, 5:-5]
    assert abs(float(np.nanmedian(slope)) - math.degrees(math.atan(.05))) < .2
    w, s, e, n = region.meta["bounds"]
    assert w < 20.0 < e and s < 10.0 < n and region.meta["cell_m"] >= 160


def test_global_region_rejects_polar_and_allows_long_or_wrapped_spans(monkeypatch):
    _plane_tiles(monkeypatch, grade=0)
    with pytest.raises(ValueError, match="polar"):
        global_dem.GlobalRegion([{"lat": 85, "lon": 0}, {"lat": 86, "lon": 1}])
    far = global_dem.GlobalRegion([{"lat": 0, "lon": 170}, {"lat": 10, "lon": -100}])  # long, crosses 180
    assert far.meta["cell_m"] > 10000 and far.meta["shape"][0] <= global_dem.MAX_CELLS
    w, s, e, n = far.meta["bounds"]
    assert w > e  # rectangle wraps across the antimeridian


def test_global_mission_plans_through_api(monkeypatch):
    _plane_tiles(monkeypatch, grade=.02)
    with TestClient(app) as client:
        created = client.post("/api/missions", json={"region_id": "global", "start": {"lat": 18.4, "lon": 77.4}, "goal": {"lat": 18.5, "lon": 77.55}})
        assert created.status_code == 201 and created.json()["region"]["coverage_level"] == "global_coarse"
        mid = created.json()["id"]
        client.post(f"/api/missions/{mid}/plan", json={"expected_map_version": 0})
        for _ in range(200):
            if missions[mid]["state"] in ("ready", "error"):
                break
            time.sleep(.05)
        assert all(r["status"] == "ready" for r in missions[mid]["routes"])
        frame = data.frame_catalog()[0]
        assert client.post(f"/api/missions/{mid}/observations", json={"frame_id": frame["id"], "base_map_version": 0}).status_code == 422


def test_nomenclature_catalogue_is_served():
    path = data.DATA_DIR / "global" / "nomenclature.json"
    if not path.exists():
        pytest.skip("nomenclature not prepared")
    with TestClient(app) as client:
        body = client.get("/api/nomenclature").json()
    names = {f["name"]: f for f in body["features"]}
    assert body["count"] == len(body["features"]) > 1000
    assert names["Olympus Mons"]["type"] == "Mons" and -180 <= names["Olympus Mons"]["lon"] < 180


def test_camera_route_is_drawn_only_through_measured_pixels():
    frames = [f for f in data.frame_catalog() if f["registration"].get("verified")]
    if not frames:
        pytest.skip("no registered frame")
    frame = frames[0]
    with TestClient(app) as client:
        vm = client.get(f"/api/frames/{frame['id']}/view-mission").json()
        mid = client.post("/api/missions", json={"region_id": "gale", "start": vm["start"], "goal": vm["goal"]}).json()["id"]
        assert client.get(f"/api/missions/{mid}/camera-route/{frame['id']}").status_code == 409
        client.post(f"/api/missions/{mid}/plan", json={"expected_map_version": 0})
        for _ in range(200):
            if missions[mid]["state"] in ("ready", "error"):
                break
            time.sleep(.05)
        projection = client.get(f"/api/missions/{mid}/camera-route/{frame['id']}").json()
    assert projection["segments"] and 0 < projection["visible_route_m"] <= projection["route_length_m"] + 1e-6
    xyz = np.load(data.DATA_DIR / frame["xyz_path"])
    valid = np.isfinite(xyz).all(-1) & (xyz != 0).any(-1)
    for seg in projection["segments"]:
        for u, v, _ in seg:
            assert valid[int(v * xyz.shape[0]), int(u * xyz.shape[1])]


def test_avoided_feature_is_blocked_and_reported(monkeypatch):
    if not (data.DATA_DIR / "global" / "nomenclature.json").exists():
        pytest.skip("nomenclature not prepared")
    _plane_tiles(monkeypatch, grade=0)
    from backend import features
    pav = features.find("Pavonis Mons")
    with TestClient(app) as client:
        body = {"region_id": "global", "start": {"lat": pav["lat"], "lon": pav["lon"] - 4}, "goal": {"lat": pav["lat"], "lon": pav["lon"] + 4}}
        ids = []
        for avoid in ([], ["Pavonis Mons"]):
            r = client.post("/api/missions", json={**body, "avoid_features": avoid}); assert r.status_code == 201
            mid = r.json()["id"]; ids.append(mid)
            client.post(f"/api/missions/{mid}/plan", json={"expected_map_version": 0})
            for _ in range(400):
                if missions[mid]["state"] in ("ready", "error"):
                    break
                time.sleep(.05)
        plain = next(r for r in missions[ids[0]]["routes"] if r["mode"] == "terrain")
        avoided = next(r for r in missions[ids[1]]["routes"] if r["mode"] == "terrain")
        assert any(f["name"] == "Pavonis Mons" and f["relation"] == "crosses" for f in plain["features"])
        assert not any(f["name"] == "Pavonis Mons" and f["relation"] == "crosses" for f in avoided["features"])
        assert avoided["distance_m"] > plain["distance_m"]
        bad = client.post("/api/missions", json={**body, "avoid_features": ["Not A Real Feature"]})
        assert bad.status_code == 422
