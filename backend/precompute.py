"""Deployment-build computation of the default replays and their local stereo routes.

Run during the Docker build (`python -m backend.precompute`): the SAME code, models and archived
products compute the three default scenarios once, so the free-tier server does not need minutes
of CPU after every cold start. Results are sealed with a hash of every input (backend sources,
models, frame/scenario/drive data) and are only used when that hash still matches at runtime.
They are labelled as computed at deployment build, never as live inference. Custom destinations
and single-frame requests are still computed live.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from . import data

ROOT = Path(__file__).resolve().parents[1]
CACHE = data.DATA_DIR / "build-cache.json.gz"
INPUTS = ["frames.json", "scenarios.json", "places-drive-path.json", "navigation.onnx", "navigation-model-card.json",
          "surface.onnx", "model-card.json"]


def input_hash() -> str:
    digest = hashlib.sha256()
    for path in sorted((ROOT / "backend").glob("*.py")):
        digest.update(path.name.encode()); digest.update(path.read_bytes())
    for name in INPUTS:
        path = data.DATA_DIR / name
        digest.update(name.encode()); digest.update(path.read_bytes() if path.is_file() else b"<missing>")
    for path in sorted(data.DATA_DIR.glob("*/metadata.json")):
        digest.update(str(path.relative_to(data.DATA_DIR)).encode()); digest.update(path.read_bytes())
    return digest.hexdigest()


def load():
    """The sealed build results, or None when absent or computed from different inputs."""
    if not CACHE.is_file():
        return None
    try:
        payload = json.loads(gzip.decompress(CACHE.read_bytes()))
    except (OSError, ValueError):
        return None
    return payload if payload.get("input_sha256") == input_hash() else None


def main():
    from . import replay, local_navigation
    data.load_regions()
    began = time.perf_counter()
    built_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    replays, local = {}, {}
    for scenario in replay.scenarios():
        sid = scenario["id"]
        tick = time.perf_counter()
        result = replay.compute(sid)
        frames = {}
        for step in result["steps"]:
            if step["frame_id"] in frames:
                continue
            try:
                frames[step["frame_id"]] = local_navigation.compute(step["frame_id"], result["goal"])
            except (OSError, ValueError, KeyError) as exc:
                frames[step["frame_id"]] = {"frame_id": step["frame_id"], "status": "unavailable",
                                            "reason": f"Local stereo/model inputs unavailable ({type(exc).__name__}).",
                                            "segments": [], "ribbons": [], "is_ground_projection": True}
        replays[sid] = result
        local[sid] = {"scenario": sid, "goal": result["goal"], "frames": frames,
                      "compute_ms": round((time.perf_counter() - tick) * 1000, 1), "computed_at": built_at}
        print(sid, len(frames), "frames", round(time.perf_counter() - tick, 1), "s", flush=True)
    payload = {"input_sha256": input_hash(), "built_at": built_at, "elapsed_s": round(time.perf_counter() - began, 1),
               "replays": replays, "local": local}
    CACHE.write_bytes(gzip.compress(json.dumps(payload, separators=(",", ":")).encode(), 6))
    print(CACHE, CACHE.stat().st_size, "bytes", payload["elapsed_s"], "s", flush=True)


if __name__ == "__main__":
    main()
