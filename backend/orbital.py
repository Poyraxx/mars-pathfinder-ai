"""Learned Orbital module: HiRISE landmark classification on the regional 25 cm ortho.

Runs live on the server with ONNX Runtime CPU. The ortho is split into square windows,
each resized to the model input and classified into 8 landmark types. Results are
regional context: the classifier was trained on labelled landmark crops, so sliding
windows over a map ortho are a domain shift and the output is not a hazard map.
"""
from __future__ import annotations

import io
import json
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .data import DATA_DIR

WINDOW_M = 64.0
_session = None
_dll = None
PALETTE = {"other": None, "crater": (240, 197, 141), "dark_dune": (120, 92, 200), "slope_streak": (90, 200, 230),
           "bright_dune": (250, 230, 150), "impact_ejecta": (238, 120, 90), "swiss_cheese": (150, 220, 160), "spider": (220, 120, 200)}


def card() -> dict:
    path = DATA_DIR / "orbital-model-card.json"
    return json.loads(path.read_text("utf-8")) if path.exists() else {"ready": False}


def session():
    global _session, _dll
    if _session is None:
        if os.name == "nt" and _dll is None:
            cfg = Path(__file__).resolve().parents[1] / ".local-run.json"
            d = os.environ.get("MARS_DLL_DIR") or (json.loads(cfg.read_text("utf-8")).get("dll_dir") if cfg.exists() else None)
            if d:
                _dll = os.add_dll_directory(d)
        import onnxruntime as ort
        path = DATA_DIR / "orbital.onnx"
        if not path.exists():
            raise ValueError("orbital_model_not_ready")
        opts = ort.SessionOptions(); opts.intra_op_num_threads = 1; opts.inter_op_num_threads = 1
        _session = ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])
    return _session


def analyze_region(region_id: str) -> tuple[dict, bytes]:
    meta = json.loads((DATA_DIR / "pds" / "region-ortho.json").read_text("utf-8"))
    ortho = np.asarray(Image.open(DATA_DIR / "pds" / meta["image"]).convert("L"))
    m = card(); classes = m["classes"]; size = m["input_size"]
    px = int(round(WINDOW_M / meta["pixel_resolution_m"]))
    rows, cols = ortho.shape[0] // px, ortho.shape[1] // px
    batch = []
    for r in range(rows):
        for c in range(cols):
            tile = Image.fromarray(ortho[r * px:(r + 1) * px, c * px:(c + 1) * px]).resize((size, size), Image.Resampling.BILINEAR)
            x = np.asarray(tile, dtype=np.float32) / 255
            batch.append(np.stack([(x - .485) / .229, (x - .456) / .224, (x - .406) / .225]))
    sess = session()
    began = time.perf_counter()
    logits = np.concatenate([sess.run(None, {"image": np.asarray(batch[i:i + 16], dtype=np.float32)})[0] for i in range(0, len(batch), 16)])
    elapsed = (time.perf_counter() - began) * 1000
    logits -= logits.max(1, keepdims=True); prob = np.exp(logits); prob /= prob.sum(1, keepdims=True)
    label = prob.argmax(1).reshape(rows, cols); score = prob.max(1).reshape(rows, cols)
    rgba = np.zeros((rows, cols, 4), dtype=np.uint8)
    for i, name in enumerate(classes):
        colour = PALETTE.get(name)
        if colour:
            sel = label == i
            rgba[sel, :3] = colour; rgba[sel, 3] = (90 + 120 * score[sel]).astype(np.uint8)
    out = io.BytesIO(); Image.fromarray(rgba).resize((cols * 8, rows * 8), Image.Resampling.NEAREST).save(out, format="PNG")
    counts = {name: int((label == i).sum()) for i, name in enumerate(classes)}
    result = {"region_id": region_id, "window_m": WINDOW_M, "grid": [rows, cols], "windows": int(rows * cols), "inference_ms": round(elapsed, 1),
              "class_counts": counts, "mean_top_score": round(float(score.mean()), 3),
              "bounds": meta["render_rectangle_degrees_west_south_east_north"],
              "model": m["name"], "test_macro_f1": round(m["test"]["macro_f1"], 4), "test_accuracy": round(m["test"]["accuracy"], 4),
              "note": "Live sliding-window classification of the HiRISE 25 cm ortho. Landmark context, not a hazard map; window classification is a domain shift from the labelled crops."}
    return result, out.getvalue()


def classify(arrays: list[np.ndarray]) -> np.ndarray:
    """Softmax probabilities for 0..1 grayscale windows already at the model input size."""
    batch = np.asarray([np.stack([(a - .485) / .229, (a - .456) / .224, (a - .406) / .225]) for a in arrays], dtype=np.float32)
    sess = session()
    logits = np.concatenate([sess.run(None, {"image": batch[i:i + 16]})[0] for i in range(0, len(batch), 16)])
    logits -= logits.max(1, keepdims=True); p = np.exp(logits); return p / p.sum(1, keepdims=True)


def analyze_route(points: list[dict], route_length_m: float, max_windows: int = 24) -> dict:
    from . import ctx
    size = float(min(5000.0, max(600.0, route_length_m / 10)))
    lat = np.array([p["lat"] for p in points]); lon = np.array([p["lon"] for p in points])
    step = np.hypot(np.diff(lat) * np.pi / 180 * ctx.R, np.diff(lon) * np.pi / 180 * ctx.R * np.cos(np.radians(lat[:-1])))
    along = np.r_[0, np.cumsum(step)]
    targets = np.linspace(0, along[-1], max(2, min(max_windows, int(along[-1] // size) + 2)))
    centres = [(float(np.interp(t, along, lat)), float(np.interp(t, along, lon)), float(t)) for t in targets]
    m = card(); classes = m["classes"]
    began = time.perf_counter()
    arrays, kept, skipped = [], [], 0
    for la, lo, t in centres:
        img, info = ctx.window(la, lo, size, m["input_size"])
        if img is None:
            skipped += 1; continue
        arrays.append(img); kept.append((la, lo, t, info))
    windows = []
    if arrays:
        prob = classify(arrays)
        for (la, lo, t, info), p in zip(kept, prob):
            k = int(p.argmax())
            windows.append({"lat": la, "lon": lo, "along_m": round(t, 1), "class": classes[k], "score": round(float(p[k]), 3),
                            "top3": [[classes[i], round(float(p[i]), 3)] for i in np.argsort(-p)[:3]], "ctx_level": info["level"]})
    counts = {c: sum(w["class"] == c for w in windows) for c in classes}
    return {"windows": windows, "window_m": round(size, 1), "skipped_no_coverage": skipped, "class_counts": counts,
            "elapsed_ms": round((time.perf_counter() - began) * 1000, 1), "imagery": ctx.SOURCE,
            "model": m["name"], "test_macro_f1": round(m["test"]["macro_f1"], 4), "test_accuracy": round(m["test"]["accuracy"], 4),
            "ctx_transfer_check": m.get("ctx_transfer_check"),
            "note": "Live classification of CTX windows along the route. Trained on HiRISE crops, applied to CTX: a domain shift; regional landmark context, not a hazard map."}
