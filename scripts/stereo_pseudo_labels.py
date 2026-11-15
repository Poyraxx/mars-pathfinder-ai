"""Stereo-derived pseudo-labels for the Surface model's weakest class (big_rock).

Source: the real PDS XYZ/XYE stereo products already prepared for 207 MSL frames
(65 Navcam, 142 Front Hazcam). No human annotation, no synthetic pixels.

Per pixel, with stereo error <= 0.10 m and range <= 4.5 m:
  * ground = 1.5 m median of the lowest measured height per 10 cm cell (robust to rocks
    smaller than ~0.7 m, follows real slopes);
  * big_rock  = compact raised component (relief >= 0.10 m, peak 0.15-0.6 m, <= 1.2 m wide)
    that rests on the ground (no depth jump below it) and does not enter from the frame
    edge; this rejects the rover's own arm turret, wheels, deck and mast;
  * other classes = the current model's own label, only where the ground is measured flat
    (|relief| <= 0.04 m) and the model is confident (>= 0.8). Where the model calls flat,
    measured ground a big_rock, the best non-rock class is used instead;
  * everything else (sky, far field, bedrock ledges, uncertain) = ignore (255).

These are PSEUDO-labels: geometry decides rock vs. not rock; texture classes come from
the model itself. Frames are split by sol so held-out sols are never trained on.
Run: python scripts/stereo_pseudo_labels.py
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend import data  # noqa: E402
from backend.perception import infer  # noqa: E402
from backend.local_navigation import frame_basis  # noqa: E402

TRAINING = Path(os.environ.get("LOCALAPPDATA", ".")) / "mars-pathfinder" / "training"
OUT = TRAINING / "pseudo"
CELL, WINDOW_M, MAX_ERR, MAX_RANGE = .10, 1.5, .10, 4.5
ROCK_RELIEF, ROCK_PEAK, ROCK_MAX_WIDTH, FLAT = .10, .15, 1.2, .04
ROCK_MAX_PEAK = .6
HELD_OUT_FRACTION = .25
FLAT_CORRECTION = True
ROCK_PEAK = .10
REVIEW_PATH = ROOT / "data" / "training" / "pseudo-rock-review.json"
REVIEW = json.loads(REVIEW_PATH.read_text("utf-8"))["accepted"] if REVIEW_PATH.exists() else None


def held_out(sol: int) -> bool:
    return int(hashlib.sha256(f"pseudo-sol-{sol}".encode()).hexdigest(), 16) % 100 < HELD_OUT_FRACTION * 100


def label_frame(frame):
    xyz = np.load(data.DATA_DIR / frame["xyz_path"]); xye = np.load(data.DATA_DIR / frame["xye_path"])
    h, w = xyz.shape[:2]
    err = np.linalg.norm(xye, axis=-1)
    valid = np.isfinite(xyz).all(-1) & (xyz != 0).any(-1) & np.isfinite(err) & (err <= MAX_ERR)
    if frame.get("camera_model"):
        origin, forward, right = frame_basis(frame)
        delta = xyz[..., :2] - origin[:2]
        x, y = delta @ forward, delta @ right
        valid &= (np.hypot(x, y) <= MAX_RANGE) & (x > 0)
        valid &= ~((x < 1.6) & (np.abs(y) <= 1.6))
    else:
        x, y = xyz[..., 0], xyz[..., 1]
    height = -xyz[..., 2]
    if valid.sum() < 500:
        return None
    gx = np.floor((x - x[valid].min()) / CELL).astype(int); gy = np.floor((y - y[valid].min()) / CELL).astype(int)
    shape = (gx[valid].max() + 1, gy[valid].max() + 1)
    low = np.full(shape, np.inf)
    np.minimum.at(low, (gx[valid], gy[valid]), height[valid])
    seen = np.isfinite(low)
    idx = ndimage.distance_transform_edt(~seen, return_distances=False, return_indices=True)
    filled = low[tuple(idx)]
    k = int(round(WINDOW_M / CELL)) | 1
    ground = ndimage.median_filter(filled, size=k, mode="nearest")
    relief = np.full((h, w), np.nan)
    relief[valid] = height[valid] - ground[gx[valid], gy[valid]]

    raised = valid & (relief >= ROCK_RELIEF)
    comps, n = ndimage.label(raised)
    rock = np.zeros((h, w), bool); rejected = np.zeros((h, w), bool); candidates = []
    accepted = REVIEW.get(frame["id"]) if REVIEW is not None else None
    for i, sl in enumerate(ndimage.find_objects(comps), start=1):
        m = comps[sl] == i
        peak = np.nanmax(relief[sl][m]); px = m.sum()
        cx, cy = gx[sl][m], gy[sl][m]
        width = max(np.ptp(cx), np.ptp(cy)) * CELL
        if px >= 8 and ROCK_PEAK <= peak <= ROCK_MAX_PEAK and width <= ROCK_MAX_WIDTH and _grounded(m, sl, xyz, valid):
            key = len(candidates)
            candidates.append({"key": key, "bbox": [sl[0].start, sl[1].start, sl[0].stop, sl[1].stop], "peak_m": round(float(peak), 3), "px": int(px)})
            if REVIEW is None or (accepted is not None and key in accepted):
                rock[sl] |= m
            else:
                rejected[sl] |= m
        else:
            rejected[sl] |= m

    _, mask, conf, _, _ = infer(frame["id"])
    mask = np.asarray(Image.fromarray(mask).resize((w, h), Image.Resampling.NEAREST))
    conf = np.asarray(Image.fromarray(conf.astype(np.float32)).resize((w, h), Image.Resampling.NEAREST))
    source = Image.open(data.DATA_DIR / frame["image_path"]).convert("RGB")
    flat = valid & (np.abs(relief) <= FLAT) & ~ndimage.binary_dilation(rock | rejected, iterations=2)
    label = np.full((h, w), 255, np.uint8)
    confident = flat & (conf >= .8) & (mask != 3)
    label[confident] = mask[confident]
    false_rock = flat & (mask == 3)
    if false_rock.any() and FLAT_CORRECTION:
        probs = _probs(source, (w, h))
        alt = probs[:3].argmax(0).astype(np.uint8)
        label[false_rock] = alt[false_rock]
    label[rock] = 3
    stats = {"rock_px": int(rock.sum()), "model_rock_on_stereo_rock": int((rock & (mask == 3)).sum()),
             "model_rock_on_flat_ground": int(false_rock.sum()), "labelled_px": int((label != 255).sum()), "pixels": h * w}
    return label, stats, candidates


def _grounded(m, sl, xyz, valid):
    """A rock sits on the ground; rover hardware (arm turret, wheels, deck, mast) floats or enters from the frame edge.

    Reject components touching the image border. Require that, for >= 50% of the
    component's columns, the measured pixel just below its lowest pixel is within
    0.35 m in 3D (contact with the surrounding ground), not a depth jump."""
    h, w = valid.shape
    r0, c0 = sl[0].start, sl[1].start
    if r0 <= 2 or c0 <= 2 or sl[0].stop >= h - 2 or sl[1].stop >= w - 2:
        return False
    checked = touching = 0
    for j in range(m.shape[1]):
        rows = np.flatnonzero(m[:, j])
        if not len(rows):
            continue
        r, c = r0 + rows[-1], c0 + j
        below = r + 2
        if below >= h or not valid[below, c] or not valid[r, c]:
            continue
        checked += 1
        touching += np.linalg.norm(xyz[below, c] - xyz[r, c]) <= .35
    return checked >= 3 and touching >= .5 * checked


def _probs(source, size):
    from backend.perception import get_session
    s = int(data.model_status().get("input_size", 320))
    img = np.asarray(source.resize((s, s), Image.Resampling.BILINEAR), dtype=np.float32) / 255
    x = ((img - [.485, .456, .406]) / [.229, .224, .225]).transpose(2, 0, 1)[None].astype(np.float32)
    sess = get_session()
    logits = sess.run(None, {sess.get_inputs()[0].name: x})[0][0]
    p = np.exp(logits - logits.max(0, keepdims=True)); p /= p.sum(0, keepdims=True)
    return np.stack([np.asarray(Image.fromarray(c.astype(np.float32)).resize(size, Image.Resampling.BILINEAR)) for c in p])


def main():
    global FLAT_CORRECTION
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--no-flat-correction", action="store_true",
        help="leave model rock calls on measured-flat ground unlabelled instead of relabelling them")
    ap.add_argument("--name", default="manifest-pseudo.json"); args = ap.parse_args()
    FLAT_CORRECTION = not args.no_flat_correction
    data.load_regions()
    out = OUT if FLAT_CORRECTION else OUT.with_name("pseudo-nocorr")
    out.mkdir(parents=True, exist_ok=True)
    entries, totals, all_candidates = [], {}, {}
    for frame in data.frame_catalog():
        if not frame.get("xyz_path"):
            continue
        result = label_frame(frame)
        if result is None or result[1]["labelled_px"] < 800:
            continue
        label, stats, candidates = result
        all_candidates[frame["id"]] = candidates
        path = out / f"{frame['id']}.png"
        Image.fromarray(label).save(path)
        split = "pseudo_test" if held_out(frame["sol"]) else "train"
        entries.append({"stem": frame["id"], "image": str(data.DATA_DIR / frame["image_path"]), "label": str(path),
                        "group": f"pseudo-sol{frame['sol']}", "split": split, "instrument": frame.get("instrument"),
                        "sol": frame["sol"], "pseudo": True, **stats})
        for k, v in stats.items():
            totals.setdefault(split, {}).setdefault(k, 0); totals[split][k] += v
    manifest = {"source": "MSL PDS XYZ/XYE stereo products (prepared frames)", "kind": "stereo pseudo-labels",
                "flat_ground_rock_correction": FLAT_CORRECTION, "review": "every candidate visually reviewed; accepted list in data/training/pseudo-rock-review.json" if REVIEW is not None else "none", "rule": {"cell_m": CELL, "ground_window_m": WINDOW_M, "max_stereo_error_m": MAX_ERR, "max_range_m": MAX_RANGE,
                         "rock_relief_m": ROCK_RELIEF, "rock_max_peak_m": ROCK_MAX_PEAK, "rock_peak_m": ROCK_PEAK, "rock_max_width_m": ROCK_MAX_WIDTH, "flat_m": FLAT,
                         "texture_classes": "current model, conf >= 0.8 on measured-flat ground"},
                "split": f"by sol, sha256 hash, {HELD_OUT_FRACTION:.0%} of sols held out as pseudo_test",
                "totals": totals, "entries": entries}
    (TRAINING / "pseudo-candidates.json").write_text(json.dumps(all_candidates), "utf-8")
    (TRAINING / args.name).write_text(json.dumps(manifest, indent=2), "utf-8")
    print(json.dumps({"frames": len(entries), "by_split": {s: sum(e["split"] == s for e in entries) for s in ("train", "pseudo_test")}, "totals": totals}, indent=2))


if __name__ == "__main__":
    main()
