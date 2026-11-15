"""Superpixel-assisted manual labelling of native Hazcam scenes (operational v6).

render  <source_id> : SLIC superpixels (deterministic) drawn and numbered on the native image,
                      written as review sheets for visual classification.
build              : turns the reviewed superpixel classes in
                      data/annotations/hazcam-operational-v6-superpixels.json into native label masks
                      and writes manifest-operational-v6.json (v5 entries + new v6 scenes).

Every superpixel listed in the JSON was looked at and classified by eye; unlisted superpixels stay
unknown (255). A 10-native-pixel collar between different classes is ignored, as in v5.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
from skimage.segmentation import slic, find_boundaries

ROOT = Path(__file__).resolve().parents[1]
TRAINING = Path(os.environ["LOCALAPPDATA"]) / "mars-pathfinder" / "training"
NATIVE = TRAINING / "manual-native"
REVIEW = ROOT / "data" / "annotations" / "hazcam-operational-v6-superpixels.json"
PARAMS = {"n_segments": 260, "compactness": 0.08, "sigma": 1.5, "start_label": 1}


def segments(image: Image.Image) -> np.ndarray:
    gray = np.asarray(image.convert("L"), np.float64) / 255
    return slic(gray, channel_axis=None, **PARAMS)


def render(source_id: str, out_dir: Path):
    image = Image.open(NATIVE / "images" / f"{source_id}.png").convert("RGB")
    seg = segments(image)
    out_dir.mkdir(parents=True, exist_ok=True)
    rgb = np.asarray(image).copy()
    rgb[find_boundaries(seg, mode="inner")] = [255, 210, 0]
    sheet = Image.fromarray(rgb)
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 15)
    for label, sl in enumerate(ndimage.find_objects(seg), start=1):
        if sl is None:
            continue
        rr, cc = np.nonzero(seg[sl] == label)
        y, x = int(rr.mean()) + sl[0].start, int(cc.mean()) + sl[1].start
        draw.text((x, y), str(label), font=font, fill="white", anchor="mm", stroke_width=3, stroke_fill="black")
    sheet.save(out_dir / f"{source_id}-numbered.png")
    grid = image.copy(); g = ImageDraw.Draw(grid); small = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 12)
    for v in range(64, 1024, 64):
        g.line([(v, 0), (v, 1023)], fill=(0, 200, 255), width=1); g.line([(0, v), (1023, v)], fill=(0, 200, 255), width=1)
        for (x, y) in ((v + 2, 2), (v + 2, 1008), (2, v + 2), (994, v + 2)):
            g.text((x, y), str(v), font=small, fill=(0, 230, 255), stroke_width=2, stroke_fill="black")
    grid.save(out_dir / f"{source_id}-grid.png")
    print(source_id, int(seg.max()), "superpixels")


def build():
    review = json.loads(REVIEW.read_text("utf-8"))
    v5 = json.loads((TRAINING / "manifest-operational-v5.json").read_text("utf-8"))
    sources = {s["id"]: s for s in json.loads((NATIVE / "sources.json").read_text("utf-8"))}
    out_dir = NATIVE / "operational-v6-labels"; out_dir.mkdir(exist_ok=True)
    entries = list(v5["entries"])
    for scene in review["scenes"]:
        sid = scene["id"]; src = sources[sid]
        image = Image.open(src["image"]).convert("RGB")
        assert hashlib.sha256(Path(src["image"]).read_bytes()).hexdigest() == src["image_sha256"]
        seg = segments(image)
        target = np.full(seg.shape, 255, np.uint8)
        for class_id in ("0", "3", "2", "1"):
            ids = scene["classes"].get(class_id, [])
            target[np.isin(seg, ids)] = int(class_id)
        for poly in scene.get("polygons", []):
            canvas = Image.new("L", (seg.shape[1], seg.shape[0]), 0)
            ImageDraw.Draw(canvas).polygon([tuple(p) for p in poly["polygon"]], fill=1)
            target[np.asarray(canvas, bool)] = poly["class_id"]
        collar = np.zeros(seg.shape, bool)
        for c in (0, 1, 2, 3):
            m = target == c
            if m.any():
                collar |= ndimage.binary_dilation(m, iterations=10) & ~m & (target != 255)
        target[collar] = 255
        out = out_dir / f"{sid}.png"
        Image.fromarray(target).save(out)
        counts = {str(c): int((target == c).sum()) for c in (0, 1, 2, 3)}
        regions = []
        for c in (0, 1, 2, 3):
            lab, _ = ndimage.label(target == c)
            for sl in ndimage.find_objects(lab):
                if sl is None or (sl[0].stop - sl[0].start) * (sl[1].stop - sl[1].start) < 64:
                    continue
                regions.append({"class_id": c, "bbox": [sl[1].start, sl[0].start, sl[1].stop - 1, sl[0].stop - 1]})
        entries.append({"source_id": sid, "image": src["image"], "label": str(out), "group": f"manual-sol-{scene['sol']}",
                        "split": scene["split"], "sol": scene["sol"], "source_sha256": src["image_sha256"],
                        "label_sha256": hashlib.sha256(out.read_bytes()).hexdigest(), "pixel_counts": counts, "regions": regions,
                        "annotation": "operational-v6 superpixel review"})
        print(sid, scene["split"], counts)
    manifest = {**{k: v for k, v in v5.items() if k != "entries"}, "version": 7,
                "protocol": v5["protocol"] + " Version6: ten more native scenes (7 train, 3 new test_v6) labelled by "
                "visually classifying deterministic SLIC superpixels; roles fixed before predictions "
                "(data/annotations/operational-v6-review-protocol.json).", "entries": entries}
    (TRAINING / "manifest-operational-v7.json").write_text(json.dumps(manifest, indent=1), "utf-8")


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("render"); r.add_argument("ids", nargs="+"); r.add_argument("--out", type=Path, required=True)
    sub.add_parser("build")
    args = ap.parse_args()
    if args.cmd == "render":
        for sid in args.ids:
            render(sid, args.out)
    else:
        build()


if __name__ == "__main__":
    main()
