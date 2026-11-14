"""Evaluate one or more Surface ONNX models on the current manifest's held-out test split.

Same protocol as train_surface.py's ONNX evaluation: 320x320 input, labels resized
nearest, ignore 255. Used to compare an older model with a retrained one on the SAME
test images before replacing the published model.
"""
import argparse
import json
import os
import sys
from pathlib import Path

_dll = os.add_dll_directory(os.environ["MARS_DLL_DIR"]) if os.name == "nt" and os.environ.get("MARS_DLL_DIR") else None
import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_surface import MarsDataset, TRAINING, scores  # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("models", nargs="+", type=Path); ap.add_argument("--split", default="test"); ap.add_argument("--manifest", type=Path, help="default: AI4Mars manifest; e.g. manifest-pseudo.json with --split pseudo_test")
    args = ap.parse_args()
    manifest = json.loads((args.manifest or TRAINING / "manifest.json").read_text("utf-8"))
    ds = MarsDataset([e for e in manifest["entries"] if e["split"] == args.split], size=320)
    out = {}
    for path in args.models:
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        conf = np.zeros((4, 4), dtype=np.int64)
        for i in range(len(ds)):
            x, y = ds[i]
            pred = sess.run(None, {"image": x.numpy()[None]})[0].argmax(1)[0]
            t = y.numpy(); v = t != 255
            conf += np.bincount(4 * t[v] + pred[v], minlength=16).reshape(4, 4)
        out[str(path)] = scores(conf)
        print(path, json.dumps({k: out[str(path)][k] for k in ("miou", "class_iou", "recall")}), flush=True)
    print(json.dumps({"split": args.split, "images": len(ds), "results": out}))


if __name__ == "__main__":
    main()
