"""Domain-transfer check of the HiRISE-trained orbital classifier on global CTX imagery.

Independent labels: IAU-approved crater names (USGS Gazetteer) with diameters 1-30 km
and |lat| <= 60. Each crater is imaged as a CTX window 1.4 x its diameter centred on the
published centre (the training crops are landmark bounding boxes plus a margin).
Offset windows 3 diameters east give a reference rate of 'crater' predictions away from
the named crater (other unnamed craters may legitimately appear there, so this is not a
precision estimate). Deterministic sample; results are written to the model card.
"""
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend import ctx, orbital  # noqa: E402

N = 150


def main():
    feats = json.loads((ROOT / "data/prepared/global/nomenclature.json").read_text("utf-8"))["features"]
    craters = [f for f in feats if f["type"] == "Crater" and 1 <= f["diameter_km"] <= 30 and abs(f["lat"]) <= 60]
    craters.sort(key=lambda f: hashlib.sha256(f["name"].encode()).hexdigest())
    sample = craters[:N]
    size = orbital.card()["input_size"]; classes = orbital.card()["classes"]
    on, off, used = [], [], []
    for f in sample:
        d = f["diameter_km"] * 1000
        a, ia = ctx.window(f["lat"], f["lon"], 1.4 * d, size)
        dlon = 3 * d / (ctx.R * 3.14159265 / 180 * max(0.2, __import__("math").cos(__import__("math").radians(f["lat"]))))
        b, ib = ctx.window(f["lat"], ((f["lon"] + dlon + 180) % 360) - 180, 1.4 * d, size)
        if a is None or b is None:
            continue
        on.append(a); off.append(b); used.append(f["name"])
        print(len(used), f["name"], flush=True)
    p_on = orbital.classify(on).argmax(1); p_off = orbital.classify(off).argmax(1)
    ci = classes.index("crater")
    dist = lambda p: {c: int((p == i).sum()) for i, c in enumerate(classes)}
    result = {"labels": "IAU-approved crater centres and diameters (USGS Gazetteer), 1-30 km, |lat|<=60",
              "imagery": ctx.SOURCE, "window": "1.4 x crater diameter, centred", "evaluated": len(used), "requested": N,
              "crater_recall_on_named_craters": round(float((p_on == ci).mean()), 4),
              "crater_rate_on_offset_windows": round(float((p_off == ci).mean()), 4),
              "prediction_distribution_on_craters": dist(p_on), "prediction_distribution_offset": dist(p_off),
              "note": "Transfer check on a different camera (CTX) than training (HiRISE). Offset rate is a reference, not precision."}
    path = ROOT / "data/prepared/orbital-model-card.json"
    card = json.loads(path.read_text("utf-8")); card["ctx_transfer_check"] = result
    path.write_text(json.dumps(card, indent=2), "utf-8")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
