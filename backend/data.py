import json
import os
from pathlib import Path
import numpy as np
from .navigation import terrain_analysis

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("MARS_DATA_DIR") or ROOT / "data" / "prepared")


class Region:
    def __init__(self, path: Path):
        self.path = path
        self.meta = json.loads((path / "metadata.json").read_text("utf-8"))
        self.elevation = np.load(path / "elevation.npy", mmap_mode="r")
        if max(self.elevation.shape) > 512:
            raise ValueError("Region exceeds bounded 512-cell planning extent")
        self.analysis = terrain_analysis(self.elevation, self.meta["cell_m"])


regions: dict[str, Region] = {}


def load_regions():
    regions.clear()
    for path in DATA_DIR.glob("*/metadata.json"):
        region = Region(path.parent)
        regions[region.meta["id"]] = region


def frame_catalog() -> list[dict]:
    path = DATA_DIR / "frames.json"
    return json.loads(path.read_text("utf-8")) if path.exists() else []


def model_status() -> dict:
    path = DATA_DIR / "model-card.json"
    return json.loads(path.read_text("utf-8")) if path.exists() else {
        "ready": False, "name":"LR-ASPP · MobileNetV3", "status":"Training checkpoint not installed",
        "classes":["soil","bedrock","sand","big_rock"], "metrics":None}
