"""Write data/prepared/scenarios.json: ordered real drive replays built from catalogued frames.

A scenario lists archived mid-drive camera frames in acquisition order. Rover poses come
from each frame's PLACES localization; nothing is interpolated or invented here. The
replay itself (route, deviations, re-routes, camera overlays) is computed live by the API.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "prepared"


def main():
    frames = json.loads((DATA / "frames.json").read_text("utf-8"))
    drive = [f for f in frames if "TRAV" in f["id"] and f.get("region_id") == "gale"]
    drive.sort(key=lambda f: int(f["id"][4:13]))
    scenarios = [{
        "id": "rocknest-approach",
        "title": "Curiosity · approach to Rocknest",
        "subtitle": "Gale Crater · Sols 55–57 · October 2012",
        "mission": "MSL Curiosity", "region_id": "gale", "camera": "Navcam left, mid-drive (TRAV) frames",
        "frames": [f["id"] for f in drive],
        "sols": sorted({f["sol"] for f in drive}),
        "story": "After Bathurst Inlet, Curiosity drove about 23.5 m in short legs to the Rocknest sand patch, "
                 "turning several times, then scuffed the drift on sol 56. The replay plans a route to the rover's "
                 "actual arrival point and re-plans whenever the real rover leaves it.",
        "references": [
            {"label": "JPL: From 'Bathurst Inlet' to 'Rocknest' (mission update)", "url": "https://www.marsdaily.com/reports/From_Bathurst_Inlet_to_Rocknest_999.html"},
            {"label": "NASA: Sol 55 left Navcam vertical projection", "url": "https://science.nasa.gov/resource/sol-0055-left-navigation-camera-vertical-projection"},
            {"label": "JPL video: Rover POV, five years of Curiosity (Hazcam time-lapse with map inset)", "url": "https://www.jpl.nasa.gov/videos/rover-pov-five-years-of-curiosity-on-mars/"},
            {"label": "PDS: MSL Navcam Sol 55 raw products", "url": "https://planetarydata.jpl.nasa.gov/img/data/msl/MSLNAV_1XXX/DATA_V1/SOL00055/"},
        ],
        "note": "Frames are real archived images replayed in order; motion between frames is not interpolated.",
    }]
    (DATA / "scenarios.json").write_text(json.dumps(scenarios, indent=2), "utf-8")
    print(len(scenarios), "scenario,", len(scenarios[0]["frames"]), "frames")


if __name__ == "__main__":
    main()
