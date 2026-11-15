"""Write data/prepared/scenarios.json: ordered real drive replays built from catalogued frames.

A scenario lists archived mid-drive camera frames in acquisition order. Rover poses come
from each frame's PLACES localization; nothing is interpolated or invented here. The
replay itself (route, deviations, re-routes, camera overlays) is computed live by the API.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "prepared"
sys.path.insert(0,str(ROOT))
from backend.camera_route import view_mission


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
    hazcam = sorted([f for f in frames if f.get('instrument')=='FRONT_HAZCAM_LEFT_A'],key=lambda f:f['acquired_at'])
    if hazcam:
        meta=json.loads((DATA/'gale/metadata.json').read_text('utf-8'))
        approach=[next(f for f in hazcam if f['sol']==sol) for sol in [50,52,53,55]]
        refs=[{'label':'NASA PDS: calibrated Front Hazcam images, XYZ and camera models',
               'url':'https://planetarydata.jpl.nasa.gov/img/data/msl/MSLHAZ_1XXX/DATA/'},
              {'label':'NASA PDS: PLACES rover localizations',
               'url':'https://planetarydata.jpl.nasa.gov/w10n/msl/msl_places/'},
              {'label':'JPL: original Hazcam time-lapse film and map',
               'url':'https://www.jpl.nasa.gov/videos/rover-pov-five-years-of-curiosity-on-mars/'}]
        common={'mission':'MSL Curiosity','region_id':'gale','camera':'Front Left Hazcam A · calibrated CAHVORE',
                'frames':[f['id'] for f in approach],'sols':[50,52,53,55], 'default_view':'navcam',
                'image_kind':'real_archive', 'references':refs,
                'note':'Real PDS images in acquisition order. This is an accelerated archive timelapse, not continuous video or a live camera. No invented frames or interpolated rover motion.'}
        forward={**common,'id':'hazcam-rocknest-approach','title':'Real footage · approach to Rocknest',
                 'subtitle':'Forward-facing camera · Sols 50–55 · ~60 m of recorded motion',
                 'story':'Follow Curiosity through four real forward-facing acquisitions. The cyan route is computed from the current recorded pose to a fixed ground point in the final view. Camera and map share the same remaining route; the server re-plans if the recorded drive leaves it.',
                 'destination':view_mission(approach[-1],meta)['goal']}
        return_trip={**common,'id':'hazcam-return-waypoint','title':'Real footage · missed waypoint',
                     'subtitle':'Same recorded drive · alternative nearby destination',
                     'story':'A counterfactual navigation task on the same real footage: the destination is a measured ground point in the first view. The recorded rover continues its historical drive. The planner recommends a return or U-turn when it passes the requested waypoint; the footage is never altered to pretend it followed that recommendation.',
                     'destination':view_mission(approach[0],meta)['goal']}
        scenarios=[forward,return_trip,*scenarios]
        if len(hazcam)>20:
            extended={**common,'id':'hazcam-extended-traverse','title':'Curiosity · extended field traverse',
                      'subtitle':f"{len(hazcam)} real acquisitions · Sols {hazcam[0]['sol']}–{hazcam[-1]['sol']}",
                      'frames':[f['id'] for f in hazcam],'sols':sorted({f['sol'] for f in hazcam}),
                      'story':f"Follow {len(hazcam)} actual forward-camera acquisitions through Curiosity's approach to Rocknest, fieldwork, turns and departure. Stationary science acquisitions remain in their historical order. Camera and map use each frame's archived pose and the same calculated remaining route; no intermediate rover motion is fabricated.",
                      'destination':view_mission(hazcam[-1],meta)['goal']}
            scenarios=[extended,*scenarios]
    (DATA / "scenarios.json").write_text(json.dumps(scenarios, indent=2), "utf-8")
    print([(s['id'],len(s['frames'])) for s in scenarios])


if __name__ == "__main__":
    main()
