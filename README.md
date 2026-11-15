# Mars Pathfinder AI

Global Mars exploration and terrain-aware mission planning with real NASA/USGS/PDS data.
Research prototype for the Martian Map challenge. **Not certified for human EVA operations.**

Latest perception audit (15 November2026): the reviewed operational Hazcam
dataset now has86 images, including33 additional PDS exposures and117 newly
traced rock-instance polygons. Fine-tuned challengers were tested but rejected
for losing previously found large objects; the deployed head remainsv11.

## Real-footage navigation

The default view is now the actual forward-facing rover camera on the left and
the same remaining route on the Mars map on the right. A 3D simulation is optional.
The camera defaults to a **rover-centred steering guide**: every cue begins at
the visible bottom-centre rover marker. Its shape and left/right/U-turn instruction
come from the remaining map route relative to the rover's archived heading. This
is a directional HUD, not a measured ground trace. **Ground projection** switches
to the original calibrated camera projection without moving its measured pixels.

**Recorded motion** is computed independently from consecutive PLACES poses:
translation in metres and the wrapped heading change in degrees. It distinguishes
forward/backward movement, left/right heading changes, rotation in place and an
unchanged position (25 cm /5° deadbands). These are changes between archived
acquisitions, not an assertion that the recorded rover followed our suggested route.
The default extended traverse contains **142 actual Front Hazcam acquisitions**
from sols40–150: about459 m of recorded movement, interspersed with stationary
fieldwork and turns. Its142 steps contain34 positional transitions over25 cm
and19 computed route updates. Two short diagnostic scenarios use sols50,52,53,55:
an approach to Rocknest, and a missed-waypoint return task.
The latter is explicitly counterfactual: recorded footage never changes in response
to a suggested U-turn. The planner re-routes on the server; it does not control a rover.

The camera projection uses the PDS-labelled CAHVORE model, rover-frame quaternion,
stereo XYZ and native DEM. Unfitted checks on142 actual Hazcam products reproduce
original stereo pixel positions with p95 error0.14–0.37 pixels. This validates
camera projection, **not absolute orbital localization**. The four featured poses
use archival PLACES placement; they did not pass the independent image-correlation
gate and never add obstacle cells. Distant route heights are DEM estimates aligned
vertically to nearby stereo, with stereo/DEM occlusion checks. Out-of-view routes
produce direction guidance instead of mirrored ground lines.
That calibration applies to Ground projection; it does not claim pixel accuracy
for the intentionally screen-anchored Centre guide.

Every featured actual acquisition has a visible forward route in the approach
scenario. At the second acquisition the recorded rover is 4.05 m off the initial
route; the server computes v2. In the return task later views correctly show a
U-turn and the route behind the camera on the map. Traversed route sections are
removed, and image overlays wait for the matching image to load.

A downloadable **two-minute142-acquisition route film**, plus the two30-second
diagnostic MP4s, are included in `data/prepared/demos/` and served
at `/api/scenarios/{scenario_id}/demo.mp4`. These are labelled accelerated
**still-image archive timelapses**, not continuous video or live camera feeds.
The interactive app computes its own results; it does not play the MP4 as live inference.
Reproduce the data and videos with:

```powershell
python scripts/prepare_navcam_set.py --front-hazcam --sequence hazcam --manifest data/prepared/hazcam-extended-manifest.json
python scripts/prepare_camera_models.py
python scripts/register_image_correlation.py --only-new --write
python scripts/build_scenarios.py
python scripts/export_replay_video.py hazcam-rocknest-approach
python scripts/export_replay_video.py hazcam-return-waypoint
python scripts/export_replay_video.py hazcam-extended-traverse --duration 120
```

The exact142-product manifest is reproducible; `--per-sol 1` can bound other
individual preparation runs. The full catalogue contains65 Navcam and142 Hazcam
products (**207 total**);21 pass the
image-registration gate. Each image, geometry array, calibration and localization
keeps its own source and hashes.

The sidebar also plays two **full original NASA films**, streamed directly from
NASA: Perseverance's175 m Sol200 drive (28.8 s,192 distinct decoded frames), and
Curiosity's five-year Hazcam film (5 min49 s, including NASA's inset location map).
These films retain their original content; the app does not claim per-frame
planner registration for them.

## Implemented

- React/TypeScript/Cesium global Mars, streamed real Viking imagery and MOLA/HRSC terrain.
- NASA Mars Trek optical, MOLA visualization and THEMIS infrared layers.
- A real 512×512 native **1 m** HiRISE/PLACES DEM and **25 cm** ortho crop at Gale/Rocknest.
- Distance and terrain-aware A*, route profiles, waypoints, comparisons and JSON reports.
- **Planning anywhere on Mars** (coarse): the server fetches the real MOLA/HRSC 200 m blended DEM
  tiles for the waypoints, builds a ≤384×384 grid (≥163 m cells) and runs the same A* live. This is
  regional context; cells are hundreds of metres wide and contain no rock-level information.
- **Feature-aware routes**: every route lists the IAU-named features it crosses or passes near, and any
  named feature can be avoided (blocked as a circle of its published diameter for the terrain-aware
  routes; the distance baseline stays unconstrained). Example: a 454 km Tharsis traverse climbs 8.8 km
  over Pavonis Mons; avoiding it gives 577 km with 3.9 km of ascent. Circular footprints are an
  approximation of real outlines.
- **Relief view (Explore / Plan)**: an Imagery ⇄ Relief switch turns on the MOLA colour shaded relief
  with Cesium vertical exaggeration ×12, so volcanoes rise and craters and canyons sink, and tilts the
  camera obliquely. A planned route is drawn at its own MOLA/HRSC profile height × 12 (+1.8 km lift) so
  it follows the exaggerated terrain instead of being buried; the legend states the colour scale and
  that the route is the lowest slope + roughness cost on the same 200 m terrain.
- **2055 IAU-approved feature names** (USGS Gazetteer of Planetary Nomenclature) on the globe, with
  type and diameter, searchable; labels appear by feature size.
- **Route drawn inside the rover camera image**: each route sample is placed at the image position of
  the nearest measured stereo ground pixel (via the registered transform); occluded or out-of-view
  ground leaves gaps. The previous route is shown dashed after a camera-driven replan.
- 30 matched Sol 57 PDS Navcam RAD/XYZ/XYE products (drive sequence + Rocknest panorama), playable
  as a frame sequence; corrected PLACES rover transforms. 13 pass the registration test (see below).
- Real LR-ASPP/MobileNetV3 Surface training, ONNX CPU inference, live SSE algorithm events.
- Camera/orbital registration measured by image correlation against the 25 cm HiRISE ortho
  (`scripts/register_image_correlation.py`): all three frames match at the archived placement with a
  0.35–0.75 m shift and empirical p ≤ 0.0025 against 2000 random placements, across four fixed
  filter settings. Evidence JSON with input hashes is stored per frame.
- Live fusion and replanning: map obstacles are **measured stereo height steps** (≥ 0.15 m above
  local ground, XYE ≤ 0.10 m), placed through the registered transform and inflated by clearance +
  registration bound. The learned big-rock class is shown but never written to the map.

## Run locally

Node24+, pnpm11+, Python3.12. Use a Python virtual environment.

```powershell
pnpm install
python -m pip install -r requirements.txt
pnpm build
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000. Development: run the backend on8000 and `pnpm dev` on5173.
No Cesium token is required: the open-data renderer uses real elevation. An optional
domain-scoped `VITE_CESIUM_ION_TOKEN` enables the Cesium Mars tileset when built with that variable.

On this Windows workspace, `scripts/start.ps1` uses the prepared local runtime.

## Reproduce data and training

```powershell
python -m pip install -r requirements-dev.txt
python scripts/prepare_pds.py --frames 3 --ortho-crop --dem-crop --region-ortho
python -m scripts.assemble_data
python scripts/fetch_training.py --count 2000 --eval-count 400 --bigrock-train-fraction 0.6 --group-cap 24
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python scripts/train_surface.py --epochs 10
python -m pytest tests -q
```

Raw downloads and training images default to `%LOCALAPPDATA%/mars-pathfinder/`, outside OneDrive.
Set `MARS_DATA_DIR` for another prepared-data location. CUDA training is optional; the published
application uses ONNX Runtime CPU and does not include PyTorch or GDAL.

The AI4Mars reader validates HTTP Range responses and downloads only selected original-publisher
ZIP members. Splits use three-sol acquisition blocks, with adjacent demo observations excluded.
The held-out test subset uses crowdsourced labels, **not expert gold labels**. See model-card.json
for the actual subset sizes, measured metrics and known limitations. The current model (v3) continues
v2 (2000 train images, big-rock-rich images preferred in train only) for one more selected epoch at
0.3× learning rate, chosen on validation (mIoU + big-rock IoU). On the same unbiased 400-image held-out
test: **66.5% mIoU**, big-rock IoU 0.21, recall 43%, precision 30% (v2: 64.7% mIoU, big-rock IoU 0.20,
recall 36%; v1: 59.6% mIoU, recall 14%). All four classes improved.

**Stereo self-labelling experiment (negative result, kept for transparency).** 303 raised, grounded
stereo components on our 207 MSL Hazcam/Navcam frames were shown as numbered crops and each was
visually classified by the AI coding assistant used on this project: 141 accepted as loose rocks; the
rover's own arm turret, wheels, deck and mast, layered bedrock slabs and ambiguous bumps rejected
(`scripts/stereo_pseudo_labels.py`, `data/training/pseudo-rock-review.json`; sol-split, 25% held out).
Fine-tuning with these labels raised overall mIoU (67.0–67.8%) but **lowered big-rock IoU to
0.14–0.15** on the human-labelled test and on held-out sols, so the published model does not use them.
A same-settings control without them is the published v3.
Big-rock detection remains unreliable and never writes the map.

## Scientific limits

Global terrain is coarse context, not local obstacle evidence. Display elevation is approximate
relative to the rendering sphere; scientific profiles retain source areoid heights. THEMIS mosaic
brightness is not current temperature. Big-rock scores are not occupancy probabilities.

Selected camera positions are interpolated archive localizations. Using Site4 alone caused about
7.3m horizontal drift; the supplied transformations correct the rover anchor. Image-correlation
residuals of the 13 accepted frames are 0–0.75 m; the planner uses max shift + 0.5 m (0.75–1.25 m).
This is an empirical research tolerance, not a confidence interval. Two panorama frames failed
(2.5–8 m, p ≥ 0.29) and 15 had too little dense ground inside the HiRISE region to measure; they are
shown but never update the map. Frames overlap and are not independent; PLACES localizations were themselves made against the HiRISE base map. Manual
control points were not usable: rocks here are 10–40 cm, 1–2 HiRISE pixels. Stereo-vs-DEM
elevations differ around1.1m, so absolute vertical agreement is not claimed; DEM-shape matching
was uninformative on this flat terrain (0.4–0.6 m relief). Stereo obstacles can miss low or
distant rocks; they are not a hazard-safety guarantee.

Orbital: route costs remain deterministic DEM calculations. A learned orbital classifier
(MobileNetV3-Small, HiRISE labeled landmarks v3.2, Wagstaff et al., CC BY 4.0, published split by
source image) reaches 92.9% accuracy / 69.5% macro-F1 on 1793 held-out originals, with 0% spider and
57% impact-ejecta recall. It runs live on CTX (~5 m) windows along any route; on 131 IAU-named craters
in CTX it says "crater" 65.6% of the time versus 8.4% three diameters away. It is landmark context,
not a hazard map, and does not change route costs. Real EVA consumables, D*Lite and arbitrary-photo
localization are not implemented.

## Deploy

**Live demo: https://mars-pathfinder-ai.onrender.com** (Render Free, Frankfurt, Docker).

Dockerfile and render.yaml define the service; no paid service is enabled. The free instance sleeps
when idle, so the first request can take about a minute (cold start). Session state is in memory
and is lost on restart or sleep; export the JSON report to keep results.

Verified on 14 November 2026 against the public URL from an external client: mission creation,
planning of the three routes on the server (about 8.8 s wall time including queueing/SSE), live SSE
events, real ONNX CPU inference on two archived Navcam frames (server-reported 401 ms first call,
244 ms second), `localization.rejected` with map version unchanged, duplicate-frame 409, overlay PNGs
and the JSON report. These are single-run Render Free measurements, not benchmarks.

## Sources and attribution

- [NASA Mars Trek](https://trek.nasa.gov/mars/) — challenge Resources source and science layers.
- [MSL PLACES](https://planetarydata.jpl.nasa.gov/w10n/msl/msl_places/) — imagery, DEM and localization.
- [MSL Navcam PDS](https://pds.nasa.gov/ds-view/pds/viewProfile.jsp?dsid=MSL-M-NAVCAM-5-RDR-V1.0) — RAD/XYZ/XYE.
- [AI4Mars v0.6](https://zenodo.org/records/15995036), Swan, Ono & Atha,2022 — CC BY4.0.
- [Global DEM service](https://astro.arcgis.com/arcgis/rest/services/OnMars/MDEM200M/ImageServer) — NASA/USGS/ESA/DLR/FU Berlin; Esri hosting.
- CesiumJS (Apache2.0), Torchvision (BSD3-Clause), ONNX Runtime (MIT), React (MIT), FastAPI (MIT).
