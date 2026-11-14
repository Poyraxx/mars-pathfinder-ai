# Mars Pathfinder AI

Global Mars exploration and terrain-aware mission planning with real NASA/USGS/PDS data.
Research prototype for the Martian Map challenge. **Not certified for human EVA operations.**

## Implemented

- React/TypeScript/Cesium global Mars, streamed real Viking imagery and MOLA/HRSC terrain.
- NASA Mars Trek optical, MOLA visualization and THEMIS infrared layers.
- A real 512×512 native **1 m** HiRISE/PLACES DEM and **25 cm** ortho crop at Gale/Rocknest.
- Distance and terrain-aware A*, route profiles, waypoints, comparisons and JSON reports.
- Three matched Sol57 PDS Navcam RAD, XYZ and XYE products; corrected PLACES rover transforms.
- Real LR-ASPP/MobileNetV3 Surface training, ONNX CPU inference, live SSE algorithm events.
- Conservative fusion/replanning pipeline, **disabled for the supplied frames until independent
  camera/orbital control-point residuals are validated**. Metadata consistency alone is insufficient.

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
python scripts/fetch_training.py --count 400
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
for the actual subset sizes, measured metrics and known limitations. The held-out ONNX test
reached 55.8% mIoU but **0% big-rock recall**; hazard detection remains unvalidated.

## Scientific limits

Global terrain is coarse context, not local obstacle evidence. Display elevation is approximate
relative to the rendering sphere; scientific profiles retain source areoid heights. THEMIS mosaic
brightness is not current temperature. Big-rock scores are not occupancy probabilities.

Selected camera positions are interpolated archive localizations. Using Site4 alone caused about
7.3m horizontal drift; the supplied transformations correct the rover anchor. Independent
horizontal residuals remain unmeasured. Stereo-vs-DEM elevations differ around1.1m for the first
frame, so absolute vertical agreement is not claimed. `scripts/validate_registration.py` accepts
at least six independently identified real camera/orbital feature pairs and enables a frame only
when empirical residuals meet the2m research tolerance. No such controls are fabricated here.

The Orbital component currently performs deterministic DEM analysis. A second trained orbital
model, real EVA consumables, D*Lite and arbitrary-photo localization are not implemented.

## Deploy

Dockerfile and render.yaml target a public Render Free Python/ONNX service. Connect this repository
to Render and create its Blueprint. No paid service is enabled. The service may sleep, cold-start,
or lose session state; export reports to preserve them. A public deployment is not asserted until
a real accessible URL has been verified.

## Sources and attribution

- [NASA Mars Trek](https://trek.nasa.gov/mars/) — challenge Resources source and science layers.
- [MSL PLACES](https://planetarydata.jpl.nasa.gov/w10n/msl/msl_places/) — imagery, DEM and localization.
- [MSL Navcam PDS](https://pds.nasa.gov/ds-view/pds/viewProfile.jsp?dsid=MSL-M-NAVCAM-5-RDR-V1.0) — RAD/XYZ/XYE.
- [AI4Mars v0.6](https://zenodo.org/records/15995036), Swan, Ono & Atha,2022 — CC BY4.0.
- [Global DEM service](https://astro.arcgis.com/arcgis/rest/services/OnMars/MDEM200M/ImageServer) — NASA/USGS/ESA/DLR/FU Berlin; Esri hosting.
- CesiumJS (Apache2.0), Torchvision (BSD3-Clause), ONNX Runtime (MIT), React (MIT), FastAPI (MIT).

Original implementation assisted by OpenAI Codex. This is an AI tool disclosure, not a contributor
identity. No NASA endorsement is implied. Data retain their individual provenance and rights;
the source code is Apache2.0. No NASA logos or mission insignia are used.
