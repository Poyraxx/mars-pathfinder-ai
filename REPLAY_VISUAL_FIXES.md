# Replay visual corrections — 15 November 2026

Follow-up to `4857937`: the user reported an abrupt middle-to-far taper,
an unclear visible destination and replay annotations disappearing during zoom
or stationary frame changes. Existing panels, styles and navigation models stay intact.

- `src/routeGeometry.ts`: distribute the paint-width transition along the entire
  near approach, instead of squeezing it into the last 0.18 image arc. Preserve
  the far width, measured centreline and rover clearance. Shared video geometry
  gets the same change.
- `backend/camera_route.py`, `backend/replay.py`, `src/types.ts`: project the exact
  requested destination using the accepted PDS camera model and the existing
  stereo/DEM height and occlusion checks. Return `destination: null` when its
  actual endpoint is not visible. The last local waypoint/visible fragment is
  never silently promoted to a destination.
- `src/RouteOverlay.tsx`, `src/Replay.tsx`: a small amber flag and Destination
  label identify that projected goal. An unseen/occluded goal gets no flag.
  Archived map placement retains its documented uncertainty.
- `src/replayScene.ts`, `src/Globe.tsx`, `src/Replay.tsx`: replace nonconstant
  clamped callbacks and per-frame entity destruction with persistent entities,
  constant properties and value comparisons. Duplicate stationary poses do not
  invalidate line geometry. First-person camera motion requires a real pose change.
- Replay lines sample the existing HiRISE DEM every 0.75 m, with a 0.15 m display
  lift. No new terrain is inferred. Missing DEM coverage keeps the clamped fallback.
  Both replay scenes seed Cesium's central-body occluder with the real loaded
  DEM minimum before render. Otherwise a temporary zero/higher LOD bound can
  classify the negative-elevation Gale annotations as inside the reference globe
  and hide them all. Terrain depth and on-demand rendering remain enabled;
  no heights are moved and far-side horizon occlusion remains effective.
  This guarded `Scene.frameState.minimumTerrainHeight` compatibility hook is
  specific to the installed Cesium 1.147 engine; check it when upgrading Cesium.
- `scripts/export_replay_video.py` and all three films: shared taper plus the
  same calibrated goal flag. Video/source hashes are recorded in their manifests.

Validation: 83 Python tests and 6 Node tests using actual Cesium Entity/Property
classes passed. New coverage includes a full-approach taper, actual goal versus
visible fragment, out-of-view/occluded goals, 120 unchanged update cycles,
stationary duplicate poses, measured DEM heights and missing coverage fallback.
An analytic negative-elevation test uses Cesium's real Occluder: the erroneous
reference sphere hides the near annotation; the measured minimum restores it
through 120 LOD resets without exposing the far side of the globe.
The TypeScript/Vite production build passed. Browser screenshots/checks are in
ignored `artifacts/refinement-*` files.

Cesium also documents that terrain LOD changes can bury surface annotations:
[Globe.depthTestAgainstTerrain](https://cesium.com/learn/cesiumjs/ref-doc/Globe.html#depthTestAgainstTerrain).
Its installed `Scene.js` constructs the central-body occluder from the preceding
frame's minimum before updating terrain; the replay datum hook repairs that bound.
Display annotation visibility is not proof of ground clearance.
