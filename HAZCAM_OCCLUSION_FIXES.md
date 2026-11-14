# Hazcam equipment and close-goal corrections — 15 November 2026

Follow-up to `380601d`. Reproduces all seven user screenshots in the extended
traverse: indices 120, 121, 129, 130, 139, 140 and 141 (zero based).

## Causes and changes

- Raw XYZ on the arm was included in `camera_route.ground_pixels`. At the same
  horizontal destination, its raised return could lift the destination onto the
  arm. The visibility cache now uses the existing navigation head's non-terrain
  mask, including a one-geometry-pixel edge exclusion, before selecting stereo
  height support. No hidden ground is filled by that mask.
- The same mask rejects route/goal projections on equipment, sky or unknown
  pixels, even when those pixels have no usable stereo depth. It also clips
  displayed route links and obstacle markers. An actual occluded goal has no
  flag; the visibility reason is carried in the API.
- Sol150 frames 139–141 previously moved the flag from normal ground onto the
  moving arm: y=0.10326 / 0.16608 / 0.16992. Filtering height support keeps it on
  measured terrain at y=0.41038 / 0.40946 / 0.41083. This is a geometry regression
  check on recorded fixtures, not a held-out model accuracy result.
- Their map path has only 1.21 m of projected remaining arc, despite a 2.0 m
  navigation readout. The display's fixed two-metre cutoff hid every point.
  Short plans retain their forward portion, with an entirely planned thin HUD
  link. That does not certify rover clearance. The longer near/map handoff and
  the user's good frames 129–130 keep their existing behavior.
- U-turn swing segments with no forward map handoff previously skipped paint
  tapering and ended as a broad chopped strip. Those swings now taper without
  inventing a visible return route behind the camera.
- The large orange ellipse in frame121 was explicitly classified as
  `large_raised_region`, not a discrete rock. Its conservative lookahead cost is
  retained, but it is no longer drawn as a single rock outline.
- Remove the duplicate large centre-screen failure box. Guidance remains in the
  existing readout and its concrete reason stays in the source caption. A
  clearance conflict is labelled unverified, rather than presented as missing
  data or a generic error. Arrival tolerance and obstacle clearance do not change.

## Actual clearance limit

The late Sol150 view contains real nearby rocks. In frame139, the conservative
start-cell clearance is about 1.612 m; the required body-plus-margin/grid bound is
about 1.621 m. It therefore remains `no_path` with `failure_kind=clearance_conflict`.
No rock is deleted, no footprint/clearance is shrunk and no measured-safe path is
asserted to satisfy a visual preference. Its visible thin map plan is labelled
`Map plan · clearance unverified`.

## Files and verification

Core files: `backend/camera_visibility.py`, `backend/camera_route.py`,
`backend/replay.py`, `backend/local_navigation.py`, `src/routeGeometry.ts`,
`src/RouteOverlay.tsx`, `src/Replay.tsx`, `src/types.ts`, the shared geometry bridge,
and the video exporter. All three downloadable films are regenerated.

89 Python tests passed, including hardware-without-depth occlusion, height
support exclusion, all-unknown ground, U-turn tapering, close-goal context and
the three actual Sol150 frames. Existing Cesium annotation persistence is
unchanged. TypeScript/Vite build and browser checks cover the seven supplied
frames. Local/public records and screenshots are in ignored `artifacts/occlusion-*`.

The navigation model remains v11. Panel layout, styles, global planning and the
previous zoom/negative-datum correction stay intact.
