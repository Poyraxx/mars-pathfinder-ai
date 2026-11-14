# Claude continuation review — 15 November 2026

Reviewed the two Claude sessions after the Codex handoff, including the latest
request about missing corridors and the disconnected thick/thin camera route.
Starting commit: `d99922b`. Claude had left edits in `local_navigation.py` and
`routeGeometry.ts`; their original working copies are preserved in local artifacts.

## Preserved completed work

- Accepted local navigation model v11, separate from the global v3 terrain model.
- Reviewed dataset expansions and the recorded v13/v14 rejection results. A larger
  independent holdout is still needed before another model promotion; the small
  consumed tests cannot establish near-perfect rock detection.
- Build-time default replay caches, live custom destinations, archived motion,
  global map, existing panel layout and calibrated PDS/DEM projections.

## Completed route and tracking changes

- Partial frontier goals are restricted to the rover's connected free component.
  Four-connectivity respects the planner's prohibition on cutting diagonal corners.
  A blocked intermediate map waypoint can return a validated stopping segment;
  an explicitly requested blocked local destination still reports no path.
- The old blanket body exclusion extended20cm beyond the footprint, leaving an
  artificial unobserved moat beside the rover. Exclusion now uses the same
  footprint as the start area. This enables measured side exits without inventing
  a free pivot disc. Clearance remains enforced inside the footprint too.
- The unfinished pivot-circle shortcut was removed: a collision-free rotation
  does not establish that every centre position in its swept circle is drivable.
- A single tapered outline joins the measured near route to a forward portion of
  the projected map plan. Paint tapers before the join; the2.8m clearance envelope
  does not narrow. A dashed centre identifies unverified map context. Actual camera
  exits remain separate, and hidden short intervals are dotted.
- Map context remains visible when no local corridor can be verified. It is labelled
  as a map plan, never as verified free ground. Arrival is distinguished from a
  missing-corridor error. Centre-guide mode may link that plan to the rover as a
  wholly dashed-centre, thin HUD cue; ground-projection mode does not invent a
  measured link. The duplicate centre-screen warning no longer hides a visible
  plan; the route-guidance badge still states that local passage is unverified.
- A remembered rock can survive a semantic-model miss when current stereo confirms
  its matching height and raised relief against nearby ground. A current flat-ground
  measurement clears it. Only new/reconfirmed stereo renews its timestamp; unseen
  records expire after12 elapsed frames, rather than12 subsequent detections.
- Video authoring calls the same pure TypeScript geometry module as the browser,
  through `scripts/route_geometry.mjs`. There is no second Python approximation of
  the near/far join. Video authoring therefore also needs the project's Node runtime.
  Valid sealed default plans are reused rather than needlessly rescanning their frames.

## Validation

Regression coverage includes disconnected frontiers, blocked map waypoints,
measured side exits around a near boulder, unchanged clearance, genuine closed
passages, semantic misses versus physically cleared ground, track expiration,
smooth one-outline joins and map-only context without an invented rover link.

The full Python suite passed79 tests; the production frontend build passed.
Default replay caches and all three route films are regenerated with the same
backend and shared geometry. Their manifests retain source and video hashes.
Browser QA includes the user's Sol54 15:21:49 reference frame. Local/public
verification records and screenshots are kept in ignored `artifacts/`.
The rebuilt local server returned the142-frame batch in0.10s and its repeat in0.16s
in one check, with the computation timestamp unchanged. These are local single-run
measurements, not a promise about cold starts or Render CPU performance.

## Remaining research limit

This finishes the interrupted route/overlay release, not flight qualification or
universal stone recognition. v11 remains the accepted model. The rejected v14 and
its validation/test records remain available for subsequent work; do not silently
promote it or reuse consumed test sets as fresh evidence.
