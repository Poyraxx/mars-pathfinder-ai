# Pathfinder interface

Surface mode: Operate, with an immersive planetary exploration canvas.

The existing NASA Eyes-inspired mission control direction is retained. The planet is
the primary visual. This is an instrument, not a company marketing page: scientific
coordinates, source captions, historical camera dates and uncertainty notes are required.
Marketing-only Taste Skill section/deal rules do not replace the user's explicit dashboard brief.

Palette: near-black #080d12; panel #0d141a; border #233039; cyan #78c7b9 for orbital/context;
amber #dba776 for camera/attention. Unknown data is stated in words. Error colors have text.
Typography: Manrope display and DM Sans controls, with system fallbacks and tabular numbers.
One line icon family (Lucide), a custom geometric project mark, no NASA logo.

Layout: in replay, a quiet scenario selector, actual camera left and the synchronized
Cesium map right, with transport controls underneath. Camera is the first/default
tab; simulation is named explicitly. The map fits the entire track and route and
offers whole-planet and fit-route controls. Mobile stacks camera and map above
transport and scenario selection; the page may scroll. Plan mode keeps optional
analysis panels and Explore keeps global feature search.

Behavior: route and model results are API-backed. Source imagery is actual PDS/NASA data.
No fabricated terrain illustration is substituted for an analysis result. Controls have
loading/disabled/error states. Reduced motion suppresses decorative animation.

QA: CUA browser at desktop and mobile widths, real WebGL; Python algorithm/API tests;
TypeScript build. Third-party headless browser launchers are not used in this Codex session.

Navigation order: Plan a route, Live GPS Test, Explore. Fresh pages open Plan a
route. Live GPS Test replaces the old Drive replay name and NASA film chooser.
Its sidebar explains AI terrain/rock detection, stereo clearance checks and
GPS-style guidance using NASA's archived rover positions. It presents calculated
test drives and downloadable test videos. The default drive contains142 real
acquisitions. Timeline speed is adjustable from0.5× to4× and waits for the correct
source image before advancing. Source dates, archive pose uncertainty and
calculation timing remain visible. The original NASA film player is removed.

Rover-centred guidance: camera defaults to Centre guide, a solid cyan directional
curve with a visible rover marker at normalized[0.5,0.94]. U-turn cues are amber.
Ground projection is a separate explicit option and keeps the measured image
coordinates. The guide's origin is never hidden under captions: camera controls,
image and source caption occupy separate rows. Recorded motion sits above the
image; route guidance is labelled separately in the distance readout. Motion uses
actual successive pose changes, not the suggested route. Mobile gives the camera
more vertical space and keeps the image-centred origin visible.

UI refinement · 15 November 2026: preserve the mission-control identity and real
canvases. Navigation uses a compact segmented bar, 23 px panel titles and a quieter
selection treatment. The test-drive selector precedes the long explanation; native
disclosures retain the method, drive story, source links and calculation evidence.
Selected drives have both a check and an accessible pressed state. Camera guidance
and distance occupy a dedicated footer row, never a permanent overlay on the image.
The image date remains visible; Geometry & limits opens above its source row.
Warnings remain explicit and amber; measured guidance stays cyan. No route, model,
calibration or clearance thresholds change as part of this visual pass.

Mobile uses a sticky two-row header, camera then map, and a two-row transport bar
with a full-width scrubber. The guidance footer fits above the transport on the
representative phone viewport. At 320 px, mode icons give way to the full labels.
Intermediate widths suppress the connection badge before labels wrap. Reduced
motion is respected, and hover/press feedback is limited to controls and selection.
Route analysis remains accessible on narrow screens as a closable floating panel;
closing it restores focus to its map-toolbar control.
A keyboard skip link reaches the workspace; Sources & limits traps focus, closes
on Escape and restores the invoking control. Existing factual captions and limits
are retained through progressive disclosure, not deleted for visual simplicity.

Validation uses real WebGL in the CUA browser at 1440, 1024, 768, 390 and 320 px.
Taste's marketing hero/deal/section checks do not apply to this operational app;
its static copy lint is supplementary (TypeScript negation operators trigger its
exclamation-mark warning). The Impeccable context launcher returned exit 1 without
output; PRODUCT.md, DESIGN.md, Operate, Polish and Craft Floor were read directly.
