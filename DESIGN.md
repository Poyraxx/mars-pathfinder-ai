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
