# Guidance continuity and Live GPS Test — 15 November 2026

Follow-up to `9bcf558`. The user asked for a coherent obstacle bypass instead of
right/left/right-looking guidance, and a clearer product entry point.

## Interface

Navigation order is Plan a route, Live GPS Test, Explore. A fresh page starts in
Plan a route. The old NASA drive-film chooser, player, state branches and its
exclusive styles are removed. Calculated test-drive videos remain downloadable.
Returning from the close rover view to Explore (or global route planning)
explicitly restores the planetary camera overview, instead of leaving the map
at the preceding replay zoom.

Live GPS Test explains the actual pipeline: AI identifies terrain/rocks, calibrated
stereo checks local clearance, and archived NASA rover poses place the camera on
the map. This does not claim that the model independently recovers a GPS position
from a single image. Recorded acquisitions and build/live calculation timing
remain visible in the existing source captions.

## Route behavior

- The image-space connection from the rover-centred origin to observed/projected
  ground is a direct bearing. Its former tangent-fitting cubic could overshoot
  the first ground point and suggest an opposite-side sweep before the real
  obstacle bypass. It is still a display link, not an extra clearance observation.
- Near/far display handoff control points are bounded between their endpoints,
  so that join cannot double back or add a new lateral extremum.
- U-turn choice has rear-sector hysteresis. In the rear 90-degree sector, an
  ongoing turn keeps its side when this frame still validates a local swing.
  If that side closes, the planner tests the other side against the same measured
  terrain. It releases the preference when the target moves materially away from
  the rear sector. Signed remaining turn angles follow the chosen direction.
- In the reported extended drive, the formerly right/left pair at indices120–121
  now stays left/left (211° then167° remaining). The actual recorded leftward
  heading change therefore advances the same indicated turn instead of reversing
  the instruction. This validates the near swing only, not unseen ground behind.
- For nearly stationary successive poses (<5 cm and2°), a prior bypass may be
  retained only with identical endpoints, <=10%+10 cm length overhead, and fresh
  collision validation of every segment. New obstacles veto history. The current
  default drive does not exercise this optional reuse branch; controlled synthetic
  tests verify both retention and rejection.

94 Python tests passed, including180° wrap, blocked preferred exits, fresh
collision validation of a previous bypass, and monotone display links. The
production build passed. Existing obstacle thresholds, measured-space policy,
equipment mask, ground destination flags and Cesium datum correction are intact.
Local/public verification files and browser screenshots are in `artifacts/guidance-*`.
