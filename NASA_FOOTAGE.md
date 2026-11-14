# Real rover footage sources

## Calibrated interactive traverse

The default `hazcam-extended-traverse` uses142 actual Curiosity Front Left Hazcam
acquisitions from sols40–150, in their original acquisition order. The142-product
manifest is `data/prepared/hazcam-extended-manifest.json`. Each product's RAD, XYZ,
XYE and label URLs and SHA256 hashes are retained in `data/prepared/frames.json`.

Source archive: https://planetarydata.jpl.nasa.gov/img/data/msl/MSLHAZ_1XXX/DATA/
Pose source: https://planetarydata.jpl.nasa.gov/w10n/msl/msl_places/

All142 labelled CAHVORE models passed unfitted stereo-pixel reprojection checks;
per-frame p95 errors range0.1446–0.3676 px. This does not independently validate
absolute orbital localization. Only frames with passing registration evidence can
contribute obstacles; no registration gate was relaxed for the larger replay.

The recorded track totals459.0 m;34 successive image pairs move more than25 cm.
Other pairs are stationary science observations or turns, not invented drive steps.
The server computes19 route updates against the fixed destination. A route can be
behind or outside the camera field of view; the UI then shows direction guidance.

The exported film is120 seconds,1600×900,24 fps H.264. All142 real acquisitions
appear without synthetic motion. The app computes replay routes live, independently
of that MP4. The MP4 has its own source manifest and SHA256 provenance JSON.

Following the user's centre-origin correction, exported films and the default
camera view use a bottom-centre directional HUD. This is computed from the same
map route and rover heading; it is not a calibrated ground projection. The original
ground projection remains selectable in the interactive viewer. All cues start at
normalized[0.5,0.94]. Recorded motion is a separate pose-difference algorithm:
25 cm translation deadband,5° heading deadband, signed angles wrapped across0/360°.
It can report stationary fieldwork while the planner recommends a turn. Neither
the guide nor a motion label changes archived camera pixels or map obstacle gates.

## NASA original films

Both films are streamed from NASA's official asset host. The application preserves
the original image content and uses native browser play/seek/fullscreen controls.
No app-calculated route is asserted on these films without per-frame camera/pose
registration. NASA's own overlays remain intact.

### Perseverance — Sol200 AutoNav drive

- Official source: https://science.nasa.gov/photojournal/perseverances-sol-200-auto-nav-drive/
- Original MP4: https://assets.science.nasa.gov/content/dam/science/psd/photojournal/pia/pia24/pia24934/PIA24934.mp4
- NASA/JPL-Caltech;12 September2021;175 m total drive,167 m autonomous portion.
- Verified locally:1280×960,28.8 seconds,192 decoded frames,192 unique decoded
  pixel hashes. The source is accelerated about200×, about30 seconds per acquisition.

### Curiosity — five years of rover POV

- Official source: https://science.nasa.gov/resource/rover-pov-five-years-of-curiosity-on-mars/
-720p stream: https://assets.science.nasa.gov/content/dam/science/psd/mars/downloadable_items/3/2/32074_RoverPOVFiveYearsOfCuriosityOnMars-1280.m4v
- NASA/JPL-Caltech;2012–2017 Hazcam imagery, date/sol annotations and inset map.
- Browser-verified duration348.832 seconds (5 min49 s). Stationary work and turns
  are retained. Native video plays/seeks on desktop and mobile.

Source research and verification:14 November2026. Media are NASA-hosted; archived
PDS pixels retain their original mission attribution. No claim of NASA endorsement.
