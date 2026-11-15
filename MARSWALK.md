# Marswalk mission layer — 15 November 2026 (Claude)

Implements the six challenge-alignment items, without touching the Hazcam model, planner thresholds or replay.

| Item | What is in the app | Data / basis |
|---|---|---|
| 1 Science stops | Plan sidebar lists sourced stops; "Add" puts a stop on the route with its work time (up to 5 stops). Gale: Jake Matijevic, Bathurst Inlet, Rocknest. Anywhere: 7 historic landing sites. Each has why / what to do / sources. | Gale positions = MSL PLACES rover pose on the investigation sol (target within ~2 m). Landing sites = USGS SIM 3292 layer. Work minutes are crewed-EVA planning assumptions. |
| 2 Profiles | Who travels: Astronaut (0.8 m/s, 25°, 0.5 m clearance, 0.3 m step), Crew + cart (0.5 m/s, 15°, 1.0 m, 0.15 m), Curiosity-class rover (0.04 m/s, 20°, 1.5 m, 0.25 m). Values shown in the UI and used by the planner. | Planning assumptions, stated in the UI. Walkers slow on slopes with Tobler's hiking function (Earth-derived, relative penalty only). |
| 3 Round trip | "Round trip" tab: outbound + science work + return to start vs an adjustable budget; reserve; per-stop detour, cost and "reserve without it". Return leg drawn dashed orange. | Real A* return leg on the same cost grid; per-stop marginals by re-planning without that stop. Budget is not a suit consumables model. |
| 4 Environment | Map "Conditions" switch (like Relief): TES Dust Cover Index, TES thermal inertia, THEMIS night IR. "Conditions" tab: season (Ls), dust-storm season, insolation, daylight, noon temperature upper bound, pressure, radiation. | Map layers = orbital measurements (NASA Mars Trek tiles). Each tab value is labelled Measurement / Historical record / Model with a source link. Radiation = MSL RAD 0.64 mSv/day at Gale (Hassler et al. 2014). |
| 5 Geology | Map layer "Geology (USGS)" and "Geology on this route" (units crossed, with km ranges). | Tanaka et al. 2014, USGS SIM 3292 (1:20M), rasterised at 1/8° by `scripts/prepare_geology.py`. |
| 6 Data confidence | Map layer "Data confidence": measured 1 m HiRISE DEM + 25 cm image area (green), camera frames (green = position verified by image correlation, amber = archived pose only); elsewhere coarse 200 m terrain. | Existing registration evidence. |

Reproduce: `python scripts/prepare_geology.py` (needs the USGS `sim3292_database.zip` cached under
`%LOCALAPPDATA%/mars-pathfinder/raw/usgs`), `python scripts/prepare_science_targets.py` (needs the cached
PLACES CSV). Tests: `tests/test_marswalk.py`.
