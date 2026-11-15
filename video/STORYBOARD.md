# Mars Pathfinder AI — 30 s Space Apps demo (draft)

Rules (Space Apps 2025 submission guide): ≤ 30 s, public link, **English subtitles required**,
show how space-agency open data made the project possible; team name, members and challenge.
Judging: Impact, Creativity, Validity, Relevance, Presentation.

Challenge: "Interplanetary Survival Guide: Martian Map" (NASA Space Apps 2026).

| Time | Picture (real app footage) | Turkish voiceover | English subtitle |
|---|---|---|---|
| 0.0–3.5 | Exaggerated relief Mars globe slowly orbiting; title "MARS PATHFINDER AI" + challenge name | Mars'ta bir astronot nereye güvenle yürüyebilir? | Where can an astronaut safely walk on Mars? |
| 3.5–8.0 | Gale crater relief zoom; text "Orbital maps: 200 m. Hazards: 20 cm." | Yörünge haritaları yüz metrelik; bir insanı durduran kaya ise yirmi santim. | Orbital maps resolve 200 m. The rock that stops you is 20 cm. |
| 8.0–14.0 | Plan mode: route with science stops, return leg, round-trip reserve bar, Conditions layer (dust) | Mars Pathfinder, NASA verisiyle bilim duraklarını, dönüş süresini, tozu ve jeolojiyi tek rotada planlar. | Pathfinder plans science stops, return time, dust and geology in one route — from NASA data. |
| 14.0–21.0 | Real Curiosity Hazcam frames: painted route from the rover's feet, rocks marked, route bends around Jake Matijevic rock | Gerçek Curiosity görüntülerinde yapay zekâ kayaları bulur, stereo ölçüm doğrular, rota onları dolaşır. | On real Curiosity images, AI finds rocks, stereo confirms them, and the route goes around. |
| 21.0–26.5 | Results cards: 0 → 62 % rocks found on an independent test; 86 hand-reviewed images; 142 real frames; 100 automated tests | Bağımsız testte kaya bulma yüzde sıfırdan altmış ikiye çıktı. | Independent test: rocks found rose from 0 % to 62 %. |
| 26.5–30.0 | Data logos as text strip (MOLA · HiRISE · PDS Hazcam · PLACES · TES · THEMIS · USGS) + team / members / URL | Gerçek NASA verisi. Mars'ta güvenli ilk adımlar için. | Real NASA data. For safe first steps on Mars. |

Claims are taken from MODEL_REFINEMENT.md / navigation-model-card.json (v11: 0/20 → 7/20 first test,
20/32 = 62 % on independent test_v6; previous runtime 0/36 and 0/20), DATASET_V8_REFINEMENT.md (86 images),
the 142-frame replay and the 100-test suite. Team name and member names are placeholders to fill in.

## Final (rendered)

`artifacts/mars-pathfinder-demo-30s.mp4` — 29.85 s, 1920×1080, Turkish voiceover, English subtitles burned in.
Advisor review (Fable) changed the results claim to the matched, honest form: the new model finds 31 of 59
hand-labelled rocks across three independent test sets; the previous model found 0 of 56 on its evaluation
scenes. "Flags" instead of "finds", "safer" instead of "safe".

Rebuild: `node capture.mjs` (dev server + backend running) → `python voice.py` (set `ELEVENLABS_API_KEY`
and optionally `ELEVENLABS_VOICE_ID` to use ElevenLabs; otherwise Edge neural voice tr-TR-AhmetNeural)
→ `pnpm render`. Team name and members: `demoSchema` in `src/Demo.tsx`, or
`--props='{"team":"...","members":"..."}'`. Music is synthesised in-repo (no third-party audio).
