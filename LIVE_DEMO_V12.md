# Live demo annotation and navigation refinement v12

Scope: the four existing Live GPS Test scenarios, 181 distinct source frames
(142 native 1024×1024 Hazcam, 39 native 256×256 Navcam). No UI, layout,
global terrain model or rover planning policy changes are part of this work.

Every source context was visually reviewed, including all 242 model-free
stereo proposal components. Components were individually accepted as rock,
marked rover hardware, split, or rejected as unknown. Real aligned PDS XYZ/XYE
provides weak geometric supervision; missing depth is never invented.
Flat-ground heuristic labels have lower training weight and are not certified
traversability measurements.

426 additional manual-region crops were reviewed (346 accepted, 80 rejected).
165 stationary-scene template-transfer proposals were reviewed (149 accepted).
Native Navcam pixels were magnified only for inspection. All 12 final mask
composition contact sheets were inspected after the corrections.
The resulting 316 positive annotation regions are overlapping region proxies,
not a count of unique physical rocks. Small and distant uncertain rocks are
retained separately in semantic class4 and ignored in four-class navigation;
they are never silently taught as definite ground or substantial obstacles.

**These are partial labels, not complete silhouettes of every rock.** Mean
per-frame labelled coverage is 19.70% (minimum8.56%). The remaining unknown
pixels stay255. Reviewed interiors have conservative unknown collars; source
pixel hashes, mask hashes, reviewer decisions and weights are retained.

All181 frames are intentionally used for fitting and model selection, as
requested. Scores on these frames measure demo conformance; they are not a
held-out accuracy or generalization result. Zero errors in sparse reviewed
pixels would not establish zero errors throughout the images or safe driving.

## Authoritative data and restoration

- `data/annotations/live-demo-v12-inventory.json`: exact scenario/source inventory.
- `live-demo-v12-review.json`: all181 source reviews and geometry decisions.
- `live-demo-v12-manual-qa.json`: final manual crop accept/reject decisions.
- `live-demo-v12-transfer-proposals.json`: reviewed template-transfer provenance.
- `live-demo-v12-integrity.json`: source/mask/hash checks and label coverage.
- `live-demo-v12-masks.zip`: all543 sealed semantic/navigation/weight PNGs.
- `live-demo-v12-manifest.json.gz`: portable source references, roles and hashes.
- `live-demo-v12-package.json`: archive hashes and member verification.

`live-demo-v12-labels.json` is an early **draft proposal** file. Its first-pass
point guesses are not accepted labels. The materializer takes only explicitly
reviewed equipment interiors and Jake polygons from that draft, then applies
the final review and QA decisions. The sealed masks and portable manifest
above are the authority for the actual training data.

Restore on a checkout with the prepared source images:

```powershell
python scripts/restore_live_demo_v12.py --output <new-label-directory>
```

Restoration verifies all actual source images, XYZ and archived masks. It
refuses to overwrite differing labels. The archive is training/review data;
the inference backend never reads it. No frame-ID lookup or manual-mask
substitution is used for predictions.

## Training and release checks

`scripts/train_live_demo_v12.py` starts from the preserved original v11 full
checkpoint and fine-tunes only the separate navigation encoder/head for12
epochs. `live-demo-v12-protocol.json` fixes weights, runtime thresholds and
selection/release gates before training. Unknown pixels remain ignored.
The global terrain ONNX and original checkpoint are unchanged.

Selection requires retaining all originally detected manual rock-region
proxies and no more than0.02 absolute rock pixel recall loss. Actual ONNX
baseline/candidate paired evaluation, export parity, CPU memory and every
default demo route are checked separately. A failed candidate stays separate.

Training and evaluation outputs are stored outside the repository under
`%LOCALAPPDATA%/mars-pathfinder/training/v23-live-demo-v12`.
The original model and historical datasets are retained for rollback.
Final results and deployment disposition are recorded in the result JSON
and appended here after the checks finish.
