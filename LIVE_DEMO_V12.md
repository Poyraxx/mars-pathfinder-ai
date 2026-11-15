# Live demo annotation and navigation refinement v12

**Release retracted:** the v24 candidate below failed actual real-image fixture
checks: it missed visible rocks in FLA_402284021 and proposed a corridor over
their image area. Weak flat-ground labels included flat-topped rock surfaces.
The original v11 weights and model card are restored, and the original tests
are not weakened. Candidate metrics below are retained as diagnostic history,
not a successful current release. See `release-retraction.json` in the v13
research results. Further source review of original detections/weak-ground
conflicts is in progress on all181 demo frames.

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

## Source QA correction and v13 continuation

The first12-epoch candidate (v23) improved the partial-mask scores but was
**rejected** after its actual output overlays were inspected. It hallucinated
rover hardware on the top of Navcam terrain, where the initial partial masks
were unknown, and extended some rock masks over clear Hazcam soil. This is why
partial annotation scores alone cannot establish full-image correctness.
Its failure and metrics are retained in `data/research/live-demo-v12`.

All39 native Navcam contexts were checked again: their visible landscape
contains no rover hardware (terrain shadows are not hardware). A separate
class-exclusion annotation therefore says only **not class2**, without
assigning ground, rock, sky or clearance. Five manually selected flat-soil
interior rectangles in source3 and5 similarly exclude rock class1, staying
outside the actual boulder and equipment. These are new source-reviewed labels,
not model predictions copied into truth.

`data/annotations/live-demo-v13-exclusions.json` and the matching ZIP preserve
362 exact complement masks. Most masks are empty;39 full-frame not-hardware
masks cover2,555,904 native pixels, and2 Hazcam not-rock masks cover111,186
native pixels. They supplement the immutable v12 semantic labels and do not
turn any unmeasured terrain into safe ground.

```powershell
python scripts/restore_demo_constraints_v13.py --output <new-constraint-directory>
```

`scripts/train_demo_constraints_v13.py` continues the separate navigation model
for6 epochs from the selected v23 weights, adding complement-label losses.
It retains all179 manual-region proxies found by stage1, with at most0.005
absolute rock pixel recall loss. Every source remains demo fitting data.
`live-demo-v13-protocol.json` adds explicit release limits for false hardware
on Navcam and false rocks on the manually reviewed clear-soil interiors.

## Corrected candidate results

Actual ONNX paired conformance on the same181 demo sources:

| Reviewed measurement | Original v11 | Corrected v24 |
| --- | ---: | ---: |
| Ground pixels incorrectly proposed as obstacle |225,934|21,081|
| Hardware pixels incorrectly proposed as obstacle |1,841|209|
| Reviewed rock pixel recall |82.06%|99.74%|
| Originally found manual-region proxies retained |147|147|
| All found manual-region proxies |147|179|
| Navcam false hardware on39 source-reviewed absence masks |not used for this comparison|0 pixels|
| Clear-soil false rock on the new exclusion labels |v23:6,630|2 pixels|

The final candidate is selected from continuation epoch6, after12 stage1
epochs. Actual ONNX/Torch argmax agreement is99.99899% across all181 sources.
CPU regions/three-route/global-model profiling stays below450MiB and imports
neither Torch nor Transformers. All four default scenarios were recomputed at
their unchanged goals. All19 changed-route/reference contact sheets (74
contexts) and6 paired appearance sheets were inspected; the exact transition
counts and limitations are retained in the result records.

The new masks change some swing U-turns into turn-in-place guidance when the
unchanged2.8m swept-clearance constraint is unmet. This is not fixed by clearing
unknown terrain. A separate image-overlay defect was found: remembered real
rocks were painted on the moving arm when occluded. `project_cells` now hides
markers on the current learned nonterrain mask while retaining world tracks;
three focused tests cover occlusion, clear pixels, image bounds and missing
masks. No screen layout or global map change is included.

**Zero overall errors were not achieved:**21,290 false obstacle pixels remain
on the reviewed partial semantic labels. Raw appearance masks can still spill
at rock boundaries and distant ridges; unverified proposals do not certify a
physical obstacle. The Navcam hardware-absence control reaches zero, which is
a narrower result. All these sources were fitted, not held out.

The corrected actual model SHA256 is
`5e0a5eb0d222549f164a34f75c166d4ba5cead9c3a5d74caf9f2d8011019c6bd`.
The original v11 SHA256 is
`6dcf488232cc3d46cfe7250c1bbfe1311437c9f3f65555cd6858708602ec82b0`.
Rollback originals, original full checkpoint and both datasets are sealed under
`%LOCALAPPDATA%/mars-pathfinder/training/backups/live-demo-v12-original-sealed`;
the original model also remains available from the prior Git revision.

`data/research/live-demo-v13/release-result.json` records the release checks;
the model card describes demo fitting honestly and does not inherit the old
model's independent test scores. Deployment-cache/marker checks are recorded
separately after integration with the concurrent main branch changes.
