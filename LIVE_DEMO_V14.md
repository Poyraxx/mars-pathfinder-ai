# Live demo dataset correction v14

Scope: all 181 distinct sources in the four existing Live GPS Test scenarios:
142 native 1024 × 1024 Hazcam images and 39 native 256 × 256 Navcam images.
No UI, layout, global map, synthetic depth or runtime annotation lookup is added.

## Why v23/v24 were rejected

The original three real-image navigation tests remain unchanged. Although the
v24 model improved conformance on sparse v12 labels, it missed visible
flat-topped foreground rocks in `FLA_402284021` and incorrectly offered a partial
route. It also suppressed the open-sand route in `FLA_404772660`. Some weak
flat-ground labels included rock tops, and the sparse positive regions did not
cover those forgotten rocks. Aggregate scores were insufficient evidence.

The original model (`6dcf488232cc3d46cfe7250c1bbfe1311437c9f3f65555cd6858708602ec82b0`)
and its card were restored. The public API rollback was checked. v24 remains
an offline rejected candidate; its earlier release acceptance was retracted.
Original v11 weights and v12/v13 label archives remain available unchanged.

## Source review and corrections

All 326 old-model detection/weak-ground conflict proposals were inspected
against the actual native source pixels on 41 review sheets. A model contour
only selects a review crop; it never becomes positive ground truth.

332 manually placed interior regions underwent a second source-overlay review.
76 ambiguous, shadow-adjacent or incorrectly positioned cores were rejected;
256 additional cores remain. Three Navcam coordinate transcription errors were
caught in this check; their adjusted interiors remained ambiguous and were
excluded. Tiny unresolved Navcam objects remain unknown for navigation.

275,308 weak flat-ground pixels under original-model conflict proposals were
withheld as unknown, rather than automatically relabelled as rocks. New labels
include visible rock interiors, rover equipment, clear tire tracks/shadowed
ground and nonterrain. No positive label is derived solely from model output.

The final masks contain 513 overlapping positive annotation region proxies,
not 513 unique physical rocks. Every frame has partial labels; mean labelled
coverage is 19.56%. This is not exhaustive pixel labelling and cannot establish
whole-image zero error. Source RGB determines appearance only; the existing
measured PDS XYZ/XYE and camera calibration still determine physical clearance.

## Reproducibility

`data/annotations/live-demo-v14-review.json` records source hashes and manual
decisions. `live-demo-v14-core-qa.json` records rejected/adjusted cores. The
versioned ZIP has 543 semantic/navigation/weight PNGs for all 181 sources.
`live-demo-v14-package.json` records all source, label and archive checks.
All twelve final source/mask composition sheets were visually inspected.

Restore portable labels with:

```powershell
python scripts/restore_demo_labels.py --version 14 --output <new-training-directory>
```

`scripts/train_demo_preserving_v14.py` starts from original v11 weights and
uses original-teacher distillation only as regularization on unknown pixels.
Teacher predictions are not new annotation truth. Each epoch runs the
unchanged real boulder, open-sand and blocked-view assertions through production
masking, stereo terrain, planning and projection code. Final ONNX parity,
source overlay review and the complete applicable tests are still required
before a candidate can replace the original model.

All demo frames are fitting and selection data at the user's request. Results
are demo conformance, not held-out accuracy or evidence of generalization.
Training results and the final release decision are recorded separately.

## Final result

Three six-epoch runs completed (18 epochs total). No new model was installed.

| Run | Result |
| --- | --- |
| v25, original weights plus unknown-pixel distillation | Higher rock recall, more false positives; all six epochs failed the unchanged open-sand route assertion. Rejected. |
| v26, frozen encoder with conservative decoder updates | Selected epoch 2 retained all 343 original-found manual region proxies and passed the three real route assertions. Actual ONNX on all 181 sources reduced hardware false pixels by 7.09%, but barely changed ground false pixels and increased ground classified as hardware. Failed existing release gates. |
| v27, classifier-only hard-negative calibration | Reduced ground false positives, but every epoch lost some original-found manual rock regions. Rejected. |

The exported v26 ONNX agrees with its Torch checkpoint on 99.9999937% of argmax
pixels over all 181 sources. Export fidelity does not establish better perception.
Detailed epoch histories, conformance, hashes and the retention decision are in
`data/research/live-demo-v14/training-result.json` and its run subdirectories.
v25/v27 selected epoch 0, meaning the unchanged original, not a successful new
fine-tune. Their rejected last checkpoints remain in the local training folders.

With the original model, all **56 applicable navigation, camera destination,
native inference, route overlay and marker occlusion tests passed** in the
research worktree. The test assertions were not weakened. This check does not
cover concurrent UI/backend edits in the main worktree.

**Zero error was not achieved.** Both original and candidate metrics above use
the same corrected v14 partial labels. They must not be compared directly with
the older v12 scores, whose weak ground labels included some rock surfaces.
The original navigation model, global terrain model and sealed rollback copies
remain unchanged. The reviewed data package and all experiment diagnostics are
preserved; no further training job is left running by this work.
