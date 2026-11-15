# False-alarm research and targeted Hazcam data revision

15 November 2026. Isolated branch `dataset-v11-hard-negative-research`.
Production navigation remains the original four-class v11 model. Dataset v11
and the v22 appearance experiment are distinct version numbers.

## What increased, and what the measurement actually means

The rejected v20 five-class experiment increased historical ground/small-stone
pixels predicted as obstacles from16.07% to20.12%. Its additional-rock class
had only0.457% precision on sparse validation labels. The deployed v11 was
never replaced by v20.

The fixed-policy paired audit found144,127 net additional historicalGT0 pixels
marked obstacle. Sols4019,4339 and4416 account for82.1% of that net increase.
Both native disagreement sheets were visually inspected:21 largest connected
regions across seven nonempty scenes. Several contain real boulder side faces
or rims omitted by sparse historical polygons; others mix soil, bedrock, raised
pieces and shadow. Therefore a GT0 disagreement is not automatically a visual
false alarm. The crop classifications do not establish the pixel purity of a
component or the fraction of all errors caused by labels. Original benchmark
labels remain immutable, so historical numbers stay comparable.

See `data/annotations/operational-v11-error-research.json` for every crop,
paired source/model hashes, thresholds and manual decisions. The two visual
sheets are in `docs/research/v11/historical-disagreements-*.jpg`.

Three plausible contributors need separate experiments; this run is not a
causal ablation of them:

- A shared five-way head must distinguish big/small rock from monocular
  appearance while RGB alone does not establish physical dimensions.
- Fixed16x rare-class weight and128-to512 crop enlargement encourage rare-class
  activation and alter apparent object scale.
- Few explicit ground negatives and limited illumination/arm poses leave gaps.
  Old class0 also deliberately includes small stones; it cannot be a certain
  negative label for a detector of all visible rocks.

[Recall Loss research](https://arxiv.org/abs/2106.14917) specifically discusses
excess false positives from constant minority-class upweighting. It supports
testing this mechanism; it does not prove it caused our run's regression.
[NASA's AI4MARS record](https://ntrs.nasa.gov/citations/20220008371) describes
terrain supervision and expert evaluation, which is a different task from
physical obstacle clearance. A terrain classification score must not be
advertised as a driving-safety score.

## New real imagery and reviewed labels

Archive discovery inspected actual available sol directories. Of26 candidate
sols,25 had a usable native forward RAD product; three were excluded for an old
custom RMC collision, leaving22 new sources. The exact selected IDs, original
PDS URLs, raw/prepared SHA256 hashes, clock groups and exclusions are committed
in `operational-v11-source-pool.json`.

Sources include dunes and tracks, continuous/fractured bedrock, protruding
blocks, rubble, deep shadows and varied arm positions. Preparation is the same
percentile1-99 stretch of actual1024x1024 PDS RAD pixels used previously. No
synthetic scenery, labels derived from predictions, or invented depth was used.
Gamma was changed only in review displays to see dark native source content.
Final native QA also caught an initial Sol502 proposal on a broad rover shadow;
it was rejected before the finalized training run or test prediction.

All22 contexts and276 numbered native region crops were inspected. Of298
proposed regions including22 generic top strips,146 were rejected. The generic
sky strips can contain the rover in some poses and are excluded. Accepted rock
polygons are inset15% and eroded3 native pixels; the surrounding12px collar is
unknown. These are sparse verified interiors, not exhaustive object silhouettes.
Adjacent cores are composed as a union so their uncertain collars cannot erase
one another. Explicit ignore polygons are also applied. Some sources contribute
only equipment negatives after rejecting ambiguous ground/rock proposals.

Final additions:55 positive rock regions, including32 substantial visual
candidates, and2,290,112 reviewed nonrock pixels, including equipment and large
dune/track/shadow patches. Pixel count is not an independent sample count.
Small visible rock appearance is included; physical blocking status is not
inferred from pixel size. Unreviewed rubble remains255, rather than becoming a
false ground negative. Final decisions are in `hazcam-operational-v11.json` and
`operational-v11-native-qa.json`.

All86 prior source images and semantic masks remain byte-identical. Their old
ground/small-stone composite pixels are ignored for binary appearance training,
except the63 previously checked bare-ground cores. Their obstacle1 and new
rock4 interiors are positives; checked hardware2 and sky3 are nonrock negatives.

The expanded108-source manifest contains84 train,8 validation,8 previously
consumed controls and8 fresh test episodes. New test sols200,502,1750,3374,3873,
4169,4560,4809 were fixed before predictions. They have no known old custom RMC
collision and no clock-group plus/minus-one overlap with the frozen terrain
training manifest. The new validation sols800,1400,1700 were also fixed before
predictions. The fresh test contains38 appearance objects and23 substantial
visual candidates after rejecting ambiguous proposals. This is a small,
partially annotated test, not a claim that every stone in those frames is traced.

`operational-v11-integrity.json` verifies hashes, roles, unknown-label handling,
prior immutability and source group guards. It runs no model predictions.

## Protected training experiment

The entire original navigation model and its encoder are immutable. A copied
binary appearance decoder learns rock versus reviewed nonrock, including small
rocks, on the frozen navigation encoder. It has no big/small competition.
Only this separate decoder is optimized. The old navigation model is not part
of its optimizer or export and remains the authoritative baseline.

The prespecified run uses six epochs,1008 nominal draws/epoch,512 input,
headLR2e-5, capped batch class balancing and moderate384/640/1024 context crops.
Initial checkpoint and all six epochs compete only on eight validation scenes'
masked binary rock IoU. No fresh-test tuning, threshold search or opening of
test images for prediction occurs during training.

A short v21 pilot was interrupted during final label QA, before epoch1 finished
and before any fresh-test prediction. It is not a completed release. Labels were
finalized and the complete v22 run restarted from the original v11 checkpoint.
Neither run resumes or deploys the rejected v20 model.

The optional research filter has a deliberately limited contract:

```text
original_obstacle = existing model's unchanged mask (0.7 obstacle,0.8 hardware)
suppress = original_obstacle AND old_rock_probability <0.95
           AND auxiliary_rock_probability <0.10
guarded_obstacle = original_obstacle AND NOT suppress
```

It cannot create an obstacle outside the original mask; baseline hardware
exclusion remains authoritative. This guarantee comes from an explicit boolean
intersection with the original ONNX session, rather than assuming two exported
graphs have numerically identical outputs. Any future integration must keep
stereo-confirmed geometry authoritative; appearance does not establish metric clearance. This research
filter is not wired into the live backend.

The release gates were fixed before new predictions: retain every baseline-found
substantial candidate, at least5% relative decrease in reviewed ground FP when
the baseline has at least1000 such pixels, no hardware FP increase, no more than
0.02 obstacle recall loss, strict baseline-mask subset, CPU peak at most450MiB,
and manual review. Otherwise retain the existing production model. A test set
becomes consumed on its first opening and cannot be relabelled/reused as fresh.

## Completed result: retain the original navigation model

All six epochs completed in539 seconds on the local RTX3050. Validation selected
epoch4: binary rock IoU improved from0.41637 at initialization to0.68883.
The original complete navigation state and copied encoder hashes are unchanged.
Actual-input ONNX argmax parity is at least99.9996% across all eight validation
scenes. A separate CPU process exercised three real local routes, the global
classifier and auxiliary inference with370.5MiB peak working set. This is a
Windows measurement, not a guarantee for Render's warmed server allocator.

The selected model was sealed before first prediction on the eight new test
episodes. On their sparse checked labels:

| Diagnostic | Original navigation mask | With fixed conservative filter |
| --- | ---: | ---: |
| Reviewed ground pixels marked obstacle |4665/575498 (0.811%)|4665/575498 (0.811%)|
| Reviewed obstacle-core pixel coverage |54.89%|54.89%|
| Substantial visual candidates detected |11/23|11/23|
| Original detections lost |—|0|

**The filter failed the predeclared5% relative false-positive reduction gate.**
It also left the original consumed v9 control diagnostic unchanged:16.07%
ground/small-stone disagreement,0.673% checked-hardware disagreement,71.12%
obstacle-core recall. Those are the original v9 masks, not relabelled v10 masks.
The candidate is retained for research, and production v11 remains installed.

Post-test diagnosis, with no threshold/checkpoint/label changes, found that2799
of4665 new-test ground disagreements are protected by the original probability
at least0.95. In the remaining1866, the auxiliary head also considers the
surface rock-like (probability at least0.10), so the fixed filter cannot remove
them. Sol3873 contributes4490 disagreements, mainly plate surfaces. This is a
concrete limitation of appearance-based filtering rather than evidence that
turning the threshold up would be safe.

A **descriptive post-test comparison**, not checkpoint selection, compares the
same binary decoder at initialization and after training. With the same binary
taxonomy, native inputs, sparse labels and argmax decision:

| Binary appearance metric | Initialized two-logit head | Fine-tuned head |
| --- | ---: | ---: |
| Rock IoU |26.94%|47.22%|
| Rock pixel precision |34.43%|54.19%|
| Rock pixel recall |55.35%|78.61%|
| Reviewed nonrock pixels predicted rock |50023|31535|

These figures show a useful appearance improvement, including nine baseline
navigation misses with visible auxiliary overlap. They are **not** additional
confirmed driving obstacles, full-image accuracy or improved navigation safety.
The navigation head's threshold/taxonomy differs and must not be used as a fair
binary baseline. The auxiliary head detects30/38 appearance interiors and20/23
substantial visual candidates under the declared sparse overlap proxy. The
miss-contact sheet was manually inspected and is committed as
`docs/research/v11/missed-candidates.jpg`.

Manual review also exposes a **hardware blind spot**: Sol4809 visibly has tool
parts predicted as rock. Its narrow checked wheel-interior hardware metric is
zero, but that metric does not cover the tool pose. Broad equipment negative
ROIs were deliberately excluded from the strict hardware metric because they
contain holes/background. Do not advertise that zero as perfect self rejection.
The unchanged Sol3873/4809 overlays and the exact manual observations are
committed. The next revision needs verified tool interiors and silhouettes,
more varied arm poses, flat-bedrock/shadow negatives, and stereo-supported
height/attachment evidence. Repeated monocular fine-tuning cannot turn a rock
surface's appearance into measured clearance.

Every result, opening/selection seal, source/label hash, training history,
export/profile check and release decision is under `data/annotations/*v11*`.
All eight new test episodes are now **consumed**. Future experiments need a new
independent group holdout for a release claim. They may remain diagnostic
controls, with that status clearly stated.

The sealed research backup is
`%LOCALAPPDATA%/mars-pathfinder/backups/v11-research-sealed` and includes the new
source images, all appearance labels, new semantic masks, metadata, original
production ONNX, auxiliary ONNX and its Torch checkpoint. The original86 source
backup remains in `before-small-rock-review-v10`. No main files were changed;
the other concurrent worker's edits and document removals were left alone.

## Reproduction and continuation

The local training root is `%LOCALAPPDATA%/mars-pathfinder/training`; existing
v9/v10 sources, manifests and original v11 Torch checkpoint must be present.
See `DATASET_V10_REVIEW.md` for their original provenance and rollback backup.
Run from this worktree with the project's CUDA Python environment. The exact
manifest/model/trainer hashes are saved alongside each run.

```text
python scripts/prepare_research_hazcam_v11.py
python scripts/review_crops_v11.py
python scripts/materialize_operational_v11.py
python scripts/audit_operational_v11.py
python -m unittest discover -s tests -p test_appearance_guard_v11.py
python scripts/train_appearance_v11.py --manifest <training>/manifest-operational-v11.json --init <training>/v11-navigation-512/best-operational.pt --output <new-run>
python scripts/check_appearance_v11.py --candidate <new-run> --manifest <training>/manifest-operational-v11.json --init <training>/v11-navigation-512/best-operational.pt
python scripts/evaluate_appearance_v11.py --candidate <new-run>/appearance.onnx --manifest <training>/manifest-operational-v11.json --output <new-test-output>
python scripts/profile_appearance_v11.py --candidate <new-run>/appearance.onnx --output <new-run>/runtime-profile.json
```

The committed annotations and QA decisions are authoritative; review rendering
does not generate truth. Never rematerialize labels during an active sealed run.
The evaluator refuses to overwrite a fresh-test opening in the same output
directory. A different output directory does not make these episodes fresh again.

The next data need is more verified rock sides versus *actual* flat-bedrock and
shadow negatives, plus stereo/pose evidence for physical height and clearance.
[Recent multimodal Mars research](https://arxiv.org/abs/2609.24952) uses stereo,
poses and driving signals to learn traversability. Its
[project page](https://darren-chiu.github.io/learning-to-drive-on-mars/) currently
marks code and dataset as coming soon; these data have not been downloaded or
included in this experiment. The22 usable new sources come from the available
[official Curiosity Hazcam PDS archive](https://planetarydata.jpl.nasa.gov/img/data/msl/MSLHAZ_1XXX/).

Production baseline SHA256:
`6dcf488232cc3d46cfe7250c1bbfe1311437c9f3f65555cd6858708602ec82b0`.
No UI, deployment, route geometry, runtime thresholds or main-checkout files
are changed in this isolated research branch.
