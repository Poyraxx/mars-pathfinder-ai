# Small-rock revision v10

The user requested a review of older datasets too, additional small-stone labels,
backups and retention of the deployed model if performance falls. This branch
is isolated from concurrent UI/EVA development in the main checkout.

## Backups and scope

- Base commit: `5aa9e5f`; branch `dataset-v10-small-rock-review`.
- Worktree: `%LOCALAPPDATA%/mars-pathfinder/worktrees/dataset-v10`.
- Verified backup: `%LOCALAPPDATA%/mars-pathfinder/backups/before-small-rock-review-v10`.
  It contains all86 native source images and historical masks, annotation files,
  operational manifests, original terrain manifest, deployed ONNX files and model
  cards. Source/mask snapshot SHA256 values were checked for all86 pairs.
- Every image in the older custom operational annotation versions is covered by
  these86 unique sources. Duplicate historical exposures are not counted again.
- All86 full-source sheets and588 proposed native stone-interior crops were
  visually reviewed. Native crops are nearest-neighbour enlarged, with gamma
  only for viewing; original prepared source bytes never change. Initial
  brightness-only previews were discarded because they clipped highlights.
-75 additional stone interiors survived conservative crop QA and protected-mask
  checks.63 clear bare-ground interior negatives were also selected by eye.
  These are sparse assistant visual labels, not independently expert-reviewed,
  exhaustive rock inventories, exact silhouettes or measured physical sizes.
- Original4616 terrain masks were checked for file dimensions, allowed values,
  empty labels and aggregate hashes. They were NOT all visually re-labelled.
  Their bedrock/soil/sand/big-rock taxonomy is kept separately from operational
  ground/obstacle/hardware/sky/additional-rock appearance.

The terrain audit found32 all255 training masks and five all255 frozen evaluation
masks (three validation, two test). A NEW clean manifest excludes only the32
empty training entries:3784 train,400 validation,400 test. Original files and
frozen evaluation membership stay unchanged. See
`data/annotations/operational-v10-external-cleaning.json`; the local clean
manifest is `training/manifest-v10-terrain-clean.json`. No new terrain model was
trained or installed by this pass.

## What was wrong with using the old ground class as a new appearance negative?

Historical class0 explicitly meant **ground OR small stone**. It is a valid
coarse operational label, but becomes contradictory supervision if a new
appearance task treats it as definite bare ground. Unreviewed255 pixels already
had no loss/score contribution; adding labels there alone does not explain a
previous accuracy decrease.

The new research masks add appearance class4 without changing historical
obstacle1, hardware2 or sky3 pixels. Around a reviewed stone interior, its
untraced64-native-pixel neighbourhood becomes unknown rather than a negative
outline. Positive cores are combined as a union so neighbouring unknown regions
cannot erase one another.

For the remaining historical0 pixels, a separate partial-label bit permits
either0 or4. The loss is `-log(p_ground + p_additional_rock)` there. Explicit
reviewed ground/stone cores use hard class0/4 supervision. Hardware, obstacle
and sky retain their historical hard labels. This avoids teaching that every
untraced small stone is definitely not a rock.

New masks contain8267 checked rock-core pixels and3843 bare-ground-core pixels.
273979 previously ground-labelled pixels surrounding the new stone cores become
unknown. These modest sparse counts must not be represented as dense masks.

## Research training and evaluation

-73 training, five unchanged-role validation, eight previously opened consumed
  controls. No control enters gradient training or checkpoint selection.
- Input512, six epochs,1008 samples/epoch, headLR1e-5 and encoderLR1e-6.
  V11 initialization, separate navigation encoder and spatial refinement;
  original terrain branch is frozen and checked by parameter hash.
- Additional-rock class weight16. Object-centred crops128/192/256 are used for
  the sparse positive/negative cores; original wider crops remain for context.
- Selection is validation only:0.8 times the existing four-class operational
  objective plus0.2 additional-rock IoU. A checkpoint must keep the old
  collapsed validation objective within0.01 of initialization.
- Five-class diagnostics exclude unresolved composite0 pixels. These are
  **sparse appearance diagnostics**, not full-image precision or accuracy.
- For a fair historical-task comparison, logsumexp joins0 and4 back into the
  old ground/small-stone channel. Obstacle1, hardware2 and sky3 remain separate.
  Original v9 masks and fixed runtime thresholds are used for regression checks.
- Both original v8/v9 tests were already consumed before this task. Their
  results cannot establish new untouched generalization, and this branch does
  not authorize automatic deployment.

A pilot run v19 was interrupted after a mask-composition issue and coarse-ground
ambiguity were identified; it was not evaluated on the controls. Its inputs,
labels, logs and partial history are retained outside Git under
`training/v19-v10-five-class-interrupted`. The corrected partial-label run is
`training/v20-v10-partial-ground`. The frozen source/label/protocol/trainer hashes
are in `operational-v10-training-seal.json`.

## Reproduction

```text
python scripts/review_operational_v10.py
python scripts/qa_operational_v10.py
python scripts/materialize_operational_v10.py
python scripts/audit_operational_v10.py --external
```

The first two commands render proposals for visual review, not labels selected
by a model. The committed visual-review, native-QA and ground-review JSON files
are the adjudication authority. Do not regenerate an active sealed training
manifest while training is running.

```text
python scripts/train_operational_v10.py --manifest <training>/manifest-operational-v10.json --init <training>/v4-segformer-expanded/best-surface.pt --resume <training>/v11-navigation-512/best-operational.pt --output <new-run> --size 512 --epochs 6 --samples 1008 --lr 1e-5 --spatial-refinement --train-navigation-encoder --ground-error-weight 1.25 --obstacle-error-weight .75
python scripts/export_operational_candidate.py --source <new-run> --output <raw-five-class> --size 512 --navigation-only
python scripts/check_operational_export.py --candidate <raw-five-class> --manifest <training>/manifest-operational-v10.json
python scripts/export_operational_v10_navigation.py --source <new-run> --output <four-class-compatibility-export>
```

## Result: keep the current model

The corrected partial-label training completed all six epochs on the local
RTX3050; validation selected epoch4. The original terrain parameter hash is
unchanged. Actual settings, initial validation and all epoch metrics are in
`operational-v10-training-run.json`.

On exactly the same original, previously opened eight v9 control sources and
original masks, with unchanged0.7/0.8/3 inference policy:

| Historical operational diagnostic | V11 baseline | V20 candidate |
| --- | ---: | ---: |
| Objects with≥25% reviewed coverage |25/35|26/35|
| Baseline-found large components retained |17/17|17/17|
| Raw obstacle IoU |0.27155|0.24659|
| Runtime obstacle-pixel recall |71.12%|73.75%|
| Reviewed ground/small stone marked obstacle |16.07%|20.12%|
| Reviewed hardware marked obstacle |0.673%|0.466%|

**Rejected for deployment.** A higher object count does not outweigh worse
coverage/false positives. The five-class head also confuses substantial rocks
with the additional-rock class: sparse validation small-rock precision is only
0.457% despite53.59% core-pixel recall. These are not usable full-image accuracy
figures or a new benchmark. The original v11 model remains available and is not
replaced. No threshold was tuned using these controls.

The five-class ONNX export has≥99.9992% argmax agreement on each of the five
validation sources. The four-class compatibility probabilities match the sum
of raw0+4 within3.58e-7. A fresh CPU-only process exercised three local routes
and the global model without importing training libraries, peaking at314.6MiB
on this Windows host; this is not a Render memory guarantee.

Next work should preserve the corrected dataset and test a separate,
size-agnostic rock-appearance head while protecting the existing navigation
output. Appearance alone does not establish physical dimensions. Further
complete object outlines and ground negatives are still needed; the75 new
cores do not make all86 sources exhaustively labelled.

Files to review: `operational-v10-revision.json` (exact accepted regions),
`operational-v10-ground-review.json` (ground negatives),
`operational-v10-external-cleaning.json` (32 empty training exclusions),
`operational-v10-evaluation.json` (full paired metrics and the release decision).

The production baseline is kept at SHA256
`6dcf488232cc3d46cfe7250c1bbfe1311437c9f3f65555cd6858708602ec82b0`.
No UI, runtime backend, thresholds, route geometry or production weights are
changed in this worktree.
