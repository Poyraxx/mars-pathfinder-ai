# Operational Hazcam datasets v8/v9 — 15 November 2026

**Final decision: data expanded; production remains v11.** Two GPU fine-tunes and
a validation-selected single-head parameter mixture were evaluated. The final
challenger failed the predeclared large-object retention gate. No UI, runtime
weights, thresholds, global map or planner geometry were changed.

The user requested more careful labels and better perception. This pass concerns
the local obstacle/hardware head. It does not change the interface, global terrain
classifier, camera calibration, rover dimensions, clearance limits or route renderer.

## Reviewed material

25 additional authentic native PDS Hazcam exposures were individually inspected:
17 training views and eight heldout views. The complete operational manifest has
78 images: 65 training, five unchanged validation and eight test_v8. Older consumed
tests/audits explicitly become training; none is presented as a fresh benchmark.

The new annotations contain 42 retained training rock polygons and 38 retained test
rock polygons, plus separately traced rover hardware, reviewed ground/distant regions
and explicit unknown exclusions. These are object instances across images, not a
claim of 80 distinct physical rocks. Sol2500 and2550 show the same drilling-stop
terrain with a different arm pose; both are training, and their correspondence is
recorded in the annotations.

The review first proposed90 training and56 heldout candidate regions. Native-pixel
object crops were then inspected before any new model predictions. Ambiguous relief,
coarse silhouettes, flat bright plates, bare soil, tiny gravel and quantised shadow
regions were excluded as255:48 training and18 heldout regions. Source pixels were
never generated or altered. Brightness/gamma transforms were used only for viewing
underexposed sources; the trainer receives the pinned original preparation.

Coordinates are explicitly512-preview coordinates, doubled onto the1024-native
mask. A six-native-pixel dilation around class/ROI boundaries is ignored. Unreviewed
pixels stay255. The labels are assistant visual review, not independent expert
truth, exact silhouettes or measured physical obstacle heights. Stereo still decides
whether an appearance candidate has sufficient measured size and relief to block.

## Split and reproducibility controls

- Source and mask hashes, source URLs and roles are pinned. The portable split
  manifest is `data/annotations/operational-v8-splits.json.gz`.
- All78 source/mask pairs passed dimension, class-value and SHA checks.
- There is no cross-split image hash, group or RMC site/drive collision. New test
  episodes are more than six sols from training and absent, with buffered acquisition
  groups, from the frozen terrain training manifest.
- The training manifest is immutable while a job is active. Heldout materialization
  reuses its sealed entries without rewriting training masks.
- Existing five-scene validation and the objective remain unchanged. Test selection
  is not based on new-test predictions. The candidate plan and release gates were
  recorded before opening the new test.

The final heldout masks contain38 scored connected obstacle instances (at least50
reviewed native pixels each), across5.2million reviewed pixels. These are sparse
research diagnostics, not detection AP, exhaustive real-world precision or a
guarantee of finding every stone.

Reproduce source preparation with `prepare_manual_hazcam.py --annotations
data/annotations/hazcam-operational-v8.json`; then use
`materialize_operational_v8.py --trainval-only` for training. Once that file is
sealed, `--heldout-only` builds the complete manifest without changing active
training inputs. Historical v5/v7 annotations and source preparation are retained.

## Training and release decision

Two chains are compared using validation only: v16 from deployed v11 (eight epochs,
1008 samples/epoch, head LR2e-5) and v17 from the earlier v14 challenger (six epochs,
840 samples/epoch, head LR1e-5). Both use512 input, a separate navigation encoder,
spatial refinement and a frozen terrain branch. Each retains its initialization if
no epoch improves the existing objective. Training uses the existing local RTX3050;
no paid service is used.

The selected checkpoint, ONNX hash, unchanged runtime policy and manifest must be
sealed before fresh test evaluation. Promotion requires object coverage recall not
below deployed v11, obstacle IoU no worse than0.01 below v11, hardware-as-obstacle no
more than0.005 above v11 and ground-as-obstacle no more than0.01 above v11. Export
parity and actual navigation regressions also have to pass. Measured results and the
final decision are recorded below after this evaluation.

### First new test: v8

The validation objective selected v17 (0.6151) over v16 (0.5671); deployed v11
scores 0.5521. Both fine-tunes preserve the original terrain branch bit for bit.
The selected v17 ONNX reproduces PyTorch argmax on all five validation images.
On the eight sealed test_v8 images, both v11 and v17 find26/38 scored connected
objects. Obstacle IoU rises0.1908→0.2250 and runtime ground-as-obstacle falls
11.89%→7.87%. Runtime obstacle-pixel recall falls62.30%→56.26%, however.

Whole-image/native-object review rejected v17: it loses three of17 baseline-found
large components on Sol4226,4425,4525. A better aggregate score was insufficient.
The original aggregate gate result is preserved alongside that additional visual
rejection in `operational-v8-test-opening.json` and the full paired results in
`operational-v8-evaluation.json`. V8 is subsequently consumed training/control
material, never called a fresh benchmark again.

### Eight further reviewed sources: v9

Eight more PDS exposures were visually traced before any predictions:
Sol4019,4080,4339,4416,4480,4584,4684,4784. Candidate4266 was excluded because it
shares an RMC site/drive with consumed4280;4234 likewise shares one with4226.
Native crop review retained37 of65 proposed rock polygons and excluded28
ambiguous regions as255. The37 polygons produce35 scored connected components
after rasterization and boundary exclusions; instances and components differ.

The cumulative operational dataset now contains **86 images**, up from53 in v7:
73 training/control, five unchanged validation, eight newly opened test_v9.
Across v8/v9, **33 additional images and117 retained rock-instance polygons**
were reviewed. The new v9 test has5,519,089 reviewed pixels. Full dataset totals
are318 obstacle components in train/test combined, plus36 validation components.
The v17 encoder was actually trained on65 images; the additional eight consumed
v8 images were used for mixture controls, not a claimed73-image gradient update.

To preserve previously found blocks, a weight grid was selected using only the
fixed validation set and the explicitly consumed v8 large-object controls. A
two-model probability ensemble was rejected at548.9MiB CPU peak. The same-size
single head selected **0.6×v11 +0.4×v17 parameter weights**, preserving the frozen
terrain parameters, and retained17/17 consumed large-object controls. Its
validation objective is0.5769 and obstacle IoU0.4957. Selection used the exact
production512 BILINEAR preprocessing on CPU; preliminary bicubic calibration was
discarded before selection. The final single-head CPU profile is311.9MiB with
three real local routes and the global model; this is a Windows measurement,
not a guarantee of Render memory use.

The candidate, coefficient, policy and reviewed manifest were sealed before
test_v9 was opened. Results on exactly the same eight new images:

| Diagnostic | Production v11 | Rejected v18 mixture |
| --- | ---: | ---: |
| Objects with≥25% reviewed mask coverage |25/35|25/35|
| Obstacle IoU, raw four-class head |0.2716|0.2855|
| Runtime obstacle-pixel recall |71.12%|70.00%|
| Reviewed ground/small stone marked obstacle |16.07%|13.88%|
| Reviewed hardware marked obstacle |0.673%|0.717%|
| Baseline-found large components retained |17/17|**16/17**|

**No promotion.** Sol4080 component7 (3028 reviewed native pixels, bbox
[818,843,887,907]) drops from26.68% to19.29% coverage and fails the frozen25%
criterion. Sol4019 gains a smaller object, masking that loss in the aggregate
25/35 count. Both models miss the two reviewed objects on Sol4784. Full-scene
prediction comparisons were inspected for all eight sources, including large
near blocks on4019,4339,4584. Hardware errors also do not uniformly improve.
V9 is now opened/consumed; no second coefficient or threshold was selected from
these results. Future work can use these misses as disclosed training controls,
but needs another sealed independent holdout for a release decision.

The audit trail is portable under `data/annotations/`:

- `hazcam-operational-v8.json`, `hazcam-operational-v9.json`: source URLs/hashes,
  reviewed polygons and excluded regions.
- `operational-v8-splits.json.gz`, `operational-v9-splits.json.gz`: exact split
  membership, source/mask hashes and native pixel counts, without machine paths.
- `operational-v8-training-runs.json`: actual fine-tune settings, checkpoint
  hashes, epoch history and validation scores.
- `operational-v9-weight-selection.json`, `operational-v9-frozen-selection.json`:
  coefficient selection and the pre-test seal.
- `operational-v8-evaluation.json`, `operational-v9-evaluation.json`: paired
  same-population baseline/candidate results; v9 includes per-component coverage.
- `operational-v9-release-decision.json`: rejected release, runtime memory and
  unchanged production hash.

`materialize_operational_v8.py --version9 --trainval-only` builds the73/5
training-control/validation manifest from the existing v8 lineage. The matching
`--version9 --heldout-only` materializes the new test without rewriting its
sealed training inputs. Prepare pinned originals first with
`prepare_manual_hazcam.py --annotations data/annotations/hazcam-operational-v9.json`.
The eight sources retain their historical test role in the archive; their now
consumed status is explicitly recorded in the evaluation and release decision.

The following commands reproduce the coefficient and its export using the
retained local checkpoints:

```text
python scripts/merge_navigation_weights.py --baseline <v11-checkpoint-directory> --challenger <v17-checkpoint-directory> --new-weight 0.4 --output <mixture-directory>
python scripts/export_operational_candidate.py --source <mixture-directory> --output <export-directory> --size 512 --navigation-only
```

These are rejected research artifacts; neither command authorizes a release.
Frozen-model test evaluation requires the exact manifest and unchanged inference
policy. V8/v9 annotation bytes are preserved by `.gitattributes` so the seals
survive a Windows/Linux checkout.

For an inference-free check of all86 native pairs and cross-split exclusions:

```text
python scripts/verify_operational_dataset.py --manifest <training-root>/manifest-operational-v9.json --terrain-manifest <training-root>/manifest-v4-final.json --annotations data/annotations/hazcam-operational-v9.json --protocol data/annotations/operational-v9-review-protocol.json --output <audit.json>
```

The parameter-mixture reproduction command was also checked against the actual
evaluated checkpoint: all423 state tensors match exactly. The production model
file was never replaced.

Production navigation SHA256 remains
`6dcf488232cc3d46cfe7250c1bbfe1311437c9f3f65555cd6858708602ec82b0`.
The original surface classifier is untouched. Existing candidate regression run:
94 Python tests passed; both exports have100% validation argmax parity. Dataset
hash/class/shape/split checks also pass. No inference of metric depth is added.

## Camera–map pair bank

This is a separate problem. A fixed, unchanged correlation audit sampled12 default
drive frames, including the existing positive control. Three passed all four
bandpasses: Sol55 (existing), Sol123 and Sol150 (two new candidates). The measurements,
source hashes and archived placements are in `camera-map-pairs-v1.json`.

The pair bank does not change runtime registration flags or enable new map-fusion
observations. A correlation failure is not proof of a wrong location, and is not a
supervised wrong-location negative. PLACES itself used the HiRISE map: these checks
are visual consistency at an archive-based placement, not independent geodetic truth
or a trained monocular GPS system. More rock labels do not solve that distinction.
