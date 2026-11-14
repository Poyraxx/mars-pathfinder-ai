# Hazcam obstacle refinement — 14 November 2026

Scope: native-image perception and the local Hazcam planner only. Existing UI,
recorded-motion classifier, route ribbon renderer, global 3D map and registration
checks are preserved. This is a research prototype, not a rover safety system.

## What is an obstacle here?

The user selected sensitivity comparable to the red reference rocks: ignore only
very small gravel. Appearance proposes a candidate; actual stereo measurements
must support its size/relief before it blocks a local route. Raised bedrock can
be an obstacle even though its terrain class is bedrock. Shadows, scoop marks,
flat bedrock and rover equipment need different treatment.

The operational head has four classes: ground/small stone, substantial obstacle
candidate, rover hardware, sky/distant unknown. Hardware and sky become unknown
terrain, never invented free ground underneath. The original four-class terrain
taxonomy remains separate from this operational task.

## Data, review and independence

- Genuine AI4Mars v0.6 MSL/Navcam data (CC BY 4.0): the original 2,000 training
  images included 831 rock-bearing scenes. Final expansion: **3,816 training
  images, 1,779 rock-bearing**, after acquisition-group exclusions.
- Five sparse native PDS Hazcam terrain-label scenes retain the original 35
  rock polygons. Three scenes enter terrain training; validation/test do not.
  These are visually reviewed labels, not independent expert annotations.
- Operational labels: **22 native 1024×1024 PDS Hazcam scenes: 17 training,
  3 validation, 2 final test**. Dense reviewed ground regions, individual rocks,
  raised blocks, multiple arm poses, wheels, sky and uncertain exclusions are
  traced in `data/annotations/hazcam-operational-v5.json`. Version4 is retained as
  immutable earlier lineage, and version2 is clearly marked historical.
- Unreviewed pixels remain 255. Object boundaries have an uncertain collar;
  polygons are not presented as exact metric silhouettes. Sol3601 includes the
  user's large platform examples, and Sol147 the red-reference rock cluster.
- Dark Rocknest scoop marks in Sol82/93/100 were corrected to ground after
  native-image review, actual XYZ checks and NASA's sampling description.
  Sol71 was excluded because it repeats the Sol82 physical scene.
- The known Jake boulder demo (catalogue Sol45) is operational **training**.
  Its route test checks geometry; it is not evidence of held-out ML accuracy.
- Original legacy validation400/test400 are frozen. The old test was repeatedly
  used during prior development, so it is not a pristine benchmark.
- New sealed terrain benchmark, **still unopened**: **166 unanimous expert-labelled images in
  23 acquisition groups**, excluding all original splits and adjacent three-sol
  groups, plus the new training episodes. It is reserved for a future terrain-model release; this local-only release
  preserves the existing global terrain model. It is not a Hazcam hardware benchmark.

Sources, image/raw hashes, polygons and split roles are portable in the two
annotation JSON files and `data/annotations/dataset-v4-splits.json.gz`. Raw
images, masks, checkpoints and rejected experiments stay outside Git under
`%LOCALAPPDATA%/mars-pathfinder/training` or ignored `artifacts/`.

## Model and selection

SegFormer-B0 starts from NVIDIA ADE20K weights at fixed revision
`489d5cd81a0b59fab9b7ea758d3548ebe99677da`. A four-class terrain head was trained
for 10 epochs at512, then refined for8 epochs at640 on the expanded data.
The operational model uses512 input, a separate navigation decoder with spatial
refinement and its own fine-tuned encoder. The expanded terrain encoder/head
remain frozen during operational training; state hashes are checked before/after.
Only the trained navigation encoder/decoder is exported for local routes. The
existing global v3 `surface.onnx` stays unchanged. Operational ground labels are
not soil/sand predictions; local planning does not invent material classes.
Production is ONNX Runtime CPU only, without the training stack.

Native crops, flips, brightness/contrast/gamma augmentation, class-weighted CE,
Dice, and explicit hardware/ground/obstacle errors train the operational branch.
Checkpoint selection uses only native-pixel validation with obstacle IoU/recall,
hardware IoU and false-positive penalties. The initialization is retained if
fine-tuning makes results worse. More epochs alone do not establish progress.

Rejected experiments are retained in local logs: LR-ASPP crop refinement traded
precision/IoU for recall; native tiling reduced rock IoU; sparse operational
labels produced broad full-image false positives despite impressive sparse
scores; decoder-only follow-up did not improve its retained initialization.
Those failed candidates are not automatically published. The first encoder
candidate (v9) improved the original validation but overclassified flat bedrock
on the two audit scenes; it was not promoted. Thirteen more native images were
reviewed at full resolution before new predictions. New test roles were fixed
in `operational-v5-review-protocol.json`. Old test results stay recorded, and
those two scenes are no longer called an unseen test.

Final local-model scores and identity belong in
`data/prepared/navigation-evaluation.json` and `navigation-model-card.json`.
The existing terrain model card describes a different task and is preserved. Whole-image validation inspection accompanies
pixel metrics. An object-coverage diagnostic counts >=25% reviewed-pixel overlap;
it is not standard detection AP or a guarantee of finding every stone.

## Final selection and promotion (v11)

| Candidate | Validation objective | Rocks found (val) | Hardware as obstacle (val) |
|---|---|---|---|
| v10-operational-reviewed | 0.548 | 23/36 | 0.03 % |
| v10-normalized (BN recalibration) | 0.534 | 23/36 | 0.20 % |
| **v11-navigation-512 (selected)** | **0.552** | **27/36** | **0.05 %** |

v11 continued v10 with encoder/spatial refinement; validation peaked at epoch 1 and declined
afterwards, so only epoch 1 was kept. Export parity: 100 % pixel agreement on 5 validation images.
Peak working set 314 MiB. Model and policy were frozen (`frozen-selection.json`) before the fresh
test, which was then run exactly once:

| Fresh test (3 scenes, 20 reviewed rocks) | v11 | Previous runtime (v3 big_rock ≥ 0.7) |
|---|---|---|
| Rocks found (≥25 % coverage) | 7/20 (35 %) | 0/20 |
| Obstacle IoU / precision / recall | 0.26 / 0.35 / 0.50 | — |
| Rover hardware as obstacle | 0.002 % | 3.8 % |
| Ground/small stone as obstacle | 2.9 % | 0.2 % |

The test gap versus validation (35 % vs 75 % object recall) is reported, not tuned away; Sol3750's
three shadowed near-field rocks were all missed. These three scenes are now consumed. Route
consequences on the 142-frame replay: the open-sand frame now shifts ~0.95 m right to keep the 2.8 m
body clear of a stereo-confirmed rock cluster (test updated to check clearance, not straightness);
U-turn side swings that start beside the rover, outside the camera view, are shown as a turn cue.

## Native pixels and actual geometry

The former network read the256×256 reference thumbnail and enlarged it. It now
reads the actual1024×1024 display image from the SAME PDS RAD exposure. Declared
dimensions/aspect ratio are checked. Output masks still align exactly with the
256×256 XYZ grid; extra image detail does not create extra depth measurements.

The local planner keeps measured near stereo with error norm<=0.10m, a10cm grid,
2.8m rover body corridor and existing checked A*. In the selected sensitive
profile, semantic rocks need measured elevated-point span>=8cm and relief lower
bound>=2cm after vertical stereo error and nearby-ground fit residuals. Each
candidate is compared against actual nearby non-rock ground in an0.8m ring;
without enough ground support the model alone cannot assert occupation. A broad
gentle terrain crest must not become a rock merely by standing above a global plane. This is a configurable prototype rule,
not a validated engineering limit. Independent25cm steps and slope checks remain.
A tiny object cannot acquire10cm width merely by occupying one grid cell.

At4–12m, less precise actual stereo (error norm<=0.5m), semantic evidence,
raised relief, vertical uncertainty and ground-fit residuals can supply an
occupied-only heading cost. It **does not establish drivable ground or extend
the ribbon into unmeasured terrain**. Slope changes can still cause false
candidates. No exact monocular depth is fabricated; absent depth stays unknown.

Path smoothing/simplification is collision checked again. Vertices use measured
ground heights and the existing PDS CAHVORE projection; perspective ribbon widths
and depth occlusion stay in the established renderer. Hardware occlusions cannot
be filled as free ground by triangulation. No global map occupancy policy changes.

## Reproduction

Use `requirements-training.txt` and a suitable CUDA PyTorch installation. On
Windows, set `MARS_DLL_DIR` from ignored `.local-run.json` before importing ONNX
or Torch. Training used the local RTX3050, with no paid compute.

1. `prepare_rock_expansion.py --count 4500 --workers 4` retrieves genuine ZIP
   members by validated byte ranges, not the entire16GB archive. The portable
   archive restores the frozen original exclusion manifest if needed.
2. `prepare_manual_hazcam.py --sols 2500 2800 3000 3400 4200`, then
   `materialize_rock_annotations.py`; check pinned source hashes.
3. `seal_rock_dataset.py` prepares the frozen3816 training and166 sealed gold.
4. Train `train_rock_refinement.py --architecture segformer-b0` on the pilot
   manifest with `--size 512 --epochs 10 --lr .0003`, then the final manifest
   with `--size 640 --epochs 8 --lr .00015` and the pilot checkpoint.
5. Prepare later operational sources with `prepare_manual_hazcam.py --sols 2600
   2800 3000 3200 3400 3601 3800 4200 4600 4798`. Prepare catalogue sources with
   `prepare_catalog_annotations.py --train-sols 48 54 55 59 61 82 93 100 111 117
   147 --include-jake-navigation-training`. This flag explicitly discloses the
   known demo exception; never call that frame independent evaluation.
6. Prepare additional sources3250,3349,3449,3549,3650,3750,3849,3951,4053,4151,
   4350,4550,4750 with `prepare_manual_hazcam.py`; roles are in the pinned protocol.
   `materialize_operational_annotations.py --version 5` checks image hashes and
   writes native masks/manifest. Do not modify an active training manifest.
7. Operational training lineage/settings and checkpoint hashes are recorded in
   the evaluation report. First train the reviewed decoder, then resume for
   spatial refinement/separate-encoder training. The final reviewed-data stage
   uses8 epochs,840 samples/epoch,512 input,head LR2e-5/encoder LR2e-6,
   ground-error weight1.25 and obstacle-error weight0.75. Use validation only
   to select; export the local branch with `export_operational_candidate.py
   --navigation-only --size 512`. Training checkpoints remain in the local
   training directory; exact checkpoint hashes/settings accompany the report.
8. `check_operational_export.py` compares both ONNX heads against PyTorch on
   actual validation images (>=99.9% pixel agreement). Profile using
   `profile_navigation_runtime.py --navigation-only` without importing the training
   stack. Lower-resolution384 export lost recall and was rejected; pruning the
   unrelated frozen branch preserves512-pixel predictions exactly.
9. Freeze the selected ONNX hash and inference policy in `frozen-selection.json`
   before `evaluate_operational_hazcam.py --split test` or the expert gold run.
   Never tune from final-test failures; collect a new independent holdout first.

## Rights and practical limits

NVIDIA pretrained weights and derivatives permit **non-commercial research and
evaluation** under [the original license](LICENSES/NVIDIA-SegFormer.txt); keep
that complete license with redistributed weights. Source code licensing does
not override model/data restrictions. AI4Mars remains CC BY4.0; PDS imagery
retains NASA/JPL-Caltech attribution.

Thirty-five manually reviewed scenes cannot establish universal robustness.
Small/distant rocks, unfamiliar arm poses, stereo holes, lighting and unseen
terrain can still fail. Scores refer only to their stated labelled populations.
Zero missed hazards, exact monocular metric geometry and flight qualification
are not claimed.
