# Dataset review in an isolated worktree

Requested15 November2026: review old imagery too and explicitly annotate small
stones, keeping backups and the deployed model as rollback.

- Dataset branch: `dataset-v10-small-rock-review` (base `5aa9e5f`).
- Isolated checkout: `C:\Users\osura\AppData\Local\mars-pathfinder\worktrees\dataset-v10`.
- Backup: `C:\Users\osura\AppData\Local\mars-pathfinder\backups\before-small-rock-review-v10`.
- Main working tree/UI/EVA work may continue independently. This pass does not
  replace `data/prepared/navigation.onnx` or modify runtime/backend/UI files.
- Preserve the v11 four-class production model. Small-rock appearance annotations
  use a separate fifth class in research labels; they do not automatically become
  blocked ground. Compatibility masks map small rock to old ground/small-stone0.
- Historical masks/manifests and opened test results remain immutable. Revised
  labels cannot turn an already consumed test into a fresh benchmark.
- New dataset artifacts/scripts will use v10 names. Coordinate merging rather
  than simultaneously rewriting older annotation files or production model cards.
