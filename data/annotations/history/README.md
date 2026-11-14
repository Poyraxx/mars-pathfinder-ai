# Historical labels — do not use for new training

Version2 explains the frozen decoder-pretraining checkpoint in the refinement
lineage. It contains older labels (including scoop/rock distinctions corrected
later) and is not the current annotation truth. Version4 and version5 retain the
subsequent review history. The default materializer uses version5.

The former sparse version1 and the unsuccessful version3 candidate remain in
ignored local research artifacts. Never mix old and new masks or overwrite an
active manifest during training.
