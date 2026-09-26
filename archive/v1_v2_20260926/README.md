# V1/V2 source backup before V7 sync

This directory preserves the main worktree's uncommitted V1/V2 source as it
existed before the 2026-09-26 sync from the Codex worktree. The Git base was
`6ba17953c92dc8abd1fe61c1585003bf92d6faeb` (`s2gsBase`). Original relative
paths are retained here; no V1/V2 output was removed from `output/v1_300` or
`output/v2_300`.

The active root `train.py` and `scene/gaussian_model.py` are replaced by the
V7 implementation after this backup. To restore V1/V2 as the active code,
copy the backed-up files to their original relative paths after first saving
any newer changes.
