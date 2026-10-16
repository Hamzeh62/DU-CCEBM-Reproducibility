# Two-layer legacy training-source provenance

`legacy_train_validate_original.py` and `legacy_evaluate_locked_test_original.py` are unchanged
historic two-layer scripts using original `/mnt/data` workspace paths. They are preserved for
provenance, and will need researcher-specific path changes and original inputs before retraining.

The frozen two-layer checkpoint is shipped separately under `checkpoints/`.
The independently recovered and verified **six-layer** training source and frozen checkpoint are
included under `extensions/ipix-six-layer-frozen/`; neither represents complete from-scratch
retraining of the two-layer program nor proof of independently rerun six-layer training.
