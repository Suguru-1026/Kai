# Changelog

## v1.2.0
- Added bounded generate → train → regenerate self-training loop.
- Added hard caps: 3 rounds, 8 prompts, 2 samples/prompt, 48 generations, 24 optimizer updates.
- Mixes accepted pseudo-labels with seed training examples; records accepted/rejected outputs and per-round stats.
- Added minimal, balanced, and normal architecture profiles. Switching profile resets weights and optimizer; weight expansion is not implemented.
- Added QA coverage for hard self-training limits.
