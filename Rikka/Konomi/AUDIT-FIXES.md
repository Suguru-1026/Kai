# v1.2.0 Audit / Changes

- Added `self-training.js` as an isolated bounded controller.
- Self-training validates configuration, checks cancellation, limits generation/output length, rejects empty/invalid UTF-8/highly repetitive responses, and mixes accepted pseudo-labels with a small anchor sample from the training split.
- Added UI for rounds, prompt count, samples per prompt, progress and cancellation.
- Added architecture presets: minimal, balanced, normal. Explicitly reports that profile changes reinitialize model and optimizer; no cross-width weight transfer is claimed.
- Model-state format bumped to `chat-specialist-v1.2` and includes active profile.
- Added a QA assertion for self-training hard caps.

## Known limitations

- Self-generated answers are not verified truth; no reward model, teacher model, or semantic quality judge is present.
- Pseudo-label training may cause confirmation bias, repetition, or degradation. Seed examples and validation monitoring mitigate but do not eliminate this.
- Profile promotion resets weights and optimizer. Weight expansion and optimizer-state migration are not implemented.
- Browser end-to-end verification has not been performed; tests use Node.js.
