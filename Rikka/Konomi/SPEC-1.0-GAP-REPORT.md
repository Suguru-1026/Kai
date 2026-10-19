# Specification 1.0 gap report — v1.2.0

This package adds a bounded self-training loop and staged architecture presets. It does **not** claim all original specification 1.0 items are complete.

| Area | v1.2 state | Remaining gap |
|---|---|---|
| Tensor Engine / Autograd | Existing v1.1 operations and QA retained | Full finite-difference gradient coverage and all P0 contracts not audited exhaustively |
| Chat model | Decoder-only causal Transformer, RoPE, RMSNorm, SwiGLU, tied embeddings, KV cache | No large pretrained weights, no semantic quality guarantee, no batched inference |
| Tokenizer | Learned byte-level BPE, UTF-8 fallback | Small seed corpus; not a general high-quality tokenizer |
| Training | Assistant-only loss, minibatches, AdamW, validation loss | No distributed training, gradient accumulation, or robust benchmark suite |
| Self-training | Generate → filter → mix with anchors → train → repeat, with hard caps | No external teacher, reward model, or semantic correctness verification; drift remains possible |
| Architecture profiles | Minimal / balanced / normal configurations | Profile change resets all weights and optimizer state; no weight expansion or optimizer migration |
| Persistence | JSON model and optimizer state save/load | No binary checkpoint format, cryptographic integrity, or atomic persistence |
| Runtime | CPU / FP32, cancelable UI loop | WebGPU, worker-based compute and performance optimization not implemented |
| Browser validation | Node syntax and integration smoke tests | Full real-browser / Chromebook end-to-end tests remain outstanding |
