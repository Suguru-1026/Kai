(function (g) {
  'use strict';
  const K = g.KNA || (g.KNA = {}), E = K.TensorEngine, A = E.errors.assert, T = E.Tensor;
  if (!K.Autograd) throw Error('Autograd required.');
  class Dataset {
    constructor(samples) {
      A(Array.isArray(samples) && samples.length > 0, 'E_DATASET_EMPTY', 'Samples required.');
      this.samples = samples.map((s, i) => {
        A(s && Object.prototype.hasOwnProperty.call(s, 'input') && Object.prototype.hasOwnProperty.call(s, 'target'), 'E_DATASET_SAMPLE', 'input and target required.', { index: i });
        return { input: s.input, target: s.target };
      });
      this.length = samples.length;
    }
    get(i) { A(Number.isInteger(i) && i >= 0 && i < this.length, 'E_DATASET_INDEX', 'Index out of range.'); return this.samples[i]; }
  }
  class DataLoader {
    constructor(dataset, o) {
      o = o || {}; A(dataset instanceof Dataset, 'E_DATALOADER_DATASET', 'Dataset required.');
      this.dataset = dataset; this.batchSize = o.batchSize === undefined ? 1 : o.batchSize;
      this.shuffle = !!o.shuffle; this.seed = o.seed === undefined ? 1 : o.seed;
      A(Number.isInteger(this.batchSize) && this.batchSize > 0, 'E_BATCH_SIZE', 'Positive integer batch size required.');
      A(Number.isSafeInteger(this.seed), 'E_DATALOADER_SEED', 'Seed must be a safe integer.');
    }
    *batches(epoch) {
      const ids = Array.from({ length: this.dataset.length }, (_, i) => i);
      if (this.shuffle) {
        let s = (this.seed + (epoch || 0)) >>> 0 || 1;
        const r = () => { s ^= s << 13; s ^= s >>> 17; s ^= s << 5; return (s >>> 0) / 4294967296; };
        for (let i = ids.length - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [ids[i], ids[j]] = [ids[j], ids[i]]; }
      }
      for (let i = 0; i < ids.length; i += this.batchSize) {
        const ss = ids.slice(i, i + this.batchSize).map(j => this.dataset.get(j));
        yield { inputs: ss.map(x => x.input), targets: ss.map(x => x.target) };
      }
    }
  }
  function crossEntropy(logits, targets) {
    A(logits instanceof T && logits.rank === 2 && logits.shape[0] > 0 && logits.shape[1] > 0, 'E_LOSS_SHAPE', 'logits [non-empty batch, classes] required.');
    A(E.ops.finite(logits), 'E_LOSS_NONFINITE', 'Logits must be finite.');
    const [b, c] = logits.shape;
    A(targets && typeof targets.length === 'number' && targets.length === b, 'E_LOSS_TARGETS', 'Target count mismatch.');
    const p = new Float32Array(logits.size), grad = new Float32Array(logits.size); let loss = 0;
    for (let i = 0; i < b; i++) {
      const t = targets[i]; A(Number.isInteger(t) && t >= 0 && t < c, 'E_LOSS_TARGET', 'Invalid target.');
      let m = -Infinity; for (let j = 0; j < c; j++) m = Math.max(m, logits.data[i * c + j]);
      let z = 0; for (let j = 0; j < c; j++) { p[i * c + j] = Math.exp(logits.data[i * c + j] - m); z += p[i * c + j]; }
      A(Number.isFinite(z) && z > 0, 'E_LOSS_NUMERIC', 'Softmax normalization failed.');
      loss += Math.log(z) - (logits.data[i * c + t] - m);
      for (let j = 0; j < c; j++) { p[i * c + j] /= z; grad[i * c + j] = p[i * c + j]; }
      grad[i * c + t] -= 1;
    }
    for (let i = 0; i < grad.length; i++) grad[i] /= b;
    return { value: loss / b, gradient: T.from(grad, { shape: logits.shape }), probabilities: T.from(p, { shape: logits.shape }) };
  }
  function validateParameters(parameters, code) {
    A(Array.isArray(parameters) && parameters.every(p => p && p.tensor instanceof T), code || 'E_OPT_PARAMS', 'Optimizer parameters must be Autograd Values.');
  }
  class AdamW {
    constructor(parameters, o) {
      o = o || {}; validateParameters(parameters); this.parameters = parameters.slice();
      this.lr = o.lr === undefined ? 1e-3 : o.lr; this.beta1 = o.beta1 === undefined ? 0.9 : o.beta1;
      this.beta2 = o.beta2 === undefined ? 0.999 : o.beta2; this.epsilon = o.epsilon === undefined ? 1e-8 : o.epsilon;
      this.weightDecay = o.weightDecay === undefined ? 0 : o.weightDecay; this.stepCount = 0; this.m = new Map(); this.v = new Map();
      A(Number.isFinite(this.lr) && this.lr > 0, 'E_OPT_LR', 'Learning rate must be positive and finite.');
      A(Number.isFinite(this.beta1) && this.beta1 >= 0 && this.beta1 < 1, 'E_OPT_BETA', 'beta1 must be in [0,1).');
      A(Number.isFinite(this.beta2) && this.beta2 >= 0 && this.beta2 < 1, 'E_OPT_BETA', 'beta2 must be in [0,1).');
      A(Number.isFinite(this.epsilon) && this.epsilon > 0, 'E_OPT_EPSILON', 'epsilon must be positive and finite.');
      A(Number.isFinite(this.weightDecay) && this.weightDecay >= 0, 'E_OPT_WEIGHT_DECAY', 'weightDecay must be non-negative and finite.');
    }
    zeroGrad() { this.parameters.forEach(p => p.zeroGrad()); }
    step() {
      const t = this.stepCount + 1, candidates = [];
      for (const p of this.parameters) {
        if (!p.grad) continue;
        A(E.sameShape(p.grad.shape, p.shape) && E.ops.finite(p.grad), 'E_OPT_GRAD', 'Gradient shape or values invalid.');
        const m = this.m.get(p) || new Float32Array(p.data.length), v = this.v.get(p) || new Float32Array(p.data.length);
        const nm = new Float32Array(m), nv = new Float32Array(v), n = new Float32Array(p.data.length);
        for (let i = 0; i < n.length; i++) {
          const q = p.grad.data[i]; nm[i] = this.beta1 * m[i] + (1 - this.beta1) * q; nv[i] = this.beta2 * v[i] + (1 - this.beta2) * q * q;
          n[i] = p.data[i] * (1 - this.lr * this.weightDecay) - this.lr * (nm[i] / (1 - this.beta1 ** t)) / (Math.sqrt(nv[i] / (1 - this.beta2 ** t)) + this.epsilon);
          A(Number.isFinite(n[i]) && Number.isFinite(nm[i]) && Number.isFinite(nv[i]), 'E_OPT_NONFINITE', 'Non-finite optimizer candidate.');
        }
        candidates.push([p, n, nm, nv]);
      }
      for (const [p, n, m, v] of candidates) { p.tensor.data.set(n); this.m.set(p, m); this.v.set(p, v); }
      this.stepCount = t; return candidates.length;
    }
  }
  class SGD {
    constructor(parameters, o) {
      o = o || {}; validateParameters(parameters); this.parameters = parameters.slice(); this.lr = o.lr === undefined ? 0.01 : o.lr; this.stepCount = 0;
      A(Number.isFinite(this.lr) && this.lr > 0, 'E_OPT_LR', 'Learning rate must be positive and finite.');
    }
    zeroGrad() { this.parameters.forEach(p => p.zeroGrad()); }
    step() {
      const candidates = [];
      for (const p of this.parameters) if (p.grad) {
        A(E.sameShape(p.grad.shape, p.shape) && E.ops.finite(p.grad), 'E_OPT_GRAD', 'Gradient shape or values invalid.');
        const n = Float32Array.from(p.data, (x, i) => x - this.lr * p.grad.data[i]);
        for (const x of n) A(Number.isFinite(x), 'E_OPT_NONFINITE', 'Non-finite optimizer candidate.'); candidates.push([p, n]);
      }
      for (const [p, n] of candidates) p.tensor.data.set(n);
      this.stepCount++; return candidates.length;
    }
  }
  class Scheduler {
    constructor(o) {
      o = o || {}; this.baseLR = o.baseLR === undefined ? 1e-3 : o.baseLR; this.warmupSteps = o.warmupSteps === undefined ? 0 : o.warmupSteps;
      this.decaySteps = o.decaySteps === undefined ? 0 : o.decaySteps; this.minLR = o.minLR === undefined ? 0 : o.minLR;
      A(Number.isFinite(this.baseLR) && this.baseLR > 0, 'E_SCHEDULER_LR', 'baseLR must be positive and finite.');
      A(Number.isInteger(this.warmupSteps) && this.warmupSteps >= 0 && Number.isInteger(this.decaySteps) && this.decaySteps >= 0, 'E_SCHEDULER_STEPS', 'Schedule steps must be non-negative integers.');
      A(Number.isFinite(this.minLR) && this.minLR >= 0 && this.minLR <= this.baseLR, 'E_SCHEDULER_MIN_LR', 'minLR must be between zero and baseLR.');
    }
    value(s) {
      A(Number.isInteger(s) && s >= 0, 'E_SCHEDULER_STEP', 'Invalid step.');
      if (this.warmupSteps && s < this.warmupSteps) return this.baseLR * (s + 1) / this.warmupSteps;
      if (this.decaySteps && s >= this.warmupSteps) return Math.max(this.minLR, this.baseLR * (1 - Math.min(1, (s - this.warmupSteps) / this.decaySteps)));
      return this.baseLR;
    }
  }
  class Trainer {
    constructor(o) {
      o = o || {}; A(typeof o.stepFunction === 'function', 'E_TRAIN_STEP', 'stepFunction required.');
      this.stepFunction = o.stepFunction; this.state = 'idle'; this.stepCount = 0; this.epoch = 0; this.pauseFlag = false; this.stopFlag = false; this.onEvent = o.onEvent || (() => {});
    }
    pause() { if (this.state !== 'running') return false; this.pauseFlag = true; return true; }
    resume() { if (this.state !== 'paused') return false; this.pauseFlag = false; this.state = 'running'; return true; }
    stop() { this.stopFlag = true; }
    async fit(loader, o) {
      o = o || {}; A(loader && typeof loader.batches === 'function', 'E_TRAIN_LOADER', 'A compatible DataLoader is required.');
      A(['idle','completed','stopped'].includes(this.state), 'E_TRAIN_STATE', 'Trainer is active or failed.');
      const max = o.maxSteps === undefined ? Infinity : o.maxSteps, epochs = o.maxEpochs === undefined ? 1 : o.maxEpochs;
      A((max === Infinity || Number.isInteger(max) && max >= 0) && Number.isInteger(epochs) && epochs >= 0, 'E_TRAIN_LIMIT', 'maxSteps and maxEpochs must be non-negative integers.');
      this.state = 'running'; this.stopFlag = false;
      try {
        outer: for (let e = 0; e < epochs; e++) {
          this.epoch++;
          for (const batch of loader.batches(this.epoch - 1)) {
            if (this.stopFlag || this.stepCount >= max) break outer;
            while (this.pauseFlag && !this.stopFlag) { this.state = 'paused'; await new Promise(r => setTimeout(r, 20)); }
            if (this.stopFlag) break outer;
            this.state = 'running'; const result = await this.stepFunction(batch, { trainer: this, step: this.stepCount, epoch: this.epoch });
            this.stepCount++; this.onEvent({ type: 'step', step: this.stepCount, result });
            if (this.stepCount >= max) break outer;
          }
        }
        this.state = this.stopFlag ? 'stopped' : 'completed'; return { state: this.state, step: this.stepCount, epoch: this.epoch };
      } catch (e) { this.state = 'failed'; throw e; }
    }
  }
  K.TrainingSystem = Object.freeze({ version: '0.6.0', Dataset, DataLoader, crossEntropy, AdamW, SGD, Scheduler, Trainer });
})(globalThis);
