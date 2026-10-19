(function (root) {
  'use strict';
  const K = root.KNA || (root.KNA = {}), E = K.TensorEngine, A = E.errors.assert, T = E.Tensor;
  function plain(value) {
    try { return JSON.parse(JSON.stringify(value)); }
    catch (e) { const x = Error('Metadata must be JSON-serializable.'); x.code = 'E_PERSIST_METADATA'; throw x; }
  }
  function tensorRecord(t) { return { shape: Array.from(t.shape), dtype: t.dtype, data: Array.from(t.data) }; }
  function readTensor(r) {
    A(r && Array.isArray(r.shape) && Array.isArray(r.data), 'E_PERSIST_TENSOR', 'Invalid tensor record.');
    return new T(r.data, r.shape, r.dtype || 'float32');
  }
  function parametersOf(input) {
    A(Array.isArray(input), 'E_PERSIST_PARAMS', 'Parameters must be an array.');
    return input.map((p, i) => { A(p && p.tensor instanceof T, 'E_PERSIST_PARAM', 'Expected Autograd parameter at index ' + i + '.'); return p; });
  }
  function exportState(parameters, options) {
    options = options || {}; const ps = parametersOf(parameters);
    const record = {
      format: 'KNA-STATE', formatVersion: 1,
      createdAt: options.createdAt || new Date().toISOString(),
      metadata: plain(options.metadata || {}),
      parameters: ps.map((p, index) => ({ index, label: p.label || '', requiresGrad: !!p.requiresGrad, tensor: tensorRecord(p.tensor) }))
    };
    if (options.optimizer) {
      const o = options.optimizer;
      if (Array.isArray(o.parameters)) A(o.parameters.length === ps.length && o.parameters.every((p, i) => p === ps[i]), 'E_PERSIST_OPTIMIZER_PARAMS', 'Optimizer parameter order must match exported parameters.');
      record.optimizer = {
        type: o.constructor && o.constructor.name || 'unknown',
        stepCount: Number.isInteger(o.stepCount) && o.stepCount >= 0 ? o.stepCount : 0,
        lr: Number.isFinite(o.lr) ? o.lr : null,
        beta1: Number.isFinite(o.beta1) ? o.beta1 : null,
        beta2: Number.isFinite(o.beta2) ? o.beta2 : null,
        epsilon: Number.isFinite(o.epsilon) ? o.epsilon : null,
        weightDecay: Number.isFinite(o.weightDecay) ? o.weightDecay : null,
        states: ps.map((p, index) => ({ index, m: o.m && o.m.has(p) ? Array.from(o.m.get(p)) : null, v: o.v && o.v.has(p) ? Array.from(o.v.get(p)) : null }))
      };
    }
    return record;
  }
  function validateState(state, parameters, options) {
    options = options || {}; const ps = parametersOf(parameters);
    A(state && state.format === 'KNA-STATE' && state.formatVersion === 1, 'E_PERSIST_FORMAT', 'Unsupported state format/version.');
    A(Array.isArray(state.parameters) && state.parameters.length === ps.length, 'E_PERSIST_PARAM_COUNT', 'Parameter count mismatch.');
    const tensors = state.parameters.map((r, i) => {
      A(r && r.index === i, 'E_PERSIST_ORDER', 'Parameter order mismatch.');
      if (r.label && ps[i].label) A(r.label === ps[i].label, 'E_PERSIST_LABEL', 'Parameter label mismatch at index ' + i + '.');
      const t = readTensor(r.tensor);
      A(E.sameShape(t.shape, ps[i].shape), 'E_PERSIST_SHAPE', 'Shape mismatch at parameter ' + i + '.');
      A(t.dtype === ps[i].tensor.dtype, 'E_PERSIST_DTYPE', 'Dtype mismatch at parameter ' + i + '.');
      A(E.ops.finite(t), 'E_PERSIST_NONFINITE', 'Non-finite parameter value.');
      return t;
    });
    let optStates = null, optimizerRecord = state.optimizer || null, targetOptimizer = options.optimizer || null;
    if (optimizerRecord) {
      A(typeof optimizerRecord.type === 'string' && optimizerRecord.type.length > 0, 'E_PERSIST_OPTIMIZER_TYPE', 'Missing optimizer type.');
      A(Number.isInteger(optimizerRecord.stepCount) && optimizerRecord.stepCount >= 0, 'E_PERSIST_OPTIMIZER_STEP', 'Invalid optimizer step count.');
      const os = optimizerRecord.states;
      A(Array.isArray(os) && os.length === ps.length, 'E_PERSIST_OPTIMIZER', 'Optimizer state count mismatch.');
      if (targetOptimizer) {
        A((targetOptimizer.constructor && targetOptimizer.constructor.name) === optimizerRecord.type, 'E_PERSIST_OPTIMIZER_TYPE', 'Optimizer type mismatch.');
        if (Array.isArray(targetOptimizer.parameters)) A(targetOptimizer.parameters.length === ps.length && targetOptimizer.parameters.every((p, i) => p === ps[i]), 'E_PERSIST_OPTIMIZER_PARAMS', 'Optimizer parameter order mismatch.');
        for (const key of ['lr','beta1','beta2','epsilon','weightDecay']) {
          const value = optimizerRecord[key];
          if (value !== null && value !== undefined && key in targetOptimizer) {
            let valid = Number.isFinite(value);
            if (key === 'lr' || key === 'epsilon') valid = valid && value > 0;
            else if (key === 'beta1' || key === 'beta2') valid = valid && value >= 0 && value < 1;
            else if (key === 'weightDecay') valid = valid && value >= 0;
            A(valid, 'E_PERSIST_OPTIMIZER_VALUE', 'Invalid optimizer setting: ' + key + '.');
          }
        }
      }
      optStates = os.map((r, i) => {
        A(r && r.index === i, 'E_PERSIST_OPTIMIZER_ORDER', 'Optimizer state order mismatch.');
        const check = x => {
          if (x === null || x === undefined) return null;
          A(Array.isArray(x) && x.length === ps[i].size, 'E_PERSIST_OPTIMIZER_SHAPE', 'Optimizer state size mismatch.');
          A(x.every(Number.isFinite), 'E_PERSIST_OPTIMIZER_NONFINITE', 'Non-finite optimizer state.');
          return Float32Array.from(x);
        };
        const m = check(r.m), v = check(r.v);
        if (targetOptimizer && targetOptimizer.m && targetOptimizer.v) A((m === null) === (v === null), 'E_PERSIST_OPTIMIZER_PAIR', 'Adam moment arrays must be both present or both absent.');
        return { m, v };
      });
    }
    return { parameters: ps, tensors, optimizer: targetOptimizer, optimizerRecord, optStates };
  }
  function importState(state, parameters, options) {
    options = options || {}; const v = validateState(state, parameters, options);
    // All records are validated before mutating any parameter or optimizer state.
    for (let i = 0; i < v.parameters.length; i++) { v.parameters[i].tensor.data.set(v.tensors[i].data); v.parameters[i].zeroGrad(); }
    let optimizerRestored = false;
    if (v.optimizer && v.optimizerRecord && v.optStates) {
      const o = v.optimizer, r = v.optimizerRecord;
      if (o.m && o.v) {
        o.m.clear(); o.v.clear();
        v.optStates.forEach((s, i) => { if (s.m) o.m.set(v.parameters[i], s.m); if (s.v) o.v.set(v.parameters[i], s.v); });
      }
      o.stepCount = r.stepCount;
      for (const key of ['lr','beta1','beta2','epsilon','weightDecay']) if (r[key] !== null && r[key] !== undefined && key in o) o[key] = r[key];
      optimizerRestored = true;
    }
    return { loadedParameters: v.parameters.length, optimizerRestored, stepCount: v.optimizerRecord ? v.optimizerRecord.stepCount : 0 };
  }
  function stringify(state, space) { return JSON.stringify(state, null, space === undefined ? 2 : space); }
  function parse(text) {
    let s; try { s = JSON.parse(text); }
    catch (_) { const x = Error('Invalid JSON state.'); x.code = 'E_PERSIST_JSON'; throw x; }
    return s;
  }
  K.Persistence = Object.freeze({ version: '0.7.0', exportState, validateState, importState, stringify, parse });
})(typeof globalThis !== 'undefined' ? globalThis : this);
