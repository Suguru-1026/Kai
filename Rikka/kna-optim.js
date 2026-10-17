(function (global) {
  "use strict";

  const KNA = global.KNA;
  const Tensor = KNA.Tensor;
  const parameterStates = new WeakMap();
  const optimizerOwners = new WeakMap();
  let nextParameterId = 1;

  const DEFAULTS = Object.freeze({
    learningRate: 0.0003,
    beta1: 0.9,
    beta2: 0.999,
    epsilon: 1e-8,
    weightDecay: 0.01,
    maxGradNorm: 1
  });

  function optimizerError(code, message) {
    const error = new Error(message);
    error.code = code;
    return error;
  }

  function assertParameter(parameter) {
    const state = parameterStates.get(parameter);
    if (!state || state.disposed) {
      throw optimizerError("OPTIMIZER_INVALID_PARAMETER", "Parameter is invalid or disposed.");
    }
    return state;
  }

  function assertFinite(value, name) {
    if (typeof value !== "number" || !Number.isFinite(value)) {
      throw optimizerError("OPTIMIZER_INVALID_CONFIG", name + " must be finite.");
    }
    return value;
  }

  function toFloat32(value, message) {
    const rounded = Math.fround(value);
    if (!Number.isFinite(value) || !Number.isFinite(rounded)) {
      throw optimizerError("OPTIMIZER_NUMERICAL_ERROR", message || "Optimizer produced a non-finite value.");
    }
    return rounded;
  }

  function normalizeConfig(options, base = DEFAULTS) {
    if (!options || typeof options !== "object" || Array.isArray(options)) {
      throw optimizerError("OPTIMIZER_INVALID_CONFIG", "Optimizer options must be an object.");
    }

    const allowed = new Set(Object.keys(DEFAULTS));
    for (const key of Object.keys(options)) {
      if (!allowed.has(key)) {
        throw optimizerError("OPTIMIZER_INVALID_CONFIG", "Unknown optimizer option: " + key);
      }
    }

    const config = { ...base, ...options };
    assertFinite(config.learningRate, "learningRate");
    assertFinite(config.beta1, "beta1");
    assertFinite(config.beta2, "beta2");
    assertFinite(config.epsilon, "epsilon");
    assertFinite(config.weightDecay, "weightDecay");
    assertFinite(config.maxGradNorm, "maxGradNorm");

    if (config.learningRate < 0) {
      throw optimizerError("OPTIMIZER_INVALID_CONFIG", "learningRate must be non-negative.");
    }
    if (config.beta1 < 0 || config.beta1 >= 1 || config.beta2 < 0 || config.beta2 >= 1) {
      throw optimizerError("OPTIMIZER_INVALID_CONFIG", "beta1 and beta2 must be in [0, 1).");
    }
    if (config.epsilon <= 0) {
      throw optimizerError("OPTIMIZER_INVALID_CONFIG", "epsilon must be positive.");
    }
    if (config.weightDecay < 0 || config.maxGradNorm < 0) {
      throw optimizerError("OPTIMIZER_INVALID_CONFIG", "weightDecay and maxGradNorm must be non-negative.");
    }

    return Object.freeze(config);
  }

  function copyMoment(values, length, name, nonNegative) {
    if (!Array.isArray(values) || values.length !== length) {
      throw optimizerError("OPTIMIZER_INVALID_STATE", name + " has an invalid length.");
    }

    const result = new Float32Array(length);
    for (let index = 0; index < length; index += 1) {
      const value = values[index];
      if (typeof value !== "number" || !Number.isFinite(value) || (nonNegative && value < 0)) {
        throw optimizerError("OPTIMIZER_INVALID_STATE", name + " contains an invalid value.");
      }
      result[index] = toFloat32(value, name + " contains a value outside float32 range.");
    }
    return result;
  }

  class Parameter {
    constructor(data, options = {}) {
      if (!(data instanceof Tensor) || data.disposed || data.dtype !== "float32") {
        throw optimizerError("OPTIMIZER_INVALID_PARAMETER", "Parameter data must be a live FP32 Tensor.");
      }
      if (!options || typeof options !== "object" || Array.isArray(options)) {
        throw optimizerError("OPTIMIZER_INVALID_PARAMETER", "Parameter options must be an object.");
      }

      const id = "parameter-" + nextParameterId++;
      const name = options.name === undefined ? id : options.name;
      const trainable = options.trainable === undefined ? true : options.trainable;
      const requiresGrad = options.requiresGrad === undefined ? true : options.requiresGrad;
      const decay = options.decay === undefined ? true : options.decay;
      if (typeof name !== "string" || name.trim().length === 0) {
        throw optimizerError("OPTIMIZER_INVALID_PARAMETER", "Parameter name must be a non-empty string.");
      }
      if (typeof trainable !== "boolean" || typeof requiresGrad !== "boolean" || typeof decay !== "boolean") {
        throw optimizerError("OPTIMIZER_INVALID_PARAMETER", "trainable, requiresGrad, and decay must be booleans.");
      }

      data.requiresGrad = requiresGrad;
      parameterStates.set(this, {
        id,
        name,
        data,
        trainable,
        decay,
        disposed: false
      });
    }

    get id() {
      return assertParameter(this).id;
    }

    get name() {
      return assertParameter(this).name;
    }

    get data() {
      return assertParameter(this).data;
    }

    get grad() {
      return assertParameter(this).data.grad;
    }

    get trainable() {
      return assertParameter(this).trainable;
    }

    set trainable(value) {
      if (typeof value !== "boolean") {
        throw optimizerError("OPTIMIZER_INVALID_PARAMETER", "trainable must be a boolean.");
      }
      assertParameter(this).trainable = value;
    }

    get decay() {
      return assertParameter(this).decay;
    }

    set decay(value) {
      if (typeof value !== "boolean") {
        throw optimizerError("OPTIMIZER_INVALID_PARAMETER", "decay must be a boolean.");
      }
      assertParameter(this).decay = value;
    }

    get requiresGrad() {
      return assertParameter(this).data.requiresGrad;
    }

    set requiresGrad(value) {
      assertParameter(this).data.requiresGrad = value;
    }

    get version() {
      return assertParameter(this).data.version;
    }

    get disposed() {
      const state = parameterStates.get(this);
      return !state || state.disposed;
    }

    dispose() {
      const state = parameterStates.get(this);
      if (state) state.disposed = true;
    }
  }

  class AdamW {
    constructor(parameters = [], options = {}) {
      if (!Array.isArray(parameters)) {
        throw optimizerError("OPTIMIZER_INVALID_ARGUMENT", "Parameters must be an array.");
      }

      this.config = normalizeConfig(options);
      this.parameters = [];
      this.states = new Map();
      this.disposed = false;
      this.busy = false;
      for (const parameter of parameters) this.addParameter(parameter);
    }

    assertOpen() {
      if (this.disposed) {
        throw optimizerError("OPTIMIZER_DISPOSED", "Optimizer has been disposed.");
      }
    }

    addParameter(parameter) {
      this.assertOpen();
      const state = assertParameter(parameter);
      if (this.parameters.includes(parameter)) {
        throw optimizerError("OPTIMIZER_DUPLICATE_PARAMETER", "Parameter is already registered.");
      }
      if (this.parameters.some((item) => assertParameter(item).name === state.name)) {
        throw optimizerError("OPTIMIZER_DUPLICATE_PARAMETER", "Parameter names must be unique in an optimizer.");
      }
      const owner = optimizerOwners.get(parameter);
      if (owner && owner !== this) {
        throw optimizerError("OPTIMIZER_PARAMETER_CONFLICT", "Parameter is already owned by another optimizer.");
      }

      optimizerOwners.set(parameter, this);
      this.parameters.push(parameter);
      return this;
    }

    zeroGrad() {
      this.assertOpen();
      for (const parameter of this.parameters) {
        assertParameter(parameter).data.zeroGrad();
      }
    }

    step() {
      this.assertOpen();
      if (this.busy) {
        throw optimizerError("OPTIMIZER_BUSY", "Optimizer step is already running.");
      }
      this.busy = true;

      try {
        const targets = [];
        let maximumAbsoluteGradient = 0;
        for (const parameter of this.parameters) {
          const parameterState = assertParameter(parameter);
          if (!parameterState.trainable) continue;

          const data = parameterState.data;
          const gradient = data.grad;
          if (gradient === null) continue;
          if (gradient.disposed || gradient.dtype !== data.dtype ||
            gradient.shape.length !== data.shape.length ||
            gradient.shape.some((dimension, axis) => dimension !== data.shape[axis])) {
            throw optimizerError("OPTIMIZER_INVALID_GRADIENT", "Gradient Shape or DType does not match Parameter.");
          }

          const gradientValues = gradient.toArray();
          for (const value of gradientValues) {
            if (!Number.isFinite(value)) {
              throw optimizerError("OPTIMIZER_NUMERICAL_ERROR", "Gradient contains NaN or Infinity.");
            }
            maximumAbsoluteGradient = Math.max(maximumAbsoluteGradient, Math.abs(value));
          }
          targets.push({ parameter, parameterState, data, gradientValues });
        }

        if (targets.length === 0 || this.config.learningRate === 0) {
          return { updated: 0, globalNorm: 0, clipScale: 1 };
        }

        let scaledSquares = 0;
        if (maximumAbsoluteGradient > 0) {
          for (const target of targets) {
            for (const value of target.gradientValues) {
              const scaled = value / maximumAbsoluteGradient;
              scaledSquares += scaled * scaled;
            }
          }
        }
        const globalNorm = maximumAbsoluteGradient * Math.sqrt(scaledSquares);
        if (!Number.isFinite(globalNorm)) {
          throw optimizerError("OPTIMIZER_NUMERICAL_ERROR", "Global gradient norm is not finite.");
        }
        const clipScale = globalNorm > this.config.maxGradNorm && globalNorm > 0
          ? this.config.maxGradNorm / globalNorm
          : 1;

        const candidates = [];
        for (const target of targets) {
          const previousState = this.states.get(target.parameter);
          const size = target.data.size;
          const previousM = previousState ? previousState.m : new Float32Array(size);
          const previousV = previousState ? previousState.v : new Float32Array(size);
          const step = (previousState ? previousState.step : 0) + 1;
          if (!Number.isSafeInteger(step)) {
            throw optimizerError("OPTIMIZER_NUMERICAL_ERROR", "Parameter update count exceeded the safe integer range.");
          }

          const nextM = new Float32Array(size);
          const nextV = new Float32Array(size);
          const nextData = target.data.toArray();
          for (let index = 0; index < size; index += 1) {
            const gradient = toFloat32(target.gradientValues[index] * clipScale);
            nextM[index] = toFloat32(
              this.config.beta1 * previousM[index] + (1 - this.config.beta1) * gradient
            );
            nextV[index] = toFloat32(
              this.config.beta2 * previousV[index] + (1 - this.config.beta2) * gradient * gradient
            );

            const correctedM = nextM[index] / (1 - this.config.beta1 ** step);
            const correctedV = nextV[index] / (1 - this.config.beta2 ** step);
            const weight = nextData[index];
            const decayTerm = target.parameterState.decay
              ? this.config.weightDecay * weight
              : 0;
            const update = correctedM / (Math.sqrt(correctedV) + this.config.epsilon) + decayTerm;
            nextData[index] = toFloat32(
              weight - this.config.learningRate * update
            );
          }

          const preparedData = KNA._internal.prepareTensorData(target.data, nextData);
          candidates.push({
            parameter: target.parameter,
            preparedData,
            state: { m: nextM, v: nextV, step }
          });
        }

        KNA._internal.commitTensorDataBatch(
          candidates.map((candidate) => [
            assertParameter(candidate.parameter).data,
            candidate.preparedData
          ])
        );
        for (const candidate of candidates) {
          this.states.set(candidate.parameter, candidate.state);
        }

        return { updated: candidates.length, globalNorm, clipScale };
      } finally {
        this.busy = false;
      }
    }

    stateDict() {
      this.assertOpen();
      return {
        type: "KNA.AdamW",
        version: 1,
        config: { ...this.config },
        parameters: this.parameters.map((parameter) => {
          const parameterState = assertParameter(parameter);
          const state = this.states.get(parameter);
          const size = parameterState.data.size;
          return {
            name: parameterState.name,
            shape: parameterState.data.shape.slice(),
            dtype: parameterState.data.dtype,
            decay: parameterState.decay,
            step: state ? state.step : 0,
            m: state ? Array.from(state.m) : new Array(size).fill(0),
            v: state ? Array.from(state.v) : new Array(size).fill(0)
          };
        })
      };
    }

    loadStateDict(value) {
      this.assertOpen();
      if (!value || value.type !== "KNA.AdamW" || value.version !== 1 ||
        !Array.isArray(value.parameters)) {
        throw optimizerError("OPTIMIZER_INVALID_STATE", "Invalid AdamW state dictionary.");
      }
      if (value.parameters.length !== this.parameters.length) {
        throw optimizerError("OPTIMIZER_INVALID_STATE", "State dictionary Parameter count does not match.");
      }

      const config = normalizeConfig(value.config, DEFAULTS);
      const byName = new Map(this.parameters.map((parameter) => [assertParameter(parameter).name, parameter]));
      const nextStates = new Map();
      const seen = new Set();

      for (const entry of value.parameters) {
        if (!entry || typeof entry.name !== "string" || seen.has(entry.name)) {
          throw optimizerError("OPTIMIZER_INVALID_STATE", "State dictionary contains an invalid Parameter name.");
        }
        seen.add(entry.name);
        const parameter = byName.get(entry.name);
        if (!parameter) {
          throw optimizerError("OPTIMIZER_INVALID_STATE", "State dictionary contains an unknown Parameter.");
        }
        const parameterState = assertParameter(parameter);
        if (entry.dtype !== parameterState.data.dtype ||
          entry.decay !== parameterState.decay ||
          !Array.isArray(entry.shape) ||
          entry.shape.length !== parameterState.data.shape.length ||
          entry.shape.some((dimension, axis) => dimension !== parameterState.data.shape[axis]) ||
          !Number.isSafeInteger(entry.step) || entry.step < 0) {
          throw optimizerError("OPTIMIZER_INVALID_STATE", "State dictionary metadata does not match Parameter.");
        }
        nextStates.set(parameter, {
          m: copyMoment(entry.m, parameterState.data.size, "m", false),
          v: copyMoment(entry.v, parameterState.data.size, "v", true),
          step: entry.step
        });
      }

      this.config = config;
      this.states = nextStates;
    }

    dispose() {
      if (this.disposed) return;
      for (const parameter of this.parameters) {
        if (optimizerOwners.get(parameter) === this) optimizerOwners.delete(parameter);
      }
      this.states.clear();
      this.parameters.length = 0;
      this.disposed = true;
    }
  }

  KNA.Parameter = Parameter;
  KNA.AdamW = AdamW;
})(globalThis);
