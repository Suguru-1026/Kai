(function (global) {
  "use strict";

  const KNA = global.KNA;
  const Tensor = KNA.Tensor;
  const ops = KNA.ops;

  function layerError(code, message) {
    const error = new Error(message);
    error.code = code;
    return error;
  }

  function positiveInteger(value, name) {
    if (!Number.isSafeInteger(value) || value <= 0) {
      throw layerError("LAYER_INVALID_CONFIG", name + " must be a positive safe integer.");
    }
    return value;
  }

  function finiteNumber(value, name) {
    if (typeof value !== "number" || !Number.isFinite(value)) {
      throw layerError("LAYER_INVALID_CONFIG", name + " must be finite.");
    }
    return value;
  }

  function requireRng(rng) {
    if (!rng || typeof rng.nextFloat !== "function" || typeof rng.normal !== "function") {
      throw layerError("LAYER_INVALID_CONFIG", "A KNA RNG stream is required for random initialization.");
    }
    return rng;
  }

  function initializeParameter(parameter, strategy, rng, options = {}) {
    if (!(parameter instanceof KNA.Parameter) || parameter.disposed) {
      throw layerError("LAYER_INVALID_PARAMETER", "Initialization requires a live Parameter.");
    }
    if (!options || typeof options !== "object" || Array.isArray(options)) {
      throw layerError("LAYER_INVALID_CONFIG", "Initializer options must be an object.");
    }

    const data = parameter.data;
    const shape = data.shape;
    const values = new Float32Array(data.size);
    let random = null;
    let mean = 0;
    let standardDeviation = 0;
    let uniformLimit = 0;

    switch (strategy) {
      case "zeros":
        break;
      case "ones":
        values.fill(1);
        break;
      case "normal":
        random = requireRng(rng);
        mean = finiteNumber(options.mean === undefined ? 0 : options.mean, "mean");
        standardDeviation = finiteNumber(options.standardDeviation === undefined ? 0.02 : options.standardDeviation, "standardDeviation");
        if (standardDeviation < 0) {
          throw layerError("LAYER_INVALID_CONFIG", "standardDeviation must be non-negative.");
        }
        for (let index = 0; index < values.length; index += 1) {
          values[index] = Math.fround(random.normal(mean, standardDeviation));
        }
        break;
      case "uniform": {
        random = requireRng(rng);
        const minimum = finiteNumber(options.minimum === undefined ? -1 : options.minimum, "minimum");
        const maximum = finiteNumber(options.maximum === undefined ? 1 : options.maximum, "maximum");
        if (minimum >= maximum) {
          throw layerError("LAYER_INVALID_CONFIG", "minimum must be less than maximum.");
        }
        const width = maximum - minimum;
        for (let index = 0; index < values.length; index += 1) {
          values[index] = Math.fround(minimum + random.nextFloat() * width);
        }
        break;
      }
      case "xavier_uniform":
      case "xavier_normal":
      case "kaiming_uniform":
      case "kaiming_normal": {
        if (shape.length !== 2) {
          throw layerError("LAYER_INVALID_SHAPE", "Fan-based initialization requires a rank-2 weight.");
        }
        const fanIn = shape[0];
        const fanOut = shape[1];
        random = requireRng(rng);
        if (strategy === "xavier_uniform") {
          uniformLimit = Math.sqrt(6 / (fanIn + fanOut));
        } else if (strategy === "xavier_normal") {
          standardDeviation = Math.sqrt(2 / (fanIn + fanOut));
        } else if (strategy === "kaiming_uniform") {
          uniformLimit = Math.sqrt(6 / fanIn);
        } else {
          standardDeviation = Math.sqrt(2 / fanIn);
        }
        for (let index = 0; index < values.length; index += 1) {
          values[index] = Math.fround(strategy.endsWith("uniform")
            ? (random.nextFloat() * 2 - 1) * uniformLimit
            : random.normal(0, standardDeviation));
        }
        break;
      }
      case "transformer_scaled_normal": {
        random = requireRng(rng);
        const numberOfLayers = positiveInteger(options.numLayers, "numLayers");
        standardDeviation = 0.02 / Math.sqrt(2 * numberOfLayers);
        for (let index = 0; index < values.length; index += 1) {
          values[index] = Math.fround(random.normal(0, standardDeviation));
        }
        break;
      }
      default:
        throw layerError("LAYER_INITIALIZER_UNSUPPORTED", "Unsupported initializer: " + strategy);
    }

    const prepared = KNA._internal.prepareTensorData(data, values);
    KNA._internal.commitTensorDataBatch([[data, prepared]]);
    return parameter;
  }

  class Layer {
    constructor(name) {
      if (typeof name !== "string" || name.trim().length === 0) {
        throw layerError("LAYER_INVALID_CONFIG", "Layer name must be a non-empty string.");
      }
      this.name = name;
      this.status = "CREATED";
      this.training = false;
      this._parameters = [];
      this._children = [];
    }

    initialize(context = {}) {
      if (this.status === "READY") {
        throw layerError("LAYER_ALREADY_INITIALIZED", "Layer is already initialized.");
      }
      if (this.status === "DISPOSED") {
        throw layerError("LAYER_DISPOSED", "Layer has been disposed.");
      }
      if (!context || typeof context !== "object") {
        throw layerError("LAYER_INVALID_CONFIG", "Initialization context must be an object.");
      }

      this.status = "INITIALIZING";
      try {
        this._initialize(context);
        this.status = "READY";
        return this;
      } catch (error) {
        this.status = "FAILED";
        this._disposeResources();
        throw error;
      }
    }

    _initialize() {}

    forward(input) {
      if (this.status !== "READY") {
        throw layerError(this.status === "DISPOSED" ? "LAYER_DISPOSED" : "LAYER_NOT_INITIALIZED", "Layer is not ready.");
      }
      return this._forward(input);
    }

    _forward() {
      throw layerError("LAYER_FORWARD_UNSUPPORTED", "Layer has no forward implementation.");
    }

    addParameter(parameter) {
      this._parameters.push(parameter);
      return parameter;
    }

    addChild(layer) {
      this._children.push(layer);
      return layer;
    }

    parameters() {
      const all = [...this._parameters];
      for (const child of this._children) all.push(...child.parameters());
      return Array.from(new Set(all));
    }

    children() {
      return this._children.slice();
    }

    train(mode = true) {
      if (typeof mode !== "boolean") {
        throw layerError("LAYER_INVALID_CONFIG", "Training mode must be a boolean.");
      }
      if (this.status === "DISPOSED") throw layerError("LAYER_DISPOSED", "Layer has been disposed.");
      this.training = mode;
      for (const child of this._children) child.train(mode);
      return this;
    }

    isDisposed() {
      return this.status === "DISPOSED";
    }

    _disposeResources() {
      for (const child of this._children) child.dispose();
      for (const parameter of this._parameters) {
        if (!parameter.disposed) {
          parameter.data.dispose();
          parameter.dispose();
        }
      }
      this._parameters.length = 0;
      this._children.length = 0;
    }

    dispose() {
      if (this.status === "DISPOSED") return;
      this.status = "DISPOSING";
      this._disposeResources();
      this.status = "DISPOSED";
    }
  }

  class Linear extends Layer {
    constructor(options) {
      if (!options || typeof options !== "object") {
        throw layerError("LAYER_INVALID_CONFIG", "Linear options are required.");
      }
      const name = options.name || "linear";
      super(name);
      this.inFeatures = positiveInteger(options.inFeatures, "inFeatures");
      this.outFeatures = positiveInteger(options.outFeatures, "outFeatures");
      this.useBias = options.useBias === true;
      this.initializer = options.initializer || "xavier_uniform";
      this.initializerOptions = options.initializerOptions || {};
      this.weight = null;
      this.bias = null;
    }

    _initialize(context) {
      const weightData = Tensor.zeros([this.inFeatures, this.outFeatures]);
      this.weight = this.addParameter(new KNA.Parameter(weightData, {
        name: this.name + ".weight",
        decay: true
      }));
      initializeParameter(this.weight, this.initializer, context.rng, this.initializerOptions);

      if (this.useBias) {
        const biasData = Tensor.zeros([this.outFeatures]);
        this.bias = this.addParameter(new KNA.Parameter(biasData, {
          name: this.name + ".bias",
          decay: false
        }));
        initializeParameter(this.bias, "zeros", context.rng);
      }
    }

    _forward(input) {
      if (!(input instanceof Tensor) || input.disposed || input.rank < 1) {
        throw layerError("LAYER_INVALID_INPUT", "Linear input must be a live Tensor with rank at least 1.");
      }
      if (input.shape[input.rank - 1] !== this.inFeatures) {
        throw layerError("LAYER_INVALID_SHAPE", "Linear input feature dimension does not match inFeatures.");
      }

      const leadingShape = input.shape.slice(0, -1);
      const rowCount = leadingShape.length === 0
        ? 1
        : leadingShape.reduce((product, dimension) => product * dimension, 1);
      const flattened = ops.reshape(input, [rowCount, this.inFeatures]);
      const projection = ops.matmul(flattened, this.weight.data);
      const biased = this.bias ? ops.add(projection, this.bias.data) : projection;
      const outputShape = [...leadingShape, this.outFeatures];
      const output = ops.reshape(biased, outputShape);
      flattened.dispose();
      if (biased !== projection) biased.dispose();
      projection.dispose();
      return output;
    }
  }

  function flattenTokenIds(input, vocabSize) {
    if (!Array.isArray(input)) {
      throw layerError("LAYER_INVALID_INPUT", "Embedding input must be an integer array or matrix.");
    }
    const shape = [];
    const values = [];

    function visit(value, depth) {
      if (Array.isArray(value)) {
        if (depth > 1) throw layerError("LAYER_INVALID_SHAPE", "Embedding supports one or two ID dimensions.");
        if (shape.length === depth) shape.push(value.length);
        else if (shape[depth] !== value.length) throw layerError("LAYER_INVALID_SHAPE", "Embedding IDs must be rectangular.");
        for (const child of value) visit(child, depth + 1);
        return;
      }
      if (!Number.isSafeInteger(value) || value < 0 || value >= vocabSize) {
        throw layerError("LAYER_INVALID_INPUT", "Embedding Token ID is outside the vocabulary.");
      }
      values.push(value);
    }

    visit(input, 0);
    return { shape, values };
  }

  class Embedding extends Layer {
    constructor(options) {
      if (!options || typeof options !== "object") {
        throw layerError("LAYER_INVALID_CONFIG", "Embedding options are required.");
      }
      super(options.name || "embedding");
      this.vocabSize = positiveInteger(options.vocabSize, "vocabSize");
      this.hiddenSize = positiveInteger(options.hiddenSize, "hiddenSize");
      this.standardDeviation = options.standardDeviation === undefined ? 0.02 : finiteNumber(options.standardDeviation, "standardDeviation");
      if (this.standardDeviation < 0) throw layerError("LAYER_INVALID_CONFIG", "standardDeviation must be non-negative.");
      this.weight = null;
    }

    _initialize(context) {
      const data = Tensor.zeros([this.vocabSize, this.hiddenSize]);
      this.weight = this.addParameter(new KNA.Parameter(data, {
        name: this.name + ".weight",
        decay: true
      }));
      initializeParameter(this.weight, "normal", context.rng, {
        mean: 0,
        standardDeviation: this.standardDeviation
      });
    }

    _forward(tokenIds) {
      const flattened = flattenTokenIds(tokenIds, this.vocabSize);
      const gathered = ops.gather(this.weight.data, flattened.values, 0);
      const outputShape = [...flattened.shape, this.hiddenSize];
      const output = ops.reshape(gathered, outputShape);
      gathered.dispose();
      return output;
    }
  }

  class RMSNorm extends Layer {
    constructor(options) {
      if (!options || typeof options !== "object") {
        throw layerError("LAYER_INVALID_CONFIG", "RMSNorm options are required.");
      }
      super(options.name || "rmsnorm");
      this.hiddenSize = positiveInteger(options.hiddenSize, "hiddenSize");
      this.epsilon = finiteNumber(options.epsilon === undefined ? 1e-6 : options.epsilon, "epsilon");
      if (this.epsilon <= 0) throw layerError("LAYER_INVALID_CONFIG", "epsilon must be positive.");
      this.scale = null;
    }

    _initialize(context) {
      const data = Tensor.zeros([this.hiddenSize]);
      this.scale = this.addParameter(new KNA.Parameter(data, {
        name: this.name + ".scale",
        decay: false
      }));
      initializeParameter(this.scale, "ones", context.rng);
    }

    _forward(input) {
      if (!(input instanceof Tensor) || input.disposed || input.rank < 1 || input.shape[input.rank - 1] !== this.hiddenSize) {
        throw layerError("LAYER_INVALID_SHAPE", "RMSNorm input must end in hiddenSize.");
      }

      const square = ops.mul(input, input);
      const meanSquare = ops.mean(square, { axis: -1, keepDims: true });
      const epsilon = Tensor.scalar(this.epsilon);
      const denominator = ops.sqrt(ops.add(meanSquare, epsilon));
      const normalized = ops.div(input, denominator);
      const output = ops.mul(normalized, this.scale.data);
      square.dispose();
      meanSquare.dispose();
      epsilon.dispose();
      denominator.dispose();
      normalized.dispose();
      return output;
    }
  }

  class SwiGLU extends Layer {
    constructor(options) {
      if (!options || typeof options !== "object") {
        throw layerError("LAYER_INVALID_CONFIG", "SwiGLU options are required.");
      }
      super(options.name || "swiglu");
      this.hiddenSize = positiveInteger(options.hiddenSize, "hiddenSize");
      this.intermediateSize = positiveInteger(options.intermediateSize, "intermediateSize");
      this.numLayers = positiveInteger(options.numLayers, "numLayers");
      this.useBias = options.useBias === true;
      this.gate = this.addChild(new Linear({
        name: this.name + ".gate",
        inFeatures: this.hiddenSize,
        outFeatures: this.intermediateSize,
        useBias: this.useBias,
        initializer: "xavier_uniform"
      }));
      this.up = this.addChild(new Linear({
        name: this.name + ".up",
        inFeatures: this.hiddenSize,
        outFeatures: this.intermediateSize,
        useBias: this.useBias,
        initializer: "xavier_uniform"
      }));
      this.down = this.addChild(new Linear({
        name: this.name + ".down",
        inFeatures: this.intermediateSize,
        outFeatures: this.hiddenSize,
        useBias: this.useBias,
        initializer: "transformer_scaled_normal",
        initializerOptions: { numLayers: this.numLayers }
      }));
    }

    _initialize(context) {
      this.gate.initialize(context);
      this.up.initialize(context);
      this.down.initialize(context);
    }

    _forward(input) {
      const gate = this.gate.forward(input);
      const up = this.up.forward(input);
      const activated = ops.silu(gate);
      const product = ops.mul(activated, up);
      const output = this.down.forward(product);
      gate.dispose();
      up.dispose();
      activated.dispose();
      product.dispose();
      return output;
    }
  }

  KNA.layers = Object.freeze({
    Layer,
    Linear,
    Embedding,
    RMSNorm,
    SwiGLU,
    initializeParameter
  });
})(globalThis);
