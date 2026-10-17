(function (global) {
  "use strict";

  const KNA = global.KNA;
  const Tensor = KNA.Tensor;
  const ops = KNA.ops;
  const DEFAULT_CONFIG = Object.freeze({
    architecture: "kna.rikka.decoder.v1",
    vocabSize: 512,
    hiddenSize: 64,
    numLayers: 2,
    numHeads: 4,
    intermediateSize: 176,
    maxSequenceLength: 128,
    normEpsilon: 1e-6,
    ropeTheta: 10000,
    tieWordEmbeddings: true,
    useBias: false,
    embeddingStd: 0.02
  });

  function modelError(code, message) {
    const error = new Error(message);
    error.code = code;
    return error;
  }

  function positiveInteger(value, name) {
    if (!Number.isSafeInteger(value) || value <= 0) {
      throw modelError("MODEL_INVALID_CONFIG", name + " must be a positive safe integer.");
    }
    return value;
  }

  function validateConfig(options) {
    if (!options || typeof options !== "object" || Array.isArray(options)) {
      throw modelError("MODEL_INVALID_CONFIG", "Model config must be an object.");
    }
    const allowed = new Set(Object.keys(DEFAULT_CONFIG));
    for (const key of Object.keys(options)) {
      if (!allowed.has(key)) throw modelError("MODEL_INVALID_CONFIG", "Unknown model config: " + key);
    }

    const config = { ...DEFAULT_CONFIG, ...options };
    if (config.architecture !== DEFAULT_CONFIG.architecture) {
      throw modelError("MODEL_UNSUPPORTED_ARCHITECTURE", "Unsupported architecture.");
    }
    positiveInteger(config.vocabSize, "vocabSize");
    positiveInteger(config.hiddenSize, "hiddenSize");
    positiveInteger(config.numLayers, "numLayers");
    positiveInteger(config.numHeads, "numHeads");
    positiveInteger(config.intermediateSize, "intermediateSize");
    positiveInteger(config.maxSequenceLength, "maxSequenceLength");
    if (config.hiddenSize % config.numHeads !== 0) {
      throw modelError("MODEL_INVALID_CONFIG", "hiddenSize must be divisible by numHeads.");
    }
    if ((config.hiddenSize / config.numHeads) % 2 !== 0) {
      throw modelError("MODEL_INVALID_CONFIG", "RoPE head dimension must be even.");
    }
    if (typeof config.tieWordEmbeddings !== "boolean" || typeof config.useBias !== "boolean") {
      throw modelError("MODEL_INVALID_CONFIG", "tieWordEmbeddings and useBias must be booleans.");
    }
    if (!Number.isFinite(config.normEpsilon) || config.normEpsilon <= 0) {
      throw modelError("MODEL_INVALID_CONFIG", "normEpsilon must be a positive finite number.");
    }
    if (!Number.isFinite(config.ropeTheta) || config.ropeTheta <= 0) {
      throw modelError("MODEL_INVALID_CONFIG", "ropeTheta must be a positive finite number.");
    }
    if (!Number.isFinite(config.embeddingStd) || config.embeddingStd < 0) {
      throw modelError("MODEL_INVALID_CONFIG", "embeddingStd must be a non-negative finite number.");
    }
    return Object.freeze(config);
  }

  function validateIds(inputIds, config) {
    if (!Array.isArray(inputIds) || inputIds.length === 0 || !Array.isArray(inputIds[0])) {
      throw modelError("MODEL_INVALID_INPUT", "inputIds must be a non-empty rectangular number matrix.");
    }
    const batchSize = inputIds.length;
    const sequenceLength = inputIds[0].length;
    if (sequenceLength === 0 || sequenceLength > config.maxSequenceLength) {
      throw modelError("MODEL_CONTEXT_LENGTH", "Input sequence length is outside the model context.");
    }
    for (const row of inputIds) {
      if (!Array.isArray(row) || row.length !== sequenceLength) {
        throw modelError("MODEL_INVALID_INPUT", "inputIds must be rectangular.");
      }
      for (const id of row) {
        if (!Number.isSafeInteger(id) || id < 0 || id >= config.vocabSize) {
          throw modelError("MODEL_INVALID_INPUT", "inputIds contains an invalid Token ID.");
        }
      }
    }
    return { batchSize, sequenceLength };
  }

  function splitHeads(input, batchSize, sequenceLength, numHeads, headDim) {
    const reshaped = ops.reshape(input, [batchSize, sequenceLength, numHeads, headDim]);
    const transposed = ops.transpose(reshaped, [0, 2, 1, 3]);
    reshaped.dispose();
    return transposed;
  }

  function applyRoPE(input, theta) {
    const shape = input.shape;
    if (shape.length !== 4 || shape[3] % 2 !== 0) {
      throw modelError("MODEL_INVALID_SHAPE", "RoPE input must be [batch, heads, sequence, evenHeadDim].");
    }

    const [batchSize, numHeads, sequenceLength, headDim] = shape;
    const inputSize = input.size;
    const halfDim = headDim / 2;
    const inputValues = input.toArray();
    const output = new Float32Array(input.size);
    const cosines = new Float32Array(sequenceLength * halfDim);
    const sines = new Float32Array(sequenceLength * halfDim);

    for (let position = 0; position < sequenceLength; position += 1) {
      for (let pair = 0; pair < halfDim; pair += 1) {
        const frequency = theta ** (-2 * pair / headDim);
        const angle = position * frequency;
        cosines[position * halfDim + pair] = Math.cos(angle);
        sines[position * halfDim + pair] = Math.sin(angle);
      }
    }

    for (let batch = 0; batch < batchSize; batch += 1) {
      for (let head = 0; head < numHeads; head += 1) {
        for (let position = 0; position < sequenceLength; position += 1) {
          const base = ((batch * numHeads + head) * sequenceLength + position) * headDim;
          for (let pair = 0; pair < halfDim; pair += 1) {
            const even = inputValues[base + pair * 2];
            const odd = inputValues[base + pair * 2 + 1];
            const cosine = cosines[position * halfDim + pair];
            const sine = sines[position * halfDim + pair];
            output[base + pair * 2] = Math.fround(even * cosine - odd * sine);
            output[base + pair * 2 + 1] = Math.fround(even * sine + odd * cosine);
          }
        }
      }
    }

    return KNA._internal.createCustomOperation(
      shape,
      output,
      [input],
      "rope",
      (outputGradient) => {
        const inputGradient = new Float32Array(inputSize);
        for (let batch = 0; batch < batchSize; batch += 1) {
          for (let head = 0; head < numHeads; head += 1) {
            for (let position = 0; position < sequenceLength; position += 1) {
              const base = ((batch * numHeads + head) * sequenceLength + position) * headDim;
              for (let pair = 0; pair < halfDim; pair += 1) {
                const evenGradient = outputGradient[base + pair * 2];
                const oddGradient = outputGradient[base + pair * 2 + 1];
                const cosine = cosines[position * halfDim + pair];
                const sine = sines[position * halfDim + pair];
                inputGradient[base + pair * 2] = Math.fround(evenGradient * cosine + oddGradient * sine);
                inputGradient[base + pair * 2 + 1] = Math.fround(-evenGradient * sine + oddGradient * cosine);
              }
            }
          }
        }
        return [inputGradient];
      }
    );
  }

  function makeCausalMask(sequenceLength) {
    const values = new Float32Array(sequenceLength * sequenceLength);
    for (let query = 0; query < sequenceLength; query += 1) {
      for (let key = query + 1; key < sequenceLength; key += 1) {
        values[query * sequenceLength + key] = 1;
      }
    }
    return Tensor.fromArray(values, [1, 1, sequenceLength, sequenceLength]);
  }

  class CausalSelfAttention {
    constructor(config, name) {
      this.config = config;
      this.name = name;
      this.status = "CREATED";
      this.headDim = config.hiddenSize / config.numHeads;
      const Linear = KNA.layers.Linear;
      this.query = new Linear({ name: name + ".q_proj", inFeatures: config.hiddenSize, outFeatures: config.hiddenSize, useBias: config.useBias, initializer: "xavier_uniform" });
      this.key = new Linear({ name: name + ".k_proj", inFeatures: config.hiddenSize, outFeatures: config.hiddenSize, useBias: config.useBias, initializer: "xavier_uniform" });
      this.value = new Linear({ name: name + ".v_proj", inFeatures: config.hiddenSize, outFeatures: config.hiddenSize, useBias: config.useBias, initializer: "xavier_uniform" });
      this.output = new Linear({ name: name + ".out_proj", inFeatures: config.hiddenSize, outFeatures: config.hiddenSize, useBias: config.useBias, initializer: "xavier_uniform" });
      this.childrenList = [this.query, this.key, this.value, this.output];
    }

    initialize(context) {
      if (this.status !== "CREATED") throw modelError("MODEL_ALREADY_INITIALIZED", "Attention is already initialized.");
      for (const child of this.childrenList) child.initialize(context);
      this.status = "READY";
    }

    parameters() {
      return this.childrenList.flatMap((child) => child.parameters());
    }

    forward(input) {
      if (this.status !== "READY") throw modelError("MODEL_NOT_INITIALIZED", "Attention is not initialized.");
      const shape = input.shape;
      if (shape.length !== 3 || shape[2] !== this.config.hiddenSize) {
        throw modelError("MODEL_INVALID_SHAPE", "Attention input must be [batch, sequence, hiddenSize].");
      }
      const [batchSize, sequenceLength] = shape;
      const queryProjection = this.query.forward(input);
      const keyProjection = this.key.forward(input);
      const valueProjection = this.value.forward(input);
      const queryHeads = splitHeads(queryProjection, batchSize, sequenceLength, this.config.numHeads, this.headDim);
      const keyHeads = splitHeads(keyProjection, batchSize, sequenceLength, this.config.numHeads, this.headDim);
      const valueHeads = splitHeads(valueProjection, batchSize, sequenceLength, this.config.numHeads, this.headDim);
      queryProjection.dispose();
      keyProjection.dispose();
      valueProjection.dispose();

      const rotatedQuery = applyRoPE(queryHeads, this.config.ropeTheta);
      const rotatedKey = applyRoPE(keyHeads, this.config.ropeTheta);
      const transposedKey = ops.transpose(rotatedKey, [0, 1, 3, 2]);
      const rawScores = ops.matmul(rotatedQuery, transposedKey);
      const scale = Tensor.scalar(1 / Math.sqrt(this.headDim));
      const scaledScores = ops.mul(rawScores, scale);
      const causalMask = makeCausalMask(sequenceLength);
      const maskedScores = ops.maskedFill(scaledScores, causalMask, -1e30);
      const probabilities = ops.softmax(maskedScores, -1);
      const contextHeads = ops.matmul(probabilities, valueHeads);
      const mergedHeads = ops.transpose(contextHeads, [0, 2, 1, 3]);
      const merged = ops.reshape(mergedHeads, [batchSize, sequenceLength, this.config.hiddenSize]);
      const output = this.output.forward(merged);

      queryHeads.dispose();
      keyHeads.dispose();
      valueHeads.dispose();
      rotatedQuery.dispose();
      rotatedKey.dispose();
      transposedKey.dispose();
      rawScores.dispose();
      scale.dispose();
      scaledScores.dispose();
      causalMask.dispose();
      maskedScores.dispose();
      probabilities.dispose();
      contextHeads.dispose();
      mergedHeads.dispose();
      merged.dispose();
      return output;
    }

    dispose() {
      if (this.status === "DISPOSED") return;
      for (const child of this.childrenList) child.dispose();
      this.status = "DISPOSED";
    }
  }

  class DecoderBlock {
    constructor(config, index) {
      this.config = config;
      this.name = "blocks." + index;
      const RMSNorm = KNA.layers.RMSNorm;
      this.attentionNorm = new RMSNorm({ name: this.name + ".attention_norm", hiddenSize: config.hiddenSize, epsilon: config.normEpsilon });
      this.attention = new CausalSelfAttention(config, this.name + ".attention");
      this.feedForwardNorm = new RMSNorm({ name: this.name + ".ffn_norm", hiddenSize: config.hiddenSize, epsilon: config.normEpsilon });
      this.feedForward = new KNA.layers.SwiGLU({ name: this.name + ".ffn", hiddenSize: config.hiddenSize, intermediateSize: config.intermediateSize, numLayers: config.numLayers, useBias: config.useBias });
      this.childrenList = [this.attentionNorm, this.attention, this.feedForwardNorm, this.feedForward];
      this.status = "CREATED";
    }

    initialize(context) {
      if (this.status !== "CREATED") throw modelError("MODEL_ALREADY_INITIALIZED", "DecoderBlock is already initialized.");
      for (const child of this.childrenList) child.initialize(context);
      this.status = "READY";
    }

    parameters() {
      return this.childrenList.flatMap((child) => child.parameters());
    }

    forward(input) {
      if (this.status !== "READY") throw modelError("MODEL_NOT_INITIALIZED", "DecoderBlock is not initialized.");
      const normalizedAttention = this.attentionNorm.forward(input);
      const attentionOutput = this.attention.forward(normalizedAttention);
      const afterAttention = ops.add(input, attentionOutput);
      const normalizedFeedForward = this.feedForwardNorm.forward(afterAttention);
      const feedForwardOutput = this.feedForward.forward(normalizedFeedForward);
      const output = ops.add(afterAttention, feedForwardOutput);
      normalizedAttention.dispose();
      attentionOutput.dispose();
      afterAttention.dispose();
      normalizedFeedForward.dispose();
      feedForwardOutput.dispose();
      return output;
    }

    dispose() {
      if (this.status === "DISPOSED") return;
      for (const child of this.childrenList) child.dispose();
      this.status = "DISPOSED";
    }
  }

  class RikkaModel {
    constructor(config = {}, options = {}) {
      this.tokenizer = options.tokenizer || null;
      const configured = { ...config };
      if (configured.vocabSize === undefined && this.tokenizer) {
        configured.vocabSize = this.tokenizer.getVocabSize();
      }
      this.config = validateConfig(configured);
      if (this.tokenizer && this.tokenizer.getVocabSize() !== this.config.vocabSize) {
        throw modelError("MODEL_TOKENIZER_MISMATCH", "Tokenizer vocabulary size does not match ModelConfig.");
      }
      this.status = "CREATED";
      this.embedding = new KNA.layers.Embedding({
        name: "embedding.token",
        vocabSize: this.config.vocabSize,
        hiddenSize: this.config.hiddenSize,
        standardDeviation: this.config.embeddingStd
      });
      this.blocks = Array.from({ length: this.config.numLayers }, (_, index) => new DecoderBlock(this.config, index));
      this.finalNorm = new KNA.layers.RMSNorm({ name: "final_norm", hiddenSize: this.config.hiddenSize, epsilon: this.config.normEpsilon });
      this.lmHead = this.config.tieWordEmbeddings
        ? null
        : new KNA.layers.Linear({ name: "lm_head", inFeatures: this.config.hiddenSize, outFeatures: this.config.vocabSize, useBias: false, initializer: "xavier_uniform" });
    }

    initialize(options = {}) {
      if (this.status !== "CREATED") throw modelError("MODEL_ALREADY_INITIALIZED", "Model has already been initialized.");
      if (!options || typeof options !== "object") throw modelError("MODEL_INVALID_CONFIG", "Initialize options must be an object.");
      const seed = options.seed === undefined ? 12345n : options.seed;
      const rng = options.rng || KNA.RNG.create(seed, "initialization");
      const context = { rng };
      this.status = "INITIALIZING";
      try {
        this.embedding.initialize(context);
        for (const block of this.blocks) block.initialize(context);
        this.finalNorm.initialize(context);
        if (this.lmHead) this.lmHead.initialize(context);
        this.status = "READY";
        return this;
      } catch (error) {
        this.status = "FAILED";
        this._disposeLayers();
        throw error;
      }
    }

    assertReady() {
      if (this.status !== "READY") {
        throw modelError(this.status === "DISPOSED" ? "MODEL_DISPOSED" : "MODEL_NOT_INITIALIZED", "Model is not ready.");
      }
    }

    parameters() {
      const parameters = [
        ...this.embedding.parameters(),
        ...this.blocks.flatMap((block) => block.parameters()),
        ...this.finalNorm.parameters()
      ];
      if (this.lmHead) parameters.push(...this.lmHead.parameters());
      return Array.from(new Set(parameters));
    }

    forward(inputIds) {
      this.assertReady();
      validateIds(inputIds, this.config);
      const input = this.embedding.forward(inputIds);
      let hidden = input;
      for (const block of this.blocks) {
        const next = block.forward(hidden);
        if (hidden !== input) hidden.dispose();
        hidden = next;
      }

      const normalized = this.finalNorm.forward(hidden);
      let logits;
      if (this.config.tieWordEmbeddings) {
        const tiedWeight = ops.transpose(this.embedding.weight.data, [1, 0]);
        logits = ops.matmul(normalized, tiedWeight);
        tiedWeight.dispose();
      } else {
        logits = this.lmHead.forward(normalized);
      }
      input.dispose();
      if (hidden !== input) hidden.dispose();
      normalized.dispose();
      return logits;
    }

    trainStep(batch, optimizer) {
      this.assertReady();
      if (!(optimizer instanceof KNA.AdamW) || optimizer.disposed) {
        throw modelError("TRAIN_INVALID_OPTIMIZER", "A live KNA AdamW optimizer is required.");
      }
      if (!batch || typeof batch !== "object") {
        throw modelError("TRAIN_INVALID_BATCH", "Training batch must be an object.");
      }
      const modelParameters = this.parameters();
      if (optimizer.parameters.length !== modelParameters.length ||
        modelParameters.some((parameter) => !optimizer.parameters.includes(parameter))) {
        throw modelError("TRAIN_PARAMETER_MISMATCH", "Optimizer Parameter set does not match Model.");
      }

      optimizer.zeroGrad();
      let logits = null;
      let loss = null;
      try {
        logits = this.forward(batch.inputIds);
        loss = KNA.losses.crossEntropy(logits, batch.targetIds, {
          reduction: "mean",
          ignoreIndex: batch.ignoreIndex === undefined ? -100 : batch.ignoreIndex,
          lossMask: batch.lossMask
        });
        const lossValue = loss.item();
        loss.backward();
        const update = optimizer.step();
        optimizer.zeroGrad();
        return { loss: lossValue, ...update };
      } catch (error) {
        optimizer.zeroGrad();
        throw error;
      } finally {
        if (loss) loss.dispose();
        if (logits) logits.dispose();
      }
    }

    generate(input, options = {}) {
      this.assertReady();
      if (!options || typeof options !== "object" || Array.isArray(options)) {
        throw modelError("GENERATION_INVALID_OPTIONS", "Generation options must be an object.");
      }
      const maxNewTokens = options.maxNewTokens === undefined ? 128 : options.maxNewTokens;
      if (!Number.isSafeInteger(maxNewTokens) || maxNewTokens < 0) {
        throw modelError("GENERATION_INVALID_OPTIONS", "maxNewTokens must be a non-negative integer.");
      }
      const stopOnEos = options.stopOnEos !== false;
      const tokenizer = options.tokenizer || this.tokenizer;
      let promptIds;
      if (typeof input === "string") {
        if (!tokenizer) throw modelError("GENERATION_TOKENIZER_REQUIRED", "A tokenizer is required for text prompts.");
        promptIds = tokenizer.encode(input, { addBos: options.addBos === true });
      } else if (Array.isArray(input) && (input.length === 0 || !Array.isArray(input[0]))) {
        promptIds = input.slice();
      } else {
        throw modelError("GENERATION_INVALID_INPUT", "Prompt must be text or a single Token ID array.");
      }
      if (promptIds.length === 0) throw modelError("GENERATION_INVALID_INPUT", "Prompt must contain at least one Token ID.");
      validateIds([promptIds], this.config);
      const stopTokenId = options.stopTokenId === undefined
        ? (tokenizer && tokenizer.getSpecialTokens().EOS)
        : options.stopTokenId;
      if (stopTokenId !== undefined &&
        (!Number.isSafeInteger(stopTokenId) || stopTokenId < 0 || stopTokenId >= this.config.vocabSize)) {
        throw modelError("GENERATION_INVALID_OPTIONS", "stopTokenId is outside the vocabulary.");
      }

      const currentIds = promptIds.slice();
      const generated = [];
      let finishReason = "length";
      for (let step = 0; step < maxNewTokens; step += 1) {
        if (currentIds.length > this.config.maxSequenceLength) break;
        const logits = KNA.noGrad(() => this.forward([currentIds]));
        try {
          const values = logits.toArray();
          const vocabularyStart = (currentIds.length - 1) * this.config.vocabSize;
          let selected = 0;
          let maximum = values[vocabularyStart];
          for (let token = 1; token < this.config.vocabSize; token += 1) {
            const candidate = values[vocabularyStart + token];
            if (candidate > maximum) {
              maximum = candidate;
              selected = token;
            }
          }
          generated.push(selected);
          currentIds.push(selected);
          if (stopOnEos && selected === stopTokenId) {
            finishReason = "eos";
            break;
          }
        } finally {
          logits.dispose();
        }
      }

      const result = {
        tokenIds: generated,
        allTokenIds: currentIds,
        generatedTokens: generated.length,
        finishReason
      };
      if (tokenizer) result.text = tokenizer.decode(generated, { skipSpecialTokens: true, replaceInvalid: true });
      return result;
    }

    stateDict() {
      this.assertReady();
      return {
        format: "KNA-Rikka-Weights",
        version: 1,
        config: { ...this.config },
        parameters: this.parameters().map((parameter) => ({
          name: parameter.name,
          shape: parameter.data.shape.slice(),
          dtype: parameter.data.dtype,
          decay: parameter.decay,
          values: parameter.data.toArray()
        }))
      };
    }

    loadStateDict(state) {
      this.assertReady();
      if (!state || state.format !== "KNA-Rikka-Weights" || state.version !== 1 ||
        !Array.isArray(state.parameters) || !state.config ||
        Object.keys(this.config).some((key) => state.config[key] !== this.config[key])) {
        throw modelError("MODEL_INVALID_STATE", "Weight state metadata does not match ModelConfig.");
      }

      const parameters = this.parameters();
      if (parameters.length !== state.parameters.length) {
        throw modelError("MODEL_INVALID_STATE", "Weight Parameter count does not match.");
      }
      const byName = new Map(parameters.map((parameter) => [parameter.name, parameter]));
      const updates = [];
      const seen = new Set();
      for (const entry of state.parameters) {
        if (!entry || typeof entry.name !== "string" || seen.has(entry.name)) {
          throw modelError("MODEL_INVALID_STATE", "Weight state contains an invalid Parameter name.");
        }
        seen.add(entry.name);
        const parameter = byName.get(entry.name);
        if (!parameter || entry.dtype !== parameter.data.dtype || entry.decay !== parameter.decay ||
          !Array.isArray(entry.shape) || entry.shape.length !== parameter.data.shape.length ||
          entry.shape.some((dimension, axis) => dimension !== parameter.data.shape[axis]) ||
          !Array.isArray(entry.values) || entry.values.length !== parameter.data.size) {
          throw modelError("MODEL_INVALID_STATE", "Weight state metadata does not match a Parameter.");
        }
        const preparedData = KNA._internal.prepareTensorData(parameter.data, entry.values);
        updates.push([parameter.data, preparedData]);
      }
      KNA._internal.commitTensorDataBatch(updates);
    }

    saveJSON() {
      return JSON.stringify(this.stateDict());
    }

    loadJSON(json) {
      if (typeof json !== "string") throw modelError("MODEL_INVALID_STATE", "Serialized weights must be JSON text.");
      let state;
      try {
        state = JSON.parse(json);
      } catch (_) {
        throw modelError("MODEL_INVALID_STATE", "Serialized weights are not valid JSON.");
      }
      return this.loadStateDict(state);
    }

    _disposeLayers() {
      this.embedding.dispose();
      for (const block of this.blocks) block.dispose();
      this.finalNorm.dispose();
      if (this.lmHead) this.lmHead.dispose();
    }

    dispose() {
      if (this.status === "DISPOSED") return;
      this._disposeLayers();
      this.status = "DISPOSED";
    }
  }

  KNA.CausalSelfAttention = CausalSelfAttention;
  KNA.RikkaModel = RikkaModel;
  KNA.createRikkaModel = (config, options) => new RikkaModel(config, options);
})(globalThis);
