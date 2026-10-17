(function (global) {
  "use strict";

  const KNA = global.KNA;
  const Tensor = KNA.Tensor;

  function lossError(code, message) {
    const error = new Error(message);
    error.code = code;
    return error;
  }

  function validateMatrix(matrix, batchSize, sequenceLength, label) {
    if (!Array.isArray(matrix) || matrix.length !== batchSize) {
      throw lossError("LOSS_INVALID_TARGET", label + " must have one row per batch item.");
    }
    for (const row of matrix) {
      if (!Array.isArray(row) || row.length !== sequenceLength) {
        throw lossError("LOSS_INVALID_TARGET", label + " must match the sequence Shape.");
      }
    }
  }

  function crossEntropy(logits, targetIds, options = {}) {
    if (!(logits instanceof Tensor) || logits.disposed || logits.dtype !== "float32") {
      throw lossError("LOSS_INVALID_LOGITS", "Logits must be a live FP32 Tensor.");
    }
    if (!options || typeof options !== "object" || Array.isArray(options)) {
      throw lossError("LOSS_INVALID_OPTIONS", "Cross Entropy options must be an object.");
    }

    const shape = logits.shape;
    if (shape.length !== 3 || shape[2] <= 0) {
      throw lossError("LOSS_INVALID_LOGITS", "Logits Shape must be [batch, sequence, vocabSize].");
    }

    const [batchSize, sequenceLength, vocabSize] = shape;
    validateMatrix(targetIds, batchSize, sequenceLength, "targetIds");

    const reduction = options.reduction === undefined ? "mean" : options.reduction;
    if (reduction !== "none" && reduction !== "sum" && reduction !== "mean") {
      throw lossError("LOSS_INVALID_OPTIONS", "Unsupported Cross Entropy reduction.");
    }

    const ignoreIndex = options.ignoreIndex === undefined ? -100 : options.ignoreIndex;
    if (!Number.isSafeInteger(ignoreIndex) || (ignoreIndex >= 0 && ignoreIndex < vocabSize)) {
      throw lossError("LOSS_INVALID_OPTIONS", "ignoreIndex must be an integer outside the vocabulary range.");
    }

    if (options.lossMask !== undefined) {
      validateMatrix(options.lossMask, batchSize, sequenceLength, "lossMask");
    }

    const positionCount = batchSize * sequenceLength;
    const logitsValues = logits.toArray();
    const targets = new Int32Array(positionCount);
    const valid = new Uint8Array(positionCount);
    const probabilities = new Float32Array(logits.size);
    const perPositionLoss = new Float32Array(positionCount);
    let validCount = 0;
    let lossSum = 0;

    for (let batch = 0; batch < batchSize; batch += 1) {
      for (let sequence = 0; sequence < sequenceLength; sequence += 1) {
        const position = batch * sequenceLength + sequence;
        const target = targetIds[batch][sequence];
        if (!Number.isSafeInteger(target)) {
          throw lossError("LOSS_INVALID_TARGET", "Target IDs must be safe integers.");
        }
        if (target !== ignoreIndex && (target < 0 || target >= vocabSize)) {
          throw lossError("LOSS_INVALID_TARGET", "A non-ignored Target ID is outside the vocabulary.");
        }
        targets[position] = target;

        let included = target !== ignoreIndex;
        if (options.lossMask !== undefined) {
          const maskValue = options.lossMask[batch][sequence];
          if (maskValue !== true && maskValue !== false && maskValue !== 0 && maskValue !== 1) {
            throw lossError("LOSS_INVALID_MASK", "Loss Mask values must be boolean or 0/1.");
          }
          included = included && (maskValue === true || maskValue === 1);
        }
        if (!included) continue;

        valid[position] = 1;
        validCount += 1;
        const base = position * vocabSize;
        let maximum = -Infinity;
        for (let token = 0; token < vocabSize; token += 1) {
          maximum = Math.max(maximum, logitsValues[base + token]);
        }

        let denominator = 0;
        for (let token = 0; token < vocabSize; token += 1) {
          denominator += Math.exp(logitsValues[base + token] - maximum);
        }
        const logDenominator = Math.log(denominator);
        perPositionLoss[position] = Math.fround(
          maximum + logDenominator - logitsValues[base + target]
        );
        if (!Number.isFinite(perPositionLoss[position])) {
          throw lossError("LOSS_NUMERICAL_ERROR", "Cross Entropy produced a non-finite loss.");
        }
        lossSum += perPositionLoss[position];

        for (let token = 0; token < vocabSize; token += 1) {
          probabilities[base + token] = Math.fround(
            Math.exp(logitsValues[base + token] - maximum) / denominator
          );
        }
      }
    }

    if (reduction === "mean" && validCount === 0) {
      throw lossError("LOSS_EMPTY_TARGET", "Mean Cross Entropy requires at least one valid target.");
    }

    let outputValues;
    let outputShape;
    if (reduction === "none") {
      outputValues = perPositionLoss;
      outputShape = [batchSize, sequenceLength];
    } else {
      const reducedLoss = reduction === "mean" ? lossSum / validCount : lossSum;
      if (!Number.isFinite(reducedLoss)) {
        throw lossError("LOSS_NUMERICAL_ERROR", "Cross Entropy reduction is not finite.");
      }
      outputValues = Float32Array.of(Math.fround(reducedLoss));
      outputShape = [];
    }

    return KNA._internal.createCustomOperation(
      outputShape,
      outputValues,
      [logits],
      "crossEntropy",
      (outputGradient) => {
        const gradient = new Float32Array(logits.size);
        for (let position = 0; position < positionCount; position += 1) {
          if (valid[position] === 0) continue;
          let scale;
          if (reduction === "none") {
            scale = outputGradient[position];
          } else if (reduction === "mean") {
            scale = outputGradient[0] / validCount;
          } else {
            scale = outputGradient[0];
          }

          const base = position * vocabSize;
          for (let token = 0; token < vocabSize; token += 1) {
            const targetGradient = probabilities[base + token] - (token === targets[position] ? 1 : 0);
            gradient[base + token] = Math.fround(targetGradient * scale);
          }
        }
        return [gradient];
      }
    );
  }

  KNA.losses = Object.freeze({
    crossEntropy
  });
})(globalThis);
