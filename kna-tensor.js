(function (global) {
  "use strict";

  const MAX_RANK = 16;
  const MAX_ELEMENTS = 16777216;
  const INTERNAL = Symbol("KNA.Tensor");
  const states = new WeakMap();
  let nextTensorId = 1;

  function tensorError(code, message) {
    const error = new Error(message);
    error.code = code;
    return error;
  }

  function validateShape(shape) {
    if (!Array.isArray(shape)) {
      throw tensorError("TENSOR_INVALID_SHAPE", "Shape must be an array.");
    }

    if (shape.length > MAX_RANK) {
      throw tensorError("TENSOR_INVALID_SHAPE", "Shape rank exceeds the limit.");
    }

    let size = 1;
    for (const dimension of shape) {
      if (!Number.isSafeInteger(dimension) || dimension < 0) {
        throw tensorError("TENSOR_INVALID_SHAPE", "Shape dimensions must be non-negative safe integers.");
      }

      if (dimension === 0) {
        size = 0;
      } else if (size !== 0) {
        if (size > MAX_ELEMENTS / dimension) {
          throw tensorError("TENSOR_INVALID_SHAPE", "Tensor element count exceeds the limit.");
        }
        size *= dimension;
      }
    }

    return { shape: shape.slice(), size };
  }

  function allocate(size) {
    try {
      return new Float32Array(size);
    } catch (error) {
      if (error instanceof RangeError) {
        throw tensorError("TENSOR_OUT_OF_MEMORY", "Unable to allocate Tensor data.");
      }
      throw error;
    }
  }

  function toFloat32(value) {
    if (typeof value !== "number" || !Number.isFinite(value)) {
      throw tensorError("TENSOR_NUMERICAL_ERROR", "Tensor values must be finite numbers.");
    }

    const rounded = Math.fround(value);
    if (!Number.isFinite(rounded)) {
      throw tensorError("TENSOR_NUMERICAL_ERROR", "Value is outside the float32 range.");
    }
    return rounded;
  }

  function flattenInput(input) {
    if (ArrayBuffer.isView(input) && !(input instanceof DataView)) {
      return {
        shape: [input.length],
        values: Array.from(input, toFloat32)
      };
    }

    if (!Array.isArray(input)) {
      return { shape: [], values: [toFloat32(input)] };
    }

    function visit(value, depth) {
      if (Array.isArray(value)) {
        if (depth >= MAX_RANK) {
          throw tensorError("TENSOR_INVALID_SHAPE", "Shape rank exceeds the limit.");
        }

        if (value.length === 0) {
          return { shape: [0], values: [] };
        }

        const first = visit(value[0], depth + 1);
        const values = first.values.slice();

        for (let index = 1; index < value.length; index += 1) {
          const child = visit(value[index], depth + 1);
          if (
            child.shape.length !== first.shape.length ||
            child.shape.some((dimension, axis) => dimension !== first.shape[axis])
          ) {
            throw tensorError("TENSOR_INVALID_SHAPE", "Nested input arrays must be rectangular.");
          }
          for (const item of child.values) values.push(item);
        }

        return { shape: [value.length, ...first.shape], values };
      }

      return { shape: [], values: [toFloat32(value)] };
    }

    return visit(input, 0);
  }

  function toFloat32Array(values) {
    try {
      return Float32Array.from(values, toFloat32);
    } catch (error) {
      if (error instanceof RangeError) {
        throw tensorError("TENSOR_OUT_OF_MEMORY", "Unable to allocate Tensor data.");
      }
      throw error;
    }
  }

  function createTensor(shape, data) {
    const validated = validateShape(shape);
    if (!(data instanceof Float32Array) || data.length !== validated.size) {
      throw tensorError("TENSOR_SHAPE_MISMATCH", "Float32 data length does not match Shape.");
    }
    return new Tensor(INTERNAL, validated.shape, data);
  }

  function assertUsable(tensor) {
    const state = states.get(tensor);
    if (!state || state.disposed) {
      throw tensorError("TENSOR_DISPOSED", "Tensor has been disposed.");
    }
    return state;
  }

  function stridesFor(shape) {
    const strides = new Array(shape.length);
    let stride = 1;

    for (let axis = shape.length - 1; axis >= 0; axis -= 1) {
      strides[axis] = stride;
      stride *= shape[axis];
    }

    return strides;
  }

  function coordinatesFor(index, shape, strides) {
    const coordinates = new Array(shape.length);
    for (let axis = 0; axis < shape.length; axis += 1) {
      coordinates[axis] = Math.floor(index / strides[axis]) % shape[axis];
    }
    return coordinates;
  }

  function normalizeAxes(axes, rank) {
    if (axes !== undefined && !Array.isArray(axes)) {
      throw tensorError("TENSOR_INVALID_ARGUMENT", "Axes must be an array.");
    }

    const result = axes === undefined
      ? Array.from({ length: rank }, (_, index) => rank - index - 1)
      : axes.slice();

    if (
      result.length !== rank ||
      result.some((axis) => !Number.isSafeInteger(axis) || axis < 0 || axis >= rank) ||
      new Set(result).size !== rank
    ) {
      throw tensorError("TENSOR_INVALID_ARGUMENT", "Axes must be a permutation of all dimensions.");
    }

    return result;
  }

  class Tensor {
    constructor(token, shape, data) {
      if (token !== INTERNAL) {
        throw tensorError("TENSOR_INVALID_ARGUMENT", "Use a Tensor factory method.");
      }

      states.set(this, {
        id: "tensor-" + nextTensorId++,
        shape: Object.freeze(shape.slice()),
        size: data.length,
        data,
        requiresGrad: false,
        grad: null,
        version: 0,
        disposed: false
      });
    }

    static fromArray(data, shape) {
      const flattened = flattenInput(data);
      const target = shape === undefined
        ? validateShape(flattened.shape)
        : validateShape(shape);
      if (flattened.values.length !== target.size) {
        throw tensorError("TENSOR_SHAPE_MISMATCH", "Data length does not match Shape.");
      }
      return createTensor(target.shape, toFloat32Array(flattened.values));
    }

    static zeros(shape) {
      const validated = validateShape(shape);
      return createTensor(validated.shape, allocate(validated.size));
    }

    static ones(shape) {
      return Tensor.full(shape, 1);
    }

    static full(shape, value) {
      const validated = validateShape(shape);
      const data = allocate(validated.size);
      data.fill(toFloat32(value));
      return createTensor(validated.shape, data);
    }

    static scalar(value) {
      return Tensor.full([], value);
    }

    get id() {
      return assertUsable(this).id;
    }

    get shape() {
      return assertUsable(this).shape;
    }

    get rank() {
      return assertUsable(this).shape.length;
    }

    get size() {
      return assertUsable(this).size;
    }

    get dtype() {
      assertUsable(this);
      return "float32";
    }

    get requiresGrad() {
      return assertUsable(this).requiresGrad;
    }

    get grad() {
      return assertUsable(this).grad;
    }

    get version() {
      return assertUsable(this).version;
    }

    get disposed() {
      const state = states.get(this);
      return !state || state.disposed;
    }

    toArray() {
      return Array.from(assertUsable(this).data);
    }

    item(index = 0) {
      const state = assertUsable(this);
      if (!Number.isSafeInteger(index) || index < 0 || index >= state.size) {
        throw tensorError("TENSOR_INVALID_ARGUMENT", "Tensor index is out of range.");
      }
      return state.data[index];
    }

    clone() {
      const state = assertUsable(this);
      return createTensor(state.shape, Float32Array.from(state.data));
    }

    reshape(shape) {
      const state = assertUsable(this);
      const target = validateShape(shape);
      if (target.size !== state.size) {
        throw tensorError("TENSOR_SHAPE_MISMATCH", "Reshape must preserve element count.");
      }
      return createTensor(target.shape, Float32Array.from(state.data));
    }

    transpose(axes) {
      const state = assertUsable(this);
      const rank = state.shape.length;
      const order = normalizeAxes(axes, rank);
      const outputShape = order.map((axis) => state.shape[axis]);
      const output = allocate(state.size);

      if (state.size > 0) {
        const inputStrides = stridesFor(state.shape);
        const outputStrides = stridesFor(outputShape);
        for (let outputIndex = 0; outputIndex < state.size; outputIndex += 1) {
          const outputCoordinates = coordinatesFor(outputIndex, outputShape, outputStrides);
          let inputIndex = 0;
          for (let axis = 0; axis < rank; axis += 1) {
            inputIndex += outputCoordinates[axis] * inputStrides[order[axis]];
          }
          output[outputIndex] = state.data[inputIndex];
        }
      }

      return createTensor(outputShape, output);
    }

    slice(spec) {
      const state = assertUsable(this);
      if (!Array.isArray(spec) || spec.length > state.shape.length) {
        throw tensorError("TENSOR_INVALID_ARGUMENT", "Slice spec must contain at most one range per axis.");
      }

      const ranges = state.shape.map((dimension, axis) => {
        const range = spec[axis];
        if (range === undefined || range === null) return [0, dimension];

        const start = Array.isArray(range) ? range[0] : range.start;
        const end = Array.isArray(range) ? range[1] : range.end;
        if (
          !Number.isSafeInteger(start) ||
          !Number.isSafeInteger(end) ||
          start < 0 ||
          end < start ||
          end > dimension
        ) {
          throw tensorError("TENSOR_INVALID_ARGUMENT", "Slice ranges must be valid half-open intervals.");
        }
        return [start, end];
      });

      const outputShape = ranges.map(([start, end]) => end - start);
      const outputSize = validateShape(outputShape).size;
      const output = allocate(outputSize);

      if (outputSize > 0) {
        const inputStrides = stridesFor(state.shape);
        const outputStrides = stridesFor(outputShape);
        for (let outputIndex = 0; outputIndex < outputSize; outputIndex += 1) {
          const outputCoordinates = coordinatesFor(outputIndex, outputShape, outputStrides);
          let inputIndex = 0;
          for (let axis = 0; axis < state.shape.length; axis += 1) {
            inputIndex += (outputCoordinates[axis] + ranges[axis][0]) * inputStrides[axis];
          }
          output[outputIndex] = state.data[inputIndex];
        }
      }

      return createTensor(outputShape, output);
    }

    dispose() {
      const state = states.get(this);
      if (state && !state.disposed) {
        state.disposed = true;
        state.data = null;
      }
    }
  }

  function requireTensor(value) {
    if (!(value instanceof Tensor)) {
      throw tensorError("TENSOR_INVALID_ARGUMENT", "Operation inputs must be Tensors.");
    }
    return assertUsable(value);
  }

  function broadcastShape(leftShape, rightShape) {
    const rank = Math.max(leftShape.length, rightShape.length);
    const shape = new Array(rank);

    for (let offset = 1; offset <= rank; offset += 1) {
      const left = leftShape[leftShape.length - offset] ?? 1;
      const right = rightShape[rightShape.length - offset] ?? 1;
      if (left === right) {
        shape[rank - offset] = left;
      } else if (left === 1) {
        shape[rank - offset] = right;
      } else if (right === 1) {
        shape[rank - offset] = left;
      } else {
        throw tensorError("TENSOR_SHAPE_MISMATCH", "Input shapes cannot be broadcast.");
      }
    }

    return validateShape(shape);
  }

  function broadcastStrides(shape, outputShape) {
    const inputStrides = stridesFor(shape);
    const offset = outputShape.length - shape.length;
    return outputShape.map((_, axis) => {
      const inputAxis = axis - offset;
      if (inputAxis < 0 || shape[inputAxis] === 1) return 0;
      return inputStrides[inputAxis];
    });
  }

  function binaryOperation(left, right, operation, operationName) {
    const leftState = requireTensor(left);
    const rightState = requireTensor(right);
    const output = broadcastShape(leftState.shape, rightState.shape);
    const data = allocate(output.size);

    if (output.size > 0) {
      const outputStrides = stridesFor(output.shape);
      const leftStrides = broadcastStrides(leftState.shape, output.shape);
      const rightStrides = broadcastStrides(rightState.shape, output.shape);

      for (let index = 0; index < output.size; index += 1) {
        const coordinates = coordinatesFor(index, output.shape, outputStrides);
        let leftIndex = 0;
        let rightIndex = 0;
        for (let axis = 0; axis < output.shape.length; axis += 1) {
          leftIndex += coordinates[axis] * leftStrides[axis];
          rightIndex += coordinates[axis] * rightStrides[axis];
        }

        const leftValue = leftState.data[leftIndex];
        const rightValue = rightState.data[rightIndex];
        if (operationName === "div" && rightValue === 0) {
          throw tensorError("TENSOR_NUMERICAL_ERROR", "Division by zero is not allowed.");
        }
        data[index] = toFloat32(operation(leftValue, rightValue));
      }
    }

    return createTensor(output.shape, data);
  }

  function normalizeReductionAxes(axis, rank) {
    if (axis === undefined) {
      return Array.from({ length: rank }, (_, index) => index);
    }

    const axes = Array.isArray(axis) ? axis.slice() : [axis];
    const normalized = axes.map((value) => {
      if (!Number.isSafeInteger(value)) {
        throw tensorError("TENSOR_INVALID_ARGUMENT", "Reduction axes must be integers.");
      }
      const result = value < 0 ? value + rank : value;
      if (result < 0 || result >= rank) {
        throw tensorError("TENSOR_INVALID_ARGUMENT", "Reduction axis is out of range.");
      }
      return result;
    });

    if (new Set(normalized).size !== normalized.length) {
      throw tensorError("TENSOR_INVALID_ARGUMENT", "Reduction axes must not contain duplicates.");
    }
    return normalized;
  }

  function sum(input, options) {
    const inputState = requireTensor(input);
    let axis;
    let keepDims = false;

    if (options === undefined) {
      axis = undefined;
    } else if (typeof options === "number" || Array.isArray(options)) {
      axis = options;
    } else if (options && typeof options === "object") {
      axis = options.axis;
      keepDims = options.keepDims === true;
    } else {
      throw tensorError("TENSOR_INVALID_ARGUMENT", "Invalid sum options.");
    }

    const axes = normalizeReductionAxes(axis, inputState.shape.length);
    if (axes.length === 0) return createTensor(inputState.shape, Float32Array.from(inputState.data));

    const reduced = new Set(axes);
    const outputShape = inputState.shape.filter((_, index) => !reduced.has(index));
    if (keepDims) {
      for (const reducedAxis of axes) outputShape.splice(reducedAxis, 0, 1);
    }

    const output = allocate(validateShape(outputShape).size);
    output.fill(0);

    if (inputState.size > 0) {
      const inputStrides = stridesFor(inputState.shape);
      const outputStrides = stridesFor(outputShape);

      for (let inputIndex = 0; inputIndex < inputState.size; inputIndex += 1) {
        const coordinates = coordinatesFor(inputIndex, inputState.shape, inputStrides);
        let outputIndex = 0;
        let outputAxis = 0;
        for (let inputAxis = 0; inputAxis < inputState.shape.length; inputAxis += 1) {
          if (reduced.has(inputAxis)) {
            if (keepDims) outputAxis += 1;
          } else {
            outputIndex += coordinates[inputAxis] * outputStrides[outputAxis];
            outputAxis += 1;
          }
        }
        output[outputIndex] = toFloat32(output[outputIndex] + inputState.data[inputIndex]);
      }
    }

    return createTensor(outputShape, output);
  }

  function reductionAxisFromOptions(options) {
    if (options === undefined) return undefined;
    if (typeof options === "number" || Array.isArray(options)) return options;
    if (options && typeof options === "object") return options.axis;
    throw tensorError("TENSOR_INVALID_ARGUMENT", "Invalid reduction options.");
  }

  function reductionShape(shape, axes, keepDims) {
    const reduced = new Set(axes);
    const outputShape = shape.filter((_, index) => !reduced.has(index));
    if (keepDims) {
      for (const axis of axes) outputShape.splice(axis, 0, 1);
    }
    return outputShape;
  }

  function mean(input, options) {
    const inputState = requireTensor(input);
    const axes = normalizeReductionAxes(
      reductionAxisFromOptions(options),
      inputState.shape.length
    );
    let count = 1;
    for (const axis of axes) {
      const dimension = inputState.shape[axis];
      if (dimension === 0) {
        throw tensorError("TENSOR_NUMERICAL_ERROR", "Mean of an empty reduction is not defined.");
      }
      count *= dimension;
    }

    const total = sum(input, options);
    const totalState = assertUsable(total);
    const data = allocate(totalState.size);
    for (let index = 0; index < totalState.size; index += 1) {
      data[index] = toFloat32(totalState.data[index] / count);
    }
    const result = createTensor(totalState.shape, data);
    total.dispose();
    return result;
  }

  function max(input, options) {
    const inputState = requireTensor(input);
    const axis = reductionAxisFromOptions(options);
    const axes = normalizeReductionAxes(axis, inputState.shape.length);
    if (axes.length === 0) {
      return createTensor(inputState.shape, Float32Array.from(inputState.data));
    }

    const keepDims = Boolean(options && typeof options === "object" && options.keepDims);
    const outputShape = reductionShape(inputState.shape, axes, keepDims);
    const outputSize = validateShape(outputShape).size;
    if (outputSize > 0 && axes.some((axisIndex) => inputState.shape[axisIndex] === 0)) {
      throw tensorError("TENSOR_NUMERICAL_ERROR", "Max of an empty reduction is not defined.");
    }

    const output = allocate(outputSize);
    output.fill(-Infinity);
    if (inputState.size > 0) {
      const reduced = new Set(axes);
      const inputStrides = stridesFor(inputState.shape);
      const outputStrides = stridesFor(outputShape);

      for (let inputIndex = 0; inputIndex < inputState.size; inputIndex += 1) {
        const coordinates = coordinatesFor(inputIndex, inputState.shape, inputStrides);
        let outputIndex = 0;
        let outputAxis = 0;
        for (let inputAxis = 0; inputAxis < inputState.shape.length; inputAxis += 1) {
          if (reduced.has(inputAxis)) {
            if (keepDims) outputAxis += 1;
          } else {
            outputIndex += coordinates[inputAxis] * outputStrides[outputAxis];
            outputAxis += 1;
          }
        }
        output[outputIndex] = Math.max(output[outputIndex], inputState.data[inputIndex]);
      }
    }

    return createTensor(outputShape, output);
  }

  function normalizeAxis(axis, rank) {
    if (!Number.isSafeInteger(axis)) {
      throw tensorError("TENSOR_INVALID_ARGUMENT", "Axis must be an integer.");
    }
    const normalized = axis < 0 ? axis + rank : axis;
    if (normalized < 0 || normalized >= rank) {
      throw tensorError("TENSOR_INVALID_ARGUMENT", "Axis is out of range.");
    }
    return normalized;
  }

  function unaryOperation(input, operation) {
    const inputState = requireTensor(input);
    const output = allocate(inputState.size);
    for (let index = 0; index < inputState.size; index += 1) {
      output[index] = toFloat32(operation(inputState.data[index]));
    }
    return createTensor(inputState.shape, output);
  }

  function softmax(input, axis, logarithmic) {
    const inputState = requireTensor(input);
    const normalizedAxis = normalizeAxis(axis, inputState.shape.length);
    const axisSize = inputState.shape[normalizedAxis];
    if (axisSize === 0) {
      throw tensorError("TENSOR_NUMERICAL_ERROR", "Softmax over an empty axis is not defined.");
    }

    const output = allocate(inputState.size);
    if (inputState.size === 0) return createTensor(inputState.shape, output);

    let outerSize = 1;
    let innerSize = 1;
    for (let axisIndex = 0; axisIndex < normalizedAxis; axisIndex += 1) {
      outerSize *= inputState.shape[axisIndex];
    }
    for (let axisIndex = normalizedAxis + 1; axisIndex < inputState.shape.length; axisIndex += 1) {
      innerSize *= inputState.shape[axisIndex];
    }

    for (let outer = 0; outer < outerSize; outer += 1) {
      for (let inner = 0; inner < innerSize; inner += 1) {
        let maximum = -Infinity;
        for (let index = 0; index < axisSize; index += 1) {
          const offset = (outer * axisSize + index) * innerSize + inner;
          maximum = Math.max(maximum, inputState.data[offset]);
        }

        let denominator = 0;
        for (let index = 0; index < axisSize; index += 1) {
          const offset = (outer * axisSize + index) * innerSize + inner;
          denominator += Math.exp(inputState.data[offset] - maximum);
        }
        const logDenominator = Math.log(denominator);

        for (let index = 0; index < axisSize; index += 1) {
          const offset = (outer * axisSize + index) * innerSize + inner;
          const value = inputState.data[offset] - maximum;
          output[offset] = toFloat32(
            logarithmic ? value - logDenominator : Math.exp(value) / denominator
          );
        }
      }
    }

    return createTensor(inputState.shape, output);
  }

  function matmul(left, right) {
    const leftState = requireTensor(left);
    const rightState = requireTensor(right);
    if (leftState.shape.length < 2 || rightState.shape.length < 2) {
      throw tensorError("TENSOR_SHAPE_MISMATCH", "MatMul inputs must have rank 2 or greater.");
    }

    const leftRank = leftState.shape.length;
    const rightRank = rightState.shape.length;
    const rows = leftState.shape[leftRank - 2];
    const inner = leftState.shape[leftRank - 1];
    const rightInner = rightState.shape[rightRank - 2];
    const columns = rightState.shape[rightRank - 1];
    if (inner !== rightInner) {
      throw tensorError("TENSOR_SHAPE_MISMATCH", "MatMul inner dimensions must match.");
    }

    const leftBatch = leftState.shape.slice(0, -2);
    const rightBatch = rightState.shape.slice(0, -2);
    const batch = broadcastShape(leftBatch, rightBatch);
    const outputShape = [...batch.shape, rows, columns];
    const outputSize = validateShape(outputShape).size;
    const output = allocate(outputSize);

    if (outputSize > 0) {
      const leftBatchStrides = broadcastStrides(leftBatch, batch.shape);
      const rightBatchStrides = broadcastStrides(rightBatch, batch.shape);
      const batchStrides = stridesFor(batch.shape);
      const leftMatrixSize = rows * inner;
      const rightMatrixSize = inner * columns;
      let batchCount = batch.size;
      if (batch.shape.length === 0) batchCount = 1;

      for (let batchIndex = 0; batchIndex < batchCount; batchIndex += 1) {
        const batchCoordinates = batch.shape.length === 0
          ? []
          : coordinatesFor(batchIndex, batch.shape, batchStrides);
        let leftOffset = 0;
        let rightOffset = 0;
        for (let axis = 0; axis < batch.shape.length; axis += 1) {
          leftOffset += batchCoordinates[axis] * leftBatchStrides[axis];
          rightOffset += batchCoordinates[axis] * rightBatchStrides[axis];
        }
        leftOffset *= leftMatrixSize;
        rightOffset *= rightMatrixSize;
        const outputOffset = batchIndex * rows * columns;

        for (let row = 0; row < rows; row += 1) {
          for (let column = 0; column < columns; column += 1) {
            let value = 0;
            for (let index = 0; index < inner; index += 1) {
              value += leftState.data[leftOffset + row * inner + index] *
                rightState.data[rightOffset + index * columns + column];
            }
            output[outputOffset + row * columns + column] = toFloat32(value);
          }
        }
      }
    }

    return createTensor(outputShape, output);
  }

  function concat(tensors, axis) {
    if (!Array.isArray(tensors) || tensors.length === 0) {
      throw tensorError("TENSOR_INVALID_ARGUMENT", "Concat requires at least one Tensor.");
    }

    const inputStates = tensors.map(requireTensor);
    const rank = inputStates[0].shape.length;
    const normalizedAxis = normalizeAxis(axis, rank);
    const outputShape = inputStates[0].shape.slice();
    outputShape[normalizedAxis] = 0;

    for (const state of inputStates) {
      if (state.shape.length !== rank) {
        throw tensorError("TENSOR_SHAPE_MISMATCH", "Concat inputs must have the same rank.");
      }
      for (let dimension = 0; dimension < rank; dimension += 1) {
        if (dimension !== normalizedAxis && state.shape[dimension] !== outputShape[dimension]) {
          throw tensorError("TENSOR_SHAPE_MISMATCH", "Concat dimensions must match outside the selected axis.");
        }
      }
      outputShape[normalizedAxis] += state.shape[normalizedAxis];
    }

    const output = allocate(validateShape(outputShape).size);
    let outerSize = 1;
    let innerSize = 1;
    for (let dimension = 0; dimension < normalizedAxis; dimension += 1) {
      outerSize *= outputShape[dimension];
    }
    for (let dimension = normalizedAxis + 1; dimension < rank; dimension += 1) {
      innerSize *= outputShape[dimension];
    }

    const outputAxisSize = outputShape[normalizedAxis];
    for (let outer = 0; outer < outerSize; outer += 1) {
      let axisOffset = 0;
      for (const state of inputStates) {
        const segmentSize = state.shape[normalizedAxis] * innerSize;
        const sourceOffset = outer * segmentSize;
        const targetOffset = (outer * outputAxisSize + axisOffset) * innerSize;
        output.set(state.data.subarray(sourceOffset, sourceOffset + segmentSize), targetOffset);
        axisOffset += state.shape[normalizedAxis];
      }
    }

    return createTensor(outputShape, output);
  }

  function validateMask(state) {
    for (const value of state.data) {
      if (value !== 0 && value !== 1) {
        throw tensorError("TENSOR_INVALID_ARGUMENT", "FP32 masks must contain only 0 or 1.");
      }
    }
  }

  function where(condition, left, right) {
    const conditionState = requireTensor(condition);
    const leftState = requireTensor(left);
    const rightState = requireTensor(right);
    validateMask(conditionState);

    const pairShape = broadcastShape(leftState.shape, rightState.shape);
    const outputShape = broadcastShape(conditionState.shape, pairShape.shape);
    const output = allocate(outputShape.size);

    if (outputShape.size > 0) {
      const outputStrides = stridesFor(outputShape.shape);
      const conditionStrides = broadcastStrides(conditionState.shape, outputShape.shape);
      const leftStrides = broadcastStrides(leftState.shape, outputShape.shape);
      const rightStrides = broadcastStrides(rightState.shape, outputShape.shape);

      for (let index = 0; index < outputShape.size; index += 1) {
        const coordinates = coordinatesFor(index, outputShape.shape, outputStrides);
        let conditionIndex = 0;
        let leftIndex = 0;
        let rightIndex = 0;
        for (let axis = 0; axis < outputShape.shape.length; axis += 1) {
          conditionIndex += coordinates[axis] * conditionStrides[axis];
          leftIndex += coordinates[axis] * leftStrides[axis];
          rightIndex += coordinates[axis] * rightStrides[axis];
        }
        output[index] = conditionState.data[conditionIndex] === 1
          ? leftState.data[leftIndex]
          : rightState.data[rightIndex];
      }
    }

    return createTensor(outputShape.shape, output);
  }

  function gather(input, indices, axis) {
    const inputState = requireTensor(input);
    if (!Array.isArray(indices)) {
      throw tensorError("TENSOR_INVALID_ARGUMENT", "Gather indices must be a number array.");
    }
    const normalizedAxis = normalizeAxis(axis, inputState.shape.length);
    const axisSize = inputState.shape[normalizedAxis];
    for (const index of indices) {
      if (!Number.isSafeInteger(index) || index < 0 || index >= axisSize) {
        throw tensorError("TENSOR_INVALID_ARGUMENT", "Gather index is outside the selected axis.");
      }
    }

    const outputShape = inputState.shape.slice();
    outputShape[normalizedAxis] = indices.length;
    const output = allocate(validateShape(outputShape).size);
    if (output.length > 0) {
      const inputStrides = stridesFor(inputState.shape);
      const outputStrides = stridesFor(outputShape);
      for (let outputIndex = 0; outputIndex < output.length; outputIndex += 1) {
        const coordinates = coordinatesFor(outputIndex, outputShape, outputStrides);
        coordinates[normalizedAxis] = indices[coordinates[normalizedAxis]];
        let inputIndex = 0;
        for (let dimension = 0; dimension < inputState.shape.length; dimension += 1) {
          inputIndex += coordinates[dimension] * inputStrides[dimension];
        }
        output[outputIndex] = inputState.data[inputIndex];
      }
    }

    return createTensor(outputShape, output);
  }

  function maskedFill(input, mask, value) {
    const inputState = requireTensor(input);
    const maskState = requireTensor(mask);
    validateMask(maskState);
    const outputShape = broadcastShape(inputState.shape, maskState.shape);
    if (outputShape.shape.length !== inputState.shape.length ||
      outputShape.shape.some((dimension, axis) => dimension !== inputState.shape[axis])) {
      throw tensorError("TENSOR_SHAPE_MISMATCH", "Mask broadcasting must preserve the input Shape.");
    }

    const fillValue = toFloat32(value);
    const output = allocate(inputState.size);
    if (inputState.size > 0) {
      const outputStrides = stridesFor(inputState.shape);
      const maskStrides = broadcastStrides(maskState.shape, inputState.shape);
      for (let index = 0; index < inputState.size; index += 1) {
        const coordinates = coordinatesFor(index, inputState.shape, outputStrides);
        let maskIndex = 0;
        for (let axis = 0; axis < inputState.shape.length; axis += 1) {
          maskIndex += coordinates[axis] * maskStrides[axis];
        }
        output[index] = maskState.data[maskIndex] === 1 ? fillValue : inputState.data[index];
      }
    }

    return createTensor(inputState.shape, output);
  }

  const ops = Object.freeze({
    transpose: (input, axes) => {
      requireTensor(input);
      return input.transpose(axes);
    },
    reshape: (input, shape) => {
      requireTensor(input);
      return input.reshape(shape);
    },
    slice: (input, spec) => {
      requireTensor(input);
      return input.slice(spec);
    },
    add: (left, right) => binaryOperation(left, right, (a, b) => a + b, "add"),
    sub: (left, right) => binaryOperation(left, right, (a, b) => a - b, "sub"),
    mul: (left, right) => binaryOperation(left, right, (a, b) => a * b, "mul"),
    div: (left, right) => binaryOperation(left, right, (a, b) => a / b, "div"),
    exp: (input) => unaryOperation(input, Math.exp),
    log: (input) => unaryOperation(input, (value) => {
      if (value <= 0) {
        throw tensorError("TENSOR_NUMERICAL_ERROR", "Log input must be positive.");
      }
      return Math.log(value);
    }),
    sqrt: (input) => unaryOperation(input, (value) => {
      if (value < 0) {
        throw tensorError("TENSOR_NUMERICAL_ERROR", "Sqrt input must be non-negative.");
      }
      return Math.sqrt(value);
    }),
    relu: (input) => unaryOperation(input, (value) => Math.max(0, value)),
    sigmoid: (input) => unaryOperation(input, (value) => {
      if (value >= 0) return 1 / (1 + Math.exp(-value));
      const exponential = Math.exp(value);
      return exponential / (1 + exponential);
    }),
    silu: (input) => unaryOperation(input, (value) => {
      const sigmoidValue = value >= 0
        ? 1 / (1 + Math.exp(-value))
        : Math.exp(value) / (1 + Math.exp(value));
      return value * sigmoidValue;
    }),
    gelu: (input) => unaryOperation(input, (value) => {
      const scaled = Math.sqrt(2 / Math.PI) * (value + 0.044715 * value ** 3);
      return 0.5 * value * (1 + Math.tanh(scaled));
    }),
    matmul,
    concat,
    where,
    gather,
    maskedFill,
    sum,
    mean,
    max,
    softmax: (input, axis) => softmax(input, axis, false),
    logSoftmax: (input, axis) => softmax(input, axis, true)
  });

  Object.defineProperties(Tensor, {
    MAX_RANK: { value: MAX_RANK },
    MAX_ELEMENTS: { value: MAX_ELEMENTS }
  });

  global.KNA = global.KNA || {};
  global.KNA.Tensor = Tensor;
  global.KNA.ops = ops;
})(globalThis);
