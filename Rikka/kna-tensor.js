(function (global) {
  "use strict";

  const MAX_RANK = 16;
  const MAX_ELEMENTS = 16777216;
  const INTERNAL = Symbol("KNA.Tensor");
  const states = new WeakMap();
  const preparedTensorData = new WeakMap();
  let nextTensorId = 1;
  let nextGraphId = 1;
  let gradTrackingEnabled = true;

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
        node: null,
        graphReleased: false,
        graphRefs: 0,
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

    set requiresGrad(value) {
      const state = assertUsable(this);
      if (typeof value !== "boolean") {
        throw tensorError("AUTOGRAD_INVALID_ARGUMENT", "requiresGrad must be a boolean.");
      }
      if (state.node) {
        throw tensorError("AUTOGRAD_INVALID_ARGUMENT", "requiresGrad cannot be changed on a graph result.");
      }
      state.requiresGrad = value;
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

    backward(options) {
      return runBackward(this, options);
    }

    zeroGrad() {
      const state = assertUsable(this);
      if (state.grad) {
        state.grad.dispose();
        state.grad = null;
      }
    }

    clone() {
      const state = assertUsable(this);
      const result = createTensor(state.shape, Float32Array.from(state.data));
      return recordOperation(result, [this], "clone", (gradient) => [Float32Array.from(gradient)]);
    }

    reshape(shape) {
      const state = assertUsable(this);
      const target = validateShape(shape);
      if (target.size !== state.size) {
        throw tensorError("TENSOR_SHAPE_MISMATCH", "Reshape must preserve element count.");
      }
      const result = createTensor(target.shape, Float32Array.from(state.data));
      return recordOperation(result, [this], "reshape", (gradient) => [Float32Array.from(gradient)]);
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

      const result = createTensor(outputShape, output);
      const inputShape = state.shape.slice();
      return recordOperation(result, [this], "transpose", (gradient) => {
        const inputGradient = allocate(state.size);
        const inputStrides = stridesFor(inputShape);
        const outputStrides = stridesFor(outputShape);
        for (let outputIndex = 0; outputIndex < state.size; outputIndex += 1) {
          const coordinates = coordinatesFor(outputIndex, outputShape, outputStrides);
          let inputIndex = 0;
          for (let axis = 0; axis < rank; axis += 1) {
            inputIndex += coordinates[axis] * inputStrides[order[axis]];
          }
          inputGradient[inputIndex] = gradient[outputIndex];
        }
        return [inputGradient];
      });
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

      const result = createTensor(outputShape, output);
      const inputShape = state.shape.slice();
      return recordOperation(result, [this], "slice", (gradient) => {
        const inputGradient = allocate(state.size);
        const inputStrides = stridesFor(inputShape);
        const outputStrides = stridesFor(outputShape);
        for (let outputIndex = 0; outputIndex < outputSize; outputIndex += 1) {
          const coordinates = coordinatesFor(outputIndex, outputShape, outputStrides);
          let inputIndex = 0;
          for (let axis = 0; axis < inputShape.length; axis += 1) {
            inputIndex += (coordinates[axis] + ranges[axis][0]) * inputStrides[axis];
          }
          inputGradient[inputIndex] = toFloat32(inputGradient[inputIndex] + gradient[outputIndex]);
        }
        return [inputGradient];
      });
    }

    dispose() {
      const state = states.get(this);
      if (state && !state.disposed) {
        state.disposed = true;
        if (state.graphRefs === 0) {
          releaseNode(this);
          cleanupDisposedState(state);
        }
      }
    }
  }

  function prepareTensorData(tensor, values) {
    const state = assertUsable(tensor);
    if (!values || typeof values.length !== "number" || values.length !== state.size) {
      throw tensorError("TENSOR_SHAPE_MISMATCH", "Updated data length does not match Tensor Shape.");
    }

    const token = Object.freeze({});
    preparedTensorData.set(token, {
      tensor,
      length: state.size,
      data: toFloat32Array(values)
    });
    return token;
  }

  function commitTensorDataBatch(updates) {
    if (!Array.isArray(updates)) {
      throw tensorError("TENSOR_INVALID_ARGUMENT", "Tensor updates must be an array.");
    }

    const prepared = updates.map((update) => {
      if (!Array.isArray(update) || update.length !== 2) {
        throw tensorError("TENSOR_INVALID_ARGUMENT", "Each update must contain a Tensor and preparation token.");
      }
      const [tensor, token] = update;
      const state = assertUsable(tensor);
      const payload = preparedTensorData.get(token);
      if (!payload || payload.tensor !== tensor || payload.length !== state.size) {
        throw tensorError("TENSOR_INVALID_ARGUMENT", "Invalid prepared Tensor update.");
      }
      if (state.graphRefs !== 0) {
        throw tensorError("TENSOR_ACTIVE_GRAPH", "Cannot update a Tensor referenced by an active graph.");
      }
      return { state, token, payload };
    });

    if (new Set(prepared.map((item) => item.state)).size !== prepared.length) {
      throw tensorError("TENSOR_INVALID_ARGUMENT", "A Tensor cannot be updated twice in one transaction.");
    }

    for (const item of prepared) {
      item.state.data = item.payload.data;
      item.state.version += 1;
      preparedTensorData.delete(item.token);
    }
  }

  function createCustomOperation(shape, values, inputs, operation, backwardRule) {
    if (!Array.isArray(inputs) || typeof operation !== "string" || operation.length === 0 ||
      typeof backwardRule !== "function") {
      throw tensorError("TENSOR_INVALID_ARGUMENT", "Custom operation metadata is invalid.");
    }
    const validated = validateShape(shape);
    const data = toFloat32Array(values);
    if (data.length !== validated.size) {
      throw tensorError("TENSOR_SHAPE_MISMATCH", "Custom operation data does not match Shape.");
    }
    const result = createTensor(validated.shape, data);
    return recordOperation(result, inputs, operation, backwardRule);
  }

  function recordOperation(output, inputs, operation, backwardRule) {
    const outputState = assertUsable(output);
    const trackedInputs = inputs.filter((input) => assertUsable(input).requiresGrad);
    if (!gradTrackingEnabled || trackedInputs.length === 0) return output;

    outputState.requiresGrad = true;
    outputState.graphReleased = false;
    outputState.node = {
      id: "graph-" + nextGraphId++,
      operation,
      inputs: inputs.slice(),
      backwardRule,
      released: false
    };
    for (const input of inputs) states.get(input).graphRefs += 1;
    return output;
  }

  function cleanupDisposedState(state) {
    if (!state.disposed || state.graphRefs !== 0) return;
    state.data = null;
    if (state.grad) {
      state.grad.dispose();
      state.grad = null;
    }
  }

  function releaseNode(tensor, visited = new Set()) {
    if (visited.has(tensor)) return;
    visited.add(tensor);
    const state = states.get(tensor);
    if (!state) return;

    const node = state.node;
    if (node) {
      node.released = true;
      state.node = null;
      state.graphReleased = true;
      for (const input of node.inputs) {
        const inputState = states.get(input);
        if (!inputState) continue;
        inputState.graphRefs = Math.max(0, inputState.graphRefs - 1);
        if (inputState.disposed && inputState.graphRefs === 0) {
          releaseNode(input, visited);
        }
      }
    }

    cleanupDisposedState(state);
  }

  function releaseGraph(tensors) {
    for (const tensor of tensors) releaseNode(tensor);
    for (const tensor of tensors) {
      const state = states.get(tensor);
      if (state) cleanupDisposedState(state);
    }
  }

  function collectGraph(root) {
    const order = [];
    const visited = new Set();
    const visiting = new Set();

    function visit(tensor) {
      const state = states.get(tensor);
      if (!state || (state.disposed && state.graphRefs === 0)) {
        throw tensorError("TENSOR_DISPOSED", "A required graph Tensor has been disposed.");
      }
      if (state.graphReleased) {
        throw tensorError("AUTOGRAD_GRAPH_RELEASED", "The computation graph has been released.");
      }
      if (visiting.has(tensor)) {
        throw tensorError("AUTOGRAD_INVALID_GRAPH", "The computation graph contains a cycle.");
      }
      if (visited.has(tensor)) return;

      visiting.add(tensor);
      if (state.node) {
        if (state.node.released) {
          throw tensorError("AUTOGRAD_GRAPH_RELEASED", "The computation graph has been released.");
        }
        for (const input of state.node.inputs) {
          const inputState = states.get(input);
          if (!inputState) {
            throw tensorError("AUTOGRAD_INVALID_GRAPH", "A graph input is unavailable.");
          }
          if (inputState.requiresGrad) visit(input);
        }
      }
      visiting.delete(tensor);
      visited.add(tensor);
      order.push(tensor);
    }

    visit(root);
    return order;
  }

  function addPendingGradient(pending, tensor, contribution) {
    const state = states.get(tensor);
    if (!state) throw tensorError("AUTOGRAD_INVALID_GRAPH", "A gradient target is unavailable.");
    if (!state.requiresGrad || contribution === null) return;
    if (!(contribution instanceof Float32Array) || contribution.length !== state.size) {
      throw tensorError("AUTOGRAD_GRADIENT_SHAPE", "Backward rule returned an invalid gradient shape.");
    }

    const current = pending.get(tensor);
    if (!current) {
      pending.set(tensor, Float32Array.from(contribution, toFloat32));
      return;
    }
    for (let index = 0; index < current.length; index += 1) {
      current[index] = toFloat32(current[index] + contribution[index]);
    }
  }

  function runBackward(root, options) {
    const rootState = assertUsable(root);
    let retainGraph = false;
    if (options !== undefined) {
      if (!options || typeof options !== "object" || Array.isArray(options)) {
        throw tensorError("AUTOGRAD_INVALID_ARGUMENT", "Backward options must be an object.");
      }
      if (options.retainGraph !== undefined && typeof options.retainGraph !== "boolean") {
        throw tensorError("AUTOGRAD_INVALID_ARGUMENT", "retainGraph must be a boolean.");
      }
      retainGraph = options.retainGraph === true;
    }

    if (rootState.shape.length !== 0) {
      throw tensorError("AUTOGRAD_NON_SCALAR_OUTPUT", "backward() requires a scalar Tensor.");
    }
    if (!rootState.requiresGrad) {
      throw tensorError("AUTOGRAD_NOT_TRACKED", "Tensor does not require gradients.");
    }

    let order = [];
    try {
      order = collectGraph(root);
      const pending = new Map([[root, Float32Array.of(1)]]);

      for (let position = order.length - 1; position >= 0; position -= 1) {
        const tensor = order[position];
        const outputGradient = pending.get(tensor);
        if (!outputGradient) continue;

        const state = states.get(tensor);
        if (!state) throw tensorError("AUTOGRAD_INVALID_GRAPH", "A graph Tensor is unavailable.");
        if (!state.node) continue;
        if (typeof state.node.backwardRule !== "function") {
          throw tensorError(
            "AUTOGRAD_BACKWARD_UNSUPPORTED",
            "No backward rule is registered for " + state.node.operation + "."
          );
        }

        const inputGradients = state.node.backwardRule(outputGradient);
        if (!Array.isArray(inputGradients) || inputGradients.length !== state.node.inputs.length) {
          throw tensorError("AUTOGRAD_INVALID_BACKWARD", "Backward rule returned an invalid result.");
        }
        for (let index = 0; index < state.node.inputs.length; index += 1) {
          addPendingGradient(pending, state.node.inputs[index], inputGradients[index]);
        }
      }

      const prepared = [];
      try {
        for (const [tensor, contribution] of pending) {
          const state = states.get(tensor);
          if (!state) throw tensorError("AUTOGRAD_INVALID_GRAPH", "A gradient target is unavailable.");
          const data = Float32Array.from(contribution);
          if (state.grad) {
            const previous = assertUsable(state.grad).data;
            for (let index = 0; index < data.length; index += 1) {
              data[index] = toFloat32(data[index] + previous[index]);
            }
          }
          prepared.push([state, createTensor(state.shape, data)]);
        }
      } catch (error) {
        for (const [, gradient] of prepared) gradient.dispose();
        throw error;
      }

      for (const [state, gradient] of prepared) {
        if (state.grad) state.grad.dispose();
        state.grad = gradient;
      }

      if (!retainGraph) releaseGraph(order);
    } catch (error) {
      if (order.length > 0) releaseGraph(order);
      throw error;
    }
  }

  function noGrad(callback) {
    if (typeof callback !== "function") {
      throw tensorError("AUTOGRAD_INVALID_ARGUMENT", "noGrad requires a function.");
    }

    const previous = gradTrackingEnabled;
    gradTrackingEnabled = false;
    try {
      const result = callback();
      if (result && typeof result.then === "function") {
        throw tensorError("AUTOGRAD_ASYNC_NOGRAD_UNSUPPORTED", "noGrad only supports synchronous callbacks.");
      }
      return result;
    } finally {
      gradTrackingEnabled = previous;
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

    const result = createTensor(output.shape, data);
    const tracked = gradTrackingEnabled && (leftState.requiresGrad || rightState.requiresGrad);
    const needsInputValues = tracked && (operationName === "mul" || operationName === "div");
    const savedLeft = needsInputValues ? Float32Array.from(leftState.data) : null;
    const savedRight = needsInputValues ? Float32Array.from(rightState.data) : null;

    return recordOperation(result, [left, right], operationName, tracked
      ? (outputGradient) => {
        const leftGradient = allocate(leftState.size);
        const rightGradient = allocate(rightState.size);
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

          const gradient = outputGradient[index];
          let leftContribution = gradient;
          let rightContribution = gradient;
          if (operationName === "mul") {
            leftContribution *= savedRight[rightIndex];
            rightContribution *= savedLeft[leftIndex];
          } else if (operationName === "div") {
            const divisor = savedRight[rightIndex];
            leftContribution /= divisor;
            rightContribution = -gradient * savedLeft[leftIndex] / (divisor * divisor);
          } else if (operationName === "sub") {
            rightContribution = -gradient;
          }

          leftGradient[leftIndex] = toFloat32(leftGradient[leftIndex] + leftContribution);
          rightGradient[rightIndex] = toFloat32(rightGradient[rightIndex] + rightContribution);
        }

        return [leftGradient, rightGradient];
      }
      : null);
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
    if (axes.length === 0) {
      const result = createTensor(inputState.shape, Float32Array.from(inputState.data));
      return recordOperation(result, [input], "sum", (gradient) => [Float32Array.from(gradient)]);
    }

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

    const result = createTensor(outputShape, output);
    const tracked = gradTrackingEnabled && inputState.requiresGrad;
    const inputShape = inputState.shape.slice();
    const outputShapeCopy = outputShape.slice();

    return recordOperation(result, [input], "sum", tracked
      ? (outputGradient) => {
        const inputGradient = allocate(inputState.size);
        if (inputState.size > 0) {
          const inputStrides = stridesFor(inputShape);
          const outputStrides = stridesFor(outputShapeCopy);
          const reduced = new Set(axes);
          for (let inputIndex = 0; inputIndex < inputState.size; inputIndex += 1) {
            const coordinates = coordinatesFor(inputIndex, inputShape, inputStrides);
            let outputIndex = 0;
            let outputAxis = 0;
            for (let inputAxis = 0; inputAxis < inputShape.length; inputAxis += 1) {
              if (reduced.has(inputAxis)) {
                if (keepDims) outputAxis += 1;
              } else {
                outputIndex += coordinates[inputAxis] * outputStrides[outputAxis];
                outputAxis += 1;
              }
            }
            inputGradient[inputIndex] = outputGradient[outputIndex];
          }
        }
        return [inputGradient];
      }
      : null);
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
    const keepDims = Boolean(options && typeof options === "object" && options.keepDims === true);
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
    const outputShape = totalState.shape.slice();
    const inputShape = inputState.shape.slice();
    const reduced = new Set(axes);
    const result = createTensor(outputShape, data);
    total.dispose();
    return recordOperation(result, [input], "mean", (outputGradient) => {
      const inputGradient = allocate(inputState.size);
      if (inputState.size > 0) {
        const inputStrides = stridesFor(inputShape);
        const outputStrides = stridesFor(outputShape);
        for (let inputIndex = 0; inputIndex < inputState.size; inputIndex += 1) {
          const coordinates = coordinatesFor(inputIndex, inputShape, inputStrides);
          let outputIndex = 0;
          let outputAxis = 0;
          for (let inputAxis = 0; inputAxis < inputShape.length; inputAxis += 1) {
            if (reduced.has(inputAxis)) {
              if (keepDims) outputAxis += 1;
            } else {
              outputIndex += coordinates[inputAxis] * outputStrides[outputAxis];
              outputAxis += 1;
            }
          }
          inputGradient[inputIndex] = toFloat32(outputGradient[outputIndex] / count);
        }
      }
      return [inputGradient];
    });
  }

  function max(input, options) {
    const inputState = requireTensor(input);
    const axis = reductionAxisFromOptions(options);
    const axes = normalizeReductionAxes(axis, inputState.shape.length);
    if (axes.length === 0) {
      const result = createTensor(inputState.shape, Float32Array.from(inputState.data));
      return recordOperation(result, [input], "max", (gradient) => [Float32Array.from(gradient)]);
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

    const result = createTensor(outputShape, output);
    const inputShape = inputState.shape.slice();
    const outputValues = Float32Array.from(output);
    const reduced = new Set(axes);
    const inputStrides = stridesFor(inputShape);
    const outputStrides = stridesFor(outputShape);

    function outputIndexFor(inputIndex) {
      const coordinates = coordinatesFor(inputIndex, inputShape, inputStrides);
      let outputIndex = 0;
      let outputAxis = 0;
      for (let inputAxis = 0; inputAxis < inputShape.length; inputAxis += 1) {
        if (reduced.has(inputAxis)) {
          if (keepDims) outputAxis += 1;
        } else {
          outputIndex += coordinates[inputAxis] * outputStrides[outputAxis];
          outputAxis += 1;
        }
      }
      return outputIndex;
    }

    return recordOperation(result, [input], "max", (outputGradient) => {
      const inputGradient = allocate(inputState.size);
      const winnerCounts = new Uint32Array(outputSize);
      for (let inputIndex = 0; inputIndex < inputState.size; inputIndex += 1) {
        const outputIndex = outputIndexFor(inputIndex);
        if (inputState.data[inputIndex] === outputValues[outputIndex]) {
          winnerCounts[outputIndex] += 1;
        }
      }
      for (let inputIndex = 0; inputIndex < inputState.size; inputIndex += 1) {
        const outputIndex = outputIndexFor(inputIndex);
        if (inputState.data[inputIndex] === outputValues[outputIndex]) {
          inputGradient[inputIndex] = toFloat32(
            outputGradient[outputIndex] / winnerCounts[outputIndex]
          );
        }
      }
      return [inputGradient];
    });
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

  function unaryOperation(input, operation, operationName, derivative) {
    const inputState = requireTensor(input);
    const output = allocate(inputState.size);
    for (let index = 0; index < inputState.size; index += 1) {
      output[index] = toFloat32(operation(inputState.data[index]));
    }
    const result = createTensor(inputState.shape, output);
    const tracked = gradTrackingEnabled && inputState.requiresGrad;
    const savedInput = tracked && derivative ? Float32Array.from(inputState.data) : null;
    const savedOutput = tracked && derivative ? Float32Array.from(output) : null;

    return recordOperation(result, [input], operationName, tracked && derivative
      ? (outputGradient) => {
        const inputGradient = allocate(inputState.size);
        for (let index = 0; index < inputState.size; index += 1) {
          inputGradient[index] = toFloat32(
            outputGradient[index] * derivative(savedInput[index], savedOutput[index])
          );
        }
        return [inputGradient];
      }
      : null);
  }

  function softmax(input, axis, logarithmic) {
    const inputState = requireTensor(input);
    const normalizedAxis = normalizeAxis(axis, inputState.shape.length);
    const axisSize = inputState.shape[normalizedAxis];
    if (axisSize === 0) {
      throw tensorError("TENSOR_NUMERICAL_ERROR", "Softmax over an empty axis is not defined.");
    }

    const output = allocate(inputState.size);

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

    const result = createTensor(inputState.shape, output);
    const tracked = gradTrackingEnabled && inputState.requiresGrad;
    const savedOutput = tracked ? Float32Array.from(output) : null;
    return recordOperation(result, [input], logarithmic ? "logSoftmax" : "softmax", tracked
      ? (outputGradient) => {
        const inputGradient = allocate(inputState.size);
        for (let outer = 0; outer < outerSize; outer += 1) {
          for (let inner = 0; inner < innerSize; inner += 1) {
            let groupSum = 0;
            for (let index = 0; index < axisSize; index += 1) {
              const offset = (outer * axisSize + index) * innerSize + inner;
              groupSum += logarithmic
                ? outputGradient[offset]
                : outputGradient[offset] * savedOutput[offset];
            }
            for (let index = 0; index < axisSize; index += 1) {
              const offset = (outer * axisSize + index) * innerSize + inner;
              inputGradient[offset] = toFloat32(logarithmic
                ? outputGradient[offset] - Math.exp(savedOutput[offset]) * groupSum
                : savedOutput[offset] * (outputGradient[offset] - groupSum));
            }
          }
        }
        return [inputGradient];
      }
      : null);
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

    const result = createTensor(outputShape, output);
    const tracked = gradTrackingEnabled && (leftState.requiresGrad || rightState.requiresGrad);
    const savedLeft = tracked ? Float32Array.from(leftState.data) : null;
    const savedRight = tracked ? Float32Array.from(rightState.data) : null;
    const savedBatchShape = batch.shape.slice();

    return recordOperation(result, [left, right], "matmul", tracked
      ? (outputGradient) => {
        const leftGradient = allocate(leftState.size);
        const rightGradient = allocate(rightState.size);
        const leftStrides = broadcastStrides(leftBatch, savedBatchShape);
        const rightStrides = broadcastStrides(rightBatch, savedBatchShape);
        const batchStrides = stridesFor(savedBatchShape);
        const batchCount = savedBatchShape.length === 0 ? 1 : batch.size;

        for (let batchIndex = 0; batchIndex < batchCount; batchIndex += 1) {
          const coordinates = savedBatchShape.length === 0
            ? []
            : coordinatesFor(batchIndex, savedBatchShape, batchStrides);
          let leftOffset = 0;
          let rightOffset = 0;
          for (let axis = 0; axis < savedBatchShape.length; axis += 1) {
            leftOffset += coordinates[axis] * leftStrides[axis];
            rightOffset += coordinates[axis] * rightStrides[axis];
          }
          leftOffset *= rows * inner;
          rightOffset *= inner * columns;
          const outputOffset = batchIndex * rows * columns;

          for (let row = 0; row < rows; row += 1) {
            for (let column = 0; column < columns; column += 1) {
              const gradient = outputGradient[outputOffset + row * columns + column];
              for (let index = 0; index < inner; index += 1) {
                const leftIndex = leftOffset + row * inner + index;
                const rightIndex = rightOffset + index * columns + column;
                leftGradient[leftIndex] = toFloat32(
                  leftGradient[leftIndex] + gradient * savedRight[rightIndex]
                );
                rightGradient[rightIndex] = toFloat32(
                  rightGradient[rightIndex] + savedLeft[leftIndex] * gradient
                );
              }
            }
          }
        }
        return [leftGradient, rightGradient];
      }
      : null);
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
    if (output.length === 0) return createTensor(outputShape, output);

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

    const result = createTensor(outputShape, output);
    return recordOperation(result, tensors, "concat", (outputGradient) => {
      const inputGradients = inputStates.map((state) => allocate(state.size));
      for (let outer = 0; outer < outerSize; outer += 1) {
        let axisOffset = 0;
        for (let inputIndex = 0; inputIndex < inputStates.length; inputIndex += 1) {
          const state = inputStates[inputIndex];
          const segmentSize = state.shape[normalizedAxis] * innerSize;
          const sourceOffset = outer * segmentSize;
          const targetOffset = (outer * outputAxisSize + axisOffset) * innerSize;
          for (let index = 0; index < segmentSize; index += 1) {
            inputGradients[inputIndex][sourceOffset + index] = outputGradient[targetOffset + index];
          }
          axisOffset += state.shape[normalizedAxis];
        }
      }
      return inputGradients;
    });
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
    if (conditionState.requiresGrad) {
      throw tensorError("AUTOGRAD_NON_DIFFERENTIABLE_INPUT", "where condition cannot require gradients.");
    }

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

    const result = createTensor(outputShape.shape, output);
    return recordOperation(result, [condition, left, right], "where", (outputGradient) => {
      const leftGradient = allocate(leftState.size);
      const rightGradient = allocate(rightState.size);
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
        if (conditionState.data[conditionIndex] === 1) {
          leftGradient[leftIndex] = toFloat32(leftGradient[leftIndex] + outputGradient[index]);
        } else {
          rightGradient[rightIndex] = toFloat32(rightGradient[rightIndex] + outputGradient[index]);
        }
      }

      return [null, leftGradient, rightGradient];
    });
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

    const result = createTensor(outputShape, output);
    const inputShape = inputState.shape.slice();
    return recordOperation(result, [input], "gather", (outputGradient) => {
      const inputGradient = allocate(inputState.size);
      const inputStrides = stridesFor(inputShape);
      const outputStrides = stridesFor(outputShape);
      for (let outputIndex = 0; outputIndex < outputGradient.length; outputIndex += 1) {
        const coordinates = coordinatesFor(outputIndex, outputShape, outputStrides);
        coordinates[normalizedAxis] = indices[coordinates[normalizedAxis]];
        let inputIndex = 0;
        for (let dimension = 0; dimension < inputShape.length; dimension += 1) {
          inputIndex += coordinates[dimension] * inputStrides[dimension];
        }
        inputGradient[inputIndex] = toFloat32(inputGradient[inputIndex] + outputGradient[outputIndex]);
      }
      return [inputGradient];
    });
  }

  function maskedFill(input, mask, value) {
    const inputState = requireTensor(input);
    const maskState = requireTensor(mask);
    validateMask(maskState);
    if (maskState.requiresGrad) {
      throw tensorError("AUTOGRAD_NON_DIFFERENTIABLE_INPUT", "maskedFill mask cannot require gradients.");
    }
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

    const result = createTensor(inputState.shape, output);
    return recordOperation(result, [input, mask], "maskedFill", (outputGradient) => {
      const inputGradient = allocate(inputState.size);
      const outputStrides = stridesFor(inputState.shape);
      const maskStrides = broadcastStrides(maskState.shape, inputState.shape);
      for (let index = 0; index < inputState.size; index += 1) {
        const coordinates = coordinatesFor(index, inputState.shape, outputStrides);
        let maskIndex = 0;
        for (let axis = 0; axis < inputState.shape.length; axis += 1) {
          maskIndex += coordinates[axis] * maskStrides[axis];
        }
        if (maskState.data[maskIndex] === 0) inputGradient[index] = outputGradient[index];
      }
      return [inputGradient, null];
    });
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
    exp: (input) => unaryOperation(input, Math.exp, "exp", (value, output) => output),
    log: (input) => unaryOperation(input, (value) => {
      if (value <= 0) {
        throw tensorError("TENSOR_NUMERICAL_ERROR", "Log input must be positive.");
      }
      return Math.log(value);
    }, "log", (value) => 1 / value),
    sqrt: (input) => unaryOperation(input, (value) => {
      if (value < 0) {
        throw tensorError("TENSOR_NUMERICAL_ERROR", "Sqrt input must be non-negative.");
      }
      return Math.sqrt(value);
    }, "sqrt", (_, output) => output === 0 ? 0 : 0.5 / output),
    relu: (input) => unaryOperation(
      input,
      (value) => Math.max(0, value),
      "relu",
      (value) => value > 0 ? 1 : 0
    ),
    sigmoid: (input) => unaryOperation(input, (value) => {
      if (value >= 0) return 1 / (1 + Math.exp(-value));
      const exponential = Math.exp(value);
      return exponential / (1 + exponential);
    }, "sigmoid", (_, output) => output * (1 - output)),
    silu: (input) => unaryOperation(input, (value) => {
      const sigmoidValue = value >= 0
        ? 1 / (1 + Math.exp(-value))
        : Math.exp(value) / (1 + Math.exp(value));
      return value * sigmoidValue;
    }, "silu", (value) => {
      const sigmoidValue = value >= 0
        ? 1 / (1 + Math.exp(-value))
        : Math.exp(value) / (1 + Math.exp(value));
      return sigmoidValue + value * sigmoidValue * (1 - sigmoidValue);
    }),
    gelu: (input) => unaryOperation(input, (value) => {
      const scaled = Math.sqrt(2 / Math.PI) * (value + 0.044715 * value ** 3);
      return 0.5 * value * (1 + Math.tanh(scaled));
    }, "gelu", (value) => {
      const coefficient = Math.sqrt(2 / Math.PI);
      const scaled = coefficient * (value + 0.044715 * value ** 3);
      const hyperbolic = Math.tanh(scaled);
      const derivative = coefficient * (1 + 3 * 0.044715 * value ** 2);
      return 0.5 * (1 + hyperbolic) +
        0.5 * value * (1 - hyperbolic * hyperbolic) * derivative;
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
  global.KNA.noGrad = noGrad;
  global.KNA.isGradEnabled = () => gradTrackingEnabled;
  global.KNA.isGradEnabled = () => gradTrackingEnabled;
  global.KNA._internal = Object.freeze({
    prepareTensorData,
    commitTensorDataBatch,
    createCustomOperation
  });
})(globalThis);
