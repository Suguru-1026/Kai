(function (global) {
  "use strict";

  function runTests() {
    const status = document.getElementById("status");
    const log = document.getElementById("log");
    const Tensor = global.KNA.Tensor;
    const ops = global.KNA.ops;
    const results = [];
    let passed = 0;
    let failed = 0;

    function assert(condition, message) {
      if (!condition) throw new Error(message || "Assertion failed.");
    }

    function equal(actual, expected) {
      assert(actual === expected, "actual=" + actual + ", expected=" + expected);
    }

    function arrayEqual(actual, expected) {
      equal(actual.length, expected.length);
      for (let index = 0; index < expected.length; index += 1) {
        assert(
          Math.abs(actual[index] - expected[index]) <= 1e-6,
          "index=" + index + ", actual=" + actual[index] + ", expected=" + expected[index]
        );
      }
    }

    function throwsCode(fn, code) {
      try {
        fn();
      } catch (error) {
        equal(error.code, code);
        return;
      }
      throw new Error("Expected error " + code + ".");
    }

    function test(name, fn) {
      try {
        fn();
        passed += 1;
        results.push("PASS " + name);
      } catch (error) {
        failed += 1;
        results.push("FAIL " + name + "\n  " + (error.code || error.name) + ": " + error.message);
      }
    }

    test("scalar and FP32 metadata", () => {
      const value = Tensor.scalar(3);
      arrayEqual(value.shape, []);
      equal(value.dtype, "float32");
      equal(value.item(), 3);
      equal(value.requiresGrad, false);
      equal(value.grad, null);
      value.dispose();
    });

    test("nested input shape and copy ownership", () => {
      const input = [[1, 2, 3], [4, 5, 6]];
      const value = Tensor.fromArray(input);
      arrayEqual(value.shape, [2, 3]);
      input[0][0] = 99;
      arrayEqual(value.toArray(), [1, 2, 3, 4, 5, 6]);
      value.dispose();
    });

    test("zeros, ones, and full", () => {
      const zeros = Tensor.zeros([2, 2]);
      const ones = Tensor.ones([3]);
      const full = Tensor.full([2], 2.5);
      arrayEqual(zeros.toArray(), [0, 0, 0, 0]);
      arrayEqual(ones.toArray(), [1, 1, 1]);
      arrayEqual(full.toArray(), [2.5, 2.5]);
      zeros.dispose();
      ones.dispose();
      full.dispose();
    });

    test("zero-element shape", () => {
      const value = Tensor.zeros([2, 0, 4]);
      arrayEqual(value.shape, [2, 0, 4]);
      equal(value.size, 0);
      arrayEqual(value.toArray(), []);
      value.dispose();
    });

    test("clone and reshape copy data", () => {
      const source = Tensor.fromArray([1, 2, 3, 4]);
      const copy = source.clone();
      const reshaped = ops.reshape(source, [2, 2]);
      assert(source.id !== copy.id, "Tensor IDs must be unique.");
      arrayEqual(reshaped.shape, [2, 2]);
      arrayEqual(copy.toArray(), [1, 2, 3, 4]);
      source.dispose();
      arrayEqual(copy.toArray(), [1, 2, 3, 4]);
      copy.dispose();
      reshaped.dispose();
    });

    test("transpose defaults to reversed axes", () => {
      const source = Tensor.fromArray([[1, 2, 3], [4, 5, 6]]);
      const transposed = ops.transpose(source);
      arrayEqual(transposed.shape, [3, 2]);
      arrayEqual(transposed.toArray(), [1, 4, 2, 5, 3, 6]);
      source.dispose();
      transposed.dispose();
    });

    test("slice uses half-open ranges", () => {
      const source = Tensor.fromArray([[1, 2, 3], [4, 5, 6]]);
      const sliced = ops.slice(source, [[0, 2], [1, 3]]);
      arrayEqual(sliced.shape, [2, 2]);
      arrayEqual(sliced.toArray(), [2, 3, 5, 6]);
      source.dispose();
      sliced.dispose();
    });

    test("broadcasted elementwise operations", () => {
      const matrix = Tensor.fromArray([[1, 2, 3], [4, 5, 6]]);
      const vector = Tensor.fromArray([10, 20, 30]);
      const scale = Tensor.scalar(2);
      const added = ops.add(matrix, vector);
      const subtracted = ops.sub(added, vector);
      const multiplied = ops.mul(vector, scale);
      const divided = ops.div(multiplied, scale);
      arrayEqual(added.shape, [2, 3]);
      arrayEqual(added.toArray(), [11, 22, 33, 14, 25, 36]);
      arrayEqual(subtracted.toArray(), [1, 2, 3, 4, 5, 6]);
      arrayEqual(divided.toArray(), [10, 20, 30]);
      matrix.dispose();
      vector.dispose();
      scale.dispose();
      added.dispose();
      subtracted.dispose();
      multiplied.dispose();
      divided.dispose();
    });

    test("rejects incompatible broadcast and division by zero", () => {
      const left = Tensor.ones([2, 3]);
      const right = Tensor.ones([4]);
      const zero = Tensor.scalar(0);
      throwsCode(() => ops.add(left, right), "TENSOR_SHAPE_MISMATCH");
      throwsCode(() => ops.div(left, zero), "TENSOR_NUMERICAL_ERROR");
      left.dispose();
      right.dispose();
      zero.dispose();
    });

    test("matrix multiplication", () => {
      const left = Tensor.fromArray([[1, 2], [3, 4]]);
      const right = Tensor.fromArray([[5, 6], [7, 8]]);
      const product = ops.matmul(left, right);
      arrayEqual(product.shape, [2, 2]);
      arrayEqual(product.toArray(), [19, 22, 43, 50]);
      left.dispose();
      right.dispose();
      product.dispose();
    });

    test("batched matrix multiplication broadcasts batch axes", () => {
      const matrices = Tensor.fromArray([
        [[1, 0], [0, 1]],
        [[2, 0], [0, 2]]
      ]);
      const right = Tensor.fromArray([[1, 2], [3, 4]]);
      const product = ops.matmul(matrices, right);
      arrayEqual(product.shape, [2, 2, 2]);
      arrayEqual(product.toArray(), [1, 2, 3, 4, 2, 4, 6, 8]);
      matrices.dispose();
      right.dispose();
      product.dispose();
    });

    test("sum supports axes, keepDims, and empty reductions", () => {
      const value = Tensor.fromArray([[1, 2, 3], [4, 5, 6]]);
      const byColumn = ops.sum(value, { axis: 0 });
      const kept = ops.sum(value, { axis: -1, keepDims: true });
      const all = ops.sum(value);
      const unchanged = ops.sum(value, { axis: [] });
      const empty = Tensor.zeros([0, 2]);
      const emptySum = ops.sum(empty);
      arrayEqual(byColumn.shape, [3]);
      arrayEqual(byColumn.toArray(), [5, 7, 9]);
      arrayEqual(kept.shape, [2, 1]);
      arrayEqual(kept.toArray(), [6, 15]);
      equal(all.item(), 21);
      arrayEqual(unchanged.toArray(), value.toArray());
      equal(emptySum.item(), 0);
      value.dispose();
      byColumn.dispose();
      kept.dispose();
      all.dispose();
      unchanged.dispose();
      empty.dispose();
      emptySum.dispose();
    });

    test("mean and max reductions", () => {
      const value = Tensor.fromArray([[1, 5, 3], [4, 2, 6]]);
      const means = ops.mean(value, { axis: 0 });
      const maxima = ops.max(value, { axis: -1 });
      const maximum = ops.max(value);
      const empty = Tensor.zeros([0, 2]);
      arrayEqual(means.toArray(), [2.5, 3.5, 4.5]);
      arrayEqual(maxima.toArray(), [5, 6]);
      equal(maximum.item(), 6);
      throwsCode(() => ops.mean(empty), "TENSOR_NUMERICAL_ERROR");
      throwsCode(() => ops.max(empty), "TENSOR_NUMERICAL_ERROR");
      value.dispose();
      empty.dispose();
      means.dispose();
      maxima.dispose();
      maximum.dispose();
    });

    test("unary math and activations", () => {
      const values = Tensor.fromArray([-1, 0, 1]);
      const relu = ops.relu(values);
      const sigmoid = ops.sigmoid(values);
      const silu = ops.silu(values);
      const exponents = Tensor.fromArray([0, 1]);
      const exponential = ops.exp(exponents);
      const positive = Tensor.fromArray([1, 4]);
      const logarithm = ops.log(positive);
      const root = ops.sqrt(positive);
      arrayEqual(relu.toArray(), [0, 0, 1]);
      arrayEqual(sigmoid.toArray(), [0.2689414, 0.5, 0.7310586]);
      arrayEqual(silu.toArray(), [-0.2689414, 0, 0.7310586]);
      arrayEqual(exponential.toArray(), [1, Math.E]);
      arrayEqual(logarithm.toArray(), [0, Math.log(4)]);
      arrayEqual(root.toArray(), [1, 2]);
      values.dispose();
      relu.dispose();
      sigmoid.dispose();
      silu.dispose();
      exponents.dispose();
      exponential.dispose();
      positive.dispose();
      logarithm.dispose();
      root.dispose();
    });

    test("softmax is stable and normalized", () => {
      const logits = Tensor.fromArray([[1000, 1000], [0, 1]]);
      const probabilities = ops.softmax(logits, -1);
      const rowSums = ops.sum(probabilities, { axis: -1 });
      arrayEqual(probabilities.toArray(), [0.5, 0.5, 0.2689414, 0.7310586]);
      arrayEqual(rowSums.toArray(), [1, 1]);
      logits.dispose();
      probabilities.dispose();
      rowSums.dispose();
    });

    test("logSoftmax and invalid numerical domains", () => {
      const logits = Tensor.fromArray([[0, 0], [0, Math.log(3)]]);
      const logProbabilities = ops.logSoftmax(logits, 1);
      const zero = Tensor.scalar(0);
      const negative = Tensor.scalar(-1);
      const emptyAxis = Tensor.zeros([2, 0]);
      arrayEqual(logProbabilities.toArray(), [
        -Math.log(2),
        -Math.log(2),
        -Math.log(4),
        -Math.log(4 / 3)
      ]);
      throwsCode(() => ops.log(zero), "TENSOR_NUMERICAL_ERROR");
      throwsCode(() => ops.sqrt(negative), "TENSOR_NUMERICAL_ERROR");
      throwsCode(() => ops.softmax(emptyAxis, 1), "TENSOR_NUMERICAL_ERROR");
      logits.dispose();
      logProbabilities.dispose();
      zero.dispose();
      negative.dispose();
      emptyAxis.dispose();
    });

    test("GELU uses the tanh approximation", () => {
      const input = Tensor.fromArray([-1, 0, 1]);
      const output = ops.gelu(input);
      arrayEqual(output.toArray(), [-0.158808, 0, 0.841192]);
      input.dispose();
      output.dispose();
    });

    test("concat joins along positive and negative axes", () => {
      const left = Tensor.fromArray([[1, 2], [3, 4]]);
      const right = Tensor.fromArray([[5], [6]]);
      const bottom = Tensor.fromArray([[5, 6]]);
      const columns = ops.concat([left, right], -1);
      const rows = ops.concat([left, bottom], 0);
      const emptyLeft = Tensor.zeros([2, 0, 3]);
      const emptyRight = Tensor.zeros([2, 0, 3]);
      const empty = ops.concat([emptyLeft, emptyRight], 1);
      arrayEqual(columns.shape, [2, 3]);
      arrayEqual(columns.toArray(), [1, 2, 5, 3, 4, 6]);
      arrayEqual(rows.shape, [3, 2]);
      arrayEqual(rows.toArray(), [1, 2, 3, 4, 5, 6]);
      arrayEqual(empty.shape, [2, 0, 3]);
      equal(empty.size, 0);
      left.dispose();
      right.dispose();
      bottom.dispose();
      columns.dispose();
      rows.dispose();
      emptyLeft.dispose();
      emptyRight.dispose();
      empty.dispose();
    });

    test("where and maskedFill broadcast binary masks", () => {
      const condition = Tensor.fromArray([[1], [0]]);
      const mask = Tensor.fromArray([0, 1, 0]);
      const values = Tensor.fromArray([[1, 2, 3], [4, 5, 6]]);
      const alternative = Tensor.scalar(-1);
      const invalidCondition = Tensor.scalar(2);
      const selected = ops.where(condition, values, alternative);
      const filled = ops.maskedFill(values, mask, -100);
      arrayEqual(selected.toArray(), [1, 2, 3, -1, -1, -1]);
      arrayEqual(filled.toArray(), [1, -100, 3, 4, -100, 6]);
      throwsCode(() => ops.where(invalidCondition, values, alternative), "TENSOR_INVALID_ARGUMENT");
      condition.dispose();
      mask.dispose();
      values.dispose();
      alternative.dispose();
      invalidCondition.dispose();
      selected.dispose();
      filled.dispose();
    });

    test("gather selects rows and columns", () => {
      const input = Tensor.fromArray([[1, 2, 3], [4, 5, 6]]);
      const rows = ops.gather(input, [1, 0], 0);
      const columns = ops.gather(input, [2, 0], -1);
      arrayEqual(rows.shape, [2, 3]);
      arrayEqual(rows.toArray(), [4, 5, 6, 1, 2, 3]);
      arrayEqual(columns.shape, [2, 2]);
      arrayEqual(columns.toArray(), [3, 1, 6, 4]);
      throwsCode(() => ops.gather(input, [2], 0), "TENSOR_INVALID_ARGUMENT");
      input.dispose();
      rows.dispose();
      columns.dispose();
    });

    test("backward accumulates gradients through branches", () => {
      const input = Tensor.fromArray([1, 2]);
      input.requiresGrad = true;
      const square = ops.mul(input, input);
      const shifted = ops.add(square, input);
      const loss = ops.sum(shifted);
      loss.backward();
      arrayEqual(input.grad.toArray(), [3, 5]);
      equal(loss.grad.item(), 1);
      throwsCode(() => loss.backward(), "AUTOGRAD_GRAPH_RELEASED");
      input.dispose();
      square.dispose();
      shifted.dispose();
      loss.dispose();
    });

    test("backward unbroadcasts gradients and differentiates mean", () => {
      const matrix = Tensor.fromArray([[1, 2, 3], [4, 5, 6]]);
      const vector = Tensor.fromArray([10, 20, 30]);
      vector.requiresGrad = true;
      const mean = ops.mean(ops.add(matrix, vector));
      mean.backward();
      arrayEqual(vector.grad.toArray(), [1 / 3, 1 / 3, 1 / 3]);
      matrix.dispose();
      vector.dispose();
      mean.dispose();
    });

    test("shape-operation backward restores input coordinates", () => {
      const input = Tensor.fromArray([1, 2, 3, 4]);
      input.requiresGrad = true;
      const reshaped = ops.reshape(input, [2, 2]);
      const transposed = ops.transpose(reshaped);
      const selected = ops.slice(transposed, [[0, 1], [0, 2]]);
      const loss = ops.sum(selected);
      loss.backward();
      arrayEqual(input.grad.toArray(), [1, 0, 1, 0]);
      input.dispose();
      reshaped.dispose();
      transposed.dispose();
      selected.dispose();
      loss.dispose();
    });

    test("matmul backward returns both matrix gradients", () => {
      const left = Tensor.fromArray([[1, 2], [3, 4]]);
      const right = Tensor.fromArray([[5, 6], [7, 8]]);
      left.requiresGrad = true;
      right.requiresGrad = true;
      const loss = ops.sum(ops.matmul(left, right));
      loss.backward();
      arrayEqual(left.grad.toArray(), [11, 15, 11, 15]);
      arrayEqual(right.grad.toArray(), [4, 4, 6, 6]);
      left.dispose();
      right.dispose();
      loss.dispose();
    });

    test("batched matmul sums gradients across broadcast batches", () => {
      const left = Tensor.fromArray([
        [[1, 0], [0, 1]],
        [[2, 0], [0, 2]]
      ]);
      const right = Tensor.fromArray([[1, 2], [3, 4]]);
      left.requiresGrad = true;
      right.requiresGrad = true;
      const loss = ops.sum(ops.matmul(left, right));
      loss.backward();
      arrayEqual(left.grad.toArray(), [3, 7, 3, 7, 3, 7, 3, 7]);
      arrayEqual(right.grad.toArray(), [3, 3, 3, 3]);
      left.dispose();
      right.dispose();
      loss.dispose();
    });

    test("unary and division backward rules", () => {
      const zero = Tensor.scalar(0);
      zero.requiresGrad = true;
      const activationLoss = ops.add(
        ops.add(ops.sigmoid(zero), ops.silu(zero)),
        ops.gelu(zero)
      );
      activationLoss.backward();
      equal(zero.grad.item(), 1.25);

      const numerator = Tensor.scalar(2);
      const denominator = Tensor.scalar(4);
      numerator.requiresGrad = true;
      denominator.requiresGrad = true;
      const value = ops.add(ops.add(ops.exp(numerator), ops.log(numerator)), ops.div(numerator, denominator));
      value.backward();
      arrayEqual([numerator.grad.item()], [Math.E ** 2 + 0.75]);
      arrayEqual([denominator.grad.item()], [-0.125]);
      zero.dispose();
      activationLoss.dispose();
      numerator.dispose();
      denominator.dispose();
      value.dispose();
    });

    test("softmax backward computes the Jacobian-vector product", () => {
      const logits = Tensor.fromArray([0, Math.log(3)]);
      logits.requiresGrad = true;
      const probabilities = ops.softmax(logits, 0);
      const first = ops.slice(probabilities, [[0, 1]]);
      const loss = ops.sum(first);
      loss.backward();
      arrayEqual(logits.grad.toArray(), [0.1875, -0.1875]);
      logits.dispose();
      probabilities.dispose();
      first.dispose();
      loss.dispose();
    });

    test("concat and where route gradients to their sources", () => {
      const left = Tensor.fromArray([1, 2]);
      const right = Tensor.fromArray([3, 4]);
      const condition = Tensor.fromArray([1, 0]);
      left.requiresGrad = true;
      right.requiresGrad = true;
      const joined = ops.concat([left, right], 0);
      const concatLoss = ops.sum(joined);
      concatLoss.backward();
      const selected = ops.where(condition, left, right);
      const whereLoss = ops.sum(selected);
      whereLoss.backward();
      arrayEqual(left.grad.toArray(), [2, 1]);
      arrayEqual(right.grad.toArray(), [1, 2]);
      left.dispose();
      right.dispose();
      condition.dispose();
      joined.dispose();
      selected.dispose();
      concatLoss.dispose();
      whereLoss.dispose();
    });

    test("gather and maskedFill route gradients through selected values", () => {
      const input = Tensor.fromArray([[1, 2, 3], [4, 5, 6]]);
      const maskInput = Tensor.fromArray([[1, 2, 3], [4, 5, 6]]);
      const indices = Tensor.fromArray([2, 0, 2]);
      const mask = Tensor.fromArray([0, 1, 0]);
      input.requiresGrad = true;
      maskInput.requiresGrad = true;
      const gathered = ops.gather(input, [2, 0], 1);
      const filled = ops.maskedFill(maskInput, mask, -100);
      const loss = ops.add(ops.sum(gathered), ops.sum(filled));
      loss.backward();
      arrayEqual(input.grad.toArray(), [1, 0, 1, 1, 0, 1]);
      arrayEqual(maskInput.grad.toArray(), [1, 0, 1, 1, 0, 1]);
      input.dispose();
      maskInput.dispose();
      indices.dispose();
      mask.dispose();
      gathered.dispose();
      filled.dispose();
      loss.dispose();
    });

    test("retainGraph accumulates and noGrad restores state", () => {
      const input = Tensor.scalar(2);
      input.requiresGrad = true;
      const loss = ops.mul(input, input);
      loss.backward({ retainGraph: true });
      loss.backward({ retainGraph: true });
      equal(input.grad.item(), 8);
      loss.backward();
      equal(input.grad.item(), 12);
      throwsCode(() => loss.backward(), "AUTOGRAD_GRAPH_RELEASED");

      const detached = global.KNA.noGrad(() => {
        equal(global.KNA.isGradEnabled(), false);
        const nested = global.KNA.noGrad(() => {
          equal(global.KNA.isGradEnabled(), false);
          return ops.add(input, input);
        });
        equal(global.KNA.isGradEnabled(), false);
        nested.dispose();
        return ops.mul(input, input);
      });
      equal(global.KNA.isGradEnabled(), true);
      equal(detached.requiresGrad, false);
      try {
        global.KNA.noGrad(() => {
          throw new Error("callback failure");
        });
      } catch (error) {
        equal(error.message, "callback failure");
      }
      equal(global.KNA.isGradEnabled(), true);
      input.dispose();
      loss.dispose();
      detached.dispose();
    });

    test("failed backward preserves previously committed gradients", () => {
      const input = Tensor.fromArray([2, 3]);
      input.requiresGrad = true;
      const initial = ops.sum(input);
      initial.backward();
      const unsupported = ops.max(input);
      equal(unsupported.requiresGrad, true);
      throwsCode(() => unsupported.backward(), "AUTOGRAD_BACKWARD_UNSUPPORTED");
      arrayEqual(input.grad.toArray(), [1, 1]);
      input.dispose();
      initial.dispose();
      unsupported.dispose();
    });

    test("rejects invalid and ragged shapes", () => {
      throwsCode(() => Tensor.zeros([-1]), "TENSOR_INVALID_SHAPE");
      throwsCode(() => Tensor.zeros(new Array(17).fill(1)), "TENSOR_INVALID_SHAPE");
      throwsCode(() => Tensor.zeros([4097, 4097]), "TENSOR_INVALID_SHAPE");
      throwsCode(() => Tensor.fromArray([[1, 2], [3]]), "TENSOR_INVALID_SHAPE");
      throwsCode(() => Tensor.fromArray([1, 2], [3]), "TENSOR_SHAPE_MISMATCH");
    });

    test("rejects non-finite values", () => {
      throwsCode(() => Tensor.scalar(Number.NaN), "TENSOR_NUMERICAL_ERROR");
      throwsCode(() => Tensor.scalar(Number.POSITIVE_INFINITY), "TENSOR_NUMERICAL_ERROR");
    });

    test("dispose is idempotent and invalidates access", () => {
      const value = Tensor.ones([1]);
      value.dispose();
      value.dispose();
      equal(value.disposed, true);
      throwsCode(() => value.toArray(), "TENSOR_DISPOSED");
    });

    log.textContent = results.join("\n");
    status.textContent = passed + " PASS / " + failed + " FAIL";
    status.className = failed === 0 ? "pass" : "fail";
  }

  global.KNA = global.KNA || {};
  global.KNA.runTensorTests = runTests;
  runTests();
})(window);
