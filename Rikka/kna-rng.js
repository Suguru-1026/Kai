(function (global) {
  "use strict";

  const MASK_64 = (1n << 64n) - 1n;
  const MASK_32 = 0xffffffff;
  const MULTIPLIER = 6364136223846793005n;
  const FNV_OFFSET = 14695981039346656037n;
  const FNV_PRIME = 1099511628211n;
  const ALGORITHM = "pcg32-kna-v1";

  function rngError(code, message) {
    const error = new Error(message);
    error.code = code;
    return error;
  }

  function parseUint64(value, name) {
    let parsed;
    try {
      if (typeof value === "bigint") {
        parsed = value;
      } else if (typeof value === "number" && Number.isSafeInteger(value)) {
        parsed = BigInt(value);
      } else if (typeof value === "string" && /^\d+$/.test(value)) {
        parsed = BigInt(value);
      } else {
        throw new Error("invalid");
      }
    } catch (_) {
      throw rngError("RNG_INVALID_ARGUMENT", name + " must be an exact unsigned 64-bit integer.");
    }

    if (parsed < 0n || parsed > MASK_64) {
      throw rngError("RNG_INVALID_ARGUMENT", name + " is outside the unsigned 64-bit range.");
    }
    return parsed;
  }

  function hashStreamId(streamId) {
    if (typeof streamId !== "string" || streamId.length === 0) {
      throw rngError("RNG_INVALID_ARGUMENT", "streamId must be a non-empty string.");
    }

    const bytes = new TextEncoder().encode(streamId);
    let hash = FNV_OFFSET;
    for (const byte of bytes) {
      hash = ((hash ^ BigInt(byte)) * FNV_PRIME) & MASK_64;
    }
    return hash;
  }

  class PCG32 {
    constructor(seed, sequence) {
      this.seed = parseUint64(seed, "seed");
      this.sequence = parseUint64(sequence, "sequence");
      this.increment = ((this.sequence << 1n) | 1n) & MASK_64;
      this.state = 0n;
      this.hasNormal = false;
      this.normalValue = 0;
      this.reset(this.seed);
    }

    reset(seed = this.seed) {
      this.seed = parseUint64(seed, "seed");
      this.state = 0n;
      this.hasNormal = false;
      this.normalValue = 0;
      this.nextUint32();
      this.state = (this.state + this.seed) & MASK_64;
      this.nextUint32();
    }

    nextUint32() {
      const oldState = this.state;
      this.state = (oldState * MULTIPLIER + this.increment) & MASK_64;
      const shifted = Number((((oldState >> 18n) ^ oldState) >> 27n) & 0xffffffffn);
      const rotation = Number(oldState >> 59n);
      return ((shifted >>> rotation) | (shifted << ((-rotation) & 31))) >>> 0;
    }

    nextFloat() {
      return Math.fround((this.nextUint32() >>> 8) / 16777216);
    }

    nextInt(min, max) {
      if (!Number.isSafeInteger(min) || !Number.isSafeInteger(max) || min > max) {
        throw rngError("RNG_INVALID_ARGUMENT", "Integer bounds must be ordered safe integers.");
      }
      const range = max - min + 1;
      const fullRange = 4294967296;
      if (!Number.isSafeInteger(range) || range <= 0 || range > fullRange) {
        throw rngError("RNG_INVALID_ARGUMENT", "Integer range must not exceed 32 bits.");
      }

      const threshold = fullRange % range;
      let value;
      do {
        value = this.nextUint32();
      } while (value < threshold);
      return min + (value % range);
    }

    normal(mean = 0, standardDeviation = 1) {
      if (!Number.isFinite(mean) || !Number.isFinite(standardDeviation) || standardDeviation < 0) {
        throw rngError("RNG_INVALID_ARGUMENT", "Normal distribution parameters are invalid.");
      }
      if (this.hasNormal) {
        this.hasNormal = false;
        return mean + standardDeviation * this.normalValue;
      }

      const first = 1 - this.nextFloat();
      const second = this.nextFloat();
      const radius = Math.sqrt(-2 * Math.log(first));
      const angle = 2 * Math.PI * second;
      this.normalValue = radius * Math.sin(angle);
      this.hasNormal = true;
      return mean + standardDeviation * radius * Math.cos(angle);
    }

    getState() {
      return {
        algorithm: ALGORITHM,
        seed: this.seed.toString(),
        sequence: this.sequence.toString(),
        state: this.state.toString(),
        increment: this.increment.toString(),
        hasNormal: this.hasNormal,
        normalValue: this.normalValue
      };
    }

    setState(value) {
      if (!value || value.algorithm !== ALGORITHM || typeof value.hasNormal !== "boolean" ||
        !Number.isFinite(value.normalValue)) {
        throw rngError("RNG_INVALID_STATE", "RNG state metadata is invalid.");
      }

      const seed = parseUint64(value.seed, "state.seed");
      const sequence = parseUint64(value.sequence, "state.sequence");
      const state = parseUint64(value.state, "state.state");
      const increment = parseUint64(value.increment, "state.increment");
      const expectedIncrement = ((sequence << 1n) | 1n) & MASK_64;
      if (increment !== expectedIncrement || (increment & 1n) === 0n) {
        throw rngError("RNG_INVALID_STATE", "RNG increment does not match its sequence.");
      }

      this.seed = seed;
      this.sequence = sequence;
      this.increment = increment;
      this.state = state;
      this.hasNormal = value.hasNormal;
      this.normalValue = value.normalValue;
    }

    fork(streamId) {
      return createRng(this.seed, streamId);
    }
  }

  function createRng(seed, streamId = "default") {
    return new PCG32(seed, hashStreamId(streamId));
  }

  global.KNA = global.KNA || {};
  global.KNA.RNG = Object.freeze({
    algorithm: ALGORITHM,
    PCG32,
    create: createRng
  });
})(globalThis);
