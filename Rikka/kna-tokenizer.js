(function (global) {
  "use strict";

  const VERSION = 1;
  const BYTE_VOCAB_SIZE = 256;
  const SPECIAL_TOKENS = Object.freeze({
    PAD: 256,
    BOS: 257,
    EOS: 258,
    UNK: 259
  });
  const VOCAB_SIZE = 260;
  const TOKEN_TO_NAME = new Map(
    Object.entries(SPECIAL_TOKENS).map(([name, id]) => [id, name])
  );

  function tokenizerError(code, message) {
    const error = new Error(message);
    error.code = code;
    return error;
  }

  function validateUnicode(text) {
    for (let index = 0; index < text.length; index += 1) {
      const value = text.charCodeAt(index);
      if (value >= 0xd800 && value <= 0xdbff) {
        const next = text.charCodeAt(index + 1);
        if (!Number.isFinite(next) || next < 0xdc00 || next > 0xdfff) {
          throw tokenizerError("TOKENIZER_INVALID_UNICODE", "Text contains an unmatched high surrogate.");
        }
        index += 1;
      } else if (value >= 0xdc00 && value <= 0xdfff) {
        throw tokenizerError("TOKENIZER_INVALID_UNICODE", "Text contains an unmatched low surrogate.");
      }
    }
  }

  class ByteTokenizer {
    constructor(options = {}) {
      if (!options || typeof options !== "object" || Array.isArray(options)) {
        throw tokenizerError("TOKENIZER_INVALID_OPTIONS", "Tokenizer options must be an object.");
      }
      this.normalization = options.normalization === undefined ? "NFC" : options.normalization;
      if (this.normalization !== "NFC" && this.normalization !== "NONE") {
        throw tokenizerError("TOKENIZER_INVALID_OPTIONS", "Supported normalization values are NFC and NONE.");
      }
      this.disposed = false;
    }

    assertOpen() {
      if (this.disposed) {
        throw tokenizerError("TOKENIZER_DISPOSED", "Tokenizer has been disposed.");
      }
    }

    getVocabSize() {
      this.assertOpen();
      return VOCAB_SIZE;
    }

    getSpecialTokens() {
      this.assertOpen();
      return { ...SPECIAL_TOKENS };
    }

    getMetadata() {
      this.assertOpen();
      return {
        type: "kna.byte.v1",
        version: VERSION,
        normalization: this.normalization,
        vocabSize: VOCAB_SIZE,
        specialTokens: this.getSpecialTokens()
      };
    }

    encode(text, options = {}) {
      this.assertOpen();
      if (typeof text !== "string") {
        throw tokenizerError("TOKENIZER_INVALID_INPUT", "Text must be a string.");
      }
      if (!options || typeof options !== "object" || Array.isArray(options)) {
        throw tokenizerError("TOKENIZER_INVALID_OPTIONS", "Encode options must be an object.");
      }
      validateUnicode(text);
      const normalized = this.normalization === "NFC" ? text.normalize("NFC") : text;
      const bytes = new TextEncoder().encode(normalized);
      const tokenIds = [];
      if (options.addBos === true) tokenIds.push(SPECIAL_TOKENS.BOS);
      for (const byte of bytes) tokenIds.push(byte);
      if (options.addEos === true) tokenIds.push(SPECIAL_TOKENS.EOS);
      return tokenIds;
    }

    decode(tokenIds, options = {}) {
      this.assertOpen();
      if (!Array.isArray(tokenIds)) {
        throw tokenizerError("TOKENIZER_INVALID_INPUT", "Token IDs must be an array.");
      }
      if (!options || typeof options !== "object" || Array.isArray(options)) {
        throw tokenizerError("TOKENIZER_INVALID_OPTIONS", "Decode options must be an object.");
      }

      const skipSpecialTokens = options.skipSpecialTokens !== false;
      const replaceInvalid = options.replaceInvalid === true;
      const textDecoder = new TextDecoder("utf-8", { fatal: !replaceInvalid });
      const parts = [];
      let byteBuffer = [];
      function flushBytes() {
        if (byteBuffer.length === 0) return;
        try {
          parts.push(textDecoder.decode(Uint8Array.from(byteBuffer)));
        } catch (_) {
          if (!replaceInvalid) {
            throw tokenizerError("TOKENIZER_INVALID_UTF8", "Token IDs do not form valid UTF-8.");
          }
          parts.push(new TextDecoder("utf-8").decode(Uint8Array.from(byteBuffer)));
        }
        byteBuffer = [];
      }
      for (const id of tokenIds) {
        if (!Number.isSafeInteger(id) || id < 0 || id >= VOCAB_SIZE) {
          throw tokenizerError("TOKENIZER_INVALID_TOKEN_ID", "Token ID is outside the vocabulary.");
        }
        if (id < BYTE_VOCAB_SIZE) {
          byteBuffer.push(id);
          continue;
        }

        flushBytes();
        if (!skipSpecialTokens) parts.push("<" + TOKEN_TO_NAME.get(id).toLowerCase() + ">");
      }
      flushBytes();
      return parts.join("");
    }

    dispose() {
      this.disposed = true;
    }
  }

  global.KNA = global.KNA || {};
  global.KNA.ByteTokenizer = ByteTokenizer;
})(globalThis);
