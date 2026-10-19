(function (root) {
  'use strict';
  const K = root.KNA || (root.KNA = {});
  const assert = K.TensorEngine.errors.assert;
  class ChatSession {
    constructor(tokenizer, options = {}) {
      assert(tokenizer && typeof tokenizer.encode === 'function', 'E_SESSION_TOKENIZER', 'Tokenizer required.');
      this.tokenizer = tokenizer;
      this.maxTurns = options.maxTurns === undefined ? 8 : options.maxTurns;
      this.maxPromptTokens = options.maxPromptTokens === undefined ? 90 : options.maxPromptTokens;
      assert(Number.isSafeInteger(this.maxTurns) && this.maxTurns > 0 && Number.isSafeInteger(this.maxPromptTokens) && this.maxPromptTokens > 4, 'E_SESSION_CONFIG', 'Invalid session limits.');
      this.turns = [];
    }
    addUser(text) { assert(typeof text === 'string' && text.trim().length > 0, 'E_SESSION_TEXT', 'User text must not be empty.'); this.turns.push({ role: 'user', content: text }); this._trimTurns(); }
    addAssistant(text) { assert(typeof text === 'string', 'E_SESSION_TEXT', 'Assistant text must be a string.'); this.turns.push({ role: 'assistant', content: text }); this._trimTurns(); }
    _trimTurns() {
      while (this.turns.filter(t => t.role === 'user').length > this.maxTurns) this.turns.splice(0, 2);
    }
    buildPrompt() {
      const format = turns => turns.map(t => t.role === 'user' ? '<|user|>\n' + t.content + '\n<|assistant|>\n' : t.content + '\n').join('');
      let turns = this.turns.slice();
      while (turns.length > 1 && this.tokenizer.encode(format(turns), { addBos: true }).length > this.maxPromptTokens) {
        // Drop the oldest complete user/assistant pair, never the current user message.
        if (turns.length >= 3 && turns[0].role === 'user' && turns[1].role === 'assistant') turns.splice(0, 2);
        else break;
      }
      if (turns.length && turns[turns.length - 1].role === 'user') return format(turns);
      return format(turns) + '<|assistant|>\n';
    }
    clear() { this.turns.length = 0; }
    export() { return { format: 'kanami-chat-session-v1', turns: this.turns.map(t => ({ role: t.role, content: t.content })) }; }
    import(data) {
      assert(data && data.format === 'kanami-chat-session-v1' && Array.isArray(data.turns) && data.turns.length <= this.maxTurns * 2 + 1, 'E_SESSION_STATE', 'Invalid session state.');
      const turns = data.turns.map((t, i) => {
        assert(t && (t.role === 'user' || t.role === 'assistant') && typeof t.content === 'string' && t.content.length <= 12000, 'E_SESSION_TURN', 'Invalid turn.', { index: i });
        return { role: t.role, content: t.content };
      });
      for (let i = 0; i < turns.length; i++) assert(turns[i].role === (i % 2 === 0 ? 'user' : 'assistant'), 'E_SESSION_ORDER', 'Turns must alternate starting with user.');
      this.turns = turns; this._trimTurns(); return this.turns.length;
    }
  }
  K.ChatSession = Object.freeze({ version: '1.0.0', ChatSession });
})(typeof globalThis !== 'undefined' ? globalThis : this);
