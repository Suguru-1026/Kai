(function (root) {
  'use strict';
  const K = root.KNA || (root.KNA = {});
  const assert = K.TensorEngine.errors.assert;
  const BASE_VOCAB = 259; // byte IDs 0..255, BOS/EOS/PAD 256..258
  const utf8 = () => new TextEncoder();
  function bytesForSymbol(id, tokenBytes) { return id < 256 ? [id] : tokenBytes[id] || []; }
  function validUtf8Prefix(bytes) {
    let i=0;
    while(i<bytes.length){const a=bytes[i++];if(a<=0x7f)continue;let n,min=0x80,max=0xbf;if(a>=0xc2&&a<=0xdf)n=1;else if(a>=0xe0&&a<=0xef){n=2;if(a===0xe0)min=0xa0;if(a===0xed)max=0x9f;}else if(a>=0xf0&&a<=0xf4){n=3;if(a===0xf0)min=0x90;if(a===0xf4)max=0x8f;}else return false;for(let j=0;j<n;j++){if(i>=bytes.length)return true;const b=bytes[i++];if(b<0x80||b>0xbf||(j===0&&(b<min||b>max)))return false;}}
    return true;
  }
  class HybridJapaneseTokenizer {
    constructor(config) {
      this.type='kanami-ja-bpe-v2';this.bosId=256;this.eosId=257;this.padId=258;this.byteOffset=0;this.baseVocabSize=BASE_VOCAB;
      this.merges=[];this.tokenBytes=[];this.pairToId=new Map();
      if(config){assert(config.type===this.type&&Array.isArray(config.merges),'E_TOKENIZER_CONFIG','Unsupported BPE tokenizer config.');this.merges=config.merges.map((m,i)=>{assert(Array.isArray(m)&&m.length===3&&m[2]===BASE_VOCAB+i&&m[0]>=0&&m[1]>=0&&m[0]<BASE_VOCAB+i&&m[1]<BASE_VOCAB+i,'E_TOKENIZER_CONFIG','Invalid BPE merge entry.',{index:i});return m.slice();});}this._rebuild();
    }
    _rebuild(){this.tokenBytes=[];for(let i=0;i<256;i++)this.tokenBytes[i]=[i];this.tokenBytes[this.bosId]=[];this.tokenBytes[this.eosId]=[];this.tokenBytes[this.padId]=[];this.pairToId.clear();for(const [a,b,id] of this.merges){this.tokenBytes[id]=this.tokenBytes[a].concat(this.tokenBytes[b]);this.pairToId.set(a+','+b,id);}this.vocabSize=BASE_VOCAB+this.merges.length;}
    static async train(texts, options={}) {
      assert(Array.isArray(texts)&&texts.length>0,'E_TOKENIZER_CORPUS','Tokenizer training requires non-empty text corpus.');
      const target=Math.max(BASE_VOCAB,Math.min(4096,Number.isSafeInteger(options.vocabSize)?options.vocabSize:512));
      const corpus=texts.filter(x=>typeof x==='string').map(s=>Array.from(utf8().encode(s)));
      assert(corpus.some(x=>x.length),'E_TOKENIZER_CORPUS','Corpus contains no UTF-8 bytes.');
      const tok=new HybridJapaneseTokenizer();const maxMerges=target-BASE_VOCAB;
      for(let step=0;step<maxMerges;step++){
        const counts=new Map();
        for(const seq of corpus)for(let i=0;i+1<seq.length;i++){const key=seq[i]+','+seq[i+1];counts.set(key,(counts.get(key)||0)+1);}
        let best=null,bestCount=1;for(const [key,count] of counts){if(count>bestCount){best=key;bestCount=count;}}
        if(!best)break;
        const [a,b]=best.split(',').map(Number),id=BASE_VOCAB+tok.merges.length;tok.merges.push([a,b,id]);tok._rebuild();
        for(let s=0;s<corpus.length;s++){const old=corpus[s],next=[];for(let i=0;i<old.length;){if(i+1<old.length&&old[i]===a&&old[i+1]===b){next.push(id);i+=2;}else next.push(old[i++]);}corpus[s]=next;}
        if(options.onProgress&&(step%16===15||step===maxMerges-1))options.onProgress({merges:tok.merges.length,target:maxMerges});
        if(step%16===15)await new Promise(resolve=>setTimeout(resolve,0));
      }
      tok._rebuild();return tok;
    }
    encode(text,options={}){assert(typeof text==='string','E_TOKENIZER_INPUT','Tokenizer input must be a string.');let ids=Array.from(utf8().encode(text));for(const [a,b,id] of this.merges){const next=[];for(let i=0;i<ids.length;){if(i+1<ids.length&&ids[i]===a&&ids[i+1]===b){next.push(id);i+=2;}else next.push(ids[i++]);}ids=next;}if(options.addBos!==false)ids.unshift(this.bosId);if(options.addEos===true)ids.push(this.eosId);return ids;}
    decode(ids,options={}){assert(Array.isArray(ids)||ArrayBuffer.isView(ids),'E_TOKENIZER_IDS','Token IDs must be an array.');const out=[];let bytes=[];const flush=()=>{if(bytes.length){out.push(new TextDecoder('utf-8',{fatal:false}).decode(Uint8Array.from(bytes)));bytes=[];}};for(const id of ids){assert(Number.isSafeInteger(id)&&id>=0&&id<this.vocabSize,'E_TOKEN_ID','Token ID outside vocabulary.',{id});if(id<=255)bytes.push(id);else if(id>=BASE_VOCAB)bytes.push(...this.tokenBytes[id]);else{flush();if(options.includeSpecialTokens)out.push(id===256?'<BOS>':id===257?'<EOS>':'<PAD>');}}flush();return out.join('');}
    isTextToken(id){return Number.isSafeInteger(id)&&id>=0&&id<this.vocabSize&&id!==this.bosId&&id!==this.eosId&&id!==this.padId;}
    _bytesForIds(ids){const out=[];for(const id of ids){if(id>=0&&id<=255)out.push(id);else if(id>=BASE_VOCAB&&id<this.vocabSize)out.push(...this.tokenBytes[id]);}return out;}
    isValidUtf8(ids){const bytes=this._bytesForIds(ids);try{new TextDecoder('utf-8',{fatal:true}).decode(Uint8Array.from(bytes));return true;}catch(_){return false;}}
    allowedNextIds(prefixIds=[]){const prefix=this._bytesForIds(prefixIds),out=[],control=/[\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F-\u009F]/;for(let id=0;id<this.vocabSize;id++){if(id===this.bosId||id===this.padId)continue;if(id===this.eosId){if(this.isValidUtf8(prefixIds))out.push(id);continue;}const b=prefix.concat(this._bytesForIds([id]));if(!validUtf8Prefix(b))continue;let bad=false;for(const byte of b){if((byte<32&&byte!==9&&byte!==10&&byte!==13)||byte===127){bad=true;break;}}if(bad)continue;const decoded=new TextDecoder('utf-8',{fatal:false}).decode(Uint8Array.from(b));if(control.test(decoded))continue;out.push(id);bad=false;}return out;}
    saveConfig(){return {type:this.type,vocabSize:this.vocabSize,bosId:this.bosId,eosId:this.eosId,padId:this.padId,baseVocabSize:BASE_VOCAB,merges:this.merges.map(m=>m.slice())};}
    static fromConfig(config){const t=new HybridJapaneseTokenizer(config);assert(t.vocabSize===config.vocabSize&&config.bosId===256&&config.eosId===257&&config.padId===258,'E_TOKENIZER_CONFIG','Tokenizer config vocabulary metadata mismatch.');return t;}
  }
  K.Tokenizer=Object.freeze({version:'2.0.0',ByteTokenizer:HybridJapaneseTokenizer,HybridJapaneseTokenizer,train:HybridJapaneseTokenizer.train,baseVocabSize:BASE_VOCAB});
})(typeof globalThis!=='undefined'?globalThis:this);
