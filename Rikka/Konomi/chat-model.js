(function (root) {
  'use strict';
  const K = root.KNA || (root.KNA = {}), TE = K.TensorEngine, AG = K.Autograd, LL = K.LayerLibrary;
  if (!TE || !AG || !LL) throw new Error('ChatModel requires TensorEngine, Autograd, and LayerLibrary.');
  const { Tensor, ops, errors } = TE, assert = errors.assert, { add, mul, matmul } = AG.ops;

  function normalValues(length, seed, std) {
    let state = (seed >>> 0) || 0x6d2b79f5;
    const uniform = () => { state ^= state << 13; state ^= state >>> 17; state ^= state << 5; return ((state >>> 0) + 1) / 4294967297; };
    const out = new Float32Array(length);
    for (let i=0;i<length;i+=2) { const r=Math.sqrt(-2*Math.log(uniform())),a=2*Math.PI*uniform();out[i]=r*Math.cos(a)*std;if(i+1<length)out[i+1]=r*Math.sin(a)*std; }
    return out;
  }
  function rotateRows(data, seq, width, heads, theta, inverse) {
    const headDim = width / heads, out = new Float32Array(data.length);
    out.set(data);
    for (let p = 0; p < seq; p++) for (let h = 0; h < heads; h++) {
      const base = p * width + h * headDim;
      for (let i = 0; i + 1 < headDim; i += 2) {
        const angle = p / Math.pow(theta, i / headDim) * (inverse ? -1 : 1);
        const c = Math.cos(angle), s = Math.sin(angle), x = data[base + i], y = data[base + i + 1];
        out[base + i] = x * c - y * s;
        out[base + i + 1] = x * s + y * c;
      }
    }
    return out;
  }

  class CausalMultiHeadAttention extends LL.Module {
    constructor(width, heads, options = {}) {
      super();
      assert(Number.isInteger(width) && width > 0 && Number.isInteger(heads) && heads > 0 && width % heads === 0, 'E_CHAT_HEADS', 'width must be divisible by positive numHeads.');
      assert((width / heads) % 2 === 0, 'E_CHAT_ROPE_DIM', 'RoPE requires an even head dimension.');
      this.width = width; this.numHeads = heads; this.headDim = width / heads; this.ropeTheta = options.ropeTheta || 10000;
      const seed = options.seed ?? 41;
      this.q = new LL.Linear(width, width, { seed }); this.k = new LL.Linear(width, width, { seed: seed + 1 });
      this.v = new LL.Linear(width, width, { seed: seed + 2 }); this.o = new LL.Linear(width, width, { seed: seed + 3 });
      this._parameters = [...this.q.parameters(), ...this.k.parameters(), ...this.v.parameters(), ...this.o.parameters()];
    }
    forward(input) {
      const x = AG.asValue(input); assert(x.shape.length === 2 && x.shape[1] === this.width, 'E_CHAT_ATTENTION_SHAPE', 'Attention expects [sequence,width].');
      const seq = x.shape[0]; assert(seq > 0, 'E_CHAT_EMPTY_SEQUENCE', 'Sequence must not be empty.');
      const qv = this.q.forward(x), kv = this.k.forward(x), vv = this.v.forward(x);
      const q = rotateRows(qv.data, seq, this.width, this.numHeads, this.ropeTheta, false);
      const k = rotateRows(kv.data, seq, this.width, this.numHeads, this.ropeTheta, false);
      const v = vv.data, hd = this.headDim, heads = this.numHeads, w = this.width, scale = 1 / Math.sqrt(hd);
      const probs = new Float32Array(heads * seq * seq), context = new Float32Array(seq * w);
      for (let h = 0; h < heads; h++) for (let i = 0; i < seq; i++) {
        let max = -Infinity;
        for (let j = 0; j <= i; j++) { let dot = 0; for (let z = 0; z < hd; z++) dot += q[i*w+h*hd+z] * k[j*w+h*hd+z]; max = Math.max(max, dot * scale); }
        let den = 0;
        for (let j = 0; j <= i; j++) { let dot = 0; for (let z = 0; z < hd; z++) dot += q[i*w+h*hd+z] * k[j*w+h*hd+z]; const p = Math.exp(dot * scale - max); probs[(h*seq+i)*seq+j] = p; den += p; }
        assert(Number.isFinite(den) && den > 0, 'E_CHAT_ATTENTION_NUMERIC', 'Attention softmax failed.');
        for (let j = 0; j <= i; j++) { const ix = (h*seq+i)*seq+j; probs[ix] /= den; for (let z = 0; z < hd; z++) context[i*w+h*hd+z] += probs[ix] * v[j*w+h*hd+z]; }
      }
      const cv = new AG.Value(Tensor.from(context, { shape: [seq, w] }), { requiresGrad: qv.requiresGrad || kv.requiresGrad || vv.requiresGrad, label: 'CausalMultiHeadAttention.context' });
      cv.parents = [qv, kv, vv];
      cv.backwardFn = () => {
        if (!cv.grad) return;
        const gqRot = new Float32Array(seq*w), gkRot = new Float32Array(seq*w), gv = new Float32Array(seq*w), gp = new Float32Array(heads*seq*seq), gs = new Float32Array(heads*seq*seq);
        for (let h=0;h<heads;h++) for(let i=0;i<seq;i++) for(let j=0;j<=i;j++) {
          const pi=(h*seq+i)*seq+j;
          for(let z=0;z<hd;z++){const col=h*hd+z;gv[j*w+col]+=probs[pi]*cv.grad.data[i*w+col];gp[pi]+=cv.grad.data[i*w+col]*v[j*w+col];}
        }
        for(let h=0;h<heads;h++) for(let i=0;i<seq;i++) {
          let dot=0;for(let j=0;j<=i;j++){const pi=(h*seq+i)*seq+j;dot+=gp[pi]*probs[pi];}
          for(let j=0;j<=i;j++){const pi=(h*seq+i)*seq+j;gs[pi]=probs[pi]*(gp[pi]-dot);for(let z=0;z<hd;z++){const qi=i*w+h*hd+z,kj=j*w+h*hd+z;gqRot[qi]+=gs[pi]*k[kj]*scale;gkRot[kj]+=gs[pi]*q[qi]*scale;}}
        }
        if(qv.requiresGrad) AG.accumulate(qv, Tensor.from(rotateRows(gqRot,seq,w,heads,this.ropeTheta,true),{shape:[seq,w]}));
        if(kv.requiresGrad) AG.accumulate(kv, Tensor.from(rotateRows(gkRot,seq,w,heads,this.ropeTheta,true),{shape:[seq,w]}));
        if(vv.requiresGrad) AG.accumulate(vv, Tensor.from(gv,{shape:[seq,w]}));
      };
      return this.o.forward(cv);
    }
    // Inference-only one-token path. Cache stores already RoPE-rotated keys and values.
    forwardCached(vector, cache) {
      const w=this.width, hd=this.headDim, heads=this.numHeads, pos=cache.keys.length;
      const linear=(layer,x)=>{const out=new Float32Array(layer.outFeatures);for(let j=0;j<layer.outFeatures;j++){let v=layer.bias?layer.bias.data[j]:0;for(let i=0;i<layer.inFeatures;i++)v+=x[i]*layer.weight.data[i*layer.outFeatures+j];out[j]=v;}return out;};
      const q=linear(this.q,vector), k=linear(this.k,vector), v=linear(this.v,vector);
      for(let h=0;h<heads;h++)for(let z=0;z+1<hd;z+=2){const a=pos/Math.pow(this.ropeTheta,z/hd),c=Math.cos(a),sn=Math.sin(a),b=h*hd+z,x=k[b],y=k[b+1],qx=q[b],qy=q[b+1];k[b]=x*c-y*sn;k[b+1]=x*sn+y*c;q[b]=qx*c-qy*sn;q[b+1]=qx*sn+qy*c;}
      cache.keys.push(k);cache.values.push(v);
      const ctx=new Float32Array(w),scale=1/Math.sqrt(hd),n=cache.keys.length;
      for(let h=0;h<heads;h++){let max=-Infinity;const scores=new Float32Array(n);for(let j=0;j<n;j++){let dot=0;for(let z=0;z<hd;z++)dot+=q[h*hd+z]*cache.keys[j][h*hd+z];scores[j]=dot*scale;if(scores[j]>max)max=scores[j];}let den=0;for(let j=0;j<n;j++){scores[j]=Math.exp(scores[j]-max);den+=scores[j];}for(let j=0;j<n;j++){const prob=scores[j]/den;for(let z=0;z<hd;z++)ctx[h*hd+z]+=prob*cache.values[j][h*hd+z];}}
      return linear(this.o,ctx);
    }
  }

  class SwiGLU extends LL.Module {
    constructor(width, hidden, options = {}) {
      super(); const seed = options.seed ?? 61;
      this.gate = new LL.Linear(width, hidden, { seed }); this.up = new LL.Linear(width, hidden, { seed: seed + 1 }); this.down = new LL.Linear(hidden, width, { seed: seed + 2 });
      this._parameters = [...this.gate.parameters(), ...this.up.parameters(), ...this.down.parameters()];
    }
    forward(input) {
      const x = AG.asValue(input), gate = this.gate.forward(x), up = this.up.forward(x), n = gate.size, outData = new Float32Array(n), sig = new Float32Array(n), silu = new Float32Array(n);
      for (let i=0;i<n;i++){sig[i]=1/(1+Math.exp(-gate.data[i]));silu[i]=gate.data[i]*sig[i];outData[i]=silu[i]*up.data[i];}
      const mid = new AG.Value(Tensor.from(outData,{shape:gate.shape}),{requiresGrad:gate.requiresGrad||up.requiresGrad,label:'SwiGLU.activation'});mid.parents=[gate,up];
      mid.backwardFn=()=>{if(!mid.grad)return;const gg=new Float32Array(n),gu=new Float32Array(n);for(let i=0;i<n;i++){gg[i]=mid.grad.data[i]*up.data[i]*(sig[i]+gate.data[i]*sig[i]*(1-sig[i]));gu[i]=mid.grad.data[i]*silu[i];}if(gate.requiresGrad)AG.accumulate(gate,Tensor.from(gg,{shape:gate.shape}));if(up.requiresGrad)AG.accumulate(up,Tensor.from(gu,{shape:up.shape}));};
      return this.down.forward(mid);
    }
  }

  class DecoderBlock extends LL.Module {
    constructor(config, index) {
      super(); const seed = config.seed + index * 100;
      this.norm1 = new LL.RMSNorm(config.hiddenSize, config.normEpsilon);
      this.attention = new CausalMultiHeadAttention(config.hiddenSize, config.numHeads, { seed: seed + 1, ropeTheta: config.ropeTheta });
      this.norm2 = new LL.RMSNorm(config.hiddenSize, config.normEpsilon);
      this.ffn = new SwiGLU(config.hiddenSize, config.intermediateSize, { seed: seed + 10 });
      this._parameters = [...this.norm1.parameters(), ...this.attention.parameters(), ...this.norm2.parameters(), ...this.ffn.parameters()];
    }
    forward(x) { const a = this.attention.forward(this.norm1.forward(x)); x = add(x,a); const f = this.ffn.forward(this.norm2.forward(x)); return add(x,f); }
  }

  class ChatModel {
    constructor(options = {}) {
      this.config = Object.assign({ vocabSize:512, maxSequenceLength:192, hiddenSize:16, numLayers:1, numHeads:2, intermediateSize:32, normEpsilon:1e-5, ropeTheta:10000, tieWordEmbeddings:true, maxGradientNorm:1.0, seed:1234 }, options);
      const c = this.config;
      for (const k of ['vocabSize','maxSequenceLength','hiddenSize','numLayers','numHeads','intermediateSize']) assert(Number.isSafeInteger(c[k])&&c[k]>0,'E_CHAT_CONFIG','Invalid positive integer config: '+k);
      assert(c.hiddenSize % c.numHeads === 0 && (c.hiddenSize/c.numHeads)%2===0,'E_CHAT_CONFIG','hiddenSize/numHeads must be an even integer.');
      assert(c.vocabSize>=259 && c.vocabSize<=4096,'E_CHAT_TOKENIZER','vocabSize must fit the configured tokenizer.');
      assert(c.maxSequenceLength<=512,'E_CHAT_RESOURCE_LIMIT','maxSequenceLength is limited to 512 in the CPU prototype.');
      assert(Number.isFinite(c.maxGradientNorm)&&c.maxGradientNorm>0,'E_CHAT_GRAD_CLIP','maxGradientNorm must be positive and finite.');
      this.embedding = new LL.Embedding(c.vocabSize,c.hiddenSize,{seed:c.seed});
      this.embedding.weight.tensor.data.set(normalValues(c.vocabSize*c.hiddenSize,c.seed+7,0.02));
      this.blocks = Array.from({length:c.numLayers},(_,i)=>new DecoderBlock(c,i));
      this.finalNorm = new LL.RMSNorm(c.hiddenSize,c.normEpsilon);
      this.lmHead = c.tieWordEmbeddings ? null : new LL.Linear(c.hiddenSize,c.vocabSize,{seed:c.seed+999});
      this._parameters = [...this.embedding.parameters(),...this.blocks.flatMap(b=>b.parameters()),...this.finalNorm.parameters(),...(this.lmHead?this.lmHead.parameters():[])];
      this._disposed = false;
    }
    parameters(){return this._parameters.slice();}
    train(mode=true){this.blocks.forEach(b=>b.train(mode));this.training=!!mode;return this;}
    eval(){return this.train(false);}
    forward(tokenIds){
      assert(!this._disposed,'E_CHAT_DISPOSED','Model has been disposed.');
      assert(Array.isArray(tokenIds)&&tokenIds.length>0&&tokenIds.length<=this.config.maxSequenceLength,'E_CHAT_SEQUENCE','Token sequence length is invalid.');
      const ids=tokenIds.map((id,i)=>{assert(Number.isSafeInteger(id)&&id>=0&&id<this.config.vocabSize,'E_TOKEN_ID','Invalid token ID.',{index:i,id});return id;});
      let x=this.embedding.forward(ids);for(const block of this.blocks)x=block.forward(x);x=this.finalNorm.forward(x);
      if(this.lmHead)return this.lmHead.forward(x);
      const w=this.embedding.weight, logits=ops.matmul(x.tensor,ops.transpose2D(w.tensor));
      const out=new AG.Value(logits,{requiresGrad:x.requiresGrad||w.requiresGrad,label:'TiedLMHead.logits'});out.parents=[x,w];
      out.backwardFn=()=>{if(!out.grad)return;if(x.requiresGrad)AG.accumulate(x,ops.matmul(out.grad,w.tensor));if(w.requiresGrad)AG.accumulate(w,ops.matmul(ops.transpose2D(out.grad),x.tensor));};
      return out;
    }
    loss(tokenIds){
      assert(Array.isArray(tokenIds)&&tokenIds.length>=2,'E_CHAT_TRAIN_SEQUENCE','Training sequence must contain at least two tokens.');
      const input=tokenIds.slice(0,-1),targets=tokenIds.slice(1),logits=this.forward(input),result=K.TrainingSystem.crossEntropy(logits.tensor,targets);
      return {loss:result.value,gradient:result.gradient,logits,targets};
    }
    trainStep(tokenIds, optimizer){
      assert(optimizer&&typeof optimizer.zeroGrad==='function'&&typeof optimizer.step==='function','E_CHAT_OPTIMIZER','Compatible optimizer required.');
      optimizer.zeroGrad();const result=this.loss(tokenIds);result.logits.backward(result.gradient);
      let norm2=0;
      for(const p of this._parameters) if(p.grad) { assert(TE.sameShape(p.grad.shape,p.shape)&&ops.finite(p.grad),'E_CHAT_GRAD_NONFINITE','Invalid gradient before clipping.'); for(const v of p.grad.data) norm2+=v*v; }
      const gradientNorm=Math.sqrt(norm2),scale=gradientNorm>this.config.maxGradientNorm?this.config.maxGradientNorm/(gradientNorm||1):1;
      if(scale<1)for(const p of this._parameters)if(p.grad)for(let i=0;i<p.grad.size;i++)p.grad.data[i]*=scale;
      const updated=optimizer.step();
      return {loss:result.loss,updatedParameters:updated,gradientNorm,gradientScale:scale};
    }
    trainChatExample(prompt, response, tokenizer, optimizer) {
      assert(typeof prompt === 'string' && typeof response === 'string', 'E_CHAT_EXAMPLE', 'prompt and response must be strings.');
      assert(tokenizer && typeof tokenizer.encode === 'function', 'E_CHAT_TOKENIZER', 'Tokenizer required.');
      assert(optimizer && typeof optimizer.zeroGrad === 'function' && typeof optimizer.step === 'function', 'E_CHAT_OPTIMIZER', 'Compatible optimizer required.');
      const prefix = tokenizer.encode(prompt, { addBos: true, addEos: false });
      const answer = tokenizer.encode(response, { addBos: false, addEos: false });
      const ids = prefix.concat(answer, [tokenizer.eosId]);
      assert(answer.length > 0, 'E_CHAT_EXAMPLE_EMPTY', 'Assistant response must not be empty.');
      assert(ids.length - 1 <= this.config.maxSequenceLength, 'E_CHAT_SEQUENCE', 'Training example exceeds maxSequenceLength.');
      const input = ids.slice(0, -1), targets = ids.slice(1);
      const logits = this.forward(input), vocab = this.config.vocabSize, rows = [], selectedTargets = [];
      // Train only on assistant response bytes and EOS, not on the user prompt.
      for (let targetIndex = prefix.length; targetIndex < ids.length; targetIndex++) {
        rows.push(logits.data.slice((targetIndex - 1) * vocab, targetIndex * vocab));
        selectedTargets.push(targets[targetIndex - 1]);
      }
      const selected = Tensor.from(rows, { shape: [rows.length, vocab] });
      const loss = K.TrainingSystem.crossEntropy(selected, selectedTargets);
      const fullGrad = new Float32Array(logits.size);
      for (let r = 0; r < rows.length; r++) fullGrad.set(loss.gradient.data.slice(r * vocab, (r + 1) * vocab), (prefix.length + r - 1) * vocab);
      optimizer.zeroGrad(); logits.backward(Tensor.from(fullGrad, { shape: logits.shape }));
      let norm2 = 0;
      for (const param of this._parameters) if (param.grad) {
        assert(TE.sameShape(param.grad.shape, param.shape) && ops.finite(param.grad), 'E_CHAT_GRAD_NONFINITE', 'Invalid gradient before clipping.');
        for (const v of param.grad.data) norm2 += v * v;
      }
      const gradientNorm = Math.sqrt(norm2), scale = gradientNorm > this.config.maxGradientNorm ? this.config.maxGradientNorm / (gradientNorm || 1) : 1;
      if (scale < 1) for (const param of this._parameters) if (param.grad) for (let i = 0; i < param.grad.size; i++) param.grad.data[i] *= scale;
      const updatedParameters = optimizer.step();
      return { loss: loss.value, updatedParameters, gradientNorm, gradientScale: scale, trainedTokens: selectedTargets.length, lossScope: 'assistant-response-only' };
    }
    createInferenceCache(){return {layers:this.blocks.map(()=>({keys:[],values:[]})),length:0,version:'kanami-kv-cache-v1'};}
    inferenceStep(tokenId, cache){
      assert(!this._disposed,'E_CHAT_DISPOSED','Model has been disposed.');
      assert(cache&&cache.version==='kanami-kv-cache-v1'&&Array.isArray(cache.layers)&&cache.layers.length===this.blocks.length,'E_KV_CACHE','Invalid KV cache.');
      assert(Number.isSafeInteger(tokenId)&&tokenId>=0&&tokenId<this.config.vocabSize,'E_TOKEN_ID','Invalid inference token ID.');
      const w=this.config.hiddenSize;let x=new Float32Array(w);for(let j=0;j<w;j++)x[j]=this.embedding.weight.data[tokenId*w+j];
      const norm=(vec,layer)=>{let ss=0;for(const v of vec)ss+=v*v;const inv=1/Math.sqrt(ss/vec.length+layer.epsilon),out=new Float32Array(vec.length);for(let j=0;j<vec.length;j++)out[j]=vec[j]*inv*layer.scale.data[j];return out;};
      const linear=(layer,vec)=>{const out=new Float32Array(layer.outFeatures);for(let j=0;j<layer.outFeatures;j++){let v=layer.bias?layer.bias.data[j]:0;for(let i=0;i<layer.inFeatures;i++)v+=vec[i]*layer.weight.data[i*layer.outFeatures+j];out[j]=v;}return out;};
      for(let i=0;i<this.blocks.length;i++){const b=this.blocks[i],a=b.attention.forwardCached(norm(x,b.norm1),cache.layers[i]);for(let j=0;j<w;j++)x[j]+=a[j];const n=norm(x,b.norm2),g=linear(b.ffn.gate,n),u=linear(b.ffn.up,n),mid=new Float32Array(g.length);for(let j=0;j<g.length;j++)mid[j]=(g[j]/(1+Math.exp(-g[j])))*u[j];const f=linear(b.ffn.down,mid);for(let j=0;j<w;j++)x[j]+=f[j];}
      x=norm(x,this.finalNorm);const logits=new Float32Array(this.config.vocabSize),weight=this.lmHead?null:this.embedding.weight.data;
      if(this.lmHead){const out=linear(this.lmHead,x);logits.set(out);}else{for(let id=0;id<this.config.vocabSize;id++){let sum=0;for(let j=0;j<w;j++)sum+=x[j]*weight[id*w+j];logits[id]=sum;}}
      cache.length++;return Tensor.from(logits,{shape:[this.config.vocabSize]});
    }
    prepareChatExample(ex, tokenizer) {
      assert(ex&&typeof ex.prompt==='string'&&typeof ex.response==='string'&&ex.response.length>0,'E_CHAT_EXAMPLE','Invalid chat example.');
      const prefix=tokenizer.encode(ex.prompt,{addBos:true,addEos:false}),answer=tokenizer.encode(ex.response,{addBos:false,addEos:false}),ids=prefix.concat(answer,[tokenizer.eosId]);
      assert(answer.length>0&&ids.length-1<=this.config.maxSequenceLength,'E_CHAT_SEQUENCE','A chat example exceeds maxSequenceLength.',{length:ids.length-1});
      return {_kanamiEncoded:true,prefix,answer,ids,prompt:ex.prompt,response:ex.response};
    }
    trainChatBatch(examples, tokenizer, optimizer) {
      assert(Array.isArray(examples)&&examples.length>0,'E_CHAT_BATCH','A non-empty example batch is required.');
      assert(optimizer&&typeof optimizer.zeroGrad==='function'&&typeof optimizer.step==='function','E_CHAT_OPTIMIZER','Compatible optimizer required.');
      const prepared=examples.map((ex,index)=>{try{return ex&&ex._kanamiEncoded?ex:this.prepareChatExample(ex,tokenizer);}catch(e){e.details=Object.assign({},e.details,{index});throw e;}});
      optimizer.zeroGrad();let lossSum=0,trainedTokens=0;
      for(const ex of prepared){const input=ex.ids.slice(0,-1),targets=ex.ids.slice(1),logits=this.forward(input),vocab=this.config.vocabSize,rows=[],selectedTargets=[];for(let ti=ex.prefix.length;ti<ex.ids.length;ti++){rows.push(logits.data.slice((ti-1)*vocab,ti*vocab));selectedTargets.push(targets[ti-1]);}const selected=Tensor.from(rows,{shape:[rows.length,vocab]}),loss=K.TrainingSystem.crossEntropy(selected,selectedTargets),fullGrad=new Float32Array(logits.size),scale=1/prepared.length;for(let r=0;r<rows.length;r++){const src=loss.gradient.data.subarray(r*vocab,(r+1)*vocab);for(let j=0;j<vocab;j++)fullGrad[(ex.prefix.length+r-1)*vocab+j]=src[j]*scale;}logits.backward(Tensor.from(fullGrad,{shape:logits.shape}));lossSum+=loss.value;trainedTokens+=selectedTargets.length;}
      let norm2=0;for(const param of this._parameters)if(param.grad){assert(TE.sameShape(param.grad.shape,param.shape)&&ops.finite(param.grad),'E_CHAT_GRAD_NONFINITE','Invalid gradient before clipping.');for(const v of param.grad.data)norm2+=v*v;}const gradientNorm=Math.sqrt(norm2),gradientScale=gradientNorm>this.config.maxGradientNorm?this.config.maxGradientNorm/(gradientNorm||1):1;if(gradientScale<1)for(const param of this._parameters)if(param.grad)for(let i=0;i<param.grad.size;i++)param.grad.data[i]*=gradientScale;const updatedParameters=optimizer.step();return {loss:lossSum/prepared.length,updatedParameters,gradientNorm,gradientScale,batchSize:prepared.length,trainedTokens,lossScope:'assistant-response-only'};
    }
    evaluateChatExamples(examples, tokenizer) {
      assert(Array.isArray(examples)&&examples.length>0,'E_CHAT_EVAL_BATCH','A non-empty evaluation set is required.');
      let lossSum=0,tokenCount=0;
      for(const [index,raw] of examples.entries()){
        const ex=raw&&raw._kanamiEncoded?raw:this.prepareChatExample(raw,tokenizer);
        const input=ex.ids.slice(0,-1),targets=ex.ids.slice(1),logits=this.forward(input),vocab=this.config.vocabSize,rows=[],selectedTargets=[];
        for(let ti=ex.prefix.length;ti<ex.ids.length;ti++){rows.push(logits.data.slice((ti-1)*vocab,ti*vocab));selectedTargets.push(targets[ti-1]);}
        if(!selectedTargets.length)throw Object.assign(Error('Evaluation example has no assistant targets.'),{code:'E_CHAT_EVAL_EMPTY',details:{index}});
        const selected=Tensor.from(rows,{shape:[rows.length,vocab]}),loss=K.TrainingSystem.crossEntropy(selected,selectedTargets);
        lossSum+=loss.value*selectedTargets.length;tokenCount+=selectedTargets.length;
      }
      return {loss:lossSum/tokenCount,examples:examples.length,tokens:tokenCount,updatedParameters:0,mode:'evaluation'};
    }
    nextLogits(tokenIds){const logits=this.forward(tokenIds).tensor,seq=logits.shape[0],vocab=logits.shape[1];return Tensor.from(logits.data.slice((seq-1)*vocab,seq*vocab),{shape:[vocab]});}
    async generate(tokenizer, prompt, options = {}) {
      assert(tokenizer && typeof tokenizer.encode === 'function' && typeof tokenizer.decode === 'function', 'E_CHAT_TOKENIZER', 'Tokenizer required.');
      const maxNew = options.maxNewTokens === undefined ? 64 : options.maxNewTokens;
      const minNew = options.minNewTokens === undefined ? 1 : options.minNewTokens;
      assert(Number.isSafeInteger(maxNew) && maxNew >= 0, 'E_CHAT_GENERATION_LIMIT', 'maxNewTokens must be a non-negative integer.');
      assert(Number.isSafeInteger(minNew) && minNew >= 0 && minNew <= maxNew, 'E_CHAT_GENERATION_LIMIT', 'minNewTokens must be between zero and maxNewTokens.');
      const temperature = options.temperature === undefined ? 0.8 : options.temperature;
      const topK = options.topK === undefined ? 40 : options.topK;
      const topP = options.topP === undefined ? 0.9 : options.topP;
      const repetitionPenalty = options.repetitionPenalty === undefined ? 1.05 : options.repetitionPenalty;
      assert(Number.isFinite(temperature) && temperature >= 0 && Number.isFinite(topP) && topP > 0 && topP <= 1 && Number.isSafeInteger(topK) && topK >= 0 && Number.isFinite(repetitionPenalty) && repetitionPenalty >= 1, 'E_CHAT_SAMPLING', 'Invalid sampling configuration.');
      let ids = tokenizer.encode(prompt, { addBos: true, addEos: false });
      if (ids.length > this.config.maxSequenceLength) ids = ids.slice(-this.config.maxSequenceLength);
      const generated = [], stop = options.stopOnEos !== false;
      let cache=this.createInferenceCache(), logits=null;
      for(const id of ids) logits=this.inferenceStep(id,cache);
      let finishReason = 'max_new_tokens';
      const rng = typeof options.rng === 'function' ? options.rng : Math.random;
      for (let step = 0; step < maxNew; step++) {
        if (options.signal && options.signal.aborted) { const e = Error('Generation cancelled.'); e.code = 'E_CANCELLED'; throw e; }
        if(!logits) throw new Error('KV cache did not produce initial logits.');
        const allowed = typeof tokenizer.allowedNextIds === 'function' ? new Set(tokenizer.allowedNextIds(generated)) : null;
        const seen = new Set(generated), candidates = [];
        for (let id = 0; id < logits.size; id++) {
          if (id === tokenizer.eosId && generated.length < minNew) continue;
          if (allowed && !allowed.has(id)) continue;
          let value = logits.data[id];
          if (seen.has(id) && (typeof tokenizer.isTextToken !== 'function' ? id <= 255 : tokenizer.isTextToken(id)) && repetitionPenalty > 1) value = value >= 0 ? value / repetitionPenalty : value * repetitionPenalty;
          if (temperature > 0) value /= temperature;
          candidates.push({ id, value });
        }
        if (!candidates.length) { finishReason = 'constraint'; break; }
        candidates.sort((a, b) => b.value - a.value);
        if (temperature === 0 || topK === 1) {
          const chosen = candidates[0].id;
          if (chosen === tokenizer.eosId && stop) { finishReason = 'eos'; break; }
          generated.push(chosen); ids.push(chosen);
          if(cache.length>=this.config.maxSequenceLength){const keep=ids.slice(-this.config.maxSequenceLength);cache=this.createInferenceCache();for(const id of keep)logits=this.inferenceStep(id,cache);}else logits=this.inferenceStep(chosen,cache);
          if (options.onToken) await options.onToken(chosen, tokenizer.decode(generated), typeof tokenizer.isValidUtf8 !== 'function' || tokenizer.isValidUtf8(generated));
        } else {
          const limited = topK > 0 ? candidates.slice(0, topK) : candidates;
          const max = limited[0].value;
          let weights = limited.map(c => Math.exp(c.value - max)), sum = weights.reduce((a, b) => a + b, 0);
          let normalized = weights.map(w => w / sum);
          if (topP < 1) {
            let cumulative = 0, keep = 0;
            for (; keep < normalized.length; keep++) { cumulative += normalized[keep]; if (cumulative >= topP) { keep++; break; } }
            const count = Math.max(1, keep); limited.length = count; normalized = normalized.slice(0, count);
            sum = normalized.reduce((a, b) => a + b, 0); normalized = normalized.map(w => w / sum);
          }
          let r = Math.max(0, Math.min(0.999999999, rng())), chosen = limited[limited.length - 1].id;
          for (let j = 0; j < normalized.length; j++) { r -= normalized[j]; if (r <= 0) { chosen = limited[j].id; break; } }
          if (chosen === tokenizer.eosId && stop) { finishReason = 'eos'; break; }
          generated.push(chosen); ids.push(chosen);
          if(cache.length>=this.config.maxSequenceLength){const keep=ids.slice(-this.config.maxSequenceLength);cache=this.createInferenceCache();for(const id of keep)logits=this.inferenceStep(id,cache);}else logits=this.inferenceStep(chosen,cache);
          if (options.onToken) await options.onToken(chosen, tokenizer.decode(generated), typeof tokenizer.isValidUtf8 !== 'function' || tokenizer.isValidUtf8(generated));
        }
        if (step % 8 === 7) await new Promise(resolve => setTimeout(resolve, 0));
      }
      while (typeof tokenizer.isValidUtf8 === 'function' && !tokenizer.isValidUtf8(generated) && generated.length) generated.pop();
      return { tokenIds: generated, text: tokenizer.decode(generated), finishReason, sampling: { temperature, topK, topP, repetitionPenalty, minNewTokens: minNew } };
    }
    dispose(){this._disposed=true;this._parameters.length=0;}
    saveConfig(){return Object.assign({},this.config);}
  }
  K.ChatModel = Object.freeze({ version:'1.1.0', ChatModel, CausalMultiHeadAttention, SwiGLU, DecoderBlock });
})(typeof globalThis !== 'undefined' ? globalThis : this);
