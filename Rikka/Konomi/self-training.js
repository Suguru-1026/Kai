(function(root){
  'use strict';
  const K=root.KNA||(root.KNA={});
  function assert(ok,code,message){if(!ok){const e=Error(message);e.code=code;throw e;}}
  async function run(options){
    const o=options||{}, model=o.model, tokenizer=o.tokenizer;
    assert(model&&typeof model.generate==='function'&&typeof model.trainChatBatch==='function','E_SELF_MODEL','A compatible ChatModel is required.');
    assert(tokenizer&&typeof tokenizer.encode==='function','E_SELF_TOKENIZER','Tokenizer is required.');
    const prompts=Array.isArray(o.prompts)?o.prompts:[];
    const rounds=Math.max(1,Math.min(3,Number.isSafeInteger(o.rounds)?o.rounds:1));
    const samplesPerPrompt=Math.max(1,Math.min(2,Number.isSafeInteger(o.samplesPerPrompt)?o.samplesPerPrompt:1));
    const maxPrompts=Math.max(1,Math.min(8,Number.isSafeInteger(o.maxPrompts)?o.maxPrompts:6));
    const batchSize=Math.max(1,Math.min(4,Number.isSafeInteger(o.batchSize)?o.batchSize:4));
    const maxNewTokens=Math.max(4,Math.min(24,Number.isSafeInteger(o.maxNewTokens)?o.maxNewTokens:16));
    const maxGenerations=rounds*Math.min(prompts.length,maxPrompts)*samplesPerPrompt;
    assert(prompts.length>0,'E_SELF_PROMPTS','At least one internal prompt is required.');
    assert(maxGenerations<=48,'E_SELF_LIMIT','Generation budget exceeds the hard limit.');
    const accepted=[], roundStats=[]; let generationCount=0, updateCount=0;
    const anchorSource=Array.isArray(o.anchorExamples)?o.anchorExamples:[];
    const prepare=(x)=>x&&x._kanamiEncoded?x:model.prepareChatExample(x,tokenizer);
    const anchors=[];
    for(const x of anchorSource){try{anchors.push(prepare(x));}catch(_){}}
    const yieldUI=()=>new Promise(resolve=>setTimeout(resolve,0));
    for(let round=1;round<=rounds;round++){
      if(o.signal&&o.signal.aborted){const e=Error('Self-training cancelled.');e.code='E_CANCELLED';throw e;}
      const roundExamples=[]; let rejected=0;
      for(const prompt of prompts.slice(0,maxPrompts)){
        for(let sample=0;sample<samplesPerPrompt;sample++){
          if(o.signal&&o.signal.aborted){const e=Error('Self-training cancelled.');e.code='E_CANCELLED';throw e;}
          if(generationCount>=48)break;
          const result=await model.generate(tokenizer,prompt,{maxNewTokens,minNewTokens:1,temperature:0.9,topK:24,topP:0.9,repetitionPenalty:1.12,signal:o.signal});
          generationCount++;
          const response=String(result.text||'').trim();
          const tokenIds=Array.isArray(result.tokenIds)?result.tokenIds:[];
          const unique=new Set(tokenIds).size;
          const repetitive=tokenIds.length>=8&&unique/Math.max(1,tokenIds.length)<0.18;
          const valid=!!response&&response.length<=300&&(!tokenizer.isValidUtf8||tokenizer.isValidUtf8(tokenIds))&&!repetitive;
          if(valid){
            try{const ex=prepare({prompt,response});roundExamples.push(ex);accepted.push({round,prompt,response,tokenCount:tokenIds.length});}
            catch(_){rejected++;}
          }else rejected++;
          if(typeof o.onProgress==='function')o.onProgress({phase:'generate',round,rounds,generationCount,maxGenerations,accepted:accepted.length,rejected});
          await yieldUI();
        }
      }
      // Keep self-generated answers anchored to a small, bounded sample of real seed examples.
      const mixed=roundExamples.slice();
      const anchorCount=Math.min(anchors.length,Math.max(1,roundExamples.length));
      for(let i=0;i<anchorCount;i++)mixed.push(anchors[(round-1+i)%anchors.length]);
      // Deterministic shuffle within this run; never exceed six optimizer updates per round.
      for(let i=mixed.length-1;i>0;i--){const j=Math.floor(Math.random()*(i+1));[mixed[i],mixed[j]]=[mixed[j],mixed[i]];}
      let lossSum=0,lossBatches=0;
      for(let i=0;i<mixed.length;i+=batchSize){
        if(o.signal&&o.signal.aborted){const e=Error('Self-training cancelled.');e.code='E_CANCELLED';throw e;}
        if(updateCount>=24)break;
        const batch=mixed.slice(i,i+batchSize); if(!batch.length)continue;
        const r=model.trainChatBatch(batch,tokenizer,o.optimizer);updateCount++;lossSum+=r.loss;lossBatches++;
        if(typeof o.onProgress==='function')o.onProgress({phase:'train',round,rounds,generationCount,maxGenerations,accepted:accepted.length,rejected,updateCount,loss:r.loss});
        await yieldUI();
      }
      roundStats.push({round,generated:roundExamples.length,rejected,trainingExamples:mixed.length,updates:lossBatches,meanBatchLoss:lossBatches?lossSum/lossBatches:null});
      if(typeof o.onRound==='function')o.onRound(roundStats[roundStats.length-1]);
      if(updateCount>=24)break;
    }
    return {roundsCompleted:roundStats.length,generationCount,acceptedCount:accepted.length,updateCount,roundStats,acceptedExamples:accepted,warning:'Self-generated responses are pseudo-labels, not verified correct answers. Anchor examples reduce but do not eliminate feedback-loop degradation.'};
  }
  K.SelfTraining=Object.freeze({version:'1.0.0',run,limits:Object.freeze({rounds:3,prompts:8,samplesPerPrompt:2,generations:48,optimizerUpdates:24,maxNewTokens:24})});
})(typeof globalThis!=='undefined'?globalThis:this);
