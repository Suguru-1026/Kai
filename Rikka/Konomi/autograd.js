(function (root) {
  'use strict';
  const KNA = root.KNA || (root.KNA = {}), TE = KNA.TensorEngine;
  if (!TE) throw new Error('KNA.Autograd requires TensorEngine first.');
  const { Tensor, ops, errors } = TE, assert = errors.assert;

  class Value {
    constructor(tensor, options = {}) {
      assert(tensor instanceof Tensor, 'E_AUTOGRAD_TENSOR', 'Value requires Tensor.');
      this.tensor = tensor;
      this.requiresGrad = !!options.requiresGrad;
      this.grad = null;
      this.parents = [];
      this.backwardFn = () => {};
      this.label = options.label || '';
    }
    get shape() { return this.tensor.shape; }
    get data() { return this.tensor.data; }
    get size() { return this.tensor.size; }
    item() { return this.tensor.item(); }
    zeroGrad() { this.grad = null; return this; }
    detach() { return new Value(this.tensor.clone()); }
    backward(seed) {
      if (seed !== undefined) {
        assert(seed instanceof Tensor, 'E_AUTOGRAD_SEED', 'Seed must be a Tensor.');
        assert(TE.sameShape(seed.shape, this.shape), 'E_AUTOGRAD_SEED_SHAPE', 'Seed shape must match output shape.');
      } else {
        assert(this.size === 1, 'E_AUTOGRAD_SEED', 'Non-scalar output requires an explicit seed Tensor.');
      }
      const topo = [], seen = new Set();
      const visit = v => { if (seen.has(v)) return; seen.add(v); v.parents.forEach(visit); topo.push(v); };
      visit(this);
      const gradientSnapshot = new Map(topo.map(v => [v, v.grad ? v.grad.clone() : null]));
      try {
        // Intermediate gradients belong to this backward pass; leaf gradients accumulate.
        for (const v of topo) if (v.parents.length) v.grad = null;
        this.grad = seed ? seed.clone() : Tensor.full(this.shape, 1);
        for (let i = topo.length - 1; i >= 0; i--) topo[i].backwardFn();
      } catch (e) {
        // A failed backward pass must not leave partially accumulated gradients behind.
        for (const [v, grad] of gradientSnapshot) v.grad = grad ? grad.clone() : null;
        throw e;
      }
      return this;
    }
  }
  const asValue = x => x instanceof Value ? x : new Value(x instanceof Tensor ? x : Tensor.scalar(x));
  function accumulate(v, g) {
    if (!v.requiresGrad) return;
    assert(g instanceof Tensor && TE.sameShape(g.shape, v.shape), 'E_AUTOGRAD_GRAD_SHAPE', 'Gradient shape mismatch.');
    if (!v.grad) v.grad = g.clone(); else v.grad = ops.add(v.grad, g);
  }
  function reduce(g, target) {
    if (TE.sameShape(g.shape, target.shape)) return g;
    if (target.rank === 0) return Tensor.scalar(ops.sum(g).item());
    assert(g.rank >= target.rank, 'E_AUTOGRAD_GRAD_SHAPE', 'Gradient rank is smaller than operand rank.');
    const pad = g.rank - target.rank, padded = Array(pad).fill(1).concat(Array.from(target.shape));
    for (let axis = 0; axis < g.rank; axis++) {
      if (padded[axis] === 1 && g.shape[axis] !== 1) g = ops.sumAxis(g, axis, true);
      else assert(padded[axis] === g.shape[axis], 'E_AUTOGRAD_GRAD_SHAPE', 'Cannot reduce gradient to operand shape.');
    }
    return ops.reshape(g, target.shape);
  }
  function binary(a, b, f, fa, fb, name) {
    a = asValue(a); b = asValue(b);
    const tensor = f(a.tensor, b.tensor);
    const out = new Value(tensor, { requiresGrad: a.requiresGrad || b.requiresGrad, label: name });
    out.parents = [a, b];
    out.backwardFn = () => {
      if (!out.grad) return;
      if (a.requiresGrad) accumulate(a, fa(a.tensor, b.tensor, out.grad));
      if (b.requiresGrad) accumulate(b, fb(a.tensor, b.tensor, out.grad));
    };
    return out;
  }
  function unary(x, f, df, name) {
    x = asValue(x); const y = f(x.tensor);
    const out = new Value(y, { requiresGrad: x.requiresGrad, label: name });
    out.parents = [x];
    out.backwardFn = () => { if (x.requiresGrad && out.grad) accumulate(x, df(x.tensor, y, out.grad)); };
    return out;
  }
  const add = (a,b) => binary(a,b,ops.add,(x,y,g)=>reduce(g,x),(x,y,g)=>reduce(g,y),'add');
  const sub = (a,b) => binary(a,b,ops.sub,(x,y,g)=>reduce(g,x),(x,y,g)=>reduce(ops.mul(g,-1),y),'sub');
  const mul = (a,b) => binary(a,b,ops.mul,(x,y,g)=>reduce(ops.mul(g,y),x),(x,y,g)=>reduce(ops.mul(g,x),y),'mul');
  const div = (a,b) => binary(a,b,ops.div,(x,y,g)=>reduce(ops.div(g,y),x),(x,y,g)=>reduce(ops.mul(ops.div(ops.mul(g,x),ops.mul(y,y)),-1),y),'div');
  const tanh = x => unary(x,ops.tanh,(x,y,g)=>ops.mul(g,ops.sub(1,ops.mul(y,y))),'tanh');
  const exp = x => unary(x,ops.exp,(x,y,g)=>ops.mul(g,y),'exp');
  const log = x => unary(x,ops.log,(x,y,g)=>ops.div(g,x),'log');
  const relu = x => unary(x,ops.relu,(x,y,g)=>{const a=new Float32Array(x.size);for(let i=0;i<x.size;i++)a[i]=x.data[i]>0?g.data[i]:0;return new Tensor(a,x.shape);},'relu');
  const sum = x => unary(x,ops.sum,(x,y,g)=>Tensor.full(x.shape,g.item()),'sum');
  const mean = x => unary(x,ops.mean,(x,y,g)=>Tensor.full(x.shape,g.item()/x.size),'mean');
  function matmul(a,b) {
    a=asValue(a); b=asValue(b);
    const out=new Value(ops.matmul(a.tensor,b.tensor),{requiresGrad:a.requiresGrad||b.requiresGrad,label:'matmul'});
    out.parents=[a,b];
    out.backwardFn=()=>{if(!out.grad)return;if(a.requiresGrad)accumulate(a,ops.matmul(out.grad,ops.transpose2D(b.tensor)));if(b.requiresGrad)accumulate(b,ops.matmul(ops.transpose2D(a.tensor),out.grad));};
    return out;
  }
  const max=x=>{x=asValue(x);const y=ops.max(x.tensor),out=new Value(y,{requiresGrad:x.requiresGrad,label:'max'});out.parents=[x];out.backwardFn=()=>{if(!x.requiresGrad||!out.grad)return;let count=0;for(const v of x.data)if(v===y.item())count++;const d=new Float32Array(x.size);if(count)for(let i=0;i<x.size;i++)if(x.data[i]===y.item())d[i]=out.grad.item()/count;accumulate(x,Tensor.from(d,{shape:x.shape}));};return out;};
  const sigmoid=x=>unary(x,ops.sigmoid,(a,y,g)=>ops.mul(g,ops.mul(y,ops.sub(1,y))),'sigmoid');
  const silu=x=>unary(x,ops.silu,(a,y,g)=>{const d=new Float32Array(a.size);for(let i=0;i<a.size;i++){const q=1/(1+Math.exp(-a.data[i]));d[i]=g.data[i]*(q+a.data[i]*q*(1-q));}return new Tensor(d,a.shape);},'silu');
  const gelu=x=>unary(x,ops.gelu,(a,y,g)=>{const d=new Float32Array(a.size),c=Math.sqrt(2/Math.PI);for(let i=0;i<a.size;i++){const v=a.data[i],u=c*(v+0.044715*v*v*v),t=Math.tanh(u);d[i]=g.data[i]*(0.5*(1+t)+0.5*v*(1-t*t)*c*(1+3*0.044715*v*v));}return new Tensor(d,a.shape);},'gelu');
  function reshape(x,shape){x=asValue(x);const y=ops.reshape(x.tensor,shape),out=new Value(y,{requiresGrad:x.requiresGrad,label:'reshape'});out.parents=[x];out.backwardFn=()=>{if(x.requiresGrad&&out.grad)accumulate(x,ops.reshape(out.grad,x.shape));};return out;}
  function transpose(x,axes){x=asValue(x);const normalized=axes===undefined?Array.from({length:x.shape.length},(_,i)=>x.shape.length-1-i):axes.slice(),inverse=new Array(normalized.length);normalized.forEach((v,i)=>inverse[v]=i);const out=new Value(ops.transpose(x.tensor,normalized),{requiresGrad:x.requiresGrad,label:'transpose'});out.parents=[x];out.backwardFn=()=>{if(x.requiresGrad&&out.grad)accumulate(x,ops.transpose(out.grad,inverse));};return out;}
  function reduceToShape(g,shape){if(TE.sameShape(g.shape,shape))return g;if(shape.length===0)return Tensor.scalar(ops.sum(g).item());const pad=g.rank-shape.length;assert(pad>=0,'E_AUTOGRAD_GRAD_SHAPE','Cannot reduce gradient rank.');const padded=Array(pad).fill(1).concat(shape);for(let axis=0;axis<g.rank;axis++){if(padded[axis]===1&&g.shape[axis]!==1)g=ops.sumAxis(g,axis,true);else assert(padded[axis]===g.shape[axis],'E_AUTOGRAD_GRAD_SHAPE','Cannot reduce broadcast gradient.');}return ops.reshape(g,shape);}
  function sumAxis(x,axis,keepDims=false,average=false){x=asValue(x);const ax=axis<0?axis+x.shape.length:axis,dim=x.shape[ax],y=average?ops.meanAxis(x.tensor,ax,keepDims):ops.sumAxis(x.tensor,ax,keepDims),out=new Value(y,{requiresGrad:x.requiresGrad,label:average?'meanAxis':'sumAxis'});out.parents=[x];out.backwardFn=()=>{if(!x.requiresGrad||!out.grad)return;let g=out.grad;if(!keepDims){const shape=Array.from(x.shape);shape[ax]=1;g=ops.reshape(g,shape);}g=ops.broadcastTo(g,x.shape);if(average)g=ops.div(g,dim);accumulate(x,g);};return out;}
  const meanAxis=(x,axis,keepDims=false)=>sumAxis(x,axis,keepDims,true);
  function maxAxis(x,axis,keepDims=false){x=asValue(x);const ax=axis<0?axis+x.shape.length:axis,dim=x.shape[ax],y=ops.maxAxis(x.tensor,ax,keepDims),out=new Value(y,{requiresGrad:x.requiresGrad,label:'maxAxis'});out.parents=[x];out.backwardFn=()=>{if(!x.requiresGrad||!out.grad)return;const outer=TE.sizeOf(x.shape.slice(0,ax)),inner=TE.sizeOf(x.shape.slice(ax+1)),d=new Float32Array(x.size);for(let o=0;o<outer;o++)for(let i=0;i<inner;i++){let v=-Infinity,count=0;for(let j=0;j<dim;j++)v=Math.max(v,x.data[(o*dim+j)*inner+i]);for(let j=0;j<dim;j++)if(x.data[(o*dim+j)*inner+i]===v)count++;const go=out.grad.data[o*inner+i]/count;for(let j=0;j<dim;j++){const ix=(o*dim+j)*inner+i;if(x.data[ix]===v)d[ix]=go;}}accumulate(x,Tensor.from(d,{shape:x.shape}));};return out;}
  function softmax(x,axis=-1,asLog=false){x=asValue(x);const y=asLog?ops.logSoftmax(x.tensor,axis):ops.softmax(x.tensor,axis),out=new Value(y,{requiresGrad:x.requiresGrad,label:asLog?'logSoftmax':'softmax'});out.parents=[x];out.backwardFn=()=>{if(!x.requiresGrad||!out.grad)return;const ax=axis<0?axis+x.shape.length:axis,outer=TE.sizeOf(x.shape.slice(0,ax)),dim=x.shape[ax],inner=TE.sizeOf(x.shape.slice(ax+1)),d=new Float32Array(x.size);for(let o=0;o<outer;o++)for(let i=0;i<inner;i++){let dot=0;for(let j=0;j<dim;j++){const ix=(o*dim+j)*inner+i;dot+=asLog?out.grad.data[ix]:out.grad.data[ix]*y.data[ix];}for(let j=0;j<dim;j++){const ix=(o*dim+j)*inner+i;d[ix]=asLog?out.grad.data[ix]-Math.exp(y.data[ix])*dot:y.data[ix]*(out.grad.data[ix]-dot);}}accumulate(x,Tensor.from(d,{shape:x.shape}));};return out;}
  const logSoftmax=(x,axis=-1)=>softmax(x,axis,true);
  function slice(x,axis,start,end,step=1){x=asValue(x);const ax=axis<0?axis+x.shape.length:axis,dim=x.shape[ax],st=start===undefined?0:start,en=end===undefined?dim:end,sp=step,count=Math.ceil((en-st)/sp),out=new Value(ops.slice(x.tensor,ax,st,en,sp),{requiresGrad:x.requiresGrad,label:'slice'});out.parents=[x];out.backwardFn=()=>{if(!x.requiresGrad||!out.grad)return;const outer=TE.sizeOf(x.shape.slice(0,ax)),inner=TE.sizeOf(x.shape.slice(ax+1)),g=new Float32Array(x.size);for(let o=0;o<outer;o++)for(let j=0;j<count;j++)for(let i=0;i<inner;i++)g[(o*dim+st+j*sp)*inner+i]+=out.grad.data[(o*count+j)*inner+i];accumulate(x,Tensor.from(g,{shape:x.shape}));};return out;}
  function concat(inputs,axis=0){const xs=inputs.map(asValue),y=ops.concat(xs.map(x=>x.tensor),axis),ax=axis<0?axis+y.rank:axis,out=new Value(y,{requiresGrad:xs.some(x=>x.requiresGrad),label:'concat'});out.parents=xs;out.backwardFn=()=>{if(!out.grad)return;let offset=0;xs.forEach(x=>{const end=offset+x.shape[ax];if(x.requiresGrad)accumulate(x,ops.slice(out.grad,ax,offset,end));offset=end;});};return out;}
  function gather(x,indices,axis=0){x=asValue(x);const ax=axis<0?axis+x.shape.length:axis,ids=Array.from(indices),out=new Value(ops.gather(x.tensor,ids,ax),{requiresGrad:x.requiresGrad,label:'gather'});out.parents=[x];out.backwardFn=()=>{if(!x.requiresGrad||!out.grad)return;const outer=TE.sizeOf(x.shape.slice(0,ax)),dim=x.shape[ax],inner=TE.sizeOf(x.shape.slice(ax+1)),g=new Float32Array(x.size);for(let o=0;o<outer;o++)for(let j=0;j<ids.length;j++)for(let i=0;i<inner;i++)g[(o*dim+ids[j])*inner+i]+=out.grad.data[(o*ids.length+j)*inner+i];accumulate(x,Tensor.from(g,{shape:x.shape}));};return out;}
  function where(mask,a,b){a=asValue(a);b=asValue(b);const y=ops.where(mask,a.tensor,b.tensor),out=new Value(y,{requiresGrad:a.requiresGrad||b.requiresGrad,label:'where'});out.parents=[a,b];out.backwardFn=()=>{if(!out.grad)return;if(a.requiresGrad)accumulate(a,reduceToShape(ops.where(mask,out.grad,0),a.shape));if(b.requiresGrad)accumulate(b,reduceToShape(ops.where(mask,0,out.grad),b.shape));};return out;}
  function maskedFill(x,mask,value){x=asValue(x);const out=new Value(ops.maskedFill(x.tensor,mask,value),{requiresGrad:x.requiresGrad,label:'maskedFill'});out.parents=[x];out.backwardFn=()=>{if(x.requiresGrad&&out.grad)accumulate(x,ops.where(mask,0,out.grad));};return out;}
  function broadcastTo(x,shape){x=asValue(x);const out=new Value(ops.broadcastTo(x.tensor,shape),{requiresGrad:x.requiresGrad,label:'broadcastTo'});out.parents=[x];out.backwardFn=()=>{if(x.requiresGrad&&out.grad)accumulate(x,reduceToShape(out.grad,x.shape));};return out;}
  function cast(x,dtype){x=asValue(x);assert(!x.requiresGrad||dtype==='float32','E_AUTOGRAD_CAST','Gradient tracking through integer/bool casts is unsupported.');const out=new Value(ops.cast(x.tensor,dtype),{requiresGrad:x.requiresGrad,label:'cast'});out.parents=[x];out.backwardFn=()=>{if(x.requiresGrad&&out.grad)accumulate(x,ops.cast(out.grad,x.tensor.dtype));};return out;}
  function parameter(value, shape, label) {
    const t=value instanceof Tensor?value:Tensor.from(value,shape?{shape}:undefined);
    return new Value(t,{requiresGrad:true,label:label||''});
  }
  function zeroGrad(values) { for (const v of values) v.zeroGrad(); }
  KNA.Autograd=Object.freeze({version:'0.6.0',Value,asValue,parameter,zeroGrad,accumulate,ops:Object.freeze({add,sub,mul,div,tanh,exp,log,relu,sigmoid,silu,gelu,sum,mean,max,sumAxis,meanAxis,maxAxis,softmax,logSoftmax,reshape,transpose,matmul,slice,concat,gather,where,maskedFill,broadcastTo,cast})});
})(typeof globalThis !== 'undefined' ? globalThis : this);
