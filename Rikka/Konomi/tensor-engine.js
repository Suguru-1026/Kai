(function (root) {
  'use strict';
  const KNA = root.KNA || (root.KNA = {});
  const MAX_ELEMENTS = 16777216;
  const DTYPES = Object.freeze({ float32: Float32Array, int32: Int32Array, bool: Uint8Array });
  function error(code, message, details) { const e = new Error(message); e.name = 'KNAError'; e.code = code; if (details !== undefined) e.details = details; throw e; }
  function assert(ok, code, message, details) { if (!ok) error(code, message, details); }
  function shapeOf(shape) {
    assert(Array.isArray(shape), 'E_SHAPE_TYPE', 'Shape must be an array.');
    const s = shape.map((d, i) => { assert(Number.isSafeInteger(d) && d >= 0, 'E_SHAPE_DIM', 'Shape dimensions must be non-negative safe integers.', { axis: i, value: d }); return d; });
    const n = s.reduce((a, b) => a * b, 1);
    assert(Number.isSafeInteger(n) && n <= MAX_ELEMENTS, 'E_RESOURCE_LIMIT', 'Tensor exceeds the configured element limit.', { elements: n, limit: MAX_ELEMENTS });
    return s;
  }
  const sizeOf = s => s.reduce((a, b) => a * b, 1);
  const stridesOf = s => { const r = new Array(s.length); let n = 1; for (let i = s.length - 1; i >= 0; i--) { r[i] = n; n *= s[i]; } return r; };
  const sameShape = (a, b) => a.length === b.length && a.every((d, i) => d === b[i]);
  function flatten(x, out) { if (Array.isArray(x) || ArrayBuffer.isView(x)) { for (const v of x) flatten(v, out); } else out.push(x); return out; }
  function inferShape(x) {
    if (!Array.isArray(x)) return [];
    const child = x.length ? inferShape(x[0]) : [];
    for (let i = 1; i < x.length; i++) assert(sameShape(inferShape(x[i]), child), 'E_RAGGED_ARRAY', 'Nested input arrays must be rectangular.');
    return [x.length, ...child];
  }
  class Tensor {
    constructor(data, shape, dtype = 'float32') {
      assert(DTYPES[dtype], 'E_DTYPE', 'Unsupported dtype: ' + dtype);
      this.shape = Object.freeze(shapeOf(shape)); this.size = sizeOf(this.shape); this.rank = this.shape.length; this.dtype = dtype;
      const values = ArrayBuffer.isView(data) ? Array.from(data) : Array.isArray(data) ? flatten(data, []) : [data];
      assert(values.length === this.size, 'E_DATA_SHAPE', 'Data length does not match shape.', { expected: this.size, actual: values.length });
      this.data = DTYPES[dtype].from(values, v => dtype === 'bool' ? (v ? 1 : 0) : dtype === 'int32' ? Math.trunc(Number(v)) : Number(v));
    }
    clone() { return new Tensor(this.data, this.shape, this.dtype); }
    item() { assert(this.size === 1, 'E_TENSOR_ITEM', 'item() requires one element.'); return this.data[0]; }
    get(indices) { assert(Array.isArray(indices) && indices.length === this.rank, 'E_INDEX_RANK', 'Index rank mismatch.'); const st = stridesOf(this.shape); let o = 0; for (let i=0;i<indices.length;i++){assert(Number.isInteger(indices[i])&&indices[i]>=0&&indices[i]<this.shape[i],'E_INDEX_RANGE','Index out of range.');o+=indices[i]*st[i];} return this.data[o]; }
    toArray() { if (!this.rank) return this.data[0]; const st=stridesOf(this.shape); const build=(dim,off)=>{const a=[];for(let i=0;i<this.shape[dim];i++)a.push(dim===this.rank-1?this.data[off+i]:build(dim+1,off+i*st[dim]));return a;};return build(0,0); }
    static from(value, options = {}) { const s = options.shape ? shapeOf(options.shape) : inferShape(value); return new Tensor(flatten(value, []), s, options.dtype || 'float32'); }
    static zeros(shape, dtype) { const s=shapeOf(shape); return new Tensor(new Array(sizeOf(s)).fill(0),s,dtype); }
    static ones(shape, dtype) { const s=shapeOf(shape); return new Tensor(new Array(sizeOf(s)).fill(1),s,dtype); }
    static full(shape, value, dtype) { const s=shapeOf(shape); return new Tensor(new Array(sizeOf(s)).fill(value),s,dtype); }
    static scalar(value, dtype) { return new Tensor([value], [], dtype); }
  }
  function binary(a,b,fn,name) { if(!(a instanceof Tensor))a=Tensor.scalar(a);if(!(b instanceof Tensor))b=Tensor.scalar(b);const s=broadcastShape(a.shape,b.shape),aa=broadcastTo(a,s),bb=broadcastTo(b,s),n=sizeOf(s),o=new Float32Array(n);for(let i=0;i<n;i++)o[i]=fn(aa.data[i],bb.data[i]);return new Tensor(o,s); }
  function unary(a,fn,name) { assert(a instanceof Tensor,'E_TENSOR_REQUIRED',name+' requires a Tensor.');const o=new Float32Array(a.size);for(let i=0;i<a.size;i++)o[i]=fn(a.data[i]);return new Tensor(o,a.shape); }
  const add=(a,b)=>binary(a,b,(x,y)=>x+y,'add'), sub=(a,b)=>binary(a,b,(x,y)=>x-y,'sub'), mul=(a,b)=>binary(a,b,(x,y)=>x*y,'mul'), div=(a,b)=>binary(a,b,(x,y)=>x/y,'div');
  const exp=a=>unary(a,Math.exp,'exp'), log=a=>unary(a,Math.log,'log'), tanh=a=>unary(a,Math.tanh,'tanh'), relu=a=>unary(a,x=>Math.max(0,x),'relu'), sqrt=a=>unary(a,Math.sqrt,'sqrt'), sigmoid=a=>unary(a,x=>1/(1+Math.exp(-x)),'sigmoid'), silu=a=>unary(a,x=>x/(1+Math.exp(-x)),'silu'), gelu=a=>unary(a,x=>0.5*x*(1+Math.tanh(Math.sqrt(2/Math.PI)*(x+0.044715*x*x*x))),'gelu');
  function sum(a) { let x=0; for(const v of a.data)x+=v;return Tensor.scalar(x); }
  function max(a) { assert(a instanceof Tensor&&a.size>0,'E_EMPTY_REDUCTION','max of an empty tensor is undefined.');let v=-Infinity;for(const x of a.data)v=Math.max(v,x);return Tensor.scalar(v); }
  function mean(a) { assert(a.size>0,'E_EMPTY_REDUCTION','mean of empty tensor is undefined.');return Tensor.scalar(sum(a).item()/a.size); }
  function axisOf(axis,rank){assert(Number.isInteger(axis),'E_AXIS','Axis must be an integer.');const a=axis<0?axis+rank:axis;assert(a>=0&&a<rank,'E_AXIS','Axis is out of range.',{axis,rank});return a;}
  function reshape(a,s) { s=shapeOf(s);assert(sizeOf(s)===a.size,'E_RESHAPE_SIZE','Reshape must preserve element count.');return new Tensor(a.data,s,a.dtype); }
  function transpose(a,axes){assert(a instanceof Tensor,'E_TENSOR_REQUIRED','transpose requires a Tensor.');axes=axes===undefined?Array.from({length:a.rank},(_,i)=>a.rank-1-i):axes.slice();assert(axes.length===a.rank&&new Set(axes).size===a.rank&&axes.every(x=>Number.isInteger(x)&&x>=0&&x<a.rank),'E_TRANSPOSE_AXES','Axes must be a permutation of tensor axes.');const shape=axes.map(i=>a.shape[i]),out=new Float32Array(a.size),inSt=stridesOf(a.shape),outSt=stridesOf(shape);for(let idx=0;idx<a.size;idx++){let rem=idx,coords=new Array(a.rank);for(let i=0;i<a.rank;i++){coords[i]=Math.floor(rem/inSt[i]);rem%=inSt[i];}let oi=0;for(let i=0;i<axes.length;i++)oi+=coords[axes[i]]*outSt[i];out[oi]=a.data[idx];}return new Tensor(out,shape,a.dtype);}
  function transpose2D(a) { assert(a.rank===2,'E_RANK','transpose2D requires rank 2.');return transpose(a,[1,0]); }
  function matmul(a,b) { assert(a instanceof Tensor&&b instanceof Tensor&&a.rank===2&&b.rank===2,'E_MATMUL_RANK','matmul requires rank-2 tensors.');assert(a.shape[1]===b.shape[0],'E_MATMUL_SHAPE','matmul inner dimensions must match.',{a:a.shape,b:b.shape});const [m,k]=a.shape,n=b.shape[1];shapeOf([m,n]);const o=new Float32Array(m*n);for(let i=0;i<m;i++)for(let p=0;p<k;p++){const v=a.data[i*k+p];for(let j=0;j<n;j++)o[i*n+j]+=v*b.data[p*n+j];}return new Tensor(o,[m,n]); }
  function reduceAxis(a,axis,keepDims,average){axis=axisOf(axis,a.rank);const outer=sizeOf(a.shape.slice(0,axis)),dim=a.shape[axis],inner=sizeOf(a.shape.slice(axis+1)),shape=a.shape.slice();if(keepDims)shape[axis]=1;else shape.splice(axis,1);const out=new Float32Array(outer*inner);for(let o=0;o<outer;o++)for(let i=0;i<inner;i++){let v=0;for(let d=0;d<dim;d++)v+=a.data[(o*dim+d)*inner+i];out[o*inner+i]=average?v/dim:v;}return new Tensor(out,shape,average?'float32':a.dtype);}
  const sumAxis=(a,axis,keepDims=false)=>reduceAxis(a,axis,keepDims,false);
  function maxAxis(a,axis,keepDims=false){axis=axisOf(axis,a.rank);const outer=sizeOf(a.shape.slice(0,axis)),dim=a.shape[axis],inner=sizeOf(a.shape.slice(axis+1));assert(dim>0,'E_EMPTY_REDUCTION','max of an empty axis is undefined.');const shape=a.shape.slice();if(keepDims)shape[axis]=1;else shape.splice(axis,1);const out=new Float32Array(outer*inner);for(let o=0;o<outer;o++)for(let i=0;i<inner;i++){let v=-Infinity;for(let d=0;d<dim;d++)v=Math.max(v,a.data[(o*dim+d)*inner+i]);out[o*inner+i]=v;}return new Tensor(out,shape);}
  const meanAxis=(a,axis,keepDims=false)=>{assert(a.shape[axisOf(axis,a.rank)]>0,'E_EMPTY_REDUCTION','mean of empty axis is undefined.');return reduceAxis(a,axis,keepDims,true);};
  function softmax(a,axis=-1,asLog=false){assert(a instanceof Tensor&&a.rank>0,'E_SOFTMAX_RANK','Softmax requires a non-scalar Tensor.');axis=axisOf(axis,a.rank);const outer=sizeOf(a.shape.slice(0,axis)),dim=a.shape[axis],inner=sizeOf(a.shape.slice(axis+1)),out=new Float32Array(a.size);for(let o=0;o<outer;o++)for(let i=0;i<inner;i++){let max=-Infinity;for(let d=0;d<dim;d++)max=Math.max(max,a.data[(o*dim+d)*inner+i]);let z=0;for(let d=0;d<dim;d++)z+=Math.exp(a.data[(o*dim+d)*inner+i]-max);const lse=max+Math.log(z);for(let d=0;d<dim;d++){const ix=(o*dim+d)*inner+i;out[ix]=asLog?a.data[ix]-lse:Math.exp(a.data[ix]-max)/z;}}return new Tensor(out,a.shape);}
  const softmaxOp=(a,axis=-1)=>softmax(a,axis,false), logSoftmax=(a,axis=-1)=>softmax(a,axis,true);
  function broadcastShape(a,b){const n=Math.max(a.length,b.length),out=new Array(n);for(let i=0;i<n;i++){const da=a[a.length-1-i]??1,db=b[b.length-1-i]??1;assert(da===db||da===1||db===1,'E_BROADCAST_SHAPE','Shapes cannot be broadcast.',{a,b});out[n-1-i]=da===1?db:db===1?da:da;}return out;}
  function broadcastTo(a,target){target=shapeOf(target);assert(a.rank<=target.length,'E_BROADCAST_RANK','Cannot broadcast to lower rank.');const pad=target.length-a.rank,srcShape=Array(pad).fill(1).concat(a.shape);for(let i=0;i<target.length;i++)assert(srcShape[i]===target[i]||srcShape[i]===1,'E_BROADCAST_SHAPE','Source shape cannot broadcast to target.',{source:a.shape,target});const out=new Float32Array(sizeOf(target)),srcSt=stridesOf(srcShape),dstSt=stridesOf(target);for(let idx=0;idx<out.length;idx++){let rem=idx,src=0;for(let i=0;i<target.length;i++){const c=Math.floor(rem/dstSt[i]);rem%=dstSt[i];if(srcShape[i]!==1)src+=c*srcSt[i];}out[idx]=a.data[src];}return new Tensor(out,target,a.dtype);}
  function slice(a,axis,start,end,step=1){axis=axisOf(axis,a.rank);const dim=a.shape[axis];start=start===undefined?0:start;end=end===undefined?dim:end;assert(Number.isInteger(start)&&Number.isInteger(end)&&Number.isInteger(step)&&step>0&&start>=0&&end>=start&&end<=dim,'E_SLICE_RANGE','Invalid slice range.');const count=Math.ceil((end-start)/step),shape=a.shape.slice();shape[axis]=count;const outer=sizeOf(a.shape.slice(0,axis)),inner=sizeOf(a.shape.slice(axis+1)),out=new Float32Array(sizeOf(shape));for(let o=0;o<outer;o++)for(let j=0;j<count;j++)for(let i=0;i<inner;i++)out[(o*count+j)*inner+i]=a.data[(o*dim+start+j*step)*inner+i];return new Tensor(out,shape,a.dtype);}
  function concat(tensors,axis=0){assert(Array.isArray(tensors)&&tensors.length>0,'E_CONCAT_INPUT','concat requires a non-empty Tensor array.');const first=tensors[0];assert(first instanceof Tensor,'E_CONCAT_TENSOR','concat inputs must be Tensors.');axis=axisOf(axis,first.rank);const shape=first.shape.slice();shape[axis]=0;for(const t of tensors){assert(t instanceof Tensor&&t.rank===first.rank&&t.dtype===first.dtype,'E_CONCAT_TENSOR','concat ranks and dtypes must match.');for(let i=0;i<t.rank;i++)if(i!==axis)assert(t.shape[i]===first.shape[i],'E_CONCAT_SHAPE','concat non-axis dimensions must match.');shape[axis]+=t.shape[axis];}shapeOf(shape);const outer=sizeOf(shape.slice(0,axis)),inner=sizeOf(shape.slice(axis+1)),out=new Float32Array(sizeOf(shape));for(let o=0;o<outer;o++){let offset=0;for(const t of tensors){const d=t.shape[axis],n=d*inner;out.set(t.data.subarray(o*n,(o+1)*n),o*shape[axis]*inner+offset*inner);offset+=d;}}return new Tensor(out,shape,first.dtype);}
  function gather(a,indices,axis=0){axis=axisOf(axis,a.rank);assert(Array.isArray(indices)||ArrayBuffer.isView(indices),'E_GATHER_INDICES','Indices must be an array.');indices=Array.from(indices);for(const ix of indices)assert(Number.isSafeInteger(ix)&&ix>=0&&ix<a.shape[axis],'E_GATHER_INDEX','Gather index out of range.',{ix,axis});const shape=a.shape.slice();shape[axis]=indices.length;shapeOf(shape);const outer=sizeOf(a.shape.slice(0,axis)),dim=a.shape[axis],inner=sizeOf(a.shape.slice(axis+1)),out=new Float32Array(sizeOf(shape));for(let o=0;o<outer;o++)for(let j=0;j<indices.length;j++)for(let i=0;i<inner;i++)out[(o*indices.length+j)*inner+i]=a.data[(o*dim+indices[j])*inner+i];return new Tensor(out,shape,a.dtype);}
  function where(mask,a,b){assert(mask instanceof Tensor&&mask.dtype==='float32','E_MASK_TYPE','where mask must be a Tensor.');if(!(a instanceof Tensor))a=Tensor.scalar(a);if(!(b instanceof Tensor))b=Tensor.scalar(b);const shape=broadcastShape(broadcastShape(mask.shape,a.shape),b.shape),m=broadcastTo(mask,shape),x=broadcastTo(a,shape),y=broadcastTo(b,shape),out=new Float32Array(m.size);for(let i=0;i<out.length;i++){assert(m.data[i]===0||m.data[i]===1,'E_MASK_VALUE','Mask values must be 0 or 1.');out[i]=m.data[i]?x.data[i]:y.data[i];}return new Tensor(out,shape);}
  function maskedFill(a,mask,value){assert(mask instanceof Tensor&&mask.dtype==='float32','E_MASK_TYPE','maskedFill mask must be a Tensor.');const m=broadcastTo(mask,a.shape),out=new Float32Array(a.size);for(let i=0;i<out.length;i++){assert(m.data[i]===0||m.data[i]===1,'E_MASK_VALUE','Mask values must be 0 or 1.');out[i]=m.data[i]?value:a.data[i];}return new Tensor(out,a.shape,a.dtype);}
  function cast(a,dtype){assert(DTYPES[dtype],'E_DTYPE','Unsupported dtype: '+dtype);return new Tensor(a.data,a.shape,dtype);}
  function finite(a) { for(const x of a.data)if(!Number.isFinite(x))return false;return true; }
  KNA.TensorEngine = Object.freeze({ version:'0.5.0', Tensor, sizeOf, stridesOf, sameShape, limits:Object.freeze({maxElements:MAX_ELEMENTS}), errors:Object.freeze({assert,error}), ops:Object.freeze({add,sub,mul,div,exp,log,tanh,relu,sqrt,sigmoid,silu,gelu,sum,mean,max,sumAxis,meanAxis,maxAxis,reshape,transpose,transpose2D,matmul,softmax:softmaxOp,logSoftmax,slice,concat,gather,where,maskedFill,cast,broadcastTo,finite}) });
})(typeof globalThis !== 'undefined' ? globalThis : this);
