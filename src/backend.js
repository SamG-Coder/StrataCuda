// Host glue only. All model arithmetic executes in kernels/strata.cu.
import {GpuRuntime} from '../vendor/webcuda/src/runtime/runtime.js';
import {ThreadedProgram} from '../vendor/webcuda/src/wasm/runtime.js';

const typed = {f32:Float32Array, i32:Int32Array, u32:Uint32Array};
export function requireInteger(n, label, min=1, max=0x7fffffff) {
    if (!Number.isSafeInteger(n) || n < min || n > max) throw new RangeError(`${label} must be an integer in [${min}, ${max}]`);
    return n;
}
export function validateDispatch(name, args) {
    const get = k => requireInteger(args[k], name + '.' + k);
    if (name === 'strata_router'||name==='strata_router_batch') {
        if (get('experts') > 512 || get('topK') > Math.min(args.experts,32)) throw new RangeError('Router supports 1–512 experts and topK <= min(experts,32)');
    }
    if (name === 'strata_quant_gemv' || name === 'strata_quant_project') {
        if (![2,4,8].includes(args.bits)) throw new RangeError('Canonical code width must be 2, 4 or 8');
        if (get('cols') % get('groupSize') || args.cols * args.rows * args.bits > 0x7fffffff) throw new RangeError('Quantized rows must contain whole groups and fit 31-bit indexing');
    }
    if(['strata_project','strata_quant_project','strata_bf16_project','strata_q2_project'].includes(name)) {
        get('tokens');get('rows');get('cols');get('outputStride');requireInteger(args.outputOffset,'projection output offset',0);
        if(args.outputOffset+args.rows>args.outputStride||args.tokens*args.outputStride>args.Y.length||args.tokens*args.cols>args.X.length)throw Error('Projection batch exceeds buffer layout');
    }
    if(name==='strata_q2_project'){
        if(args.cols%64)throw Error('Q2 expert columns must contain complete groups');
        for(const key of ['codeOffset','scaleOffset','weightRow'])requireInteger(args[key],key,0);
        requireInteger(args.rowStride,'expert row stride',1,2);
    }
    if(name==='strata_q2_gate_up'||name==='strata_q2_down'){
        if(get('width')%64||get('ff')%64)throw Error('Packed expert dimensions must contain complete groups');
        const tokens=get('tokens');if(args.Hidden.length<tokens*args.ff||args.mapped&&args.Mapping.length<tokens*2)throw Error('Packed expert batch exceeds buffer layout');
        if(name==='strata_q2_down')requireInteger(args.outputOffset,'expert output offset',0);
    }
    if(name==='strata_embedding'){
        const count=get('width')*get('streams');requireInteger(args.outputOffset,'embedding output offset',0);
        if(args.outputOffset+count>args.Residual.length||![0,1,2].includes(args.format))throw Error('Embedding exceeds buffer layout');
        if(args.format===2&&(![2,4,8].includes(args.bits)||args.width%get('groupSize')))throw Error('Invalid packed embedding layout');
    }
    if(name==='strata_ple_decode'&&(get('count')%32||Math.ceil(args.count/32*18/4)>args.Packed.length||args.count>args.Values.length))throw Error('Invalid packed PLE layout');
    if(name==='strata_rope_position'){
        requireInteger(args.position,'RoPE position',0,2047);
        if(get('rotary')%2||args.rotary>get('dim')||args.position+get('tokens')>2048||args.Cos.length<(args.position+args.tokens)*args.rotary/2||args.Sin.length!==args.Cos.length||args.X.length!==args.dim*get('heads')*args.tokens||args.Y.length!==args.X.length)throw Error('Invalid RoPE layout');
    }
    if(name==='strata_rope_frequencies'&&(!Number.isFinite(args.base)||args.base<=1||get('rotary')%2))throw Error('Invalid RoPE frequencies');
    if (name === 'strata_attention') {
        if(get('heads') % get('kvHeads'))throw new RangeError('Attention heads must be divisible by KV heads');
        requireInteger(args.count,'attention selected cells',1,2048);
    }
    if(name==='strata_attention_batch') {
        if(get('heads')%get('kvHeads'))throw Error('Attention heads must be divisible by KV heads');
        requireInteger(args.position,'attention position',0,2047);
        const tokens=args.Q.length/(get('heads')*get('dim'));
        if(!Number.isInteger(tokens)||args.position+tokens>2048)throw Error('Batched attention exceeds 2048 cells');
    }
    if (name === 'strata_rope' && (get('rotary') % 2 || args.rotary > get('dim'))) throw new RangeError('Rotary width must be even and <= head dimension');
    if (name === 'strata_norm' && get('rows') % get('weightRows')) throw new RangeError('Normalization weight rows must divide input rows');
}

export class GpuBackend {
    static async create({base=new URL('../generated/',import.meta.url), onError, batch=true}={}) {
        const runtime = await GpuRuntime.create({useAdapterBufferLimits:true, onError});
        const manifest = await (await fetch(new URL('manifest.json',base))).json();
        const kernels = new Map();
        try {
            for (const {entry} of manifest.kernels) {
                const response = await fetch(new URL(entry + '.json',base));
                if (!response.ok) throw Error('Missing generated kernel: ' + entry);
                kernels.set(entry,await runtime.kernel(await response.json()));
            }
            return new GpuBackend(runtime,kernels,manifest,{batch});
        } catch (error) { runtime.dispose(); throw error; }
    }
    constructor(runtime,kernels,manifest,{batch=true}={}) {
        this.runtime=runtime;this.kernels=kernels;this.manifest=manifest;this.kind='webgpu';this.buffers=new Set();this.batching=batch;this.pendingBatch=null;this.retired=new Set();
        this.pool=new Map();this.poolBytes=0;this.poolBudget=128*1024**2;this.bufferSequence=0;this.bindings=new Map();this.poolHits=0;this.bindingHits=0;
    }
    commandBatch() {
        if(this.pendingBatch&&(this.pendingBatch.dispatchCount>=128||this.pendingBatch.cursor+this.runtime.uniformAlignment*2>this.runtime.uniformCapacity))this.flush();
        return this.pendingBatch??=this.runtime.batch();
    }
    flush() {
        if(this.pendingBatch){this.pendingBatch.submit();this.pendingBatch=null;}
        for(const buffer of this.retired)this.runtime.destroyBuffer(buffer);this.retired.clear();
    }
    alloc(dataOrLength,type='f32',{zero=true}={}) {
        if(!typed[type])throw Error('Unsupported buffer type '+type);
        // WebGPU initializes new buffers to zero; avoid a duplicate CPU allocation
        // and upload for every state and output buffer.
        const data = typeof dataOrLength === 'number' ? requireInteger(dataOrLength,'allocation')*4 : dataOrLength;
        const key=typeof data==='number'?`${type}:${data}`:null,available=key?this.pool.get(key):null;
        if(available?.length){const handle=available.pop();this.poolBytes-=handle.byteLength;this.poolHits++;this.buffers.add(handle);if(zero)this.zero(handle);return handle;}
        const handle = this.runtime.createBuffer(data);handle.poolKey=key;handle.bindingId=++this.bufferSequence;handle.type=type;handle.length=(typeof data==='number'?data:data.byteLength)/4;this.buffers.add(handle);return handle;
    }
    write(buffer,data,offset=0) { this.flush();this.runtime.write(buffer,data,offset*4); }
    zero(buffer){this.commandBatch().clear(buffer);if(!this.batching)this.flush();}
    async read(buffer,type=buffer.type) { this.flush();return this.runtime.read(buffer,typed[type]); }
    copy(source,destination,count=source.length,sourceOffset=0,destinationOffset=0) {
        this.runtime.assertAlive();
        this.commandBatch().copy(source,destination,{sourceOffset:sourceOffset*4,targetOffset:destinationOffset*4,byteLength:count*4});
        if(!this.batching)this.flush();
    }
    run(name,args,groups) {
        validateDispatch(name,args);
        const kernel = this.kernels.get(name); if (!kernel) throw Error('Unknown kernel ' + name);
        const buffers={},scalars={};
        for (const [key,value] of Object.entries(args)) (typeof value === 'number' ? scalars : buffers)[key]=value;
        const key=name+':'+Object.entries(buffers).map(([key,value])=>key+'='+value.bindingId).join(','),cached=this.bindings.get(key);
        let invocation;
        if(cached&&Object.values(cached.buffers).every(b=>!b.destroyed)){invocation=cached.setScalars(scalars);this.bindingHits++;this.bindings.delete(key);}
        else invocation=kernel.bind(buffers,scalars);
        this.bindings.set(key,invocation);if(this.bindings.size>4096)this.bindings.delete(this.bindings.keys().next().value);
        this.commandBatch().dispatch(invocation,groups);
        if(!this.batching)this.flush();
    }
    free(buffer) {
        if(!this.buffers.delete(buffer))return;
        if(buffer.poolKey&&this.poolBytes+buffer.byteLength<=this.poolBudget){if(!this.pool.has(buffer.poolKey))this.pool.set(buffer.poolKey,[]);this.pool.get(buffer.poolKey).push(buffer);this.poolBytes+=buffer.byteLength;return;}
        if(this.pendingBatch)this.retired.add(buffer);else this.runtime.destroyBuffer(buffer);
    }
    async idle() { this.flush();await this.runtime.idle(); }
    info() { return {backend:this.kind,...this.runtime.describe(),...this.runtime.stats,pooledBytes:this.poolBytes,poolHits:this.poolHits,bindingCacheHits:this.bindingHits}; }
    async dispose() {try{await this.idle();}finally{this.pendingBatch?.discard();this.pendingBatch=null;this.retired.clear();this.bindings.clear();this.pool.clear();this.poolBytes=0;this.runtime.dispose();this.buffers.clear();} }
}

export class WasmBackend {
    constructor(module,abi,threads=4) {
        this.program=new ThreadedProgram(module,abi,threads); this.kind='wasm'; this.buffers=new Set(); this.dispatches=0;this.groupTotals=new Array(threads).fill(0);
    }
    alloc(dataOrLength,type='f32') {
        if(!typed[type])throw Error('Unsupported buffer type '+type);
        const bytes=typeof dataOrLength==='number'?requireInteger(dataOrLength,'allocation')*4:dataOrLength.byteLength;
        const b=this.program.alloc(bytes);b.type=type;b.length=bytes/4;
        // ThreadedProgram.alloc already zeroes its heap range.
        if(typeof dataOrLength!=='number')this.program.write(b,dataOrLength);this.buffers.add(b);return b;
    }
    write(b,data,offset=0) {
        if (!this.buffers.has(b) || offset<0 || offset*4+data.byteLength>b.bytes) throw Error('Invalid WASM write');
        this.program.module.HEAPU8.set(new Uint8Array(data.buffer,data.byteOffset,data.byteLength),b.ptr+offset*4);
    }
    read(b,type=b.type) { if (!this.buffers.has(b)) throw Error('Invalid WASM read'); const a=this.program.read(b); return new typed[type](a.buffer,a.byteOffset,a.byteLength/4); }
    copy(a,b,count=a.length,aOffset=0,bOffset=0) {
        if (!this.buffers.has(a)||!this.buffers.has(b)||aOffset<0||bOffset<0||count+aOffset>a.length||count+bOffset>b.length) throw Error('Invalid WASM copy');
        this.program.module.HEAPU8.copyWithin(b.ptr+bOffset*4,a.ptr+aOffset*4,a.ptr+(aOffset+count)*4);
    }
    run(name,args,groups) { validateDispatch(name,args); this.program.dispatch(name,groups,args); this.dispatches++;this.program.groups().forEach((n,i)=>this.groupTotals[i]+=n); }
    free(b) {
        if (!this.buffers.delete(b)) return;
        this.program.module._free(b.ptr); this.program.buffers.splice(this.program.buffers.indexOf(b),1);
    }
    async idle() {}
    info() { return {backend:'wasm',threads:this.program.threads,dispatches:this.dispatches,workerGroups:[...this.groupTotals],heapBytes:this.program.module.HEAPU8.length}; }
    async dispose() { this.program.dispose(); this.buffers.clear(); }
}

// One ordered RPC stream per WASM module; dispatches on that module never overlap.
export class WorkerBackend {
    static async create({threads=4}={}) {
        const b=new WorkerBackend(new Worker(new URL('./wasm-worker.js',import.meta.url),{type:'module'}));
        try {await b.call('init',{threads});return b;}
        catch(error) {b.closed=true;b.fail(error);b.worker.terminate();throw error;}
    }
    constructor(worker) {
        this.worker=worker; this.pending=new Map(); this.sequence=0; this.kind='wasm';this.closed=false;this.failure=null;
        this.commands=[];this.commandBytes=0;this.nextBuffer=0;this.lastBatch=Promise.resolve();
        worker.onmessage=({data})=>{ const p=this.pending.get(data.id); if (!p) return; this.pending.delete(data.id); data.error ? p.reject(Error(data.error)) : p.resolve(data.value); };
        worker.onerror=e=>this.fail(Error(e.message||'WASM worker failed'));
        worker.onmessageerror=()=>this.fail(Error('Cannot decode WASM worker response'));
    }
    fail(error) {this.failure??=error;for(const p of this.pending.values())p.reject(this.failure);this.pending.clear();}
    call(method,args={}) {
        if(this.closed||this.failure)return Promise.reject(this.failure||Error('WASM worker is disposed'));
        return new Promise((resolve,reject)=>{
            const id=++this.sequence;this.pending.set(id,{resolve,reject});
            try {this.worker.postMessage({id,method,args});}
            catch(error) {this.pending.delete(id);reject(error);}
        });
    }
    enqueue(method,args,bytes=0){
        if(this.closed||this.failure)throw this.failure||Error('WASM worker is disposed');
        this.commands.push({method,args});this.commandBytes+=bytes;
        if(this.commands.length>=64||this.commandBytes>=8*1024**2)this.submitCommands();
    }
    submitCommands(){
        if(this.commands.length){const commands=this.commands;this.commands=[];this.commandBytes=0;this.lastBatch=this.call('batch',{commands});this.lastBatch.catch(error=>this.fail(error));}
        return this.lastBatch;
    }
    alloc(dataOrLength,type='f32') {
        if(!typed[type])throw Error('Unsupported buffer type '+type);
        const length=typeof dataOrLength==='number'?requireInteger(dataOrLength,'allocation'):dataOrLength.byteLength/4;
        // Structured clone copies an entire backing ArrayBuffer, even for a tiny
        // typed view into a bulk layer read. Send only the requested weight tile.
        if(typeof dataOrLength!=='number'&&dataOrLength.byteLength!==dataOrLength.buffer.byteLength)dataOrLength=dataOrLength.slice();
        const handle={id:++this.nextBuffer,type,length};this.enqueue('alloc',{dataOrLength,type,handle},typeof dataOrLength==='number'?0:dataOrLength.byteLength);return handle;
    }
    write(buffer,data,offset=0) { if(data.byteLength!==data.buffer.byteLength)data=data.slice();this.enqueue('write',{buffer,data,offset},data.byteLength); }
    async read(buffer,type=buffer.type) { await this.submitCommands();return this.call('read',{buffer,type}); }
    copy(source,destination,count=source.length,sourceOffset=0,destinationOffset=0) { this.enqueue('copy',{source,destination,count,sourceOffset,destinationOffset}); }
    run(name,args,groups) { validateDispatch(name,args);this.enqueue('run',{name,args,groups}); }
    free(buffer) { this.enqueue('free',{buffer}); }
    async idle() { await this.submitCommands();return this.call('idle'); }
    async info() { await this.submitCommands();return this.call('info'); }
    async dispose() {
        if(this.closed)return;
        try {if(!this.failure){await this.submitCommands();await this.call('dispose');}}
        finally {this.closed=true;this.fail(Error('WASM worker is disposed'));this.worker.terminate();}
    }
}
