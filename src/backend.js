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
    if (name === 'strata_router') {
        if (get('experts') > 512 || get('topK') > Math.min(args.experts,32)) throw new RangeError('Router supports 1–512 experts and topK <= min(experts,32)');
    }
    if (name === 'strata_quant_gemv') {
        if (![2,4,8].includes(args.bits)) throw new RangeError('Canonical code width must be 2, 4 or 8');
        if (get('cols') % get('groupSize') || args.cols * args.rows * args.bits > 0x7fffffff) throw new RangeError('Quantized rows must contain whole groups and fit 31-bit indexing');
    }
    if (name === 'strata_attention') {
        if(get('heads') % get('kvHeads'))throw new RangeError('Attention heads must be divisible by KV heads');
        requireInteger(args.count,'attention selected cells',1,2048);
    }
    if (name === 'strata_rope' && (get('rotary') % 2 || args.rotary > get('dim'))) throw new RangeError('Rotary width must be even and <= head dimension');
    if (name === 'strata_norm' && get('rows') % get('weightRows')) throw new RangeError('Normalization weight rows must divide input rows');
}

export class GpuBackend {
    static async create({base=new URL('../generated/',import.meta.url), onError}={}) {
        const runtime = await GpuRuntime.create({useAdapterBufferLimits:true, onError});
        const manifest = await (await fetch(new URL('manifest.json',base))).json();
        const kernels = new Map();
        try {
            for (const {entry} of manifest.kernels) {
                const response = await fetch(new URL(entry + '.json',base));
                if (!response.ok) throw Error('Missing generated kernel: ' + entry);
                kernels.set(entry,await runtime.kernel(await response.json()));
            }
            return new GpuBackend(runtime,kernels,manifest);
        } catch (error) { runtime.dispose(); throw error; }
    }
    constructor(runtime,kernels,manifest) { this.runtime=runtime; this.kernels=kernels; this.manifest=manifest; this.kind='webgpu'; this.buffers=new Set(); }
    alloc(dataOrLength,type='f32') {
        if(!typed[type])throw Error('Unsupported buffer type '+type);
        // WebGPU initializes new buffers to zero; avoid a duplicate CPU allocation
        // and upload for every state and output buffer.
        const data = typeof dataOrLength === 'number' ? requireInteger(dataOrLength,'allocation')*4 : dataOrLength;
        const handle = this.runtime.createBuffer(data); handle.type=type; handle.length=(typeof data==='number'?data:data.byteLength)/4; this.buffers.add(handle); return handle;
    }
    write(buffer,data,offset=0) { this.runtime.write(buffer,data,offset*4); }
    async read(buffer,type=buffer.type) { return this.runtime.read(buffer,typed[type]); }
    copy(source,destination,count=source.length,sourceOffset=0,destinationOffset=0) {
        this.runtime.assertAlive();
        const encoder = this.runtime.device.createCommandEncoder();
        encoder.copyBufferToBuffer(source.gpuBuffer,sourceOffset*4,destination.gpuBuffer,destinationOffset*4,count*4);
        this.runtime.device.queue.submit([encoder.finish()]);
    }
    run(name,args,groups) {
        validateDispatch(name,args);
        const kernel = this.kernels.get(name); if (!kernel) throw Error('Unknown kernel ' + name);
        const buffers={},scalars={};
        for (const [key,value] of Object.entries(args)) (typeof value === 'number' ? scalars : buffers)[key]=value;
        this.runtime.batch().dispatch(kernel.bind(buffers,scalars),groups).submit();
    }
    free(buffer) { if (this.buffers.delete(buffer)) this.runtime.destroyBuffer(buffer); }
    async idle() { await this.runtime.idle(); }
    info() { return {backend:this.kind,...this.runtime.describe(),...this.runtime.stats}; }
    async dispose() { await this.idle(); this.runtime.dispose(); this.buffers.clear(); }
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
    async alloc(dataOrLength,type='f32') { return this.call('alloc',{dataOrLength,type}); }
    write(buffer,data,offset=0) { return this.call('write',{buffer,data,offset}); }
    read(buffer,type=buffer.type) { return this.call('read',{buffer,type}); }
    copy(source,destination,count=source.length,sourceOffset=0,destinationOffset=0) { return this.call('copy',{source,destination,count,sourceOffset,destinationOffset}); }
    run(name,args,groups) { validateDispatch(name,args); return this.call('run',{name,args,groups}); }
    free(buffer) { return this.call('free',{buffer}); }
    idle() { return this.call('idle'); }
    info() { return this.call('info'); }
    async dispose() {
        if(this.closed)return;
        try {if(!this.failure)await this.call('dispose');}
        finally {this.closed=true;this.fail(Error('WASM worker is disposed'));this.worker.terminate();}
    }
}
