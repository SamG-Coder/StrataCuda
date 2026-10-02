import test from 'node:test';
import assert from 'node:assert/strict';
import {Ops} from '../src/ops.js';
import {StrataEngine} from '../src/engine.js';
import {WorkerBackend} from '../src/backend.js';
import {createFixture} from '../src/fixture.js';
import {logitBars} from '../web/logits.js';
import {attention} from './reference.mjs';

const deferred=()=>{let resolve;const promise=new Promise(r=>{resolve=r;});return {promise,resolve};};
const tick=()=>new Promise(r=>setImmediate(r));

// A bounded allocator and controllable queues exercise orchestration, not model math.
// Numerical correctness is covered separately by the real GPU/WASM conformance suite.
class TestBackend {
    constructor({capacity=Infinity,failAllocation=0,kind='webgpu'}={}) {
        Object.assign(this,{capacity,failAllocation,kind});this.buffers=new Set();this.used=0;this.allocations=0;
    }
    alloc(input,type='f32') {
        if(++this.allocations===this.failAllocation)throw Error('injected allocation failure');
        const Type=type==='i32'?Int32Array:type==='u32'?Uint32Array:Float32Array;
        const data=typeof input==='number'?new Type(input):new Type(input);
        if(this.used+data.byteLength>this.capacity)throw Error('allocation exceeds capacity');
        const b={data,type,length:data.length};this.buffers.add(b);this.used+=data.byteLength;return b;
    }
    async read(b){if(this.readGate){const gate=this.readGate;this.readGate=null;gate.entered.resolve();await gate.release.promise;}return b.data.slice();}
    async write(b,a,offset=0){if(this.writeGate){const gate=this.writeGate;this.writeGate=null;gate.entered.resolve();await gate.release.promise;}b.data.set(a,offset);}
    copy(a,b,n=a.length,from=0,to=0){b.data.set(a.data.subarray(from,from+n),to);}
    run(name,args){if(name==='strata_router')for(let i=0;i<args.topK;i++)args.Ids.data[i]=i;}
    free(b){if(this.buffers.delete(b))this.used-=b.data.byteLength;}
    async idle(){}
}

test('weight eviction happens before an allocation at the memory limit',async()=>{
    const b=new TestBackend({capacity:32}),store={readRows:async()=>({format:'f32',data:new Float32Array(8)})},ops=new Ops(b,store,{cacheBytes:32});
    try {await ops.weights('first',0,1);await ops.weights('second',0,1);assert.equal(b.used,32);assert.equal(ops.used,32);}
    finally {await ops.dispose();}
    assert.equal(b.used,0);
});

test('failed multi-plane uploads release every partially allocated buffer',async()=>{
    const b=new TestBackend({failAllocation:2}),store={readRows:async()=>({format:'s',codes:new Uint32Array(8),scales:new Float32Array(1),offsets:new Float32Array(1)})},ops=new Ops(b,store);
    await assert.rejects(()=>ops.weights('quantized',0,1),/injected/);
    assert.equal(b.used,0,'a failed upload must not retain its code plane');await ops.dispose();
});

test('a tile larger than the cache stays temporary and is released',async()=>{
    const b=new TestBackend(),store={readRows:async()=>({format:'f32',data:new Float32Array(8)})},ops=new Ops(b,store,{cacheBytes:4});
    try {await ops.weights('large',0,1);assert.equal(ops.used,0);assert.equal(ops.cache.size,0);await ops.clearTemporary();assert.equal(b.used,0);}
    finally {await ops.dispose();}
});

test('hybrid failures wait for the other lane before releasing the session',async()=>{
    const b=new TestBackend(),cpu=new TestBackend({kind:'wasm'}),engine=await StrataEngine.create(createFixture(),b,{cpuBackend:cpu});
    engine.hot.set('0:0',true);const releaseGPU=deferred(),cpuStarted=deferred();
    engine.expert=async(ops)=>{if(ops===engine.cpuOps){cpuStarted.resolve();throw Error('injected expert read failure');}await releaseGPU.promise;return ops.alloc(engine.g.width);};
    let settled=false;const work=engine.step(2).then(()=>{settled=true;return null;},error=>{settled=true;return error;});
    try {
        await cpuStarted.promise;await tick();assert.equal(settled,false,'the GPU lane is still active');
        await assert.rejects(()=>engine.reset(),/decode step/);
    } finally {releaseGPU.resolve();const error=await work;assert.match(error.message,/injected expert/);await engine.reset();await engine.dispose();}
    assert.equal(b.used,0);assert.equal(cpu.used,0);
});

test('GPU expert selection survives a full scan larger than its slot count',async()=>{
    const b=new TestBackend(),cpu=new TestBackend({kind:'wasm'}),engine=await StrataEngine.create(createFixture(),b,{cpuBackend:cpu,gpuExperts:2});
    try {for(let i=0;i<3;i++)await engine.step(2);assert.ok(engine.stats.gpuExperts>=2,'recurring experts should reach the GPU despite the layer scan');assert.ok(engine.stats.cpuExperts>0);assert.ok(engine.hot.size<=2);}
    finally {await engine.dispose();}
});

for(const operation of ['checkpoint','reset'])test(operation+' excludes decode until state I/O finishes',async()=>{
    const b=new TestBackend(),engine=await StrataEngine.create(createFixture(),b),gate={entered:deferred(),release:deferred()};
    b[operation==='checkpoint'?'readGate':'writeGate']=gate;
    const work=engine[operation]();
    try {await gate.entered.promise;await assert.rejects(()=>engine.step(2),/already running|in progress/);}
    finally {gate.release.resolve();await work;await engine.dispose();}
});

class TestWorker {postMessage(){}terminate(){this.terminated=true;}}
const timeout=async(p)=>{let timer;try{return await Promise.race([p,new Promise((_,reject)=>{timer=setTimeout(()=>reject(Error('RPC remained pending after worker failure')),100);})]);}finally{clearTimeout(timer);}};

test('worker crashes reject future RPC and disposal still terminates the worker',async()=>{
    const worker=new TestWorker(),backend=new WorkerBackend(worker),pending=backend.call('read');
    worker.onerror({message:'injected worker crash'});await assert.rejects(()=>pending,/injected worker crash/);
    await assert.rejects(()=>timeout(backend.call('info')),/injected worker crash/);
    await timeout(backend.dispose());assert.equal(worker.terminated,true);
});

test('an uncloneable worker request does not leak a pending RPC',async()=>{
    const worker=new TestWorker(),backend=new WorkerBackend(worker);
    worker.postMessage=()=>{throw Error('injected clone failure');};
    await assert.rejects(()=>backend.call('write'),/clone failure/);assert.equal(backend.pending.size,0);
});

test('production vocabulary visualization stays bounded and shows the selected token',()=>{
    const values=new Float32Array(248320);values[248319]=9;values[100]=-4;
    const {bars,min,max}=logitBars(values,248319);
    assert.equal(min,-4);assert.equal(max,9);assert.ok(bars.length<=120);
    assert.deepEqual(bars.filter(b=>b.best).map(b=>b.id),[248319]);
    assert.equal(bars.at(-1).height,95);
});

test('prompt-only steps update the session without loading the vocabulary projection',async()=>{
    const store=createFixture(),readRows=store.readRows.bind(store);let outputReads=0;
    store.readRows=async(name,...args)=>{if(name==='output.weight')outputReads++;return readRows(name,...args);};
    const b=new TestBackend(),engine=await StrataEngine.create(store,b);
    try {
        assert.equal((await engine.step(2,{predict:false})).token,undefined);await engine.step(7,{predict:false});
        assert.equal(engine.position,2);assert.equal(outputReads,0);
        await engine.step(4);assert.equal(engine.position,3);assert.ok(outputReads>0);
    } finally {await engine.dispose();}
});

test('the attention oracle keeps fractional scores for typed cell indices',()=>{
    const out=attention(new Float32Array([1]),new Float32Array([0,.5]),new Float32Array([0,1]),new Int32Array([0,1]),1,1,1);
    assert.ok(Math.abs(out[0]-Math.exp(.5)/(1+Math.exp(.5)))<1e-12);
});

test('aborting after a layer rejects reuse of partial state and frees all allocations',async()=>{
    const b=new TestBackend(),engine=await StrataEngine.create(createFixture(),b),controller=new AbortController();let progress=0;
    await assert.rejects(engine.step(2,{signal:controller.signal,onProgress:layer=>{progress=layer;controller.abort();}}),{name:'AbortError'});
    assert.equal(progress,1);assert.equal(engine.position,0);assert.equal(engine.busy,false);assert.equal(engine.failed,true);
    await assert.rejects(engine.step(2),/Session failed/);await engine.dispose();assert.equal(b.used,0);
});
