import {requireInteger} from './backend.js';
export const grid=n=>[Math.ceil(n/64),1,1];

// Kernel orchestration and bounded, evictable weight residency. No tensor arithmetic.
export class Ops {
    constructor(backend,store,{cacheBytes=64*1024*1024,tileRows=256}={}) {
        this.b=backend; this.store=store;
        this.cacheBytes=requireInteger(cacheBytes,'weight cache bytes',0,Number.MAX_SAFE_INTEGER);
        this.tileRows=requireInteger(tileRows,'weight tile rows',1,65535);
        this.cache=new Map(); this.used=0; this.hits=0; this.misses=0; this.temporary=new Set();this.readyLayers=new Set();
    }
    // Every temporary output is completely written by its producing kernel.
    // Persistent recurrent/KV buffers use backend.alloc directly and stay zeroed.
    async alloc(dataOrLength,type='f32') { const b=await this.b.alloc(dataOrLength,type,{zero:false}); this.temporary.add(b); return b; }
    async release(...buffers) { for (const b of buffers) if (this.temporary.delete(b)) await this.b.free(b); }
    async clearTemporary() { if(this.b.flush)this.b.flush();else await this.b.idle();for (const b of this.temporary) await this.b.free(b); this.temporary.clear(); }
    tileSize(cols){return Math.min(this.tileRows,Math.max(1,Math.floor(16*1024*1024/(cols*4))));}
    readRows(name,start,rows){return this.store.readRows(name,start,rows,{compact:this.b.kind==='webgpu',packedScales:this.b.kind==='wasm'});}
    async prepareLayer(layer){if(!this.residency||!this.readyLayers.has(layer))await this.store.prefetchDenseLayer?.(layer);}
    finishLayer(layer){if(this.residency)this.readyLayers.add(layer);this.store.denseWindow=null;}
    async upload(t) {
        const planes=['data','codes','scales','offsets'].filter(key=>t[key]),w={...t,buffers:[],bytes:planes.reduce((n,key)=>n+t[key].byteLength,0)};
        try{for(const plane of planes){w[plane]=await this.b.alloc(t[plane],plane==='codes'?'u32':'f32');w.buffers.push(w[plane]);}}
        catch(error){for(const b of w.buffers)await this.b.free(b);throw error;}return w;
    }
    async weights(name,start,rows) {
        if(this.residency){const w=await this.residency.get(name,start,rows);if(w)return w;}
        const key=`${name}:${start}:${rows}`;
        if (this.cache.has(key)) {const w=this.cache.get(key); this.cache.delete(key); this.cache.set(key,w); this.hits++; return w;}
        this.misses++;
        const t=await this.readRows(name,start,rows);
        if(!t.packedScales)return this.cacheWeight(key,t);
        const {packedScales,scaleCount,...planes}=t,w=await this.cacheWeight(key,{...planes,scales:{allocateLength:scaleCount,byteLength:scaleCount*4}});
        const raw=await this.alloc(packedScales,'u32');
        await this.b.run('strata_expert_scales',{Blob:raw,Scales:w.scales,offset:0,count:scaleCount},grid(scaleCount));await this.release(raw);return w;
    }
    async cacheWeight(key,t) {
        const planes=['data','codes','scales','offsets'].filter(key=>t[key]);
        const w={...t,buffers:[],bytes:planes.reduce((n,key)=>n+t[key].byteLength,0)};
        // Retire old tiles before allocating their replacements, including on WASM
        // where the heap has a fixed upper bound. One fence covers all evictions.
        if(this.used+w.bytes>this.cacheBytes && this.cache.size){if(this.b.flush)this.b.flush();else await this.b.idle();}
        while (this.used+w.bytes>this.cacheBytes && this.cache.size) {
            const [oldKey,old]=this.cache.entries().next().value; this.cache.delete(oldKey);
            for(const b of old.buffers) await this.b.free(b); this.used-=old.bytes;
        }
        try {
            for(const plane of planes) {
                w[plane]=await this.b.alloc(t[plane].allocateLength??t[plane],plane==='codes'?'u32':'f32');w.buffers.push(w[plane]);
            }
        } catch(error) {for(const buffer of w.buffers)await this.b.free(buffer);throw error;}
        if(w.bytes<=this.cacheBytes) {this.cache.set(key,w);this.used+=w.bytes;}
        else for(const buffer of w.buffers)this.temporary.add(buffer);
        return w;
    }
    async packedExpert(layer,id,x,tokens=1,{output=null,mapping=null,outputOffset=0}={}) {
        const ex=this.store.manifest.experts,g=this.store.config,key=`${layer}:${id}`,resident=this.residency?.experts.get(key);
        let blob=resident?.rawBuffer,scales=null;const expanded=this.b.kind==='wasm';
        if(!blob){
            const cacheKey=`expert:${key}`;let w=this.cache.get(cacheKey);
            if(w){this.cache.delete(cacheKey);this.cache.set(cacheKey,w);this.hits++;}
            else{this.misses++;const {raw}=await this.store.readExpertBlob(layer,id),count=(raw.byteLength-ex.offsets.gate_up_scales)/2;w=await this.cacheWeight(cacheKey,{codes:raw,...(expanded?{scales:{allocateLength:count,byteLength:count*4}}:{})});
                if(expanded)await this.b.run('strata_expert_scales',{Blob:w.codes,Scales:w.scales,offset:ex.offsets.gate_up_scales/2,count},grid(count));}
            blob=w.codes;scales=w.scales;
        }else this.residency.hits++;
        this.dummy??=await this.b.alloc(new Int32Array(1),'i32');
        const map=mapping?await this.alloc(mapping,'i32'):this.dummy,hidden=await this.alloc(g.ff*tokens),result=output??await this.alloc(g.width*tokens);
        const scaleBase=expanded?ex.offsets.gate_up_scales/2:0;
        await this.b.run('strata_q2_gate_up',{X:x,Blob:blob,Scales:scales??this.dummy,Mapping:map,Hidden:hidden,width:g.width,ff:g.ff,tokens,mapped:mapping?1:0,expanded:expanded?1:0,codeOffset:ex.offsets.gate_up_codes/4,scaleOffset:ex.offsets.gate_up_scales/2-scaleBase},[g.ff,tokens,1]);
        await this.b.run('strata_q2_down',{Hidden:hidden,Blob:blob,Scales:scales??this.dummy,Mapping:map,Y:result,width:g.width,ff:g.ff,tokens,mapped:mapping?1:0,expanded:expanded?1:0,outputOffset,codeOffset:ex.offsets.down_codes/4,scaleOffset:ex.offsets.down_scales/2-scaleBase},[g.width,tokens,1]);
        await this.release(hidden,...(mapping?[map]:[]),blob,...(scales?[scales]:[]));return result;
    }
    async embedding(token,output,outputOffset=0){
        const g=this.store.config,w=await this.weights('token_embd.weight',token,1);this.dummy??=await this.b.alloc(new Int32Array(1),'i32');
        await this.b.run('strata_embedding',{Values:w.data??this.dummy,Codes:w.codes??this.dummy,Scales:w.scales??this.dummy,Offsets:w.offsets??this.dummy,Residual:output,
            width:g.width,streams:g.streams,outputOffset,format:w.format==='f32'?0:w.format==='bf16'?1:2,bits:w.bits??0,groupSize:w.groupSize??1,bias:w.bias??0,codebook:w.codebook??0,hasOffset:w.hasOffset?1:0},grid(g.width));
        await this.release(...w.buffers);
    }
    async ple(token,previous){
        if(!this.store.ple)return this.alloc(await this.store.pleRows(token,previous));
        const raw=await this.store.plePackedRows(token,previous),packed=await this.alloc(raw,'u32'),out=await this.alloc(this.store.config.width);
        await this.b.run('strata_ple_decode',{Packed:packed,Values:out,count:out.length},grid(out.length));await this.release(packed);return out;
    }
    async rotate(x,heads,position,tokens=1){
        const g=this.store.config,y=await this.alloc(x.length);
        if(!this.rope){
            this.rope={};this.rope.cos=await this.b.alloc(g.context*g.rotary/2);this.rope.sin=await this.b.alloc(g.context*g.rotary/2);
            const hi=await this.alloc(g.rotary/2),lo=await this.alloc(g.rotary/2);
            await this.b.run('strata_rope_frequencies',{High:hi,Low:lo,rotary:g.rotary,base:g.ropeBase},grid(g.rotary/2));
            await this.b.run('strata_rope_table',{High:hi,Low:lo,Cos:this.rope.cos,Sin:this.rope.sin,rotary:g.rotary,context:g.context},grid(g.context*g.rotary/2));
            await this.release(hi,lo);
        }
        await this.b.run('strata_rope_position',{X:x,Cos:this.rope.cos,Sin:this.rope.sin,Y:y,dim:g.headDim,heads,rotary:g.rotary,position,tokens},grid(x.length));return y;
    }
    async mat(name,x,tokens=1) {
        const {shape}=this.store.describe(name); const cols=shape[0], rows=shape.slice(1).reduce((a,b)=>a*b,1);
        if(x.length!==cols*tokens) throw Error(`${name}: expected ${cols*tokens} inputs, got ${x.length}`);
        const y=await this.alloc(rows*tokens);
        // Keep every tile below both WebGPU buffer and WASM heap limits.
        const tile=this.tileSize(cols);
        for(let start=0;start<rows;start+=tile) {
            const n=Math.min(tile,rows-start); const w=await this.weights(name,start,n);
            try {
                const layout={cols,rows:n,tokens,outputStride:rows,outputOffset:start};
                if(w.format==='f32') await this.b.run('strata_project',{X:x,W:w.data,Y:y,...layout},[n,tokens,1]);
                else if(w.format==='bf16')await this.b.run('strata_bf16_project',{X:x,Codes:w.codes,Y:y,...layout},[n,tokens,1]);
                else if(w.format==='q2blob')await this.b.run('strata_q2_project',{X:x,Blob:w.codes,Y:y,...layout,codeOffset:w.codeOffset,scaleOffset:w.scaleOffset,weightRow:w.weightRow,rowStride:w.rowStride},[n,tokens,1]);
                else await this.b.run('strata_quant_project',{X:x,Codes:w.codes,Scales:w.scales,Offsets:w.offsets,Y:y,...layout,bits:w.bits,groupSize:w.groupSize,bias:w.bias,codebook:w.codebook,hasOffset:w.hasOffset?1:0},[n,tokens,1]);
            } finally {await this.release(...w.buffers);}
        }
        return y;
    }
    async vector(name) {if(this.residency)return this.residency.vector(name); const d=this.store.describe(name); const n=d.shape.reduce((a,b)=>a*b,1); const a=await this.store.readValues(name,0,n); return this.alloc(a); }
    async norm(x,w,cols,rows=1,weightRows=1,mode=0,scale=1) {
        const y=await this.alloc(cols*rows);
        await this.b.run('strata_norm',{X:x,W:w,Y:y,cols,rows,weightRows,mode,epsilon:1e-6,scale},[rows,1,1]); return y;
    }
    async elem(a,b=a,mode=0,scale=1) {
        if(a.length!==b.length) throw Error('Elementwise input shapes disagree');
        const y=await this.alloc(a.length); await this.b.run('strata_elementwise',{A:a,B:b,Y:y,n:a.length,mode,scale},grid(a.length)); return y;
    }
    async slice(x,width,rows=1,stride=width,offset=0) {
        requireInteger(width,'slice width'); requireInteger(rows,'slice rows');
        if(offset<0||(rows-1)*stride+offset+width>x.length) throw Error('Slice exceeds buffer');
        const y=await this.alloc(width*rows); await this.b.run('strata_slice',{X:x,Y:y,width,rows,stride,offset},grid(width*rows)); return y;
    }
    async dispose() { await this.clearTemporary();await this.residency?.dispose();for(const w of this.cache.values()) for(const b of w.buffers) await this.b.free(b);for(const buffer of Object.values(this.rope??{}))await this.b.free(buffer);this.rope=null;if(this.dummy)await this.b.free(this.dummy);this.dummy=null;this.cache.clear(); this.used=0; }
}
