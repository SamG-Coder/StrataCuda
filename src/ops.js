import {requireInteger} from './backend.js';
export const grid=n=>[Math.ceil(n/64),1,1];

// Kernel orchestration and bounded, evictable weight residency. No tensor arithmetic.
export class Ops {
    constructor(backend,store,{cacheBytes=64*1024*1024,tileRows=256}={}) {
        this.b=backend; this.store=store;
        this.cacheBytes=requireInteger(cacheBytes,'weight cache bytes',0,Number.MAX_SAFE_INTEGER);
        this.tileRows=requireInteger(tileRows,'weight tile rows',1,65535);
        this.cache=new Map(); this.used=0; this.hits=0; this.misses=0; this.temporary=new Set();
    }
    // Every temporary output is completely written by its producing kernel.
    // Persistent recurrent/KV buffers use backend.alloc directly and stay zeroed.
    async alloc(dataOrLength,type='f32') { const b=await this.b.alloc(dataOrLength,type,{zero:false}); this.temporary.add(b); return b; }
    async release(...buffers) { for (const b of buffers) if (this.temporary.delete(b)) await this.b.free(b); }
    async clearTemporary() { if(this.b.flush)this.b.flush();else await this.b.idle();for (const b of this.temporary) await this.b.free(b); this.temporary.clear(); }
    tileSize(cols){return Math.min(this.tileRows,Math.max(1,Math.floor(16*1024*1024/(cols*4))));}
    readRows(name,start,rows){return this.store.readRows(name,start,rows,{compact:this.b.kind==='webgpu'});}
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
                w[plane]=await this.b.alloc(t[plane],plane==='codes'?'u32':'f32');w.buffers.push(w[plane]);
            }
        } catch(error) {for(const buffer of w.buffers)await this.b.free(buffer);throw error;}
        if(w.bytes<=this.cacheBytes) {this.cache.set(key,w);this.used+=w.bytes;}
        else for(const buffer of w.buffers)this.temporary.add(buffer);
        return w;
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
    async dispose() { await this.clearTemporary();await this.residency?.dispose();for(const w of this.cache.values()) for(const b of w.buffers) await this.b.free(b); this.cache.clear(); this.used=0; }
}
