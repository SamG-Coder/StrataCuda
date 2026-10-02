import {requireInteger} from './backend.js';

const product=shape=>shape.reduce((a,b)=>a*b,1);
export const expertKey=name=>/^blk\.(\d+)\.expert\.(\d+)\./.exec(name)?.slice(1).join(':')??null;
export function residentBytes(store,name) {
    const t=store.describe(name),n=product(t.shape);
    if(t.expert)return n/4+n/64*4+4;
    if(t.values||t.format==='f32')return n*4;
    return Math.ceil(n*t.code_bits/8/4)*4+n/t.group_elems*4+(t.has_offset?n/t.group_elems*4:4);
}

// Dense tensors are never evicted by an expert scan. Expert entries own all
// three projections, so a scheduling hit means the entire expert is resident.
export class WeightResidency {
    constructor(ops,denseNames,budgetBytes) {
        this.ops=ops;this.b=ops.b;this.store=ops.store;
        this.budgetBytes=requireInteger(budgetBytes,'GPU weight budget',1,Number.MAX_SAFE_INTEGER);
        this.denseNames=new Set(denseNames);this.denseBytes=[...this.denseNames].reduce((n,name)=>n+this.bytesFor(name),0);
        this.expertBudget=budgetBytes-this.denseBytes;
        if(this.expertBudget<0)throw Error(`Resident weights need ${(this.denseBytes/1024**3).toFixed(2)} GiB; choose a larger GPU budget or streaming mode.`);
        this.tiles=new Map();this.experts=new Map();this.expertUsed=0;this.denseUsed=0;this.frequency=new Map();this.hits=0;this.misses=0;this.evictions=0;this.pending=new Map();
    }
    bytesFor(name){const t=this.store.describe(name),rows=product(t.shape)/t.shape[0],dummy=t.expert||t.codes&&!t.has_offset,vector=/norm|conv1d|ssm_a$|ssm_dt/.test(name);if(!vector&&t.source_type==='BF16'&&t.values&&!t.values_fp16)return Math.ceil(product(t.shape)/2)*4;return residentBytes(this.store,name)+(dummy?4*(Math.ceil(rows/this.ops.tileSize(t.shape[0]))-1):0);}
    tileKey(name,start,rows){return `${name}:${start}:${rows}`;}
    hasExpert(key){return this.experts.has(key);}
    async get(name,start,rows) {
        const key=this.tileKey(name,start,rows),cached=this.tiles.get(key);
        if(cached){this.hits++;return cached;}
        if(!this.denseNames.has(name))return null;
        this.misses++;const w=await this.ops.upload(await this.ops.readRows(name,start,rows));
        this.tiles.set(key,w);this.denseUsed+=w.bytes;return w;
    }
    async vector(name) {
        const key=`vector:${name}`,cached=this.tiles.get(key);if(cached){this.hits++;return cached.data;}
        const n=product(this.store.describe(name).shape),w=await this.ops.upload({format:'f32',data:await this.store.readValues(name,0,n)});
        this.misses++;this.tiles.set(key,w);this.denseUsed+=w.bytes;return w.data;
    }
    note(key,count=1){this.frequency.set(key,(this.frequency.get(key)||0)+count);}
    async dropExpert(key) {
        const entry=this.experts.get(key);if(!entry)return;
        if(entry.rawBuffer&&this.store.expertCacheBudget){const words=await this.b.read(entry.rawBuffer,'u32'),[layer,id]=key.split(':').map(Number);this.store.retainExpertBlob(layer,id,new Uint8Array(words.buffer,words.byteOffset,words.byteLength));}
        this.b.flush?.();
        for(const tile of entry.tiles){const w=this.tiles.get(tile);if(w){for(const buffer of w.buffers)await this.b.free(buffer);this.tiles.delete(tile);}}
        this.expertUsed-=entry.bytes;this.experts.delete(key);this.evictions++;
    }
    async admit(key,{force=false}={}) {
        if(this.experts.has(key))return true;
        if(this.pending.has(key))return this.pending.get(key);
        const work=this.loadExpert(key,force);this.pending.set(key,work);
        try{return await work;}finally{this.pending.delete(key);}
    }
    async loadExpert(key,force) {
        const [layer,id]=key.split(':').map(Number),names=['gate','up','down'].map(role=>`blk.${layer}.expert.${id}.${role}`);
        const compact=this.store.describe(names[0]).expert&&this.store.readExpertBlob;
        const bytes=compact?this.store.manifest.experts.blob_bytes:names.reduce((n,name)=>n+this.bytesFor(name),0);
        if(bytes>this.expertBudget)return false;
        const victims=this.expertUsed+bytes>this.expertBudget?[...this.experts].sort((a,b)=>(this.frequency.get(a[0])||0)-(this.frequency.get(b[0])||0)):[];
        let available=this.expertBudget-this.expertUsed;const remove=[];
        for(const [old,entry] of victims){if(available>=bytes)break;if(!force&&(this.frequency.get(old)||0)>=(this.frequency.get(key)||0))return false;remove.push(old);available+=entry.bytes;}
        for(const old of remove)await this.dropExpert(old);
        const tiles=[];let used=0;
        try {
            if(compact){
                const {raw,layout}=await this.store.readExpertBlob(layer,id),buffer=await this.b.alloc(raw,'u32');
                for(const name of names){const {shape,expert}=this.store.describe(name),rows=product(shape)/shape[0],tile=this.ops.tileSize(shape[0]),down=expert.role==='down';
                    for(let start=0;start<rows;start+=tile){const n=Math.min(tile,rows-start),tk=this.tileKey(name,start,n);this.tiles.set(tk,{format:'q2blob',codes:buffer,buffers:[buffer],bytes:raw.byteLength,codeOffset:layout.offsets[down?'down_codes':'gate_up_codes']/4,scaleOffset:layout.offsets[down?'down_scales':'gate_up_scales']/2,weightRow:down?start:2*start+(expert.role==='up'?1:0),rowStride:down?1:2});tiles.push(tk);}
                }
                this.experts.set(key,{tiles,bytes:raw.byteLength,rawBuffer:buffer});this.expertUsed+=raw.byteLength;this.misses++;this.store.forgetExpertBlob(layer,id);return true;
            }
            // One coalesced read of the canonical blob, shared by gate/up/down.
            await this.store.prefetchExpert?.(layer,id);
            for(const name of names){const {shape}=this.store.describe(name),rows=product(shape)/shape[0],tile=this.ops.tileSize(shape[0]);
                for(let start=0;start<rows;start+=tile){const n=Math.min(tile,rows-start),w=await this.ops.upload(await this.ops.readRows(name,start,n)),tk=this.tileKey(name,start,n);this.tiles.set(tk,w);tiles.push(tk);used+=w.bytes;}
            }
            this.experts.set(key,{tiles,bytes:used});this.expertUsed+=used;this.misses++;return true;
        }catch(error){for(const tile of tiles){for(const b of this.tiles.get(tile).buffers)await this.b.free(b);this.tiles.delete(tile);}throw error;}
    }
    info(){return {budgetBytes:this.budgetBytes,densePlannedBytes:this.denseBytes,denseResidentBytes:this.denseUsed,expertResidentBytes:this.expertUsed,residentExperts:this.experts.size,hits:this.hits,misses:this.misses,evictions:this.evictions};}
    async dispose(){this.b.flush?.();for(const w of this.tiles.values())for(const b of w.buffers)await this.b.free(b);this.tiles.clear();this.experts.clear();this.expertUsed=0;this.denseUsed=0;}
}
