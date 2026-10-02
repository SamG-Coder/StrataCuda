import {requireInteger} from './backend.js';
const product=a=>a.reduce((x,y)=>x*y,1);
export const IQ4NL=[-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113];
export const STRATA_GEOMETRY={width:2560,streams:4,rank:320,layers:48,qsaInterval:4,ssmDim:128,keyHeads:16,valueHeads:48,
    heads:24,kvHeads:2,headDim:256,rotary:64,experts:512,topK:10,ff:640,vocab:0,context:256,
    pleLayer:1,pleHeads:16,pleHeadDim:160,pleTaps:4,pleDilation:3,ropeBase:1e7,eos:248044};
export const PLE_CONSTANTS={multipliers:['23703573157769','20109073645365','8052911324071'],
    vocab:[20000003,20000023,20000033,20000047,20000059,20000063,20000069,20000077,20000081,20000093,20000107,20000147,20000153,20000159,20000161,20000171],
    offsets:[0,20000003,40000026,60000059,80000106,100000165,120000228,140000297,160000374,180000455,200000548,220000655,240000802,260000955,280001114,300001275]};

export function validateGeometry(g) {
    for(const key of ['width','streams','rank','layers','qsaInterval','ssmDim','keyHeads','valueHeads','heads','kvHeads','headDim','rotary','experts','topK','ff','vocab','context','pleHeads','pleHeadDim','pleTaps','pleDilation']) requireInteger(g[key],key);
    if(g.context>2048) throw Error('Portable decode currently supports at most 2048 context cells; sparse QSA selection is not enabled');
    if(g.experts>512||g.topK>Math.min(32,g.experts)||g.heads%g.kvHeads||g.valueHeads%g.keyHeads) throw Error('Invalid attention or expert geometry');
    if(g.rotary%2||g.rotary>g.headDim||g.pleHeads%2||g.pleHeads*g.pleHeadDim!==g.width) throw Error('Invalid rotary or PLE geometry');
    if(!Number.isFinite(g.ropeBase)||g.ropeBase<=1) throw Error('Invalid RoPE base');
    if(!Number.isInteger(g.pleLayer)||g.pleLayer<0||g.pleLayer>=g.layers) throw Error('PLE layer is outside model');
    requireInteger(g.eos,'PLE EOS token',0);
    return g;
}
export function ngramRows(token,previous,constants=PLE_CONSTANTS,eos=248044) {
    requireInteger(token,'token',0); const ctx=[BigInt(token)]; let cut=false;
    for(let back=1;back<3;back++) {
        const t=previous[previous.length-back]??-1; cut ||= t<0||t===eos; ctx.push(BigInt(cut?eos:t));
    }
    const n=constants.vocab.length; if(n%2||constants.offsets.length!==n) throw Error('Invalid n-gram constants');
    const terms=ctx.map((t,i)=>BigInt.asUintN(64,t*BigInt(constants.multipliers[i])));
    return Uint32Array.from(constants.vocab,(v,i)=>{
        const mixed=i<n/2 ? terms[0]^terms[1] : terms[0]^terms[1]^terms[2];
        return Number(mixed%BigInt(v)+BigInt(constants.offsets[i]));
    });
}
export function halfToFloat(h) {
    const sign=(h&32768)?-1:1, exponent=(h>>>10)&31, fraction=h&1023;
    return exponent===0 ? sign*2**-14*(fraction/1024) : exponent===31 ? (fraction?NaN:sign*Infinity) : sign*2**(exponent-15)*(1+fraction/1024);
}
function floats(bytes,half=false) {
    const stride=half?2:4;
    if(bytes.byteLength%stride) throw Error('Misaligned float plane');
    const result=!half&&bytes.byteOffset%4===0&&littleEndian?new Float32Array(bytes.buffer,bytes.byteOffset,bytes.byteLength/4):new Float32Array(bytes.byteLength/stride);
    if(half){const v=new DataView(bytes.buffer,bytes.byteOffset,bytes.byteLength);for(let i=0;i<result.length;i++)result[i]=halfTable[v.getUint16(i*2,true)];}
    else if(result.buffer!==bytes.buffer){const v=new DataView(bytes.buffer,bytes.byteOffset,bytes.byteLength);for(let i=0;i<result.length;i++)result[i]=v.getFloat32(i*4,true);}
    for(let i=0;i<result.length;i++)if(!Number.isFinite(result[i]))throw Error('Non-finite model weights');return result;
}
const littleEndian=new Uint8Array(new Uint32Array([1]).buffer)[0]===1;
const halfTable=Float32Array.from({length:65536},(_,h)=>halfToFloat(h));
function words(bytes) {if(littleEndian&&bytes.byteOffset%4===0&&bytes.byteLength%4===0)return new Uint32Array(bytes.buffer,bytes.byteOffset,bytes.byteLength/4);const out=new Uint8Array(Math.ceil(bytes.length/4)*4);out.set(bytes);return new Uint32Array(out.buffer);}
export function dequantCanonical(t) {
    if(t.format==='f32') return t.data;
    const out=new Float32Array(product(t.shape)); const mask=(1<<t.bits)-1;
    for(let i=0;i<out.length;i++) {const bit=i*t.bits, code=(t.codes[bit>>>5]>>>(bit&31))&mask;
        const value=t.codebook===1?IQ4NL[code]:code+t.bias;
        out[i]=value*t.scales[Math.floor(i/t.groupSize)]+(t.hasOffset?t.offsets[Math.floor(i/t.groupSize)]:0);
    } return out;
}

export class MemoryStore {
    constructor(config,tensors,constants=PLE_CONSTANTS) {this.config=validateGeometry(config);this.tensors=tensors;this.constants=constants;this.label='Untrained deterministic fixture';}
    describe(name) {const t=this.tensors.get(name);if(!t)throw Error('Missing tensor '+name);return t;}
    async readRows(name,start,rows) {
        const t=this.describe(name),cols=t.shape[0],total=product(t.shape)/cols;
        if(!Number.isInteger(start)||start<0||!Number.isInteger(rows)||rows<1||start+rows>total)throw Error('Tensor row range invalid: '+name);
        if(t.format!=='f32')throw Error('Memory fixture expects F32');
        return {format:'f32',shape:[cols,rows],data:t.data.slice(start*cols,(start+rows)*cols)};
    }
    async readValues(name,start,count) {const t=this.describe(name);if(!Number.isSafeInteger(start)||!Number.isSafeInteger(count)||start<0||count<1||start+count>t.data.length)throw Error('Tensor value range invalid');return t.data.slice(start,start+count);}
    async pleRows(token,previous) {const rows=ngramRows(token,previous,this.constants,this.config.eos),out=new Float32Array(this.config.width);for(let h=0;h<rows.length;h++)out.set(await this.readValues('per_layer_token_embd.weight',rows[h]*this.config.pleHeadDim,this.config.pleHeadDim),h*this.config.pleHeadDim);return out;}
}

// Random access into user-selected Blobs. No whole-model ArrayBuffer or upload.
export class BlobSource {
    constructor(files) {
        this.files=new Map(); for(const f of files) {if(this.files.has(f.name))throw Error('Duplicate file basename: '+f.name);this.files.set(f.name,f);}
    }
    size(name) { const f=this.files.get(name);if(!f)throw Error('Select the model file '+name);return f.size; }
    async read(name,offset,bytes) {
        if(!Number.isSafeInteger(offset)||!Number.isSafeInteger(bytes)||offset<0||bytes<0||offset+bytes>this.size(name))throw Error('File range out of bounds: '+name);
        return new Uint8Array(await this.files.get(name).slice(offset,offset+bytes).arrayBuffer());
    }
}

export class PackStore {
    constructor(manifest,source,{config,ple}={}) {
        this.manifest=manifest;this.source=source;this.ple=ple;this.constants=manifest.pleConstants||PLE_CONSTANTS;
        if(!['strata-webcuda-v1','strata-pack'].includes(manifest.format))throw Error('Expected a Strata canonical pack or strata-webcuda-v1 manifest');
        if(manifest.format==='strata-pack' && manifest.experts?.source_type!=='Q2_0')throw Error('Canonical expert import currently requires Q2_0; IQ expert packs are not supported');
        this.config=validateGeometry({...STRATA_GEOMETRY,...manifest.config,...config,
            vocab:manifest.config?.vocab||manifest.tensors['token_embd.weight']?.shape?.[1]||0});
        if(this.constants.vocab.length!==this.config.pleHeads||this.constants.offsets.length!==this.config.pleHeads||this.constants.vocab.some(v=>!Number.isSafeInteger(v)||v<1)||this.constants.offsets.some(v=>!Number.isSafeInteger(v)||v<0))throw Error('PLE constants do not match the head geometry');
        this.label=manifest.name||'Local Strata pack';
        this.pleRowCache=new Map();this.plePending=new Map();
        for(const [name,bytes] of Object.entries(manifest.files||{}))if(source.size(name)!==bytes)throw Error(name+' size differs from manifest');
        for(const [name,t] of Object.entries(manifest.tensors)) {
            if(!Array.isArray(t.shape)||!t.shape.length||t.shape.some(x=>!Number.isSafeInteger(x)||x<1)||!Number.isSafeInteger(product(t.shape)))throw Error('Invalid tensor shape: '+name);
            if(!['P16','P32','S2','S4','S8','f32'].includes(t.form||t.format))throw Error('Unsupported tensor format: '+name);
            if(!!t.values===!!t.codes)throw Error('Tensor needs exactly one value or code plane: '+name);
            for(const key of ['values','codes','scales','offsets'])if(t[key]){const p=t[key];if(!Number.isSafeInteger(p.offset)||!Number.isSafeInteger(p.bytes)||p.offset<0||p.bytes<0||p.offset+p.bytes>source.size(t.file))throw Error('Invalid plane '+name+'.'+key);}
            if(t.values&&t.values.bytes!==product(t.shape)*(t.values_fp16?2:4))throw Error('Invalid value plane size: '+name);
            if(t.codes) {
                const groups=product(t.shape)/t.group_elems;
                if(![2,4,8].includes(t.code_bits)||!Number.isInteger(t.code_bias)||!Number.isSafeInteger(t.group_elems)||t.group_elems<1||t.shape[0]%t.group_elems||t.codes.bytes!==product(t.shape)*t.code_bits/8||t.scales?.bytes!==groups*(t.scales_fp16?2:4)||!!t.has_offset!==!!t.offsets||t.offsets&&t.offsets.bytes!==groups*(t.offsets_fp16?2:4))throw Error('Invalid canonical plane layout: '+name);
                if(!['Affine','affine','IQ4NL',null,undefined].includes(t.codebook)||t.codebook==='IQ4NL'&&t.code_bits!==4)throw Error('Unsupported codebook or code width: '+t.codebook);
                if(t.code_bias< -0x80000000||t.code_bias>0x7fffffff-((1<<t.code_bits)-1))throw Error('Canonical code bias exceeds signed kernel indexing');
            }
        }
    }
    describe(name) {
        const direct=this.manifest.tensors[name];if(direct)return direct;
        const e=/^blk\.(\d+)\.expert\.(\d+)\.(gate|up|down)$/.exec(name);
        if(!e||this.manifest.format!=='strata-pack')throw Error('Missing tensor '+name);
        const layer=Number(e[1]),expert=Number(e[2]); if(layer>=this.config.layers||expert>=this.config.experts)throw Error('Expert out of range');
        return {shape:e[3]==='down'?[this.config.ff,this.config.width]:[this.config.width,this.config.ff],expert:{layer,id:expert,role:e[3]}};
    }
    setExpertCacheBudget(bytes) {
        requireInteger(bytes,'expert RAM budget',0,Number.MAX_SAFE_INTEGER);
        this.expertCacheBudget=bytes;this.expertBlobs??=new Map();this.expertReads??=new Map();this.expertCacheUsed??=0;
        this.expertCacheStats??={hits:0,misses:0,bytesRead:0};
        while(this.expertCacheUsed>bytes&&this.expertBlobs.size){const [key,b]=this.expertBlobs.entries().next().value;this.expertBlobs.delete(key);this.expertCacheUsed-=b.byteLength;}
    }
    async prefetchExpert(layer,id) {
        if(!this.expertCacheBudget||!this.manifest.experts)return null;
        const key=`${layer}:${id}`,cached=this.expertBlobs.get(key);
        if(cached){this.expertBlobs.delete(key);this.expertBlobs.set(key,cached);this.expertCacheStats.hits++;return cached;}
        if(this.expertReads.has(key))return this.expertReads.get(key);
        const ex=this.manifest.experts,info=ex.layers.find(l=>l.layer===layer);
        if(!info||!Number.isInteger(id)||id<0||id>=this.config.experts)throw Error('Expert out of range');
        if(ex.blob_bytes>this.expertCacheBudget)return null;
        const work=(async()=>{
            const blob=await this.source.read('experts.bin',info.offset+id*ex.blob_bytes,ex.blob_bytes);
            this.retainExpertBlob(layer,id,blob);this.expertCacheStats.misses++;this.expertCacheStats.bytesRead+=blob.byteLength;return blob;
        })();
        this.expertReads.set(key,work);try{return await work;}finally{this.expertReads.delete(key);}
    }
    forgetExpertBlob(layer,id){const key=`${layer}:${id}`,old=this.expertBlobs?.get(key);if(old){this.expertBlobs.delete(key);this.expertCacheUsed-=old.byteLength;}}
    retainExpertBlob(layer,id,blob){
        if(!this.expertCacheBudget||blob.byteLength>this.expertCacheBudget)return;
        this.forgetExpertBlob(layer,id);
        while(this.expertCacheUsed+blob.byteLength>this.expertCacheBudget&&this.expertBlobs.size){const [old,b]=this.expertBlobs.entries().next().value;this.expertBlobs.delete(old);this.expertCacheUsed-=b.byteLength;}
        this.expertBlobs.set(`${layer}:${id}`,blob);this.expertCacheUsed+=blob.byteLength;
    }
    clearExpertCache(){this.expertBlobs?.clear();this.expertCacheUsed=0;}
    async readExpertBlob(layer,id){
        const ex=this.manifest.experts;if(!ex||ex.source_type!=='Q2_0')return null;
        let raw=await this.prefetchExpert(layer,id);
        if(!raw){const info=ex.layers.find(l=>l.layer===layer);if(!info||!Number.isInteger(id)||id<0||id>=this.config.experts)throw Error('Expert out of range');raw=await this.source.read('experts.bin',info.offset+id*ex.blob_bytes,ex.blob_bytes);}
        const view=new DataView(raw.buffer,raw.byteOffset,raw.byteLength);
        for(let i=ex.offsets.gate_up_scales;i<raw.byteLength;i+=2)if((view.getUint16(i,true)&0x7c00)===0x7c00)throw Error('Non-finite expert scales');
        return {raw:words(raw),layout:ex};
    }
    async readRows(name,start,rows,{compact=false}={}) {
        const t=this.describe(name),cols=t.shape[0],total=product(t.shape)/cols;
        if(!Number.isInteger(start)||start<0||!Number.isInteger(rows)||rows<1||start+rows>total)throw Error('Tensor range invalid: '+name);
        if(t.expert)return this.readExpert(t,start,rows);
        const count=cols*rows,begin=start*cols;
        if(t.values) {
            const stride=t.values_fp16?2:4;
            const raw=await this.source.read(t.file,t.values.offset+begin*stride,count*stride);
            if(compact&&t.source_type==='BF16'&&!t.values_fp16){
                const v=new DataView(raw.buffer,raw.byteOffset,raw.byteLength),codes=new Uint32Array(Math.ceil(count/2));
                for(let i=0;i<count;i++){const word=v.getUint32(i*4,true);if((word&65535)!==0||(word&0x7f800000)===0x7f800000)throw Error('Invalid canonical BF16 value');codes[i>>>1]|=(word>>>16)<<((i&1)*16);}
                return {format:'bf16',shape:[cols,rows],codes};
            }
            return {format:'f32',shape:[cols,rows],data:floats(raw,t.values_fp16)};
        }
        const gs=t.group_elems, groupStart=begin/gs,groups=count/gs;
        const [c,s,o]=await Promise.all([
            this.source.read(t.file,t.codes.offset+begin*t.code_bits/8,count*t.code_bits/8),
            this.source.read(t.file,t.scales.offset+groupStart*(t.scales_fp16?2:4),groups*(t.scales_fp16?2:4)),
            t.offsets?this.source.read(t.file,t.offsets.offset+groupStart*(t.offsets_fp16?2:4),groups*(t.offsets_fp16?2:4)):null
        ]);
        return {format:'s',shape:[cols,rows],codes:words(c),scales:floats(s,t.scales_fp16),offsets:o?floats(o,t.offsets_fp16):new Float32Array(1),
            bits:t.code_bits,groupSize:gs,bias:t.code_bias,codebook:t.codebook==='IQ4NL'?1:0,hasOffset:!!t.offsets};
    }
    async readExpert(t,start,rows) {
        const {layer,id,role}=t.expert, ex=this.manifest.experts, cols=t.shape[0];
        if(cols%64)throw Error('Q2_0 expert width must be divisible by 64');
        const layerInfo=ex.layers.find(l=>l.layer===layer);if(!layerInfo)throw Error('Missing expert layer');
        const base=layerInfo.offset+id*ex.blob_bytes, down=role==='down', stride=down?1:2, first=down?start:2*start+(role==='up'?1:0);
        const codeStart=base+ex.offsets[down?'down_codes':'gate_up_codes'],scaleStart=base+ex.offsets[down?'down_scales':'gate_up_scales'];
        const span=(rows-1)*stride+1;
        const blob=await this.prefetchExpert(layer,id),read=(offset,bytes)=>blob?blob.subarray(offset-base,offset-base+bytes):this.source.read('experts.bin',offset,bytes);
        const [cb,sb]=await Promise.all([read(codeStart+first*cols/4,span*cols/4),read(scaleStart+first*cols/64*2,span*cols/64*2)]);
        // Q2_0 uses consecutive two-bit codes (unlike Q4_0's split halves).
        const packed=new Uint8Array(rows*cols/4),scales=new Float32Array(rows*cols/64),sv=new DataView(sb.buffer,sb.byteOffset,sb.byteLength);
        for(let r=0;r<rows;r++){
            packed.set(cb.subarray(r*stride*cols/4,(r*stride+1)*cols/4),r*cols/4);
            for(let b=0;b<cols/64;b++)scales[r*cols/64+b]=halfTable[sv.getUint16((r*stride*cols/64+b)*2,true)];
        }
        if(scales.some(x=>!Number.isFinite(x)))throw Error('Non-finite expert scales');
        return {format:'s',shape:[cols,rows],codes:words(packed),scales,offsets:new Float32Array(1),bits:2,groupSize:64,bias:-1,codebook:0,hasOffset:false};
    }
    async readValues(name,start,count) {
        const t=this.describe(name),cols=t.shape[0];
        if(!Number.isSafeInteger(start)||!Number.isSafeInteger(count)||start<0||count<1||start+count>product(t.shape))throw Error('Tensor value range invalid: '+name);
        const begin=Math.floor(start/cols),end=Math.ceil((start+count)/cols);
        return dequantCanonical(await this.readRows(name,begin,end-begin)).slice(start-begin*cols,start-begin*cols+count);
    }
    async pleRows(token,previous) {
        const g=this.config,ids=ngramRows(token,previous,this.constants,g.eos),out=new Float32Array(g.width);
        if(this.manifest.tensors['per_layer_token_embd.weight']) {
            for(let h=0;h<ids.length;h++)out.set(await this.readValues('per_layer_token_embd.weight',ids[h]*g.pleHeadDim,g.pleHeadDim),h*g.pleHeadDim);
        } else {
            if(!this.ple)throw Error('Select the original PLE GGUF shard as well as the canonical pack files');
            if(this.ple.type!==20||g.pleHeadDim%32)throw Error('PLE GGUF import currently requires IQ4_NL rows');
            const rowBytes=g.pleHeadDim/32*18;
            const rows=await Promise.all(Array.from(ids,async id=>{
                if(this.pleRowCache.has(id)){const row=this.pleRowCache.get(id);this.pleRowCache.delete(id);this.pleRowCache.set(id,row);return row;}
                if(this.plePending.has(id))return this.plePending.get(id);
                const work=this.source.read(this.ple.file,this.ple.offset+id*rowBytes,rowBytes).then(row=>{while(this.pleRowCache.size>=65536)this.pleRowCache.delete(this.pleRowCache.keys().next().value);this.pleRowCache.set(id,row);return row;});
                this.plePending.set(id,work);try{return await work;}finally{this.plePending.delete(id);}
            }));
            for(let h=0;h<rows.length;h++){const raw=rows[h],v=new DataView(raw.buffer,raw.byteOffset,raw.byteLength);for(let b=0;b<g.pleHeadDim/32;b++){const scale=halfToFloat(v.getUint16(b*18,true));if(!Number.isFinite(scale))throw Error('Non-finite PLE scales');for(let i=0;i<32;i++)out[h*g.pleHeadDim+b*32+i]=scale*IQ4NL[(raw[b*18+2+i%16]>>>(i<16?0:4))&15];}}
        }
        return out;
    }
}
