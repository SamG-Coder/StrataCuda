import test from 'node:test';
import assert from 'node:assert/strict';
import {createFixture} from '../src/fixture.js';
import {BlobSource,PackStore,ngramRows,PLE_CONSTANTS,halfToFloat,dequantCanonical,validateGeometry} from '../src/model.js';
import {encodeFixturePack} from '../src/pack.js';
import {readGGUF} from '../src/gguf.js';
import {validateDispatch} from '../src/backend.js';
const source=(binary,name='weights.bin')=>new BlobSource([new File([binary],name)]);

test('portable pack preserves every weight, row and PLE gather',async()=>{
    const original=createFixture(),p=encodeFixturePack(original),s=new PackStore(p.manifest,source(p.binary));
    for(const [name,t] of original.tensors)assert.deepEqual(await s.readValues(name,0,t.data.length),t.data,name);
    for(const history of [[],[0],[2,7],[39,4],[4,39]])assert.deepEqual(await s.pleRows(8,history),await original.pleRows(8,history));
});
test('bad plane length, missing file and out-of-file offsets fail before execution',()=>{
    const p=encodeFixturePack(createFixture());let m=structuredClone(p.manifest);m.tensors['output.weight'].values.bytes-=4;assert.throws(()=>new PackStore(m,source(p.binary)),/plane size/);
    m=structuredClone(p.manifest);m.tensors['output.weight'].values.offset=p.binary.length;assert.throws(()=>new PackStore(m,source(p.binary)),/plane/);
    assert.throws(()=>new PackStore(p.manifest,source(p.binary,'different.bin')),/Select the model file/);
});
test('non-finite float weights and unsafe random access are rejected',async()=>{
    const p=encodeFixturePack(createFixture());new DataView(p.binary.buffer).setFloat32(0,NaN,true);const s=new PackStore(p.manifest,source(p.binary));
    await assert.rejects(()=>s.readValues('token_embd.weight',0,32),/Non-finite/);
    await assert.rejects(()=>s.source.read('weights.bin',-4,16),/bounds/);await assert.rejects(()=>s.readRows('output.weight',40,1),/range/);
});
test('canonical half scales and float offset planes are read with their own widths',async()=>{
    const p=encodeFixturePack(createFixture()),raw=new Uint8Array(16+4+8),dv=new DataView(raw.buffer);raw.fill(0xe4,0,16);dv.setUint16(16,0x3800,true);dv.setUint16(18,0xbc00,true);dv.setFloat32(20,.25,true);dv.setFloat32(24,.5,true);
    p.manifest.files['quant.bin']=raw.length;p.manifest.tensors.q={shape:[32,2],file:'quant.bin',form:'S2',code_bits:2,code_bias:-1,codebook:'Affine',group_elems:32,has_offset:true,codes:{offset:0,bytes:16},scales:{offset:16,bytes:4},scales_fp16:true,offsets:{offset:20,bytes:8},offsets_fp16:false};
    const store=new PackStore(p.manifest,new BlobSource([new File([p.binary],'weights.bin'),new File([raw],'quant.bin')]));
    const q=await store.readRows('q',0,2),values=dequantCanonical(q);assert.deepEqual(Array.from(values.slice(0,4)),[-.25,.25,.75,1.25]);assert.deepEqual(Array.from(values.slice(32,36)),[1.5,.5,-.5,-1.5]);
    const broken=structuredClone(p.manifest);broken.tensors.q.scales_fp16=false;assert.throws(()=>new PackStore(broken,store.source),/canonical plane/);
});
test('canonical Q2_0 expert gate/up interleave preserves consecutive code order',async()=>{
    const p=encodeFixturePack(createFixture()),g={...p.manifest.config,width:64,ff:64,pleHeadDim:16},rows=64,cols=64,codesBytes=rows*cols/4,scalesBytes=rows*cols/64*2,blobBytes=3*(codesBytes+scalesBytes),raw=new Uint8Array(blobBytes),v=new DataView(raw.buffer);
    raw.fill(0xe4,0,2*codesBytes);raw.fill(0x1b,2*codesBytes,3*codesBytes);for(let i=3*codesBytes;i<raw.length;i+=2)v.setUint16(i,0x3c00,true);
    const m={format:'strata-pack',config:g,pleConstants:p.manifest.pleConstants,tensors:{...p.manifest.tensors},files:{'weights.bin':p.binary.length,'experts.bin':raw.length},experts:{source_type:'Q2_0',blob_bytes:blobBytes,offsets:{gate_up_codes:0,down_codes:2*codesBytes,gate_up_scales:3*codesBytes,down_scales:3*codesBytes+2*scalesBytes},layers:[{layer:0,offset:0}]}};
    for(const name of Object.keys(m.tensors))if(name.includes('.expert.'))delete m.tensors[name];
    const store=new PackStore(m,new BlobSource([new File([p.binary],'weights.bin'),new File([raw],'experts.bin')]));
    for(const role of ['gate','up'])assert.deepEqual(Array.from(dequantCanonical(await store.readRows(`blk.0.expert.0.${role}`,3,1)).slice(0,8)),[-1,0,1,2,-1,0,1,2]);
    assert.deepEqual(Array.from(dequantCanonical(await store.readRows('blk.0.expert.0.down',3,1)).slice(0,4)),[2,1,0,-1]);
});
test('PLE hash honors oldest-first history, token zero, EOS cut and 64-bit wrap',()=>{
    const a=ngramRows(7,[2,3]);assert.notDeepEqual(a,ngramRows(7,[3,2]));assert.notDeepEqual(ngramRows(7,[0,0]),ngramRows(7,[]));
    assert.deepEqual(ngramRows(7,[99,248044]),ngramRows(7,[]));assert.deepEqual(ngramRows(7,[248044,8]),ngramRows(7,[-1,8]));
    const i=2147483000,terms=[i,2147482000,2147481000].map((t,j)=>BigInt.asUintN(64,BigInt(t)*BigInt(PLE_CONSTANTS.multipliers[j]))),rows=ngramRows(i,[2147481000,2147482000]);
    assert.equal(rows[0],Number((terms[0]^terms[1])%BigInt(PLE_CONSTANTS.vocab[0])));assert.equal(rows[8],Number((terms[0]^terms[1]^terms[2])%BigInt(PLE_CONSTANTS.vocab[8]))+PLE_CONSTANTS.offsets[8]);
});
test('half decoder preserves subnormals and signed zero',()=>{assert.equal(halfToFloat(1),2**-24);assert.equal(halfToFloat(0x3c00),1);assert.ok(Object.is(halfToFloat(0x8000),-0));assert.equal(halfToFloat(0x7c00),Infinity);});
test('unsupported geometry and kernel contracts fail explicitly',()=>{
    assert.throws(()=>validateGeometry({...createFixture().config,context:2049}),/2048/);
    assert.throws(()=>validateDispatch('strata_router',{experts:6,topK:7}),/topK/);
    assert.throws(()=>validateDispatch('strata_attention',{heads:7,kvHeads:2}),/divisible/);
    assert.throws(()=>validateDispatch('strata_rope',{rotary:7,dim:8}),/even/);
    assert.throws(()=>validateGeometry({...createFixture().config,pleLayer:1.5}),/PLE layer/);
    assert.throws(()=>validateDispatch('strata_attention',{heads:4,kvHeads:2,count:2049}),/2048/);
});

test('value reads reject fractional, non-finite and empty ranges in both stores',async()=>{
    const fixture=createFixture(),p=encodeFixturePack(fixture),pack=new PackStore(p.manifest,source(p.binary));
    for(const store of [fixture,pack])for(const [start,count] of [[.5,1],[0,1.5],[0,0],[NaN,1],[0,Infinity],[-1,1]])
        await assert.rejects(()=>store.readValues('token_embd.weight',start,count),/value range/);
});

test('canonical codebooks reject incompatible bit widths and overflowing biases',()=>{
    const p=encodeFixturePack(createFixture());p.manifest.files['q.bin']=24;
    p.manifest.tensors.q={shape:[64],file:'q.bin',form:'S2',code_bits:2,code_bias:-1,codebook:'IQ4NL',group_elems:32,has_offset:false,codes:{offset:0,bytes:16},scales:{offset:16,bytes:8}};
    const files=new BlobSource([new File([p.binary],'weights.bin'),new File([new Uint8Array(24)],'q.bin')]);
    assert.throws(()=>new PackStore(p.manifest,files),/codebook/);
    p.manifest.tensors.q.codebook='Affine';p.manifest.tensors.q.code_bias=2**32;
    assert.throws(()=>new PackStore(p.manifest,files),/bias/);
});

test('raw PLE tables reject non-finite IQ4_NL scales before dispatch',async()=>{
    const p=encodeFixturePack(createFixture()),rows=p.manifest.pleConstants.vocab.reduce((a,b)=>a+b,0),raw=new Uint8Array(rows*18),v=new DataView(raw.buffer);
    for(let r=0;r<rows;r++)v.setUint16(r*18,0x7c00,true);
    delete p.manifest.tensors['per_layer_token_embd.weight'];
    const files=new BlobSource([new File([p.binary],'weights.bin'),new File([raw],'ple.gguf')]);
    const pack=new PackStore(p.manifest,files,{config:{width:128,pleHeadDim:32},ple:{file:'ple.gguf',offset:0,type:20}});
    await assert.rejects(()=>pack.pleRows(2,[]),/Non-finite PLE/);
});

function ggufFixture(){
    const parts=[];const u32=v=>{const b=Buffer.alloc(4);b.writeUInt32LE(v);parts.push(b);},u64=v=>{const b=Buffer.alloc(8);b.writeBigUInt64LE(BigInt(v));parts.push(b);},str=s=>{u64(Buffer.byteLength(s));parts.push(Buffer.from(s));};
    u32(0x46554747);u32(3);u64(1);u64(2);str('general.architecture');u32(8);str('qwen4exp');str('general.alignment');u32(4);u32(32);
    str('per_layer_token_embd.weight');u32(2);u64(160);u64(1);u32(20);u64(0);let out=Buffer.concat(parts);out=Buffer.concat([out,Buffer.alloc((32-out.length%32)%32),Buffer.alloc(90)]);return out;
}
test('GGUF parser reads only bounded header windows and resolves absolute offsets',async()=>{
    const raw=ggufFixture(),s=source(raw,'ple.gguf'),g=await readGGUF(s,'ple.gguf');assert.equal(g.metadata['general.architecture'],'qwen4exp');assert.equal(g.tensors[0].typeName,'IQ4_NL');assert.equal(g.tensors[0].bytes,90);assert.equal(g.tensors[0].offset+90,raw.length);
});
test('GGUF rejects wrong magic, truncated tensor body and header bombs',async()=>{
    let raw=ggufFixture();raw[0]=0;await assert.rejects(()=>readGGUF(source(raw,'x.gguf'),'x.gguf'),/magic/);
    raw=ggufFixture().subarray(0,-1);await assert.rejects(()=>readGGUF(source(raw,'x.gguf'),'x.gguf'),/outside file/);
    raw=ggufFixture();raw.writeBigUInt64LE(100001n,8);await assert.rejects(()=>readGGUF(source(raw,'x.gguf'),'x.gguf'),/header limits/);
});
