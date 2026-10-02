import {router,gdn,conv,attention,rope,ReferenceEngine} from './reference.mjs';
import {createFixture} from '../src/fixture.js';
import {StrataEngine} from '../src/engine.js';
import {halfToFloat,PackStore,BlobSource} from '../src/model.js';
import {encodeFixturePack} from '../src/pack.js';
import {fusedConformance} from './fused-conformance.js';
export function compare(actual,expected,tolerance=2e-5,label='values') {
    if(actual.length!==expected.length)throw Error(label+' length mismatch');let maxError=0;
    for(let i=0;i<actual.length;i++){const e=Math.abs(actual[i]-expected[i]);if(!Number.isFinite(actual[i])||e>tolerance*(1+Math.abs(expected[i])))throw Error(`${label}[${i}]: ${actual[i]} versus ${expected[i]}, error ${e}`);maxError=Math.max(maxError,e);}return maxError;
}
export async function conformance(b,{cpu=null}={}) {
    const checks=[];let owned=[];
    const alloc=async(a,type='f32')=>{const h=await b.alloc(a,type);owned.push(h);return h;};
    const clean=async()=>{await b.idle();for(const x of owned)await b.free(x);owned=[];};
    const pass=(name,error=0)=>checks.push({name,maxAbsoluteError:error});
    try {
        await fusedConformance(b,{alloc,clean,pass,compare});
        const halves=Uint32Array.from({length:32768},(_,i)=>(2*i)|((2*i+1)<<16)),decoded=await alloc(65536);
        await b.run('strata_unpack_half',{Packed:await alloc(halves,'u32'),Values:decoded,count:65536},[1024,1,1]);
        const actualHalf=await b.read(decoded);for(let h=0;h<65536;h++){const expected=halfToFloat(h);if(!(Number.isNaN(expected)?Number.isNaN(actualHalf[h]):Object.is(actualHalf[h],expected)))throw Error('Packed FP16 decode differs at '+h);}
        pass('All 65536 packed FP16 bit patterns');await clean();
        // A complete expert blob with interleaved gate/up rows, signed scales,
        // three prompt rows and output padding exercises the actual resident ABI.
        {
            const cols=64,rows=65,tokens=3,cb=rows*cols/4,sb=rows*2,raw=new Uint8Array(Math.ceil(3*(cb+sb)/4)*4),view=new DataView(raw.buffer);
            for(let i=0;i<3*cb;i++)raw[i]=(i*13+7)&255;
            for(let i=0;i<3*rows;i++)view.setUint16(3*cb+i*2,0x3800+(i%6)*64+(i%3===0?0x8000:0),true);
            const x=Float32Array.from({length:tokens*cols},(_,i)=>Math.sin(i*.23)),blob=await alloc(new Uint32Array(raw.buffer),'u32'),input=await alloc(x),outputStride=rows+7,outputOffset=3;
            for(const role of ['gate','up','down']){
                const down=role==='down',codeOffset=(down?2*cb:0)/4,scaleOffset=(3*cb+(down?2*sb:0))/2,weightRow=role==='up'?1:0,rowStride=down?1:2,out=await alloc(tokens*outputStride),ref=new Float64Array(out.length);
                for(let t=0;t<tokens;t++)for(let r=0;r<rows;r++){const row=weightRow+r*rowStride,scale=halfToFloat(view.getUint16((scaleOffset+row)*2,true));let sum=0;for(let c=0;c<cols;c++)sum+=x[t*cols+c]*(((raw[codeOffset*4+row*cols/4+(c>>2)]>>((c%4)*2))&3)-1)*scale;ref[t*outputStride+outputOffset+r]=sum;}
                await b.run('strata_q2_project',{X:input,Blob:blob,Y:out,cols,rows,tokens,outputStride,outputOffset,codeOffset,scaleOffset,weightRow,rowStride},[rows,tokens,1]);pass('Packed Q2 '+role+' with strided prompt output',compare(await b.read(out),ref,2e-5));
            }await clean();
        }
        // Full 512-expert router, equal-logit ties, large magnitudes, k=10.
        for(const style of ['ties','spread']) {
            const logits=Float32Array.from({length:512},(_,i)=>style==='ties'?7:Math.sin(i*3.1)*90),reference=router(logits,10),ids=await alloc(10,'i32'),w=await alloc(10);
            await b.run('strata_router',{Logits:await alloc(logits),Ids:ids,Weights:w,experts:512,topK:10},[1,1,1]);compare(await b.read(ids,'i32'),reference.ids,0,'router IDs');pass('router '+style,compare(await b.read(w),reference.weights,2e-5));await clean();
        }
        // Full-width projection uses non-multiple workgroup tails and signed inputs.
        const cols=2560,rows=37,x=Float32Array.from({length:cols},(_,i)=>Math.sin(i*.73)),w=Float32Array.from({length:cols*rows},(_,i)=>Math.cos(i*.17)*.03),y=await alloc(rows);
        await b.run('strata_gemv',{X:await alloc(x),W:await alloc(w),Y:y,cols,rows,tokens:1},[rows,1,1]);
        const expected=Float64Array.from({length:rows},(_,r)=>x.reduce((sum,v,c)=>sum+v*w[r*cols+c],0));pass('2560-wide projection',compare(await b.read(y),expected));await clean();
        for(const bits of [2,4,8]) {
            const cols=64,rows=5,groupSize=16,mask=(1<<bits)-1,codes=new Uint32Array(Math.ceil(cols*rows*bits/32)),scales=Float32Array.from({length:cols*rows/groupSize},(_,i)=>(i%7-3)*.03),offsets=Float32Array.from(scales,(_,i)=>i*.001),xx=Float32Array.from({length:cols},(_,i)=>Math.sin(i));
            for(let i=0;i<cols*rows;i++){const bit=i*bits;codes[bit>>>5]|=((i*13+7)&mask)<<(bit&31);}
            const out=await alloc(rows),ref=Float64Array.from({length:rows},(_,r)=>xx.reduce((s,x,c)=>{const i=r*cols+c;return s+x*((((i*13+7)&mask)-3)*scales[Math.floor(i/groupSize)]+offsets[Math.floor(i/groupSize)]);},0));
            await b.run('strata_quant_gemv',{X:await alloc(xx),Codes:await alloc(codes,'u32'),Scales:await alloc(scales),Offsets:await alloc(offsets),Y:out,cols,rows,bits,groupSize,bias:-3,codebook:0,hasOffset:1},[rows,1,1]);pass(`S${bits} signed scales and offset plane`,compare(await b.read(out),ref));await clean();
        }
        // Production DeltaNet state shape; multiple steps and modulo head pairing.
        const dim=128,kh=16,vh=48,state=new Float32Array(dim*vh*dim),oracle=new Float64Array(state),stateBuffer=await alloc(state),out=await alloc(vh*dim);
        for(let step=0;step<3;step++) {
            const q=Float32Array.from({length:kh*dim},(_,i)=>Math.sin(i+step)*.04),k=Float32Array.from(q,(_,i)=>Math.cos(i*.3-step)*.04),v=Float32Array.from({length:vh*dim},(_,i)=>Math.sin(i*.1+step)),d=Float32Array.from({length:vh},(_,i)=>.9+i*.001),beta=Float32Array.from(d,()=>.6);
            const ref=gdn(oracle,q,k,v,d,beta,dim,kh,vh);
            await b.run('strata_gdn_step',{State:stateBuffer,Q:await alloc(q),K:await alloc(k),V:await alloc(v),Decay:await alloc(d),Beta:await alloc(beta),Y:out,dim,keyHeads:kh,valueHeads:vh},[Math.ceil(vh*dim/64),1,1]);pass('GDN production state step '+step,compare(await b.read(out),ref));
        }pass('GDN persistent state',compare(await b.read(stateBuffer),oracle));await clean();
        const channels=7,taps=4,dilation=3,hist=new Float64Array(channels*9),hb=await alloc(channels*9),weights=Float32Array.from({length:channels*taps},(_,i)=>Math.cos(i)),wb=await alloc(weights),cy=await alloc(channels);
        for(let step=0;step<12;step++){const input=Float32Array.from({length:channels},(_,i)=>Math.sin(i+step)),ref=conv(hist,input,weights,taps,dilation,true);await b.run('strata_conv',{History:hb,X:await alloc(input),W:wb,Y:cy,channels,taps,dilation,activation:1},[1,1,1]);compare(await b.read(cy),ref);}
        pass('PLE dilated convolution and 12 history shifts',compare(await b.read(hb),hist));await clean();
        const q=Float32Array.from({length:32},(_,i)=>Math.sin(i)),keys=Float32Array.from({length:9*16},(_,i)=>Math.cos(i*.13)),values=Float32Array.from(keys,(_,i)=>Math.sin(i*.07)),cells=[0,3,8],ay=await alloc(32);
        await b.run('strata_attention',{Q:await alloc(q),Keys:await alloc(keys),Values:await alloc(values),Cells:await alloc(Int32Array.from(cells),'i32'),Y:ay,dim:8,heads:4,kvHeads:2,count:cells.length,scale:1/Math.sqrt(8)},[4,1,1]);pass('GQA attention selected cells',compare(await b.read(ay),attention(q,keys,values,cells,8,4,2)));await clean();
        for(const [dim,count] of [[70,65],[256,2048]]) {
            const heads=4,kvHeads=2,q=Float32Array.from({length:heads*dim},(_,i)=>Math.sin(i*.13)*.2),k=Float32Array.from({length:count*kvHeads*dim},(_,i)=>Math.cos(i*.07)),v=Float32Array.from(k,(_,i)=>Math.sin(i*.11));
            const cells=Int32Array.from({length:count},(_,i)=>count-1-i),y=await alloc(heads*dim);
            await b.run('strata_attention',{Q:await alloc(q),Keys:await alloc(k),Values:await alloc(v),Cells:await alloc(cells,'i32'),Y:y,dim,heads,kvHeads,count,scale:1/Math.sqrt(dim)},[heads,Math.ceil(dim/64),1]);
            pass(`GQA attention ${dim} dimensions and ${count} cells`,compare(await b.read(y),attention(q,k,v,cells,dim,heads,kvHeads)));await clean();
        }
        const rx=Float32Array.from({length:24},(_,i)=>Math.sin(i)),cos=Float32Array.from({length:2},(_,i)=>Math.cos(13/10000**(i/2))),sin=Float32Array.from({length:2},(_,i)=>Math.sin(13/10000**(i/2))),ry=await alloc(24);
        await b.run('strata_rope',{X:await alloc(rx),Cos:await alloc(cos),Sin:await alloc(sin),Y:ry,dim:8,heads:3,rotary:4},[1,1,1]);pass('Partial NeoX RoPE',compare(await b.read(ry),rope(rx,8,4,13,10000)));await clean();
        const store=createFixture(),engine=await StrataEngine.create(store,b,{cpuBackend:cpu,gpuExperts:48}),reference=new ReferenceEngine(store);
        let max=0;const tokens=[];
        try {
            for(const token of [2,7,4,7,4]) {const expected=await reference.step(token),actual=await engine.step(token,{logits:true});max=Math.max(max,compare(actual.logits,expected.logits,8e-5,'full logits'));if(actual.token!==expected.token)throw Error('Autoregressive argmax differs');tokens.push(actual.token);}
            pass('4-layer GDN + QSA + PLE + HC + MoE decode, 5 tokens',max);
            const snapshot=await engine.checkpoint(),first=await engine.step(8,{logits:true});await engine.restore(snapshot);const replay=await engine.step(8,{logits:true});pass('Checkpoint restore replays logits',compare(first.logits,replay.logits,0));
            await engine.reset();const again=await engine.step(2,{logits:true}),fresh=await new ReferenceEngine(store).step(2);pass('Reset clears recurrent, KV and PLE state',compare(again.logits,fresh.logits,8e-5));
            await engine.reset();const promptReference=new ReferenceEngine(store);
            for(const token of [2,7]){await engine.step(token,{predict:false});await promptReference.step(token);}
            const predicted=await engine.step(4,{logits:true}),promptExpected=await promptReference.step(4);
            if(predicted.token!==promptExpected.token)throw Error('Prompt-only steps changed the generated token');
            pass('Prompt-only state updates preserve final logits',compare(predicted.logits,promptExpected.logits,8e-5));
            // Layer-major batching must preserve every recurrent/KV/PLE state,
            // including nonzero starts, chunk boundaries and the next decode.
            for(const chunkSize of [1,2,5]){
                await engine.reset();const ref=new ReferenceEngine(store);let wanted;
                await engine.step(2,{predict:false});await ref.step(2);
                for(const t of [7,4,8,7,4])wanted=await ref.step(t);
                const actual=await engine.prefill([7,4,8,7,4],{chunkSize,logits:true});
                pass(`Batched prompt chunk ${chunkSize} at nonzero position`,compare(actual.logits,wanted.logits,8e-5));
                const next=await engine.step(8,{logits:true}),expectedNext=await ref.step(8);
                pass(`Decode after prompt chunk ${chunkSize}`,compare(next.logits,expectedNext.logits,8e-5));
            }
            if(b.kind==='webgpu'){
                const resident=await StrataEngine.create(store,b,{weightBudgetBytes:8*1024*1024,tileRows:7,cpuBackend:cpu});
                try{
                    const ref=new ReferenceEngine(store);let expected;for(const t of [2,7,4])expected=await ref.step(t);
                    const out=await resident.prefill([2,7,4],{logits:true});pass('Resident tiled weights with batched prompt',compare(out.logits,expected.logits,8e-5));
                    const before=resident.ops.residency.denseUsed;await resident.step(7,{logits:true});
                    if(resident.ops.residency.denseUsed!==before||!resident.ops.residency.hits||!resident.ops.residency.experts.size)throw Error('Dense/expert residency did not persist');
                }finally{await resident.dispose();}
                const original=createFixture(),pack=encodeFixturePack(original),dense=pack.manifest.tensors['blk.0.hc_attn_down.weight'];
                // Losslessly round the fixture weights to a BF16 source, then
                // compare compact GPU storage with its original F32 container.
                const tensor=original.tensors.get(dense.name),bits=new Uint32Array(tensor.data.buffer);for(let i=0;i<bits.length;i++)bits[i]&=0xffff0000;
                pack.binary.set(new Uint8Array(tensor.data.buffer),dense.values.offset);dense.source_type='BF16';
                const compactStore=new PackStore(pack.manifest,new BlobSource([new File([pack.binary],'weights.bin')])),compact=await StrataEngine.create(compactStore,b,{weightBudgetBytes:8*1024**2,tileRows:7});
                try{const ref=new ReferenceEngine(original);let expected;for(const t of [2,7,4])expected=await ref.step(t);const out=await compact.prefill([2,7,4],{logits:true});pass('Lossless compact BF16 projections',compare(out.logits,expected.logits,8e-5));}finally{await compact.dispose();}
            }
            const stats={...engine.stats,weightCacheHits:engine.ops.hits,weightCacheMisses:engine.ops.misses};
            return {backend:await b.info(),checks,tokens,stats};
        } finally {await engine.dispose();}
    } finally {await clean();}
}
