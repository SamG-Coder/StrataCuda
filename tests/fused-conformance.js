// Independent scalar equations and layout sentinels for CUDA-owned decoding.
import {halfToFloat,IQ4NL} from '../src/model.js';
export async function fusedConformance(b,{alloc,clean,pass,compare}){
    {
        const width=128,ff=64,tokens=3,cb=width*ff/4,sb=width*ff/64*2,bytes=3*(cb+sb),raw=new Uint8Array(bytes),v=new DataView(raw.buffer);
        for(let i=0;i<3*cb;i++)raw[i]=(i*71+13)&255;
        for(let i=0;i<3*sb/2;i++)v.setUint16(3*cb+2*i,0x2800+(i%11)*51+(i%3===0?32768:0),true);
        const x=Float32Array.from({length:tokens*width},(_,i)=>Math.sin(i*.17)*.3),mapping=new Int32Array([2,4,0,1,1,3]),hiddenRef=new Float64Array(tokens*ff);
        const weight=(index,down)=>{const offset=down?2*cb:0,scaleOffset=3*cb+(down?2*sb:0);return (((raw[offset+(index>>2)]>>((index%4)*2))&3)-1)*halfToFloat(v.getUint16(scaleOffset+2*Math.floor(index/64),true));};
        for(let t=0;t<tokens;t++)for(let r=0;r<ff;r++){let gate=0,up=0;for(let c=0;c<width;c++){gate+=weight(2*r*width+c,false)*x[mapping[2*t]*width+c];up+=weight((2*r+1)*width+c,false)*x[mapping[2*t]*width+c];}hiddenRef[t*ff+r]=gate/(1+Math.exp(-gate))*up;}
        const outputOffset=7,ref=new Float64Array(5*width+outputOffset).fill(-99);
        for(let t=0;t<tokens;t++)for(let r=0;r<width;r++){let sum=0;for(let c=0;c<ff;c++)sum+=weight(r*ff+c,true)*hiddenRef[t*ff+c];ref[outputOffset+mapping[2*t+1]*width+r]=sum;}
        const blob=await alloc(new Uint32Array(raw.buffer),'u32'),map=await alloc(mapping,'i32'),hidden=await alloc(hiddenRef.length),out=await alloc(new Float32Array(ref.length).fill(-99));
        const input=await alloc(x),dummy=await alloc(1);
        await b.run('strata_q2_gate_up',{X:input,Blob:blob,Scales:dummy,Mapping:map,Hidden:hidden,width,ff,tokens,mapped:1,expanded:0,codeOffset:0,scaleOffset:3*cb/2},[ff,tokens,1]);
        pass('Fused Q2 gate/up/SiLU with reordered input rows',compare(await b.read(hidden),hiddenRef,2e-5));
        await b.run('strata_q2_down',{Hidden:hidden,Blob:blob,Scales:dummy,Mapping:map,Y:out,width,ff,tokens,mapped:1,expanded:0,outputOffset,codeOffset:2*cb/4,scaleOffset:(3*cb+2*sb)/2},[width,tokens,1]);
        pass('Fused Q2 down with sparse destinations and untouched padding',compare(await b.read(out),ref,2e-5));
        const scales=await alloc(3*sb/2);await b.run('strata_expert_scales',{Blob:blob,Scales:scales,offset:3*cb/2,count:3*sb/2},[Math.ceil(3*sb/2/64),1,1]);
        await b.run('strata_q2_gate_up',{X:input,Blob:blob,Scales:scales,Mapping:map,Hidden:hidden,width,ff,tokens,mapped:1,expanded:1,codeOffset:0,scaleOffset:0},[ff,tokens,1]);
        await b.run('strata_q2_down',{Hidden:hidden,Blob:blob,Scales:scales,Mapping:map,Y:out,width,ff,tokens,mapped:1,expanded:1,outputOffset,codeOffset:2*cb/4,scaleOffset:sb},[width,tokens,1]);
        pass('CUDA-expanded expert scales preserve fused output',compare(await b.read(out),ref,2e-5));await clean();
    }
    for(const bits of [2,4,8]){
        const width=96,streams=3,groupSize=16,mask=(1<<bits)-1,codes=new Uint32Array(width*bits/32),scales=Float32Array.from({length:width/groupSize},(_,i)=>(i-2)*.03125),offsets=Float32Array.from(scales,(_,i)=>i*.125),outputOffset=5;
        for(let i=0;i<width;i++)codes[Math.floor(i*bits/32)]|=((i*7+3)&mask)<<((i*bits)%32);
        const expected=new Float64Array(width*streams+outputOffset).fill(47);
        for(let s=0;s<streams;s++)for(let i=0;i<width;i++)expected[outputOffset+s*width+i]=(((i*7+3)&mask)-3)*scales[Math.floor(i/groupSize)]+offsets[Math.floor(i/groupSize)];
        const out=await alloc(new Float32Array(expected.length).fill(47)),dummy=await alloc(1);
        await b.run('strata_embedding',{Values:dummy,Codes:await alloc(codes,'u32'),Scales:await alloc(scales),Offsets:await alloc(offsets),Residual:out,width,streams,outputOffset,format:2,bits,groupSize,bias:-3,codebook:0,hasOffset:1},[2,1,1]);
        pass('CUDA S'+bits+' embedding decode and stream replication',compare(await b.read(out),expected,2e-6));await clean();
    }
    {
        const blocks=19,raw=new Uint8Array(Math.ceil(blocks*18/4)*4),v=new DataView(raw.buffer),expected=new Float64Array(blocks*32);
        for(let block=0;block<blocks;block++){const half=0x2000+block*61+(block%3===0?32768:0);v.setUint16(block*18,half,true);for(let i=0;i<16;i++)raw[block*18+2+i]=(i*23+block*13)&255;for(let i=0;i<32;i++)expected[block*32+i]=halfToFloat(half)*IQ4NL[(raw[block*18+2+i%16]>>(i<16?0:4))&15];}
        const out=await alloc(expected.length);await b.run('strata_ple_decode',{Packed:await alloc(new Uint32Array(raw.buffer),'u32'),Values:out,count:expected.length},[Math.ceil(expected.length/64),1,1]);
        pass('CUDA IQ4_NL PLE decode with unaligned blocks and signed scales',compare(await b.read(out),expected,0));await clean();
    }
    for(const position of [0,17,2045]){
        const dim=96,heads=3,rotary=position===17?48:64,tokens=3,base=10000000,x=Float32Array.from({length:dim*heads*tokens},(_,i)=>Math.sin(i*.1)),ref=Float64Array.from(x);
        for(let t=0;t<tokens;t++)for(let h=0;h<heads;h++)for(let i=0;i<rotary/2;i++){const at=(t*heads+h)*dim+i,angle=(position+t)/base**(2*i/rotary),c=Math.cos(angle),s=Math.sin(angle),a=x[at],bb=x[at+rotary/2];ref[at]=a*c-bb*s;ref[at+rotary/2]=bb*c+a*s;}
        const context=position+tokens,high=await alloc(rotary/2),low=await alloc(rotary/2),cos=await alloc(context*rotary/2),sin=await alloc(context*rotary/2);
        await b.run('strata_rope_frequencies',{High:high,Low:low,rotary,base},[1,1,1]);await b.run('strata_rope_table',{High:high,Low:low,Cos:cos,Sin:sin,rotary,context},[Math.ceil(context*rotary/2/64),1,1]);
        const out=await alloc(x.length);await b.run('strata_rope_position',{X:await alloc(x),Cos:cos,Sin:sin,Y:out,dim,heads,rotary,position,tokens},[Math.ceil(x.length/64),1,1]);
        pass('CUDA RoPE trigonometry at position '+position,compare(await b.read(out),ref,8e-7));await clean();
    }
}
