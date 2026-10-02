import {attention} from './reference.mjs';
import {compare} from './conformance.js';

export async function measureAttention(backend,variant,{iterations=1,samples=5}={}) {
    const results=[];
    for(const count of [256,1024,2048]) {
        const dim=256,heads=24,kvHeads=2,owned=[];
        const alloc=async(data,type='f32')=>{const b=await backend.alloc(data,type);owned.push(b);return b;};
        const q=Float32Array.from({length:heads*dim},(_,i)=>Math.sin(i*.13)*.2);
        const keys=Float32Array.from({length:count*kvHeads*dim},(_,i)=>Math.cos(i*.07));
        const values=Float32Array.from(keys,(_,i)=>Math.sin(i*.11)),cells=Int32Array.from({length:count},(_,i)=>i);
        try {
            const y=await alloc(heads*dim),args={Q:await alloc(q),Keys:await alloc(keys),Values:await alloc(values),Cells:await alloc(cells,'i32'),Y:y,dim,heads,kvHeads,count,scale:1/Math.sqrt(dim)};
            const groups=variant==='before'?[Math.ceil(heads*dim/64),1,1]:[heads,Math.ceil(dim/64),1];
            await backend.run('strata_attention',args,groups);await backend.idle();
            const maxAbsoluteError=compare(await backend.read(y),attention(q,keys,values,cells,dim,heads,kvHeads));
            const timings=[];
            for(let sample=0;sample<samples;sample++) {
                const started=performance.now();
                for(let i=0;i<iterations;i++)await backend.run('strata_attention',args,groups);
                await backend.idle();timings.push((performance.now()-started)/iterations);
            }
            timings.sort((a,b)=>a-b);
            results.push({count,dim,heads,kvHeads,medianMs:timings[Math.floor(timings.length/2)],minMs:timings[0],maxMs:timings.at(-1),samples,iterations,maxAbsoluteError});
        } finally {await backend.idle();for(const b of owned)await backend.free(b);}
    }
    return results;
}
