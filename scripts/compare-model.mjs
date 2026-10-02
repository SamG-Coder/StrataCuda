// Summarize independent backend runs without treating native Strata as an oracle.
import {readFile,writeFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import assert from 'node:assert/strict';
const modes=['webgpu','wasm','hybrid'],runs=[],arrays=[];
for(const mode of modes){
    const run=JSON.parse(await readFile('reports/model-'+mode+'.json','utf8'));
    const bytes=await readFile(run.logits.path);
    assert.equal(createHash('sha256').update(bytes).digest('hex'),run.logits.sha256,'Logit file hash');
    const values=new Float32Array(Uint8Array.from(bytes).buffer);
    assert.equal(values.length,run.logits.length);assert.ok(values.every(Number.isFinite));
    runs.push(run);arrays.push(values);
}
const base=runs[0],comparisons=[];
for(let n=1;n<runs.length;n++){
    const other=runs[n];
    for(const field of ['model','sourceSha256','kernelSourceSha256','prompt','inputTokens','generated','packVerification'])
        assert.deepEqual(other[field],base[field],'Mismatched run '+field);
    assert.equal(arrays[n].length,arrays[0].length);assert.equal(other.steps.length,base.steps.length);
    let maxAbsoluteError=0,sumSquaredError=0,differentRoutingLayers=0;
    for(let i=0;i<arrays[0].length;i++){
        const diff=arrays[n][i]-arrays[0][i];maxAbsoluteError=Math.max(maxAbsoluteError,Math.abs(diff));sumSquaredError+=diff*diff;
    }
    for(let step=0;step<base.steps.length;step++){
        assert.equal(other.steps[step].input,base.steps[step].input);assert.equal(other.steps[step].output,base.steps[step].output);
        assert.equal(other.steps[step].routes.length,base.steps[step].routes.length);
        for(let layer=0;layer<base.steps[step].routes.length;layer++)
            if(String(other.steps[step].routes[layer].ids)!==String(base.steps[step].routes[layer].ids))differentRoutingLayers++;
    }
    comparisons.push({against:'webgpu',mode:modes[n],values:arrays[0].length,maxAbsoluteError,rootMeanSquareError:Math.sqrt(sumSquaredError/arrays[0].length),differentRoutingLayers});
}
const report={timestamp:new Date().toISOString(),model:base.model,kernelSourceSha256:base.kernelSourceSha256,prompt:base.prompt,inputTokens:base.inputTokens,generated:base.generated,text:base.text,
    runs:runs.map(r=>({mode:r.mode,seconds:r.seconds,lastDecodeMs:r.steps.at(-1).ms,gpuExperts:r.stats.gpuExperts,cpuExperts:r.stats.cpuExperts,promotions:r.stats.promotions,errors:r.errors})),comparisons};
await writeFile('reports/model-comparison.json',JSON.stringify(report,null,2)+'\n');
console.log(JSON.stringify(report,null,2));
