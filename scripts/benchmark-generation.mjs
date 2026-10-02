// Real browser generation timings with the same native benchmark prompt.
import {chromium} from 'playwright';
import {writeFile} from 'node:fs/promises';
import {startServer} from './serve.mjs';

const started=performance.now(),server=await startServer(0);
const browser=await chromium.launch({channel:process.env.STRATA_BROWSER||'msedge',headless:true,args:['--enable-unsafe-webgpu']});
let result;
try{
    const page=await browser.newPage(),errors=[];
    page.on('pageerror',e=>errors.push(e.message));
    page.on('console',m=>{if(m.type()==='error')errors.push(m.text());if(m.text().startsWith('GENBENCH '))console.log(m.text());});
    await page.goto(`http://127.0.0.1:${server.address().port}/test.html`);
    result=await page.evaluate(async()=>{
        const {loadLocal}=await import('/src/model-files.js'),{GpuBackend}=await import('/src/backend.js'),{StrataEngine}=await import('/src/engine.js');
        const setupStart=performance.now(),model=await loadLocal(),store=model.store(256),backend=await GpuBackend.create({onError:e=>console.error(e.message)});
        const engine=await StrataEngine.create(store,backend,{weightBudgetBytes:8*1024**3,tileRows:4096}),setupMs=performance.now()-setupStart;
        const inputs=[760,6511,314,9338,369,11751],samples=[];
        try{
            for(let run=0;run<3;run++){
                if(run)await engine.reset();
                const start=performance.now();
                await engine.prefill(inputs.slice(0,-1),{predict:false});
                const prefillMs=performance.now()-start,decodeStart=performance.now(),output=[];let token=inputs.at(-1),firstTokenMs;
                for(let i=0;i<32;i++){
                    const out=await engine.step(token);output.push(out.token);token=out.token;
                    if(!i)firstTokenMs=performance.now()-start;
                    if((i+1)%8===0)console.log(`GENBENCH run ${run+1}, generated ${i+1}/32`);
                }
                const decodeMs=performance.now()-decodeStart;
                const sample={run:run+1,retainedWeights:run>0,prefillTokens:inputs.length-1,prefillMs,generatedTokens:output.length,decodeMs,tokensPerSecond:output.length*1000/decodeMs,timeToFirstTokenMs:firstTokenMs,output,residency:engine.residencyInfo()};
                if(run&&String(output)!==String(samples[0].output))throw Error('Reset changed generated tokens');
                samples.push(sample);console.log('GENBENCH '+JSON.stringify(sample));
            }
            return {timestamp:new Date().toISOString(),kernelSourceSha256:backend.manifest.sha256,inputs,setupMs,samples,backend:await backend.info(),note:'32 greedy tokens from the same six-token prompt as native. Stop at the fixed benchmark length. First sample starts with empty weight caches; subsequent samples reset state and retain weights. OS file cache is not flushed. Native and portable arithmetic and generated sequences may differ.'};
        }finally{await engine.dispose();await backend.dispose();}
    });
    if(errors.length)throw Error(errors.join('\n'));
}finally{await browser.close();await new Promise(resolve=>server.close(resolve));}
result.totalHarnessSeconds=(performance.now()-started)/1000;
await writeFile('reports/benchmark-generation-webgpu.json',JSON.stringify(result,null,2)+'\n');
console.log('PASS '+result.samples.map(s=>s.tokensPerSecond.toFixed(2)+' tok/s').join(', '));
