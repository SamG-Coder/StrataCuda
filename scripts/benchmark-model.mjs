// Teacher-forced, full-model comparison with I/O and queue counters.
import {chromium} from 'playwright';
import {mkdir,writeFile,readFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {startServer} from './serve.mjs';
const args=Object.fromEntries(process.argv.slice(2).map(a=>{const i=a.indexOf('=');return [a.slice(2,i),a.slice(i+1)];}));
const label=args.label||'resident',resident=Number(args.resident??8),batch=args.batch!=='false',prefill=args.prefill==='true',repeat=args.repeat==='true';
const mode=args.mode||'webgpu';if(!['webgpu','hybrid','wasm'].includes(mode))throw Error('Invalid backend');
const repeats=Number(args.repeats??1),profile=args.profile==='true';if(!Number.isInteger(repeats)||repeats<1||repeats>20)throw Error('Invalid repeat count');
if(!/^[a-z0-9-]+$/.test(label)||!Number.isFinite(resident)||resident<0||resident>12)throw Error('Invalid benchmark options');
await mkdir('.local/benchmarks',{recursive:true});
const server=await startServer(0),browser=await chromium.launch({channel:process.env.STRATA_BROWSER||'msedge',headless:true,args:['--enable-unsafe-webgpu']});
try {
    const page=await browser.newPage(),errors=[];
    page.on('pageerror',e=>errors.push(e.message));
    page.on('console',m=>{if(m.text().startsWith('BENCH '))console.log(m.text());if(m.type()==='error')errors.push(m.text());});
    await page.goto(`http://127.0.0.1:${server.address().port}/test.html`);
    let profiler,profilePhase='cold';
    if(profile){profiler=await page.context().newCDPSession(page);await profiler.send('Profiler.enable');await profiler.send('Profiler.setSamplingInterval',{interval:1000});await profiler.send('Profiler.start');
        await page.exposeFunction('benchmarkWarmPhase',async()=>{const {profile}=await profiler.send('Profiler.stop');await writeFile(`.local/benchmarks/${label}-cold.cpuprofile`,JSON.stringify(profile));profilePhase='warm';await profiler.send('Profiler.start');});}
    const result=await page.evaluate(async({resident,batch,prefill,repeat,repeats,mode})=>{
        const {loadLocal}=await import('/src/model-files.js'),{GpuBackend,WorkerBackend}=await import('/src/backend.js'),{StrataEngine}=await import('/src/engine.js');
        const model=await loadLocal(),store=model.store(256),backend=mode==='wasm'?await WorkerBackend.create({threads:4}):await GpuBackend.create({onError:e=>console.error(e.message),batch}),cpu=mode==='hybrid'?await WorkerBackend.create({threads:4}):null;
        const kernelSourceSha256=backend.manifest?.sha256||(await(await fetch('/generated/manifest.json')).json()).sha256;
        const counters={sourceBytes:0,sourceReads:0,sourceMs:0,decodeMs:0,uploads:0,uploadBytes:0,fences:0,fenceMs:0};
        const read=store.source.read.bind(store.source);store.source.read=async(...args)=>{const t=performance.now();const r=await read(...args);counters.sourceBytes+=r.byteLength;counters.sourceReads++;counters.sourceMs+=performance.now()-t;return r;};
        const rows=store.readRows.bind(store);store.readRows=async(...args)=>{const t=performance.now();const r=await rows(...args);counters.decodeMs+=performance.now()-t;return r;};
        const alloc=backend.alloc.bind(backend);backend.alloc=(a,...args)=>{if(typeof a!=='number'){counters.uploads++;counters.uploadBytes+=a.byteLength;}return alloc(a,...args);};
        const idle=backend.idle.bind(backend);backend.idle=async()=>{const t=performance.now();counters.fences++;await idle();counters.fenceMs+=performance.now()-t;};
        const inputs=[760,6511,314,9338,369,11751],steps=[];
        const start=performance.now(),engine=await StrataEngine.create(store,backend,{cpuBackend:cpu,weightBudgetBytes:mode==='wasm'?0:resident*1024**3,tileRows:resident?4096:256});
        const initializedMs=performance.now()-start;
        let last;
        const record=async(input,out,ms)=>{
            steps.push({input,output:out.token,position:out.position,ms,counters:{...counters},backend:await backend.info(),residency:engine.residencyInfo?.()||{used:engine.ops.used,hits:engine.ops.hits,misses:engine.ops.misses},routes:structuredClone(engine.stats.routes)});
            console.log('BENCH '+JSON.stringify({input,output:out.token,ms,ioMiB:counters.sourceBytes/1024**2,residency:steps.at(-1).residency}));
        };
        try {
            if(prefill){const t=performance.now();last=await engine.prefill(inputs.slice(0,5),{logits:true,onProgress:(layer,total)=>console.log('BENCH prompt layer '+layer+'/'+total)});await record(inputs.slice(0,5),last,performance.now()-t);}
            for(let i=prefill?5:0;i<inputs.length;i++){
                const t=performance.now();last=await engine.step(inputs[i],{logits:i>=4,predict:i>=4,onProgress:l=>{if(l%12===0)console.log('BENCH token '+i+' layer '+l);}});await record(inputs[i],last,performance.now()-t);
            }
            const coldSeconds=(performance.now()-start)/1000;let warm=null;const warmRuns=[];
            if(repeat&&window.benchmarkWarmPhase)await window.benchmarkWarmPhase();
            for(let iteration=0;iteration<(repeat?repeats:0);iteration++){
                const expected=last.logits.slice(),before={...counters},warmStart=performance.now();await engine.reset();
                const promptStart=performance.now();last=await engine.prefill(inputs.slice(0,5),{logits:true});const promptMs=performance.now()-promptStart;
                const decodeStart=performance.now();last=await engine.step(inputs[5],{logits:true});const decodeMs=performance.now()-decodeStart;
                let maxError=0;for(let i=0;i<expected.length;i++)maxError=Math.max(maxError,Math.abs(expected[i]-last.logits[i]));if(maxError!==0)throw Error('Warm repetition changed logits');
                const sample={seconds:(performance.now()-warmStart)/1000,promptMs,decodeMs,maxError,counters:Object.fromEntries(Object.keys(counters).map(k=>[k,counters[k]-before[k]])),residency:engine.residencyInfo?.()};warmRuns.push(sample);warm??=sample;console.log('BENCH warm '+JSON.stringify(sample));
            }
            return {timestamp:new Date().toISOString(),kernelSourceSha256,mode,resident:mode==='wasm'?0:resident,tileRows:resident?4096:256,batch,prefill,inputs,initializedMs,seconds:coldSeconds,warm,warmRuns,steps,logits:Array.from(last.logits),backend:await backend.info(),cpu:cpu?await cpu.info():null};
        }finally{await engine.dispose();if(cpu)await cpu.dispose();await backend.dispose();}
    },{resident,batch,prefill,repeat,repeats,mode});
    if(profiler){const {profile}=await profiler.send('Profiler.stop');await writeFile(`.local/benchmarks/${label}-${profilePhase}.cpuprofile`,JSON.stringify(profile));await profiler.detach();}
    if(errors.length)throw Error(errors.join('\n'));
    const logits=Float32Array.from(result.logits);delete result.logits;
    result.logits={count:logits.length,sha256:createHash('sha256').update(new Uint8Array(logits.buffer)).digest('hex')};
    await writeFile(`.local/benchmarks/${label}-logits.f32`,new Uint8Array(logits.buffer));
    await writeFile(`reports/benchmark-${label}.json`,JSON.stringify(result,null,2)+'\n');
    console.log('PASS '+label+' '+result.seconds.toFixed(3)+' s; logits '+result.logits.sha256);
}finally{await browser.close();await new Promise(r=>server.close(r));}
