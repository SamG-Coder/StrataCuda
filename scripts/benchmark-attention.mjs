import {readFile,writeFile} from 'node:fs/promises';
import {resolve} from 'node:path';
import {pathToFileURL} from 'node:url';
import {createHash} from 'node:crypto';
import {cpus} from 'node:os';
import {chromium} from 'playwright';
import {startServer} from './serve.mjs';
import {WasmBackend} from '../src/backend.js';
import {measureAttention} from '../tests/attention-benchmark.mjs';

const baseline=resolve(process.argv[2]||'.local/attention-before');
const folders={before:baseline,after:resolve('generated')},results={},hashes={};
for(const [variant,folder] of Object.entries(folders)) {
    const wasm=await readFile(resolve(folder,'strata.wasm')),abi=JSON.parse(await readFile(resolve(folder,'strata.abi.json'),'utf8'));
    hashes[variant]={wasm:createHash('sha256').update(wasm).digest('hex')};
    const {default:createModule}=await import(pathToFileURL(resolve(folder,'strata.mjs')));
    const backend=new WasmBackend(await createModule({wasmBinary:wasm}),abi,4);
    try {results['wasm_'+variant]=await measureAttention(backend,variant);console.log('WASM '+variant,JSON.stringify(results['wasm_'+variant]));}
    finally {await backend.dispose();}
}
const server=await startServer(0),browser=await chromium.launch({channel:'msedge',headless:true,args:['--enable-unsafe-webgpu']});
let gpu;
try {
    const page=await browser.newPage();await page.goto('http://127.0.0.1:'+server.address().port);
    for(const [variant,folder] of Object.entries(folders)) {
        const raw=await readFile(resolve(folder,'strata_attention.json'),'utf8'),artifact=JSON.parse(raw);
        hashes[variant].attentionArtifact=createHash('sha256').update(raw).digest('hex');
        const measured=await page.evaluate(async({variant,artifact})=>{
            const {GpuBackend}=await import('/src/backend.js'),{measureAttention}=await import('/tests/attention-benchmark.mjs');
            const backend=await GpuBackend.create();
            try {backend.kernels.set('strata_attention',await backend.runtime.kernel(artifact));return {info:backend.info(),result:await measureAttention(backend,variant,{iterations:5,samples:7})};}
            finally {await backend.dispose();}
        },{variant,artifact});
        results['webgpu_'+variant]=measured.result;gpu=measured.info;console.log('WebGPU '+variant,JSON.stringify(measured.result));
    }
} finally {await browser.close();await new Promise(r=>server.close(r));}
const speedups=[];
for(const backend of ['wasm','webgpu'])for(let i=0;i<results[backend+'_before'].length;i++) {
    const before=results[backend+'_before'][i],after=results[backend+'_after'][i];
    speedups.push({backend,count:before.count,beforeMs:before.medianMs,afterMs:after.medianMs,speedup:before.medianMs/after.medianMs});
}
const report={timestamp:new Date().toISOString(),cpu:cpus()[0].model,gpu,hashes,
    scope:'Attention kernel only; warm dispatch plus completion wall time, excluding upload and readback; deterministic synthetic inputs, no trained-model throughput claim. Download was active during measurement.',results,speedups};
await writeFile('reports/attention-performance.json',JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(speedups,null,2));
process.exit(0);
