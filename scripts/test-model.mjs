// Real file-picker inference on the converted model. No inference runs in Python:
// upstream Python only tokenizes/decodes; all model arithmetic uses WebCuda.
import {readFile,writeFile,mkdir,stat} from 'node:fs/promises';
import {resolve,basename} from 'node:path';
import {execFile} from 'node:child_process';
import {createHash} from 'node:crypto';
import {chromium} from 'playwright';
import {startServer} from './serve.mjs';

const options=Object.fromEntries(process.argv.slice(2).map(a=>{const at=a.indexOf('=');if(!a.startsWith('--')||at<0)throw Error('Use --name=value options');return [a.slice(2,at),a.slice(at+1)];}));
const folder=resolve(options.pack||'models/Qwen3.8-Flash-Next-WebCuda-Q2_0');
const modes=(options.modes||'webgpu').split(','),steps=Number(options.steps||2),prompt=options.prompt||'The capital of France is';
if(modes.some(mode=>!['webgpu','wasm','hybrid'].includes(mode))||!Number.isInteger(steps)||steps<1||steps>16)throw Error('Invalid model test options');
const manifest=JSON.parse(await readFile(resolve(folder,'manifest.json'),'utf8'));
if(!manifest.verification||!await stat(resolve(folder,'VERIFIED.json')).catch(()=>null))throw Error('Verify the full pack before inference');
const python=options.python||process.env.STRATA_MODEL_PYTHON||resolve('.local/pack-env',process.platform==='win32'?'Scripts/python.exe':'bin/python');
const tokenize=input=>new Promise((fulfill,reject)=>{
    const child=execFile(python,['scripts/model-tokenizer.py','--pack',folder],{cwd:resolve('.'),encoding:'utf8',maxBuffer:4*1024*1024},(error,out,stderr)=>{
        if(error)reject(Error(stderr||String(error)));else try{fulfill(JSON.parse(out));}catch(e){reject(e);}
    });child.stdin.end(JSON.stringify(input));
});
const encoded=await tokenize({text:prompt});if(encoded.roundTrip!==prompt)throw Error('Tokenizer round trip failed');
console.log('Prompt:',JSON.stringify(prompt),'tokens:',encoded.tokens.join(','));
const localPle=resolve(folder,manifest.pleSource.file),plePath=await stat(localPle).catch(()=>null)?localPle:resolve(manifest.source.shard2);
const files=[resolve(folder,'manifest.json'),...Object.keys(manifest.files).map(name=>{if(basename(name)!==name)throw Error('Invalid pack file name');return resolve(folder,name);}),plePath];
await mkdir('.local/model-validation',{recursive:true});
const server=await startServer(0),browser=await chromium.launch({channel:process.env.STRATA_BROWSER||'msedge',headless:true,args:['--enable-unsafe-webgpu']});
try {
    for(const mode of modes) {
        const page=await browser.newPage({viewport:{width:1440,height:1240}});page.setDefaultTimeout(30*60*1000);
        const errors=[];page.on('pageerror',e=>errors.push(e.message));
        page.on('console',message=>{if(message.type()==='error')errors.push(message.text());});
        page.on('response',response=>{if(response.status()>=400)errors.push(response.status()+' '+response.url());});
        page.on('console',message=>{if(message.text().startsWith('MODEL '))console.log(mode,message.text());});
        try {
            await page.goto('http://127.0.0.1:'+server.address().port+'/test.html');await page.waitForFunction(()=>window.strataDiagnostics?.ready);
            await page.evaluate(async()=>{
                const {StrataEngine}=await import('/src/engine.js');const originalStep=StrataEngine.prototype.step,originalMoe=StrataEngine.prototype.moe;
                window.modelSteps=[];
                StrataEngine.prototype.step=async function(token,options){
                    console.log('MODEL step '+this.position+' input '+token);const out=await originalStep.call(this,token,options);
                    if(out.logits)window.modelLogits=Array.from(out.logits);
                    window.modelSteps.push({input:token,output:out.token,position:out.position,ms:this.stats.lastMs,routes:structuredClone(this.stats.routes)});
                    console.log('MODEL completed '+out.position+' in '+(this.stats.lastMs/1000).toFixed(2)+' s; output '+(out.token??'prompt-only'));return out;
                };
                StrataEngine.prototype.moe=async function(layer,x){const out=await originalMoe.call(this,layer,x);if((layer+1)%12===0)console.log('MODEL position '+this.position+' layer '+(layer+1)+'/'+this.g.layers);return out;};
            });
            await page.locator('#files').setInputFiles(files);
            await page.waitForFunction(name=>(window.strataDiagnostics.model===name||window.strataDiagnostics.errors.length)&&!document.getElementById('run').disabled,manifest.name);
            const loadErrors=await page.evaluate(()=>window.strataDiagnostics.errors);if(loadErrors.length)throw Error(loadErrors.join('\n'));
            await page.locator('#backend').selectOption(mode);await page.locator('#threads').selectOption('4');
            await page.locator('#tokens').fill(encoded.tokens.join(', '));await page.locator('#steps').fill(String(steps));
            const started=Date.now();await page.locator('#run').click();
            await page.waitForFunction(()=>window.strataDiagnostics.last||window.strataDiagnostics.errors.length,{},{timeout:30*60*1000});
            await page.waitForFunction(()=>!document.getElementById('run').disabled);
            const result=await page.evaluate(()=>({diagnostics:window.strataDiagnostics,steps:window.modelSteps,logits:window.modelLogits}));
            if(errors.length||result.diagnostics.errors.length)throw Error(JSON.stringify([...errors,...result.diagnostics.errors]));
            if(result.logits.length!==248320||result.logits.some(x=>!Number.isFinite(x)))throw Error('Invalid full-vocabulary logits');
            const logits=Float32Array.from(result.logits),logitsPath=resolve('.local/model-validation/'+mode+'-logits.f32');
            await writeFile(logitsPath,new Uint8Array(logits.buffer));
            const generated=result.diagnostics.last.generated,decoded=await tokenize({ids:generated});
            const report={timestamp:new Date().toISOString(),mode,model:manifest.name,packVerification:manifest.verification,sourceSha256:manifest.source.shard1_sha256,
                kernelSourceSha256:JSON.parse(await readFile('generated/manifest.json','utf8')).sha256,prompt,inputTokens:encoded.tokens,generated,text:decoded.text,
                seconds:(Date.now()-started)/1000,steps:result.steps,backend:result.diagnostics.backend,stats:result.diagnostics.last.stats,
                logits:{length:logits.length,path:logitsPath,sha256:createHash('sha256').update(new Uint8Array(logits.buffer)).digest('hex')},errors};
            await writeFile('reports/model-'+mode+'.json',JSON.stringify(report,null,2)+'\n');
            await page.screenshot({path:'reports/model-'+mode+'.png',fullPage:true});
            console.log('PASS real model '+mode+': '+JSON.stringify({generated,text:decoded.text,seconds:report.seconds}));
            await page.evaluate(()=>window.strataTests.release());
        }finally{await page.close();}
    }
}finally{await browser.close();await new Promise(r=>server.close(r));}
