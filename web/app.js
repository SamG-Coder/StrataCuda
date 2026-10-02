import {GpuBackend,WorkerBackend} from '../src/backend.js';
import {StrataEngine} from '../src/engine.js';
import {createFixture} from '../src/fixture.js';
import {BlobSource,PackStore} from '../src/model.js';
import {readGGUF} from '../src/gguf.js';
import {conformance} from '../tests/conformance.js';
import {logitBars} from './logits.js';
const $=id=>document.getElementById(id);
let store=createFixture(),backend,cpu,engine,key='',busy=false;
const diagnostic=window.strataDiagnostics={ready:true,errors:[],model:'fixture',backend:null};
function log(message){$('log').textContent+='\n'+message;$('log').scrollTop=$('log').scrollHeight;}
function state(label,error=false){$('status').textContent=label;$('status').className='pill'+(error?' error':busy?' busy':'');}
function lock(value){busy=value;for(const id of ['run','verify','reset','fixture','backend','threads','files'])$(id).disabled=value;}
function modelDetails(){
    $('model-name').textContent=store.label;
    $('token-hint').textContent=`Vocabulary: 0–${store.config.vocab-1} · context: ${store.config.context} tokens`;
    delete diagnostic.last;
}
async function release(){if(engine)await engine.dispose();engine=null;if(cpu)await cpu.dispose();cpu=null;if(backend)await backend.dispose();backend=null;key='';}
async function ensure(){
    const next=$('backend').value+':'+$('threads').value;if(engine&&key===next)return;
    await release();state('Starting');log('Starting '+$('backend').value+' backend…');
    try{
        if($('backend').value==='wasm')backend=await WorkerBackend.create({threads:Number($('threads').value)});
        else {backend=await GpuBackend.create({onError:e=>{diagnostic.errors.push(e.message);log(e.message);}});if($('backend').value==='hybrid')cpu=await WorkerBackend.create({threads:Number($('threads').value)});}
        engine=await StrataEngine.create(store,backend,{cpuBackend:cpu,gpuExperts:48});key=next;const info=await backend.info();
        diagnostic.backend=info;$('adapter').textContent=info.backend==='webgpu'?(info.vendor+' · '+info.architecture):`WASM · ${info.threads} CPU threads`;
        log('Ready: '+store.config.layers+' layers, '+store.config.experts+' experts per layer, '+store.config.context+' context cells.');state('Ready');
    }catch(error){await release();throw error;}
}
function render(result){
    $('position').textContent=engine.position+' / '+store.config.context;$('latency').textContent=engine.stats.lastMs.toFixed(1)+' ms';$('gpu-count').textContent=engine.stats.gpuExperts;$('cpu-count').textContent=engine.stats.cpuExperts;
    $('logits').replaceChildren();
    for(const item of logitBars(result.logits,result.token).bars){const bar=document.createElement('div');bar.className='bar'+(item.best?' best':'');bar.style.height=item.height+'%';bar.title=`Token ${item.id}: ${item.value.toFixed(6)}`;$('logits').append(bar);}
    $('routing').replaceChildren();for(const r of engine.stats.routes){const row=document.createElement('div');row.className='route';const layer=document.createElement('span');layer.className='layer';layer.textContent='Layer '+r.layer;row.append(layer);r.ids.forEach((id,i)=>{const x=document.createElement('span');x.className='expert '+(r.devices[i]==='wasm'?'cpu':'gpu');x.textContent='E'+id;row.append(x);});const type=document.createElement('span');type.className='type';type.textContent=r.layer%store.config.qsaInterval===store.config.qsaInterval-1?'QSA + MoE':'GDN + MoE';row.append(type);$('routing').append(row);}
}
async function action(fn){if(busy)return;lock(true);try{await fn();state('Ready');}catch(e){diagnostic.errors.push(e.message);log('ERROR: '+e.message);state('Error',true);}finally{lock(false);}}
$('run').onclick=()=>action(async()=>{
    delete diagnostic.last;
    await ensure();const tokens=$('tokens').value.split(/[\s,]+/).filter(Boolean).map(Number),steps=Number($('steps').value);
    if(!tokens.length||tokens.some(n=>!Number.isInteger(n)||n<0||n>=store.config.vocab)||!Number.isInteger(steps)||steps<1||steps>16)throw Error('Use valid token IDs and 1–16 decode steps');
    if(engine.position+tokens.length+steps-1>store.config.context)throw Error('Request exceeds remaining context; reset the session');
    state('Decoding');let last;for(let i=0;i<tokens.length;i++){const final=i===tokens.length-1;last=await engine.step(tokens[i],{logits:final,predict:final});}const generated=[last.token];render(last);$('output').textContent=generated.join(' · ');
    for(let i=1;i<steps;i++){last=await engine.step(last.token,{logits:true});generated.push(last.token);render(last);$('output').textContent=generated.join(' · ');await new Promise(r=>requestAnimationFrame(r));}
    $('output-note').textContent=store.label+' · greedy decoding';log('Generated ['+generated.join(', ')+']');diagnostic.last={generated,position:engine.position,stats:engine.stats};
});
$('reset').onclick=()=>action(async()=>{if(engine)await engine.reset();$('position').textContent='0';$('output').textContent='State reset. Ready for the next prompt.';log('Recurrent, KV and PLE history cleared.');});
$('fixture').onclick=()=>action(async()=>{await release();store=createFixture();diagnostic.model='fixture';modelDetails();$('tokens').value='2, 7, 4';log('Loaded the untrained fixture.');});
async function verify(){
    await ensure();state('Verifying');log('Comparing against independent scalar equations…');
    const report=await conformance(backend,{cpu});$('checks').replaceChildren();
    for(const c of report.checks){const row=document.createElement('div');row.className='check';const tick=document.createElement('span');tick.className='tick';tick.textContent='✓';const name=document.createElement('span');name.textContent=c.name;const error=document.createElement('span');error.className='error-value';error.textContent=c.maxAbsoluteError.toExponential(1);row.append(tick,name,error);$('checks').append(row);}
    $('validation-summary').textContent=report.checks.length+' checks passed';log('PASS: '+report.checks.length+' checks. Maximum errors are shown beside each check.');diagnostic.validation=report;state('Ready');return report;
}
$('verify').onclick=()=>action(verify);
$('files').onchange=()=>action(async()=>{
    const files=Array.from($('files').files),manifestFile=files.find(f=>f.name==='manifest.json');if(!manifestFile)throw Error('Include manifest.json with the model files');if(manifestFile.size>16*1024*1024)throw Error('Manifest exceeds 16 MiB');
    const manifest=JSON.parse(await manifestFile.text()),source=new BlobSource(files);let ple;
    for(const file of files.filter(f=>f.name.endsWith('.gguf'))){const gguf=await readGGUF(source,file.name),tensor=gguf.tensors.find(t=>t.name==='per_layer_token_embd.weight');if(tensor)ple={...tensor,file:file.name};}
    const loaded=new PackStore(manifest,source,{ple});await release();store=loaded;modelDetails();diagnostic.model=store.label;log('Loaded pack header. Weights will stream in bounded tiles.');
});
window.strataTests={
    async run(mode='webgpu'){await release();$('backend').value=mode;store=createFixture();await ensure();return verify();},
    async decode(mode='hybrid'){await release();$('backend').value=mode;store=createFixture();await ensure();const out=await engine.step(2,{logits:true});render(out);$('output').textContent=out.token;return {token:out.token,logits:Array.from(out.logits),stats:engine.stats};},
    release
};
