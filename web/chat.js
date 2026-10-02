import {loadLocal,loadFolder} from '../src/model-files.js';
import {formatChat} from '../src/tokenizer.js';
import {TextSession} from '../src/text-session.js';
import {GpuBackend,WorkerBackend} from '../src/backend.js';
import {StrataEngine} from '../src/engine.js';
const $=id=>document.getElementById(id),diagnostic=window.strataChat={ready:false,model:null,busy:false,errors:[]};
let model=null,session=null,messages=[],busy=false,controller=null,timer;
const settings=()=>({mode:$('format').value,context:Number($('context').value),maxTokens:Number($('reply-limit').value),system:$('system').value});
function error(message=''){$('error').textContent=message;$('error').hidden=!message;}
function connection(text,ready=false){$('connection').replaceChildren(Object.assign(document.createElement('i'),{}),document.createTextNode(text));$('connection').classList.toggle('ready',ready);}
function budget(){
    $('send').disabled=busy||!model||!$('prompt').value.trim();$('budget').classList.remove('over');
    if(!model){$('budget').textContent='Load a model to count tokens';return;}
    try{const s=settings(),history=[...messages,{role:'user',content:$('prompt').value}],text=s.mode==='chat'?formatChat(history,s.system):$('prompt').value,n=model.tokenizer.encode(text,{parseSpecial:s.mode==='chat'}).length;
        $('budget').textContent=`${n} prompt + ${s.maxTokens} reply / ${s.context}`;$('budget').classList.toggle('over',n+s.maxTokens>s.context);
    }catch(e){$('budget').textContent=e.message;}
}
function lock(value){busy=value;diagnostic.busy=value;$('settings').disabled=value;for(const id of ['load-local','choose-folder','model-folder','new-chat','clear'])$(id).disabled=value;$('prompt').disabled=value;$('send').hidden=value;$('stop').hidden=!controller;$('stop').disabled=false;budget();}
function scroll(){const area=$('scroll-area');area.scrollTop=area.scrollHeight;}
function message(role,text=''){
    const article=document.createElement('article');article.className='message '+role;
    const avatar=document.createElement('span');avatar.className='avatar';avatar.textContent=role==='user'?'Y':'S';
    const content=document.createElement('div'),label=document.createElement('div'),body=document.createElement('div'),meta=document.createElement('div');
    label.className='message-label';label.textContent=role==='user'?'You':'Strata';body.className='message-body';body.textContent=text;meta.className='message-meta';
    if(role==='assistant'){const copy=document.createElement('button');copy.className='copy-button';copy.textContent='Copy';copy.onclick=async()=>{try{await navigator.clipboard.writeText(body.textContent);copy.textContent='Copied';setTimeout(()=>copy.textContent='Copy',1500);}catch{error('Select the reply text to copy it.');}};label.append(copy);}
    content.append(label,body,meta);article.append(avatar,content);$('messages').append(article);$('welcome').hidden=true;scroll();return {article,body,meta};
}
async function createEngine(context){
    const mode=$('backend').value,threads=Number($('threads').value);let backend,cpu,engine;
    const dispose=async()=>{try{if(engine)await engine.dispose();}finally{try{if(cpu)await cpu.dispose();}finally{if(backend)await backend.dispose();}}};
    try{
        if(mode==='wasm')backend=await WorkerBackend.create({threads});
        else{backend=await GpuBackend.create({onError:e=>{diagnostic.errors.push(e.message);controller?.abort();error(e.message);}});if(mode==='hybrid')cpu=await WorkerBackend.create({threads});}
        engine=await StrataEngine.create(model.store(context),backend,{cpuBackend:cpu,gpuExperts:48});diagnostic.backend=await backend.info();diagnostic.context=engine.g.context;return {engine,dispose};
    }catch(e){await dispose();throw e;}
}
async function open(loader){
    if(busy)return;error();lock(true);connection('Loading model…');$('model-state').textContent='Reading model and tokenizer…';
    try{
        const loaded=await loader();await session?.reset();model=loaded;session=new TextSession(model.tokenizer,createEngine);messages=[];$('messages').replaceChildren();$('welcome').hidden=false;$('conversation-title').textContent='New conversation';
        $('model-name').textContent=model.label;$('model-state').textContent='Ready · 48 layers · Q2_0';$('load-local').textContent='Reload downloaded model';diagnostic.model=model.label;diagnostic.chatSupported=model.chatSupported;
        $('format').querySelector('[value=chat]').disabled=!model.chatSupported;if(!model.chatSupported)$('format').value='completion';$('system').disabled=$('format').value==='completion';
        $('run-note').textContent='Weights stream from disk. Full-model replies can take several minutes.';connection('Model ready',true);document.body.classList.remove('settings-open');$('toggle-settings').setAttribute('aria-expanded','false');
    }catch(e){error(e.message);diagnostic.errors.push(e.message);connection(model?'Model ready':'No model loaded',!!model);$('model-state').textContent=model?'Ready':'Model not loaded';}
    finally{lock(false);}
}
$('load-local').onclick=()=>open(()=>loadLocal());$('choose-folder').onclick=()=>$('model-folder').click();$('model-folder').onchange=()=>{if($('model-folder').files.length)open(()=>loadFolder($('model-folder').files));};
async function clear(){if(busy)return;error();lock(true);try{await session?.reset();messages=[];delete diagnostic.last;$('messages').replaceChildren();$('welcome').hidden=false;$('conversation-title').textContent='New conversation';$('prompt').value='';}catch(e){error(e.message);}finally{lock(false);}}
$('new-chat').onclick=clear;$('clear').onclick=clear;
for(const id of ['backend','threads','context','format'])$(id).onchange=async()=>{
    if(busy)return;$('thread-control').hidden=$('backend').value==='webgpu';$('system').disabled=$('format').value==='completion';error();lock(true);try{await session?.reset();}catch(e){error(e.message);}finally{lock(false);}
};
for(const id of ['system','reply-limit'])$(id).addEventListener('input',budget);
$('prompt').addEventListener('input',()=>{budget();$('prompt').style.height='auto';$('prompt').style.height=Math.min(200,$('prompt').scrollHeight)+'px';});
$('prompt').addEventListener('keydown',event=>{if(event.key==='Enter'&&!event.shiftKey&&!event.isComposing){event.preventDefault();if(!$('send').disabled)$('composer').requestSubmit();}});
for(const button of document.querySelectorAll('[data-prompt]'))button.onclick=()=>{if(busy)return;$('prompt').value=button.dataset.prompt;$('prompt').focus();budget();};
$('toggle-settings').onclick=()=>{const open=document.body.classList.toggle('settings-open');$('toggle-settings').setAttribute('aria-expanded',String(open));};
$('stop').onclick=()=>{controller?.abort();$('stop').disabled=true;$('progress-label').textContent='Stopping after the current layer…';};
$('composer').onsubmit=async event=>{
    event.preventDefault();if(busy||!model)return;error();const prompt=$('prompt').value;if(!prompt.trim())return;
    const s=settings(),history=s.mode==='chat'?[...messages,{role:'user',content:prompt}]:[{role:'user',content:prompt}];
    try{session.plan({...s,messages:history});}catch(e){error(e.message);return;}
    messages=history;message('user',prompt);const reply=message('assistant');$('conversation-title').textContent=prompt.trim().slice(0,80);$('prompt').value='';
    controller=new AbortController();lock(true);$('progress').hidden=false;delete diagnostic.last;const start=performance.now();
    const tick=()=>$('elapsed').textContent=Math.round((performance.now()-start)/1000)+'s';tick();timer=setInterval(tick,1000);
    try{
        const result=await session.generate({...s,messages:history,signal:controller.signal,
            onProgress:p=>{diagnostic.progress=p;if(controller.signal.aborted)return;$('progress-label').textContent=p.phase==='starting'?'Preparing model…':p.phase==='prompt'?`Reading prompt · token ${p.done+1} of ${p.total}`:`Writing reply · ${p.done} tokens`;$('progress-bar').value=p.phase==='starting'?0:(p.done+(p.layer||0)/(p.layers||1))/p.total;},
            onText:(text,info)=>{reply.body.textContent=text;reply.meta.textContent=info.generated+' tokens';if(text)scroll();}
        });
        if(result.text)messages.push({role:'assistant',content:result.text});else reply.body.textContent=result.stop==='stopped'?'Stopped before a reply.':'The model ended without a text reply.';
        reply.meta.textContent=`${result.generated.length} tokens · ${result.seconds.toFixed(1)}s · ${result.stop==='end'?'Complete':result.stop==='stopped'?'Stopped':'Reply limit reached'}`;
        diagnostic.last=result;connection('Model ready',true);
    }catch(e){reply.body.textContent='The reply could not be completed.';reply.meta.textContent='Generation error';error(e.message);diagnostic.errors.push(e.message);}
    finally{clearInterval(timer);controller=null;$('progress').hidden=true;lock(false);$('prompt').focus();scroll();}
};
diagnostic.ready=true;budget();
