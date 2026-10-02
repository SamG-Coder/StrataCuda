import {BlobSource,PackStore} from './model.js';
import {readGGUF} from './gguf.js';
import {Tokenizer,CHAT_TEMPLATE_SHA256} from './tokenizer.js';
export const LOCAL_PACK='/models/Qwen3.8-Flash-Next-WebCuda-Q2_0/';
const safeName=name=>{if(typeof name!=='string'||!name||/[\\/]/.test(name)||name==='.'||name==='..')throw Error('Invalid model filename.');return name;};
export class RangeSource {
    constructor(base,sizes){this.base=new URL(base,globalThis.location?.href||'http://localhost/');this.sizes=sizes;}
    size(name){safeName(name);const size=this.sizes[name];if(!Number.isSafeInteger(size)||size<0)throw Error('Missing model file: '+name);return size;}
    async read(name,offset,bytes){
        const size=this.size(name);if(!Number.isSafeInteger(offset)||!Number.isSafeInteger(bytes)||offset<0||bytes<0||offset+bytes>size)throw Error('Invalid model file range.');if(!bytes)return new Uint8Array();
        const response=await fetch(new URL(encodeURIComponent(name),this.base),{headers:{Range:`bytes=${offset}-${offset+bytes-1}`}});
        if(response.status!==206||response.headers.get('Content-Range')!==`bytes ${offset}-${offset+bytes-1}/${size}`){await response.body?.cancel();throw Error('The model server must support exact byte ranges. Use Choose folder instead.');}
        const data=new Uint8Array(await response.arrayBuffer());if(data.length!==bytes)throw Error('Truncated model range.');return data;
    }
}
async function tokenizerFrom(read){
    const [vocab,merges,types,config,template]=await Promise.all(['vocab.json','merges.txt','token_type.json','tokenizer.json','chat_template.jinja'].map(name=>read('tokenizer/'+name)));
    const tokenizer=new Tokenizer({vocab:JSON.parse(vocab),merges,types:JSON.parse(types),config:JSON.parse(config)});
    const hash=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',new TextEncoder().encode(template))),b=>b.toString(16).padStart(2,'0')).join('');
    return {tokenizer,chatSupported:hash===CHAT_TEMPLATE_SHA256};
}
async function finish(manifest,source,read){
    if(manifest.format!=='strata-pack')throw Error('Open a trained canonical model pack. The fixture is available in the Test lab.');
    const pleName=safeName(manifest.pleSource?.file),gguf=await readGGUF(source,pleName),tensor=gguf.tensors.find(t=>t.name==='per_layer_token_embd.weight');
    if(!tensor)throw Error('The model folder needs its PLE GGUF shard.');
    const ple={...tensor,file:pleName},store=new PackStore(manifest,source,{ple}),data=await tokenizerFrom(read);
    if(data.tokenizer.tokens.length!==store.config.vocab)throw Error('The tokenizer does not match the model vocabulary.');
    return {manifest,source,ple,label:store.label,...data,store:context=>new PackStore(manifest,source,{ple,config:{context}})};
}
export async function loadFolder(files){
    const list=Array.from(files),find=name=>list.find(f=>(f.webkitRelativePath||f.name).replaceAll('\\','/').endsWith('/'+name)||f.name===name);
    const read=async name=>{const file=find(name);if(!file)throw Error('The selected folder is missing '+name+'. Choose the converted pack folder, including tokenizer/.');if(file.size>64*1024*1024)throw Error('Model metadata is too large.');return file.text();};
    const manifest=JSON.parse(await read('manifest.json'));
    const needed=[...Object.keys(manifest.files||{}),manifest.pleSource?.file].map(name=>{safeName(name);const file=find(name);if(!file)throw Error('Missing model file: '+name);return file;});
    return finish(manifest,new BlobSource(needed),read);
}
export async function loadLocal(base=LOCAL_PACK){
    const url=new URL(base,location.href);
    const read=async name=>{const response=await fetch(new URL(name,url));if(!response.ok)throw Error('The downloaded model was not found here. Use Choose folder to open your converted pack.');const size=Number(response.headers.get('Content-Length'));if(size>64*1024*1024){await response.body?.cancel();throw Error('Model metadata is too large.');}return response.text();};
    const manifest=JSON.parse(await read('manifest.json')),ple=manifest.pleSource;if(!ple)throw Error('This pack needs conversion verification before loading.');
    return finish(manifest,new RangeSource(url,{...manifest.files,[safeName(ple.file)]:ple.offset+ple.bytes}),read);
}
