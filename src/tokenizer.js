// Byte-level BPE ported from the pinned Strata tokenizer. This is text I/O;
// all model arithmetic remains in kernels/strata.cu.
export const QWEN35_PATTERN=String.raw`(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`;
const byteChars=new Array(256),charBytes=new Map(),visible=[];
for(const [lo,hi] of [[33,126],[161,172],[174,255]])for(let i=lo;i<=hi;i++)visible.push(i);
let extra=256;
for(let i=0;i<256;i++){const c=String.fromCodePoint(visible.includes(i)?i:extra++);byteChars[i]=c;charBytes.set(c,i);}
const utf8=new TextEncoder();
const append=(target,values)=>{for(const value of values)target.push(value);};
const escape=s=>s.replace(/[.*+?^${}()|[\]\\]/g,'\\$&');
const alternation=values=>values.length?new RegExp(values.sort((a,b)=>b.length-a.length).map(escape).join('|'),'gu'):null;
class Heap {
    a=[];
    before(a,b){return a[0]<b[0]||a[0]===b[0]&&a[1]<b[1];}
    push(value){let i=this.a.length;this.a.push(value);while(i){const p=(i-1)>>1;if(!this.before(value,this.a[p]))break;this.a[i]=this.a[p];i=p;}this.a[i]=value;}
    pop(){const top=this.a[0],last=this.a.pop();if(this.a.length){let i=0;while(i*2+1<this.a.length){let c=i*2+1;if(c+1<this.a.length&&this.before(this.a[c+1],this.a[c]))c++;if(!this.before(this.a[c],last))break;this.a[i]=this.a[c];i=c;}this.a[i]=last;}return top;}
}
export class Tokenizer {
    constructor({vocab,merges,types,config}){
        if(config.model!=='gpt2'||config.pre!=='qwen35'||config.pre_pattern!==QWEN35_PATTERN)throw Error('This model needs the supported Qwen35 byte-level tokenizer.');
        this.ids=new Map(Object.entries(vocab));this.tokens=new Array(this.ids.size);this.types=types;this.config=config;
        for(const [token,id] of this.ids){if(!Number.isInteger(id)||id<0||id>=this.tokens.length||this.tokens[id]!==undefined)throw Error('Invalid tokenizer vocabulary.');this.tokens[id]=token;}
        if(config.vocab_size!==this.tokens.length||this.tokens.some(t=>t===undefined)||types.length!==this.tokens.length)throw Error('Tokenizer vocabulary and metadata disagree.');
        this.ranks=new Map();const lines=Array.isArray(merges)?merges:merges.split(/\r?\n/).filter(Boolean);
        if(config.n_merges!==lines.length)throw Error('Tokenizer merge count differs from metadata.');
        for(let rank=0;rank<lines.length;rank++){const pair=lines[rank].split(' ');if(pair.length!==2||pair.some(p=>!this.ids.has(p)))throw Error('Invalid tokenizer merge.');this.ranks.set(lines[rank],rank);}
        // Python regex \s is Unicode White_Space. JavaScript \s differs for NEL/BOM.
        this.plain=new RegExp(QWEN35_PATTERN.replaceAll('\\s','\\p{White_Space}').replaceAll('\\S','\\P{White_Space}'),'gu');
        this.special=new Map(this.tokens.map((t,i)=>[t,i]).filter(([,i])=>types[i]===3||types[i]===4));
        this.always=alternation([...this.special].filter(([,i])=>types[i]===4).map(([t])=>t));
        this.all=alternation([...this.special.keys()]);this.cache=new Map();this.bytesCache=new Map();
        this.endTokens=new Set([config.special_ids?.['tokenizer.ggml.eos_token_id'],this.ids.get('<|im_end|>'),this.ids.get('<|endoftext|>')].filter(Number.isInteger));
    }
    bpe(word){
        const cached=this.cache.get(word);if(cached)return cached;
        const parts=Array.from(word);
        if(parts.length<=64){
            while(parts.length>1){let best=-1,rank=Infinity;for(let i=0;i<parts.length-1;i++){const r=this.ranks.get(parts[i]+' '+parts[i+1]);if(r!==undefined&&r<rank){best=i;rank=r;}}if(best<0)break;parts.splice(best,2,parts[best]+parts[best+1]);}
        }else{
            const next=parts.map((_,i)=>i+1<parts.length?i+1:-1),prev=parts.map((_,i)=>i-1),heap=new Heap();
            const offer=i=>{const j=next[i];if(j<0)return;const rank=this.ranks.get(parts[i]+' '+parts[j]);if(rank!==undefined)heap.push([rank,i,parts[i],parts[j]]);};
            for(let i=0;i<parts.length-1;i++)offer(i);
            while(heap.a.length){const [,i,left,right]=heap.pop(),j=next[i];if(parts[i]!==left||j<0||parts[j]!==right)continue;parts[i]=left+right;parts[j]=null;const k=next[j];next[i]=k;if(k>=0){prev[k]=i;offer(i);}if(prev[i]>=0)offer(prev[i]);}
        }
        const ids=parts.filter(p=>p!==null).map(p=>{const id=this.ids.get(p);if(id===undefined)throw Error('Tokenizer produced an unknown token.');return id;});
        if(this.cache.size>=4096)this.cache.delete(this.cache.keys().next().value);this.cache.set(word,ids);return ids;
    }
    encodePlain(text){
        const ids=[];let end=0;this.plain.lastIndex=0;
        for(const match of text.matchAll(this.plain)){if(match.index!==end)throw Error('Tokenizer skipped text.');end=match.index+match[0].length;let mapped='';for(const byte of utf8.encode(match[0]))mapped+=byteChars[byte];append(ids,this.bpe(mapped));}
        if(end!==text.length)throw Error('Tokenizer skipped trailing text.');return ids;
    }
    encode(text,{parseSpecial=false}={}){
        if(typeof text!=='string'||text.length>1024*1024||text.isWellFormed&&!text.isWellFormed())throw Error('Use well-formed text up to 1 MiB.');
        const pat=parseSpecial?this.all:this.always;if(!pat)return this.encodePlain(text);
        pat.lastIndex=0;const ids=[];let start=0;
        for(const match of text.matchAll(pat)){append(ids,this.encodePlain(text.slice(start,match.index)));ids.push(this.special.get(match[0]));start=match.index+match[0].length;}
        append(ids,this.encodePlain(text.slice(start)));return ids;
    }
    tokenBytes(id){
        if(!Number.isInteger(id)||id<0||id>=this.tokens.length)throw Error('Token outside vocabulary.');
        let bytes=this.bytesCache.get(id);if(!bytes){bytes=Uint8Array.from(Array.from(this.tokens[id],ch=>{const b=charBytes.get(ch);if(b===undefined)throw Error('Token outside byte alphabet.');return b;}));this.bytesCache.set(id,bytes);}return bytes;
    }
    decode(ids,{stream=false}={}){const chunks=ids.map(id=>this.tokenBytes(id)),bytes=new Uint8Array(chunks.reduce((n,b)=>n+b.length,0));let at=0;for(const chunk of chunks){bytes.set(chunk,at);at+=chunk.length;}return new TextDecoder('utf-8',{ignoreBOM:true}).decode(bytes,{stream});}
}

// Text-only, non-thinking rendering of the exact supplied Qwen chat template.
// Tools, images and arbitrary Jinja templates are intentionally not interpreted.
export const CHAT_TEMPLATE_SHA256='12827f24b742ea4e80cdc12dbcf9622227056b9f797252a3149263d4f9aaadce';
const trim=s=>s.replace(/^[\p{White_Space}\u001c-\u001f]+|[\p{White_Space}\u001c-\u001f]+$/gu,'');
export function formatChat(messages,system=''){
    if(!messages.length)throw Error('Enter a message first.');
    let text=trim(system)?'<|im_start|>system\n'+trim(system)+'<|im_end|>\n':'';
    for(const message of messages){if(!['user','assistant'].includes(message.role)||typeof message.content!=='string')throw Error('Unsupported chat message.');text+='<|im_start|>'+message.role+'\n'+(message.role==='assistant'?'<think>\n\n</think>\n\n':'')+trim(message.content)+'<|im_end|>\n';}
    return text+'<|im_start|>assistant\n<think>\n\n</think>\n\n';
}
