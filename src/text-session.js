import {formatChat} from './tokenizer.js';

export class TextSession {
    constructor(tokenizer,create){this.tokenizer=tokenizer;this.create=create;this.owner=null;this.consumed=[];this.busy=false;}
    async reset(){if(this.busy)throw Error('Stop generation before resetting.');await this.drop();}
    async drop(){const owner=this.owner;this.owner=null;this.consumed=[];if(owner)await owner.dispose();}
    plan({messages,system='',mode='chat',context,maxTokens}){
        if(!Number.isInteger(context)||context<32||context>2048||!Number.isInteger(maxTokens)||maxTokens<1||maxTokens>context)throw Error('Choose a context of 32–2048 tokens and a reply limit within it.');
        if(!['chat','completion'].includes(mode))throw Error('Unknown prompt format.');
        const text=mode==='chat'?formatChat(messages,system):messages.at(-1)?.content;
        const tokens=this.tokenizer.encode(text,{parseSpecial:mode==='chat'});
        if(!tokens.length)throw Error('Enter a prompt first.');
        if(tokens.length+maxTokens>context)throw Error(`This request needs ${tokens.length+maxTokens} tokens (${tokens.length} prompt + ${maxTokens} reply). Increase context, shorten the prompt, or reduce the reply limit.`);
        return {tokens,text};
    }
    async generate(request){
        if(this.busy)throw Error('Generation is already running.');
        const {tokens}=this.plan(request),{context,maxTokens,signal,onProgress=()=>{},onText=()=>{}}=request;
        this.busy=true;const started=performance.now();let generated=[],stop='limit';
        try{
            signal?.throwIfAborted();
            if(this.owner&&(this.owner.engine.g.context!==context||this.consumed.length>=tokens.length||this.consumed.some((t,i)=>t!==tokens[i])))await this.drop();
            if(!this.owner){onProgress({phase:'starting',done:0,total:tokens.length});this.owner=await this.create(context);}
            signal?.throwIfAborted();
            const engine=this.owner.engine;let result;
            const step=async(token,predict,phase,done,total)=>{
                onProgress({phase,done,total,layer:0,layers:engine.g.layers,position:engine.position});
                const out=await engine.step(token,{predict,signal,onProgress:(layer,layers)=>onProgress({phase,done,total,layer,layers,position:engine.position})});
                this.consumed.push(token);return out;
            };
            for(let i=this.consumed.length;i<tokens.length;i++)result=await step(tokens[i],i===tokens.length-1,'prompt',i,tokens.length);
            for(let n=0;n<maxTokens;n++){
                signal?.throwIfAborted();
                if(this.tokenizer.endTokens.has(result.token)){stop='end';break;}
                generated.push(result.token);onText(this.tokenizer.decode(generated,{stream:true}),{generated:generated.length,position:engine.position,ms:engine.stats.lastMs});
                if(n+1<maxTokens)result=await step(result.token,true,'reply',n+1,maxTokens);
            }
            const text=this.tokenizer.decode(generated,{stream:stop!=='end'});onText(text,{generated:generated.length,position:engine.position,ms:engine.stats.lastMs});
            return {text,generated,promptTokens:tokens.length,position:engine.position,stop,seconds:(performance.now()-started)/1000};
        }catch(error){
            // A layer may have changed recurrent state before Stop was observed.
            // Rebuild from visible conversation next time; never reuse partial state.
            await this.drop();
            if(error.name==='AbortError'){const text=this.tokenizer.decode(generated,{stream:true});onText(text,{generated:generated.length,position:0});return {text,generated,promptTokens:tokens.length,stop:'stopped',seconds:(performance.now()-started)/1000};}
            throw error;
        }finally{this.busy=false;}
    }
}
