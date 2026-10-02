import {Ops,grid} from './ops.js';
import {validateGeometry} from './model.js';
import {requireInteger} from './backend.js';
import {WeightResidency} from './residency.js';
import {prefillChunks} from './prefill.js';

export class StrataEngine {
    static async create(store,backend,options={}) {
        const engine=new StrataEngine(store,backend,options);
        try {await engine.initialize();return engine;}catch(error){await engine.dispose();throw error;}
    }
    constructor(store,backend,{cpuBackend=null,cacheBytes=64*1024*1024,cpuCacheBytes=64*1024*1024,gpuExperts=8,weightBudgetBytes=0,expertRamBytes=1024**3,tileRows=256}={}) {
        this.store=store;this.g=validateGeometry(store.config);this.b=backend;this.cpu=cpuBackend;
        this.ops=new Ops(backend,store,{cacheBytes,tileRows});this.cpuOps=cpuBackend?new Ops(cpuBackend,store,{cacheBytes:cpuCacheBytes,tileRows}):null;
        this.weightBudgetBytes=requireInteger(weightBudgetBytes,'GPU weight budget',0,Number.MAX_SAFE_INTEGER);this.expertRamBytes=requireInteger(expertRamBytes,'expert RAM budget',0,Number.MAX_SAFE_INTEGER);
        this.gpuExperts=requireInteger(gpuExperts,'GPU expert slots');this.hot=new Map();this.frequency=new Map();
        this.position=0;this.previous=[];this.states=[];this.persistent=[];this.busy=false;this.failed=false;this.disposed=false;
        this.checkpointOwner=crypto.randomUUID();
        this.stats={tokens:0,gpuExperts:0,cpuExperts:0,promotions:0,routes:[],lastMs:0};
    }
    async keep(n,type='f32') {const b=await this.b.alloc(n,type);this.persistent.push(b);return b;}
    assertIdle(message) {if(this.disposed)throw Error('Session is disposed');if(this.busy)throw Error(message);}
    selectGpuExperts() {
        if(!this.cpu)return;
        // Choose once per token. Updating an LRU after each layer lets a sequential
        // layer scan evict every promoted expert before it is used again.
        const candidates=[...this.frequency].filter(([,count])=>count>=2);
        candidates.sort((a,b)=>b[1]-a[1]||Number(this.hot.has(b[0]))-Number(this.hot.has(a[0])));
        const next=new Map(candidates.slice(0,this.gpuExperts).map(([key])=>[key,true]));
        for(const key of next.keys())if(!this.hot.has(key))this.stats.promotions++;
        this.hot=next;
    }
    expectedShapes() {
        const g=this.g,hc=g.width*g.streams,c=(2*g.keyHeads+g.valueHeads)*g.ssmDim, specs=[];
        const add=(name,shape)=>specs.push([name,shape]);
        const gr=(prefix,inject=true)=>{add(prefix+'norm.weight',[hc]);add(prefix+'down.weight',[hc,g.rank]);add(prefix+'up.weight',[g.rank,hc]);if(inject)add(prefix+'inject.weight',[hc,g.streams]);};
        add('token_embd.weight',[g.width,g.vocab]);add('output.weight',[g.width,g.vocab]);gr('output_hc_',false);
        for(let l=0;l<g.layers;l++) {
            const p=`blk.${l}.`;gr(p+'hc_attn_');gr(p+'hc_ffn_');
            add(p+'ffn_gate_inp.weight',[g.width,g.experts]);add(p+'ffn_gate_inp_shexp.weight',[g.width]);
            for(const role of ['gate','up','down'])add(p+'ffn_'+role+'_shexp.weight',role==='down'?[g.ff,g.width]:[g.width,g.ff]);
            for(let e=0;e<g.experts;e++)for(const role of ['gate','up','down'])add(p+`expert.${e}.${role}`,role==='down'?[g.ff,g.width]:[g.width,g.ff]);
            if(l%g.qsaInterval===g.qsaInterval-1) {
                add(p+'attn_q.weight',[g.width,2*g.heads*g.headDim]);add(p+'attn_k.weight',[g.width,g.kvHeads*g.headDim]);add(p+'attn_v.weight',[g.width,g.kvHeads*g.headDim]);add(p+'attn_output.weight',[g.heads*g.headDim,g.width]);add(p+'attn_q_norm.weight',[g.headDim]);add(p+'attn_k_norm.weight',[g.headDim]);
            } else {
                add(p+'attn_qkv.weight',[g.width,c]);add(p+'attn_gate.weight',[g.width,g.valueHeads*g.ssmDim]);add(p+'ssm_out.weight',[g.valueHeads*g.ssmDim,g.width]);add(p+'ssm_conv1d.weight',[4,c]);
                add(p+'ssm_alpha.weight',[g.width,g.valueHeads]);add(p+'ssm_beta.weight',[g.width,g.valueHeads]);add(p+'ssm_a',[g.valueHeads]);add(p+'ssm_dt.bias',[g.valueHeads]);add(p+'ssm_norm.weight',[g.ssmDim]);
            }
        }
        const p=`blk.${g.pleLayer}.ple_`;add(p+'key.weight',[g.width,hc]);add(p+'value.weight',[g.width,g.width]);
        for(const suffix of ['norm_key','norm_query','norm_conv'])add(p+suffix+'.weight',[hc]);add(p+'conv1d.weight',[g.pleTaps,hc]);
        return specs;
    }
    async initialize() {
        // Shape gates run before GPU/CPU allocations or the first token.
        for(const [name,expected] of this.expectedShapes()) {
            const shape=[...this.store.describe(name).shape];while(shape.length>1&&shape.at(-1)===1)shape.pop();
            const want=[...expected];while(want.length>1&&want.at(-1)===1)want.pop();
            if(String(shape)!==String(want))throw Error(`${name}: expected [${expected}], found [${shape}]`);
        }
        if(this.weightBudgetBytes){
            if(this.b.kind!=='webgpu')throw Error('GPU weight residency requires the WebGPU backend');
            const dense=this.expectedShapes().map(([name])=>name).filter(name=>name!=='token_embd.weight'&&!name.includes('.expert.'));
            this.ops.residency=new WeightResidency(this.ops,dense,this.weightBudgetBytes);
        }
        this.store.setExpertCacheBudget?.(this.expertRamBytes);
        const g=this.g,c=(2*g.keyHeads+g.valueHeads)*g.ssmDim;
        this.residual=await this.keep(g.width*g.streams);
        this.pleHistory=await this.keep((g.pleTaps-1)*g.pleDilation*g.width*g.streams);
        for(let l=0;l<g.layers;l++) {
            this.states.push(l%g.qsaInterval===g.qsaInterval-1
                ?{keys:await this.keep(g.context*g.kvHeads*g.headDim),values:await this.keep(g.context*g.kvHeads*g.headDim)}
                :{state:await this.keep(g.ssmDim*g.valueHeads*g.ssmDim),conv:await this.keep(c*3)});
        }
    }
    async hc(prefix,inject=true) {
        const {g,ops:o,b}=this;const hc=g.width*g.streams;
        const norm=await o.norm(this.residual,await o.vector(prefix+'norm.weight'),g.width,g.streams,g.streams);
        const down=await o.mat(prefix+'down.weight',norm),lo=await o.elem(down,down,1,1/g.streams);
        const gate=await o.mat(prefix+'up.weight',lo),mixed=await o.alloc(g.width);
        await b.run('strata_hc_mix',{Norm:norm,Gate:gate,Mixed:mixed,width:g.width,streams:g.streams},grid(g.width));
        return {mixed,inject:inject?await o.mat(prefix+'inject.weight',norm):null};
    }
    async hcWrite(output,inject) {const g=this.g;await this.b.run('strata_hc_write',{Residual:this.residual,Output:output,Inject:inject,width:g.width,streams:g.streams},grid(g.width*g.streams));}
    async ple(token) {
        const {g,ops:o,b}=this,p=`blk.${g.pleLayer}.ple_`,hc=g.width*g.streams;
        const emb=await o.alloc(await this.store.pleRows(token,this.previous));
        const rawKey=await o.mat(p+'key.weight',emb),key=await o.norm(rawKey,await o.vector(p+'norm_key.weight'),g.width,g.streams,g.streams);
        const query=await o.norm(this.residual,await o.vector(p+'norm_query.weight'),g.width,g.streams,g.streams);
        const value=await o.mat(p+'value.weight',emb),gate=await o.alloc(g.streams),gated=await o.alloc(hc);
        await b.run('strata_ple_gate',{Key:key,Query:query,Gate:gate,width:g.width,streams:g.streams},[g.streams,1,1]);
        await b.run('strata_ple_broadcast',{Value:value,Gate:gate,Y:gated,width:g.width,streams:g.streams},grid(hc));
        const norm=await o.norm(gated,await o.vector(p+'norm_conv.weight'),g.width,g.streams,g.streams),conv=await o.alloc(hc);
        await b.run('strata_conv',{History:this.pleHistory,X:norm,W:await o.vector(p+'conv1d.weight'),Y:conv,channels:hc,taps:g.pleTaps,dilation:g.pleDilation,activation:1},grid(hc));
        await b.run('strata_ple_add',{Residual:this.residual,Gated:gated,Conv:conv,n:hc},grid(hc));
    }
    async gdn(layer,x) {
        const {g,ops:o,b}=this,p=`blk.${layer}.`,st=this.states[layer],qk=g.ssmDim*g.keyHeads,vn=g.ssmDim*g.valueHeads,c=2*qk+vn;
        const qkv=await o.mat(p+'attn_qkv.weight',x),convolved=await o.alloc(c);
        await b.run('strata_conv',{History:st.conv,X:qkv,W:await o.vector(p+'ssm_conv1d.weight'),Y:convolved,channels:c,taps:4,dilation:1,activation:1},grid(c));
        const qr=await o.slice(convolved,qk),kr=await o.slice(convolved,qk,1,qk,qk),v=await o.slice(convolved,vn,1,vn,2*qk);
        const q=await o.norm(qr,qr,g.ssmDim,g.keyHeads,1,1,1/Math.sqrt(g.ssmDim)),k=await o.norm(kr,kr,g.ssmDim,g.keyHeads,1,1);
        const alpha=await o.mat(p+'ssm_alpha.weight',x),beta=await o.mat(p+'ssm_beta.weight',x),decay=await o.alloc(g.valueHeads),strength=await o.alloc(g.valueHeads);
        await b.run('strata_gdn_gates',{Alpha:alpha,Beta:beta,A:await o.vector(p+'ssm_a'),Dt:await o.vector(p+'ssm_dt.bias'),Decay:decay,Strength:strength,heads:g.valueHeads},grid(g.valueHeads));
        const y=await o.alloc(vn);
        await b.run('strata_gdn_step',{State:st.state,Q:q,K:k,V:v,Decay:decay,Beta:strength,Y:y,dim:g.ssmDim,keyHeads:g.keyHeads,valueHeads:g.valueHeads},grid(vn));
        const z=await o.mat(p+'attn_gate.weight',x),norm=await o.norm(y,await o.vector(p+'ssm_norm.weight'),g.ssmDim,g.valueHeads);
        return o.mat(p+'ssm_out.weight',await o.elem(norm,z,4));
    }
    async rotate(x,heads) {
        const {g,ops:o,b}=this,cos=new Float32Array(g.rotary/2),sin=new Float32Array(g.rotary/2);
        // Position tables are host-side in upstream Strata too; kernels perform rotation.
        for(let i=0;i<cos.length;i++){const theta=this.position/g.ropeBase**(2*i/g.rotary);cos[i]=Math.cos(theta);sin[i]=Math.sin(theta);}
        const y=await o.alloc(x.length);await b.run('strata_rope',{X:x,Cos:await o.alloc(cos),Sin:await o.alloc(sin),Y:y,dim:g.headDim,heads,rotary:g.rotary},grid(x.length));return y;
    }
    async qsa(layer,x) {
        const {g,ops:o,b}=this,p=`blk.${layer}.`,st=this.states[layer];
        const full=await o.mat(p+'attn_q.weight',x),qr=await o.slice(full,g.headDim,g.heads,g.headDim*2);
        const q=await this.rotate(await o.norm(qr,await o.vector(p+'attn_q_norm.weight'),g.headDim,g.heads),g.heads);
        const kr=await o.mat(p+'attn_k.weight',x),k=await this.rotate(await o.norm(kr,await o.vector(p+'attn_k_norm.weight'),g.headDim,g.kvHeads),g.kvHeads),v=await o.mat(p+'attn_v.weight',x);
        await b.run('strata_kv_append',{K:k,V:v,Keys:st.keys,Values:st.values,width:g.kvHeads*g.headDim,position:this.position},grid(g.kvHeads*g.headDim));
        // At <=2048 context all causal cells fit the upstream sparse selection budget.
        const cells=await o.alloc(Int32Array.from({length:this.position+1},(_,i)=>i),'i32'),attn=await o.alloc(g.heads*g.headDim),gated=await o.alloc(g.heads*g.headDim);
        await b.run('strata_attention',{Q:q,Keys:st.keys,Values:st.values,Cells:cells,Y:attn,dim:g.headDim,heads:g.heads,kvHeads:g.kvHeads,count:this.position+1,scale:1/Math.sqrt(g.headDim)},[g.heads,Math.ceil(g.headDim/64),1]);
        await b.run('strata_attention_gate',{Attention:attn,FullQ:full,Y:gated,dim:g.headDim,heads:g.heads},grid(g.heads*g.headDim));
        return o.mat(p+'attn_output.weight',gated);
    }
    async expert(o,prefix,x,shared=false,tokens=1) {
        const name=role=>shared?prefix+'ffn_'+role+'_shexp.weight':prefix+role;
        const gate=await o.mat(name('gate'),x,tokens),up=await o.mat(name('up'),x,tokens),activation=await o.elem(gate,up,3);
        return o.mat(name('down'),activation,tokens);
    }
    async moe(layer,x) {
        const {g,ops:o,b}=this,p=`blk.${layer}.`;
        const logits=await o.mat(p+'ffn_gate_inp.weight',x),ids=await o.alloc(g.topK,'i32'),weights=await o.alloc(g.topK);
        await b.run('strata_router',{Logits:logits,Ids:ids,Weights:weights,experts:g.experts,topK:g.topK},[1,1,1]);
        const selected=await b.read(ids,'i32');
        if(selected.some(id=>id<0||id>=g.experts)||new Set(selected).size!==g.topK)throw Error('Invalid expert routing result');
        const resident=o.residency;
        if(resident)for(const id of selected)resident.note(`${layer}:${id}`);
        const parts=await o.alloc(g.width*g.topK),jobs=Array.from(selected,(id,index)=>({id,index,key:`${layer}:${id}`,gpu:!this.cpu||(resident?resident.hasExpert(`${layer}:${id}`):this.hot.has(`${layer}:${id}`))}));
        const cpuX=jobs.some(j=>!j.gpu)?await b.read(x):null;
        this.stats.routes.push({layer,ids:Array.from(selected),devices:jobs.map(j=>j.gpu?b.kind:'wasm')});
        // The two independent command queues execute concurrently. Each WASM module
        // has a single ordered stream and persistent pooled weights.
        const lanes=await Promise.allSettled([
            (async()=>{for(const job of jobs.filter(j=>j.gpu)){await resident?.admit(job.key);const result=await this.expert(o,p+`expert.${job.id}.`,x);await b.copy(result,parts,g.width,0,job.index*g.width);this.stats.gpuExperts+=b.kind==='webgpu'?1:0;this.stats.cpuExperts+=b.kind==='wasm'?1:0;}})(),
            (async()=>{for(const job of jobs.filter(j=>!j.gpu)){const cx=await this.cpuOps.alloc(cpuX),result=await this.expert(this.cpuOps,p+`expert.${job.id}.`,cx);await b.write(parts,await this.cpu.read(result),job.index*g.width);await this.cpuOps.clearTemporary();this.stats.cpuExperts++;}})()
        ]);
        const failure=lanes.find(lane=>lane.status==='rejected');if(failure)throw failure.reason;
        if(resident&&this.cpu)for(const job of jobs.filter(j=>!j.gpu))if((resident.frequency.get(job.key)||0)>=2&&await resident.admit(job.key))this.stats.promotions++;
        if(this.cpu)for(const job of jobs)this.frequency.set(job.key,(this.frequency.get(job.key)||0)+1);
        const shared=await this.expert(o,p,x,true),sharedGate=await o.mat(p+'ffn_gate_inp_shexp.weight',x),y=await o.alloc(g.width);
        await b.run('strata_moe_combine',{Parts:parts,Weights:weights,Shared:shared,SharedGate:sharedGate,Y:y,width:g.width,count:g.topK},grid(g.width));return y;
    }
    residencyInfo(){return {...(this.ops.residency?.info()||{streamingCacheBytes:this.ops.used,hits:this.ops.hits,misses:this.ops.misses}),hostExpertCacheBytes:this.store.expertCacheUsed||0,hostExpertCache:this.store.expertCacheStats};}
    async prefill(tokens,options={}){
        this.assertIdle('A session operation is already running');if(this.failed)throw Error('Session failed; reset it before continuing');
        tokens=Array.from(tokens);if(!tokens.length)throw Error('Prompt is empty');for(const t of tokens)requireInteger(t,'token',0,this.g.vocab-1);
        if(this.position+tokens.length>this.g.context)throw Error('Context capacity reached');
        requireInteger(options.chunkSize??16,'prompt chunk size',1,64);if(options.logits&&options.predict===false)throw Error('Logits require prediction');
        options.signal?.throwIfAborted();this.busy=true;
        try{return await prefillChunks(this,tokens,options);}catch(error){this.failed=true;throw error;}finally{this.busy=false;}
    }
    async step(token,{logits=false,predict=true,signal,onProgress}={}) {
        this.assertIdle('A session operation is already running');if(this.failed)throw Error('Session failed; reset it before continuing');
        signal?.throwIfAborted();
        if(logits&&!predict)throw Error('Logits require prediction');
        requireInteger(token,'token',0,this.g.vocab-1);if(this.position>=this.g.context)throw Error('Context capacity reached; reset or create a larger session');
        this.busy=true;const start=performance.now();this.stats.routes=[];
        try {
            const {g,ops:o,b}=this,row=await o.alloc(await this.store.readValues('token_embd.weight',token*g.width,g.width));
            await b.run('strata_embed',{Row:row,Residual:this.residual,width:g.width,streams:g.streams},grid(g.width*g.streams));
            for(let l=0;l<g.layers;l++) {
                if(l===g.pleLayer)await this.ple(token);
                const attn=await this.hc(`blk.${l}.hc_attn_`),mixed=l%g.qsaInterval===g.qsaInterval-1?await this.qsa(l,attn.mixed):await this.gdn(l,attn.mixed);
                await this.hcWrite(mixed,attn.inject);
                const ffn=await this.hc(`blk.${l}.hc_ffn_`);await this.hcWrite(await this.moe(l,ffn.mixed),ffn.inject);
                await o.clearTemporary();
                onProgress?.(l+1,g.layers);signal?.throwIfAborted();
            }
            const result={position:this.position};
            // Prompt tokens before the last only update state. Their full-vocabulary
            // projection, argmax and readback would be discarded by the caller.
            if(predict) {
                const head=await this.hc('output_hc_',false),values=await o.mat('output.weight',head.mixed),next=await o.alloc(1,'i32');
                await b.run('strata_argmax',{Logits:values,Token:next,count:g.vocab},[1,1,1]);
                result.token=(await b.read(next,'i32'))[0];requireInteger(result.token,'generated token',0,g.vocab-1);
                if(logits)result.logits=await b.read(values);
                if(result.logits?.some(v=>!Number.isFinite(v)))throw Error('Non-finite logits');
            }
            await o.clearTemporary();signal?.throwIfAborted();this.selectGpuExperts();this.previous.push(token);if(this.previous.length>2)this.previous.shift();this.position++;this.stats.tokens++;this.stats.lastMs=performance.now()-start;return result;
        } catch(error) {this.failed=true;throw error;} finally {this.busy=false;}
    }
    async reset() {
        this.assertIdle('Cannot reset during a decode step or state operation');this.busy=true;
        try {
            await this.ops.clearTemporary();if(this.cpuOps)await this.cpuOps.clearTemporary();
            for(const buffer of this.persistent){if(this.b.zero)await this.b.zero(buffer);else await this.b.write(buffer,new Float32Array(buffer.length));}this.position=0;this.previous=[];this.failed=false;
        } catch(error) {this.failed=true;throw error;}finally {this.busy=false;}
    }
    async checkpoint() {
        this.assertIdle('Checkpoint requires an idle session');if(this.failed)throw Error('Checkpoint requires a healthy session');this.busy=true;
        try {return {owner:this.checkpointOwner,geometry:JSON.stringify(this.g),position:this.position,previous:[...this.previous],buffers:await Promise.all(this.persistent.map(b=>this.b.read(b)))};}
        finally {this.busy=false;}
    }
    async restore(state) {
        this.assertIdle('Cannot restore during decode or state operation');
        if(state.owner!==this.checkpointOwner||state.geometry!==JSON.stringify(this.g)||state.buffers.length!==this.persistent.length||state.buffers.some((a,i)=>!(a instanceof Float32Array)||a.length!==this.persistent[i].length||a.some(x=>!Number.isFinite(x)))||!Number.isInteger(state.position)||state.position<0||state.position>this.g.context||state.previous.length!==Math.min(state.position,2)||state.previous.some(x=>!Number.isInteger(x)||x<0||x>=this.g.vocab))throw Error('Checkpoint does not match this model/session geometry');
        this.busy=true;
        try {
            await this.ops.clearTemporary();if(this.cpuOps)await this.cpuOps.clearTemporary();
            for(let i=0;i<this.persistent.length;i++)await this.b.write(this.persistent[i],state.buffers[i]);this.position=state.position;this.previous=[...state.previous];this.failed=false;
        } catch(error) {this.failed=true;throw error;}finally {this.busy=false;}
    }
    async dispose() {
        if(this.disposed)return;this.assertIdle('Cannot dispose during a decode step or state operation');this.busy=true;
        try {await this.ops.dispose();if(this.cpuOps)await this.cpuOps.dispose();for(const b of this.persistent)await this.b.free(b);this.persistent=[];this.store.clearExpertCache?.();this.disposed=true;}
        catch(error) {this.failed=true;throw error;}finally {this.busy=false;}
    }
}
