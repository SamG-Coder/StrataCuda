import {grid} from './ops.js';

// Layer-major prompt execution: each projection and each distinct routed expert
// reads its weights once for the whole chunk. Tensor arithmetic stays in .cu.
class PromptBatch {
    constructor(engine,tokens){this.e=engine;this.g=engine.g;this.o=engine.ops;this.b=engine.b;this.tokens=tokens;this.n=tokens.length;this.position=engine.position;}
    mat(name,x){return this.o.mat(name,x,this.n);}
    async hc(prefix,inject=true){
        const {g,o,b,n}=this,norm=await o.norm(this.residual,await o.vector(prefix+'norm.weight'),g.width,n*g.streams,g.streams);
        const down=await this.mat(prefix+'down.weight',norm),lo=await o.elem(down,down,1,1/g.streams),gate=await this.mat(prefix+'up.weight',lo),mixed=await o.alloc(n*g.width);
        await b.run('strata_hc_mix_batch',{Norm:norm,Gate:gate,Mixed:mixed,width:g.width,streams:g.streams,tokens:n},grid(n*g.width));
        return {mixed,inject:inject?await this.mat(prefix+'inject.weight',norm):null};
    }
    async write(output,inject){const {g,b,n}=this;await b.run('strata_hc_write_batch',{Residual:this.residual,Output:output,Inject:inject,width:g.width,streams:g.streams,tokens:n},grid(n*g.width*g.streams));}
    async ple(){
        const {g,o,b,n,e}=this,p=`blk.${g.pleLayer}.ple_`,hc=g.width*g.streams,emb=await o.alloc(n*g.width),previous=[...e.previous];
        for(let i=0;i<n;i++){const row=await o.ple(this.tokens[i],previous);await b.copy(row,emb,g.width,0,i*g.width);await o.release(row);previous.push(this.tokens[i]);if(previous.length>2)previous.shift();}
        const rawKey=await this.mat(p+'key.weight',emb),key=await o.norm(rawKey,await o.vector(p+'norm_key.weight'),g.width,n*g.streams,g.streams);
        const query=await o.norm(this.residual,await o.vector(p+'norm_query.weight'),g.width,n*g.streams,g.streams),value=await this.mat(p+'value.weight',emb),gate=await o.alloc(n*g.streams),gated=await o.alloc(n*hc);
        await b.run('strata_ple_gate',{Key:key,Query:query,Gate:gate,width:g.width,streams:n*g.streams},[n*g.streams,1,1]);
        await b.run('strata_ple_broadcast_batch',{Value:value,Gate:gate,Y:gated,width:g.width,streams:g.streams,tokens:n},grid(n*hc));
        const norm=await o.norm(gated,await o.vector(p+'norm_conv.weight'),g.width,n*g.streams,g.streams),conv=await o.alloc(n*hc);
        await b.run('strata_conv_batch',{History:e.pleHistory,X:norm,W:await o.vector(p+'conv1d.weight'),Y:conv,channels:hc,taps:g.pleTaps,dilation:g.pleDilation,activation:1,tokens:n},grid(hc));
        await b.run('strata_ple_add',{Residual:this.residual,Gated:gated,Conv:conv,n:n*hc},grid(n*hc));
    }
    async gdn(layer,x){
        const {g,o,b,n,e}=this,p=`blk.${layer}.`,st=e.states[layer],qk=g.ssmDim*g.keyHeads,vn=g.ssmDim*g.valueHeads,c=2*qk+vn;
        const qkv=await this.mat(p+'attn_qkv.weight',x),convolved=await o.alloc(n*c);
        await b.run('strata_conv_batch',{History:st.conv,X:qkv,W:await o.vector(p+'ssm_conv1d.weight'),Y:convolved,channels:c,taps:4,dilation:1,activation:1,tokens:n},grid(c));
        const qr=await o.slice(convolved,qk,n,c),kr=await o.slice(convolved,qk,n,c,qk),v=await o.slice(convolved,vn,n,c,2*qk);
        const q=await o.norm(qr,qr,g.ssmDim,n*g.keyHeads,1,1,1/Math.sqrt(g.ssmDim)),k=await o.norm(kr,kr,g.ssmDim,n*g.keyHeads,1,1);
        const alpha=await this.mat(p+'ssm_alpha.weight',x),beta=await this.mat(p+'ssm_beta.weight',x),decay=await o.alloc(n*g.valueHeads),strength=await o.alloc(n*g.valueHeads),y=await o.alloc(n*vn);
        await b.run('strata_gdn_gates_batch',{Alpha:alpha,Beta:beta,A:await o.vector(p+'ssm_a'),Dt:await o.vector(p+'ssm_dt.bias'),Decay:decay,Strength:strength,heads:g.valueHeads,tokens:n},grid(n*g.valueHeads));
        await b.run('strata_gdn_batch',{State:st.state,Q:q,K:k,V:v,Decay:decay,Beta:strength,Y:y,dim:g.ssmDim,keyHeads:g.keyHeads,valueHeads:g.valueHeads,tokens:n},grid(vn));
        const z=await this.mat(p+'attn_gate.weight',x),norm=await o.norm(y,await o.vector(p+'ssm_norm.weight'),g.ssmDim,n*g.valueHeads);
        return this.mat(p+'ssm_out.weight',await o.elem(norm,z,4));
    }
    async rotate(x,heads){
        return this.o.rotate(x,heads,this.position,this.n);
    }
    async qsa(layer,x){
        const {g,o,b,n,e}=this,p=`blk.${layer}.`,st=e.states[layer];
        const full=await this.mat(p+'attn_q.weight',x),qr=await o.slice(full,g.headDim,n*g.heads,g.headDim*2);
        const q=await this.rotate(await o.norm(qr,await o.vector(p+'attn_q_norm.weight'),g.headDim,n*g.heads),g.heads);
        const kr=await this.mat(p+'attn_k.weight',x),k=await this.rotate(await o.norm(kr,await o.vector(p+'attn_k_norm.weight'),g.headDim,n*g.kvHeads),g.kvHeads),v=await this.mat(p+'attn_v.weight',x);
        await b.run('strata_kv_append_batch',{K:k,V:v,Keys:st.keys,Values:st.values,width:g.kvHeads*g.headDim,position:this.position,tokens:n},grid(n*g.kvHeads*g.headDim));
        const attn=await o.alloc(n*g.heads*g.headDim),gated=await o.alloc(attn.length);
        await b.run('strata_attention_batch',{Q:q,Keys:st.keys,Values:st.values,Y:attn,dim:g.headDim,heads:g.heads,kvHeads:g.kvHeads,position:this.position,scale:1/Math.sqrt(g.headDim)},[g.heads,Math.ceil(g.headDim/64),n]);
        await b.run('strata_attention_gate_batch',{Attention:attn,FullQ:full,Y:gated,dim:g.headDim,heads:g.heads,tokens:n},grid(attn.length));
        return this.mat(p+'attn_output.weight',gated);
    }
    async moe(layer,x){
        const {g,o,b,n,e}=this,p=`blk.${layer}.`,logits=await this.mat(p+'ffn_gate_inp.weight',x),ids=await o.alloc(n*g.topK,'i32'),weights=await o.alloc(n*g.topK);
        await b.run('strata_router_batch',{Logits:logits,Ids:ids,Weights:weights,experts:g.experts,topK:g.topK},[n,1,1]);
        const selected=await b.read(ids,'i32'),groups=new Map(),parts=await o.alloc(n*g.topK*g.width);
        for(let t=0;t<n;t++){
            const row=Array.from(selected.subarray(t*g.topK,(t+1)*g.topK));
            if(row.some(id=>id<0||id>=g.experts)||new Set(row).size!==g.topK)throw Error('Invalid prompt expert routing');
            e.stats.routes.push({position:this.position+t,layer,ids:row,devices:row.map(()=>b.kind)});
            for(let k=0;k<g.topK;k++){const id=row[k];if(!groups.has(id))groups.set(id,[]);groups.get(id).push({token:t,part:t*g.topK+k});}
        }
        await e.store.prefetchExperts?.(layer,[...groups.keys()].filter(id=>!o.residency?.hasExpert(`${layer}:${id}`)));
        for(const [id,jobs] of groups){
            const key=`${layer}:${id}`;o.residency?.note(key,jobs.length);const resident=await o.residency?.admit(key);if(!resident)await e.store.prefetchExpert?.(layer,id);
            if(b.kind==='webgpu'&&e.store.manifest?.experts?.source_type==='Q2_0'){
                const mapping=Int32Array.from(jobs.flatMap(j=>[j.token,j.part]));
                await o.packedExpert(layer,id,x,jobs.length,{output:parts,mapping});
                e.stats[b.kind==='webgpu'?'gpuExperts':'cpuExperts']+=jobs.length;continue;
            }
            const indices=await o.alloc(Int32Array.from(jobs,j=>j.token),'i32'),input=await o.alloc(jobs.length*g.width);
            await b.run('strata_gather_rows',{X:x,Indices:indices,Y:input,width:g.width,rows:jobs.length},grid(input.length));
            const output=await e.expert(o,p+`expert.${id}.`,input,false,jobs.length),destinations=await o.alloc(Int32Array.from(jobs,j=>j.part),'i32');
            await b.run('strata_scatter_rows',{X:output,Indices:destinations,Y:parts,width:g.width,rows:jobs.length},grid(output.length));
            e.stats[b.kind==='webgpu'?'gpuExperts':'cpuExperts']+=jobs.length;
        }
        const shared=await e.expert(o,p,x,true,n),sharedGate=await this.mat(p+'ffn_gate_inp_shexp.weight',x),y=await o.alloc(n*g.width);
        await b.run('strata_moe_combine_batch',{Parts:parts,Weights:weights,Shared:shared,SharedGate:sharedGate,Y:y,width:g.width,count:g.topK,tokens:n},grid(y.length));return y;
    }
    async run({signal,onProgress}){
        const {g,o,b,n,e}=this,hc=g.width*g.streams;
        this.residual=await b.alloc(n*hc);
        try{
            for(let t=0;t<n;t++)await o.embedding(this.tokens[t],this.residual,t*hc);
            for(let l=0;l<g.layers;l++){
                await o.prepareLayer(l);
                if(l===g.pleLayer)await this.ple();
                const attn=await this.hc(`blk.${l}.hc_attn_`),mixed=l%g.qsaInterval===g.qsaInterval-1?await this.qsa(l,attn.mixed):await this.gdn(l,attn.mixed);
                await this.write(mixed,attn.inject);const ffn=await this.hc(`blk.${l}.hc_ffn_`);await this.write(await this.moe(l,ffn.mixed),ffn.inject);
                o.finishLayer(l);
                await o.clearTemporary();onProgress?.(l+1,g.layers);signal?.throwIfAborted();
            }
            await b.copy(this.residual,e.residual,hc,(n-1)*hc,0);
            e.position+=n;e.previous=[...e.previous,...this.tokens].slice(-2);e.stats.tokens+=n;
        }finally{await b.free(this.residual);}
    }
}

export async function prefillChunks(engine,tokens,{signal,onProgress,chunkSize=16,logits=false,predict=true}={}){
    const start=performance.now();engine.stats.routes=[];
    for(let offset=0;offset<tokens.length;offset+=chunkSize){
        signal?.throwIfAborted();const chunk=tokens.slice(offset,offset+chunkSize);
        await new PromptBatch(engine,chunk).run({signal,onProgress:(layer,layers)=>onProgress?.(layer,layers,offset,chunk.length)});
    }
    const result={position:engine.position-1};
    if(predict){const head=await engine.hc('output_hc_',false),values=await engine.ops.mat('output.weight',head.mixed),next=await engine.ops.alloc(1,'i32');
        await engine.b.run('strata_argmax',{Logits:values,Token:next,count:engine.g.vocab},[1,1,1]);result.token=(await engine.b.read(next,'i32'))[0];
        if(!Number.isInteger(result.token)||result.token<0||result.token>=engine.g.vocab)throw Error('Invalid prompt prediction');
        if(logits){result.logits=await engine.b.read(values);if(result.logits.some(v=>!Number.isFinite(v)))throw Error('Non-finite prompt logits');}
    }
    await engine.ops.clearTemporary();signal?.throwIfAborted();engine.stats.lastMs=performance.now()-start;return result;
}
