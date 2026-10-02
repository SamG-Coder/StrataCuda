// Independent scalar/F64 equations from upstream Strata's gdn.cu, gr.cu,
// ple.cu, qsa.cu and layer.cpp. Never calls the ported kernels or runtime.
export const sigmoid=x=>1/(1+Math.exp(-x));
export const silu=x=>x*sigmoid(x);
export const dot=(a,b)=>a.reduce((s,x,i)=>s+x*b[i],0);
export function mat(weights,x) {return Float64Array.from({length:weights.length/x.length},(_,r)=>dot(weights.subarray(r*x.length,(r+1)*x.length),x));}
export function norm(x,weights,width,{l2=false,scale=1}={}) {
    const y=new Float64Array(x.length);for(let r=0;r<x.length/width;r++) {
        const row=x.subarray(r*width,(r+1)*width),inverse=scale/Math.sqrt(dot(row,row)/(l2?1:width)+1e-6);
        for(let i=0;i<width;i++)y[r*width+i]=row[i]*inverse*(l2?1:weights[(r*width+i)%weights.length]);
    }return y;
}
export function router(logits,k) {
    const max=Math.max(...logits),p=Array.from(logits,x=>Math.exp(x-max)),total=p.reduce((a,b)=>a+b,0),prob=p.map(x=>x/total);
    const ids=Array.from(logits,(_,i)=>i).sort((a,b)=>prob[b]-prob[a]||a-b).slice(0,k),denom=Math.max(ids.reduce((s,i)=>s+prob[i],0),2**-14);
    return {ids,weights:ids.map(i=>prob[i]/denom)};
}
export function conv(history,x,weights,taps,dilation,activate) {
    const span=(taps-1)*dilation,y=new Float64Array(x.length);
    for(let c=0;c<x.length;c++) {
        let value=x[c]*weights[c*taps+taps-1];
        for(let tap=0;tap<taps-1;tap++)value+=history[c*span+tap*dilation]*weights[c*taps+tap];
        y[c]=activate?silu(value):value;
        history.copyWithin(c*span,c*span+1,(c+1)*span);if(span)history[(c+1)*span-1]=x[c];
    }return y;
}
export function gdn(state,q,k,v,decay,beta,dim,keyHeads,valueHeads) {
    const out=new Float64Array(v.length);
    for(let h=0;h<valueHeads;h++)for(let j=0;j<dim;j++) {
        const offset=h*dim+j,source=h%keyHeads;let predicted=0;
        for(let i=0;i<dim;i++){const ix=(i*valueHeads+h)*dim+j;state[ix]*=decay[h];predicted+=state[ix]*k[source*dim+i];}
        const update=beta[h]*(v[offset]-predicted);
        for(let i=0;i<dim;i++){const ix=(i*valueHeads+h)*dim+j;state[ix]+=k[source*dim+i]*update;out[offset]+=state[ix]*q[source*dim+i];}
    }return out;
}
export function rope(x,dim,rotary,pos,base) {
    const y=Float64Array.from(x),half=rotary/2;
    for(let h=0;h<x.length/dim;h++)for(let i=0;i<half;i++) {const a=x[h*dim+i],b=x[h*dim+half+i],angle=pos/base**(2*i/rotary);y[h*dim+i]=a*Math.cos(angle)-b*Math.sin(angle);y[h*dim+half+i]=a*Math.sin(angle)+b*Math.cos(angle);}
    return y;
}
export function attention(q,keys,values,cells,dim,heads,kvHeads) {
    const y=new Float64Array(heads*dim);
    for(let h=0;h<heads;h++) {
        const kh=Math.floor(h/(heads/kvHeads)),scores=Array.from(cells,t=>dot(q.subarray(h*dim,(h+1)*dim),keys.subarray((t*kvHeads+kh)*dim,(t*kvHeads+kh+1)*dim))/Math.sqrt(dim));
        const max=Math.max(...scores),probs=scores.map(s=>Math.exp(s-max)),sum=probs.reduce((a,b)=>a+b,0);
        for(let d=0;d<dim;d++)for(let t=0;t<cells.length;t++)y[h*dim+d]+=probs[t]/sum*values[(cells[t]*kvHeads+kh)*dim+d];
    }return y;
}

export class ReferenceEngine {
    constructor(store) {
        this.store=store;this.g=store.config;this.position=0;this.previous=[];const g=this.g;
        this.states=Array.from({length:g.layers},()=>({state:new Float64Array(g.ssmDim*g.valueHeads*g.ssmDim),conv:new Float64Array((2*g.keyHeads+g.valueHeads)*g.ssmDim*3),keys:new Float64Array(g.context*g.kvHeads*g.headDim),values:new Float64Array(g.context*g.kvHeads*g.headDim)}));
        this.history=new Float64Array((g.pleTaps-1)*g.pleDilation*g.width*g.streams);
    }
    weight(name) {return this.store.describe(name).data;}
    project(name,x) {return mat(this.weight(name),x);}
    hc(prefix) {
        const g=this.g,x=norm(this.r,this.weight(prefix+'norm.weight'),g.width),lo=this.project(prefix+'down.weight',x).map(v=>silu(v/g.streams)),gate=this.project(prefix+'up.weight',lo);
        return {x,mixed:Float64Array.from({length:g.width},(_,d)=>{let s=0;for(let c=0;c<g.streams;c++)s+=x[c*g.width+d]*sigmoid(gate[c*g.width+d]);return s/g.streams;})};
    }
    write(y,inject) {const g=this.g;for(let i=0;i<this.r.length;i++)this.r[i]+=y[i%g.width]*2*sigmoid(inject[Math.floor(i/g.width)]/g.streams);}
    expert(p,x,shared=false) {const name=r=>shared?p+'ffn_'+r+'_shexp.weight':p+r,gate=this.project(name('gate'),x),up=this.project(name('up'),x);return this.project(name('down'),gate.map((v,i)=>silu(v)*up[i]));}
    async step(token) {
        const g=this.g,hc=g.width*g.streams,row=await this.store.readValues('token_embd.weight',token*g.width,g.width);
        this.r=Float64Array.from({length:hc},(_,i)=>row[i%g.width]);
        for(let l=0;l<g.layers;l++) {
            const p=`blk.${l}.`,s=this.states[l];
            if(l===g.pleLayer) {
                const pr=p+'ple_',emb=await this.store.pleRows(token,this.previous),key=norm(this.project(pr+'key.weight',emb),this.weight(pr+'norm_key.weight'),g.width),query=norm(this.r,this.weight(pr+'norm_query.weight'),g.width),value=this.project(pr+'value.weight',emb),gated=new Float64Array(hc);
                for(let c=0;c<g.streams;c++){const score=dot(key.subarray(c*g.width,(c+1)*g.width),query.subarray(c*g.width,(c+1)*g.width))/Math.sqrt(g.width),gate=sigmoid(Math.sign(score)*Math.sqrt(Math.max(Math.abs(score),1e-6)));for(let i=0;i<g.width;i++)gated[c*g.width+i]=value[i]*gate;}
                const n=norm(gated,this.weight(pr+'norm_conv.weight'),g.width),cv=conv(this.history,n,this.weight(pr+'conv1d.weight'),g.pleTaps,g.pleDilation,true);this.r=this.r.map((v,i)=>v+gated[i]+cv[i]);
            }
            const a=this.hc(p+'hc_attn_'),x=a.mixed;let y;
            if(l%g.qsaInterval===g.qsaInterval-1) {
                const full=this.project(p+'attn_q.weight',x),qraw=Float64Array.from({length:g.heads*g.headDim},(_,i)=>full[Math.floor(i/g.headDim)*2*g.headDim+i%g.headDim]);
                const q=rope(norm(qraw,this.weight(p+'attn_q_norm.weight'),g.headDim),g.headDim,g.rotary,this.position,g.ropeBase),k=rope(norm(this.project(p+'attn_k.weight',x),this.weight(p+'attn_k_norm.weight'),g.headDim),g.headDim,g.rotary,this.position,g.ropeBase),v=this.project(p+'attn_v.weight',x);
                s.keys.set(k,this.position*g.kvHeads*g.headDim);s.values.set(v,this.position*g.kvHeads*g.headDim);
                const attn=attention(q,s.keys,s.values,Array.from({length:this.position+1},(_,i)=>i),g.headDim,g.heads,g.kvHeads);
                y=this.project(p+'attn_output.weight',attn.map((v,i)=>v*sigmoid(full[Math.floor(i/g.headDim)*g.headDim*2+g.headDim+i%g.headDim])));
            } else {
                const c=conv(s.conv,this.project(p+'attn_qkv.weight',x),this.weight(p+'ssm_conv1d.weight'),4,1,true),qk=g.ssmDim*g.keyHeads;
                const q=norm(c.subarray(0,qk),null,g.ssmDim,{l2:true,scale:1/Math.sqrt(g.ssmDim)}),k=norm(c.subarray(qk,2*qk),null,g.ssmDim,{l2:true});
                const alpha=this.project(p+'ssm_alpha.weight',x),beta=this.project(p+'ssm_beta.weight',x).map(sigmoid),wa=this.weight(p+'ssm_a'),dt=this.weight(p+'ssm_dt.bias'),decay=alpha.map((a,i)=>Math.exp(wa[i]*Math.log1p(Math.exp(a+dt[i]))));
                const out=gdn(s.state,q,k,c.subarray(2*qk),decay,beta,g.ssmDim,g.keyHeads,g.valueHeads),z=this.project(p+'attn_gate.weight',x),n=norm(out,this.weight(p+'ssm_norm.weight'),g.ssmDim);
                y=this.project(p+'ssm_out.weight',n.map((v,i)=>v*sigmoid(z[i])));
            }
            this.write(y,this.project(p+'hc_attn_inject.weight',a.x));
            const f=this.hc(p+'hc_ffn_'),routing=router(this.project(p+'ffn_gate_inp.weight',f.mixed),g.topK),shared=this.expert(p,f.mixed,true),sg=sigmoid(this.project(p+'ffn_gate_inp_shexp.weight',f.mixed)[0]);
            const output=shared.map(v=>v*sg);for(let i=0;i<g.topK;i++){const part=this.expert(p+`expert.${routing.ids[i]}.`,f.mixed);for(let d=0;d<g.width;d++)output[d]+=part[d]*routing.weights[i];}
            this.write(output,this.project(p+'hc_ffn_inject.weight',f.x));
        }
        const logits=this.project('output.weight',this.hc('output_hc_').mixed),tokenOut=Array.from(logits).indexOf(Math.max(...logits));this.position++;this.previous.push(token);if(this.previous.length>2)this.previous.shift();return {token:tokenOut,logits};
    }
}
