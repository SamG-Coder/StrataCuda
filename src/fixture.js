import {MemoryStore} from './model.js';
// Untrained weights exercise the full architecture; never presented as a language model.
export function createFixture(seed=1234) {
    const g={width:32,streams:2,rank:8,layers:4,qsaInterval:4,ssmDim:8,keyHeads:2,valueHeads:4,
        heads:4,kvHeads:2,headDim:8,rotary:8,experts:6,topK:2,ff:24,vocab:40,context:32,
        pleLayer:1,pleHeads:4,pleHeadDim:8,pleTaps:4,pleDilation:3,ropeBase:1e7,eos:39};
    let state=seed>>>0;const rnd=()=>{state=(Math.imul(state,1664525)+1013904223)>>>0;return state/4294967296*2-1;};
    const tensors=new Map();
    const add=(name,shape,mode='weight')=>{const n=shape.reduce((a,b)=>a*b,1),data=new Float32Array(n);for(let i=0;i<n;i++)data[i]=mode==='norm'?1+rnd()*.05:mode==='negative'?-.1-Math.abs(rnd()):rnd()*(mode==='embedding'?.4:.3/Math.sqrt(shape[0]));tensors.set(name,{format:'f32',shape,data});};
    const hc=g.width*g.streams;
    const gr=prefix=>{add(prefix+'norm.weight',[hc],'norm');add(prefix+'down.weight',[hc,g.rank]);add(prefix+'up.weight',[g.rank,hc]);add(prefix+'inject.weight',[hc,g.streams]);};
    add('token_embd.weight',[g.width,g.vocab],'embedding');gr('output_hc_');add('output.weight',[g.width,g.vocab]);
    for(let l=0;l<g.layers;l++) {
        const p=`blk.${l}.`;gr(p+'hc_attn_');gr(p+'hc_ffn_');
        add(p+'ffn_gate_inp.weight',[g.width,g.experts]);add(p+'ffn_gate_inp_shexp.weight',[g.width,1]);
        for(const role of ['gate','up','down'])add(p+'ffn_'+role+'_shexp.weight',role==='down'?[g.ff,g.width]:[g.width,g.ff]);
        for(let e=0;e<g.experts;e++)for(const role of ['gate','up','down'])add(p+`expert.${e}.${role}`,role==='down'?[g.ff,g.width]:[g.width,g.ff]);
        if(l%g.qsaInterval===g.qsaInterval-1) {
            add(p+'attn_q.weight',[g.width,2*g.heads*g.headDim]);add(p+'attn_k.weight',[g.width,g.kvHeads*g.headDim]);add(p+'attn_v.weight',[g.width,g.kvHeads*g.headDim]);add(p+'attn_output.weight',[g.heads*g.headDim,g.width]);
            add(p+'attn_q_norm.weight',[g.headDim],'norm');add(p+'attn_k_norm.weight',[g.headDim],'norm');
        } else {
            const c=(2*g.keyHeads+g.valueHeads)*g.ssmDim;
            add(p+'attn_qkv.weight',[g.width,c]);add(p+'attn_gate.weight',[g.width,g.ssmDim*g.valueHeads]);add(p+'ssm_out.weight',[g.ssmDim*g.valueHeads,g.width]);
            add(p+'ssm_alpha.weight',[g.width,g.valueHeads]);add(p+'ssm_beta.weight',[g.width,g.valueHeads]);
            add(p+'ssm_conv1d.weight',[4,c]);add(p+'ssm_norm.weight',[g.ssmDim],'norm');add(p+'ssm_dt.bias',[g.valueHeads]);add(p+'ssm_a',[g.valueHeads],'negative');
        }
    }
    const p=`blk.${g.pleLayer}.ple_`;
    add(p+'key.weight',[g.width,hc]);add(p+'value.weight',[g.width,g.width]);
    for(const n of ['norm_key','norm_query','norm_conv'])add(p+n+'.weight',[hc],'norm');add(p+'conv1d.weight',[g.pleTaps,hc]);
    add('per_layer_token_embd.weight',[g.pleHeadDim,88],'embedding');
    const constants={multipliers:['23703573157769','20109073645365','8052911324071'],vocab:[17,19,23,29],offsets:[0,17,36,59]};
    return new MemoryStore(g,tensors,constants);
}
