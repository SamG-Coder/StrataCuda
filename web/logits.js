// Bound DOM work without spreading a production vocabulary into function arguments.
// Each bar retains its bucket's peak, so the selected token remains visible.
export function logitBars(logits,token,maxBars=120) {
    if(!Number.isInteger(maxBars)||maxBars<1)throw Error('Invalid logit bar count');
    let min=Infinity,max=-Infinity;
    for(const value of logits) {if(!Number.isFinite(value))throw Error('Non-finite logit');min=Math.min(min,value);max=Math.max(max,value);}
    const stride=Math.max(1,Math.ceil(logits.length/maxBars)),bars=[];
    for(let start=0;start<logits.length;start+=stride) {
        let id=start;
        for(let i=start+1;i<Math.min(start+stride,logits.length);i++)
            if(logits[i]>logits[id]||(i===token&&logits[i]===logits[id]))id=i;
        bars.push({id,value:logits[id],best:id===token,height:5+90*(logits[id]-min)/Math.max(1e-9,max-min)});
    }
    return {min,max,bars};
}
