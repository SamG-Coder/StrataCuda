import test from 'node:test';
import assert from 'node:assert/strict';
import {TextSession} from '../src/text-session.js';
const tokenizer={encode:s=>Array.from(s,c=>c.codePointAt(0)),decode:ids=>String.fromCodePoint(...ids),endTokens:new Set([0])};
function harness(outputs=[65,66,0]){
    const created=[],session=new TextSession(tokenizer,async context=>{
        const calls=[],owner={engine:{g:{context,layers:2},stats:{lastMs:1},position:0,async step(token,options){calls.push({token,predict:options.predict});options.onProgress(1,2);options.signal?.throwIfAborted();this.position++;return options.predict?{token:outputs.shift()??0}:{};}},async dispose(){this.disposed=true;},calls};created.push(owner);return owner;
    });return {session,created};
}
const request=(content='Hi',extra={})=>({messages:[{role:'user',content}],mode:'completion',context:256,maxTokens:2,...extra});
test('text sessions reject over-budget prompts before allocating an engine',async()=>{const {session,created}=harness();await assert.rejects(session.generate(request('a'.repeat(255))),/needs 257 tokens/);assert.equal(created.length,0);});
test('generation predicts only at the last prompt token and stops on EOS',async()=>{const {session,created}=harness([65,0]);const r=await session.generate(request('Hi',{maxTokens:8}));assert.equal(r.text,'A');assert.equal(r.stop,'end');assert.deepEqual(created[0].calls,[{token:72,predict:false},{token:105,predict:true},{token:65,predict:true}]);});
test('prefix reuse consumes the final generated token before extending the prompt',async()=>{const {session,created}=harness([65,66,67]);await session.generate(request('Hi'));const r=await session.generate(request('HiAB!',{maxTokens:1}));assert.equal(created.length,1);assert.deepEqual(created[0].calls.slice(3),[{token:66,predict:false},{token:33,predict:true}]);assert.equal(r.text,'C');});
test('changed context or prefix rebuilds the session instead of mixing states',async()=>{const {session,created}=harness();await session.generate(request());await session.generate(request('Other',{context:512,maxTokens:1}));assert.equal(created.length,2);assert.ok(created[0].disposed);assert.equal(created[1].engine.g.context,512);});
test('stopping inside a layer discards partial state and permits a fresh request',async()=>{const {session,created}=harness(),abort=new AbortController();const r=await session.generate(request('Hi',{signal:abort.signal,onProgress:p=>{if(p.layer===1)abort.abort();}}));assert.equal(r.stop,'stopped');assert.ok(created[0].disposed);assert.equal(session.owner,null);await session.generate(request());assert.equal(created.length,2);});

test('new conversations reset recurrent state while retaining loaded weights',async()=>{
    const {session,created}=harness([65,66,67,68]);await session.generate(request());const owner=created[0];let resets=0;
    owner.engine.reset=async()=>{owner.engine.position=0;resets++;};
    await session.reset();assert.equal(resets,1);assert.equal(owner.disposed,undefined);assert.equal(session.consumed.length,0);
    await session.generate(request('New',{maxTokens:1}));assert.equal(created.length,1);
    await session.generate(request('Other',{maxTokens:1}));assert.equal(resets,2);assert.equal(created.length,1);
    await session.reset({release:true});assert.equal(owner.disposed,true);assert.equal(session.owner,null);
});
