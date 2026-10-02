import test from 'node:test';
import assert from 'node:assert/strict';
import {RangeSource} from '../src/model-files.js';
test('HTTP model reads validate exact ranges before accepting bytes',async t=>{
    const source=new RangeSource('http://localhost/pack/',{'weights.bin':100});let requested;
    t.mock.method(globalThis,'fetch',async(url,options)=>{requested=options.headers.Range;return new Response(new Uint8Array([7,8]),{status:206,headers:{'Content-Range':'bytes 9-10/100'}});});
    assert.deepEqual(await source.read('weights.bin',9,2),new Uint8Array([7,8]));assert.equal(requested,'bytes=9-10');
    await assert.rejects(source.read('weights.bin',99,2),/Invalid model file range/);
});
test('HTTP model reads reject ignored ranges and incorrect bodies',async t=>{
    const source=new RangeSource('http://localhost/pack/',{'weights.bin':100});
    t.mock.method(globalThis,'fetch',async()=>new Response(new Uint8Array([7,8]),{status:200}));
    await assert.rejects(source.read('weights.bin',9,2),/exact byte ranges/);
    t.mock.method(globalThis,'fetch',async()=>new Response(new Uint8Array([7]),{status:206,headers:{'Content-Range':'bytes 9-10/100'}}));
    await assert.rejects(source.read('weights.bin',9,2),/Truncated model range/);
});
