import test from 'node:test';
import assert from 'node:assert/strict';
import {startServer} from '../scripts/serve.mjs';
test('local server supplies WASM isolation, exact ranges and rejects hidden paths',async()=>{
    const server=await startServer(0),base='http://127.0.0.1:'+server.address().port;
    try{
        const page=await fetch(base);assert.equal(page.status,200);assert.equal(page.headers.get('cross-origin-opener-policy'),'same-origin');assert.equal(page.headers.get('cross-origin-embedder-policy'),'require-corp');
        const full=await(await fetch(base+'/kernels/strata.cu')).text(),part=await fetch(base+'/kernels/strata.cu',{headers:{Range:'bytes=5-12'}});assert.equal(part.status,206);assert.equal(await part.text(),full.slice(5,13));
        assert.equal((await fetch(base+'/kernels/strata.cu',{headers:{Range:'bytes=999999999-'}})).status,416);
        assert.equal((await fetch(base+'/.git/config')).status,403);assert.equal((await fetch(base+'/%2e%2e%5cpackage.json')).status,403);
        assert.equal((await fetch(base,{method:'POST'})).status,405);
    }finally{await new Promise(resolve=>server.close(resolve));}
});
