import {chromium} from 'playwright';
import {readFile,writeFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {startServer} from './serve.mjs';
import {createFixture} from '../src/fixture.js';
import {ReferenceEngine} from '../tests/reference.mjs';
import {fileURLToPath} from 'node:url';
const server=await startServer(0),url='http://127.0.0.1:'+server.address().port;
const browser=await chromium.launch({channel:process.env.STRATA_BROWSER||'msedge',headless:true,args:['--enable-unsafe-webgpu']});
const page=await browser.newPage({viewport:{width:1440,height:1240}}),errors=[];
page.on('pageerror',e=>errors.push(e.message));page.on('console',m=>{if(m.type()==='error')errors.push(m.text());});page.on('response',r=>{if(r.status()>=400)errors.push(r.status()+' '+r.url());});
try {
    await page.goto(url+'/test.html');await page.waitForFunction(()=>window.strataDiagnostics?.ready);
    if(!await page.evaluate(()=>crossOriginIsolated))throw Error('Server did not isolate page for WASM');
    const manifest=JSON.parse(await readFile(new URL('../generated/manifest.json',import.meta.url),'utf8'));
    for(const {entry,wgslSha256} of manifest.kernels){const disk=await readFile(new URL('../generated/'+entry+'.wgsl',import.meta.url),'utf8'),served=await(await fetch(url+'/generated/'+entry+'.wgsl')).text();if(createHash('sha256').update(disk).digest('hex')!==wgslSha256||createHash('sha256').update(served).digest('hex')!==wgslSha256)throw Error('Stale served WGSL '+entry);}
    const reports=[];
    for(const mode of ['webgpu','wasm','hybrid']) {
        console.log('Testing '+mode);const report=await page.evaluate(mode=>window.strataTests.run(mode),mode);reports.push({mode,...report});console.log('PASS '+mode+': '+report.checks.length+' checks; tokens '+report.tokens.join(','));
        if(mode==='hybrid'&&(!report.stats.gpuExperts||!report.stats.cpuExperts||!report.stats.promotions))throw Error('Hybrid path did not exercise both processors and promotion');
    }
    await page.locator('#files').setInputFiles(['manifest.json','weights.bin'].map(n=>fileURLToPath(new URL('../generated/fixture/'+n,import.meta.url))));
    await page.waitForFunction(()=>window.strataDiagnostics.model==='Untrained deterministic fixture'&&!document.getElementById('run').disabled);
    await page.locator('#run').click();await page.waitForFunction(()=>window.strataDiagnostics.last&&!document.getElementById('run').disabled);
    const reference=new ReferenceEngine(createFixture());let value;for(const t of [2,7,4])value=await reference.step(t);const expected=[value.token];for(let i=1;i<4;i++){value=await reference.step(value.token);expected.push(value.token);}
    const fileDecode=await page.evaluate(()=>window.strataDiagnostics.last);if(String(fileDecode.generated)!==String(expected))throw Error('File pack UI decode differs from reference');
    console.log('PASS local file picker, file-backed autoregressive decode and UI controls');
    await page.screenshot({path:fileURLToPath(new URL('../reports/browser.png',import.meta.url)),fullPage:true});
    await page.setViewportSize({width:390,height:844});await page.screenshot({path:fileURLToPath(new URL('../reports/mobile.png',import.meta.url)),fullPage:true});
    if(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth))throw Error('Mobile page overflows viewport');
    // Production routing has ten selected experts rather than the fixture's two.
    // Exercise that layout without requiring a 40 GB model for the UI suite.
    await page.evaluate(()=>{
        for(const row of document.querySelectorAll('#routing .route')){
            const chip=row.querySelector('.expert'),type=row.querySelector('.type');
            while(row.querySelectorAll('.expert').length<10)row.insertBefore(chip.cloneNode(true),type);
        }
    });
    if(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth))throw Error('Production top-10 routing overflows the mobile viewport');
    console.log('PASS production top-10 routing layout at mobile width');
    await page.evaluate(()=>window.strataTests.release());
    const runtimeErrors=await page.evaluate(()=>window.strataDiagnostics.errors);if(errors.length||runtimeErrors.length)throw Error(JSON.stringify([...errors,...runtimeErrors]));
    await writeFile(new URL('../reports/browser.json',import.meta.url),JSON.stringify({sourceSha256:manifest.sha256,errors,reports,fileDecode},null,2));
}finally{await browser.close();await new Promise(r=>server.close(r));}
