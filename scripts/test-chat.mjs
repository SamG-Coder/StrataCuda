import {chromium} from 'playwright';
import {resolve} from 'node:path';
import {writeFile} from 'node:fs/promises';
import assert from 'node:assert/strict';
import {startServer} from './serve.mjs';
const server=await startServer(0),browser=await chromium.launch({channel:process.env.STRATA_BROWSER||'msedge',headless:true,args:['--enable-unsafe-webgpu']});
const page=await browser.newPage({viewport:{width:1440,height:1024}}),errors=[];page.setDefaultTimeout(30000);
page.on('pageerror',e=>errors.push(e.message));page.on('console',m=>{if(m.type()==='error')errors.push(m.text());});page.on('response',r=>{if(r.status()>=400)errors.push(r.status()+' '+r.url());});
let ticker;
try{
    await page.goto('http://127.0.0.1:'+server.address().port+'/');await page.waitForFunction(()=>window.strataChat?.ready);
    assert.ok(await page.locator('#send').isDisabled());assert.equal(await page.locator('a[href="./test.html"]').count(),1);
    await page.screenshot({path:'reports/chat-desktop.png',fullPage:true,animations:'disabled'});
    await page.setViewportSize({width:390,height:844});await page.screenshot({path:'reports/chat-mobile.png',fullPage:true,animations:'disabled'});
    assert.ok(!await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth));
    await page.locator('#toggle-settings').click();assert.equal(await page.locator('#toggle-settings').getAttribute('aria-expanded'),'true');
    await page.screenshot({path:'reports/chat-mobile-settings.png',fullPage:true,animations:'disabled'});
    assert.deepEqual(errors,[]);console.log('PASS desktop/mobile text interface and Test lab link');
    if(!process.argv.includes('--ui-only')){
    await page.setViewportSize({width:1440,height:1024});
    await page.locator('#load-local').click();await page.waitForFunction(()=>window.strataChat.model&&!window.strataChat.busy||window.strataChat.errors.length);
    assert.deepEqual(await page.evaluate(()=>window.strataChat.errors),[]);assert.ok(await page.evaluate(()=>window.strataChat.chatSupported));
    await page.locator('#context').selectOption('128');await page.locator('#reply-limit').fill('128');await page.locator('#prompt').fill('Hi');await page.locator('#send').click();
    assert.match(await page.locator('#error').textContent(),/needs 141 tokens/);assert.equal(await page.locator('#messages .message').count(),0);
    await page.locator('#reply-limit').fill('2');await page.locator('#format').selectOption('completion');await page.locator('#send').click();
    await page.waitForFunction(()=>window.strataChat.progress?.layer>=1);const stoppedAt=Date.now();await page.locator('#stop').click();
    await page.waitForFunction(()=>window.strataChat.last?.stop==='stopped'&&!window.strataChat.busy);const stopSeconds=(Date.now()-stoppedAt)/1000;
    console.log('PASS HTTP model loading, context rejection and real mid-layer cancellation ('+stopSeconds.toFixed(2)+' s)');
    await page.locator('#new-chat').click();await page.locator('#model-folder').setInputFiles(resolve('models/Qwen3.8-Flash-Next-WebCuda-Q2_0'));
    await page.waitForFunction(()=>window.strataChat.model&&!window.strataChat.busy);assert.deepEqual(await page.evaluate(()=>window.strataChat.errors),[]);
    await page.locator('#format').selectOption('chat');await page.locator('#context').selectOption('512');await page.locator('#prompt').fill('Hi');
    console.log('Starting real chat: Hi, 512 context, 2 reply tokens');
    let last='';ticker=setInterval(async()=>{try{const p=await page.evaluate(()=>window.strataChat.progress);if(!p)return;const key=p.phase+':'+p.done;if(key!==last){last=key;console.log('CHAT '+JSON.stringify(p));}}catch{}},1000);
    await page.locator('#prompt').press('Enter');await page.waitForFunction(()=>window.strataChat.last&&!window.strataChat.busy||window.strataChat.errors.length,{},{timeout:30*60*1000});
    clearInterval(ticker);ticker=null;const result=await page.evaluate(()=>window.strataChat);assert.deepEqual(result.errors,[]);assert.deepEqual(errors,[]);assert.equal(result.context,512);assert.equal(result.last.promptTokens,13);assert.ok(result.last.generated.length>0);assert.equal(await page.locator('.assistant .message-body').last().textContent(),result.last.text);
    await page.screenshot({path:'reports/chat-model.png',fullPage:true,animations:'disabled'});
    await writeFile('reports/chat.json',JSON.stringify({timestamp:new Date().toISOString(),model:result.model,context:result.context,backend:result.backend,stopSeconds,result:result.last,errors},null,2)+'\n');
    console.log('PASS real text chat '+JSON.stringify(result.last));
    await page.locator('#new-chat').click();await page.waitForFunction(()=>!window.strataChat.busy);assert.equal(await page.locator('#messages .message').count(),0);
    }
}finally{clearInterval(ticker);await browser.close();await new Promise(r=>server.close(r));}
