import {readFile,writeFile} from 'node:fs/promises';
import {execFile} from 'node:child_process';
import {resolve} from 'node:path';
import assert from 'node:assert/strict';
import {Tokenizer,formatChat} from '../src/tokenizer.js';
const pack=resolve('models/Qwen3.8-Flash-Next-WebCuda-Q2_0'),read=async name=>readFile(resolve(pack,'tokenizer',name),'utf8');
const tokenizer=new Tokenizer({vocab:JSON.parse(await read('vocab.json')),merges:await read('merges.txt'),types:JSON.parse(await read('token_type.json')),config:JSON.parse(await read('tokenizer.json'))});
const corpus=['','The capital of France is','Hello, world!','  leading and trailing  ','a\n\n\nb','def f(x):\n\treturn x # comment\n','你好，世界','العربية','😀🚀🇺🇸','é́ combining','\x00\x01\x7f control','\u00a0non-breaking\u00a0space','1234567890','MixedCASE_and-dashes','\r\n\r\n','\u2028\u2029','\u0085next line\uFEFFBOM','\uFEFF','\u001c\u001d\u001e\u001f','<|im_start|>user\nHi<|im_end|>','<think>hi</think><tool_response>x</tool_response>','<|im_star','x'.repeat(5000),'测试'.repeat(4000)];
let seed=417;const atoms=['a',' ','\n','é','́','🦀','中','7','<','_','\u0085','\uFEFF'];
for(let i=0;i<80;i++){let s='';for(let j=0;j<37;j++){seed=(Math.imul(seed,1664525)+1013904223)>>>0;s+=atoms[seed%atoms.length];}corpus.push(s);}
const checks=corpus.flatMap(text=>[false,true].map(parse_special=>({text,parse_special})));
const chats=[{messages:[{role:'user',content:'Hi'}]},{system:' Be concise. ',messages:[{role:'user',content:'  hello\n'}]},{messages:[{role:'user',content:'Hi'},{role:'assistant',content:' Hello! '},{role:'user',content:'Again?'}]},{system:'\u001c\u0085test\uFEFF',messages:[{role:'user',content:'\u001eafé́ 中文 \n'},{role:'assistant',content:'Yes.'},{role:'user',content:'Why?'}]}];
const expected=await new Promise((fulfill,reject)=>{const child=execFile(resolve('.local/pack-env',process.platform==='win32'?'Scripts/python.exe':'bin/python'),['scripts/model-tokenizer.py','--pack',pack],{encoding:'utf8',maxBuffer:8*1024*1024},(e,out,err)=>e?reject(Error(err)):fulfill(JSON.parse(out)));child.stdin.end(JSON.stringify({checks,chats}));});
for(let i=0;i<checks.length;i++){const actual=tokenizer.encode(checks[i].text,{parseSpecial:checks[i].parse_special});assert.deepEqual(actual,expected.checks[i].tokens,'Token IDs '+i);assert.equal(tokenizer.decode(actual),checks[i].text,'Round trip '+i);}
for(let i=0;i<chats.length;i++)assert.equal(formatChat(chats[i].messages,chats[i].system),expected.chats[i],'Exact Jinja chat template '+i);
const emoji=tokenizer.encode('🦀');for(let i=1;i<emoji.length;i++)assert.ok(!tokenizer.decode(emoji.slice(0,i),{stream:true}).includes('\uFFFD'),'Streaming UTF-8 must not flash replacement characters');
const report={checks:checks.length,chatTemplates:chats.length,exact:true,chatHiTokens:tokenizer.encode(formatChat(chats[0].messages),{parseSpecial:true})};await writeFile('reports/tokenizer.json',JSON.stringify(report,null,2)+'\n');console.log('PASS '+JSON.stringify(report));
