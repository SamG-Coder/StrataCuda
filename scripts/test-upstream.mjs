import {readFile,writeFile,mkdir} from 'node:fs/promises';
import {spawnSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';
import path from 'node:path';
import assert from 'node:assert/strict';
import {ngramRows,halfToFloat,IQ4NL} from '../src/model.js';
const root=fileURLToPath(new URL('../',import.meta.url)),out=path.join(root,'.local');await mkdir(out,{recursive:true});
// Extract upstream function bodies unchanged. The native oracle never compiles
// kernels/strata.cu or a JavaScript reimplementation of Strata's bit layouts.
const source=await readFile(path.join(root,'vendor/strata/src/kernels/ngram.cpp'),'utf8');
const functions=['PleConsts ple_artifact_consts()','uint64_t ngram_mixed(','void ngram_rows('].map(anchor=>{
    const begin=source.indexOf(anchor);if(begin<0)throw Error('Upstream oracle anchor missing');const end=source.indexOf('\n}',begin);return source.slice(begin,end+2);
}).join('\n');
const cpp=`#include "strata/artifact/dequant.hpp"
#include "strata/kernels/ngram.hpp"
#include <fstream>
namespace strata::kernels { ${functions} }
int main(int argc,char**argv){
 if(argc!=2)return 2;std::ofstream out(argv[1],std::ios::binary);
 for(unsigned h=0;h<65536;h++){float f=strata::fp16_to_fp32(h);out.write((char*)&f,4);}
 for(int b=0;b<256;b++) {unsigned char block[18];unsigned h=((b%16)+8)*1024+(b*7)%1024+(b%2?32768:0);block[0]=h&255;block[1]=h>>8;for(int i=0;i<16;i++)block[2+i]=(i*19+b*73)&255;float values[64];strata::dequantize_q2_0(block,values);out.write((char*)values,sizeof(values));}
 for(int b=0;b<256;b++) {unsigned char block[18];unsigned h=((b%16)+8)*1024+(b*7)%1024+(b%2?32768:0);block[0]=h&255;block[1]=h>>8;for(int i=0;i<16;i++)block[2+i]=(i*19+b*73)&255;float values[32];strata::dequantize_iq4_nl(block,values);out.write((char*)values,sizeof(values));}
 auto c=strata::kernels::ple_artifact_consts();
 int32_t tokens[]={7,7,7,7,2147483000};int32_t prev[]={2,3,3,2,0,0,99,248044,2147481000,2147482000};uint32_t rows[80];strata::kernels::ngram_rows(tokens,prev,5,c,rows);out.write((char*)rows,sizeof(rows));return out?0:3;
}`;
const cppPath=path.join(out,'upstream-oracle.cpp'),exe=path.join(out,process.platform==='win32'?'upstream-oracle.exe':'upstream-oracle'),binary=path.join(out,'upstream-oracle.bin');await writeFile(cppPath,cpp);
let run=spawnSync(process.env.CXX||'clang++',['-O2','-std=c++20','-I',path.join(root,'vendor/strata/include'),cppPath,'-o',exe],{encoding:'utf8'});if(run.status!==0)throw Error(run.error?.message||run.stderr);
run=spawnSync(exe,[binary],{encoding:'utf8'});if(run.status!==0)throw Error(run.error?.message||run.stderr);
const bytes=await readFile(binary),view=new DataView(bytes.buffer,bytes.byteOffset,bytes.byteLength);let offset=0;
for(let h=0;h<65536;h++){const a=halfToFloat(h),b=view.getFloat32(offset,true);assert.ok(Object.is(a,b)||Number.isNaN(a)&&Number.isNaN(b),'half '+h);offset+=4;}
for(const size of [64,32])for(let b=0;b<256;b++) {
    const h=((b%16)+8)*1024+(b*7)%1024+(b%2?32768:0),scale=halfToFloat(h);
    for(let i=0;i<size;i++){const raw=((size===64?Math.floor(i/4):i%16)*19+b*73)&255,code=size===64?(raw>>>((i%4)*2))&3:(raw>>>(i<16?0:4))&15,value=scale*(size===64?code-1:IQ4NL[code]);assert.equal(Math.fround(value),view.getFloat32(offset,true));offset+=4;}
}
for(const [token,prev] of [[7,[2,3]],[7,[3,2]],[7,[0,0]],[7,[99,248044]],[2147483000,[2147481000,2147482000]]])for(const row of ngramRows(token,prev)){assert.equal(row,view.getUint32(offset,true));offset+=4;}
assert.equal(offset,bytes.length);
const report={upstream:'1678de333d0e0711bc414ad992b640e1a37dd814',halfValues:65536,q2Values:16384,iq4Values:8192,ngramIndices:80,exact:true};await writeFile(path.join(root,'reports/upstream.json'),JSON.stringify(report,null,2));console.log('PASS: native upstream oracle '+JSON.stringify(report));
