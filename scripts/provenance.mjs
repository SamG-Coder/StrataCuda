import {readFile,writeFile,readdir} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {fileURLToPath} from 'node:url';
import path from 'node:path';
const root=fileURLToPath(new URL('../',import.meta.url)),sha=bytes=>createHash('sha256').update(bytes).digest('hex');
async function walk(dir){const out=[];for(const d of await readdir(path.join(root,dir),{withFileTypes:true})){const name=dir+'/'+d.name;out.push(...(d.isDirectory()?await walk(name):[name]));}return out;}
const sources={};for(const file of await walk('vendor'))sources[file]=sha(await readFile(path.join(root,file)));
const provenance={
    strata:{repository:'https://github.com/Niko1221/Strata',commit:'1678de333d0e0711bc414ad992b640e1a37dd814',license:'MIT'},
    webcuda:{repository:'https://github.com/SamG-Coder/cuda-webshader',commit:'ef46ff1bf02a306bad94ddc18286d25d3d902c14',license:'MIT',localSource:'D:/cuda-webshader',modifications:[]},
    port:{source:'kernels/strata.cu',sha256:sha(await readFile(path.join(root,'kernels/strata.cu'))),numericContract:'F32 activations/reductions/KV; portable baseline, not native bit parity',upstreamMapping:{
        projections:['src/kernels/cuda/bf16_gemv.cu','src/kernels/cuda/s_gemv.cu'],router:['src/kernels/cuda/router_top10.cu'],
        gatedResidual:['src/kernels/cuda/gr.cu'],deltaNet:['src/kernels/cuda/gdn.cu','src/kernels/cuda/elementwise.cu'],
        attention:['src/kernels/cuda/qsa.cu','src/kernels/cuda/rope.cu'],ple:['src/kernels/cuda/ple.cu','src/kernels/ngram.cpp'],
        orchestration:['src/core/layer.cpp','src/core/layout.cpp'],fileLayouts:['tools/strata_pack.py','tools/pack_layer.py','include/strata/artifact/dequant.hpp']
    }},vendorSha256:sources
};
await writeFile(path.join(root,'PROVENANCE.json'),JSON.stringify(provenance,null,2)+'\n');console.log('Recorded '+Object.keys(sources).length+' vendored source/license hashes');
