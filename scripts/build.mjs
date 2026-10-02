import {readFile, writeFile, mkdir, access} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import path from 'node:path';
import {createHash} from 'node:crypto';
import {compile, serializableArtifact} from '../vendor/webcuda/src/compiler/compiler.js';
import {compileThreaded} from '../vendor/webcuda/src/wasm/compile.mjs';

const root = fileURLToPath(new URL('../', import.meta.url));
const source = await readFile(path.join(root, 'kernels/strata.cu'), 'utf8');
const specs = [...source.matchAll(/__global__\s+void\s+(\w+)\s*\(/g)].map(m => ({entry: m[1], workgroupSize: [64, 1, 1]}));
const outDir = path.join(root, 'generated');
const hash = data => createHash('sha256').update(data).digest('hex');
const records=[];
await mkdir(outDir, {recursive:true});
for (const spec of specs) {
    const artifact = compile(source, spec);
    await writeFile(path.join(outDir, spec.entry + '.json'), JSON.stringify(serializableArtifact(artifact)));
    await writeFile(path.join(outDir, spec.entry + '.wgsl'), artifact.wgsl);
    records.push({...spec,wgslSha256:hash(artifact.wgsl)});
    console.log('WebGPU ' + spec.entry);
}
const manifest = {source: 'kernels/strata.cu', sha256:hash(source), kernels:records};
await writeFile(path.join(outDir, 'manifest.json'), JSON.stringify(manifest, null, 2));
if (!process.argv.includes('--gpu-only')) {
    let emcc = process.env.EMXX;
    if (!emcc && process.platform === 'win32') {
        const local = 'D:/WaterCuda/.native/emsdk/upstream/emscripten/em++.py';
        try { await access(local); emcc = local.replace(/\.py$/, '.bat'); } catch {}
    }
    await compileThreaded(source, specs, {outDir, name:'strata', ...(emcc ? {emcc} : {})});
    console.log(`WebAssembly ${specs.length} kernels (shared memory, SIMD, 1–8 threads)`);
}
console.log('Source SHA-256 ' + manifest.sha256);
await import('./fixture-pack.mjs');
