import {readFile,writeFile} from 'node:fs/promises';
import createModule from '../generated/strata.mjs';
import {WasmBackend} from '../src/backend.js';
import {conformance} from '../tests/conformance.js';
const abi=JSON.parse(await readFile(new URL('../generated/strata.abi.json',import.meta.url),'utf8'));
const backend=new WasmBackend(await createModule({wasmBinary:await readFile(new URL('../generated/strata.wasm',import.meta.url))}),abi,Number(process.env.STRATA_THREADS||4));
let code=0;
try {
    const report=await conformance(backend);await writeFile(new URL('../reports/wasm.json',import.meta.url),JSON.stringify(report,null,2));
    for(const check of report.checks)console.log(`PASS ${check.name}: max absolute error ${check.maxAbsoluteError}`);
    if(report.backend.workerGroups.filter(x=>x>0).length<2)throw Error('Threaded backend did not engage multiple workers');
    console.log(JSON.stringify({tokens:report.tokens,backend:report.backend,stats:report.stats},null,2));
}catch(error){console.error(error);code=1;}finally{await backend.dispose();}
process.exit(code);
