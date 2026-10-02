import createModule from '../generated/strata.mjs';
import {WasmBackend} from './backend.js';
let backend;
const buffers=new Map(); let next=0;
const resolve=b=>{const actual=buffers.get(b.id); if(!actual) throw Error('Unknown WASM buffer'); return actual;};
async function execute(method,args){
        let value;
        if (method==='init') {
            if (!crossOriginIsolated) throw Error('WASM threads need COOP/COEP headers; use npm start');
            const abi=await (await fetch(new URL('../generated/strata.abi.json',import.meta.url))).json();
            backend=new WasmBackend(await createModule(),abi,args.threads); value=backend.info();
        } else if (method==='alloc') {
            const b=backend.alloc(args.dataOrLength,args.type);const handle=args.handle||{id:++next,type:b.type,length:b.length};if(buffers.has(handle.id))throw Error('Duplicate WASM handle');buffers.set(handle.id,b);value=handle;
        } else if (method==='write') backend.write(resolve(args.buffer),args.data,args.offset);
        else if (method==='read') value=backend.read(resolve(args.buffer),args.type);
        else if (method==='copy') backend.copy(resolve(args.source),resolve(args.destination),args.count,args.sourceOffset,args.destinationOffset);
        else if (method==='run') {
            const values=Object.fromEntries(Object.entries(args.args).map(([k,v])=>[k,typeof v==='number'?v:resolve(v)]));
            backend.run(args.name,values,args.groups);
        } else if (method==='free') { backend.free(resolve(args.buffer)); buffers.delete(args.buffer.id); }
        else if (method==='idle') await backend.idle();
        else if (method==='info') value=backend.info();
        else if (method==='dispose') { await backend.dispose(); buffers.clear(); }
        else if(method==='batch'){for(const command of args.commands){if(!['alloc','write','copy','run','free'].includes(command.method))throw Error('Invalid batched WASM operation');await execute(command.method,command.args);}}
        else throw Error('Unknown worker command');
        return value;
}
let queue=Promise.resolve();
self.onmessage=({data})=>{queue=queue.then(async()=>{
    const {id,method,args}=data;
    try{
        const value=await execute(method,args);
        self.postMessage({id,value});
    } catch(error) { self.postMessage({id,error:error.stack||String(error)}); }
});};
