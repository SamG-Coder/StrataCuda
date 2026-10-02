import http from 'node:http';
import {createReadStream} from 'node:fs';
import {stat} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import path from 'node:path';
export const root=fileURLToPath(new URL('../',import.meta.url));
const mime={'.html':'text/html; charset=utf-8','.js':'text/javascript; charset=utf-8','.mjs':'text/javascript; charset=utf-8','.css':'text/css','.json':'application/json','.wasm':'application/wasm','.svg':'image/svg+xml','.cu':'text/plain; charset=utf-8','.md':'text/plain; charset=utf-8','.png':'image/png'};
export function startServer(port=Number(process.env.PORT||8094)) {
    const server=http.createServer(async(req,res)=>{
        res.setHeader('Cross-Origin-Opener-Policy','same-origin');res.setHeader('Cross-Origin-Embedder-Policy','require-corp');res.setHeader('Cross-Origin-Resource-Policy','same-origin');res.setHeader('Cache-Control','no-store');res.setHeader('X-Content-Type-Options','nosniff');
        try {
            if(!['GET','HEAD'].includes(req.method)){res.writeHead(405);res.end();return;}
            const pathname=decodeURIComponent(new URL(req.url,'http://127.0.0.1').pathname),relative=pathname==='/'?'index.html':pathname.slice(1);
            if(relative.split(/[\\/]/).some(p=>p.startsWith('.')||p==='node_modules'||p==='upstream')){res.writeHead(403);res.end('Forbidden');return;}
            const file=path.resolve(root,relative);if(!file.startsWith(root)||file===root){res.writeHead(403);res.end('Forbidden');return;}
            const s=await stat(file);if(!s.isFile())throw Error('Not a file');
            res.setHeader('Content-Type',mime[path.extname(file)]||'application/octet-stream');res.setHeader('Accept-Ranges','bytes');
            let start=0,end=s.size-1,status=200;
            if(req.headers.range){const m=/^bytes=(\d+)-(\d*)$/.exec(req.headers.range);if(!m||Number(m[1])>=s.size||(m[2]&&Number(m[2])<Number(m[1]))){res.writeHead(416,{'Content-Range':`bytes */${s.size}`});res.end();return;}start=Number(m[1]);end=m[2]?Math.min(Number(m[2]),end):end;status=206;res.setHeader('Content-Range',`bytes ${start}-${end}/${s.size}`);}
            res.setHeader('Content-Length',Math.max(0,end-start+1));res.writeHead(status);if(req.method==='HEAD'||!s.size){res.end();return;}createReadStream(file,{start,end,highWaterMark:1024*1024}).on('error',()=>res.destroy()).pipe(res);
        } catch {if(!res.headersSent)res.writeHead(404);res.end('Not found');}
    });
    return new Promise((resolve,reject)=>{server.once('error',reject);server.listen(port,'127.0.0.1',()=>resolve(server));});
}
if(process.argv[1]&&path.resolve(process.argv[1])===fileURLToPath(import.meta.url)){const server=await startServer();console.log('Strata WebCuda: http://127.0.0.1:'+server.address().port);}
