// An explicit portable interchange pack, also used to exercise real file loading
// with the deterministic fixture. Binary payload is not embedded in the manifest.
export function encodeFixturePack(store) {
    const tensors=Object.create(null),chunks=[];let offset=0;
    for(const [name,t] of store.tensors) {
        const bytes=new Uint8Array(t.data.buffer,t.data.byteOffset,t.data.byteLength);
        tensors[name]={name,shape:t.shape,form:'P32',source_type:'F32',file:'weights.bin',values:{offset,bytes:bytes.length},values_fp16:false};
        chunks.push(bytes);offset+=bytes.length;
    }
    const binary=new Uint8Array(offset);let at=0;for(const bytes of chunks){binary.set(bytes,at);at+=bytes.length;}
    return {manifest:{format:'strata-webcuda-v1',name:store.label,config:store.config,pleConstants:store.constants,tensors,files:{'weights.bin':binary.length}},binary};
}
