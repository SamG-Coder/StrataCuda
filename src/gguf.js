// Bounded, random-access GGUF v3 header reader. Tensor bodies remain in the File.
const formats=new Map([[0,[1,4,'F32']],[1,[1,2,'F16']],[2,[32,18,'Q4_0']],[3,[32,20,'Q4_1']],
    [6,[32,22,'Q5_0']],[7,[32,24,'Q5_1']],[8,[32,34,'Q8_0']],[9,[32,36,'Q8_1']],
    [10,[256,84,'Q2_K']],[11,[256,110,'Q3_K']],[12,[256,144,'Q4_K']],[13,[256,176,'Q5_K']],[14,[256,210,'Q6_K']],
    [15,[256,292,'Q8_K']],[16,[256,66,'IQ2_XXS']],[17,[256,74,'IQ2_XS']],[18,[256,98,'IQ3_XXS']],
    [19,[256,50,'IQ1_S']],[20,[32,18,'IQ4_NL']],[21,[256,110,'IQ3_S']],[22,[256,82,'IQ2_S']],
    [23,[256,136,'IQ4_XS']],[24,[1,1,'I8']],[25,[1,2,'I16']],[26,[1,4,'I32']],[27,[1,8,'I64']],[28,[1,8,'F64']],[29,[256,56,'IQ1_M']],[30,[1,2,'BF16']],[42,[64,18,'Q2_0']]]);
class Reader {
    constructor(source,name){this.source=source;this.name=name;this.size=source.size(name);this.pos=0;this.begin=0;this.window=new Uint8Array();}
    async take(n) {
        if(!Number.isSafeInteger(n)||n<0||this.pos+n>this.size)throw Error('Truncated GGUF at '+this.pos);
        if(this.pos<this.begin||this.pos+n>this.begin+this.window.length){this.begin=this.pos;this.window=await this.source.read(this.name,this.pos,Math.min(this.size-this.pos,Math.max(65536,n)));}
        const out=this.window.subarray(this.pos-this.begin,this.pos-this.begin+n);this.pos+=n;return out;
    }
    async number(type) {
        const shape={0:[1,'getUint8'],1:[1,'getInt8'],2:[2,'getUint16'],3:[2,'getInt16'],4:[4,'getUint32'],5:[4,'getInt32'],6:[4,'getFloat32'],7:[1,'getUint8'],10:[8,'getBigUint64'],11:[8,'getBigInt64'],12:[8,'getFloat64']}[type];
        if(!shape)throw Error('Unknown GGUF metadata type '+type);const bytes=await this.take(shape[0]),v=new DataView(bytes.buffer,bytes.byteOffset,bytes.byteLength)[shape[1]](0,true);
        if(type===7){if(v>1)throw Error('Invalid GGUF bool');return !!v;}
        return typeof v==='bigint'?(v<=BigInt(Number.MAX_SAFE_INTEGER)&&v>=BigInt(Number.MIN_SAFE_INTEGER)?Number(v):v.toString()):v;
    }
    async length(max,label) {const n=await this.number(10);if(!Number.isSafeInteger(n)||n<0||n>max)throw Error('GGUF '+label+' exceeds supported header limits');return n;}
    async string() {const n=await this.length(16*1024*1024,'string');return new TextDecoder('utf-8',{fatal:true}).decode(await this.take(n));}
    async value(type,depth=0) {
        if(type===8)return this.string();
        if(type!==9)return this.number(type);
        if(depth>0)throw Error('Nested GGUF arrays are not supported');
        const subtype=await this.number(4),count=await this.length(2_000_000,'array'),a=[];
        for(let i=0;i<count;i++)a.push(await this.value(subtype,depth+1));return a;
    }
}
export async function readGGUF(source,name) {
    const r=new Reader(source,name);if(await r.number(4)!==0x46554747)throw Error(name+': invalid GGUF magic');
    const version=await r.number(4);if(version!==3)throw Error('Only GGUF v3 headers are supported');
    const count=await r.length(100000,'tensor count'),metadataCount=await r.length(100000,'metadata count'),metadata=Object.create(null),tensors=[];
    for(let i=0;i<metadataCount;i++){const key=await r.string();if(Object.hasOwn(metadata,key))throw Error('Duplicate GGUF metadata '+key);metadata[key]=await r.value(await r.number(4));}
    const names=new Set();
    for(let i=0;i<count;i++) {
        const name=await r.string();if(names.has(name))throw Error('Duplicate GGUF tensor '+name);names.add(name);
        const dims=await r.number(4);if(dims<1||dims>4)throw Error('Invalid GGUF tensor rank');
        const shape=[];for(let d=0;d<dims;d++){const n=await r.length(Number.MAX_SAFE_INTEGER,'dimension');if(!n)throw Error('Empty GGUF tensor');shape.push(n);}
        const type=await r.number(4),offset=await r.length(Number.MAX_SAFE_INTEGER,'offset'),format=formats.get(type),elements=shape.reduce((a,b)=>a*b,1);
        if(!Number.isSafeInteger(elements)||!format||elements%format[0])throw Error('Unsupported or invalid GGUF tensor '+name+' type '+type);
        tensors.push({name,shape,type,typeName:format[2],offset,bytes:elements/format[0]*format[1]});
    }
    const alignment=metadata['general.alignment']??32;if(!Number.isInteger(alignment)||alignment<1||alignment>65536||(alignment&(alignment-1)))throw Error('Invalid GGUF alignment');
    const dataOffset=Math.ceil(r.pos/alignment)*alignment;
    for(const t of tensors){if(t.offset%alignment||dataOffset+t.offset+t.bytes>r.size)throw Error('GGUF tensor outside file: '+t.name);t.offset+=dataOffset;}
    const intervals=[...tensors].sort((a,b)=>a.offset-b.offset);for(let i=1;i<intervals.length;i++)if(intervals[i].offset<intervals[i-1].offset+intervals[i-1].bytes)throw Error('Overlapping GGUF tensors');
    return {name,version,dataOffset,metadata,tensors,size:r.size};
}
