"""Verify every canonical dense value and every expert code/scale byte using bounded reads.

Requires the vendored upstream tools plus numpy and gguf in .local/pack-env.
Expert verification reconstructs the raw role layout independently of the writer.
The source GGUF files and pack files are never modified; only verification metadata
and the manifest's source hashes/name/PLE locator are written after all checks pass.
"""
from __future__ import annotations
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT/'vendor/strata/tools'))
import numpy as np
import strata_pack as pack
from gguf_reader import GGUFFile


def read_at(stream, offset, count):
    stream.seek(offset)
    data=stream.read(count)
    if len(data)!=count:
        raise ValueError(f'Short read at {offset}: expected {count}, got {len(data)}')
    return data


def decode_chunk(stream, entry, begin, count):
    entry=dict(entry, elements=count)
    body=bytearray()
    for name in ('values','codes','scales','offsets'):
        if name not in entry:
            continue
        if name=='values':
            stride=2 if entry.get('values_fp16') else 4
            skip, size=begin*stride, count*stride
        elif name=='codes':
            skip, size=begin*entry['code_bits']//8, count*entry['code_bits']//8
        else:
            stride=2 if entry.get(name+'_fp16') else 4
            skip, size=begin//entry['group_elems']*stride, count//entry['group_elems']*stride
        raw=read_at(stream, entry[name]['offset']+skip, size)
        entry[name]={'offset':len(body),'bytes':len(raw)}
        body.extend(raw)
    return pack.decode_entry(body,entry)


def self_test():
    # Multiple chunks at nonzero offsets, then a deliberately corrupted code plane.
    raw=b''.join(b'\x00\x3c'+bytes([(i*47)&255])*16 for i in range(4))
    meta=SimpleNamespace(shape=[64,4],offset=0)
    blob,entry=pack.tensor_entry('q',meta,'Q2_0',raw,64)
    stream=io.BytesIO(bytes(64)+blob)
    for begin in (0,64,128,192):
        expected=pack.reference_values('Q2_0',raw[begin//64*18:(begin+64)//64*18])
        assert np.array_equal(decode_chunk(stream,entry,begin,64).view(np.uint32),expected.view(np.uint32))
    broken=bytearray(stream.getvalue());broken[entry['codes']['offset']]^=1
    assert not np.array_equal(decode_chunk(io.BytesIO(broken),entry,0,64),pack.reference_values('Q2_0',raw[:18]))
    print('Verifier self-test: chunk offsets and corrupt-code negative control passed',flush=True)


def verify_dense(gguf, source, folder, manifest):
    by_name={t.name:t for t in gguf.tensors}
    tensors=elements=zero_sign_differences=0
    with ExitStack() as stack:
        files={name:stack.enter_context((folder/name).open('rb')) for name in ('dense.bin','embd.bin')}
        for name,entry in manifest['tensors'].items():
            t=by_name[name]
            block=entry['block_elems']; chunk=max(block,1048576//block*block)
            for begin in range(0,t.elements,chunk):
                count=min(chunk,t.elements-begin)
                raw=read_at(source,gguf.data_start+t.offset+begin//block*entry['block_bytes'],count//block*entry['block_bytes'])
                expected=pack.reference_values(t.type_name,raw).reshape(-1)
                actual=decode_chunk(files[entry['file']],entry,begin,count).reshape(-1)
                if not np.array_equal(actual,expected):
                    mismatch=np.flatnonzero(actual!=expected)
                    raise ValueError(f'{name}: value mismatch at element {begin+int(mismatch[0])}')
                bit_mismatch=actual.view(np.uint32)!=expected.view(np.uint32)
                if np.any(bit_mismatch & ((actual!=0)|(expected!=0))):
                    raise ValueError(f'{name}: nonzero bit-pattern mismatch')
                zero_sign_differences+=int(np.count_nonzero(bit_mismatch))
                elements+=count
            tensors+=1
            if tensors%100==0 or tensors==len(manifest['tensors']):
                print(f'Dense verification: {tensors}/{len(manifest["tensors"])} tensors, {elements:,} values',flush=True)
    return dict(tensors=tensors,elements=elements,nonzeroBitMismatches=0,zeroSignDifferences=zero_sign_differences)


def verify_experts(gguf, source, folder, manifest):
    ex=manifest['experts']; ne=manifest['n_experts_per_layer']; blob=ex['blob_bytes']; offsets=ex['offsets']
    by_name={t.name:t for t in gguf.tensors}; compared=experts=0
    with (folder/'experts.bin').open('rb') as arena:
        for layer in ex['layers']:
            for first in range(0,ne,16):
                batch=min(16,ne-first)
                raw=read_at(arena,layer['offset']+first*blob,batch*blob)
                for role in ('gate','up','down'):
                    t=by_name[f'blk.{layer["layer"]}.ffn_{role}_exps.weight']
                    cols,rows,n=t.shape
                    assert n==ne and cols%64==0 and t.type_name=='Q2_0'
                    blocks=cols//64; role_bytes=rows*blocks*18
                    expected=np.frombuffer(read_at(source,gguf.data_start+t.offset+first*role_bytes,batch*role_bytes),dtype=np.uint8).reshape(batch,rows,blocks,18)
                    down=role=='down'; stride=1 if down else 2; slot=1 if role=='up' else 0
                    code_offset=offsets['down_codes' if down else 'gate_up_codes']+slot*blocks*16
                    scale_offset=offsets['down_scales' if down else 'gate_up_scales']+slot*blocks*2
                    codes=np.ndarray((batch,rows,blocks,16),dtype=np.uint8,buffer=raw,offset=code_offset,strides=(blob,stride*blocks*16,16,1))
                    scales=np.ndarray((batch,rows,blocks,2),dtype=np.uint8,buffer=raw,offset=scale_offset,strides=(blob,stride*blocks*2,2,1))
                    if not np.array_equal(codes,expected[:,:,:,2:]) or not np.array_equal(scales,expected[:,:,:,:2]):
                        raise ValueError(f'Expert byte mismatch: layer {layer["layer"]}, experts {first}:{first+batch}, {role}')
                    compared+=batch*role_bytes
                experts+=batch
            print(f'Expert verification: layer {layer["layer"]+1}/{len(ex["layers"])}, {experts:,} experts, {compared:,} bytes exact',flush=True)
    return dict(experts=experts,roleChecks=experts*3,bytesCompared=compared,byteMismatches=0)


def digest(path):
    print(f'Hashing {path.name}: {path.stat().st_size:,} bytes',flush=True)
    h=hashlib.sha256()
    with path.open('rb') as f:
        while data:=f.read(16*1024*1024):h.update(data)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--gguf',type=Path);ap.add_argument('--pack',type=Path);ap.add_argument('--self-test',action='store_true');args=ap.parse_args()
    self_test()
    if args.self_test:return
    if not args.gguf or not args.pack:ap.error('--gguf and --pack are required')
    started=time.monotonic();folder=args.pack.resolve();path=args.gguf.resolve();manifest=json.loads((folder/'manifest.json').read_text())
    for name,size in manifest['files'].items():
        if (folder/name).stat().st_size!=size:raise ValueError('Pack file size mismatch: '+name)
    gguf=GGUFFile(path)
    with path.open('rb') as source:
        dense=verify_dense(gguf,source,folder,manifest)
        experts=verify_experts(gguf,source,folder,manifest)
    shard2=Path(manifest['source']['shard2'])
    if not shard2.is_absolute():shard2=(ROOT/shard2).resolve()
    source_hashes={path.name:digest(path),shard2.name:digest(shard2)}
    download=path.parent/'DOWNLOAD.json'
    if download.exists():
        for entry in json.loads(download.read_text())['files']:
            if source_hashes[entry['name']]!=entry['expected_sha256']:raise ValueError('Source SHA-256 mismatch: '+entry['name'])
    file_hashes={name:digest(folder/name) for name in manifest['files']}
    g2=GGUFFile(shard2);ple=next(t for t in g2.tensors if t.name=='per_layer_token_embd.weight')
    manifest['name']='Qwen3.8-Flash-Next GSQ-RCO Q2_0'
    manifest['source'].update(shard1=str(path),shard2=str(shard2),shard1_sha256=source_hashes[path.name],shard2_sha256=source_hashes[shard2.name])
    manifest['pleSource']={'file':shard2.name,'offset':g2.data_start+ple.offset,'type':ple.type_id,'shape':list(ple.shape),'bytes':ple.expected_bytes()}
    report=dict(verifiedUTC=datetime.now(timezone.utc).isoformat(),seconds=time.monotonic()-started,dense=dense,experts=experts,sourceSha256=source_hashes,packSha256=file_hashes,method='All dense values compared with upstream reference dequantization in chunks; all expert code and scale bytes compared with original GGUF roles. Signed-zero differences counted explicitly.')
    manifest['verification']={'file':'VERIFIED.json','verifiedUTC':report['verifiedUTC']}
    (folder/'VERIFIED.json').write_text(json.dumps(report,indent=2)+'\n')
    (folder/'manifest.json').write_text(json.dumps(manifest,indent=1)+'\n')
    (folder/'SHA256SUMS.txt').write_text(''.join(f'{sha}  {name}\n' for name,sha in file_hashes.items()))
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
