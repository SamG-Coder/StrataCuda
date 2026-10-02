"""Run the pinned upstream executable and report numerical differences honestly.

Install upstream v0.1.34 separately; this script never downloads or executes an
unselected binary. All output stays in .local/native-upstream and reports/.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--exe', type=Path, default=ROOT/'.local/native-upstream/bin/strata.exe')
parser.add_argument('--pack', type=Path, default=ROOT/'models/Qwen3.8-Flash-Next-WebCuda-Q2_0')
parser.add_argument('--portable', type=Path, default=ROOT/'reports/benchmark-release.json')
parser.add_argument('--label', default='release', help='Portable raw-logit filename label')
parser.add_argument('--existing', action='store_true', help='Compare saved native runs instead of running them')
args = parser.parse_args()
report = json.loads(args.portable.read_text())
manifest = json.loads((args.pack/'manifest.json').read_text())
output = ROOT/'.local/native-upstream'
output.mkdir(parents=True, exist_ok=True)
if not args.exe.is_file():
    parser.error('Select the upstream v0.1.34 executable with --exe; see reports/execution-port.md')
if not args.existing:
    subprocess.run([sys.executable, '-B', str(ROOT/'vendor/strata/tools/pack_index.py'), '--pack', str(args.pack)], check=True)

def digest(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

portable_path = ROOT/f'.local/benchmarks/{args.label}-logits.f32'
if digest(portable_path) != report['logits']['sha256']:
    raise ValueError('Portable logits do not match their report')
x = np.fromfile(portable_path, dtype='<f4').astype(np.float64)
portable_routes = {}
for step in report['steps']:
    for route in step['routes']:
        portable_routes[(route.get('position', step['position']), route['layer'])] = route['ids']

results = []
for mode in ('canonical', 'native'):
    logits_path, routing_path = output/f'{mode}-logits.f32', output/f'{mode}-routing.bin'
    command = [str(args.exe), '--pack', str(args.pack), '--tokens', ','.join(map(str, report['inputs'])),
               '--max-new', '1', '--max-context', '256', '--mmap-experts', '--pool-workers', '4',
               '--ple-gguf', manifest['source']['shard2'], '--dump-logits', str(logits_path),
               '--dump-routing', str(routing_path)]
    if mode == 'native':
        command += ['--native', manifest['source']['shard1']]
    if not args.existing:
        with open(output/f'{mode}.log', 'w', encoding='utf-8') as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    raw = logits_path.read_bytes()
    vocab, rows = struct.unpack_from('<ii', raw)
    if vocab != x.size or rows != len(report['inputs']) or len(raw) != 8+vocab*rows*4:
        raise ValueError('Invalid native logits dimensions')
    native = np.frombuffer(raw, dtype='<f4', offset=8).reshape(rows, vocab)
    if not np.isfinite(native).all():
        raise ValueError('Non-finite native logits')
    y = native[-1].astype(np.float64)
    diff = x-y
    routing = routing_path.read_bytes()
    offset = record = exact = same_set = overlap = 0
    while offset < len(routing):
        layer, count = struct.unpack_from('<ii', routing, offset)
        if layer != record % 48 or count != 10 or offset+8+8*count > len(routing):
            raise ValueError('Invalid native routing record')
        ids = list(struct.unpack_from('<'+'i'*count, routing, offset+8))
        ref = portable_routes[(record//48, layer)]
        exact += ids == ref
        same_set += set(ids) == set(ref)
        overlap += len(set(ids) & set(ref))
        offset += 8+8*count
        record += 1
    if record != rows*48:
        raise ValueError('Incomplete native routing trace')
    results.append(dict(mode=mode, command=command, nativeLogitsSha256=digest(logits_path),
                        nativeRoutingSha256=digest(routing_path), positions=rows,
                        nativeLastToken=int(y.argmax()), portableLastToken=int(x.argmax()),
                        bitExact=bool(np.array_equal(x,y)), maxAbsoluteError=float(abs(diff).max()),
                        rmsError=float(np.sqrt(np.mean(diff*diff))),
                        cosine=float(x@y/(np.linalg.norm(x)*np.linalg.norm(y))),
                        routing=dict(records=record, orderedMatches=exact, setMatches=same_set,
                                     selectedExpertOverlap=overlap, selectedExperts=record*10)))
result = dict(timestamp=datetime.now(timezone.utc).isoformat(), upstreamVersion='0.1.34',
              upstreamCommit='1678de333d0e0711bc414ad992b640e1a37dd814',
              executableSha256=digest(args.exe), inputs=report['inputs'],
              portableKernelSha256=report['kernelSourceSha256'], portableLogitsSha256=digest(portable_path),
              note='This measures differences; agreement on one token is not a native parity or model-quality pass.',
              comparisons=results)
(ROOT/'reports/native-comparison.json').write_text(json.dumps(result,indent=2)+'\n')
for r in results:
    print(f"{r['mode']}: bitExact={r['bitExact']}, RMS={r['rmsError']:.6g}, max={r['maxAbsoluteError']:.6g}, "
          f"ordered routes={r['routing']['orderedMatches']}/{r['routing']['records']}")
