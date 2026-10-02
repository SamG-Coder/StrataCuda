"""Time the pinned upstream binary without changing its source or model weights."""
import argparse
import hashlib
import json
import re
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--runs', type=int, default=3)
parser.add_argument('--new-tokens', type=int, default=1)
parser.add_argument('--variant', choices=['native', 'cached'], default='native')
args = parser.parse_args()
if not 1 <= args.runs <= 10 or not 1 <= args.new_tokens <= 128:
    parser.error('Use 1-10 runs and 1-128 generated tokens')
exe = ROOT / '.local/native-upstream/bin/strata.exe'
pack = ROOT / 'models/Qwen3.8-Flash-Next-WebCuda-Q2_0'
manifest = json.loads((pack / 'manifest.json').read_text())
build = json.loads((exe.parent / 'BUILD.json').read_text())
digest = hashlib.sha256(exe.read_bytes()).hexdigest()
if build['version'] != '0.1.34' or digest != 'f0838beeb5b630483262456b1d0a596c76c7e19633bbb58eebe27ab86b497bb4':
    raise RuntimeError('Expected the verified, unchanged upstream v0.1.34 executable')
inputs = [760, 6511, 314, 9338, 369, 11751]
command = [str(exe), '--pack', str(pack), '--tokens', ','.join(map(str, inputs)),
           '--max-new', str(args.new_tokens), '--max-context', '256', '--mmap-experts',
           '--pool-workers', '4', '--ple-gguf', manifest['source']['shard2'],
           '--native', manifest['source']['shard1']]
if args.variant == 'cached':
    command += ['--expert-cache', '3072', '--expert-cache-per-layer', '--prefill', '16']
label = f'upstream-{args.variant}-{args.new_tokens}'
folder = ROOT / '.local/native-benchmark'
folder.mkdir(parents=True, exist_ok=True)
samples = []
for run in range(args.runs):
    start = time.perf_counter()
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=600)
    wall = time.perf_counter() - start
    log = result.stdout + '\n' + result.stderr
    log_path = folder / f'{label}-{run + 1}.log'
    log_path.write_text(log, encoding='utf-8')
    if result.returncode:
        raise RuntimeError(f'Upstream failed: {log_path}\n{log[-3000:]}')
    decode = re.search(r'^decode\s+(\d+) tokens in ([\d.]+) ms\s+->\s+([\d.]+) tok/s', log, re.M)
    prefill = re.search(r'^prefill\s+(\d+) tokens in ([\d.]+) ms\s+->\s+([\d.]+) tok/s\s+\(time to first token ([\d.]+) ms\)', log, re.M)
    output = re.search(r'^output\s*:\s*(.*)$', log, re.M)
    if not decode or not prefill or not output:
        raise RuntimeError(f'Missing upstream timing fields: {log_path}')
    sample = {'run': run + 1, 'processSeconds': wall, 'generatedTokens': int(decode[1]),
              'decodeMs': float(decode[2]), 'tokensPerSecond': float(decode[3]),
              'prefillTokens': int(prefill[1]), 'prefillMs': float(prefill[2]),
              'timeToFirstTokenMs': float(prefill[4]),
              'output': [int(x) for x in output[1].split()],
              'logSha256': hashlib.sha256(log_path.read_bytes()).hexdigest()}
    if sample['generatedTokens'] != args.new_tokens or sample['output'][0] != 13:
        raise RuntimeError(f'Unexpected output: {sample}')
    if samples and sample['output'] != samples[0]['output']:
        raise RuntimeError('Repeated native runs generated different token sequences')
    samples.append(sample)
    print(json.dumps(sample), flush=True)
report = {'timestamp': datetime.now(timezone.utc).isoformat(), 'version': build['version'],
          'upstreamCommit': '1678de333d0e0711bc414ad992b640e1a37dd814',
          'executableSha256': digest, 'variant': args.variant, 'inputs': inputs,
          'command': command, 'samples': samples,
          'median': {key: statistics.median(s[key] for s in samples) for key in
                     ['processSeconds', 'decodeMs', 'tokensPerSecond', 'prefillMs', 'timeToFirstTokenMs']},
          'note': 'Each sample starts a fresh process. Upstream decode timing excludes weight loading and initial setup; processSeconds includes them and shutdown. Reported time to first token also excludes weight loading. Fixed generation length continues past end-of-turn tokens. OS file cache is not flushed. Native arithmetic differs from the WebCuda portable F32 contract. No MTP/speculative decoding.'}
destination = ROOT / 'reports' / f'benchmark-{label}.json'
destination.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
print('MEDIAN', json.dumps(report['median']), flush=True)
