"""Build and fully verify the supported Strata Q2_0 pack, then extract its tokenizer."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--gguf',required=True,type=Path,help='First original Q2_0 GGUF shard; both shards must be alongside one another')
ap.add_argument('--out',required=True,type=Path,help='New or empty output directory')
args=ap.parse_args()
source=args.gguf.resolve();destination=args.out.resolve()
if not source.is_file():ap.error('The first GGUF shard does not exist')
if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
    ap.error('Output must be new or empty; an existing pack will not be overwritten')

def run(script,*arguments):
    subprocess.run([sys.executable,'-B',str(ROOT/script),*map(str,arguments)],cwd=ROOT,check=True)

# The bounded verifier hashes the source and output after checking every value.
run('vendor/strata/tools/strata_pack.py','build','--gguf',source,'--out',destination,'--skip-hash')
run('scripts/verify-model.py','--gguf',source,'--pack',destination)
run('vendor/strata/tools/strata_tokenizer.py','--gguf',source,'--out',destination,'--check')
# A single browser file dialog must be able to select the pack and PLE together.
# On the same volume, a hard link shares the existing 28.8 GB file without a copy.
manifest=json.loads((destination/'manifest.json').read_text(encoding='utf-8'))
ple=Path(manifest['source']['shard2'])
try:
    os.link(ple,destination/ple.name)
    print('PLE shard linked without duplicating its data.',flush=True)
except OSError:
    print('Hard links unavailable across these locations; copying the PLE shard.',flush=True)
    shutil.copyfile(ple,destination/ple.name)
card=source.parent/'MODEL_CARD.md'
if card.is_file():shutil.copyfile(card,destination/'MODEL_CARD.md')
print(f'Converted and verified: {destination}',flush=True)
