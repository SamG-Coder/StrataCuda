"""Use Strata's unchanged tokenizer with the extracted pack metadata (JSON stdin/stdout)."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0,str(ROOT/'vendor/strata/tools'))
from strata_tokenizer import Tokenizer

ap=argparse.ArgumentParser();ap.add_argument('--pack',required=True,type=Path);args=ap.parse_args()
p=args.pack/'tokenizer'
vocab=json.loads((p/'vocab.json').read_text(encoding='utf-8'));tokens=[None]*len(vocab)
for text,index in vocab.items():tokens[index]=text
cfg=json.loads((p/'tokenizer.json').read_text(encoding='utf-8'))
tokenizer=Tokenizer(tokens,(p/'merges.txt').read_text(encoding='utf-8').splitlines(),json.loads((p/'token_type.json').read_text(encoding='utf-8')),cfg['pre'],cfg['special_ids'])
sys.stdin.reconfigure(encoding='utf-8')
request=json.load(sys.stdin);result={}
if 'text' in request:
    result['tokens']=tokenizer.encode(request['text'],parse_special=request.get('parse_special',False))
    result['roundTrip']=tokenizer.decode(result['tokens'])
if 'ids' in request:result['text']=tokenizer.decode(request['ids'])
if 'checks' in request:
    result['checks']=[{'tokens':tokenizer.encode(c['text'],parse_special=c.get('parse_special',False)),'text':c['text']} for c in request['checks']]
if 'chats' in request:
    from jinja2.sandbox import SandboxedEnvironment
    environment=SandboxedEnvironment()
    def raise_exception(message):raise ValueError(message)
    environment.globals['raise_exception']=raise_exception
    template=environment.from_string((p/'chat_template.jinja').read_text(encoding='utf-8'))
    result['chats']=[template.render(messages=([{'role':'system','content':c['system']}] if c.get('system') else [])+c['messages'],enable_thinking=False,add_generation_prompt=True) for c in request['chats']]
sys.stdout.reconfigure(encoding='utf-8')
print(json.dumps(result,ensure_ascii=False))
