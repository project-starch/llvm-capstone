#!/usr/bin/env python3
"""Fetch pinned upstream source archives into scratch, never into the repository."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
DEST = Path(os.environ.get('CAPSTONE_TMP_ROOT', '/tmp/capstone')) / 'native-survey'
SOURCES = json.loads((HERE / 'source-lock.json').read_text())

def fetch(item):
 name, spec = item
 suffix='.zip' if spec['url'].endswith('.zip') else '.tar'
 target=DEST/'downloads'/(name+suffix)
 target.parent.mkdir(parents=True,exist_ok=True)
 if not target.exists():
  subprocess.run(['curl','--fail','--location','--silent','--show-error','--retry','2',spec['url'],'-o',str(target)+'.part'],check=True)
  Path(str(target)+'.part').rename(target)
 digest=hashlib.sha256(target.read_bytes()).hexdigest()
 if spec.get('sha256') and digest != spec['sha256']:
  raise ValueError(name+': upstream hash mismatch')
 print(name,spec['version'],digest,flush=True)
 return name,dict(spec,sha256=digest,archive='downloads/'+target.name)

if __name__=='__main__':
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
  result=dict(pool.map(fetch,SOURCES.items()))
 (DEST/'sources.json').write_text(json.dumps(result,indent=2)+'\n')
