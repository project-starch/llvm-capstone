#!/usr/bin/env python3
"""Run the declared native workload matrix and validate every measured process."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import pwd
import random
import re
import shutil
import signal
import socket
import subprocess
import time

HERE = Path(__file__).resolve().parent
WORK = Path(os.environ.get('CAPSTONE_TMP_ROOT', '/tmp/capstone')) / 'native-survey'
PLAN = json.loads((HERE/'workloads.json').read_text())

def sha(data):
    return hashlib.sha256(data).hexdigest()

def binary(app, variant):
    root = WORK/'build'/variant/app
    if app == 'sqlite': return WORK/'install'/variant/app/'speedtest1'
    if app == 'postgresql': return WORK/'install'/variant/app/'bin/postgres'
    if app == 'cpython':
        return root/('python.exe' if (root/'python.exe').is_file() else 'python')
    if app == 'mruby': return root/'build/host/bin/mruby'
    if app == 'perl': return Path('/var/tmp/native-survey-build')/variant/app/'perl'
    if app == 'ffmpeg': return root/'ffmpeg'
    if app == 'wireshark': return root/'out/run/tshark'
    if app == 'memcached': return root/'memcached'
    raise ValueError(app)

def run(command, folder, label, env=None, data=None, user=None, timeout=600):
    command = [str(x) for x in command]
    kwargs = {}
    if user is not None:
        kwargs.update(user=user.pw_uid, group=user.pw_gid, extra_groups=[])
    with (folder/(label+'.stdout')).open('wb') as out, (folder/(label+'.stderr')).open('wb') as err:
        result = subprocess.run(command, input=data, stdout=out, stderr=err, env=env,
                                timeout=timeout, **kwargs)
    if result.returncode:
        raise RuntimeError(f'{label}: exit {result.returncode}, see {folder}')
    return (folder/(label+'.stdout')).read_bytes()

def check_counts(folder):
    paths = list(folder.glob('counts.*.json'))
    if len(paths) != 1:
        raise ValueError(f'expected exactly one process report, found {len(paths)}')
    data = json.loads(paths[0].read_text())
    if not data['allocators'] or not data['backing_acquires']:
        raise ValueError('empty instrumentation report')
    for row in data['allocators']:
        if row['alloc'] != row['free'] + row['live']:
            raise ValueError('lifetime accounting mismatch')
        if row['reuse'] != row['inside'] + row['outside'] + row['unknown_reuse']:
            raise ValueError('reuse partition mismatch')
        if row['unknown_free']:
            raise ValueError(f"unmatched retirement: {row['family']} {row['unknown_free']}")
        # Unknown backing is preserved, never interpreted as invisible reuse.
        if row['unknown_alloc']:
            raise ValueError(f"incomplete backing coverage: {row['family']} {row['unknown_alloc']}")
    return data

def sqlite(spec, variant, folder, env):
    database=folder/'test.db'
    command=[binary('sqlite',variant), *spec['arguments'], database]
    run(command,folder,'application',env)
    shell=WORK/'install/baseline/sqlite/sqlite3'
    integrity=run([shell,database,'PRAGMA integrity_check;'],folder,'integrity')
    if integrity.strip()!=b'ok': raise ValueError('SQLite integrity check failed')
    dump=run([shell,database,'.dump'],folder,'oracle')
    if b'CREATE TABLE' not in dump: raise ValueError('empty database oracle')
    return command, dict(dump_sha256=sha(dump),integrity='ok')

def cpython(spec, variant, folder, env):
    kind='loads' if spec['id'].endswith('loads') else 'dumps'
    env=dict(env,PYTHONPATH=str(WORK/'python-deps'),PYTHONHASHSEED='0')
    command=[binary('cpython',variant),HERE/'workload-json.py',WORK/'downloads',kind,spec['iterations']]
    output=run(command,folder,'application',env)
    return command,json.loads(output)

def mruby(spec, variant, folder, env):
    command=[binary('mruby',variant),WORK/'original/mruby/benchmark/bm_ao_render.rb',spec['width']]
    output=run(command,folder,'application',env)
    header=f"P6\n{spec['width']} {spec['width']}\n255\n".encode()
    if not output.startswith(header) or len(output)!=len(header)+3*spec['width']**2:
        raise ValueError('invalid AO image')
    return command,dict(ppm_sha256=sha(output),width=spec['width'],height=spec['width'])

def perl(spec, variant, folder, env):
    exe=binary('perl',variant)
    command=[exe,'-I'+str(exe.parent/'lib'),WORK/'downloads/binarytrees.pl',spec['depth']]
    output=run(command,folder,'application',env)
    depth=spec['depth']
    checks=[(depth+1,2**(depth+2)-1)]
    expected=f'stretch tree of depth {checks[0][0]}\t check: {checks[0][1]}\n'
    for d in range(4,depth+1,2):
        iterations=2**(depth-d+4)
        expected+=f'{iterations}\t trees of depth {d}\t check: {iterations*(2**(d+1)-1)}\n'
    expected+=f'long lived tree of depth {depth}\t check: {2**(depth+1)-1}\n'
    if output!=expected.encode(): raise ValueError('binary-trees analytic oracle failed')
    return command,dict(stdout_sha256=sha(output),depth=depth)

def ffmpeg(spec, variant, folder, env):
    sample='xvid.h263' if spec['id'].endswith('xvid') else 'resize.h263'
    command=[binary('ffmpeg',variant),'-nostdin','-v','error','-threads','1',
             '-f','m4v','-i',WORK/'downloads'/sample,'-an','-threads','1',
             '-filter_threads','1','-f','framemd5','-']
    output=run(command,folder,'application',env)
    lines=[s for s in output.splitlines() if s and not s.startswith(b'#')]
    if len(lines)!=spec['frames']: raise ValueError(f'frame count {len(lines)}')
    return command,dict(framemd5_sha256=sha(output),frames=len(lines))

def wireshark(spec, variant, folder, env):
    capture=WORK/'original/wireshark/test/captures'/spec['capture']
    # The full tree includes decoded protocol values. Disable name resolution
    # and use the same original capture path in both configurations.
    command=[binary('wireshark',variant),'-n','-r',capture,'-T','json']
    output=run(command,folder,'application',env)
    packets=json.loads(output)
    if not packets: raise ValueError('capture produced no packets')
    normalized=json.dumps(packets,sort_keys=True,separators=(',',':')).encode()
    return command,dict(protocol_tree_sha256=sha(normalized),packets=len(packets),
                        capture_sha256=sha(capture.read_bytes()))

def pg_input(spec):
    # Reuse the exact upstream transaction SQL. Parameter sampling is this
    # harness's deterministic Python PRNG, not pgbench's client PRNG.
    text=(WORK/'original/postgresql/src/bin/pgbench/pgbench.c').read_text()
    start=text.index('\t\t"BEGIN;\\n"',text.index('static const BuiltinScript builtin_script[]'))
    end=text.index('\t\t"END;\\n"',start)+len('\t\t"END;\\n"')
    sql=''.join(ast.literal_eval(line.strip()) for line in text[start:end].splitlines())
    rng=random.Random(spec['seed'])
    output='SET max_parallel_workers_per_gather=0;\n'
    total=0
    for _ in range(spec['transactions']):
        values=dict(aid=rng.randint(1,100000),bid=1,tid=rng.randint(1,10),delta=rng.randint(-5000,5000))
        total+=values['delta']
        transaction=sql
        for key,value in values.items(): transaction=transaction.replace(':'+key,str(value))
        output+=transaction
    output+="SELECT 'NS_ORACLE', (SELECT sum(abalance) FROM pgbench_accounts), (SELECT sum(tbalance) FROM pgbench_tellers), (SELECT sum(bbalance) FROM pgbench_branches), (SELECT count(*) FROM pgbench_history), (SELECT sum(delta) FROM pgbench_history);\n"
    return output.encode(),total

def postgresql(spec, variant, folder, env):
    user=pwd.getpwnam('survey')
    folder.chmod(0o777)
    data=folder/'cluster'
    data.mkdir(mode=0o700)
    os.chown(data,user.pw_uid,user.pw_gid)
    init=binary('postgresql','baseline').parent/'initdb'
    run([init,'-D',data,'--no-locale','--encoding=UTF8','--auth=trust'],folder,'initdb',user=user)
    setup=b'''CREATE TABLE pgbench_accounts (aid integer PRIMARY KEY, bid integer, abalance integer, filler char(84));
CREATE TABLE pgbench_branches (bid integer PRIMARY KEY, bbalance integer, filler char(88));
CREATE TABLE pgbench_tellers (tid integer PRIMARY KEY, bid integer, tbalance integer, filler char(84));
CREATE TABLE pgbench_history (tid integer, bid integer, aid integer, delta integer, mtime timestamp, filler char(22));
INSERT INTO pgbench_accounts SELECT i,1,0,'' FROM generate_series(1,100000) i;
INSERT INTO pgbench_branches VALUES (1,0,'');
INSERT INTO pgbench_tellers SELECT i,1,0,'' FROM generate_series(1,10) i;
VACUUM ANALYZE;
'''
    run([binary('postgresql','baseline'),'--single','-D',data,'postgres'],folder,'setup',data=setup,user=user)
    payload,total=pg_input(spec)
    (folder/'work.sql').write_bytes(payload)
    command=[binary('postgresql',variant),'--single','-D',data,'postgres']
    output=run(command,folder,'application',env,data=payload,user=user)
    errors=(folder/'application.stderr').read_text()
    if re.search(r'\b(ERROR|FATAL|PANIC):',errors): raise ValueError('backend reported SQL failure')
    if b'NS_ORACLE' not in output: raise ValueError('missing database oracle')
    final=output[output.rfind(b'NS_ORACLE'):]
    values=[int(x) for x in re.findall(rb'= "(-?\d+)"',final)]
    expected=[total,total,total,spec['transactions'],total]
    if values!=expected: raise ValueError(f'pgbench balance oracle {values}, expected {expected}')
    return command,dict(stdout_sha256=sha(output),sql_sha256=sha(payload),transactions=spec['transactions'],balance_sum=total)

def memcached(spec, variant, folder, env):
    with socket.socket() as probe:
        probe.bind(('127.0.0.1',0))
        port=probe.getsockname()[1]
    command=[binary('memcached',variant),'-u','root','-l','127.0.0.1','-p',str(port),
             '-U','0','-t',str(spec['workers']),'-m','64']
    with (folder/'application.stdout').open('wb') as out, (folder/'application.stderr').open('wb') as err:
        server=subprocess.Popen([str(x) for x in command],env=env,stdout=out,stderr=err)
        try:
            for _ in range(100):
                if server.poll() is not None: raise ValueError('server stopped during startup')
                try:
                    with socket.create_connection(('127.0.0.1',port),timeout=.2): pass
                    break
                except OSError: time.sleep(.05)
            else: raise ValueError('server did not start')
            client=WORK/'tools/memtier_benchmark-2.2.1/memtier_benchmark'
            common=[client,'--server=127.0.0.1',f'--port={port}','--protocol=memcache_text',
                    '--threads=1','--clients=1','--pipeline=1','--key-prefix=ns:',
                    '--key-minimum=1',f"--key-maximum={spec['keys']}",
                    f"--data-size={spec['value_bytes']}",'--key-pattern=S:S','--hide-histogram']
            run([*common,'--ratio=1:0',f"--requests={spec['warmup_sets']}"],folder,'warmup')
            run([*common,f"--ratio={spec['ratio']}",f"--requests={spec['requests']}",
                 '--json-out-file='+str(folder/'memtier.json')],folder,'client')
            with socket.create_connection(('127.0.0.1',port),timeout=5) as sock:
                stream=sock.makefile('rb')
                sock.sendall(b'stats\r\n')
                stats={}
                while True:
                    line=stream.readline()
                    if line==b'END\r\n': break
                    if not line: raise ValueError('short stats response')
                    _,key,value=line.strip().split(maxsplit=2)
                    stats[key.decode()]=value.decode()
                expected_set=spec['warmup_sets']+spec['requests']//2
                expected_get=spec['requests']//2
                if int(stats['cmd_set'])!=expected_set or int(stats['cmd_get'])!=expected_get:
                    raise ValueError('memtier operation count mismatch')
                if int(stats['get_misses']) or int(stats['curr_items'])!=spec['keys']:
                    raise ValueError('missing cache keys')
                digest=hashlib.sha256()
                for key in range(1,spec['keys']+1):
                    sock.sendall(f'get ns:{key}\r\n'.encode())
                    header=stream.readline()
                    expected=f"VALUE ns:{key} 0 {spec['value_bytes']}\r\n".encode()
                    if header!=expected: raise ValueError(f'invalid cache value header {header!r}')
                    value=stream.read(spec['value_bytes']+2)
                    if value!=b'x'*spec['value_bytes']+b'\r\n' or stream.readline()!=b'END\r\n':
                        raise ValueError('cache readback mismatch')
                    digest.update(value[:-2])
                oracle=dict(sets=expected_set,gets=expected_get,readback_keys=spec['keys'],values_sha256=digest.hexdigest())
            server.send_signal(signal.SIGUSR1)
            if server.wait(timeout=20): raise ValueError('unclean server exit')
        finally:
            if server.poll() is None:
                server.terminate()
                try: server.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    server.kill(); server.wait()
    return command,oracle

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--applications',nargs='+')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--repetitions',type=int,default=PLAN['repetitions'])
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    specs=[]
    for original in PLAN['workloads']:
        if args.applications and original['application'] not in args.applications: continue
        if original['application']=='wireshark':
            for capture in original['captures']:
                specs.append(dict(original,id='tshark-'+Path(capture).stem,capture=capture))
        else: specs.append(original)
    summaries=[]
    for spec in specs:
        reference=None
        for repetition in range(1,args.repetitions+1):
            for variant in ['baseline','observed']:
                folder=args.output/spec['id']/f'{variant}-{repetition}'
                folder.mkdir(parents=True,exist_ok=False)
                env=dict(os.environ,LC_ALL='C',TZ='UTC')
                env.pop('NS_OUT',None)
                env.pop('LD_PRELOAD',None)
                if variant=='observed': env['NS_OUT']=str(folder/'counts')
                record=dict(workload=spec['id'],application=spec['application'],variant=variant,
                            repetition=repetition,status='running',spec=spec,
                            binary_sha256=sha(binary(spec['application'],variant).read_bytes()))
                record_path=folder/'result.json'
                record_path.write_text(json.dumps(record,indent=2)+'\n')
                try:
                    command,oracle=globals()[spec['application']](spec,variant,folder,env)
                    if reference is None: reference=oracle
                    if oracle!=reference: raise ValueError(f'functional oracle differs: {oracle} != {reference}')
                    record.update(command=[str(x) for x in command],oracle=oracle,status='pass')
                    if variant=='observed': record['measurement']=check_counts(folder)
                except Exception as error:
                    record.update(status='failed',error=str(error))
                    raise
                finally:
                    record_path.write_text(json.dumps(record,indent=2)+'\n')
                summaries.append(record)
                print(spec['id'],variant,repetition,'PASS',flush=True)
    (args.output/'summary.json').write_text(json.dumps(summaries,indent=2)+'\n')

if __name__=='__main__':
    main()
