#!/usr/bin/env python3
"""Build prepared SQLite memory-study arms using the existing port builders.

Source capstone/tests/capstone-test-env.sh first. Fetched and generated files
stay in the explicitly supplied scratch directory. Every tool invocation is
recorded. This tool does not manage VMs.
"""
import argparse
import json
import os
import subprocess
from pathlib import Path

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
PORT=REPO/'capstone/ports/sqlite'


def recorder(path, real):
    path.write_text('#!/usr/bin/env python3\nimport os,sys,json\n'
                    'with open(os.environ["STUDY_TOOL_LOG"],"a") as f:\n'
                    f' f.write(json.dumps([{str(real)!r},*sys.argv[1:]])+"\\n")\n'
                    f'os.execv({str(real)!r},[{str(real)!r},*sys.argv[1:]])\n')
    path.chmod(0o755)


def run(cmd,log,env=None):
    with log.open('w') as f:
        subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env=env,check=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('prepared',type=Path)
    p.add_argument('--platform',choices=['capstone','poisoncap'],required=True)
    p.add_argument('--sdk',type=Path,help='PoisonCap CHERI SDK')
    p.add_argument('--sysroot',type=Path,help='Published PoisonCap purecap sysroot')
    p.add_argument('--skip-host',action='store_true',
                   help='reuse a separately identified compatible domain host')
    args=p.parse_args();out=args.prepared.resolve()
    if not (out/'inputs.json').exists():p.error('run prepare-sqlite-memory.py first')
    if args.platform=='capstone':
        for key in ['CAPSTONE_CLANG','CAPSTONE_LD_LLD','CAPSTONE_BUILDROOT_DIR','GUEST_CC']:
            if not os.environ.get(key):p.error(f'{key} must name the intended tool/platform')
        recorder(out/'record-clang.py',Path(os.environ['CAPSTONE_CLANG']).absolute())
        recorder(out/'record-lld.py',Path(os.environ['CAPSTONE_LD_LLD']).absolute())
        for arm in ['capstone','capstone-sublet']:
            env=os.environ.copy()
            # Do not inherit a diagnostic/feature override from another experiment.
            for key in ['SQLITE_SUBLET_PATCH','SQLITE_FLOAT','SQLITE_TRIM','SQLITE_JSON','SQLITE_RTREE',
                        'SQLITE_DIAG','EXTRA_MLLVM','AMALGAM_EXTRA_MLLVM','SUPPORT_EXTRA_MLLVM']:
                env.pop(key,None)
            env.update(CAPSTONE_CLANG=str(out/'record-clang.py'),CAPSTONE_LD_LLD=str(out/'record-lld.py'),
                       STUDY_TOOL_LOG=str(out/(arm+'-tools.jsonl')),OUT_DIR=str(out/(arm+'-build')),
                       PATCHED_SQLITE=str(out/'capstone-port-base.c'),DOMAIN_SRC=str(PORT/'speedtest1_measure.c'),
                       SQLITE_AMALGAMATION_HEADER=str(out/'sqlite3.h'),
                       SQLITE_SPEEDTEST1_SRC=str(out/'speedtest1.c'),SPEEDTEST1_STUB_CLOCK='1',
                       SQLITE_OPT_LEVEL='-O0',SQLITE_FEATURE_SET='deployed',
                       DOMAIN_EXTRA_DEFS='-DSQLITE_HC_REGION_SIZE=1048576UL')
            if arm=='capstone':
                env.update(SQLITE_STUDY_PATCH=str(out/'capstone.patch'))
                env['DOMAIN_EXTRA_DEFS']+=' -DCAPSTONE_SPEEDTEST1_REGION_ARENA=1'
            else:
                env.update(SQLITE_SUBLET_PATCH=str(PORT/'sublet/sublet-3220000-memsys5.patch'),
                           SQLITE_STUDY_PATCH=str(out/'capstone-sublet-observer.patch'))
                env['DOMAIN_EXTRA_DEFS']+=' -DSPEEDTEST1_SUBLET=1'
            run(['bash',str(PORT/'build-sqlite-silicon.sh')],out/(arm+'-build.log'),env)
        if not args.skip_host:
            env=os.environ.copy();env.update(OUT_DIR=str(out),HOST_EXTRA_DEFS='-DSQLITE_HC_REGION_SIZE=1048576UL')
            run(['bash',str(PORT/'build-sqlite-host.sh')],out/'host-build.log',env)
    else:
        if not args.sdk or not args.sysroot:p.error('--sdk and --sysroot are required')
        sdk=args.sdk.resolve();sysroot=args.sysroot.resolve()
        text=(PORT/'build-sqlite-capstone.sh').read_text().split('SQLITE_DEFINES=(\n')[1].split('\n)')[0]
        flags=[s for s in text.split() if s!='-DSQLITE_OS_OTHER=1']
        (out/'speedtest1-shared.c').write_text((out/'speedtest1.c').read_text().replace('randomFunc','speedtest1_randomFunc'))
        for arm in ['poisoncap-spatial','poisoncap-temporal']:
            tu=out/(arm+'-tu.c');tu.write_text(f'#include "{arm}.c"\n#include "speedtest1-shared.c"\n')
            cmd=[str(sdk/'bin/clang'),'--target=riscv64-unknown-freebsd13','--sysroot='+str(sysroot),
                 '-march=rv64imafdcxcheri','-mabi=l64pc128d','-mno-relax','-fuse-ld='+str(sdk/'bin/ld.lld'),
                 '-O0',*flags,'-DSQLITE_HEAP_SIZE=262144','-DSQLITE_HC_REGION_SIZE=1048576UL',
                 '-include','stdint.h','-include','sys/types.h','-include','sys/uio.h','-include','signal.h',
                 '-I'+str(out),str(tu),'-lm','-o',str(out/arm)]
            (out/(arm+'-argv.json')).write_text(json.dumps(cmd,indent=2)+'\n')
            run(cmd,out/(arm+'-build.log'))


if __name__=='__main__':main()
