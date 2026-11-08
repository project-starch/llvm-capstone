#!/usr/bin/env python3
"""Summarize sampled host cycles; categories are disjoint self-cost groups."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import subprocess


def category(dso, symbol):
    if dso.startswith('[JIT]'):
        return 'translated_guest_code'
    if dso != 'qemu-system-riscv64':
        return 'host_libraries_or_unknown'
    if symbol.startswith('cap_mem_map') or symbol == 'find_entry':
        return 'tag_metadata'
    if symbol.startswith(('capstone_ldc', 'capstone_stc', 'helper_capstone_ldc', 'helper_capstone_stc')):
        return 'capability_load_store_helpers'
    if symbol in {'decode.isra.0','cap_compress','cap_uncompress','capstone_cursor_representable',
                  'capstone_cap_move_offset','capstone_set_moved_cap','helper_reg_set_cap_compressed',
                  'helper_cscincoffsetimm','helper_cscincoffset','helper_csmovc'}:
        return 'capability_representation_and_moves'
    if symbol.startswith('cap_rev_') or symbol in {'guest_read64','capstone_watch_page',
            'capstone_pre_mem_access','capstone_mem_access_allowed','capstone_pcc_fetch_ok'}:
        return 'lifetime_and_access_validation'
    if symbol.startswith(('tb_lookup','tb_htable_lookup','qht_lookup','cpu_exec')) or symbol in {
            'cpu_get_tb_cpu_state','helper_lookup_tb_ptr','cpu_tb_exec'}:
        return 'translation_block_dispatch'
    if symbol.startswith(('bitmap_set_atomic','cpu_physical_memory_set_dirty','find_next_bit')):
        return 'dirty_bitmap_bookkeeping'
    return 'other_qemu'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--perf',type=Path,required=True)
    p.add_argument('--vm',type=Path,required=True)
    a=p.parse_args()
    cmd=[str(a.perf),'report','-i',str(a.vm/'host-perf.data'),'--stdio','--no-children','-g','none',
         '--sort','dso,symbol','--show-total-period','--show-nr-samples','--percent-limit','0','-t',';']
    # perf uses /tmp/perf-PID.map. Restore the archived map only if absent;
    # never replace another run's map with the same PID.
    for saved in a.vm.glob('perf-*.map'):
        target=Path('/tmp')/saved.name
        if target.exists():
            if target.read_bytes()!=saved.read_bytes():
                raise RuntimeError(f'Conflicting JIT map: {target}; use an isolated /tmp')
        else:
            with target.open('xb') as f:f.write(saved.read_bytes())
    completed=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True)
    (a.vm/'host-perf-flat.txt').write_text(completed.stdout)
    (a.vm/'host-perf-report.stderr').write_text(completed.stderr)
    rows=[];groups=defaultdict(int);dsos=defaultdict(int)
    for line in completed.stdout.splitlines():
        if not re.match(r'\s*[\d.]+%\s*;',line):continue
        fields=[x.strip() for x in line.split(';')]
        _,samples,period,dso,symbol=fields[:5]
        symbol=re.sub(r'^\[.\]\s*','',symbol)
        group=category(dso,symbol)
        row=dict(symbol=symbol,dso=dso,samples=int(samples),period=int(period),category=group)
        rows.append(row);groups[group]+=row['period'];dsos[dso]+=row['period']
    total=sum(x['period'] for x in rows)
    events=[int(x) for x in re.findall(r'^# Event count \(approx\.\): (\d+)',completed.stdout,re.M)]
    if not total or total!=sum(events):raise ValueError(('Missing/inconsistent periods',total,events))
    for row in rows:row['percent']=100*row['period']/total
    result=dict(total_sampled_period=total,samples=sum(x['samples'] for x in rows),
                lost_samples=sum(map(int,re.findall(r'Total Lost Samples: (\d+)',completed.stdout))),
                categories={k:100*v/total for k,v in sorted(groups.items(),key=lambda x:-x[1])},
                shared_objects={k:100*v/total for k,v in sorted(dsos.items(),key=lambda x:-x[1])},
                symbols=sorted(rows,key=lambda x:-x['period']),
                scope='Userspace host cycle samples, self cost; no inclusive-cost addition or target-cycle claim.',
                classification_note='Explicit symbol groups; inlining and unknown callers limit attribution. decode.isra.0 and find_entry resolved to cap_compress.c and cap_mem_map.c in this pinned binary.')
    (a.vm/'host-perf-summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['symbols']},indent=2))


if __name__=='__main__':main()
