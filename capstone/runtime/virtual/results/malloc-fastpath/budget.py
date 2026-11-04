#!/usr/bin/env python3
"""Per-call heap budget from a profile.sh run (capprof windows), one arm.

usage: budget.py RUN --kind cap|rv64 --nm NM --objdump OBJDUMP [--json OUT] [--annotate DIR] [--exe RETAINED_EXECUTABLE]

Sums the attempts of every program instruction over ALL windows, maps each to its function by the
executable's symbol table (domain images: slid by the launcher's entry from image-records.txt; all
placements must agree), and prints: per-function attempts and entry counts for the heap functions,
the public call counts (entries of malloc/calloc/realloc/free/...), and the heap total per public
call. With --annotate it writes, per heap function, the disassembly with source lines and counts.
No data is an error, never a zero.
"""
import argparse, collections, json, os, re, subprocess, sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('run'); ap.add_argument('--kind', required=True, choices=('cap', 'rv64'))
ap.add_argument('--nm', required=True); ap.add_argument('--objdump', required=True)
ap.add_argument('--json'); ap.add_argument('--annotate')
ap.add_argument('--exe', help='retained copy of the recorded executable')
A = ap.parse_args()
run = Path(A.run)

invs = []
for f in sorted((run / 'invocations').glob('[0-9][0-9][0-9][0-9].json')):
    d = json.loads(f.read_text())
    if d.get('route') in ('guest', 'guest-plain'):
        invs.append(d)
if not invs:
    sys.exit(f'no guest invocations under {run}')
exes = {d['exe'] for d in invs}
if len(exes) != 1:
    sys.exit(f'more than one executable: {exes}')
exe = A.exe or exes.pop()

nm = subprocess.run([A.nm, '--format=sysv', '--defined-only', exe], capture_output=True, text=True, check=True).stdout
syms = []
for line in nm.splitlines():
    f = [x.strip() for x in line.split('|')]
    if len(f) >= 7 and f[3] == 'FUNC' and f[1] and f[4]:
        syms.append((int(f[1], 16), int(f[4], 16), f[0]))
syms.sort()
starts = [s[0] for s in syms]
code_lo, code_hi = None, None
slide = 0
if A.kind == 'cap':
    recs = re.findall(r'entry=(0x[0-9a-f]+) code=(0x[0-9a-f]+)-(0x[0-9a-f]+)', (run / 'image-records.txt').read_text())
    if not recs or len(set(recs)) != 1:
        sys.exit(f'domain placements are not unique: {sorted(set(recs))[:4]}')
    entry, code_lo, code_hi = (int(x, 16) for x in recs[0])
    dm = [s[0] for s in syms if s[2] == 'domain_main']
    slide = entry - dm[0]

import bisect
def func_of(link):
    i = bisect.bisect_right(starts, link) - 1
    if i < 0: return None
    a, size, name = syms[i]
    if size and link >= a + size: return None
    return name

per_insn = collections.Counter()
words = {}
windows = sorted((run / 'capprof').glob('window-*.tsv'))
if not windows:
    sys.exit('no capprof windows')
raw_user_total = 0
user_total = json.loads((run / 'profile.json').read_text())['classes']['program']
for w in windows:
    for line in w.open():
        if line.startswith('#'): continue
        va, ln, b, att, tr = line.rstrip('\n').split('\t')
        va = int(va, 16); att = int(att)
        if not att or va >= 0x8000000000000000: continue
        if code_lo is not None and not (code_lo <= va < code_hi): continue
        raw_user_total += att
        per_insn[va - slide] += att
        if func_of(va - slide):
            word = int(b, 16)
            if va - slide in words and words[va - slide] != word:
                sys.exit('instruction bytes differ across windows')
            words[va - slide] = word
if not per_insn:
    sys.exit('no program attempts')

func_att = collections.Counter(); func_entry = collections.Counter()
symaddr = {s[2]: s[0] for s in syms}
for link, att in per_insn.items():
    fn = func_of(link)
    func_att[fn or '?'] += att
for name, a in symaddr.items():
    if a in per_insn: func_entry[name] = per_insn[a]

HEAP = {
 'cap': ['malloc', 'allocate', 'publish', 'calloc', 'realloc', 'free', 'checked', 'aligned_alloc',
         'posix_memalign', 'malloc_usable_size', '__libc_malloc', '__libc_free', '__libc_realloc',
         '__capstone_mallocng_malloc', 'enframe', 'alloc_slot', 'alloc_group', 'try_avail',
         '__capstone_mallocng_free_slot', '__capstone_mallocng_free', '__capstone_mallocng_realloc',
         'nontrivial_free', 'free_group', '__capstone_malloc_prepare_free', '__capstone_malloc_slot_take',
         '__capstone_malloc_slot_key', '__capstone_malloc_slot_release', '__capstone_malloc_slot_start',
         '__capstone_malloc_find', '__capstone_malloc_nested_release', '__capstone_malloc_group_nested',
         '__capstone_malloc_group_map', '__capstone_malloc_group_unmap', '__capstone_malloc_slots_new',
         '__capstone_malloc_slots_free', 'bind_slots', 'exact_or_moved', 'record_of_result', 'unindex',
         'prepare_slot', 'issue_child', 'coarse_take', 'coarse_release', 'recycle_take', 'root_remove',
         'root_enter', '__malloc_alloc_meta', 'alloc_meta', 'lock_heap', 'ensure_ids'],
 'rv64': ['malloc', 'calloc', 'realloc', 'free', 'aligned_alloc', 'posix_memalign', '__libc_malloc_impl',
          '__libc_malloc', '__libc_free', '__libc_realloc', 'enframe', 'alloc_slot', 'alloc_group', 'try_avail',
          'get_meta', 'nontrivial_free', 'free_group', '__malloc_alloc_meta', 'alloc_meta',
          '__malloc_allzerop', 'is_allzero'],
}[A.kind]
API = ['malloc', 'calloc', 'realloc', 'free', 'aligned_alloc', 'posix_memalign', 'malloc_usable_size']
heap_total = sum(func_att[f] for f in HEAP)
calls = {f: func_entry.get(f, 0) for f in API}
ncalls = sum(calls.values())
print(f'exe {exe}')
print(f'windows {len(windows)} program_attempts {user_total}')
print('api_entries', calls)
print(f'heap_attempts {heap_total} ({100*heap_total/user_total:.2f}% of program) per_public_call {heap_total/ncalls:.2f}')
print(f'{"function":40s} {"attempts":>14s} {"entries":>11s} {"per_entry":>9s}')
for f in sorted(HEAP, key=lambda f: -func_att[f]):
    if func_att[f]:
        e = func_entry.get(f, 0)
        print(f'{f:40s} {func_att[f]:14d} {e:11d} {func_att[f]/e if e else float("nan"):9.2f}')
for name in ('memset', 'memcpy', 'Perl_safesysmalloc', 'Perl_safesysfree', 'Perl_safesysrealloc', 'Perl_safesyscalloc'):
    print(f'{name:40s} {func_att[name]:14d} {func_entry.get(name, 0):11d}')
primitives = collections.defaultdict(collections.Counter)
for pc, word in words.items():
    fn = func_of(pc)
    count = per_insn[pc]
    if A.kind == 'cap' and word & 127 == 0x5b and (word >> 12) & 7 == 1:
        if word >> 25 == 0x51: primitives[fn]['CDERIVE'] += count
        if word >> 25 == 0x52: primitives[fn]['CREVOKE'] += count
    if word & 127 == 0x2f: primitives[fn]['atomic'] += count
    if word & 127 == 0x0f: primitives[fn]['fence'] += count
    if fn == 'checked' and word & 0x7fff == 0x4003:
        primitives[fn]['validation_load'] += count
print('heap_primitives', {f: dict(primitives[f]) for f in HEAP if primitives[f]})
if A.json:
    Path(A.json).write_text(json.dumps(dict(exe=exe, program=user_total, api=calls, heap=heap_total, primitives={f: dict(primitives[f]) for f in HEAP if primitives[f]},
        functions={f: [func_att[f], func_entry.get(f, 0)] for f in HEAP + ['memset', 'memcpy'] if func_att[f]}), indent=1) + '\n')
if A.annotate:
    out = Path(A.annotate); out.mkdir(parents=True, exist_ok=True)
    for f in HEAP + ['memset']:
        if not func_att[f]: continue
        a = symaddr[f]
        size = next(s[1] for s in syms if s[2] == f and s[0] == a)
        dis = subprocess.run([A.objdump, '-d', '-l', '--mattr=+m,+a', '--no-show-raw-insn', f'--start-address={a:#x}',
                              f'--stop-address={a + size:#x}', exe], capture_output=True, text=True, check=True).stdout
        lines = []
        for l in dis.splitlines():
            m = re.match(r'\s*([0-9a-f]+):\s*(.*)$', l)
            if m:
                pc = int(m[1], 16)
                asm = m[2]
                word = words.get(pc, 0)
                if asm == '<unknown>':
                    if word & 127 == 0x5b and (word >> 12) & 7 == 1 and word >> 25 in (0x51, 0x52):
                        asm = ('CDERIVE' if word >> 25 == 0x51 else 'CREVOKE') + f' [encoding {word:08x}]'
                    else:
                        asm = f'<encoding {word:08x}>'
                lines.append(f'{per_insn.get(pc, 0):12d}  {m[1]}: {asm}')
            elif l.startswith(('/', ';')) or l.endswith('):') or ':' in l:
                lines.append(f'{"":12s}  {l}')
        (out / f'{f}.txt').write_text(f'{f}: {func_att[f]} attempts, {func_entry.get(f,0)} entries\n' + '\n'.join(lines) + '\n')
