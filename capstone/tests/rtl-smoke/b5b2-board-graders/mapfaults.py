"""mapfaults.py UART_TXT ARM : map every 'sublet: fault' line of a b5b2-spectest capture to the image's instruction.
The loader maps the image's vaddr 0x10000 at 'entry' (its 'code=' range equals the R E LOAD segment), so vaddr = pc -
entry + 0x10000. The image comes from board-stages/ARM/<bench>/stage/bin/native/<comm-prefix>*.native; its sha16 is
printed. A fault whose image cannot be found unambiguously is reported as such, never guessed."""
import glob, hashlib, os, re, subprocess, sys
if len(sys.argv) != 3: sys.exit(__doc__)
OD = os.environ.get('CAPSTONE_LLVM_OBJDUMP') or f"{os.environ['CAPSTONE_REPO_ROOT']}/llvm/cmake-build-debug/bin/llvm-objdump"  # (a literal in the run copy)
ROOT = f"{os.environ['CAPSTONE_HDD']}/c4/b5b2spec/board-stages/{sys.argv[2]}"  # (a literal in the run copy)
t = open(sys.argv[1], errors='replace').read().replace('\r', '')
seen = set()
for m in re.finditer(r'sublet: fault cause=(\d+) pc=(0x[0-9a-f]+) address=(0x[0-9a-f]+) pid=(\d+) comm=(\S+) ra=(0x[0-9a-f]+) fp=(\S+) entry=(0x[0-9a-f]+) code=(0x[0-9a-f]+)-(0x[0-9a-f]+)', t):
    key = m.group(0)
    if key in seen: continue
    seen.add(key)
    cause, pc, addr, pid, comm, ra, fp, entry, c0, c1 = m.groups()
    pc, ra, entry, c0, c1 = (int(x, 16) for x in (pc, ra, entry, c0, c1))
    alias = {'.perl-native': 'perlbench_r_base.native'}   # run.sh copies the image under another name (perlbench's test.pl subprocesses)
    want = alias.get(comm, comm.rstrip('.'))
    cands = [f for f in glob.glob(f'{ROOT}/*/stage/bin/native/*.native') if os.path.basename(f).startswith(want)]
    print(f'fault cause={cause} pid={pid} comm={comm} pc={pc:#x} ra={ra:#x} address={addr} code {c0:#x}-{c1:#x}')
    if len(cands) != 1:
        print(f'  image: {len(cands)} candidates {[os.path.basename(c) for c in cands]}: NOT MAPPED'); continue
    f = cands[0]; sha = hashlib.sha256(open(f, 'rb').read()).hexdigest()[:16]
    v, vr = pc - entry + 0x10000, ra - entry + 0x10000
    print(f'  image {os.path.relpath(f, ROOT)} {sha}: pc vaddr {v:#x}, ra vaddr {vr:#x}' + ('' if c0 <= pc < c1 else ' (pc OUTSIDE the code range)'))
    d = subprocess.run([OD, '-d', '--triple=capstone64-unknown-elf', f'--start-address={v - 20:#x}', f'--stop-address={v + 6:#x}', f],
                       capture_output=True, text=True).stdout
    for l in d.splitlines():
        if re.match(r'^\s*[0-9a-f]+:', l): print(('  > ' if int(l.split(':')[0], 16) == v else '    ') + l.strip()[:90])
if not seen: print('no fault lines')
