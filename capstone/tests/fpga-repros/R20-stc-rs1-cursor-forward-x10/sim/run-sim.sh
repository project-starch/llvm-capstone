#!/usr/bin/env bash
# R-20 in RTL simulation: ~14 s, NO BOARD and no SQLite.
#
# Needs the `cva6-build-rv` docker image and capstone-ariane's tools/ (Verilator 5.008 + Spike).
# Everything else is here.
#
# What it shows that the board cannot: the RVFI trace prints the value every instruction wrote,
# so the LOAD's own retired value and its CONSUMER's operand are both visible. The load retires
# with the correct 0 while the consumer reads the store's base address.
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"
D=capstone/tests/fpga-repros/R20-stc-rs1-cursor-forward-x10
A=capstone/capstone-ariane

[ -d "$A/verif/sim" ] || { echo "capstone-ariane submodule not checked out"; exit 2; }
docker image inspect cva6-build-rv >/dev/null 2>&1 || { echo "docker image cva6-build-rv missing"; exit 2; }

cp -f "$D/sim/r20-stc-ld-x10.S" "$A/verif/tests/custom/capstone/"
python3 - "$A" <<'PY'
import sys
p = sys.argv[1] + '/verif/tests/testlist_capstone.yaml'
s = open(p).read()
if 'r20-stc-ld-x10' not in s:
    e = ('- test: r20-stc-ld-x10\n  iterations: 1\n  path_var: TESTS_PATH\n'
         '  gcc_opts: "-static -mcmodel=medany -fvisibility=hidden -nostdlib -nostartfiles '
         '-I<path_var>/riscv-tests/isa/macros/scalar/ -I<path_var>/riscv-tests/env/p/"\n'
         '  asm_tests: <path_var>/custom/capstone/r20-stc-ld-x10.S\n\n')
    open(p, 'w').write(s.replace('- test: stc\n', e + '- test: stc\n', 1))
    print('  testlist entry added')
PY

# ALWAYS delete stale artifacts first. A failed COMPILE leaves the previous run's log in place,
# and the parser will happily report it as this run's result.
rm -f "$A"/verif/sim/out_*/veri-testharness_sim/r20-stc-ld-x10*

( cd "$A" && docker run --rm -v "$(pwd)":/workdir --user "$(id -u):$(id -g)" --entrypoint bash \
    -e HOME=/tmp -e RISCV=/toolchain -e CVA6_REPO_DIR=/workdir -e NUM_JOBS=16 \
    -e VERILATOR_THREADS=1 cva6-build-rv -c '
    set -e; cd /workdir
    source verif/regress/install-verilator.sh >/dev/null 2>&1
    source verif/regress/install-spike.sh     >/dev/null 2>&1
    source verif/sim/setup-env.sh             >/dev/null 2>&1
    cd verif/sim
    python3 cva6.py --testlist=../tests/testlist_capstone.yaml --test r20-stc-ld-x10 \
      --iss_yaml cva6.yaml --target capstone_cv64a6_imafdc_sv39 --iss=veri-testharness \
      --issrun_opts=+debug_disable=1+UVM_VERBOSITY=UVM_NONE --issrun_opts=+time_out=2000000' ) >/dev/null 2>&1

python3 - "$A" <<'PY'
import sys, glob, re
A = sys.argv[1]
lg = sorted(glob.glob(A + '/verif/sim/out_*/veri-testharness_sim/r20-stc-ld-x10*.log'))
if not lg:
    print('NO LOG PRODUCED -- the test did not build'); raise SystemExit(1)
iss = open(lg[-1] + '.iss', errors='replace').read()
m = re.search(r'after (\d+) cycles', iss)
n = int(m.group(1)) if m else -1
exc = [l.strip() for l in iss.split('\n') if 'Exception' in l]
# "*** SUCCESS *** after 2000013 cycles" is the TIMEOUT, not a pass.
if n >= 2000000:
    print(f'TIMEOUT/HANG ({n} cycles) -- SUCCESS at the timeout is NOT a pass'); raise SystemExit(1)
if exc:
    print('EXCEPTION:', exc[0]); raise SystemExit(1)

t = open(lg[-1], errors='replace').read().split('\n')
names = {28: 'A  a0, adjacent          ', 29: 'B  a3, adjacent  CONTROL ',
         30: 'C  a0, 1 nop             ', 20: 'D  a0, 2 nops            ',
         22: 'E  a0, 4 nops    CONTROL '}
seen, loads = {}, []
for i, l in enumerate(t):
    if not l.startswith('core   0:'):
        continue
    nxt = t[i + 1] if i + 1 < len(t) else ''
    v = re.search(r'\bx\s?(\d+)\s+(0x[0-9a-f]{16})', nxt)
    d = re.search(r'\((0x[0-9a-f]+)\)\s+(.*)$', l)
    if not (v and d):
        continue
    dis = d.group(2).strip()
    if dis.startswith('ld '):
        loads.append((dis, v.group(1), v.group(2)))
    if re.match(r'(mv|addi)\b', dis):
        r = int(v.group(1))
        if r in names and r not in seen:
            seen[r] = v.group(2)

print(f'  genuine completion in {n} cycles, no exceptions\n')
print('  every LOAD retires with the CORRECT value:')
for dis, r, val in loads:
    print(f'    {dis:20s} -> x{r} = {val}')
print("\n  but the CONSUMER after it reads (correct answer 0; store base 0x80003000):")
for r in sorted(names):
    if r in seen:
        print(f'    {names[r]} -> {seen[r]}   ' + ('*** DEFECT' if int(seen[r], 16) else 'correct'))

ok = (28 in seen and int(seen[28], 16) != 0) and (29 in seen and int(seen[29], 16) == 0)
print('\n  ' + ('REPRODUCED, and register-specific: arm A wrong, arm B correct in the same run'
                if ok else 'NOT the expected pattern -- read the arms above before concluding'))
raise SystemExit(0 if ok else 1)
PY
