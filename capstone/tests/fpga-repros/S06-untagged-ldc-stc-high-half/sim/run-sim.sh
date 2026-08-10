#!/usr/bin/env bash
# S-06 in RTL simulation: ~14 s, NO BOARD.
#
# Needs the `cva6-build-rv` docker image and capstone-ariane's tools/ (Verilator 5.008 + Spike).
# Everything else is here.
#
# What this shows that the board cannot: the RVFI trace prints the value every instruction wrote,
# so the two halves of the round-tripped word are read DIRECTLY, alongside a plain sd/ld control
# over the same buffer and the same capability in the same run.
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"
D=capstone/tests/fpga-repros/S06-untagged-ldc-stc-high-half
A=capstone/capstone-ariane

[ -d "$A/verif/sim" ] || { echo "capstone-ariane submodule not checked out"; exit 2; }
docker image inspect cva6-build-rv >/dev/null 2>&1 || { echo "docker image cva6-build-rv missing"; exit 2; }

cp -f "$D/sim/untagged-ldc-stc-128.S" "$A/verif/tests/custom/capstone/"
python3 - "$A" <<'PY'
import sys
p = sys.argv[1] + '/verif/tests/testlist_capstone.yaml'
s = open(p).read()
if 'untagged-ldc-stc-128' not in s:
    e = ('- test: untagged-ldc-stc-128\n  iterations: 1\n  path_var: TESTS_PATH\n'
         '  gcc_opts: "-static -mcmodel=medany -fvisibility=hidden -nostdlib -nostartfiles '
         '-I<path_var>/riscv-tests/isa/macros/scalar/ -I<path_var>/riscv-tests/env/p/"\n'
         '  asm_tests: <path_var>/custom/capstone/untagged-ldc-stc-128.S\n\n')
    open(p, 'w').write(s.replace('- test: stc\n', e + '- test: stc\n', 1))
    print('  testlist entry added')
PY

# ALWAYS delete stale artifacts first. A failed COMPILE leaves the previous run's log in place,
# and the parser will happily report it as this run's result.
rm -f "$A"/verif/sim/out_*/veri-testharness_sim/untagged-ldc-stc-128*

( cd "$A" && docker run --rm -v "$(pwd)":/workdir --user "$(id -u):$(id -g)" --entrypoint bash \
    -e HOME=/tmp -e RISCV=/toolchain -e CVA6_REPO_DIR=/workdir -e NUM_JOBS=16 \
    -e VERILATOR_THREADS=1 cva6-build-rv -c '
    set -e; cd /workdir
    source verif/regress/install-verilator.sh >/dev/null 2>&1
    source verif/regress/install-spike.sh     >/dev/null 2>&1
    source verif/sim/setup-env.sh             >/dev/null 2>&1
    cd verif/sim
    python3 cva6.py --testlist=../tests/testlist_capstone.yaml --test untagged-ldc-stc-128 \
      --iss_yaml cva6.yaml --target capstone_cv64a6_imafdc_sv39 --iss=veri-testharness \
      --issrun_opts=+debug_disable=1+UVM_VERBOSITY=UVM_NONE --issrun_opts=+time_out=2000000' ) >/dev/null 2>&1

python3 - "$A" <<'PY'
import sys, glob, re
A = sys.argv[1]
lg = sorted(glob.glob(A + '/verif/sim/out_*/veri-testharness_sim/untagged-ldc-stc-128*.log'))
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

LO, HI = 0x0123456789abcdef, 0xfedcba9876543210
names = {28: ('t3  ldc/stc round trip  LOW ', LO),
         29: ('t4  ldc/stc round trip  HIGH', HI),
         30: ('t5  CONTROL plain sd/ld LOW ', LO),
         31: ('t6  CONTROL plain sd/ld HIGH', HI)}
seen = {}
for l in open(lg[-1], errors='replace'):
    v = re.search(r'\(([0-9a-fx]+)\)\s+x\s*(\d+)\s+(0x[0-9a-f]+)', l)
    if v:
        r = int(v.group(2))
        if r in names:
            seen[r] = int(v.group(3), 16)

if not seen:
    print('NONE of t3-t6 were written -- the test did not run as written'); raise SystemExit(1)

print(f'  genuine completion in {n} cycles, no exceptions\n')
for r in sorted(names):
    label, want = names[r]
    got = seen.get(r)
    if got is None:
        print(f'    {label} -> NEVER WRITTEN'); continue
    mark = 'as expected' if got == want else ('*** LOST -- reads as zero' if got == 0 else '*** UNEXPECTED')
    print(f'    {label} -> 0x{got:016x}   {mark}')

ctl_ok  = seen.get(30) == LO and seen.get(31) == HI
defect  = seen.get(28) == LO and seen.get(29) == 0
print()
if not ctl_ok:
    print('  CONTROL FAILED -- the plain sd/ld path is wrong too, so this run says NOTHING')
    print('  about ldc/stc. Investigate the buffer/capability before reading the arms above.')
    raise SystemExit(1)
if defect:
    print('  REPRODUCED. The control survives bit-exact over the same buffer and the same')
    print('  capability, so the loss belongs to ldc/stc: the low half round-trips, the high')
    print('  half comes back as zero.')
    raise SystemExit(0)
print('  NOT reproduced on this RTL revision -- both halves survived. If that is expected,')
print('  the fix is present; otherwise read the arms above before concluding.')
raise SystemExit(1)
PY
