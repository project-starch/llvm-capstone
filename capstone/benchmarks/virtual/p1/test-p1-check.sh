#!/usr/bin/env bash
# test-p1-check.sh: the kernel-splice repair of p1-check.py and parse-board.py (unsplice_printk), on a synthetic 2-process
# transcript in the P1 run-line format. Exit 0 when every case reads as stated, 1 otherwise.
#   A  clean                                             p1-check PASS, 0 fragments; parse-board rc 0
#   B  A with 4 kernel lines spliced into user lines:   p1-check PASS, 4 fragments; parse-board rc 0, TSV identical to A's;
#      a BENCH-DONE cut mid-word, a BENCH line cut      every @@NS-DMESG quote line kept, and the line after them too
#      inside check=, two fragments back to back, plus
#      an @@NS-DMESG dmesg quote block (kept on purpose)
#   B0 positive control: B has fewer whole BENCH-DONE lines than A (the splice is really there)
#   C  B with step 0's BENCH-DONE deleted (the negative control): p1-check FAIL "step 0: no BENCH-DONE failed=0";
#      parse-board rc 1
main() {
  set -u
  local V; V=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
  TEST_DIR=$(mktemp -d); trap 'rm -rf "$TEST_DIR"' EXIT; local D=$TEST_DIR
  python3 - "$D" <<'EOF' || return 1
import sys
d = sys.argv[1]
open(f'{d}/steps.tsv', 'w').write('n\tprogram\tarm\tproc\tK\targs\tsrc_n\n0\tcrc32\tT1\t0\t0\t1 2\t0\n1\tcrc32\tT3\t0\t0\t1 2\t1\n')
def proc(n, arm, rc0, rc1):
    L = [f'@@PERF-BEGIN {n} crc32 {arm} 0 K0',
         f'@@NATIVE-CNT {n} begin runs=10 run_cycles={rc0} between_cycles=5 checks=1 check_cycles=2',
         'BENCH-INFO beebs/crc32 source=context-csr rdcycle_rdinstret=ok',
         'BENCH-INFO beebs/crc32 counters=present warmup=1 iterations=2']
    for it in ('w0', '0', '1'):
        L.append(f'BENCH beebs/crc32 it={it} cycles=1234567 instret=7654321 entries=0 ns=49382680 check=0xabcdef12')
    L += ['BENCH-DONE beebs/crc32 failed=0',
          f'@@NATIVE-CNT {n} end runs=12 run_cycles={rc1} between_cycles=9 checks=2 check_cycles=4',
          f'@@PERF-END {n} crc32 {arm} 0 K0 rc=0 stderr_lines=0']
    return L
head = ['@@NS-DMESG [    0.000000] Kernel command line: earlycon', '@@NS-DMESG [  112.004191] capstone_vm: release retires attempted=12',
        '@@NS-RESULT 70']
A = head + proc(0, 'T1', 1000, 5000) + proc(1, 'T3', 6000, 9000)
k = '[  143.107178] capstone_vm: release retires attempted=13 succeeded=13 pushed=13 first_id=1024\r\n'
k2 = '[  143.207178] capstone_vm: release retires attempted=9 succeeded=9 pushed=9 first_id=1024\r\n'
open(f'{d}/A.txt', 'w', newline='').write('\r\n'.join(A) + '\r\n')
B = '\r\n'.join(A) + '\r\n'
def cut(t, line, at, frag):
    i = t.index(line); assert t.count(line) >= 1
    return t[:i + at] + frag + t[i + at:]
B = cut(B, 'BENCH-DONE beebs/crc32 failed=0', 14, k)                    # step 0, mid-word: "BENCH-DONE bee[..]\nbs/crc32 ..."
j = B.index('@@PERF-BEGIN 1 ')
B = B[:j] + cut(B[j:], 'it=0 cycles=1234567', 70, k)                    # step 1, inside check=
j = B.index('@@PERF-BEGIN 1 ')
B = B[:j] + cut(B[j:], 'BENCH-DONE beebs/crc32 failed=0', 20, k + k2)  # step 1, two back to back
open(f'{d}/B.txt', 'w', newline='').write(B)
# C: delete step 0's (spliced) BENCH-DONE line together with its fragment
s0 = B.index('BENCH-DONE bee'); e0 = B.index('failed=0\r\n', s0) + len('failed=0\r\n')
open(f'{d}/C.txt', 'w', newline='').write(B[:s0] + B[e0:])
EOF
  local bad=0 out rc
  ok() { echo "ok   $1"; }; no() { echo "FAIL $1"; bad=1; }
  out=$(python3 "$V/p1/p1-check.py" "$D/A.txt" "$D/steps.tsv"); rc=$?
  [ $rc = 0 ] && grep -qx 'kernel fragments removed from mid-line: 0' <<<"$out" && grep -qx PASS <<<"$out" && ok "A p1-check PASS, 0 fragments" || no "A p1-check rc=$rc: $out"
  python3 "$V/parse-board.py" "$D/A.txt" --board t --boot t --out "$D/A.tsv" > "$D/A.perr" 2>&1; rc=$?
  [ $rc = 0 ] && ok "A parse-board rc 0" || no "A parse-board rc=$rc: $(cat "$D/A.perr")"
  local na nb
  na=$(python3 -c "import sys;print(sum(1 for l in open(sys.argv[1],newline='').read().split('\r\n') if l=='BENCH-DONE beebs/crc32 failed=0'))" "$D/A.txt")
  nb=$(python3 -c "import sys;print(sum(1 for l in open(sys.argv[1],newline='').read().split('\r\n') if l=='BENCH-DONE beebs/crc32 failed=0'))" "$D/B.txt")
  [ "$na" = 2 ] && [ "$nb" = 0 ] && ok "B0 whole BENCH-DONE lines: A $na, B $nb" || no "B0 whole BENCH-DONE lines: A $na, B $nb"
  out=$(python3 "$V/p1/p1-check.py" "$D/B.txt" "$D/steps.tsv"); rc=$?
  [ $rc = 0 ] && grep -qx 'kernel fragments removed from mid-line: 4' <<<"$out" && grep -qx PASS <<<"$out" && ok "B p1-check PASS, 4 fragments" || no "B p1-check rc=$rc: $out"
  python3 "$V/parse-board.py" "$D/B.txt" --board t --boot t --out "$D/B.tsv" > "$D/B.perr" 2>&1; rc=$?
  [ $rc = 0 ] && grep -q 'rejoined 4 guest line(s) split by a mid-line kernel console line' "$D/B.perr" && ok "B parse-board rc 0, 4 rejoined" || no "B parse-board rc=$rc: $(cat "$D/B.perr")"
  cmp -s "$D/A.tsv" "$D/B.tsv" && cmp -s "$D/A.tsv.ncnt.tsv" "$D/B.tsv.ncnt.tsv" && ok "B parse-board TSVs identical to A's" || no "B parse-board TSVs differ from A's"
  python3 - "$V/p1/p1-check.py" "$D/B.txt" <<'EOF' && ok "B @@NS-DMESG lines and @@NS-RESULT kept" || no "B @@NS-DMESG/@@NS-RESULT not kept"
import re, sys
s = open(sys.argv[1]).read()
g = {'re': re}; exec(s[s.index('KSPLICE ='):s.index('ap = argparse')], g)
t, n = g['unsplice_printk'](open(sys.argv[2], newline='').read().replace('\r', ''))
L = t.split('\n')
sys.exit(0 if n == 4 and L[:3] == ['@@NS-DMESG [    0.000000] Kernel command line: earlycon',
                                   '@@NS-DMESG [  112.004191] capstone_vm: release retires attempted=12', '@@NS-RESULT 70'] else 1)
EOF
  out=$(python3 "$V/p1/p1-check.py" "$D/C.txt" "$D/steps.tsv"); rc=$?
  [ $rc = 1 ] && grep -qx '  FAIL step 0: no BENCH-DONE failed=0' <<<"$out" && ok "C p1-check FAIL: step 0: no BENCH-DONE failed=0" || no "C p1-check rc=$rc: $out"
  python3 "$V/parse-board.py" "$D/C.txt" --board t --boot t --out "$D/C.tsv" > "$D/C.perr" 2>&1; rc=$?
  [ $rc = 1 ] && ok "C parse-board rc 1" || no "C parse-board rc=$rc: $(cat "$D/C.perr")"
  [ $bad = 0 ] && echo "test-p1-check: ALL PASS" || echo "test-p1-check: FAILED"
  return $bad
}
main "$@"; exit
