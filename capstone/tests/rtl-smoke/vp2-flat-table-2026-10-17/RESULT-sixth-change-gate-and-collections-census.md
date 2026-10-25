# Result — the sixth change reads identically two-sided, and nothing in Tier 3 collects

Sibling records in this folder: `RESULT-collection-row.md` (the flat table's collection row, whose
sparse-churn figures are this census's positive control) and `PREREGISTRATION-boot4-one-slot.md`
(the seven FALSE checks Tier 1 reads below). The change is lane commit `e5775346cc8e`
(`module-resume-rd-x0`): the entry/resume CSRUNV is issued with rd = x0 and every event kind is read
from the frame's event record at +16, with ~0 as the refusal sentinel (S-27).

## Builds

| kit | parent source (control) | e5775346cc8e |
|---|---|---|
| board (build-fpga linux-6.4.14, rv64g, flat) | `3005ed52e860c7af`, byte-for-byte the deployed module | `d71f207a3af5546b`, rebuilt twice, byte-identical |
| QEMU (caplifive-buildroot linux-custom, rv64g, flat) | `.text` equal to the deployed `5bfc38f808e1effd` | `28882779c95b94ad`; a rebuild differs only in the embedded build path (`.rodata.str1.8`) and the build-id |

`d71f207a3af5546b`: vermagic `6.4.14 SMP riscv`; 0 compressed instructions of 4,762 (the counter
reads 2,051 on a default-march build); the kit's `qemu-only-csr-census.py` 0 and `vp2-silicon-gate.py`
(trusted_S) 0 refused / 0 candidates / 0 attributed / 4 conditional / 0 unimp, the same profile as
`3005ed52` (4,715 instructions).

## The gate: capstone-qemu b290f5656600 (binary 7a4f3a64776f0c88), svadu=false

**Tier 1** — `run.py` with `CAPSTONE_QEMU_CPU_EXTRA=svadu=false`, the v9 kit's contract.dom /
sqlite3.dom / mruby, the flat adapter's launcher, `--module-args ""`, one arm per module:

| | old `5bfc38f808e1effd` | new `28882779c95b94ad` |
|---|---|---|
| checks | 34/41 | 34/41 |
| FALSE | heap-threads, heap_threads, shared_heap_pages_released, threads, threads_output, vm, vm_contract | the same seven |
| VM_EXIT, in order | normal:0 threads:1 vm:1 nested:0 nested-stale:139 heap-threads:1 sparse-churn:0 stale:139 bounds:139 retired:139 vm-ro-write:139 vm-none-read:139 vm-guard:139 vm-padding:139 sparse-stale:139 preempt:143 parallel_one:0 parallel_two:0 sqlite_write:0 sqlite_read:0 mruby:0 cleanup:0 | identical |
| cause 29 / "wrote no event record" | 0 / 0 | 0 / 0 |

The seven are the one-slot pre-registration's seven. **Where the kernel-side counts come from:**
`run.py`'s stdout carries no kernel lines at all, not even the module's load banner, so a count
taken there is void. It was first reported from that file and corrected. The counts above are from
each arm's guest console (`virtual-app.*/serial.log`), where the module's pr_emerg load banner
appears in every arm as the route's control; the refusal line is pr_emerg too. An Oops prints below
KERN_EMERG and so is not visible on this level-1 console; no-crash rests on every step running
through `cleanup:0` with identical exit codes. run.py's end-of-run ring-buffer read
(`dmesg | grep capstone_vm:`) printed nothing in any arm, not even the banner its first read
showed, so it was not used. The 11 `CAPSTONE_VM_STATS` lines agree field by
field except `steps`, `rounds` and the byte counts on the last five (parallel_one/two, sqlite, mruby),
which vary run to run on this invocation whatever the module: twelve earlier records of it give mruby
`bytes_out` in {16898, 16905, 17161, 21122, 21129}, `rounds` 109-140, `steps` 1674-1792, and the new
arm's values occur in two of them. Every escape kind now comes from +16, so the pass also shows QEMU
writes the record on every kind the run exercises.

**Tier 3** — the 15-arm nb chain (memcached, FFmpeg, Wireshark, the prebuilt pool and allocator
bundles, postgres c-repros and mmgr) plus memcached plain-temporal, each run on the QEMU prediction
kit and on a copy of it that differs only in the module. `compare-gate.py` compares, per case, the
verdict, reason, cause, symbol, offset, attribution, reached/completed, image sha256 and every
control's (name, observed, cause), and per arm the multiset of serial `domain fault` signatures
(cause, zero address, image sha256). Its controls: two different arms under one name DIFFER, one
edited cause DIFFERs, an empty arm is NO-DATA (exit 2).

**16 of 16 arms identical**, including pb13's VOID (20 faults, the same cause-29 signature), 0
refusal lines on either side; the old side also equals the morning's chain record on all 16, so the
tools did not drift between the two.

**Where QEMU writes +16** (`target/riscv/capstone_supervisor.c` at b290f5656600): a virtual entry sets
`call->event = frame + 16` (:963), and only `event_write` writes it — :456 (escape, every kind), :512
(domain return, kind 0), :696 (terminal resource fault, kind 2). The `base + 16` store at :396 is the
classic CSSUPERVISOR path: `capstone_supervisor_call` needs `s->armed`, which is set only at :354,
where `call->event` points at that path's own buffer (:352).

## The collections census (Q-27: vp2 refuses CSRUNV action 3)

**Tier 1/2**: `collections` is 0 on every measured step except sparse-churn's (9, reclaimed 584,902,
both modules). A step that ends at 139 prints no stats line, so 11 of the 22 steps are measured.
sparse-stale runs the same 200,000-allocation loop as sparse-churn (`runtime/virtual/contract.c`
197-233) and so needs collection by source; it is unmeasured.

**Tier 3**: the same 16 arms on the new module, with `capstone-vexec --stats` (CV_GLOBAL) printed
before the first step and after every step (scratch tooling: dev 07cfbd87b805 plus the per-step
readout of lane 4754cce159e2; `census-parse.py` reads it, and a step without its line is NO-DATA).
**0 collections on all 16**, all 332 steps measured, peak per-step node usage 31-102 against 65,536 —
far from the trigger, which needs fewer than 2 + 256 (`CAP_REV_TABLE_RESERVE`) records available.
The census readings equal the gate's new arm on all 16.

**Positive control, the same instrument**: contract.dom in one virtualvm batch on the new-module kit,
`ok argument` then `sparse-churn` (`CAPSTONE_VM_CONTRACT=environment CAPSTONE_VM_STATS=1`): the
per-step readout gives collections 0 then 9, equal to each step's own `CAPSTONE_VM_STATS`, reclaimed
584,902 and allocated 600,041 — `RESULT-collection-row.md`'s figures. The parser reads it COLLECTED.

**A control that could not fire, kept on record**: loading the module with `node_flat_records=256`
to force collection refuses every image grant (`CAPSTONE_VM_ADD_REFUSED ... errno=12`), because ADD
needs 2 + 256 records available, so nothing ran. Every corpus step is a fresh context using at most
about a hundred nodes, so no record count (a multiple of 256) can make one collect; the churn workload
is the control instead.

**The appvm bundles** (pymalloc pym0/pym1, SQLite engine sq0/sq1 as full sets — supersets of the
board's per-boot subsets — and Wireshark wmem wm0/wm1), each one session through the board lane's
`tier3-appvm-arm.sh` with only its tool root pointed at the readout tree; each collect equals the
board's image list by sha256, so the reconstructed arguments are checked, not assumed:

| bundle | steps measured | peak nodes per step | collections | reading (= its prediction tally) |
|---|---:|---:|---:|---|
| pym0 | 22 | 44 | 0 | MISSED 20 |
| pym1 | 22 | 46 | 0 | CAUGHT 20 |
| sq0 | 36 | 22 | 0 | CAUGHT 5, MISSED 26, NO-READING 2 |
| sq1 | 36 | 56 | 0 | NO-READING control-failed 33 |
| wm0 | 27 | 47 | 0 | MISSED 23 |
| wm1 | 27 | 55 | 0 | NO-READING control-failed 23 |

So on bf3db3175 the rows that need collection are sparse-churn (measured) and sparse-stale (by
source). **None of the 26 pilot bundles collects** (502 steps measured, peak 102 nodes per step).

## Tools in this folder

- `compare-gate.py A B [tags]` — the two-sided per-case comparison above; a refusal count is NO-DATA
  unless every guest console of the arm carries the module's load banner.
- `census-parse.py ROOT [tags]` — the per-step readout; a step without its line is NO-DATA.
- `kcount.py FILE PATTERN...` — counts module lines only when the load banner is in the same file
  (exit 2, VOID, otherwise). Negative-tested: the stopgap arm's console reads `collect skipped` = 2,
  its `run.py` stdout log is VOID, a missing file is VOID.
