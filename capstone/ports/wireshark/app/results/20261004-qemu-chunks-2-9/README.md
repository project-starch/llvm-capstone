# The `chunks` arm's last eight registered cells, measured (2026-10-04)

**Question.** `chunks` fixtures **2-9** were registered in `host/safety-expect.txt` on **2026-09-23** and
had never appeared in any committed bundle — the arm's only measured cells were 1, 10-13 and 14-15. Eight
registered rows with no reading.

**Verdict: 9 of 9 cells AS PREDICTED, in one boot, runner exit 0.** Fixture 1 is included as the in-boot
setup control.

| fixture | registered | measured | attribution |
|---|---|---|---|
| **1** `heap_len` (control) | RETURN `100001` + LEN `64` | **RETURN `100001`** | `len=64` |
| **2** `heap_neighbour` | FAULT `oob` | **FAULT `oob`** | `touch=5`, `len=64` |
| **3** `heap_one_past` | FAULT `oob` | **FAULT `oob`** | `touch=3`, `len=64` |
| **4** `uaf_noreuse` | FAULT `temporal` | **FAULT `temporal`** | `touch=3` |
| **5** `uaf_reuse` | FAULT `temporal` | **FAULT `temporal`** | `touch=5` |
| **6** `stale_free` | FAULT `temporal` | **FAULT `temporal`** | `touch=5` |
| **7** `global_oob` | FAULT `oob` | **FAULT `oob`** | `touch=4` |
| **8** `stack_oob` | FAULT `oob` | **FAULT `oob`** | `touch=4` |
| **9** `global_merged` | RETURN `90005b` | **RETURN `90005b`** | `touch=5` |

Every faulting cell exited **139** (`128+11`, SIGSEGV) with a fault record, and each fault is attributed
*after* the fixture's own touch line — that is what the port classifier checks, not merely that a fault
happened.

## What these cells do and do not add

**Read them as a consistency check, not as new nested-allocator evidence.** Fixtures 2-9 are
**plain-heap** synthetic probes — neighbour write, one-past-the-end, use-after-free with and without
reuse, double free, global and stack overflow, merged globals. They exercise the *heap* under an arm
whose distinguishing feature is the **wmem BLOCK chunk port**. So the finding is that `chunks` behaves on
plain heap exactly as `sublet` does (`host/safety-expect.txt:102` predicted precisely that), which is
worth having measured rather than assumed, and is **not** a detection the paper's table counts — that
table counts upstream corpus cases.

The arm's nested-allocator evidence remains fixtures **11, 12, 13**, measured 2026-10-03 in
[`../2026-10-03-qemu-wmem-chunks-arm/`](../2026-10-03-qemu-wmem-chunks-arm/README.md), where `chunks`
faults and every heap arm returns.

## How it ran, and why that is new

`capstone-vm` is this port's only runner and it needs ssh; the rootfs has no dropbear, and no riscv64
dropbear binary exists on this host to hand it via `--ssh-server`. This bundle used
**`ports/common/application/run-fixtures-9p.py`**, committed with it: it stages the images and the
pinned `capstone-exec`/`capstone-job`/`capstone.ko` over 9p, issues **the same guest command
`capstone-vm` issues** (`capstone-job <result> -- capstone-exec -- <image>`), and judges with
`check-safety.py`'s own `classify()`/`matches()`, imported rather than reimplemented.

**One boot for nine cells.** A domain fault SIGSEGVs the launcher, not the guest, so a faulting fixture
does not end the batch. The runner still orders RETURN-expected cells first, so a lost boot costs least.

**The judge was negative-tested before any of this was believed.** Run against a deliberately wrong row
(fixture 1 forced to `FAULT any`), it reported `fx1: DIFFERS: RETURN 100001` and exited 1 — so it both
reads the true outcome and can fail. Without that, nine passes would be nine untested zeros.

**Two infrastructure failures are worth recording**, because both were classified as infrastructure
rather than as results:

1. The first attempt died before booting: the runner resolved the repo root one level short and asked
   python for `capstone/capstone/tests/...`. Exit 75, `NO SERIAL CAPTURE`.
2. The second was **cut mid-stream at four cells**. `run-domain-smoke.py` defaults the guest-command
   timeout to `30 × --timeout-multiplier` — a *setup*-sized number — and nine 47 MB fixtures blew
   through 240 s. Exit 75, `BOOT PRODUCED NO RESULT`. The runner now sizes the **workload** budget from
   the batch (`--seconds-per-fixture`, default 180), which is what `run-domain-smoke.py:450-458` itself
   says the split is for. Neither failure produced a verdict, which is the point.

## Provenance

- **Platform:** the pinned process-ABI monitor. Application images need the monitor's `PROCESS_*` ecalls,
  and the installed buildroot `fw_jump.elf` has none — 0 occurrences of `context_step` against 41 in the
  pinned one. `SHA256SUMS` records the monitor actually used, plus every image, the launcher, the module
  and the emulator.
- **Images:** built 2026-10-03, the same build that produced the measured 10-13, so these cells and those
  share one image set.
- **Fidelity:** the runner exports `CAPSTONE_GP_NONLIN=1` and `CAPSTONE_REV_NODES=65536`, which
  `capstone-vm` forces (`capstone_vm/cli.py:326-327`) and the earlier ad-hoc scripts did not.
- QEMU only. **N = 1 per cell.**

Files: `result-lines.txt`, `SHA256SUMS`.
