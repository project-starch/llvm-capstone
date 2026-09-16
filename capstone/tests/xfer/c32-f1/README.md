# F1 / C-32 design A: cell ⑥ at pure -O2 — TRANSFER BRANCH, NOT FOR MERGE

Compiler lane, 2026-09-16, for the board lane on apollo, whose host glibc cannot compile the
SQLite domain TU (hosted-header wall; not a design-A problem). Cite by hash, never by filename.

## What is in this branch

    cell6-O2-c32fixed.dom   sha256/16  113221f93b0ac994

Built by the committed flow — `run-speedtest1-measure.sh` with `SPEEDTEST1_SUBLET=1`,
`SQLITE_OPT_LEVEL=-O2`, `SPEEDTEST1_STACK=385024`, `SQLITE_OPTNONE_FUNCS` unset — from dev
carrying design A (`46c53b7b6ae2` + `19bc05cf21b1`, on dev at `3e105357b00c`). Toolchain rebuilt
from that tree first.

## WHAT IS NOT IN THIS BRANCH, AND WHY — read before looking for the logs

**The four QEMU log files are deliberately absent.** They cannot be committed: they carry the
operator's name in file content, both as `/home/...` paths throughout the host captures and as the
guest kernel banner `Linux version 6.1.0 (<user>@<host>)` — a user@host build string. CLAUDE.md's
"Commit result lines, not the capture they came from" names exactly this ("a raw log is
contaminated by construction — kernel and driver banners carry account names and emails"), and
`precommit-scan.sh` blocks on them. That is the gate working as designed, not a misfire, and it is
not the kind of thing to route around.

Their sha256 are recorded here so they remain verifiable however they are moved:

    2391bc667cd96f313f8de1c106827d141e7561a9413e9a9d34b481b359aa4f5c  default-arena.serial.log
    9c7a4a6f8c20a9e9062971d875ebb0d390838e904e057d8b2cb7e95596940c45  default-arena.stdout.log
    4431d070674695e4956b34f95235a40a95d3ed38189734f179350836499bb88e  2mib-arena.serial.log
    eef14a1c2a3aa312d7544eda856e6ec6acc984f441004a15b2336bdb92a6a332  2mib-arena.stdout.log

`board-c6var.sh` needs exactly two lines out of them per record, and both lines are clean of names.
They are reproduced verbatim below so the numbers are on record even before the files move:

    default-arena.serial.log:  SPEEDTEST1-CYCLES 338496911 HIGHWATER n/a HEAP 911104 DROPPED 0 RC 0
    default-arena.stdout.log:  == Sublet: pool 1419584 bytes (arena, REV_BORROWED), tables 1750285 bytes
    2mib-arena.serial.log:     SPEEDTEST1-CYCLES 340817188 HIGHWATER n/a HEAP 1344064 DROPPED 0 RC 0
    2mib-arena.stdout.log:     == Sublet: pool 2097152 bytes (arena, REV_BORROWED), tables 1750285 bytes

**These four lines are quoted, not a substitute for the files.** An extract is something no tool
emitted natively, and the gate they feed exists precisely to stop a passing condition being met by
the wrong thing — so the receiving lane decides whether its gate may read an extract, or whether
the raw logs must move by a route outside git. That is not this lane's call to make for them.

## The two records

| | arena | tables | cycles | HEAP | sublet counters |
|---|---:|---:|---:|---:|---|
| default | 1,419,584 | 1,750,285 | **338,496,911** | 911104 | 5481/37874/32565/37874/5309 |
| 2 MiB | 2,097,152 | 1,750,285 | **340,817,188** | 1344064 | **5568/37966/32565/37966/5401** |

Both carry the oracle `Verification Hash: 112006 38bb59fd…3925d8518`, `DROPPED 0 RC 0`, and both
are records of the SAME image — verified byte-identical across the two build directories — so the
driver's single `C6_HASH` is correct. Arena and tables are host arguments; they never enter the
image.

**The 2 MiB counters match the boot's pre-registration exactly.** That was the prediction on
record: C-32 is a codegen change, not an allocator change, so the allocation trace had to be
identical, and a mismatch would have been a finding about design A rather than a number to report.

**The default-arena counters `5481/37874/…` are not a mismatch.** They are the documented
default-arena counts; the pre-registration is the 2 MiB set.

## Gate result, measured before sending

`board-c6var.sh`'s four checks, run against the files as produced:

    :41 default CYCLES/HEAP rc=0   :42 2 MiB CYCLES/HEAP rc=0
    :59 default CFG         rc=0   :60 2 MiB CFG         rc=0

Negative control, because a passing check is not evidence until shown able to fail: the default
CFG grep against the SERIAL log returns **rc=1**. The two-stream split is load-bearing — the cycles
line is emitted by the guest, the configuration line by the host script — and no single artefact
the flow produces carries both.

## Limits

Not merged, ever: this path exists to move bytes, and `capstone/tests/xfer/` is used rather than
`rtl-smoke/drivers/artifacts/` because `.gitignore:106` ignores `/capstone/tests/rtl-smoke/**/*.dom`
and would have dropped the image silently while `git add` reported success.

These records do NOT establish that C-32 is fixed on silicon: QEMU is the permissive side for this
class (Q-04) and cannot observe the defect. The +2 cycles against the pre-fix image at both arenas
is not a result and must not be reported as one, for the same reason. The boot does not exercise
design A's PHI residue — `renameResolveTrigger` is reached per trigger and `main` defines none.

The arena grant rounds up from the request (1,419,584 → 1,421,312 → 22,208 atoms → HEAP 911,104,
which these runs confirm). The granularity is NOT settled: 2 KiB and 4 KiB both reproduce it,
because 347×4096 = 694×2048, and the 2 MiB point is exact at every candidate. It matters only if
the new `SPEEDTEST1_SUBLET_ARENA` override is used at an arena not already a multiple of 4,096.
