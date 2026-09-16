# F1 / C-32 design A: cell ⑥ at pure -O2 — TRANSFER BRANCH, NOT FOR MERGE

Compiler lane, 2026-09-16, for the board lane on apollo, whose host glibc cannot compile the
SQLite domain TU (hosted-header wall; not a design-A problem). Cite by hash, never by filename.

## What is in this branch

    cell6-O2-c32fixed.dom   sha256/16  113221f93b0ac994

Built by the committed flow — `run-speedtest1-measure.sh` with `SPEEDTEST1_SUBLET=1`,
`SQLITE_OPT_LEVEL=-O2`, `SPEEDTEST1_STACK=385024`, `SQLITE_OPTNONE_FUNCS` unset — from dev
carrying design A (`46c53b7b6ae2` + `19bc05cf21b1`, on dev at `3e105357b00c`). Toolchain rebuilt
from that tree first.

## THE LOGS ARE REDACTED — read this before using them as evidence

**The four log files ARE here, and they are NOT byte-original.** They are the full transcripts
with two identifier substitutions applied, per the lead's ruling of 2026-09-16. Nothing else was
touched: no cropping, no trimming, no removal of uninteresting lines. Line counts are identical to
the originals (1218 / 464 / … verified).

**The transformation, described so it can be checked rather than trusted.** Two literal byte
substitutions over the whole file, nothing else:

    <operator-account>@<build-host>   ->   <user>@<host>
    /home/<operator-account>          ->   /home/<user>

where `<operator-account>` is the Unix account the runs were made under and `<build-host>` the
machine name in the guest kernel's build banner.

**The removed strings are deliberately NOT reproduced here.** Writing the account name into a
committed file is precisely what this redaction exists to prevent, and `precommit-scan.sh` blocks
a README that quotes it — which it did, on the first attempt at this paragraph. A disclosure
cannot contain the token it is disclosing the removal of. What makes the change verifiable instead
is the pair of hashes below: anyone holding an original can diff it against the committed file and
see every byte that moved, without this document having to name it.

**Why:** the raw captures carry the operator's name in content — home paths through the host
captures, and the guest kernel banner's `Linux version … (<user>@<host>)` build string.
`precommit-scan.sh` blocks on them, correctly; CLAUDE.md's "commit result lines, not the capture
they came from" names this exact contamination. Redaction is the route that keeps a real transcript
in the gate's hands rather than an extract assembled to satisfy it.

**Pre-redaction sha256, so a reader holding an original can verify exactly what changed:**

    2391bc667cd96f313f8de1c106827d141e7561a9413e9a9d34b481b359aa4f5c  default-arena.serial.log
    9c7a4a6f8c20a9e9062971d875ebb0d390838e904e057d8b2cb7e95596940c45  default-arena.stdout.log
    4431d070674695e4956b34f95235a40a95d3ed38189734f179350836499bb88e  2mib-arena.serial.log
    eef14a1c2a3aa312d7544eda856e6ec6acc984f441004a15b2336bdb92a6a332  2mib-arena.stdout.log

The originals are retained unmodified outside the repo. `SHA256SUMS` in this directory covers the
files AS COMMITTED, i.e. post-redaction, so the bundle's own hash chain is intact.

**The redaction cannot disturb what the driver reads:** none of the four gated lines contains an
identifier. Confirmed by re-running all four gate checks plus the negative control AFTER the
substitution — `:41 :42 :59 :60` all rc=0, and the default CFG grep against the serial log rc=1.

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
