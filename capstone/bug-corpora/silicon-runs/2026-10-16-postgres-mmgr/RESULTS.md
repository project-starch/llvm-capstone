# PostgreSQL's memory-context corpus on silicon: results (2026-10-16, caplifive_sublet_d93a8d66b)

Pre-registered in README.md (lane a13fe2ed2cbd, before the boot). One boot, 16:41-16:53:
- firmware 9dc96b2345000482, Image 89295c34884a3378;
- two arms, 30 runs, every one ended by itself (each arm's rung returned 0).

**21 of 22 rows as pre-registered, by outcome, cause and site,** on the pre-registered image (sha256). Every
negative control, the corpus control and the tool's four controls are as pre-registered on both arms. The one row
that differs is case 00 on the Sublet arm. Its outcome and cause are as predicted, and it faults at the case's other
declared site.

| case | pgm-silicon-pg-stock | pgm-silicon-pg-sublet | its negative control |
|---|---|---|---|
| 00 tuplestore double free | completes | CAUGHT 25 in GetMemoryChunkMethodID (declared; **pre-registered: SubletRelease**) | completes on both |
| 01 vacuum dead_items stale struct | completes | CAUGHT 25 at pg_probe | completes on both |
| 02 live_parts stale alias | completes | CAUGHT 25 at pg_probe | completes on both |
| 03 pgoutput entry context teardown | completes | CAUGHT 25 at pg_probe | completes on both |
| 04 reorderbuffer spec-insert record | completes | CAUGHT 25 at pg_probe | completes on both |
| corpus control 00 (pfree, then read through the alias) | completes | CAUGHT 25 at pg_probe | |

## Case 00: where the fault lands, and a correction to the pre-registration

The two readings are at different instructions, in the same image (4ac778ecd0a7ecdf):
- **QEMU:** cause 24 at `SubletRelease+0x24`, a `cincoffsetimm a0, a0, -0x20`. That is the step from the pointer
  pfree is handed back to its chunk header, done on a capability QEMU has untagged.
- **Board:** cause 25 at `GetMemoryChunkMethodID+0x2c`, an `ld a0, -0x8(a0)`. That is the first read of the chunk
  header through that pointer. SubletRelease calls GetMemoryChunkMethodID out of line at -O0.

This is ISSUES Q-11's mechanism. QEMU untags a revoked child, so the first instruction to check the tag faults, and
here that is pointer arithmetic. The RTL keeps the child tagged, so the arithmetic succeeds and the first access
refuses it, with 25.

**The correction.** README.md describes QEMU's 00 as "the second pfree's CREVOKE of a revoked child". That was written
without disassembling the fault pc, and it is wrong: QEMU faults before SubletRelease reaches its CREVOKE. With the
right instruction in hand, the board's site was predictable. The pre-registration's rule ("each 24 reading 25, same
site") holds only where QEMU's faulting instruction is itself an access. Where it is capability arithmetic on the
untagged child, the silicon fault moves to the next access through that child. The outcome and cause predictions
are unaffected.

**What it shows:**
- PostgreSQL 17's own memory-context managers (aset, mcxt, generation, slab, bump) run on silicon in a capstone-exec
  domain, on both arms.
- **Stock arm:** all five use-after-free and double-free defects run to completion unseen, as pre-registered.
- **Sublet arm:** patch 0003 catches all five. Every chunk is a CDERIVE child of its block, CREVOKEd at pfree and when
  its context dies. Each defect's stale access is refused with 25.
- Each case's negative control completes on both arms. So the Sublet arm's faults are the defects', and its revocation
  leaves the permitted accesses alone.

**Judged by** `tools/run-silicon-cases.py board --prereg` on the run-scoped transcript, then compared row by row with
the pre-registered outcome, cause and site, keyed by image hash.
