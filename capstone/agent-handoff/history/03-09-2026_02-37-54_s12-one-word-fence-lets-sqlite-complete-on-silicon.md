# S-12: a one-word fence removes the fault and SQLite completes on silicon

## The result

| build | draws | outcome |
|---|---|---|
| trap-on baseline, sha `b316decabbc9e04d` | 3 / 3 | **TRAPPED** — mcause 25, VA `0x104874`, `slt did not run` |
| + one word: `fence rw, rw`, sha `0e3a2264c0bba5f0` | **0 / 4 trapped** | **`SLT-SUMMARY records=2 stmt_pass=1 stmt_fail=0 query_pass=0 query_fail=1 … completed=1`** |

The two images differ in **one 4-byte word** at file offset `0xf5860`, verified by whole-file
comparison. Every fence draw carried the correct per-run sha256 and the known-good control returned
in 6 s. The SLT summary is byte-identical to what the same image produces under QEMU emulation, so
the query did not merely start — it ran to the same answer.

One-sided Fisher on 3/3 against 0/4 is p = 0.029.

## The patch

At `0x104860`, three instructions ahead of the capability store:

    before                         after
    104860  sw   a4, 0x0(a5)       104860  fence rw, rw
    104864  cincoffsetimm a5, s0, -0x120
    104868  movc a4, zero
    10486c  stc  a4, 0x0(a5)
    104870  ldc  a4, 0x0(a0)
    104874  cincoffsetimm a4, a4, 0xb0     <- the fault site

The replaced instruction is the zero-init store the QEMU functional gate had already shown to be
behaviour-preserving — its slot is re-stored some 700 instructions later — so the fence costs the
program nothing, and QEMU confirms identical output.

## What it establishes, and what it does not

**Establishes:** S-12 is removed, on the real workload, by draining the store buffer immediately
ahead of the faulting window. Combined with the localisation — the fault requires the store's
source, the load's destination and the faulting instruction's operand to be the same register — and
with the RTL structure (STC decodes `rd := rs2` so the store is a scoreboard producer;
`create_cnull()` is cursor 0 AND `cap_type` 0, which is `mcause 25` with `tval 0` exactly; a
store-buffer-stalled STC never retracts `we_gpr`), the account is coherent end to end.

**Does NOT establish:** that the store-buffer stall specifically is the trigger. **A fence is not a
surgical instrument.** It orders all prior memory operations and delays everything after it, so it
perturbs issue timing broadly, not just buffer occupancy. This result is *consistent with* the
stall hypothesis and does not isolate it. Saying otherwise would repeat the mistake this folder has
already retracted twice — treating a correlate that fits as a cause that is proven.

It is also worth stating plainly that every reconstruction remains clean: ~53,000 executions of the
exact shape in a real capability domain on silicon, across seven eliminated variables (shape alone,
register identity, ABI class, producer distance, stored value, domain context, store pressure, slot
provenance), with zero faults. Whatever the real workload supplies is still not reproduced outside
it.

## Why this matters beyond the mechanism

**It is a validated workaround.** S-12 was the blocker on running the SQLite logic-test corpus on
silicon, and a one-word, behaviour-preserving patch now lets the query complete, 4/4. That is
directly actionable independently of whether the mechanism is ever confirmed.

The obvious next question is whether the fence generalises — whether a compiler-inserted barrier
ahead of `stc`/`ldc` pairs that share a register clears the corpus, at what cost, and whether a
narrower barrier than `fence rw, rw` suffices. Each of those is now cheap to measure, because the
fault returns instead of taking the board.
