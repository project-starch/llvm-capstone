# SQLite returns rows on silicon — and the remaining fault is a spilled linear capability

## The result

`sqfixed.dom`, 2026-08-13, control green:

```
SQ: G/enter
SQ: H/return
row name=beta value=22
row name=gamma value=33
SQ: obs=4131427634
```

**Two rows came back out of the database.** That is the first time this workload has produced
query results on the FPGA. The prior best was a wild `Mem*` inside `CREATE TABLE`.

Not a pass: `row name=alpha value=11` is MISSING, and the run ends in a trap.

## What got it here

Three defects, in the order they were removed:

1. **S-06 in the C library copy** — the `stc` in `BEEBS_CHUNK_COPY` was unconditional. Fixed by
   asking LCC's (now total) type query and skipping the capability store for plain data.
2. **S-06 in the COMPILER's aggregate copies** — 283 bare `ldc`/`stc` granule stores across 41
   copy runs, including a 112-byte `Mem` copy inlined into `sqlite3VdbeExec`. Fixed by a new IR
   pass (`-capstone-guard-cap-granule-copies`).
3. **A miscompile in that pass** — see below. This one mattered most and was mine.

## The miscompile, because the shape will recur

`stc` WRITES cnull BACK INTO rs2 for the LINEAR/UNINIT/SEALED family — move semantics,
`capstone_dyn_unit.anvil:458-461`. **LLVM does not know**: `STC` is declared with an empty
`(outs)` list (`CapstoneInstrInfo.td:2402`). `MOVC` and `SCC` have the same gap.

The pass left the loaded capability live across the branch it inserts. At `-O0` RegAllocFast
spilled it *with `stc`*, and the spill cleared the register the guard then queried:

```
ldc a0, 0(a0)
stc a0, 80(sp)      # Folded Spill  <- clears a0 to cnull
lcc a0, a0, 1       #               <- reads 7
beq a0, a1, ...     #               <- "plain data", capability store SKIPPED
```

Destination untagged, source already cleared by the `ldc`. The fault then surfaced in a
function the pass never touched, which is why `sqlite3_strnicmp` and `memcpy` were
byte-identical between builds and why three of my hypotheses died before this one.

Fixed by keeping the capability's live range inside ONE block — `ldc`, query, `stc`, nothing
between — and letting only `lo`/`hi` cross the branch, since integer spills use `sd`/`ld` and
clear nothing. The capability store is now unconditional and the plain stores are the
conditional repair.

## The remaining fault, and the hypothesis it points at

`mcause 25` at `sqlite3DbMallocRawNN+0xf8` on `ldc a2, 0x0(a1)`, where `a1` was reloaded from
a stack slot — **another untagged capability out of a spill**.

That is the SAME shape as the miscompile, but this time in code the pass did not write. The
hypothesis that follows directly: because LLVM does not model `stc` clobbering rs2, **any**
spill of a LINEAR-family capability destroys it, anywhere in the program. The pass merely
created more opportunities for it.

NOT ESTABLISHED. Competing explanation: `s06spill` returns 65535 on this silicon — 16/16
capabilities spilled to stack slots survive reload — so spilling is *not* universally
destructive. Those were NONLIN. The discriminator is whether the capability at this site is
LINEAR, which needs a directed rung that spills a LINEAR capability specifically and reports
its type after reload. That rung does not exist yet and is the next step.

If confirmed, the fix is to model the rs2 clobber for `STC`/`MOVC`/`SCC` in TableGen so the
allocator stops spilling live capabilities with a destructive instruction — a real backend
change, not a workaround.

## Instruments built along the way

* `s06agg` / `s06aggcap` / `s06aggwide` — guarded-copy acceptance: plain data, capability
  preservation, and non-zero granules + multi-granule + stack destination. 15 / 15 / 255.
* `s06spill` — 16 capabilities spilled and reloaded; refuted the spill hypothesis in its
  general form.
* `trapctl` + `INTERP_DOMAIN_MTVEC` — an in-domain trap handler that packs mcause and
  `mepc - _start` into the result word, calibrated to the bit against a deliberate fault.
  Every fault above located itself because of it.
* `CAPSTONE_ARG_PROBE` — reports a function's incoming argument types and its caller.
