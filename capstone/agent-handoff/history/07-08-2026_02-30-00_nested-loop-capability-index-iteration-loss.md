# A four-way conjunction loses loop iterations on silicon -- characterised, NOT root-caused

Date: 2026-08-07
Status: reproducible board-vs-QEMU divergence with a fully bisected TRIGGER and NO mechanism.
Handed over as a characterisation plus seven controls, not as a diagnosed defect.

## The observation

A domain loop executes FEWER iterations than it should, deterministically, returning a WRONG
NUMBER rather than hanging. QEMU runs the identical binary correctly every time.

    L31   nested 64x9, inner body reads a capability field indexed by the inner counter
          board 567 on SIX separate runs      QEMU 576

## The trigger, bisected to a conjunction of FOUR conditions

Seven controls, each removing exactly ONE condition from L31, each exactly correct on the
board and matching its measured QEMU oracle. Every probe reuses existing globals, so all share
`.capstone_cap_init@0x160570` and `.capstone_gp_table@0x1a0ea0` and none is an
image-perturbation draw.

| level | what it removes | board | QEMU |
|---|---|---|---|
| L28 | the capability access entirely (`qcount++` only) | 576 | 576 |
| L30 | the capability access, keeps a `jalr` | 576 | 576 |
| L36 | the nest (single loop, constant index) | 576 | 576 |
| L35 | the counter index (nest, constant index) | 576 | 576 |
| L38 | the capability (integer field at counter index) | 576 | 576 |
| L37 | the counter (capability at a dynamic NON-counter index) | 576 | 576 |
| L39 | the nest, keeping a counter-derived index (`qk%9`, monotone) | 576 | 576 |
| L40 | the RESET (nest, index is the OUTER counter, never resets) | 576 | 576 |
| L41 | the nest, keeping an explicitly RESETTING index | 576 | 576 |
| **L31** | **nothing -- all four present** | **567 x6** | 576 |

So the fault requires ALL of:

1. a NESTED loop  (L36, L39, L41 remove it -> correct)
2. a CAPABILITY access in the inner body  (L28, L30, L38 remove it -> correct)
3. an index that is the INNER loop counter  (L37, L40 remove it -> correct)
4. that index RESETS to 0 each outer pass  (L40 removes it -> correct)

Conditions 1 and 4 were confounded until L40/L41 separated them: in every failing level the
capability index is also the only frame word written to 0 in an outer body and counted past
its own bound. L39 looked like it tested a counter-derived index without a nest, but `qk%9`
comes from a MONOTONE counter, so it did not.

## What is NOT established

**The mechanism.** Nothing read so far in `capstone-ariane` is sensitive to loop NESTING, and
every per-access mechanism proposed has been refuted by a level that measurably returned the
correct answer:

* store-to-load forwarding / wbuffer eviction -- `wt_dcache_wbuffer.sv:331` selects its path on
  `(|wr_data_be_o) && (|hit_oh)`, neither of which depends on `wr_ack_i`;
* an issue-stage clobber window -- the trigger is byte-identical in L31 (567) and in
  L35/L36/L37/L39 (576), and `NrCommitPorts=1` makes `stall_waw` inert;
* a spill slot aliasing the counter -- L32 miscomputes with NO 16-byte stack traffic in its
  nest, and L30 does a stack `stc`/`ldc` round trip 576 times and is exactly correct;
* a clearing-type `ldc` -- `delin(sp)` plus SPLIT preserving cap_type makes every cap-table
  entry NONLIN, which `load_unit.sv:448` excludes; and such an `ldc` would self-destruct on
  iteration 2 of a 576-iteration loop.

**Which 9 iterations are lost.** L31 reports only the accumulator, and 576-9 is equally "one
lost inner pass", "nine lost read-modify-writes", or "63 outer passes that each lost nothing".
UNRESOLVED.

## This is NOT the SQLite blocker

`sqlite3InsertBuiltinFuncs` is a SINGLE loop reading a capability field indexed by its own
counter. L39 was built as exactly that shape and returns 576. Condition 1 fails, so this fault
is inert there. The blocker (qr15 returns / qr16 wedges, images TWO BYTES apart) remains a
separate, unexplained divergence.

## What would close it

Board bisection has gone as far as it can: the trigger is pinned and every remaining question
is about internal state. What is needed now is a SIMULATION or waveform of L31's inner loop --
specifically, what differs in the fetch/issue/replay path between an outer pass that runs its
inner body and one that does not. S01's README asks for the same thing for its own hang, and
that request is still open.

Reproduce with:

    OUT_DIR=/tmp/capstone/sq-l31d4 DOMAIN_EXTRA_DEFS="-DCAPSTONE_SQLITE_QUICKRET=31 -DQR_DRAW=4" \
      bash capstone/benchmarks/sqlite/build-sqlite-silicon.sh
    # stage as q31.dom, k800 FIRST in the boot; expect obs 0x9E310237 (567) against QEMU 0x9E310240
