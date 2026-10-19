# The next RTL revision: four items in ONE bitstream, with every acceptance reading written before the build

*RTL lane, 2026-10-20, at the semantics owner's request and under the sufficiency rule ("a bitstream costs ~90
minutes plus a reflash; audit for SUFFICIENCY, not just correctness — and BATCH"). Nothing here is blocked on a
bitstream: both S-23 faces have module-side workarounds with field evidence (the srevroot-zero module for the
data face, the node floor for the fetch face, dev 794d7c5815f5). **The reflash of the resulting bitstream is the
project lead's own word, as always; this document does not assume it.***

## Why these four together

Each is a small RTL change with a question only silicon or a full-tree simulation can answer, and three of the
four share one subsystem (the revocation-node broadcast and the capability encoder), so splitting them across
builds would pay the ~90-minute cycle three times for changes that will interact. **Discovering the second one
after the first has been built costs another full cycle**, which is the rule this batch exists to respect.

The go/no-go question each item has to pass before the build: **what will it READ if the current best hypothesis
is true, and would that reading be uninformative?** Every item below answers it, and the answers are the
acceptance readings.

---

## (1) The namespace-aware invalidation broadcast — the real fix for both S-23 sites

**What is wrong.** The broadcast carries a BARE index under the table namespace
(`ex_stage.sv:1644-1645`, `revnode_invalidation_id_o = node_wr_req[29:0]`, with the generation hardwired `14'd0`
in the virtual record decode at `:1431-1433`), and the CPMP tracker compares **16 bits only**
(`pmp_data_if.sv:107`). So a VIRTUAL index clears a tracker holding a PHYSICAL one. That is the root cause of
both S-23 sites.

**Why it is not a comparator widening.** `pmp_data_if.sv:96-100` records why the compare is index-only: *"a
30-bit compare against a tracked (g, i) with g >= 1 would never fire and invalidation would silently stop
working at generation >= 1"*. The fix must therefore **carry the namespace in the broadcast** and compare it,
not widen the existing compare.

**Acceptance readings, pre-registered.**
1. `cap-fetchdeny` firing arm: **still 1 cause-1 fault** at the following instruction. A same-namespace
   collision is a legitimate revocation and must keep denying — a fix that silences this arm has broken
   invalidation, not fixed it.
2. `cap-fetchdeny` control arm: **still 0 faults**.
3. A NEW cross-namespace arm (the `va-fetchctx` shape with the two coverages split, see below): **0 faults**
   where today's mechanism predicts a refusal. This is the reading the fix exists for.
4. **The generation case, which is the one most likely to be forgotten**: an arm that revokes a node at a
   RECYCLED index, whose generation has been bumped to ≥ 1 by a physical pop (`capstone_rev_node.anvil:289-307`),
   must **still deny**. That is precisely what the index-only compare was chosen to preserve, so it is the
   regression this fix can most easily cause.

**A dependency to clear first.** The cross-namespace arm does not exist yet. `va-fetchctx` wedged because the
caller's CPMP grant covers the whole image, so it covers the context's code too and the collision denies the
CONTEXT's own fetches; and an M handler cannot repair a CPMP entry while a context is live (CCSRRW to
0x010..0x01F is a privilege violation while `sup_active_q`, `csr_regfile.sv:3018-3021`). The arm needs the
caller's S code under the tracked entry and the context's code under a different one. **That is simulation work
and must be finished before the build**, or item (1) ships with its decisive reading unavailable.

## (2) R-58: the caller's `srevroot` restored at the escape

**What is wrong.** The switcher's SAVE/RESTORE window covers register ids 0..66 and `srevroot` is id 67, so the
caller's value is never saved and never restored; after any entry or resume the caller runs in the CONTEXT's
lifetime namespace (`capstone_dom_switcher.anvil:170-173`, `:481`, `:315`, `:527-541`).

**The change**, scoped in R-58's row: id 67 in the caller's snapshot, which extends SAVE and RESTORE together
through the shared `walk_advance`, with the 8-byte stride placing it at snapshot offset +944. **The reference
model and any software definition of the snapshot size move with it**, or the two diverge silently — which this
project files as its own issue class.

**Acceptance readings, pre-registered.**
1. `va-srevroot`: the caller must read **its OWN value back** — `s3 = 0x80006000` and **`s4 = 0x80006000`**, where
   today `s4` reads `0x80007000`, the context's table — with the twenty baseline readings byte-identical.
2. `va-srevroot -DSREV_CSNODE`: the post-escape CSNODE must be **illegal at decode again (mcause 2)**, where
   today it succeeds with a node id, because the caller is back outside the table namespace. Two leading `0x2`
   instead of three.
3. On silicon, the DATA face at **0 with the OLD module** on the new bitstream — the reading that shows the fix
   removes the persistence rather than the workaround masking it.
4. **NOT an acceptance reading**: the surviving CSRETIRE+4 fetch site. R-58 is not required for it (inside the
   window `select_root` deliberately puts the context's table in `srevroot`), and an acceptance test that cannot
   fail for the right reason is worse than none.

## (3) R-59: the precheck accepts a frame word the id-67 write masks to zero

**What is wrong.** `runv_precheck_step` accepts any **non-zero** word at `dom_base + 8`
(`capstone_dom_switcher.anvil:381-387`) while the id-67 write masks the low 12 bits
(`csr_regfile.sv:2051`), so a word in **1..4095** passes the non-zero check and installs **0**: the context is
admitted and then runs outside the table namespace.

**The change**: make the two agree — either the precheck rejects a word with low bits set, or it tests the
masked value. The semantics owner's call which.

**Acceptance readings, pre-registered.** A frame with `+8 = 0x800`: the CSRUNV must be **REFUSED** (the refusal
path, ids 74 and 0, no entry), where today the context is admitted. Control: the same arm with a real
4 KiB-aligned non-zero base must **admit** exactly as it does now. Both arms in one load; the refusal is the
reading, and the control is what makes it a reading rather than a broken arm.

## (4) R-60: an out-of-window cursor must not alias

**What is wrong.** A load or store through a capability whose cursor lies outside its representable window is
bounds-checked against the bounds **decoded at that cursor** — the true bounds shifted by whole window spans — so
when the shifted decode contains the cursor the access **passes and touches memory a span away, with no trap**.
CINCOFFSET checks only the type: it neither clears the tag nor checks representability
(`capstone_flu_unit.anvil:29-60`, writeback `ex_stage.sv:1739`/`:1753`). Characterised on silicon, stores
measured; QEMU traps 28 in every arm.

**This lane's own earlier finding is the benign half of the same mechanism and must not be read as a reassurance.**
`cap-cursor-window` (capstone-ariane `3c2cdf3f2`) measured that the compressed word SURVIVES an excursion —
`compress_bounds` does not take the cursor — so the bounds READBACK self-heals when the cursor returns. It read
base and top with LCC and **never dereferenced**, which is exactly the case R-60 measures. Self-healing bounds
and an aliasing access are the same cursor-relative decode seen from two sides.

**The change**, as ruled in `isa.md` (`6a7038fb5fe3`): CINCOFFSET and SCC **clear the tag, or refuse**, when the
cursor leaves the representable window, so an alias-load or alias-store traps or reads untagged.

**Acceptance readings, pre-registered — and the strongest of the batch, because an arm already records the
current values.** `cap-cursor-window`'s W0 ladder today reads base and top EXACT at `+0` and `+0x800`, WRONG at
`+0x4000` and `+0x10004000`, and EXACT AGAIN all the way back. After the fix:
1. at `+0x4000` and beyond the capability must read **type 7 (NOT_CAP)** from LCC zimm 1, or the CINCOFFSET must
   refuse — the tag is gone;
2. **the self-heal must be GONE**: the return to `+0` must NOT restore the bounds, because there is no longer a
   tagged capability to restore them to. That is a reading no other change in this batch can produce;
3. W1's allocator transit (`neg`/`cincoffset` to cursor 0 and back) must then **fail where it passes today** —
   its SHRINK currently passes and must now refuse or operate on an untagged capability. **This is a
   COMPATIBILITY reading, not a regression**: it is the runtime's hoisted-cursor pattern, and the compiler lane
   must have landed its fix (lane `runtime/cursor-window`) before this bitstream, or correct code breaks on it;
4. a NEW alias-store arm: an out-of-window cursor whose shifted decode contains it must **trap (28)** instead of
   writing a span away, with an in-window control still passing.

Reading 3 is the item's real cost and the reason it is listed last: **it changes the ABI for code that is in the
field today.** It must be sequenced behind the runtime fix, and the board's corpus images rebuilt under the new
rule, or the acceptance boots will fail for a reason that is not a defect.

---

## Gates before the hash is handed to synthesis

Unchanged from every previous build, and none of them is evidence of synthesizability:

- `bash verif/sim/rtl-lint-gate.sh` **PASS** at baseline by NAME, not only by count, run inside the container
  (it fails to start on the host: `verilator: command not found`, which the gate correctly calls an ERROR);
- the 95-test sweep identical, `testlist_sup.yaml` identical plus the new arms at their recorded readings;
- `claim-auditor` over the diff, asked specifically what could break synthesis — inferred latches, combinational
  loops, undriven struct fields, positional struct literals whose field count changed;
- **`UNOPTFLAT` loop membership by STEM, diffed and not just counted.** Item (1) adds a term to the invalidation
  broadcast, which crosses a module boundary into the CPMP path — the highest-risk edit available, and every
  check we have is blind to it. If the change touches a cone on the standing loop list, **it goes to synthesis
  before it goes anywhere else**;
- the hash handed over carries its lint numbers, and **a hash is ready when synthesis has RUN**, not when the
  checks pass.

## What this batch deliberately does NOT include

- The index-only compare's **alternative** fix (a generation in the virtual record) — deferred by a prior ruling
  and not needed once the broadcast carries the namespace.
- Anything for the **third, unobserved S-23 face** (a context clearing the caller grant it is itself fetching
  through). The node floor prevents it for any id the floor covers, and it has never been observed; item (1)
  removes it as a side effect.
- The **CSRUNV+4 open anomaly** (pb10's 2 → 0 under the srevroot-zero module, which no mechanism predicts). It
  needs a reading, not a fix, and the reading is a one-variable pair isolating the inserted instruction from the
  srevroot value — simulation work, not a bitstream item.
