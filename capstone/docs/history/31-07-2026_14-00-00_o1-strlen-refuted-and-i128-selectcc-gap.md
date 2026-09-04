# 31-07-2026 — `-O1` on the string primitives: REFUTED. Plus a real backend gap found on the way.

## Result first

**SQLite still stalls at `SQ: G/enter` on silicon.** Rebuilding the domain's string
primitives at `-O1` did **not** move it. No rows. Position is unchanged from the previous
session: both share entries complete (`SHA6` twice), the domain reaches its run entry, and
then nothing.

Board run, firmware freshness gate passed (`initramfs 10490880 bytes, verified by
decompressed content`), domain `a41c6a6a`:

```
SQ: E/share1 ... SHA6, ECSZ
SQ: F/share2 ... SHA6, ECSZ
SQ: G/enter
<nothing>
```

## What was tested and why

The previous session froze at image VA `0x14d884`, inside `strlen`, `ra` -> `sqlite3Strlen30`,
`mcause = 0`, pc not advancing under `stepi`. The instruction at that pc sits in this loop,
which is what `-O0` makes of `while (*p) p++`:

```
  14d868: cincoffsetimm a0, s0, -0x40
  14d86c: ldc  a0, 0x0(a0)          <- load p from a stack CAPABILITY slot
  14d870: lbu  a0, 0x0(a0)
  14d874: beqz a0, done
  14d87c: cincoffsetimm a1, s0, -0x40
  14d880: ldc  a0, 0x0(a1)          <- load p AGAIN, same slot, same iteration
  14d884: cincoffsetimm a0, a0, 0x1 <- FROZEN PC
  14d888: stc  a0, 0x0(a1)
```

Two capability loads from one stack slot per iteration, plus a capability store back. At
`-O1` the entire pattern disappears — the pointer stays in a register and the loop contains
no `ldc`/`stc`:

```
  14cc70: movc          a1, a2
  14cc74: lbu           a3, 0x0(a2)
  14cc78: cincoffsetimm a2, a2, 0x1
  14cc7c: bnez          a3, 14cc70
```

That made it the cheapest available discriminator. It is now spent: **the stack round-trip
is not the cause.** The frozen pc was a symptom.

`build-sqlite-silicon.sh` keeps the `SQLITE_SUPPORT_OPT_LEVEL` knob (default = `$OPT`), so
the `-O1` string build is reproducible, but it is not the default and buys nothing yet.

Checked before spending the board session, since `-O1` introduces `movc` into the hot loop
and `movc` is destructive on silicon (`capstone_flu_unit.anvil:6-27` writes cnull to the
source unless it is `CAP_TYPE_NONLIN`): SQLite's strings are cap-table storage capabilities,
which arrive NONLIN (`sqlite_capstone_domain.c:30-34`), so the `movc` is safe. The C-14 pass
correctly leaves capability `movc`s alone — it only rewrites provably-scalar ones.

## RETRACTED on the way

* **"A tight capability store→load can lose its shadow tag."** This was the mechanism I was
  going to attribute the wedge to, and it is wrong: the AXI adapter explicitly interlocks.
  A load that needs a tag read enters `TAG_WAIT` and holds until every outstanding shadow-tag
  write has taken its B-response (`wt_axi_adapter.sv:406-427`, `TAG_IDLE`/`TAG_WAIT`,
  `if (tag_wr_pend_q == '0 && !tag_rd_inflight_q)`). The decoupling is real but it is ordered.
* **Consequence for the board owner.** The drafted question ("is the decoupled tag write
  intentional, and is there a way to drain/fence pending shadow-tag writes before a domain
  switch?") is **answered in-tree — yes, `TAG_WAIT` is the drain.** Do not send it.
* **"`ldc` of a LINEAR capability writes cnull back to memory, so a parked linear cap is
  retrievable once."** Held as a settled fact; it is NOT what this RTL does. `LDC`
  (`capstone_dyn_unit.anvil:296-352`) is a plain load — it validates type, permission,
  bounds, alignment and revocation-node validity, then issues the load. Nothing writes the
  source location. QEMU's `trans_csldc` likewise just loads. Whatever the spec says, neither
  implementation here consumes the memory copy, so "double `ldc` consumes a linear cap" is
  not available as an explanation on this platform.

## Genuine finding: no i128 `SELECT_CC` pattern on RV64

Trying to build the whole amalgamation at `-O1` fails in the backend:

```
fatal error: error in backend: Cannot select:
  t22: i128 = CapstoneISD::SELECT_CC t18, Constant:i64<10>, seteq:ch, t4, t6
```

Two-line reproducer (`/tmp/capstone/o1probe/sel.c`):

```c
char *pick (int n, char *a, char *b) { return n == 10 ? a : b; }   /* FAILS at -O1 */
char *pick0(int n, char *a, char *b) { return n ==  0 ? a : b; }   /* compiles     */
```

`i128` is a capability, so this is `cond ? capA : capB` — ordinary C. In
`CapstoneGenDAGISel.inc` the only two `Select_GPRCAP_Using_CC_GPR` entries are both guarded
by `OPC_CheckPatternPredicate, 35, // !((Subtarget->is64Bit()))`, i.e. RV32 only. On
capstone64 there is **no i128 select pattern at all**. The `n == 0` form survives because
`SelectCC_GPR_rrirr` adds a separate explicit `Pat` for a zero rhs.

Why it was never hit: every purecap domain in this tree is built at `-O0`, where
SelectionDAG keeps the select as control flow and never forms the node.

**Attempted fix, reverted.** Adding explicit `let Predicates = [IsRV64]` `Pat`s with the
compared operands spelled `i64` instead of `XLenVT` changes nothing — after a full rebuild
the matcher still contains exactly the same 4 references, so TableGen drops the new patterns
too, silently (`--warn-on-skipped-patterns` reports nothing for them). Reverted rather than
left in place looking like a fix; a comment at `CapstoneInstrInfo.td:1740` records the state.
Root cause not established.

This is worth fixing on its own merits — it blocks `-O1`/`-O2` for *any* purecap code with a
pointer select, which is a precondition for meaningful performance numbers — but it is not on
the SQLite critical path, since the amalgamation does not need to be optimised to run.

## Where the SQLite investigation actually stands

Unchanged and honest: **runs through both share entries, reaches `G/enter`, stalls, no rows.**
The `-O0` codegen shape at the frozen pc is now ruled out. What has NOT been done is
re-probing the wedge on the current (`a41c6a6a`) build — the previous pc/`mcause`/`stepi`
evidence was taken on `ad0aca1f`, a different binary, so the stall location must be
re-measured before anything is inferred from it.

## Next step

Re-run `probe_sqlite_wedge.py` against `a41c6a6a` with `PROBE_STEPI=1` to get the current
frozen pc, `ra`, and the disassembly around it. Only then choose the next discriminator.
Cheap, one board session, and it replaces stale evidence rather than reasoning from it.

---

# UPDATE, same day — "THE WEDGE IS GONE" is RETRACTED

## The retraction

Commit e03a3124 claimed the linear-safe `strlen` cleared the wedge, on the evidence that
pc advanced across three `stepi`s (`0x14cc74 -> 0x14cc78 -> 0x14cc7c`) where it had
previously been pinned. **That inference was wrong.** Three back-to-back single-steps
prove the debug module can force the core past an instruction; they do not show the core
runs when resumed. It is the mirror image of the ambiguity `probe_sqlite_wedge.py`'s own
comment documents for `monitor resume`, and it was walked into anyway.

`probe_sqlite_progress.py` samples over wall-clock instead, and settles it:

```
[sample 0..4] pc = 0x14cc74 [in strlen]  a0 = 31342951   -- 5 samples, 20 s apart
VERDICT: STUCK -- pc stayed in strlen and a0 did NOT climb
```

Same pc, same register, ~100 s. The core is **not** executing. The domain is still wedged.

The general lesson, which has now cost two wrong conclusions: **a pc measured under
single-step says nothing about free-running execution.** Any liveness claim needs samples
separated by wall-clock with the core resumed in between.

## What the run DID establish, and it is the useful part

`a0` is `strlen`'s index (`li a0, -1` then `addi a0, a0, 1`). It read **31,342,951** --
31.3 MB of characters scanned -- against a total domain image of 1.37 MB (`.text` is
0x13f5b0). Every one of those 31 M `lbu`s through the string capability was in bounds and
did not fault.

So the capability handed to `strlen` has bounds of **at least 31 MB, ~23x the entire
domain**. No object in this domain is remotely that size. That is not a linear-capability
problem and not a codegen-shape problem: **`strlen` is being called on a pointer with
wildly oversized bounds**, it runs away, and the core eventually wedges -- plausibly on
crossing into unmapped address space, which fits the AXI error-slave response
(`0xca11ab1ebadcab1e`, `axi_err_slv.sv:25`) seen in `$a1` in an earlier dump.

The linear-safe `strlen` is still correct on its own merits and is kept -- the walking form
really is unsafe for a linear argument -- but it is not the fix, and it was never the
disease.

## Where the bad capability is NOT coming from

`sqlite3Strlen30` (VA 0x16adc, the caller in `$ra`) is clean:

```
16af0: movc a1, a0          <- string capability arrives as the argument
16af8: stc  a1, 0x0(a0)     <- parked as a full 16-byte capability
16afc: ld   a0, 0x0(a0)     <- low 64 bits as an integer == the `if (z == 0)` null check
16b1c: ldc  a0, 0x0(a0)     <- reloaded as a capability
16b28: jalr a1              <- strlen(a0)
```

Stored with `stc`, reloaded with `ldc`, no narrowing lost. The over-wide capability is
already over-wide on entry, so the defect is in whoever calls `sqlite3Strlen30`.

## Next step

Find where a string pointer acquires oversized bounds. The prime suspect is the untouched
**C-13 `lla`/`auipc` sub-lead**: `strlen` itself is reached by
`auipc a1, 0x136; addi a1, a1, 0x138; jalr a1`, a plain integer PC-relative address. That
is fine for a call target (cap-mode `jalr` derives authority from `pcc`), but the same
pattern used to form a DATA pointer applies an integer offset to whatever capability is in
the base register rather than to a bounded cap-table entry -- which produces exactly a
capability with region-sized bounds.

Offline and cheap: scan the domain for `auipc`/`lla`-formed pointers that reach a
string-consuming call, and compare against the gp cap-table path. No board time needed
until there is a candidate.
