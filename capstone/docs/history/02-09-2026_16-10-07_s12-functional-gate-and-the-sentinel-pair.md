# S-12: a QEMU functional gate, and the matched pair that asks the RTL's question directly

## The gate

`probes/s12-funcgate.py` runs a patched image under QEMU and requires the guest's own markers —
`SQ: G/enter`, `SQ: H/return`, the `SLT-SUMMARY` counters — to match the base's. QEMU never
reproduces S-12, and that is exactly what makes it the right instrument: any difference it reports
is caused by the cut and by nothing else.

It must be run **serially**. Two concurrent instances produced one guest that died at boot with an
EXT4 error and another that hung past four minutes where the base finishes in thirty seconds.
Neither is a statement about the variant, and both read like one — the first arm re-run alone was
clean.

Results on the four cuts the static gates left admissible:

| cut | verdict | evidence |
|---|---|---|
| `{28}` | behaviour-preserving | `SLT-SUMMARY` identical to base |
| `{30}` | behaviour-preserving | `SLT-SUMMARY` identical to base |
| `{33}` | **DIVERGED** | capability fault, pc `0x101cbb27c`, never returns |
| `{32,33}` | **DIVERGED** | same fault, same pc |

`{33}` and `{32,33}` are the null-capability store immediately before the faulting pair — the
plan's priority board candidate and the direct deletion test of the correlation retracted twice.
Both break the program, and they break it in the same downstream function EJF faulted in on
silicon. Boarding either would have bought another unreadable verdict at three draws apiece.

## What the RTL says, and the one thing that separates the two accounts

Two mechanisms are on the table, from a read of the operand path:

* **Writeback-port displacement.** An `ldc` retiring through a scalar port rather than the
  capability port; the tag is tied to zero there while the cursor forwards correctly. Already
  instrumented in the shipping bitstream as `sw=204`. It predicts `tval` = the **real, non-zero
  cursor**, and every S-12 wedge has latched `tval = 0`. (The `sw=204` read at the wedge itself is
  VOID in every log checked — the readback faults once the core is wedged — so this is a structural
  argument, not an instrument reading.)
* **A missed RAW hazard.** The consumer issues without the hazard on the in-flight `ldc` being
  detected, neither stalls nor forwards, and takes rs1 from the register file — the value from
  *before* the load. In this window that value is written by `[32] movc a4, zero`, two instructions
  ahead, so it predicts `mcause 25` with `tval = 0`: the observed signature, bit for bit.

The second matches and the first does not, but the match is not evidence, because `tval = 0` is
also what a delivery path that simply presents zero would produce. **The two are indistinguishable
while the stale value IS zero.** So change the stale value.

## The pair

`probes/s12-sentinel.py` builds both arms out of the pinned base. Every null the program stored is
still stored, to the same slot, with the same value — `t0` carries it, and `t0` is referenced zero
times in the function's 4600 instructions.

    [26] movc a4,zero       -> movc t0,zero
    [27] stc  a4,-0x5a0(s0) -> stc  t0,-0x5a0(s0)
    [28] sw   a4,0x0(a5)    -> sw   t0,0x0(a5)
    [30] sw   a4,0x0(a5)    -> sw   t0,0x0(a5)
    [33] stc  a4,0x0(a5)    -> stc  t0,0x0(a5)
    [32] movc a4,zero       -> li a4, 0x5a5   (SENTINEL)   |   movc a4,zero   (CONTROL)

The two images differ in **exactly one 4-byte word**, at file offset `0xf5808`: `0x5a500713`
against `0x1400175b`. Both pass the functional gate with `SLT-SUMMARY` identical to the base.
`li` is safe for this: `lui`/`addi` metadata leakage was refuted in simulation with a firing
self-check (`cincoffset-stale-metadata.S`, 2026-08-08), so a4 ends up genuinely
`{cursor 0x5a5, NOT_CAP}` rather than stale-tagged.

Pre-registered reading, on the sentinel arm:

* `tval = 0x5a5` — the consumer read the stale register file. Decisive at n=1: nothing else in the
  image writes `0x5a5`, and a load delivers a capability cursor, not a 12-bit immediate.
* `tval = 0` — refutes the stale-read account outright. Nothing writes zero to a4 in this image any
  more, so a zero operand has to be manufactured by the delivery path itself.
* `tval` = a large address — the operand came from elsewhere; displacement predicts precisely this.
* clean on every draw — evidence *against* the stale-read account, which predicts the fault at the
  unchanged rate, but confounded with plain perturbation. This is what the control is for.

The control is what makes a cure attributable. The register-patch arm cured 0/4 by changing which
register held the null, and that result could never be attributed because nothing else was held
fixed.

## A bound on the existing record

Of 134 logs in the working area carrying a latched trap `mepc`, 31 latch the canonical site
`0x828f4814`, and exactly **one** carries a per-run `.dom` sha256 — the pinned base. The freshness
gate that prints it was added later, so the other 30 are attributed by name and by segment
geometry, and geometry cannot distinguish the base from a NOP-patch of it (identical size,
identical segment). The signature stands. The word "attested" should be used precisely about it.

## The instrument defect found alongside

`run_sqlite_stages_fpga.py` printed *"the STALE-OPERAND ACCOUNT IS CONFIRMED"* whenever `x14` was
non-zero, with no reference to where the core trapped — a check that cannot fail in the informative
direction. It fired on a draw that wedged on a different instruction in a different image, and that
reading became a finding that had to be retracted. It now requires the latched `mepc` to equal the
site where a4 *is* the faulting operand, and says explicitly that it is withholding a verdict
otherwise. Negative-tested both ways.
