# vp2 on the board, and the whole bug corpus on it in the virtual profile (2026-10-17)

The lead's instruction (2026-10-17, evening): do everything needed for a working vp2, then run every test of the
bug corpus on it in fully virtual Capstone mode. This plan is the decision tree and the work split; it is kept
current by the helper lane. The vp2 design and its acceptance definition live in
[`2026-10-16-virtual-profile-on-silicon.md`](2026-10-16-virtual-profile-on-silicon.md) ("What done means", items
3.2-3.4); the Tier 1/2 kit in `tests/rtl-smoke/vp2-kit-2026-10-17/README.md`; the Tier 3 manifest in its `tier3/`.

**Where things stand at this writing.** `caplifive_vp2_bf3db3175.bit` is sealed (sha256 `776cdd8c…7475`, routed WNS
-19.019, graded at 9c3a27dc54c6 and amended at 6a2062ff338d). vp1 `caplifive_vp1_9ed9a2594.bit` is sealed (-15.615)
and its flash chain is armed behind ST100; vp2's chain is armed behind vp1's with the lead's word (ff4725abd412,
b1516214e63e). The monitor with the CSMINT hook is on capstone-bootstrap c3293b3 and booted Linux on d93a8d66b (boot
BL, 26fd95e2861c). The v6 kit passes `run.py` 41/41 on QEMU (fa22c4592250). Tier 3 on QEMU: 20 of 26 bundles run.
Two attribution builds of vp2 (A: no D3 port steal; B: no Q-15 cone) are in synthesis (93bee16ee3f7).

## The decision tree for the bitstream

```
vp1 session (tonight)        ── differs from d93a8d66b ──► STOP, hold vp2; RTL lane reproduces in sim first
      │ clean
vp2 flash + canary           ── canary fails ──► revert to vp1 (one reflash); build vp2b (below)
      │ passes                                          │
25 arms + corpus set 7       ── an arm differs ──► sim reproduction (RTL lane) before any call of "silicon"
      │ as pre-registered
Tier 1 (contract.dom)  →  Tier 2 (run.py on the board)  →  Tier 3 (the 26 virtual bundles)
```

- **The canary** (f05265150556): S-16 and the Sublet floor exactly, s1gap by its functional readings (the eight cycle
  counts reported, not compared: d93a8d66b's own runs differ in them), two passes plus pass-against-pass. A timing
  failure on this design shows as a wrong or nondeterministic functional reading on a known program; that is what
  the canary measures, and why cycle counts are not its signal.
- **vp2b**, built only if the canary fails: the RTL change is selected by the attribution seals, pre-registered by
  the RTL lane before they land (B names Q-15's cone → the mask test in the DYN unit, a documented equivalence to
  the encoder, no reading changes; A names D3 → a narrow registered FP port for the switcher; neither → the two-level
  id decode, costed first). The synth lane builds verified handoffs without a per-build go (the lead, 2026-10-17).
  Every vp2 arm and Tier re-runs on vp2b; nothing below is specific to bf3db3175 except its hash.

## The rule that binds every reading: one pinned kit, predicted and run on the same bytes

- **One kit revision for Tiers 1-3** (LLVM build, runtime, the virtual SDK with `-fno-builtin` as decided on
  2026-10-17 — a PARTIAL workaround for C-81, explicit `memcpy` only — module, launcher). The compiler lane pins it
  and the board lane builds the board module (6.4.14, 0 RVC) from it.
- **Its identity is recorded in every bundle's `inputs.json`:** the compiler binary's sha256 beside `--version`, the
  runtime archive hash and the full SDK flags. (The manifest recorded none of the flags before 2026-10-17; the
  archive hash cannot see `-fno-builtin`, measured by the compiler lane.)
- **Prediction on image X, silicon on image X.** Every one of the 26 bundles is rebuilt on the pinned SDK and re-run
  on QEMU's virtual profile with exact bounds off; that reading, cell by cell, is the board's prediction. The Tier 3
  readings taken before the pin (20/26 on v6-fix without the flag; the three pool bundles on arm-6) are not
  predictions for the board's images.
- **A positive control that the run is virtual, in the first rung of every boot:** `contract.dom`'s stale access
  denied with 25 (exit 139), or a Q-15 refusal (29 at `capstone_cap_derive`). A boot without one says nothing about
  the profile it ran under.
- **Table geometry must match between the prediction kit and the board kit:** a flat RECYCLE table of at most
  65,536 records, the 256-identity reserve inside a context. A heavy-allocating image reading node pressure (kind 5,
  cause 30) on silicon where QEMU did not is a kit-mismatch candidate first and a silicon finding only after the
  module's request is shown equal on both. The RTL lane's statement of the rule, verbatim, follows. (An earlier line
  here said vp2 refuses PAGED at CSRUNV; it does not: the refusal comes at the context's first allocation, as 29.)

**What the module's lifetime table must look like for the board kit and the QEMU prediction kit to agree (the RTL
lane, from capstone-ariane `virtual-profile-2-d` bf3db3175, 2026-10-18).** The rev-node unit reads the table's
header at EVERY allocation (`capstone_rev_node.anvil:225-238`, QEMU's `table_load` made per allocation) and refuses
the header, status 3, when any of these fails: RECYCLE set (slot 0 word 0 bit 63), PAGED clear (bit 62), next-unused
at least 2 and at most capacity, and a free list whose head and count are both zero or both non-zero. Inside a
context the DYN unit turns status 3 into 29 at the allocating instruction (`capstone_dyn_unit.anvil:284-287`: Q-20's
"a corrupt table reads 29"); CSRUNV itself reads no header, so a PAGED or non-RECYCLE table is NOT refused at entry:
the context enters and its first CDERIVE, CSMINT-emulated mint or MREV reads 29. The glue presents capacity and
next-unused saturated at 65,536 (`ex_stage.sv:1397-1398`: any set bit in [61:16] reads as 65,536) and indexes
records by the id's LOW 16 BITS (`ex_stage.sv:1545-1548`), while QEMU accepts a capacity up to 2^30 and ids up to
UINT32_MAX (`cap_rev_table.c:51`, `:105`, `cap_rev_tree.h:11`). So the kit rule is: a flat RECYCLE table, capacity at
most 65,536 records (1 MiB), PAGED never set, ids 0 and 1 the header, and the module must never let next-unused pass
65,536. Node pressure (kind 5) fires inside a context when capacity − next + free_count ≤ 256 (`:239-246`, QEMU's
`capstone_require_node` with `CAP_REV_TABLE_RESERVE` 256); CSNODE's first allocation needs two identities. **Where a
heavy-allocating image (cpython, perl) can read differently on silicon than under QEMU, and is pre-registered as a
KIT MISMATCH, not a silicon finding:** (1) a table the module sizes above 65,536 records: QEMU allocates past id
65,535 and sees pressure only 256 below its own capacity, silicon saturates capacity at 65,536, so kind 5 fires when
65,536 − next + free_count ≤ 256 and, if the software collects and continues, ids at or above 65,536 alias records
by their low 16 bits (corruption that QEMU cannot show); (2) CSRETIRE of an id at or above 65,536 or below 2 answers
rd 1 without a table read on silicon (`capstone_rev_node.anvil:620`), where QEMU's bound is UINT32_MAX; (3)
next-unused written by software above 65,536 reads as exactly 65,536 on silicon, so a "next > capacity" refusal QEMU
would raise does not fire there. None of the three can occur with a table at or below 65,536 records whose
next-unused stays inside it, and the module's table size is the one input the kit controls. Everything else in the
header checks is bit-identical between the two (the three Q-entries carried into vp2: the table barrier, the
emulation window's rules, CSRETIRE's answers without a read).

## Tier 3's comparison rule, settled from the sources (2026-10-17)

The question was whether the board can give the cell comparison QEMU gives (verdict, cause, faulting symbol,
attribution, reached) or only verdict-plus-exit-status. It can give the full one:

- the launcher itself prints the `domain fault cause= pc= address=` line when `CAPSTONE_EXEC_DIAGNOSTICS=1`
  (`runtime/virtual/exec.c`, `fault()`: `capstone_delegate_fault_record` to fd 2), and `tools/virtualvm.py`'s gate
  script exports that variable; nothing comes from a QEMU-side trace;
- `run-virtual-cases.py` resolves the pc against the image's own symbols with `llvm-nm` on the host, after the run.

So the board kind is **the same gate script in a board boot instead of a QEMU boot**: the same `BEGIN`/`END` step
markers, `virtualvm.split()` over the board's serial log, the same `observe()`, and `verdicts.py`'s bundle writer
producing a verdicts-v1 bundle for a new arm whose `inputs.json` names the bitstream, the firmware hash and the kit
identity. Proposed as a board backend for `Batch.execute()` (stage dir → the board lane's bake into the initramfs
with the module and the launcher → boot → `serial.log`), everything else untouched. The board lane confirms or
amends this before writing it, and pre-registers the rule with anything the launcher cannot produce on the board.

## The work, by owner

| track | owner | what | when |
|---|---|---|---|
| bitstream | board | vp1 session; vp2 flash, canary, 25 arms, set 7 (armed) | tonight, after ST100 |
| bitstream | RTL | vp2b pre-registered against A/B before the seals; sim standby for every differing reading | now |
| bitstream | synth | A and B sealed with stems; vp2b if asked, without a per-build go | ~1 h |
| Tier 1/2 | board | the board arm from the kit README (`VM_EXIT:<name>:<status>`, `BASELINE-PROBE vm_steps`), `contract.dom` first; prediction = the QEMU record c2e0bfa7901e1c76 step by step | pre-registered before vp2's first session |
| Tier 3 kit | compiler | pin the revision; identity into `inputs.json`; table geometry shown equal on both kits; rebuild 26 bundles on the pinned SDK; QEMU re-run with exact bounds off (serialized on the QEMU lock with its C-81 runs); the invocation manifest (argv, stdin, fixtures, env, cc_args per bundle, machine-readable, from the records) | today and tomorrow |
| Tier 3 runner | board | the board backend above; the boot plan computed from the compressed initramfs size of the actual images against the 20 MB working / 32 MB cap, one wedge risk last per boot, the virtual positive control first; boots and hours per bundle group | after the pin |
| Tier 3 fold | helper | `arms.json` configurations, `SCHEMA.md` rows, `check-corpus` oracles, `derive-verdicts`, `build-index`; the independent re-verification (images in the booted firmware by sha256, readings regenerated from the serial log) | as bundles land |
| ports | compiler | the three Sublet (iii) placement repairs (memcached 0006, wmem 0001, memsys5); the per-port SDK soft-float fix (cpython, postgres/sql) | continuing |

**Order of the bundles on the board.** The 17 that read as their records on QEMU first; then the three Sublet (iii)
arms BEFORE their placement repairs, with "29 at `capstone_cap_derive`" as the prediction (on silicon it is the first
reading of Q-15's refusal path), and again after; the six blocked on the per-port SDK fix last, recorded "pending"
until the fix lands rather than holding the campaign.

**Not a gate for any of this:** C-81's backend fix (its own track; the partial workaround is the pinned
configuration and the caveat is in the measurements doc); the attribution builds' outcome when the canary passes.

## Status

| item | owner | state |
|---|---|---|
| vp1 session | board | armed behind ST100 (chain-vp1.sh, 14:18); vb-trans-plain in the batch (76c8fe446fcd) |
| vp2 flash + canary + arms + set 7 | board | armed behind vp1 (f05265150556; the lead's word ff4725abd412) |
| vp2b contingency | RTL | asked 2026-10-17 evening, before the A/B seals |
| Tier 1/2 board arm | board | asked 2026-10-17 evening; to be pre-registered before vp2's first session |
| pinned kit + identity + geometry | compiler | asked 2026-10-17 evening |
| 26 bundles rebuilt and re-predicted on the pin | compiler | to do; 20/26 read on the unpinned kit |
| invocation manifest | compiler | asked 2026-10-17 evening |
| Tier 3 board backend + boot plan | board | asked 2026-10-17 evening; the comparison rule above to confirm first |
| Tier 3 fold tooling | helper | to do |
| Sublet (iii) placement repairs ×3 | compiler | to do |
| per-port SDK fix (cpython ×4, postgres/sql ×2) | compiler | root-caused, to do |
