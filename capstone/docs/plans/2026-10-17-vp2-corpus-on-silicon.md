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

## Kit blockers found from the sources before any board time (2026-10-17, evening)

Each of these would have ended vp2's first virtual boot at its first allocation or its first CSR access, and none is
visible on the QEMU prediction kit, which implements what the module assumes. All are the compiler lane's, in the
module, behind one parameter (`node_table_flat`, default off so the paged kit is unchanged and the record tells
them apart), regressed on QEMU with the parameter on and off (run.py 41/41, bundle 5 13/13; the off arm is the null
control of the paged path) before any hash reaches the board lane.

1. **The module publishes a PAGED table.** `c->table[0] = CAP_REV_TABLE_FLAGS`, and capstone-qemu d35af47b163c
   `target/riscv/cap_rev_table_abi.h:13-15` defines `FLAGS = RECYCLE | PAGED`; the table grows by pages
   (`node_add_page`). vp2 implements only the flat format and refuses a PAGED header with status 3, which reads 29 at
   the context's first allocation. **Flat mode:** one physically contiguous 1 MiB allocation at context creation
   (order 8, `__GFP_RETRY_MAYFAIL`; the root is a physical address, QEMU's flat preflight wants the whole
   `capacity*16` mapped), capacity published as 65,536 from `open()`, no growth path, ids 0 and 1 the header.
2. **Two paged-only header words sit inside record id 2 in the flat layout.** The module writes MAGIC at root+32
   (`table[4]`) and reads the RETIRED count at root+40 (`table[5]`, written by QEMU only under `t->paged`,
   `cap_rev_table.c:177-181`) in its collection heuristic (`capstone_vm.c:392, :401, :864`). In flat mode the first is
   dropped and the second is kept by the module itself: incremented on each `CSRETIRE` that returned 0, reset at the
   collection that reclaims the retired records. The RTL publishes no retired count in flat mode (the unit writes only
   the four header words, `ex_stage.sv:1559-1565`). The collection trigger gets its own two-sided regression row (a
   workload that collects under the paged module collects under the flat one, counts comparable).
3. **CSR 0xcc0 (urevavail) does not exist on vp2**: a read is an illegal instruction (`csr_regfile.sv:1079`). QEMU
   implements it as `capacity − next + free_count` read live (`capstone_table.c:199-205`). The module computes the
   same from the header words, saturated at 65,536, and never reads the CSR on silicon. Q-21.
4. **CSR 0x5c0 (scapctl) does not exist on vp2** either (any access traps). On QEMU it is "protected U" (bit 0 PU:
   U mode outside a context uses the guest table; the read ORs 0x100 when exact bounds are on). The RTL's namespace is a
   context, S with srevroot ≠ 0, or M with csupctl bit 3; U outside a context never reaches the table. The module
   must not execute the 0x5c0 access in flat mode, and answers the exact-bounds query with 0. Q-22. Open: whether
   anything run.py or the corpus runs in U outside a CSRUNV context with capabilities (believed not: the launcher's
   own code is plain Linux, every domain instruction runs inside the context); the compiler lane answers from the
   module's lines.
5. **The module's CSR set on silicon is exactly** srevroot 0x5c1 (S and M RW, 4 KiB aligned, the write flushes the
   pipeline), satp, and CSRUNV/CSRETIRE/CSNODE as instructions. csupctl/csupquantum are M-only; csupstatus/cause/
   epc/tval and csnodefree are M-only read-only; an S-mode CSRUNV caller reads the event from rd and the frame (+16
   kind, +24 cause, +32 pc, +40 addr, reply +224). Control: an objdump of the flat-mode module showing no reachable
   csrr/csrw of 0x5c0 or 0xcc0, with the paged build as the positive control that the grep finds them.

**The flat layout, confirmed word for word by the RTL lane against QEMU's `table_get`/`table_put`** (not the stale
comment at `cap_rev_table.h:8-16`): record i at root + 16·i (QEMU `cap_rev_table.c:9-11`; RTL `ex_stage.sv:1555-1556`
indexes by id[15:0]); word 0 = prev[31:0] | next[63:32]; word 1 = depth[29:0] | parent[59:30] | VALID bit 60 |
LINEAR bit 61 | FREE bit 62 (the RTL writes parent as 16 bits and depth as 17, exact inside a 65,536-record table); a
FREE record has word 1 == exactly FREE<<60 and word 0 = the link; header slot 0 = {capacity | RECYCLE<<63, next-unused},
slot 1 = {free_head | free_count<<32, allocations}. **No RTL arm has ever read a table QEMU's code wrote**: the
cross-check is the flat-mode module's own table on QEMU, dumped whole (1 MiB) after a run.py run with the
allocation/retirement sequence, fed by the RTL lane as `.data` into an arm whose readings are QEMU's.

**Tier 3's payload budget (decided by the helper lane under the lead's delegation, 2026-10-17):** a Tier 3 bake may
total 26,000,000 bytes when every file beyond the clean base is named in the bake manifest with its sha256; the 20 MB
rule stays for every other boot. The preflight's C10 purpose (no undeclared stale domain) is kept as that manifest
check. Negative tests recorded with the change: an undeclared file BLOCKS; a declared bake at 26,000,001 bytes
BLOCKS. Measured by the board lane: a clean firmware is about 17.5 MB, case bundles gzip to 0.7-2.3 MB each, so the
old rule gave one bundle per boot (about 17 boots); the budget gives 4-5. The gate changes only after vp1's armed
chain has passed its second preflight. Bundles over the payload are split by case across boots or recorded as not
stageable at their measured size, never squeezed.

## The first virtual boot, and the ruling it forced (2026-10-17 night, Q-23 and Q-24)

vp2's first virtual boot (board lane, lane vp2-tier12 c1fa1a863d09) wedged at `contract.dom` before any output:
EXCX cause 24 from S at `capstone_vm.ko` add()'s `ldc t0, 0(a5)`, a5 the slot (an integer kernel pointer), right
after the emulated CSMINT; then a second 24 at the monitor's own `_cap_trap_entry+4`. Neither pre-registered branch
matched, so it went to the RTL lane before anyone called it silicon. **Not silicon**, three findings of record:

1. **Q-23.** The RTL's DYN unit refuses `LDC`/`STC` with a NOT_CAP rs1 in every mode
   (`capstone_dyn_unit.anvil:641-654`, `:716-719`); QEMU alone had an "experimental trusted-kernel context spill"
   branch (`trans_capstone.c.inc:214-224`, S-mode scalar-address `LDC`, no bounds or permission check), exercised on
   purpose by `tests/trusted-linux-u-access`, and no such branch for `STC`. The module's four integer-based accesses
   (all in add(): block A `ldc/delin/scc/stc` to retype LIN → NONLIN and set the cursor; block B `ldc/stc` copying the
   image capability into register slot 0) run on EVERY domain launch (five of the eight grants are `linear = 0`),
   so nothing of Tiers 1-3 could run on the resident module. Block B needs a scalar `STC`, which QEMU does not have
   either, so "QEMU as reference" decided nothing.
   **Ruling (the semantics owner, under the lead's delegation, 2026-10-18): the RTL stays; trusted S performs no
   capability memory access; QEMU is tightened to the RTL (the scalar-S `LDC` branch kept only under the
   test-only property `x-capstone-test-s-ldc`, default off, for three directed suites' bootstrap; no production kit
   sets it, and the kit's QEMU invocation is grepped for it as a gate); the CSMINT descriptor becomes v7** (`isa.md`: base, end, perms | bit 8 NONLIN, cursor with `base <= cursor <=
   end`, a second slot address or 0, reserved 0; one node in both slots; LINEAR with a second slot is 29). Feasible
   on the RTL because CSNODE accepts any tagged rs1 (`capstone_dyn_unit.anvil:585-592`), SCC is a pure register op
   (`capstone_flu_unit.anvil:104-126`) and the switcher loads the context's PC capability from frame +64
   (`capstone_dom_switcher.anvil:157, :319`), so slot 0 is the hardware's and the second slot fills it. No bitstream.
   The alternative, scalar `LDC` and `STC` from trusted S on the RTL plus the `STC` branch in QEMU (vp2c), was
   weighed and rejected: a day plus a flash cycle to codify a branch QEMU itself marked experimental.
   Consequences: the module loses all four accesses (gate: 0 `ldc`/`stc` encodings in the `.ko`, the pre-ruling
   module as the positive control with 4); the monitor routine mints NONLIN directly (CAPCREATE/CAPTYPE/CAPBOUND/
   CAPPERM → CSNODE in the window → SCC outside it → STC, STC); the emulation arms are re-pre-registered for v7; the
   Tier 0 suites that build descriptors move to six words and are re-measured; run.py's 41/41 on the old QEMU is
   NOT inherited (it passed through the removed branch), the regression reruns on the tightened QEMU.
2. **The emulation's restore is exact** (`sbi_capstone.S:14-45`, `return_to_sumode` `:163-199`); a5 was the slot;
   the 24 is finding 1.
3. **Q-24, the wedge.** `handle_exception`'s default (`sbi_capstone.c:3520-3554`) calls `fault_return_from_domain`
   with no domain active (MPP = S), never runs `return_to_sumode`'s CSCRATCH swap, so the next trap entry runs
   CINCOFFSETIMM on an integer sp: the second 24, tval = the kernel stack. medeleg cannot delegate causes 24-30 on
   this RTL (`csr_regfile.sv:1642-1659`, the write mask stops below 24; the switcher's unmasked write at `:2023` is an
   M-owned path, an inconsistency to record, not a loophole), where QEMU makes them delegable. RTL fix: the mask gains
   24-30, cycle-neutral, pre-registered for the next bitstream. Monitor fix now (board lane, reviewed by the RTL
   lane): an unhandled cause with MPP = S (sufficient for OS-versus-domain: a classic domain traps with MPP = M) is
   injected into S as `csm_inject` does and leaves through `return_to_sumode`; `fault_return_from_domain` refuses to
   run without an active domain.

**Boot 2 (2026-10-18 00:38-00:50; vp2-tier12 279d993e273d; module ed24f827b5beb373 at runtime 179686bcf198, firmware
41fb669c9f77462d with monitor vp2/tier1-monitor 1e9451a; prediction run.py 41/41 on f2f15f8fc70c).** Controls pass. The
result of record: **the S-trap injection works on silicon**: the module's CSRUNV trapped from S with cause 29, the
monitor injected it (`INJS 1D`), and the kernel printed a correct oops (epc in vm_ioctl, cause 0x1d, SPP = 1) instead
of wedging as in boot 1. Every virtual step faulted earlier with cause 26 at contract.dom VA 0x11f60, `delin a0` in
`__capstone_cap_init` right after `scc a0, gp, a0`: gp is granted NONLIN, the RTL's DELIN raises 26 on any non-LINEAR
rs1 (`capstone_dyn_unit.anvil:794-801`), QEMU's `helper_csdelin` no-ops it, so the glue's delin has been dead code on
QEMU since the module delinned the grants itself (C-13 class; S-15 in the silicon-mimic). **Ruled the same way as
Q-23: the RTL is the reference; QEMU's default DELIN raises 26 on a non-LINEAR rs1; the images are rebuilt on the new
pin with the module unchanged; run.py on the UNFIXED images must fail at the same delin on the new QEMU before the
fixed ones pass.** The board lane's census then showed the delin is not one glue site but the backend's
`-capstone-image-gp` global-address materialization, `scc rd, gp, x; delin rd` with no type test: 525 of contract.dom's
530 delins, 5,983 of sqlite3.dom's 5,988, 9,859 of mruby's 9,863. **This is C-27**, filed 2026-09 and recorded FIXED in
ISSUES-ARCHIVE ("the DELIN is dropped under the NONLIN-gp contract") while the backend still emits it unconditionally
(`CapstoneExpandPseudoInsts.cpp:381`; the cited pin asserts the delin is still emitted; no NONLIN-gp option exists;
`CapstoneISASemantics.md:52` still lists it): the false FIXED is why boot 2 was spent, and the prior-art search that
would have caught it (the registry's archive) was not made before the boot. The retraction lands as its own commit.
Fix (b), recommended: the backend drops the delin under the ASSERTED NONLIN-gp ABI (two consumers depend on the base
being NONLIN: the copy hazard and the SHRINK double read, so the contract is asserted, not assumed; a LINEAR gp would
also make every global access write the shared node's linear bit on the RTL). **The decision is reserved to the
project lead** (`CapstoneISASemantics.md:52`, "the Tier 4 DELIN decision"); the compiler lane stopped on it and the lead
is asked. Then: the census script is the gate on the rebuilt images (0, against 525/5,983/9,859); the 5 non-gp delins
per image are named one by one; the backend revision joins the compiler identity in `inputs.json`.
**Context slots:** QEMU holds 32 (`SUPERVISOR_SLOTS`), the RTL one; run.py's parallel steps pass on QEMU for that
reason and boot 2's 29 at parallel_one was the pre-registered refusal of a second launch. QEMU gains a slot-count
property defaulting to 1 (the DELIN commit); the module serializes contexts on a one-slot machine and never issues a
CSRUNV the hardware would refuse (a 29 trap in S is a kernel panic); the parallel/preempt/heap-threads expectations on
vp2 are derived from the serialization mechanism and pre-registered on the 1-slot QEMU before the run. Open with the RTL lane: whether CSRUNV from S with an invalid frame or
action traps with 29 (as boot 2 read) or returns 29 in rd (as the plan's item-D text says); QEMU and the kit's
expectation follow the RTL. Boot 3 is boot 2's pre-registration with the images' hashes moved and the delin branch
added. No strict-DELIN property for prediction runs: the strict behaviour is the default, a test-only property exists
only where a suite's bootstrap needs the old one.

Owners: the RTL lane the QEMU change and the re-run arms; the compiler lane the module (one post-ruling revision,
with the early-clobber fix, the collection row, the ceiling reading and the flat define); the board lane the monitor
routine and the injection fix; the helper lane `isa.md` and this record. No board boot before: the tightened QEMU
pinned, the new module through its gates and the from-scratch regression, the emulation arms re-run, the 1 MiB
table cross-check.

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
with the module and the launcher → boot → `serial.log`), everything else untouched.

**CONFIRMED by the board lane (2026-10-18) with seven amendments, before the backend is written** (`BoardBatch(Batch)`
whose `execute()` stages into the Tier 1 bake and firmware recipe and drives the baked-rung driver; dry-run end to end
with the driver stubbed before any board time): (1) the backend reads the batch rung's own segment from the driver's
per-rung `board.txt`, never the raw capture, because the console replays the previous boot's tail on connect (the
replay trap, made structural); (2) the stage lives at `/test-domains/<batch>/`, `insmod` is guarded on `/proc/modules`
(an earlier rung may have loaded the module), the final `rmmod` fails harmlessly on the board kernel; (3) a wedge takes
the core and the later steps read kind `none` → `infra`, so each boot runs the virtual positive control first, the
expected-to-return cases, and at most one wedge risk last; (4) the gate script carries a heartbeat no `observe()` reads,
since the driver ends a rung after `BAKED_IDLE_S` of silence; (5) each boot is the clean base plus the batch under the
declared 26,000,000-byte cap, the plan computed from the actual compressed sizes, a bundle group over several boots
each repeating the positive control; (6) a pre-HELLO fault reads `entry=unknown` on BOTH machines (the launcher prints
`code=lo-hi` only after the domain's HELLO, `runtime/linux/delegate-service.c:977-984`, and `verdicts.FAULT_LINE`
requires `code=`), so such a NO-READING is the launcher's limit, not the board's; (7) `inputs.json` names the
bitstream, the fw_payload and Image sha256, the module and launcher hashes, and writes the `qemu` key as
`board:<bitstream>` so the arm cannot be read as an emulator run. Cycle and time readings are not cell fields.

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
| vp1 session | board | DONE 2026-10-17 (c5e696ac7328): S-16 98/98, floor 77/77, s1gap functional identical to d93a8d66b, the four vp1 arms as pre-registered, set 7 identical; one JTAG non-start re-run, pre-registered before the retry |
| vp2 flash + canary + arms + set 7 | board | DONE 2026-10-17 (987bf254ee40): flashed 20:40:46 on the lead's word, readback names caplifive_vp2_bf3db3175; canary 9 of 9 (two passes and pass-against-pass); 25 of 25 arms equal to their simulated vectors on bf3db3175's RTL; set 7 33 of 33 identical to d93a8d66b; every run matched to its image hash; the comparator negative-tested on the session's own readings. **vp2 is resident; the physical regression holds (done-means 4).** |
| vp2b contingency | RTL | pre-registered before the seals, dev 733ad96033df; NOT built: the canary passed. B is graded (44f14d011de5: Q-15's cone is not the ten thousand); A pending, for the record |
| Tier 1/2 board arm | board | stage, bake recipe, firmware recipe and judge ready in its private tree; HELD on the flat-mode module (kit blockers 1-5); then pre-registered on dev and armed behind vp2's set 7 |
| flat-mode module (kit blockers 1-5) | compiler | built and gated (board ko 0d62a1bcdeecc13f flat; run.py 41/41 and bundle 5 13/13 on both adapters, c14933904de4) — superseded by the v7 module below |
| Q-23/Q-24: the first virtual boot's wedge | all | ruled 2026-10-18 (a2739d9313ca, 2ed59b4d0e6c, 8865adec18f9): no bitstream. Boot 1's attribution landed (9536a7531531). Monitor: capstone-sbi capstone-bootstrap 83966bc (the S-trap injection) + dafe4f2 (the v7 routine, gated to S) = vp2/tier1-monitor 1e9451a's tree; caplifive-opensbi capstone-bootstrap bumped to it at bc55760 after BLb (five arms, placement confirmed as the mechanism of a −0.77 % so_lists / +0.32 % fib shift; fa049994c1eb; physical-line timing baselines are re-taken on the bumped firmware and cite B/B2). v7 module: board ko 9ff967445799c02b at the delin-era revision, run.py 41/41 and bundle 5 13/13 on both adapters at f2f15f8fc70c; the collection row measured (both arms collect at the default, reclaim within 0.33 %); emulation arms on the final monitor 81/81, 81/81, 23/23 |
| boot 2 (Tier 1/2 on the v7 module) | board | DONE 2026-10-18 (8350197f7649; attribution audited at vp2-tier12 2a01393a18e4): the injection works on silicon; every virtual step faults at the glue's delin of the NONLIN gp = C-27 (falsely FIXED in the archive; retraction by the compiler lane); the 29 at parallel_one = the one-slot refusal (Q-26) |
| QEMU reference pin | RTL | capstone-qemu `virtual-capstone-v7` 3b8040d5ff6e (32d1a2e97fe7): v7 descriptor and the scalar-S gate (f2f15f8fc70c), strict DELIN and `x-capstone-supervisor-slots` default 1 (852c5bc76774), `scapctl`/`urevavail` illegal by default under `x-capstone-test-csrs` (3b8040d5ff6e); Tier 0 re-measured at the defaults. **Q-29 RETRACTED the same day by the RTL lane, and the ruling built on it withdrawn:** QEMU 3b8040d5ff6e DOES check every virtual-C block's fetch against the PCC (`capstone_supervisor.c:982-1062`, `translate.c:1425-1433`, one-instruction exact; tests/virtual-capstone-runtime's pcc_bounds, fetch_full_width, cap_execute, revoked_pcc and revoked_cached_pcc prove it) with the RTL's codes. What remains is a low-priority entry: the order when two PCC conditions hold at once (QEMU liveness first, the RTL type/permission/bounds then liveness, `commit_stage.sv:379-391`) and the legacy-C monitor's own PCC unchecked on QEMU; neither touches a corpus context. **The 26-bundle Tier 3 re-prediction runs on 3b8040d5ff6e.** The lesson (the RTL lane's): run the suite that would test a behaviour before claiming the model lacks it |
| C-27 fix (the backend drops the delin under the asserted NONLIN-gp ABI) | compiler, the lead's word | recommended; reserved to the lead by `CapstoneISASemantics.md:52`; the compiler lane holds the change ready; the control (run.py on the current images on 852c5bc76774 failing at the delin and at the second CSRUNV) runs meanwhile |
| context serialization in the module (one slot on vp2) | compiler | to do, its own commit; the parallel/preempt/heap-threads expectations pre-registered from the mechanism on the 1-slot QEMU |
| C-27 validated on the rebuilt images | compiler | DONE 2026-10-18 (lane 6fa0e2483a71): 0 gp-derived delins with the totals falling by exactly the old counts (530→5, 5,988→5, 9,863→4); run.py 41/41 zero false on the old pin as the control; images contract f2017064bea77dd2, sqlite 233640f7193b034c, mruby bba891bc16576d98; compiler 01b62884c0c7 with a feature probe beside the version |
| the QEMU kit's firmware lacks the S-trap injection | board | FOUND 2026-10-18 by the matrix on 852c5bc76774: new images pass `normal` and die at `threads` (step 2) with the kit's M firmware (c3293b3's tree, injection FPGA-only) faulting 24 at its own pc after the one-slot refusal 29; silicon injected the same refusal cleanly in boot 2. Fix: a QEMU-kit firmware with the injection under a vp2-kit define, pinned, RTL-reviewed; the matrix re-run on it |
| thread steps on one slot | compiler, RTL | RULED from the switcher's lines (RTL lane, 2026-10-18): a paused context occupies the one slot and blocks a second one's entry (`commit_stage.sv:271-276`, mcause 29); re-ENTRY after a discard is not a resume (CSRs and FP zeroed, no pc advance; `capstone_dom_switcher.anvil:190-227`). So `threads`, heap-threads, preempt and any step where a live context waits on another are **"not available on a one-slot bitstream"**, Q-26's consequence: the module's queue admits sequential contexts only and returns -EBUSY where a second live context is required, never a trap. The RTL remedy (N saved contexts in the CSR file, the refusal becoming "no free slot") is costed by the RTL lane at N = 2/4/8 (dev fe641742ce4c; +74·N + log2 N flops as the sharp key, ~+85·N LUT below the netlist band, the compare path's slack +20.712 ns combinational / +36.130 registered from vp2's checkpoint) and stays DEFERRED: the compiler lane's count is **zero of the 26 bundles** needing a second live context (236 domain sources of the 16 case.c corpora scanned, 0 thread or process creations, contract.c as the firing control; the two script-trigger corpora's images define neither thread-create symbol, contract.dom as the control; two void passes on the way recorded). Caveat, Tier 2 not Tier 3: the kit apps sqlite3.dom (THREADSAFE=1) and mruby LINK the thread-create frame symbol; linked is not called, and those three steps have no one-slot reading yet (the matrix died at `threads` first), so they run alone on 852c5bc76774 before the queue is designed; if one needs a slot, the 39/41 pre-registration changes |
| boot 3-T1, first run (2026-10-18 03:17-03:23, 2d8559e63ed0) | board | READ: C-27's fix holds on silicon (the run passes `__capstone_cap_init` for the first time); then both `ok` and `stale` exit 139 with cause 2 at contract.dom VA 0x20ccc, `csrr a0, 0xcc0` in the virtual runtime's malloc (tval = the encoding): **Q-21 reaching the application runtime** (malloc and realloc read urevavail twice per image; the module has 0; QEMU implements the CSR, so every record passes). Ruled as DELIN was: the application runtime never reads 0xcc0 or 0x5c0 (availability inside a context is the kind-5 pressure event, not a polled CSR; the compiler lane removes the reads and rebuilds); QEMU's default makes both CSRs illegal (test-only property only where a Tier 0 suite needs it; the RTL lane re-pins); the Tier 1 and Tier 2 app-step records are re-made on that pin (the one-slot app-step record is withdrawn as QEMU-only); the board lane adds a CSR census beside delin-census.py (today's images the positive control, 2/2/2). **General instrument asked for:** a decode-whitelist census of every staged image against the RTL's decoder (the RTL lane supplies the table from decoder.sv/riscv_pkg.sv with file:line), replacing one-off censuses: the three classes that cost boots today are its positive controls, and a staged image with a non-empty report does not boot |
| the decode-whitelist gate, first reading | board | BUILT on the census TSV (lane df45a9907d74). The rebuilt images (contract 125705371837ea92, sqlite a5942a1add489597, mruby a902e740fc5d3b0f): 0 refused, 0 gp-derived delins, 0 QEMU-only CSR accesses, ONE candidate each: libc_exit_fini's `addi s1, s1, -0x10; mv a0, s1; ldc a0, 0(a0)`, an integer ALU result as an LDC base (untagged on both machines, cause 24), unreachable in these images (`.fini_array` size 0, the `bgeu` at +0x44 skips the loop). **Ruling: attributions are admitted** in a committed file beside the gate, one row per (image sha256, address, reason, checkable evidence); the gate passes only when every candidate has a row for that exact hash (a rebuild invalidates every row); every report prints the attributed rows. **RETRACTED the same day** (board lane): "any non-empty .fini_array faults 24 at exit on both machines" is wrong, refuted by the compiler lane's positive control (C-82, c82-destructor.dom a250d387ddfd352d: one destructor, a 16-byte .fini_array, the same addi/mv/ldc sequence present; prints C82_DTOR_RAN and exits 0). musl's `libc_exit_fini` has zero call sites: the runtime's strong `__libc_exit_fini` overrides the weak alias and is capability-correct, so no image is exposed; and it is not a codegen defect: musl does integer arithmetic through `(uintptr_t)` and the backend emitted what it was told. What stands: the gate reports the pattern, now refuses it under the TSV's row 54 ("integer ALU op on a tagged register", dev 35b14e455880 / b29410a7fe1a), and the three attribution rows carry it with checked evidence (their reason understates: the function is never called). The lesson, the same as sqlite's pthread_create the same day: linked is not called; check reachedness before claiming exposure |
| boot 3-T1, repeat, read (2026-10-18; the board lane's details and auditor pending) | board, RTL | C-27 and Q-21 fixed on silicon: the run reached the first reply; vm-contract stops with EFAULT at the first reply-slot LINEAR grant, the adapter requiring the frame's +224 slot EMPTY before writing the reply. **Q-30, ruled on the RTL lane's source read:** on bf3db3175 every escape writes x1..x31 with their tags into the frame's own slots (`capstone_dom_switcher.anvil:155-161`, `:513-526`), so x10 sits at +224 until the reply resume loads and zeroes it (`:297`, `:338-340`); QEMU 3b8040d5ff6e keeps the callee in save_state and leaves the slots at ENTRY's zero (`capstone_supervisor.c:419`, `:951-958`), so the empty-slot check passed only on QEMU. The RTL's frame is the reference: the adapter OVERWRITES +224 with the reply (the compiler lane, in the queue's revision); QEMU's escape writes the registers into the frame and restores the caller over the register file as the RTL does (the RTL lane, the next pin, a Tier 0 arm reading the frame after an escape); the Tier 1/2 records are re-made on that pin before boot 4. The third run of 3-T1 goes on the changed module against the current pin (the frame write is QEMU-side faithfulness, not a prediction change for contract.dom) |
| silicon-only privilege escalation through child-thread frames | board (S-entry), RTL (demonstration, fix costed), compiler (module fix now) | FOUND 2026-10-18 by the board lane's auditor at bf3db3175, not yet executed: ENTRY saves the S caller at frame+1024 (`commit_stage.sv:617`) and every escape restores it unchecked (`:856`; `dom_switcher.anvil:508`, ids 0..66; CSR ids 1..25 and 57..66 with no privilege gate, `csr_regfile.sv:2008-2070`, M state included); a child thread's frame is a page the domain keeps a NONLIN RW copy of (`thread.c:36-61`; CV_THREAD_CREATE pins only). QEMU keeps the caller in its slot and cannot show it; item D's decision 1 (the semantics owner's, 2026-10-17) placed the snapshot in the frame on that assumption. **Ruled: both.** (a) now, the compiler lane in the queue's revision: CSRETIRE the child frame's arena before first entry, page kernel-only, refuse the thread if the retire fails; two-sided run.py check (a write to frame+1408 before the ECALL reads 25 on the fixed module, reaches the ECALL on the pre-fix one). (b) the RTL lane, costed and pre-registered for the next bitstream: the caller snapshot out of any domain-reachable page, and a consuming resume (the switcher clears the 31 slots and their tags after loading them), because the RTL's restore read does not consume and only +224 is zeroed after a reply, so a resumed context's frame keeps tagged LINEAR copies of x1..x31 beside the live registers: a second consequence of a domain-reachable child frame, in the same S-entry. QEMU is brought to the RTL AS IT IS (slots written on escape, +224 zeroed on action 2 only), not to the fixed behaviour. Demonstration first: a va-runv2 variant storing a marker at frame+1408 before its ECALL, the supervisor's sp reading the marker. Tier 1/2 proceed (benign programs); the paper's security claims about the virtual profile on silicon must not outrun the entry (the lead's) |
| **Tier 1 on silicon: PASS** (boot r6, 2026-10-18; lane 2482b9c61c59, pre-registered 452c1611a52e; vp2 bf3db3175, module 0cc4760d21fde17a) | board | b0-stats 0, b0-hello 0, vm-contract 0: contract.dom `ok` printed VIRTUAL_APPLICATION_OK and exited 0, `stale` read 139 with cause 25, the identity branch; the ok run's 24 grants equal QEMU's 24 by script; the Q-30 line holds. **Caveat on every claim that cites it:** the firmware carries a monitor WORKAROUND, capstone-sbi vp2/fetch-retry 007d81d (CAPSTONE_VP2_FETCH_RETRY), for **S-23**: on bf3db3175 an S/U instruction fetch is sometimes refused although a loaded RWX CPMP entry covers it when M looks, always the caller's first fetch after a CSRUNV or CSRETIRE returns; without the retry boots r3-r5 panicked (N = 3); with it, it fired twice in r6 (after a CSRETIRE, before any grant) and never again across hundreds of escapes. The RTL lane root-causes it (candidate: ENTRY's init strobe zeroes the CPMP bank and ESCAPE restores the pc before the CPMPs) and pre-registers the RTL fix for the next bitstream; a workaround is not a fix. **Tier 2 (vm-suite) in the same boot:** normal 0 and threads 1 as pre-registered, then `vm` made no progress for 25 minutes in its second launch, right after `mmap(8193, PROT_READ)` and its read-only first touch; boot r7 runs a DIAG module printing the live PTE on a repeated resolve() fault. ~~The A/D-loop candidate was REFUTED by r8 (no DIAG/-ELOOP line; the module touches no PTE, but the fault is not a loop)~~ **RETRACTED 2026-10-18: the A/D-loop candidate is unresolved, not refuted.** The DIAG/-ELOOP lines are pr_warn, and the baked driver sets `/proc/sys/kernel/printk` to 1 before the first rung (`fpga_driver/run_ladder_perf_fpga.py:201`), so during rungs the console shows only KERN_EMERG, so their absence read nothing. What r8 read: `vm` exits 139 with a store fault (cause 15) after the mprotect to RW at p[8192], by deduction ~~on its first occurrence,~~ through resolve()'s `!same` -EACCES branch (the read-only first touch pinned one page, the write pin after mprotect returned another). The kernel-version candidate (6.1.0 vs 6.4.14 FOLL_FORCE semantics) was WITHDRAWN on the compiler lane's quote: `can_follow_write_pte` is byte-identical in both trees (the PageAnonExclusive rule predates 6.1), and the only FOLL_FORCE gate difference is a hugetlb block that cannot apply. The synthesis under test, no board needed: with svadu=true QEMU's hardware sets A and D, so after the read-only resolution the store needs no fault; silicon never sets them, faults, reaches resolve() and is refused by `!same` (the write pin returning a different page than the read-only first touch). So svadu explains why the store faults only on silicon and `!same` why the fault is terminal. The discriminating run (compiler lane fe8c4283f8ad): the suite on the prediction QEMU with svadu=false, module cb4a8d758fd55d41, one slot: **34/41, identical line for line to svadu=true** (the same 22 VM_EXIT lines; the wrapper's match count makes the no-difference a reading), `vm` exiting 1 at the thread refusal with cause 15 ZERO times, so **the svadu synthesis is refuted by measurement** (~~three~~ two hypotheses dead: ~~the A/D loop,~~ the kernel version, svadu; the A/D loop's refutation was RETRACTED 2026-10-18, above) and what stands is the `!same` deduction from the code's structure, whose mechanism only the first-fault print on a board boot can show (the store passes on QEMU). **Rules from here:** the prediction QEMU runs with svadu=false, free, recorded in `inputs.json` (`platform.qemu` = 3b8040d5ff6e + svadu=false; svadu defaults to true in QEMU, cpu.c:1922, so every earlier prediction ran with hardware A/D); **the one-slot pre-registration for boot 4 is 34/41**, not 39/41: the seven false are threads, vm, heap-threads and their dependent checks threads_output, vm_contract, heap_threads, shared_heap_pages_released, with `vm`'s 1 on QEMU vs 139/15 on silicon the one row judged as a silicon divergence under diagnosis. The next Tier 2 boot carries the module with the first-fault print at a console-visible level (pr_err: the board's console hides KERN_INFO; CORRECTED 2026-10-18: during rungs it shows only KERN_EMERG, so a module line must be read with dmesg in the guest script) and the -ELOOP bound, with sparse-churn cut out as its own LAST rung because it wedges the core.
**r9** (production module ce3a542d00acce86, sparse-churn last): Tier 1 identity, the fourth boot; vm-suite read 13 steps exactly as pre-registered (normal 0, threads 1, vm 139, nested 0, nested-stale 139, heap-threads 1, stale/bounds/retired/vm-ro-write/vm-none-read/vm-guard/vm-padding 139 each); then **sparse-STALE wedged at the same point with the same apertures as r8's sparse-churn**: the domain's commit pc at contract.dom 0x20e98 = free+4, free()'s first store, both sparse modes beginning `*slot = malloc(64); free(*slot)`. One wedge with a two-boot, two-step reproducer. **Corrected brief (compiler lane's disassembly and the project's records):** the domain's commit pc is slot-0 residue (exact only within 8 issues after the last flush_ex), and 0x20e98 is a plain `sd ra, 0x80(sp)` right after free()'s prologue cincoffsetimm (no allocator metadata, no stc; no ecall or CSRUNV anywhere in free() or __capstone_mallocng_free; the only escape in that window is the lock-contention wait; the lifetime retire is an in-context CREVOKE); the image is the v9 contract.dom 125705371837ea92 (bba891bc is mruby). The sparse modes fault on nearly every first touch, so the context is resumed by the module many times; the solid facts are M-side: the trap-log mepc six instructions after the module's CSRUNV, the switcher at index 56 with a register write in flight, a pending STORE. **Candidate for the directed arm:** the module writes the frame (the resolve's bookkeeping, the event words, the reply slot) and issues CSRUNV action 0 within a few instructions with no fence, so the switcher's RESUME walk reads the frame while the kernel's store is still in the store buffer and collides at index 56. The RTL lane builds the arm on that (a trusted-S caller storing to the frame then CSRUNV with 0..3 instructions between, memory latency as the board has it, a context paused at a kind-4 escape; what index 56 is in the RESUME walk); the compiler lane quotes the module's resume path (every frame store before the CSRUNV, the fences). If the arm wedges: a fence before CSRUNV is the cycle-neutral ABI fix now (r11 can carry it), the switcher draining the store buffer the RTL item. **Measured in the built board module (compiler lane):** on the kind-4 RESUME path the module stores NOTHING to the frame before CSRUNV action 0 (the event words are read after the loop; the reply is written only on action 2; resolve() writes a kernel array and two counters; the page-table writes are inside pin_user_pages_remote), so there the pending store at the wedge is GUP's page-table write or a kernel array, and a fence orders PTE visibility, not the frame; on the ENTRY path the module stores frame[0] (satp), frame[1] (the table root) and frame[7] = 3 two instructions before CSRUNV; **no fence at any of the four CSRUNV sites** (two fences in 4,323 instructions, neither on this path; the asm "memory" clobber and local_irq_save's csrrc order nothing; mmap_read_unlock emits none within 500 instructions). The module gains `fence_before_entry` (module_param, default 0; `fence rw,rw` at all four sites when set; afbaff6e029e; verified in the object, fence count 2 → 6) so the hypothesis is two-sided in ONE boot: **r11** runs the suite with the print, then the sparse pair with the fence set (expected to return), then the sparse pair with the fence clear LAST (the expected wedge, the control); the four readings are pre-registered (fence orders it / not the fence / timing moved it / another variable). A wedge that vanishes with the print alone is its own finding, not a fence pass. Ahead of S-23's root cause since S-23 has a workaround and this has none. Method note of record: a mnemonic grep over llvm-objdump is VOID for the .insn-encoded lifetime ops (238 instructions print as <unknown>); decode from the funct3/funct7 tables in CapstoneInstrFormats.td/CapstoneInstrInfo.td. **r10** boots with both sparse steps last and reads preempt, the parallel pair, sqlite and mruby for the first time on silicon; **r11** carries the first-fault print module with the two `vm` branches (equal page pointers = the `!same` deduction wrong; different = the COW mechanism and the re-pin-and-replace fix). **Tier 1 holds on three boots (r6, r7, r8), each with S-23's workaround** (kit record on dev 7b294492ca9f). r8's vm-suite: normal 0, threads 1, nested 0, nested-stale 139/25, heap-threads 1 as the queue's record; `sparse-churn` WEDGED the core mid-step (M, stall, a pending store, the dom switcher at index 56 with a register write in flight; trap-log mepc six instructions after the module's CSRUNV): with the RTL lane, the first real wedge on the virtual path. Also recorded by the board lane: this board's console hides KERN_INFO, so r7's pr_info diagnostic was void (CORRECTED 2026-10-18: the baked driver sets `/proc/sys/kernel/printk` to 1 before the first rung (`fpga_driver/run_ladder_perf_fpga.py:201`), so during rungs the console shows only KERN_EMERG; pr_warn is hidden too) |
**r10/r11: a SECOND wedge signature**, from the first-ever silicon runs of the kit apps: sqlite_write (r10) and mruby (r11) wedge with all 15 apertures equal, M, NOT stalled, LSU idle, the dom switcher at idx 56; the sparse wedge reads M, stalled, a pending store, idx 56. idx 56 is common, the LSU state differs; its own S-entry (board lane), with the RTL lane treating the two as one question with two faces. r11 therefore gave no fence reading (mruby wedged before the sparse pair); **r12** runs the sparse pair alone in the two-sided order (returning steps, fence=1 with read-back, sparse-stale, fence=0 with read-back, sparse-stale, sparse-churn last), module 06d663f3.
**The two faces read from bf3db3175's source (RTL lane):** index 56 is x31 (ids 26..56 = x1..x31) and is the switcher's REST value after a non-reply RESUME (cur_idx is never cleared at finish(); that walk's last restore is id 56, anvil :326-333; ENTRY and a reply resume rest at 31, an ESCAPE at 107); the walk is in flight in none of the wedges (commit_dom_switch_valid and dom_switch_busy read 0 in all three), and the "register write in flight" is the last restore request's enables still on the wires. **So the switcher's handshake is not the candidate and the arm is not built around index 56.** Sparse face (r8/r9): a DYN instruction waiting on the STORE SYNCER (store_syncer_req_set 1, lsu_ready 0, dyn_ready 0, issue stalled), the store unit in WAIT_STORE_READY, the store buffer's commit queue reading count 1 with NO valid bit and no data request: the count disagrees with its valid bits, R-49's ghost-count SHAPE but not its mechanism (R-49 needed a switcher write into a full queue; a RESUME walk only reads); the question is which count path moves commit_status_cnt without a valid entry, or clears a valid bit without the count, on the virtual path; the arm: a DYN op behind stores right after a RESUME returns, with the kit's memory latency. Large-application face (r10/r11): every unit ready, commit valid 1, LSU idle, count 4, M code running without stores; whether 4 is a ghost needs aperture 221, not yet read for those boots (asked of the board lane, with 193/221 after every healthy boot as the control).
**RETRACTION (board lane, dev 9e645777a5fb) that binds every transcript claim:** during board rungs the driver sets printk=1, so the console shows ONLY KERN_EMERG; an absent kernel or module line in a board transcript reads nothing. r8's "no -ELOOP line, so the A/D loop is refuted" is withdrawn: the A/D loop is back beside `!same` for the `vm` fault, and the discriminator is the first-fault print READ FROM THE RING BUFFER (`dmesg | grep capstone_vm:` in the guest script, which the console level does not filter; busybox 1.36.1's dmesg reads the whole 128 KB buffer), with threads' own refusal line in the same read as the positive control (a read without it is void); the fence read-back is a userspace echo beside each step, never through printk; only the banner moves to pr_emerg to show the module loaded. Transcript-based claims from r8-r11 stand only on exit codes and the launcher's own lines. Ahead of S-23's root cause since S-23 has a workaround and this has none. Method note of record: a mnemonic grep over llvm-objdump is VOID for the .insn-encoded lifetime ops (238 instructions print as <unknown>); decode from the funct3/funct7 tables in CapstoneInstrFormats.td/CapstoneInstrInfo.td. **r10** boots with both sparse steps last and reads preempt, the parallel pair, sqlite and mruby for the first time on silicon; **r11** carries the first-fault print module with the two `vm` branches (equal page pointers = the `!same` deduction wrong; different = the COW mechanism and the re-pin-and-replace fix). **Tier 1 holds on three boots (r6, r7, r8), each with S-23's workaround** (kit record on dev 7b294492ca9f). r8's vm-suite: normal 0, threads 1, nested 0, nested-stale 139/25, heap-threads 1 as the queue's record; `sparse-churn` WEDGED the core mid-step (M, stall, a pending store, the dom switcher at index 56 with a register write in flight; trap-log mepc six instructions after the module's CSRUNV): with the RTL lane, the first real wedge on the virtual path. Also recorded by the board lane: this board's console hides KERN_INFO, so r7's pr_info diagnostic was void |
| boot 3-T1, third run (Tier 1 alone: contract.dom on the rebuilt images, the adapter overwriting +224) | board | after the runtime fix, the census at 0/0/0 and the new-pin record; the previous entry's GO text follows: the new-pin record exists (852c5bc76774, new images: ok → 0 with VIRTUAL_APPLICATION_OK, stale → 139 cause 25, zero 24/26, geometry 1 MiB/65,536); b0-stats, b0-hello, vm-contract; single-context; the identity reading is Tier 1's headline; 26 = C-27 not in the staged bytes; 24 from S = an integer-based access the gate says cannot exist |
| run.py's steps on one slot, classified from contract.c | compiler | a vexec process is one context (keyed on mm) and each thread has its own frame, so threads and contexts are one slot problem. Serialize completely (no second live frame): every single-process step, `preempt` (one thread, SIGTERM), and `parallel_one`/`parallel_two` (two independent processes that never synchronize: the second just waits longer). Not available on one slot: `threads` (busy-waits 100 M iterations on the child) and `heap-threads` (joins). The queue refuses at `thread_create` with -EBUSY (never admits a frame it cannot schedule), so both fail immediately and legibly at their `require(tid > 0)`, with one stderr line naming the slot limit. Pre-registered expectation for boot 4: measured on the one-slot QEMU as 34 of 41 (the two refused steps plus their five dependent checks), superseding the 39/41 first stated |
| the QEMU-kit firmware with the injection | board, RTL | the first build (vp2/qemu-kit-inject f5f1bbe, lane only, never handed over) WITHDRAWN 2026-10-18 on claim-auditor's source read: on the kit's QEMU (cap_mem by CAPENTER, scapctl never written) an S-origin trap writes CEPC, not mepc, and mret returns through CEPC (cpu_helper.c:1986-1991, op_helper.c:352-356), so the mepc-based block would re-fault into an INJL spin; a dropped `return 0` in swap_cpmp; the classic-domain branch unreachable on QEMU. Define-off byte-identity had no define-on reference. **Rule:** ONE CEPC-based `_inject_s_trap` in the shared sbi_capstone.S with no target #ifdef (the RTL confirms CCSRRW CEPC reads/writes mepc's cursor, csr_regfile.sv:2616-2621, so the S-visible state is identical on silicon), FPGA and QEMU byte-paired again; proven by va-osinject on bf3db3175 (silicon half) and the kit's `threads` step on QEMU reading INJS 0x1d, mret to stvec and the kernel's trap report (QEMU half, f5f1bbe as the spin control), both before handover. The CEPC/mepc divergence is filed as a low-priority Q-entry |
| boot 4 (Tier 2: vm-suite) | board | after the shared CEPC-based injection entry passes both halves, the kit firmware is handed over and the matrix re-run on it; the serialized module with the step expectations pre-registered (threads/heap-threads refused at thread_create); and the C-27 word for the landing |
| pinned kit + identity + geometry | compiler | geometry rule in; identity with the flat module's revision |
| Tier 3 payload budget (26 MB declared) | board | accepted; lands on dev after vp1's second preflight, with both negative tests |
| 26 bundles rebuilt and re-predicted on the pin | compiler | to do; 20/26 read on the unpinned kit |
| invocation manifest | compiler | asked 2026-10-17 evening |
| Tier 3 board backend + boot plan | board | asked 2026-10-17 evening; the comparison rule above to confirm first |
| Tier 3 fold tooling | helper | to do |
| Sublet (iii) placement repairs ×3 | compiler | to do |
| per-port SDK fix (cpython ×4, postgres/sql ×2) | compiler | root-caused, to do |
