# The full virtual Capstone profile on silicon: the RTL lane's task (2026-10-16)

**The lead's decision, 2026-10-16:** silicon is to run the **full** virtual Capstone profile. The landed virtual stack
is to run on the board as it runs on QEMU's virtual profile: ABI v5, local mallocng and the virtual application ports.
Applications are ported after it lands, not before: the lead paused the silicon-ABI ports of the board campaign's
wave 5 the same day. This answers the first of the three questions in the RTL lane's audit,
[`virtual-capstone-rtl-feasibility.md`](virtual-capstone-rtl-feasibility.md) (3fe4541c1299, 2026-10-10). That
audit's verdict is "partly, and not as a port". Its item list (A-H) and its order of work are the starting point here.

**The target, as the lead put it on 2026-10-16: the final, fully virtual Capstone/Sublet bitstream.** It is reached
through intermediate bitstreams, each synthesised and then flashed on the lead's word. It carries:
- B, translation in C mode;
- A, the per-context lifetime table, flat and paged;
- C', bounds as decided below: representable bounds that the allocator enforces (option (iii)). The RTL keeps its
  compressed format, so there is no side store;
- D, `CSRUNV`, `CSMINT`/`CSRETIRE` and the event kinds;
- E, the policy list;
- H, the collection flush;
- the Sublet instructions as the virtual profile defines them (Q-17, Q-18).
The physical profile stays underneath. Removing it would cost work, not save it, and it carries every recorded
silicon result.

Today's bitstream, `caplifive_sublet_d93a8d66b`, implements the physical profile plus CDERIVE/CREVOKE. Every silicon
result so far is a physical-profile result: the bug corpus was rebuilt for the silicon ABI, not run from its virtual
images ([`2026-10-15-bug-corpus-on-fpga.md`](2026-10-15-bug-corpus-on-fpga.md) §1).

## What done means

1. **One bitstream, both profiles.** A virtual-context flag selects the virtual profile per context, as the audit's
   item B proposes. The physical profile stays, because the board campaign
   ([`2026-10-16-board-campaign-all.md`](2026-10-16-board-campaign-all.md)) and every recorded silicon result run on it.
2. **The reference is QEMU.** capstone-qemu `virtual-capstone`, at d35af47b163c or its head when work starts, with the
   guide `docs/design/virtual-capstone/*`. Where the RTL and QEMU disagree, the disagreement goes into an ISSUES
   Q-entry; it is never settled silently. The first is Q-16 (fa38c2520b85).
3. **Acceptance, in this order:**
   1. **In the RTL harness:** the directed tests ported from capstone-qemu's own suites at the reference:
      - `tests/virtual-capstone-model`;
      - `tests/sublet-lifetimes`;
      - `tests/virtual-capstone-m1`;
      - `tests/virtual-capstone-runtime` and its `exact-bounds.sh`;
      - `tests/capstone-x0-writes` and `tests/capstone-untagged-arith`;
      - dev's `runtime/tests/sublet-lifetimes`.
      Their counts are re-measured on the pin before porting.
   2. **On the board, the guide's Linux smoke** (`docs/design/virtual-capstone/development.md`):
      - `capstone_vm.ko` and `capstone-vexec` print `VIRTUAL_APPLICATION_OK` for `contract.dom`, under the option-(iii)
        runtime (no exact-bounds bit);
      - a stale access is denied with cause 25 (exit 139).
   3. **The runtime's `run.py` gate.** It needs the SQLite and mruby images.
   4. **The bug corpus's virtual arms.** The selection is every verdicts-v1 bundle whose configuration's target is
      `capstone-virtual`: 26 bundles, 382 cells and 304 distinct images at 44d82a9eb3f3. Each cell reads the verdict
      recorded for it on QEMU, taken from the bundles by that rule rather than from a copied list. Under option (iii)
      every image is rebuilt from its bundle's recorded inputs with the option-(iii) virtual SDK. Today's images
      require the exact-bounds bit, and the launcher refuses to run them without it. Each rebuild's reading on QEMU's
      virtual profile, with exact bounds off, is its prediction. A verdict that moves between the recorded bundle and
      the rebuild is a measured cost of (iii), not a silicon finding. 91 of the 304 old images survive, copied with a
      SHA256SUMS to `/mnt/hdd/<user>/capstone/virtual-corpus-images-2026-10-16` on apollo, as references.
   This is the payoff: the corpus runs on silicon as built for the virtual profile, not rebuilt for the silicon ABI.
4. **The physical regression holds:** the S-16 bare set, the Sublet directed arms and one corpus set read the same on
   the new bitstream as on d93a8d66b.

## First deliverable: re-baseline the audit

The audit read the virtual spec as of 2026-10-07/08 and QEMU 1a6dd20732. Since then:
- **capstone-qemu `virtual-capstone`:**
  - #21, Sublet lifetimes (CDERIVE/CREVOKE in the virtual profile, and the stale-reload rule: a reloaded stale
    non-linear capability keeps its tag in the virtual profile only);
  - #22, #23, #26 and #29, performance;
  - #25, untagged capability arithmetic;
  - #28, x0 capability writes;
  - #30, floating point in virtual C;
  - acbad0f7ec, CDERIVE/CREVOKE decoded on the physical profile;
  - #27, MOVC with an untagged source (Q-04).
- **dev:**
  - 6576dca09c37: virtual mallocng manages lifetimes with CDERIVE/CREVOKE;
  - 601c8d52c260, cbbc2ca958af and b02ddf9043cc;
  - 7901153f10fa: the superseded allocator backends retired.

Write an addendum to the audit that says, item by item, what changed. Include where d93a8d66b's CDERIVE/CREVOKE
differ from the virtual profile's. The known case is rounding: silicon rounds an inexact CDERIVE's base down.

## Bounds (audit item C'): option (iii), representable bounds that the allocator enforces

**HELD 2026-10-16, pending a re-cost:** the cost below was RETRACTED by its author the same day; see the next paragraph. Neither (iii) nor (ii) is being built meanwhile. **Decided 2026-10-16, superseding the same day's choice of (ii).** The lead delegated the choice to the lanes, then
asked whether exact bounds are needed at all. The answer is that the hardware does not need them; only the landed
runtime's choices did. The guide states that dependency: "Local mallocng requires this profile to retain upstream
slot geometry and expose only requested object bounds". QEMU marks its shadow store as a prototype: `cap_mem_map.h:20`
"we hack this to prevent precision loss for now", and 24b1e95b1dac "does not specify an RTL encoding".

**RETRACTED 2026-10-16 by its author, the compiler lane: "no size-class, slot or group layout, or alignment change"
is wrong.** The estimate checked only the length. Representability also needs the BASE to be a multiple of the
granule. modcapstone capstone.c:338 says so ("both its length and its base are granule multiples"), and the RTL
enforces it in its own way.
- At d93a8d66b, `compress_bounds` (`ariane_pkg.sv:777-833`) truncates the base to the granule and rounds only the end
  up. An unaligned base therefore silently gains authority BELOW the object: up to 240 bytes for the largest class,
  past the in-band header and into the previous slot.
- QEMU's physical profile refuses the same derive with 29 instead (b6-sublet.c:19).
- In mallocng's real geometry, `slot_base = group_base + 16 + i*stride` (heap.c), so 12 of the 20 classes with a
  stride of at least 4096 have an unrepresentable base for their first slot.
- Rounding the base down and the length up fits inside the slot for only 8 of those 20.
- Only granule-aligned slot offsets are usable, so mallocng's double-free offset entropy falls to 1 in 2 (granule 32)
  through 1 in 16 (granule 256) for those classes.
So (iii) needs the group header and the strides of the large classes to be granule-aligned: a size-class and layout
change, which is what the estimate said would not be needed.

The hope that this collapses is also settled. The author asked whether the cursorless scheme keeps a freshly derived
child exact. It does not exist on the bitstream: 4a4261bb4 removed it ("one scheme for every capability",
`ariane_pkg.sv:778` at d93a8d66b). The author read the main checkout's older capstone-ariane, 062f4453b. The same
author also counts 430 CAUGHT rows among the corpus's 825 judged virtual-arm rows that could flip under (iii).
`fault.offset` is a pc offset, not an overrun distance, so the recorded rows cannot settle it; Tier 3's re-run does.

The original estimate follows, kept for the record:

**What (iii) costs, as first estimated.** The compiler lane computed it (`lane/compiler-exact-bounds-iii` 91d5b48e4c23,
`docs/plans/2026-10-16-exact-bounds-option-iii-cost.md`): 3-5 days in the runtime, and nothing in the RTL.
- mallocng changes in two expressions: it classifies on `repr_round(n)` and sizes the big-path mapping from it. ~~Its
  size classes, slot and group layout, and alignment do not change.~~ (retracted above)
- 628 of 131,051 small sizes (0.48%) move to a larger class. These are the sizes in the last 16 bytes below a class's
  capacity, and no power of two or page multiple is among them. The worst is n = 13073, 25.1% more. 107 sizes near
  the mmap threshold move to the mmap path. No distribution of application sizes exists, so these figures are over
  the size space and not weighted by use.
- The CSMINT descriptors the launcher and the image path build must be representable. probe.S builds one that is not
  today. That audit, 1-2 days, is the largest item.
- Mechanically: the ABI's exact-bounds requirement statements, the launcher's profile gate (exec.c:697-699, probe.S),
  seven runners, and the documents.

**What (iii) gives up, stated rather than hidden:**
- Above 4 KiB, a bound rounds outward to the granule: 8 bytes at 4 KiB, 128 at 64 KiB, at most 1/512 of the size.
  An overflow into that slack is not trapped. It cannot reach a neighbour: the allocator keeps the rounded bound
  inside the slot.
- The slack can span mallocng's own footer bytes for that slot, so the slot's in-band footer becomes writable through
  the user pointer.
- One recorded verdict flips: allocator.json `bounds`. malloc-contract.c stores at `saved+200003` for
  `malloc(200003)`, which rounds to 200192. The test needs a trigger past `repr_round(n)`.
- exact-bounds.sh changes meaning: its profile-off arms become the production configuration.
- On the bug corpus, rounding touched one silicon reading of 130 (memcached 08, 16 extra bytes).
- The four ports' recorded functional verdicts do not change, because a wider bound can only remove a fault.

**What it saves:**
- the side store and its L1 bounds array;
- the shared core widening;
- the 496 MiB memory map, so the map stays at 963 MiB of data and the CMA and monitor conditions for moving it lapse;
- about 1.9x capability traffic;
- one to two bitstream cycles.

**Still in force from the 33cd17910a37 decision:**
- the format freeze (the word as it is, 27 bounds bits, 30 node bits as a flat index under the flag, no generation
  bits in the word). The exact pair is no longer stored;
- the owner of the model's semantics: the helper lane, with the RTL lane owning conformance.

**Owners:**
- the compiler lane does the runtime change on its lane branch, re-records what it moves, and the re-run;
- the semantics owner states (iii) in the guide and in QEMU's default profile;
- the RTL lane drops C' from the sequence.

The superseded record (the costed (i) against (ii), and the dense map with its RAM budget) stays below and in
`2026-10-16-exact-bounds-options.md` as the evidence the choice was weighed.

## Work that does not wait for the decision

These items follow the audit's order. Each is proven in simulation, against directed tests ported from QEMU's virtual
suites, before it goes to synthesis:
- **B, translation in C mode.** This is the largest item and needs no format decision:
  - the virtual-context flag;
  - translation under M privilege with USER permissions, for data and for fetch;
  - page faults as a resumable escape kind;
  - the consuming LDC's same-granule rule;
  - an audit of every `PRIV_LVL_M` assumption.
  The audit plans one bitstream cycle for it, with a bare test before any runtime.
- **A, the lifetime table:**
  - an `srevroot` register;
  - the 16-byte record, the reserved ids and the header;
  - the flat format first, the paged format second;
  - the generation path removed, and replaced by the collection flush (H).
- **D, the `CSRUNV` front end:**
  - frame addressing and actions 0-3;
  - the page-fault, node-pressure and service-ECALL kinds;
  - `CSMINT`/`CSRETIRE` over the new table.
  The spec leaves context storage for 32 contexts open.
- **E, the policy list,** with the relaxation of SPLIT/MREV under supervision. It folds into D.

The audit estimated four to five bitstream cycles, plus the format decision. Options (i) and (ii) add to that.

## SUPERSEDED by option (iii): the 1 GiB memory map with a dense bounds region (RTL lane, 2026-10-16)

Today (every bitstream since April, `ariane_pkg.sv:602-608`, `wt_axi_adapter.sv:161-162`): data `[0x8000_0000,
0xBC2D_2D2D)` = 963 MiB; the tag region at one byte per 16-byte granule, `tag = 0xBC2D_2D2D + ((paddr - 0x8000_0000)
>> 4)`, which ends exactly at `0xBFF0_0000`; the rev-node pool `[0xBFF0_0000, 0xC000_0000)` = 1 MiB (65,536 records
x 16 B). The Genesys2 DDR is 1 GiB at `0x8000_0000`.

A dense {base 64 | end 64} pair per granule is a region the size of the data it covers, so with D bytes of data the
regions sum to D + D/16 + D + 1 MiB = 1 GiB, D = 496 MiB. Proposed map, every boundary a multiple of 1 MiB and the
pool unmoved:

| region | range | size | addressing |
|---|---|---|---|
| data (firmware, Linux, initramfs, the virtual tables at srevroot, every image) | `[0x8000_0000, 0x9F00_0000)` | 496 MiB | as today |
| tags (1 B per granule) | `[0x9F00_0000, 0xA0F0_0000)` | 31 MiB | `0x9F00_0000 + ((paddr - 0x8000_0000) >> 4)` |
| bounds (16 B per granule, authoritative under the virtual-context flag) | `[0xA0F0_0000, 0xBFF0_0000)` | 496 MiB | `0xA0F0_0000 + (paddr - 0x8000_0000)` |
| rev-node pool (the physical profile's table) | `[0xBFF0_0000, 0xC000_0000)` | 1 MiB | unchanged |

The sum is `0x1F00_0000 + 0x1F0_0000 + 0x1F00_0000 + 0x10_0000 = 0x4000_0000`. `CAP_TAG_MEM_BASE` moves from
`0xBC2D_2D2D` to `0x9F00_0000` and `MEMORY_TOP` with it; a new `CAP_BOUNDS_MEM_BASE = 0xA0F0_0000` joins them, and
the three are the only constants the adapter needs. Costs to state plainly: the physical profile on the final
bitstream also sees 496 MiB of data, not 963, because one bitstream carries both profiles; firmware, the device tree
and the test harness's memory size follow the new top. **The board lane confirms the acceptance set fits 496 MiB**
(the current initramfs is ~28 MB; the virtual tables grow with the number of live nodes); if it does not, the
directory comes back and costs a dependent walk on the WNS path. The bounds region is written only by capability
stores and read only by capability loads, so an image that never stores a capability into a page never touches
its slice; dense provisioning costs RAM, not traffic.

**The board lane's answer, 2026-10-16: 496 MiB holds the acceptance set, with about 250 MiB to spare,** so dense stands.
The figures, measured unless marked:
- today Linux gets 962.8 MiB (DT memory@80000000, 0x3c2d2000). The kernel, firmware and memblock take 30-34 MiB,
  and CMA is 256 MiB at 0xAC000000;
- the built-in initramfs stays resident: 5.3 MB of rootfs plus at most about 9 MB of staged images in any corpus set
  (set 9: 8.8 MB);
- the largest physical region is one 128 MiB CMA region: set 9's pymalloc and set 4's memcached;
- virtual-profile peaks per process (the 2026-10-11 memory-behaviour READMEs): Wireshark about 54 MB, CPython about
  33.7 MB, Perl 29.1 MB, mruby 28.4 MB, PostgreSQL 27.3 MiB, memcached 14.3 MB, SQLite 11.7 MiB, FFmpeg 7.3 MB;
- the budget: about 34 MiB reserved, 15-20 MiB initramfs, about 20 MiB kernel runtime (an estimate), and CMA of at
  least 160 MiB. CMA is reusable, so virtual pages and a physical region overlap rather than add. The peak is about
  230-240 MiB.

Still open:
- the peak RSS of the corpus's own virtual PostgreSQL server (postgres/sql-repros) and CPython interpreter
  (cpython/allocator-boundary-repros). It is measured when Tier 3 rebuilds those images on QEMU's virtual profile,
  and the margin holds if each is within 2x of its program's figure above;
- the idle kernel figure, which the board lane reads from /proc/meminfo on its next classic boot.

Conditions for the flash that moves the map:
- CMA moves inside `[0x8000_0000, 0x9F00_0000)`, 4 MiB aligned, as the DTS requires;
- the exact-regions monitor's bound on Linux's region (0xC0000000, R-55) follows the new top;
- the monitor's boot-carve replay is re-run on the new map before the flash;
- the monitor programs an all-permitting PMP entry before the first virtual context, because the page-table
  walker's accesses are PMP-checked as supervisor (the RTL lane's B simulation).

## The bitstream sequence (RTL lane, re-planned 2026-10-16 after option (iii))

With exact bounds dropped there is no C' bitstream, no side store, no core widening, and the map stays at 963 MiB.
Three bitstreams, batched so each answers what the simulation cannot:
1. **B + Q-17 + R-56 + Q-19**: the virtual-context flag with translation and the resumable page fault, the CDERIVE
   check order, the dropped page fault on a translated capability load, and the consuming-ldc rule as ruled. Reads on
   the board: vb-trans's seven arms and its flag-off control, Q-17's arm, the S-16 set, the Sublet arms and one corpus
   set identical to d93a8d66b. With the option-(iii) runtime the launcher no longer refuses for want of scapctl bit 8,
   but its first full run needs A and D, so bitstream 1's Tier 1 reading is the bare arms plus the launcher's refusal.
2. **A flat + the CSRUNV front end of D**: the srevroot register, QEMU's 16-byte record with the flat 30-bit id, the
   reserved ids and the header (the generation path leaves the id under the flag); CSRUNV's frame addressing, actions
   0-3 and the page-fault, node-pressure and service-ECALL kinds over the existing switcher. Reads: Tier 1's launcher
   smoke (contract.dom reading VIRTUAL_APPLICATION_OK and the stale access denied 25), the first virtual runtime suite.
3. **A paged + the rest of D + E + H + Q-18**: the paged table, FP state per context (#30), CSMINT/CSRETIRE over the
   new table, the policy list with #25's untagged-arithmetic rule, the collection flush, and DROP's virtual rule.
   Reads: Tier 1 and Tier 2 in full, then Tier 3's corpus; integration of whatever 1 and 2 leave open.
Every bitstream is pre-registered before its handoff and carries its lint and loop-name numbers; every reflash
waits for the lead's own word; the physical regression is read on every one.

## Constraints

- **No reflash without the lead's own word.** The board campaign is running on d93a8d66b and a reflash interrupts it,
  so agree the timing with the board lane.
- **CLAUDE.md "Simulation green is not synthesis clean" applies in full:**
  - the lint gate and claim-auditor on every change;
  - a predicted reading written down before synthesis, and used as the go/no-go;
  - every change a bitstream needs batched into it;
  - no change to the synthesis flow;
  - a hash handed on only with its lint and synthesis numbers.
  A and B both reach the rev-node, LSU and D-cache cones that carry the loops and the WNS. They are the high-risk
  class that rule names.
- **No paper edits.**

## Status

| item | owner | state |
|---|---|---|
| re-baseline addendum | RTL lane | DONE 2026-10-16: `virtual-capstone-rtl-feasibility.md` "Addendum, 2026-10-16"; Q-17 and Q-18 filed; reference pinned at d35af47b163c |
| exact-bounds options, costed (i) vs (ii), format-freeze proposal | RTL lane | DONE 2026-10-16: `2026-10-16-exact-bounds-options.md`. DECIDED the same day: (ii) dense, the format frozen as proposed, the helper lane owning the semantics |
| C' bounds | RTL lane | DROPPED 2026-10-16 for option (iii), which is now HELD pending a re-cost (the first estimate was retracted). The RTL lane does not restart C' until the decision is re-made |
| option (iii) in the runtime: mallocng `repr_round`, the CSMINT descriptor audit, the ABI and launcher gate, runners, documents, re-records | compiler lane | HELD: the 3-5-day estimate was retracted by its author. A re-cost with the layout change (granule-aligned group header and large-class strides, the memory cost, the offset-entropy loss) comes first |
| B translation in C mode | RTL lane | SIMULATION GREEN 2026-10-16: capstone-ariane `virtual-profile` ea88d8e5d (the Q-19 ruling implemented: the consuming ldc decides at writeback from the captured page permission, a non-consuming ldc completes, arm 6 added); seven arms and the flag-off control as pre-registered; lint at baseline, the 40 loop names identical; Q-17's reorder and arm joining, then the flag-off neutrality sweep and the sup lists, then the synthesis request with the pre-registered prediction |
| A lifetime table | RTL lane | to do |
| D `CSRUNV`, `CSMINT`/`CSRETIRE` | RTL lane | to do |
| E policy list | RTL lane | to do, folded into D |
| board acceptance (done-means 3, 4) | board lane | after the first virtual bitstream |
