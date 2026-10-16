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
- C', exact bounds as decided below;
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
      - `capstone_vm.ko` and `capstone-vexec` print `VIRTUAL_APPLICATION_OK` for `contract.dom`, with `scapctl` bit 8
        (exact bounds) set;
      - a stale access is denied with cause 25 (exit 139).
   3. **The runtime's `run.py` gate.** It needs the SQLite and mruby images.
   4. **The bug corpus's virtual arms.** The selection is every verdicts-v1 bundle whose configuration's target is
      `capstone-virtual`: 26 bundles, 382 cells and 304 distinct images at 44d82a9eb3f3. Each cell reads the verdict
      recorded for it on QEMU, taken from the bundles by that rule rather than from a copied list. Only 91 of the 304
      images survive. They cover seven bundles: FFmpeg carved (both arms), plain-heap and subobject; memcached
      plain-heap; Wireshark plain-heap and plain-temporal. They are copied, with a SHA256SUMS, to
      `/mnt/hdd/<user>/capstone/virtual-corpus-images-2026-10-16` on apollo, and run as built. The other 213 were not
      found: the search stopped at 15 minutes, and many were built on a-p13, whose setup is gone. Each of those is
      rebuilt from its bundle's recorded inputs with the current virtual SDK, and its prediction is that rebuild's
      reading on QEMU's virtual profile.
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

## The decision that gates qualification: exact bounds (audit item C')

The full profile needs exact bounds. ABI v5 and local mallocng run only under `scapctl` bit 8 (`CPONVVM5`), and the
launcher runs nothing without it. The audit's option (iii), representable bounds enforced by the allocator, changes
the runtime rather than the hardware, so it is not the full profile. That leaves:
- **(i)** a wider capability format: two granules per capability;
- **(ii)** a side store of 24 bytes per capability-bearing granule.

**Decided 2026-10-16.** The lead delegated the decision to the lanes, and it rests on the RTL lane's costed comparison
[`2026-10-16-exact-bounds-options.md`](2026-10-16-exact-bounds-options.md) (955851735f4f) and the compiler lane's
measurements:
- **(ii), the side store.** It is maintained by hardware alone and authoritative under the virtual-context flag. Each
  granule has a 16-byte {base, end} entry, with an L1 bounds array beside the tag array.
  - Both options cost the same in bitstream cycles (3) and in traffic: 5 transactions per capability access against 3
    today.
  - (i) would also double `sizeof(void*)`. The compiler lane measured 27 instructions per intcap add against 6, and
    3-5 weeks to audit the ports' 16-byte assumptions. That amounts to re-porting the whole virtual stack.
  - (ii) leaves the compiler, the runtime and every acceptance image unchanged, and it is the reference QEMU's model,
    whose store also keeps the cursor (24 bytes per granule).
- **Dense provisioning, not a per-page directory.** A directory adds a second dependent walk on the path that carries
  the WNS. The cost of dense is RAM:
  - the board is a Genesys2 with 1 GiB of DDR (`ariane_soc_pkg.sv:50`);
  - a 1:1 bounds region leaves about 0.5 GiB for data, beside the tag region and the rev-node pool.
  The RTL lane writes down the memory map, and the board lane confirms that the acceptance set fits it. If it does
  not, the directory is reopened.
- **The format freeze, as the RTL lane proposed:**
  - the in-memory word as both sides have it today: {revnode_id 30 | perm 4 | cap_type 3 | bounds 27};
  - 27 bounds bits;
  - 30 node bits, which mean a flat index under the virtual-context flag;
  - no generation bits in the word on either profile;
  - the RTL's compress_bounds with renormalisation as the reference encoder until QEMU lands Q-16's block. After
    that the two are proven bit-identical by the shared model;
  - the exact pair {base, end} without the cursor;
  - QEMU's 16-byte record as the virtual table's layout.
- **Owner of the model's semantics: the helper lane**, which owns capstone-qemu's `virtual-capstone` branch and the
  guide. A semantic change lands in QEMU, the guide and a Q-entry before the RTL follows. The RTL lane owns
  conformance: the Q-entries and the directed tests that read each one on both sides.

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

## The 1 GiB memory map with a dense bounds region (the first C' item, RTL lane, 2026-10-16)

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

## The bitstream sequence (RTL lane's proposal, 2026-10-16)

Batched so that each bitstream answers questions the previous one cannot, and no reflash is spent on an observation
the simulation already gives:
1. **B + Q-17**: the virtual-context flag with translation and the resumable page fault (in simulation on
   `virtual-profile` now), the CDERIVE check-order fix (one Anvil reorder), and E's untagged-arithmetic rule (#25)
   if its flu change lints at baseline. Reads on the board: the six vb-trans arms as bare JTAG arms, then Tier 1's
   launcher smoke without exact bounds (the launcher refuses, which is itself the reading until C').
2. **C' + A flat**: the common core widening (exact bounds in the register lanes), the side store with its L1
   array and adapter leg, the new memory map, and the srevroot table in its flat form with the flat 30-bit id.
   Reads: exact-bounds.sh's four cases as arms, Tier 1 with the launcher running, the first virtual runtime suite.
3. **D + A paged + E + H + Q-18**: CSRUNV with its frame, actions and kinds, FP state per context, the paged table,
   the policy list, the collection flush, and DROP's virtual rule. Reads: Tier 1 and Tier 2 in full.
4. **Integration**: whatever 1 to 3 leave open, and Tier 3's corpus. The physical regression (S-16, the Sublet arms,
   one corpus set) is read on every one of the four.
Every bitstream is pre-registered before its handoff and carries its lint and loop-name numbers; every reflash
waits for the lead's own word.

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
| C' exact bounds: the shared core widening, then the side store and the L1 bounds array; first the memory map for 1 GiB | RTL lane | to do |
| B translation in C mode | RTL lane | IN SIMULATION 2026-10-16: capstone-ariane branch `virtual-profile` 31c8ebe17 (the flag csupctl.virt, the two translation enables, the MMU permission privilege, the consuming-LDC store check; vb-trans.S with six pre-registered arms and its flag-off control); lint by name, the two arms, the old-tree control and the flag-off sweep are the gates before any synthesis request |
| A lifetime table | RTL lane | to do |
| D `CSRUNV`, `CSMINT`/`CSRETIRE` | RTL lane | to do |
| E policy list | RTL lane | to do, folded into D |
| board acceptance (done-means 3, 4) | board lane | after the first virtual bitstream |
