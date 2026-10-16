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

**RE-DECIDED 2026-10-16 on the corrected cost: (iii) stands.** The first estimate was retracted (below). The re-cost with the layout change does not turn the choice. **Decided 2026-10-16, superseding the same day's choice of (ii).** The lead delegated the choice to the lanes, then
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

**The re-cost** (the compiler lane, `lane/compiler-exact-bounds-iii` 3d7f5128a961) was measured against the RTL's own
encoder model, `verif/tests/custom/capstone/bounds-model.py` at capstone-ariane d93a8d66b, loaded by revision. Its
self-check reproduces R-54's 1,050 of 400,000.
- 7 of the 48 size-class entries change, by 1-4 units, mostly to round numbers: [36] 1170, [38] 1638, [39] 2048,
  [41] 2732, [43] 4096, [45] 5464, [47] 8192. Slot space grows by 0.022%.
- Each group's header grows from 16 bytes to the class's granule (32-256), once per group.
- Offset cycling keeps only granule-aligned offsets. Mallocng's double-free entropy is untouched on 8 of the 20 large
  classes and falls to 1 in 2 through 1 in 16 on 12.
- 0 of 254,304 (class, request, offset) combinations escape their slot.
- The table edit mis-sizes 0 of 131,051 requests. Shrinking [43] by 8 units, the positive control, mis-sizes 112.
- The effort is 4-6 days.

**One requirement the re-cost's containment check does not cover, found by the semantics owner with the same model.**
Mallocng's in-band header for slot i+1 is the last 4 bytes of slot i: `p[-4..-1]` with `p = storage + stride*(i+1)`.
Over the repaired geometry (offset 0, slots 0 and 1, every request each large class serves):
- no request leaves its slot;
- but the rounded end can cover those 4 bytes, so an object's capability would reach its neighbour's allocator
  header, which exact bounds never do.
- **CORRECTED the same day by the compiler lane** (`lane/compiler-exact-bounds-iii` 2dd6f5f2cb92). The figure first
  given here, 912 near-capacity sizes, was measured at offset 0 and one fixed base, which is too narrow a view.
  - Offset cycling moves the object to the end of its slot. Over the lowest and highest surviving offsets, 94,088 of
    252,736 (request, offset) combinations in the affected classes reach the next header without the fix, and 0 with
    it.
  - ~~The promotion count is 1,024 of 131,051, not 912.~~ **Withdrawn by its author the same day**
    (`lane/compiler-exact-bounds-iii` 30d6ddfc5274). 1,024 is the worst case over all bases, but the repaired geometry
    puts every slot on a granule-aligned base. With both edges granule multiples, R-54's renormalisation cannot fire:
    `tg - bg` is `len/g`, which is below 1024. So the base-dependent case never occurs. No class in the repaired table renormalises. **The count, CORRECTED
    2026-10-16 by the semantics owner's audit** (`capstone/tests/exact-bounds-iii-audit/`): "912" counted, against the
    edited table, 17 sizes at or above `MMAP_THRESHOLD` (131052), which never use a slot. Below the threshold, 895
    sizes reach the next header against the edited table. Measured against upstream behaviour, the figure that
    matters, **735 of 131,051 requests (0.56%) end in a different class than upstream, 107 of them on the mmap
    path**. Some of the 895 sit in the 16-byte band the table edit adds and return to their upstream class.
  - The arithmetic now lives in one header, `runtime/include/capstone/representable.h` (`capstone_repr_ok`,
    `capstone_repr_round` and two more), on that branch. `repr-cross-check.py` compiles it and compares it with the RTL
    oracle over 24,168 cases, with 0 disagreements. A mutant that ignores the base disagrees 46 times.
  - Slots 0, 1 and 31 read 0 escapes and 0 header hits with the fix.
So the runtime keeps the first estimate's classification rule as well: classify on `repr_round(n)`, with the
granule its slot base gives, against the class's usable size `stride - IB`. The rounded end then stays at or below
the next slot's header. Against upstream, 735 requests change class, 107 of them onto the mmap path.

**The cost omits the nested allocators: CORRECTED 2026-10-16 by the semantics owner.** QEMU's compact profile, the
reference once exact bounds are off, refuses an inexact CDERIVE child with 29 (`op_helper.c:1239-1247` at d35af47b163c:
"Never silently widen a new object's authority in the compact profile"). ISSUES Q-15 records that the allocators
migrated to CDERIVE do not handle this, because ABI v5 required exact bounds. Under (iii) every virtual port's Sublet
patch that derives children of 4 KiB or more must place them representably, as mallocng now does. Candidates:
- FFmpeg's pools (0003, frame buffers);
- memcached (0006, slab chunks up to the item size);
- Wireshark's wmem jumbo blocks (0001);
- PostgreSQL's large chunks (0003);
- SQLite's memsys5 blocks.
~~pymalloc, mruby's GC slots and Perl's SV heads derive only small children.~~ **CORRECTED the same day by the compiler
lane** (`lane/compiler-exact-bounds-iii` ac7898060694, probe `capstone/tests/exact-bounds-iii/mrbpage.c`). The
per-object children are small in all three, but each also derives a parent lifetime:
- **mruby's page derive is refused under (iii).** Patch 0008 derives the page's `objects` array, at least 16 KiB, at
  offset 16464 (compiled with the capability toolchain). That is 16 mod 32 against a granule of at least 32,
  whatever `sizeof(RVALUE)` is. The patch's own `issued[]` array moved it there. The repair is padding plus a
  `_Static_assert`, under 1% of a page.
- pymalloc's arena derive is 1 MiB at offset 0 on an mmap base, so it is representable and needs no change.
- Perl's arena is `PERL_ARENA_SIZE` 4080, below 4096, so it is exact at any base. It enters scope if raised to
  4096.

The audit is bounded: a child shorter than 4096 bytes is exact at any base and any offset, so only derive sites that
can produce a child of 4 KiB or more need checking, which in practice means the parent-lifetime derives. The compiler
lane re-costed these sites port by port; the Tier 3 nested arms depend on them.

**The ports, re-costed 2026-10-16; the comparison with (ii) does not reopen** (`lane/compiler-exact-bounds-iii`
9cd5fb63cfb1, sites extracted from the patches by `capstone/tests/exact-bounds-iii/derives.py`):
- Twenty derive sites across FFmpeg 0003, memcached 0006, wmem 0001, PostgreSQL 0003 and SQLite memsys5.
- At the eleven offset-0 sites, the site ALLOCATES `repr_round(size)` and derives over it. CDERIVE checks
  containment against the parent's exact bounds (`capstone_dyn_unit.anvil:250`), so the parent has to hold the
  rounded length. The cost is at most 1/512 of the size: a measured maximum of 0.1953% over 4 KiB-4 MiB, and zero
  for powers of two and page multiples.
- At the nine non-zero-offset sites, the offset must also be granule-aligned. The fixes are a startup assertion for
  memsys5's atom stride, the mruby-style placement of a sub-object, and an assertion where `cursor - base` is zero in
  practice.
- No site has a caller-controlled length with no ceiling at an offset the port does not choose.
- The total for (iii) is 6-9 days of software.

**The size_classes lock: option (a) chosen, 2026-10-16, after an independent audit.** The lead asked the semantics
owner to audit and choose. `ports/musl-capstone/check-mallocng-policy.py`, the port's guard (adc891bb920c), requires
the `size_classes` tables and the decision functions to be token-identical to musl 1.2.5. That forbids the 7 stride
edits. Three ways out were measured with `capstone/tests/exact-bounds-iii-audit/audit-abc.py`:
- musl 1.2.5's own `size_to_class` (archive sha256 a9a118bb..., the guard's);
- the RTL's `bounds-model.py` read from capstone-ariane d93a8d66b by revision;
- every request n in [1, MMAP_THRESHOLD), against upstream as the baseline.

| | requests whose class differs from upstream | of them onto mmap | slot bytes vs upstream | worst request |
|---|---|---|---|---|
| (a) edit the 7 entries, narrow the guard | 735 (0.56%) | 107 | +0.110% | +25.1% |
| (b) keep the table, skip the 7 classes whose stride is not a granule multiple | 71,752 (54.75%) | 26,347 | +2.129% | +71.6% |
| (b') keep the table, shift each object to a granule-aligned offset inside its slot | 1,215 (0.93%) | 235 | +0.132% | +25.1% |

These agree with the compiler lane's (b) (26,347 onto mmap, +2.13%, worst +71.7% at n=21793), and the in-class part
of (a) is its first estimate's 628. Every option keeps bounds representable and no rounded end reaches a neighbour's
header.

**(a) is chosen:**
- It is the cheapest, and it moves the allocator's behaviour least.
- It is the only one in which an object's nominal size equals its capability's length. In (b'), the wrapper inflates
  a request so the slot has room for the alignment shift. `set_size` would then record more than the capability
  covers, so `malloc_usable_size` and in-place `realloc` would report bytes the pointer cannot reach, needing more
  changes outside musl.
- (b) keeps the guard verbatim, but it sends more than half of all large requests to another class.

The guard is narrowed, not removed:
- `size_classes` must equal upstream except exactly entries 36, 38, 39, 41, 43, 45 and 47, each at its named new value
  (1170, 1638, 2048, 2732, 4096, 5464, 8192). Any other difference still fails.
- The decision functions stay token-identical.
- The guard gains a positive control: a table with an eighth entry changed is refused.

The port's qualification claim becomes: mallocng's decision code is token-identical to musl 1.2.5, and its size-class
table differs in exactly seven entries, each raised by 1-4 units so that every large class's stride is a multiple of
its bounds granule. heap.c derives each capability over `repr_round(n)`, the nominal size it passes, so the two
agree.

The guard and the principle it encodes, keeping upstream allocation policy, are the external collaborator's work on
the port (adc891bb920c). The lead authorised this narrowing; the landing commit says so.

**Two base alignments the re-cost has to include**, found by the semantics owner with the RTL's bounds model.
"The base comes free" holds only for parents that mallocng placed:
- **mallocng's big path at 4 MiB and up.** There the granule (8 KiB and up) exceeds the page, and `mmap` guarantees
  only a page. With a page-aligned but not granule-aligned base, a region is unrepresentable even with its length
  rounded: 4, 16, 64 and 88 MiB all read inexact in the model, and 2 MiB and 4 MiB-4 KiB read exact. The virtual
  workloads allocate 8, 16 and 64 MiB (the pymalloc replay). The mapping must be granule-aligned: over-map by a
  granule and trim.
- **Parents that mallocng did not place.** memsys5's arena is a static array in the virtual build, so its base
  alignment comes from the linker and needs an explicit `aligned` of its granule. The same applies to any other
  static or mmap-backed parent. What remains reachable through the rounding is the slot's own footer bytes,
stated above.

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

## The bitstream sequence: TWO bitstreams (the lead, 2026-10-17: the simplest path to the full virtual profile)

1. **vp1 = B + R-56 + Q-19 as ruled + Q-17**: capstone-ariane `virtual-profile` (the commit named in the status row
   below and in the pre-registration). Synthesis requested on 2026-10-17 in parallel with the last sweep; the flash
   waits for both comparisons and for the lead's own word to the board lane. Reads on the board: vb-trans's seven arms
   and its flag-off control, Q-17's arm, then the physical regression (the S-16 bare set, the Sublet arms, one corpus
   set) identical to d93a8d66b.
2. **vp2 = the full profile in its minimal form, one bitstream, on top of vp1:**
   - A, the per-context lifetime table at `srevroot`, FLAT only (QEMU's 16-byte record, the flat 30-bit id, the
     reserved ids and the header; the generation path leaves the id under the flag); the paged format deferred;
   - D, `CSRUNV` with its frame, actions 0-3 and every event kind (page fault, node pressure, service ECALL),
     `CSMINT`/`CSRETIRE` over the new table, ONE saved context (the spec leaves context storage open; 32 deferred);
   - **FP state per context: included.** Tier 2 (SQLite's REAL arithmetic, mruby's floats) and Tier 3 (FFmpeg,
     Wireshark) execute hard-float code inside the context, the virtual QEMU saves fpr/frm/fp_status per context
     (#30), and the RTL's switcher saves no FP state today, so without it any FP use across an escape reads back
     stale registers (S-19's shape, now R-53, inside a domain);
   - E, the policy list with #25's untagged-arithmetic rule;
   - H, the collection flush of the R-35 cache and the CPMP/PC trackers; the sweep stays in software;
   - Q-15's mask check and Q-18's DROP rule.
   Deferred as follow-ups only if acceptance needs them: the paged table, 32 contexts. Not done: exact bounds in
   hardware (option (iii) stands), removing the physical profile.
   Acceptance on vp2: Tier 1 (the launcher with the ABI v6 runtime), Tier 2, Tier 3.
   **What cannot share one bitstream without real risk: nothing in this list.** Two items are the high-risk class the
   synthesis rule names and are proven alone in simulation before they join the batch: A (the rev-node unit's memory
   interface, which sits in the standing loop cone; the 2026-08-21 arbiter lesson) and D's frame traffic through the
   switcher and the LSU. Each item is verified independently (its own directed arms and a lint-by-name run at
   baseline) before the next is added, so a gate failure names its item. vp2 depends on no vp1 board reading: if
   vp1's translation reads wrong on silicon, vp2 inherits the same fix, and its synthesis request is not held for it.
   **Q-15 under the flag (ruled 2026-10-16, f224649773ee; in this bitstream):** a CDERIVE child the compact format
   cannot hold exactly is refused with 29 before any mutation, QEMU's compact rule; the physical profile keeps
   rounding outward. Mechanism: not a round trip through compress_cap (a decode in series after one of the three
   instances that set the WNS) but the equivalent mask test in the dyn unit before derive_req: exact iff len < 4096
   (the iE = 0 branch, `E == 0 && !len[12]`), or both start and end are multiples of 2^(E0 + 3) with E0 from the
   length's priority encode and no renormalisation term (aligned ends have zero residues, so the R-54 step never fires
   on an exact region). The runtime lane checked the rule against bounds-model.py at d93a8d66b on 200,000 random
   regions with 0 mispredictions (an 8 KiB exemption would mispredict 749). Directed arm, three cases: a 5,000-byte
   child at an odd base (29 under the flag, rounded on the physical profile), the same child 8-aligned (exact on both),
   len 4095 at an odd base (exact on both).

**Critical path (RTL lane's estimate, 2026-10-17):** A flat with its arms 2026-10-18..21; D with CSRUNV, the kinds,
CSMINT/CSRETIRE and the FP save 2026-10-21..25; E, H, Q-15, Q-18 2026-10-25..27; integration gates (lint by name,
the flag-off sweep and the sup lists readings-identical, the vp1 arms unchanged) 2026-10-27..28; **vp2's synthesis
request 2026-10-28**, in parallel with its last sweep as vp1's was. Board acceptance of vp1 runs in that window and
does not gate it.

## Synthesis of `virtual-profile` 9ed9a2594, bitstream 1 of the virtual profile (B + Q-17), pre-registered 2026-10-16 BEFORE the handoff

Tree: capstone-sublet b7e3287e4 (= the resident d93a8d66b's RTL) + 31c8ebe17 + 16345658f (item B) + ea88d8e5d (Q-19 as ruled) + 9ed9a2594 (Q-17).
Name `caplifive_vp1_9ed9a2594.bit`. Gates on the committed tree before the handoff: lint 52/3/0/0/40/0/2/25/896/2/100
with the 40 UNOPTFLAT names identical (`lint-loop-names.py`), the 95-test sweep readings-identical to the tip sweep
of b7e3287e4 (sweep-compare.py), every testlist_sup.yaml entry readings-identical on the two trees, vb-trans's seven
arms and its flag-off control as pre-registered, Q-17's arm reading 25 (29 on the old tree). The flow untouched;
DTS 65536 unchanged (the pool is unmoved); the capability word unchanged, so every image built for d93a8d66b runs.

Predictions (refutation keys in brackets), written before the build:
- Routed WNS within the noise band of d93a8d66b's -15.856 (the same compress_bounds path sets the slack; this
  change adds no term to any compress or forwarding cone). [WNS worse by more than the noise band: the worst path
  is read; the discriminator is a failing endpoint inside cva6_mmu (the two 2:1 privilege muxes feed the
  permission compares) or at ex_stage's new flops (the kept LSU exception, ~130 bits + trans_id); either names B.]
- Loop membership by stem unchanged; LUTLP-1 0; the 40 UNOPTFLAT names identical (the MMU is in none of them; the
  load unit's new term is a 3-input AND on the request path, not in a loop cone). [a new stem through cva6_mmu or
  load_unit: the translation enable OR'd at the ex_stage boundary entered a cone]
- Area: csr_regfile +1 flop and a 2-bit read mux; cva6_mmu two priv muxes (tens of LUT); ex_stage +~140 flops and
  a 2:1 exception mux at the capability writeback port; load_unit a few LUT; everything else flat. [any module
  moving by hundreds of LUT: a synthesis duplication the model does not see, read the hierarchy report]
- Route time: no prediction; recorded for the spread.
Readings the bitstream must answer (bare JTAG arms, pre-registered values from the simulation): the six vb-trans arms
and the flag-off control; Q-17's arm; the S-16 set, the Sublet arms and one corpus set identical to d93a8d66b (the
physical regression). What the resident fallback stays: caplifive_sublet_d93a8d66b.
Gate results on 9ed9a2594 at the handoff (2026-10-17): lint 52/3/0/0/40/0/2/25/896/2/100 with the 40 loop names
identical (UNUSEDSIGNAL 894 -> 896: seven anonymous thread wires appear and five disappear in the regenerated
capstone_dyn_unit.anvil.sv, no hand-written file moved; the reference moved by name as every Sublet step did); the
seven vb-trans arms, the flag-off control and the Q-17 arm as pre-registered; the Sublet subset (12 tests) identical
to b7e3287e4. On the predecessor 4b958dc3f the flag-off 95-test sweep read 92 identical, 0 differ, 3 missing, and the
full sup list REFUSED it (40 Sublet entries: the reordered CDERIVE read the request's operands after the query's
cycle boundary), the defect fixed in 9ed9a2594. The 95-test sweep and the full sup list rerun on 9ed9a2594 with
their comparisons; synthesis runs in parallel with them on the lead's word of 2026-10-17; the flash waits for both
comparisons and the lead's own word to the board lane.

## Constraints

- **No reflash without the lead's own word.** On 2026-10-16 (evening) the lead set the priority: the full virtual
  Capstone comes first, and the physical campaign yields to it. The lead gave their word for the vp1 flash as soon
  as its synthesis passes the gates and matches the pre-registered prediction (39bf2e4938ae). The board lane stops
  its chain at the next boot boundary when the handoff arrives. A synthesis outside the prediction is a new
  question for the lead. After vp1's physical regression reads identical to d93a8d66b, the campaign's remaining
  items continue on vp1, with the bitstream recorded in every result.
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
| C' bounds | RTL lane | DROPPED 2026-10-16: option (iii), re-decided on the corrected cost, needs nothing in the RTL |
| option (iii) in the runtime: mallocng `repr_round`, the CSMINT descriptor audit, the ABI and launcher gate, runners, documents, re-records | compiler lane | to do, 4-6 days: the re-costed geometry (7 table entries, granule group header, granule offset steps) plus classification on `repr_round(n)` against `stride - IB`, so no rounded end reaches the next slot's header |
| B translation in C mode | RTL lane | GATES GREEN 2026-10-17 on capstone-ariane `virtual-profile` 9ed9a2594 (vp1's RTL: B, R-56, Q-19 as ruled, Q-17 with its operand copy): seven arms, the flag-off control and the Q-17 arm as pre-registered; lint 52/3/0/0/40/0/2/25/896/2/100 with the 40 loop names identical (+2 anonymous generated wires, attributed by name in 9ed9a2594); the Sublet subset identical to b7e3287e4; the flag-off sweep and the full sup list rerunning on this commit with their comparisons (4b958dc3f's sup-list comparison refused its predecessor, 40 Sublet entries, which is what the gate is for); synthesis handed off in parallel on the lead's word; the flash waits for both comparisons and the lead's own word to the board lane |
| A lifetime table | RTL lane | to do |
| D `CSRUNV`, `CSMINT`/`CSRETIRE` | RTL lane | to do |
| E policy list | RTL lane | to do, folded into D |
| board acceptance (done-means 3, 4) | board lane | after the first virtual bitstream |
