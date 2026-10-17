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

**The module's arena rule, decided 2026-10-17 by the semantics owner under the lead's delegation.** With exact
bounds off, `capstone_vm.c:466` refused any arena that was not a naturally aligned power of two. That rule was a
conservative stand-in for representability, moot under the exact-bounds profile and wrong under (iii): mallocng's
group mappings are page multiples, so 413 of 600 allocations failed, and it is the 16384-byte threshold the v6
malloc-contract investigation found. The module's test becomes the representability predicate itself,
`capstone_repr_ok(address, bytes)` from `representable.h`, shared into the module build, and nothing stricter: the
encoder is the authority on representability, and the same predicate already guards the launcher's `grant()` and the
CSMINT audit. The alternative, rounding every arena to a naturally aligned power of two, costs 46% more mapped memory
over 400,000 sizes for no safety gain. Both profiles build the module from `capstone_vm.c`, so the board module's
enforcement changes from "naturally aligned power of two" to "representable"; the landing commit says so, the refusal
path and its errno stay, and the negative test feeds the four refused shapes.

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
   rounding outward. Mechanism (final for vp2, the RTL lane 2026-10-17, superseding the mask test proposed here on 2026-10-16): a
   registered round trip through the package encoder, ex_stage.sv:895-914 at cab4ee891. The child's bounds [rs1.cursor,
   rs1.cursor + rs2.cursor) are latched with the DYN request, decompress_bounds(compress_bounds(child)) is compared
   against both ends one cycle later, and the dyn unit samples the settled bit (capstone_dyn_unit.anvil:277) only after
   its liveness query, so nothing sits on the request path. One implementation is bit-identical to QEMU's cap_compress
   by construction; the mask test, validated against the same encoder on 200,000 regions with 0 mispredictions, is
   kept in the runtime as capstone_repr_ok. The extra compress+decompress instance is named in vp2's synthesis
   pre-registration (area, and as a candidate WNS cone keyed on a worst path ending at dyn_child_repr_q). The runtime lane checked the rule against bounds-model.py at d93a8d66b on 200,000 random
   regions with 0 mispredictions (an 8 KiB exemption would mispredict 749). Directed arm, three cases: a 5,000-byte
   child at an odd base (29 under the flag, rounded on the physical profile), the same child 8-aligned (exact on both),
   len 4095 at an odd base (exact on both).

**Critical path (RTL lane's estimate, 2026-10-17):** A flat with its arms 2026-10-18..21; D with CSRUNV, the kinds,
CSMINT/CSRETIRE and the FP save 2026-10-21..25; E, H, Q-15, Q-18 2026-10-25..27; integration gates (lint by name,
the flag-off sweep and the sup lists readings-identical, the vp1 arms unchanged) 2026-10-27..28; **vp2's synthesis
request 2026-10-28**, in parallel with its last sweep as vp1's was. Board acceptance of vp1 runs in that window and
does not gate it.

## Synthesis of `virtual-profile` 9ed9a2594, bitstream 1 of the virtual profile (B + Q-17), pre-registered 2026-10-16 BEFORE the handoff

Tree: capstone-sublet b7e3287e4 + 31c8ebe17 + 16345658f (item B) + ea88d8e5d (Q-19 as ruled) + 4b958dc3f + 9ed9a2594 (Q-17).
**Base correction (2026-10-17, from the synth lane's check):** b7e3287e4's core/ is NOT identical to the resident
d93a8d66b's: `git diff d93a8d66b b7e3287e4 -- core` is ariane_pkg.sv +29/-4 with 14 non-comment lines, the residue-form
compress_bounds (af7fe9738 carried onto the Sublet tree, the same change built on FP as caplifive_fpb_afcd19bea). So
against the resident bitstream this build changes EIGHT core files and the compress cone, and the WNS arm below
carries two variables: the virtual profile and the residue form. The residue form's effect was measured alone on FP
(afcd19bea against 4a4261bb4: WNS +0.050, TNS better by 13%, the critical path moved to the ra_data compress
instance, carry halved and LUT depth up). b7e3287e4 itself was never built. The handoff's sentence "whose core/ is
identical to the resident" is withdrawn.
Name `caplifive_vp1_9ed9a2594.bit`. Gates on the committed tree before the handoff: lint 52/3/0/0/40/0/2/25/896/2/100
with the 40 UNOPTFLAT names identical (`lint-loop-names.py`), the 95-test sweep readings-identical to the tip sweep
of b7e3287e4 (sweep-compare.py), every testlist_sup.yaml entry readings-identical on the two trees, vb-trans's seven
arms and its flag-off control as pre-registered, Q-17's arm reading 25 (29 on the old tree). The flow untouched;
DTS 65536 unchanged (the pool is unmoved); the capability word unchanged, so every image built for d93a8d66b runs.

Predictions (refutation keys in brackets), written before the build:
- Routed WNS within the noise band of d93a8d66b's -15.856, with the residue form's measured contribution (+0.050 on
  FP, within noise) as the second variable: the compress_bounds path still sets the slack, and the VIRTUAL-PROFILE
  half adds no term to any compress or forwarding cone. [WNS worse by more than the noise band: the worst path is
  read and the two halves are told apart by its endpoint: inside cva6_mmu (the two 2:1 privilege muxes feed the
  permission compares) or at ex_stage's new flops (the kept LSU exception, ~130 bits + trans_id) names B; a path
  through a compress_cap instance whose cells show the residue form's shape (carry halved, LUT depth up, the ra_data
  instance) names the residue form, and the next build then reverts the residue form on this branch to isolate it.]
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
**Read on 2026-10-17 (both gates):** the flag-off 95-test sweep on 9ed9a2594 `SUMMARY 95 tests: 92 identical (exceptions,
readings, retired), 0 differ, 3 missing` (the three the sourceless entries of every sweep); the full sup list against
b7e3287e4's run of the same 152 entries `SUMMARY 152 tests: 152 identical, 0 differ, 0 missing`. Every gate on the RTL
side is green; the flash waits for synthesis inside the prediction, the board lane's controls on the resident bitstream,
and the lead's own word to the board lane.
**Sealed 2026-10-18 (the synth lane; flow 2 h 06 m, synthesis peak 24.91 GB), graded against the section above.** Bitstream
`caplifive_vp1_9ed9a2594.bit` 11,443,722 B, sha256 01072623f8f24b2081556c193113ca906d68316fdc1de15a55ae8267ddf82905; provenance HEAD 9ed9a2594; tarball synth-9ed9a2594-exit0.tar.gz (407,023,106 B, sha256 298476efae9c8afa...). Routed WNS -15.615 (the
router's estimate was -15.672 and the checkpoint query -15.615), TNS -885,099.250 (100,844 of 256,036 endpoints failing), WHS +0.044, one router excursion (3,274 -> 58,792 -> 0), worst path 131 levels (CARRY4 38); route time 41:03.
- **WNS: the prediction HOLDS.** The worst twelve paths start at csr_regfile's cpmp_q[3] bounds (iE) and end at
  issue_read_operands' fu_data_q rd_metadata / cap_metadata_a / cap_metadata_b [17..25] (#1 rd_metadata[25]): d93a8d66b's
  endpoint family; no path ends in cva6_mmu or at ex_stage's new flops, so neither refutation key fires. The sealed reading
  sits 0.241 inside d93a8d66b's sealed -15.856, within the band on the number; the translation half
  added no measurable term.
- **Loop membership: REFUTED in shape, without a timing or bitgen consequence.** The loop count held at 7 (base 7), but
  membership by stem differs from d93a8d66b (the comparator controlled: d93a8d66b against 0568f93a9 reads SAME): gone
  cap_clear_addr_q[n]; new ldc_store_ok_q and gen_register.d_o[n] (i_pipe_reg_load), both members of the existing
  rd_req_q ring. The key "a new stem through cva6_mmu or load_unit" fired in shape, and the mechanism it guessed (the
  translation enable OR'd at the ex_stage boundary) is not the stem: it is R-56/Q-19's clear refusal, `ldc_clear_refuse =
  vctx_i && !ldc_store_ok_d; clear_o = clear_wanted & ~ldc_clear_refuse` (load_unit.sv:186-187), the MMU's store permission
  captured at :511/:565 reaching the store buffer's clear request in the capture cycle. cap_clear_addr_q's exit from the
  listing is unexplained (clear_addr_o read the _d side on both trees). On vp1 as on vp2 the one loop the router leaves unbroken (check_timing 1, TIMING-23 arc
  ex_stage/lsu_bypass/ldc_store_ok_q_i_2 I1->O; d93a8d66b's broke at cap_clear_addr_q[55]_i_2) runs through this gate. If a later build's worst path lands on it: decide the refusal from the registered
  ldc_store_ok_q, delaying the same-cycle-hit case (SEND_TAG_LDC with the result back at once, :504-520) by one cycle.
  LUTLP-1 0 (CFGBVS-1 x2): bitgen not blocked.
- **Area against d93a8d66b, LUT/FF:** TOP -178/+124; issue_read_operands -1,346/-11; scoreboard +646/+5; ex_stage +35/+115 in total, own cells -342/+74; dyn_unit -4,062/+5; rev_node +5,803/+28; lsu -413/+6; cva6_mmu -83/0; load_unit +138/+1; csr_regfile +732/+2; commit_stage 0/0; dom_switcher +9/0; id_stage -239/+11; frontend -39/0. Predicted: csr_regfile +1 flop and a 2-bit read mux; cva6_mmu tens of LUT;
  ex_stage ~+140 flops and a 2:1 exception mux; load_unit a few LUT; everything else flat. **The TOP line holds (flat) and ex_stage's +115 flops sit near the ~140 predicted, but the module rows REFUTE the prediction and fire the hundreds-of-LUT key: rev_node +5,803 against dyn_unit -4,062 (a re-partitioning between the two hierarchies the model does not see), csr_regfile +732 against "+1 flop and a 2-bit mux", issue_read_operands -1,346, scoreboard +646, lsu -413; cva6_mmu's tens of LUT read -83. Recorded, not explained: the flow retimes across hierarchy boundaries, so a module's own-cell count is not the size of its RTL change.**
- Route time: recorded for the spread (no prediction).
The flash waits on the board lane's controls (its amendment on b0-silicon-runtime) and the lead's own word to the board
lane; nothing is staged. vp2's grade (below) compares against this reading.

## Synthesis of `virtual-profile-2-d` bf3db3175, bitstream 2 of the virtual profile (A flat + D + E + H + Q-15 + Q-18, on vp1's B + Q-17 + R-56 + Q-19), pre-registered 2026-10-17 BEFORE the handoff

Name `caplifive_vp2_bf3db3175.bit`. The tree: vp1's 9ed9a2594 plus item A step 1 fbc67c26b and step 2 724e8d393 (the flat
per-context lifetime table at srevroot, the held rev-node namespace, the trackers' namespace flush), item D (D1
441f2aa9e, D2 87a5e633b, D3 8e8b69693, D5/D6/E/H aca8c4929, D4 41fa120bf, Q-15 c4d80699b, the decoder's illegal-at-decode defaults and the flushing csupctl/srevroot writes bf3db3175: CSRUNV with the frame as the
one context's storage, kinds 1-5, FP state per context, CSRETIRE and CSNODE, the policy list, the collection strobe,
the reserve, the representability refusal). Gates on the committed tree before the handoff: lint 52/3/0/0/40/0/2/25/1395/2/100 with the 40
UNOPTFLAT names identical; the flag-off 95-test sweep 92 identical / 0 differ / 3 sourceless against step 2's run; the full sup list
150 of 152 identical, 2 differ by timing only (sup-stcsb-ev-q2k retired 19874 -> 19898; sup-s16-esc-n32 retired 566 -> 539 with its resume count 0x17 -> 0x14, a count the arm registers as '>= 1'): both values reproduced exactly by the D4 tree plus the csupctl-write flush alone, so the flush's cycles per csupctl write move where a 200- or 2000-cycle quantum lands; every registered reading unchanged against step 2's; every item-D arm as pre-registered in its header (va-runv..va-runv7, va-barrier,
va-alloc and its controls at capacity 1024, va-q15, the board lane's va-csmint-emu, va-csmint-mon and va-csmint-entry). The flow untouched; DTS 65536 unchanged (the physical pool is
unmoved and the physical profile is intact: the Sublet and S-16 subsets identical); the capability word unchanged, so
every image built for d93a8d66b runs.

Predictions (refutation keys in brackets), written before the build:
- Routed WNS within the noise band of vp1's reading once it lands, and of d93a8d66b's -15.856 until then (the band is several
  ns; vp1 has not been built at this writing). The compress_bounds path still
  sets the slack. Two new cones are candidates to take it over, both registered on both sides and in no loop: (a)
  Q-15's round trip in ex_stage, a second compress_bounds + decompress_bounds instance from a 64-bit add, latched
  with the DYN request and judged one cycle later; (b) the rev-node unit's retire path, a 30-bit compare against
  next-unused plus the widened kill_mode selector at REVOKE_NODE's two exits. [WNS worse by more than the band with
  the worst path's endpoint at dyn_child_repr_q names (a): the next build splits the judge into two cycles (compress
  registered, decompress + compare the cycle after); an endpoint inside capstone_rev_node names (b); an endpoint in
  issue_read_operands at the FP register file's port 0 names D3's port steal; one in commit_stage at the kind mux
  names the escape's 64-bit cause compare (D1/D5).]
- Loop membership by stem unchanged; LUTLP-1 0; the 40 UNOPTFLAT names identical in every lint on the way (step 2,
  D3, D6, D4, Q-15). The two new rev_node_ch endpoints enter R-27's owed/drain vector (11 bits), which is outside
  every loop cone; the DYN unit's new vctx/derive_repr endpoints are sampled registers. [a new stem through
  capstone_rev_node's IDLE_STAGE priority chain (eleven requesters now) or through the dom switcher's write selector:
  the arbiter-in-a-loop-cone case lint cannot see, isolate by reverting D4's two endpoints on the branch]
- Area: the dom switcher grows most (the frame walk over 4 KiB: 31 + 67 + 33 register ids, the FP ids 75..107, the
  views, the precheck and refusal paths: hundreds of LUT and the frame address registers); capstone_rev_node +6
  30-bit registers, +1 stage, +2 endpoints (a 62-bit reply); ex_stage +2 channels in the drain vector, the Q-15 cone
  (one compress + one decompress instance, ~the size of one write-back encoder) and two 64-bit bound registers;
  csr_regfile the slot's state (used/paused/terminal/kind/frame/options/priv, the strobe, the table bit) and the
  vctx_table decode; decoder two admissions; issue_read_operands the FP port steal; commit_stage the kind mux and
  the escape record. Everything else flat. [a module moving by thousands of LUT: a synthesis duplication the model
  does not see, read the hierarchy report; the switcher's write selector is the first suspect (31+ write sites)]
- Route time: longer than vp1's by the switcher's growth; recorded for the spread, no number.
Readings the bitstream must answer (bare JTAG arms, pre-registered values from the simulation): the item-D arms
(va-runv..va-runv7, va-barrier, va-q15, va-csmint-emu, va-csmint-mon) and item A's (va-flat, va-alloc and its three controls, va-ns-collide and its
control, va-straddle) with their headers' values; vp1's vb-trans arms and Q-17's; the S-16 set, the Sublet arms and
one corpus set identical to d93a8d66b (the physical regression). What the resident fallback stays: whichever of
caplifive_sublet_d93a8d66b and caplifive_vp1_9ed9a2594 is registered on the console at flash time.
Known divergences carried into the build (the Q-entries): the table barrier (a load younger than a non-ending table
writer may read the previous word; a serialising event orders it); the emulation window's two rules; CSRETIRE's
answers without a read; the reserve of 256 per context allocation; capacity and next saturated at 65,536.
**Sealed 2026-10-18 (the synth lane; flow 2 h 05 m, synthesis peak 24.34 GB), graded against the section above.** Bitstream
`caplifive_vp2_bf3db3175.bit` 11,443,722 B, sha256 776cdd8c40d985708a09d4d43ab1c5106414ae1347cc4436a711381d2c197475;
provenance HEAD bf3db3175; tarball synth-bf3db3175-exit0.tar.gz sha256 934e1f9adc6a7b9b... Routed WNS -19.019, TNS
-1,134,141.750 (105,011 of 259,042 endpoints failing), WHS +0.047, route 22:29, worst path 141 levels (CARRY4 49).
- **WNS: the prediction HOLDS on its rule, with a caveat.** The worst twelve paths start at csr_regfile's cpmp_q[0]
  bounds and end at issue_read_operands' fu_data_q cap_metadata_a/b operands and the scoreboard's result_metadata: the
  forwarding / write-back-encoder family that set FP's and Sublet's slack, an unchanged cone; none ends at
  dyn_child_repr_q, inside capstone_rev_node, at the FP register file's port 0 (the worst path through we_fpr_pack has
  +13.096 of slack, through any fprdata net +12.700: D3's port is more than 30 ns from critical) or at commit's kind
  mux. The number is 3.400 ns worse than vp1's -15.672 and 3.163 worse than d93a8d66b's -15.856, inside a band that rests
  on one null sample (S2, 3.74 ns), so the number alone neither confirms nor refutes; the caveat written down: the same
  endpoints read 3.771 ns slower on vp2 for FP's worst (scoreboard mem_q[0] rs1_metadata[25]) and only 0.475 slower for
  0568's worst (mem_q[2] sbe.rs1[0]/CE), so the slowdown is concentrated in the capability-metadata forwarding cone, the
  growth's placement pressure landing beside it (issue_read_operands through D3's port steal, ex_stage through the Q-15
  latch's fanout on the forwarded operands, the switcher) with no changed logic on the path. A null duplicate of vp2 is
  the honest next measurement if the 3.4 ns is to be attributed further.
- **Loop membership: REFUTED, without a timing or bitgen consequence.** Synthesis-stage stems 16 against d93a8d66b's 14:
  +Mem_DP, +ldc_store_ok_q. Neither named suspect (the rev-node's IDLE_STAGE chain, the switcher's write selector) is in
  any loop; the synthesis-stage listing's source annotations for the new block (compress_bounds, range_check) were WRONG
  for its cells, which a netlist query placed inside i_wt_dcache (the write buffer's rd_req_masked and read port 1's
  controller: the read-request handshake ring of R-43's family, an 8-326 exception through dcache_req_ports[1].data_req
  present on vp2 only), Q-15's instance cleared. The one loop the router leaves unbroken (check_timing 1, TIMING-23 arc
  ex_stage/lsu_bypass/ldc_store_ok_q_i_2 I1->O; d93a8d66b's arc was cap_clear_addr_q[55]_i_2 in the same cell family) now
  runs through vp1's R-56 refusal gate, `clear_o = clear_wanted & ~(vctx_i && !ldc_store_ok_d)` (load_unit.sv:186-187),
  where the MMU's store permission reaches the store buffer's clear request in the capture cycle: the key fired in shape,
  by a stem the key did not name. LUTLP-1 0 (CFGBVS-1 x2): bitgen not blocked.
- **Area against d93a8d66b (vp1 and vp2 together; the split against vp1 at vp1's seal), LUT/FF:** TOP +9,488/+1,341;
  **issue_read_operands +9,196/-19, the thousands the duplication key names, in a module the prediction did not list
  (its share between vp1's and D3's port steal is the open question for vp1's seal)**; scoreboard +910/+18; ex_stage
  -3,956/+821 in total with own cells +2,025/+370 (the Q-15 cone, the drain vector, the held namespace); dyn_unit
  -6,025/+16 (a shrink the prediction did not foresee: Q-18's register-only DROP and Q-20's status mapping removed logic);
  rev_node +1,205/+435 (D4: within "hundreds of LUT and the six registers"); dom_switcher +2,883/+354 (D1-D3: "hundreds"
  was an under-prediction by an order of magnitude at the LUT level); csr_regfile +1,161/+140 (the slot's state); lsu
  +423/+18; commit_stage +71/-1; id_stage +137/0; frontend -146/-1; flu_unit 0/+2.
- Route time: 22:29 against vp1's ~27 min; the prediction "longer than vp1's" is refuted.
The flash waits on the board lane's controls and the lead's own word to the board lane; nothing is staged.
**Amended at vp1's seal (2026-10-18).** The comparison reads against vp1's SEALED -15.615 (the -15.672 above was the router's estimate): vp2 is 3.404 ns worse than vp1 and 3.163 worse than d93a8d66b's -15.856. The area split (vp2 against vp1, LUT/FF): TOP +9,666/+1,217; **issue_read_operands +10,542/-8, its OWN cells 17,505 -> 27,845 (+10,340), the FP register file 1,850 -> 1,733 (no LUTRAM: not D3's read port duplicating a RAM) and the integer file 2,186 -> 2,322**; scoreboard +264/+13; ex_stage -3,991/+706 in total with own cells +2,367/+296; dyn_unit -1,963/+11; rev_node -4,598/+407; lsu +836/+12; cva6_mmu -259/-1; load_unit -88/0; csr_regfile +429/+138; commit_stage +71/-1; dom_switcher +2,874/+354; id_stage +376/-11; frontend -107/-1. **So the area prediction's ranking is REFUTED: the largest mover is issue_read_operands' own logic, not the dom switcher, and what the 10,340 LUT are is not shown.** That weighs against the WNS caveat as first written ("placement pressure from neighbours"): the module holding the critical capability-metadata cone itself grew by ten thousand cells between vp1 and vp2. The register deltas (rev_node +407, ex_stage own +296, dom_switcher +354, csr_regfile +138) match the predictions' sizes; the LUT moves do not, and the flow's cross-boundary retiming means an own-cell count is not the size of a module's RTL change (vp1's rows show the same re-partitioning with no RTL change in those modules). The attribution the lead can buy, one bitstream cycle each: vp2 minus D3's FP port steal, and vp2 minus Q-15's cone, against this seal; a null duplicate of vp2 sizes the band but cannot name the 10k. Neither bitstream is flashable on the synth machine's rules (negative WNS); flashing is the lead's word to the board lane.

## Attribution builds for vp2's issue_read_operands growth, pre-registered 2026-10-18 BEFORE any build (synthesis only, not for the board)

vp2 (bf3db3175) seals with issue_read_operands' OWN cells at 27,845 LUT against vp1's 17,505 (+10,340), the module where
the worst paths end (the capability-metadata forwarding family), and routed WNS -19.019 against vp1's -15.615; the FP
register file is not a LUTRAM duplicate (1,850 -> 1,733). Two builds name the ten thousand, each one line off bf3db3175,
lint-gated by name, built in parallel on the synth host on the lead's go; neither is flashable (a context's FP save would
hang on A, an inexact child would be admitted on B), and nothing is staged for the board.

- **A = `virtual-profile-2-attr-a` 993043c97: D3's FP port steal held off** (`fp_dom_sel = 1'b0`, issue_read_operands.sv),
  so the switcher's FP read/write muxes through issue_read_operands prune. Lint 52/3/0/0/40/0/2/25/1395/2/100 at bf3db3175's baseline, the 40 loop names identical.
- **B = `virtual-profile-2-attr-b` a8c873ac3: Q-15's judge held at 1** (`dyn_child_repr_q <= 1'b1`, ex_stage.sv), so the
  compress+decompress cone and the latched child bounds prune. Lint 52/3/0/0/40/0/2/25/1396/2/100, the 40 loop names identical (UNUSEDSIGNAL +1 = dyn_child_back, now unread).

Predictions (what each build READS, written before either runs; the keys in brackets):
- **If D3's port steal is the ten thousand:** A reads issue_read_operands' own cells near vp1's 17,505 (within ~1,000)
  and TOP down by ~10,000 LUT; routed WNS back within vp1's -15.615 ± the one-sample band, the worst path still the
  metadata-forwarding family. B reads own cells near 27,845 and WNS near -19.0. [Then the RTL change is D3's access path:
  the switcher reads and writes FP state through a dedicated narrow port with a registered id decode, not the operand
  read/write ports' muxes; a re-synthesis of vp2 with that change is the next bitstream.]
- **If Q-15's cone, retimed into issue_read_operands, is the ten thousand:** the mirror image: B recovers (own cells near
  17,505, WNS within vp1's band), A does not. [Then the judge moves: the mask test in the DYN unit (capstone_repr_ok's
  rule, already validated against the encoder on 200,000 regions) in place of the encoder round trip, or the round trip
  pipelined over three cycles with the compressed child registered; a re-synthesis follows.]
- **If neither:** both A and B read own cells near 27,845 and WNS near -19.0 (within the band of each other). [Then the
  growth is D1/D2's switcher register-access path through issue_read_operands (the frame walk's 108 register ids on the
  read and write ports) or the flow's cross-boundary retiming of the switcher's growth into the module; the RTL change
  would be a registered two-level id decode for the switcher's register access, a redesign to be costed before any build,
  and the flash decision for vp2 rests on the board's canary alone.]
- **Mixed outcomes** (both recover part; the sum near 10,000) name both and the two changes above are both taken.
Each reading changes the next RTL step, so the builds go; a null duplicate of vp2 would size the band but name nothing,
and is not requested. The grade is written against this section at each seal: own-cell LUT of issue_read_operands, TOP
LUT, routed WNS and the worst endpoint, loop count and LUTLP-1 (both expected unchanged from vp2's: 8 blocks, 0).

**B sealed 2026-10-18 (the synth lane; flow 2 h 24 m 51 s, synthesis peak 22.31 GB), graded against the keys above.** Build
a8c873ac3 (Q-15's judge held at 1), bitstream 11,443,722 B, sha256 906941f54787efdfca25c53153e7cee5c20054a84a223f24801059828e33411b,
synthesis-only, not for the board. Routed WNS -17.537 (TNS -837,683.375, WHS +0.047, 101,314 failing endpoints of 258,951,
worst path 129 levels), the worst endpoints the same family as vp1 and vp2 (csr_regfile cpmp_q[7] iE -> issue_read_operands
fu_data_q cap_metadata_b[20] / rd_metadata[19..20] / cap_metadata_a[20]). LUTLP-1 0; the router leaves one loop, its
TIMING-23 arc cap_clear_addr_q[55]_i_2 (d93a8d66b's arc, not vp1's and vp2's ldc_store_ok_q_i_2: which member of the ring
the router leaves varies by netlist).
- **The deciding key: issue_read_operands' own cells 28,457, against vp2's 27,845 (+612) and vp1's 17,505. Q-15's cone is
  NOT the ten thousand.** The second hypothesis is refuted on its own key; "D3" and "neither" stay open on A's reading.
- **Where Q-15's cone was: ex_stage's own cells 2,410 -> 337 (vp1 43), 2,073 LUT** for the latched child bounds and the
  compress + decompress round trip, in the module that holds them, not retimed across the boundary. TOP 178,406 -> 177,515
  (-891): the cone's 2,073 is offset by netlist-to-netlist movement elsewhere (scoreboard 14,451 -> 16,727, dom_switcher
  9,467 -> 9,762, dyn 3,381 -> 3,052, rev_node 2,842 -> 2,737, csr_regfile 28,405 -> 28,133), none of it predicted and
  none of it in a module the one-line change touches.
- **WNS: B recovers 1.482 of vp2's 3.404 ns gap to vp1, and that is INSIDE the one-sample null band (3.74 ns)**, so the
  1.5 ns is not attributable to Q-15 on this evidence; the worst-path family is unchanged, consistent with the round trip's
  cone sitting off the critical path. TNS -837,683 is better than vp1's -885,099.
- **Consequence for vp2b:** the Q-15 variant is not named by this seal. The mask-form judge stays validated and available
  (capstone-ariane `virtual-profile-2-d` 947ebb48c: equal to the round trip on 11,092,561 children, 0 mismatches, both
  controls firing) as a 2,073-LUT reduction of ex_stage with no reading change; it joins a vp2b build only alongside the
  change A's seal names, never as the fix for the ten thousand. If A also reads near 27,845 the pre-registered "neither"
  branch applies: the two-level id decode, costed before any build.

**A's area reading, 2026-10-18, BEFORE its seal** (the synth lane; A 993043c97, D3's FP port steal held off, crossed the
host's 3 h 30 m rule without a bitstream and is treated as failing until it seals; route_design still converging, 567
overlaps at 285 min after a host load of ~60 from another user's jobs). Basis: routing does not change cell counts and the
flow has no post-route phys_opt; control: vp1's and vp2's post-phys_opt checkpoints read exactly their routed counts
(21,084 and 32,878), so A's post-phys_opt checkpoint reads what its routed one will. WNS, endpoints and LUTLP-1 come at
the seal, if it comes (first estimate -18.142, vp2's).
**A FAILED TO ROUTE (2026-10-18, after ~5 h in route_design): "Design is not legally routed", 171 node overlaps (Route
35-2), 242 signals failed to route due to routing congestion (Route 35-162); no bitstream, NO routed WNS.** The last
WNS in the log (-14.662, WHS -0.400) is an intermediate figure on an illegally routed design before the hold fix and is
not A's WNS. Grade line: "A: routing failed (congestion; 242 signals, 171 overlaps); no routed WNS". The run exited 2 after 6 h 39 m 41 s
(synthesis peak 21.76 GB); the guard kept a 45-file diagnostic artifact (synth-993043c97-exit2.tar.gz, 1,589,889 B) and
the timing enumeration was skipped for want of a routed checkpoint. The area reading
below stands (post-phys_opt; routing does not change cell counts). **Routability on this design is marginal:** A's TOP
178,757 LUT against vp2's 178,406, which routed in 22 min; the same size routes on one netlist and not on another, and
the congestion is local, not occupancy-driven. Every future bitstream of this size carries that risk, and a hierarchy
re-homing like A's (the DYN result cones moved into `dyn_unit/_spawn_0`) is enough to tip it.
- **By the pre-registered key alone, A names D3:** issue_read_operands own LUT cells vp1 21,084 | vp2 32,878 | A 21,977;
  the module's `mem_q` stem 11,370 | 22,741 | 11,465.
- **By the control the key lacked, it does not: the logic did not leave the design.** Total LUTs vp1 168,740 | vp2
  178,406 | A 178,757 (A keeps vp2's +9.7k, +351 above it). issue_stage 39,374 | 50,182 | 37,562; scoreboard 14,187 |
  14,451 | 11,968; ex_stage 41,659 | 37,668 | 48,051; dyn_unit 5,344 | 3,381 | 13,923 (+10,542 against vp2), the growth
  in its Anvil child `dyn_unit/_spawn_0`: 14,366 LUT cells on A against 870 on vp2; the `mem_q` stem summed under
  dyn_unit 3,019 -> 14,914 (+11,895). The SAME scoreboard write-back input cones are homed inside the DYN unit on A,
  where the `_ep_res_0` chain lives, and under issue_read_operands on vp2 and B.
- **Grade: the "D3" hypothesis is REFUTED on the total and the key is REFUTED as an instrument.** The pre-registration
  measured where the flow homes a cone, not whether the cone exists: issue_read_operands' own cells are not a stable
  attribution metric on this design; TOP plus the stem summed over all hierarchies is. Both attribution builds agree with
  the source diagnosis below: the ten thousand is the DYN result chain's write-back cones (90 -> 105 sources), and which
  module's hierarchy they land in is netlist-dependent (D3's port steal changes it, Q-15's cone does not). This is the
  "a check that fires correctly and still under-determines" shape: the key fired exactly as written and could not tell
  "D3 is the logic" from "D3 moves the logic". The control that catches it is the one every per-module area key needs
  from now on: the design total beside the module's count, and the stem summed over every hierarchy.

## vp2b, the timing fix for vp2: pre-registered 2026-10-18 BEFORE the A/B seals, built only if vp2's canary fails

The lead's standing instruction (relayed by the runtime lane, 2026-10-18): everything needed for a working vp2 and the
whole corpus on it in fully virtual mode, without further asks. So the contingency is written down now, conditional on the
attribution readings, and runs without a decision point: if vp2's first board session (the canary) fails, the RTL change
named below starts at once on capstone-ariane lane branch `virtual-profile-2-timing` (from fc679dc68, RTL bf3db3175), and the
synth lane builds it overnight on the verified handoff (its standing change, dev 896759b07aef) as `caplifive_vp2b_<hash>.bit`.
If the canary passes, vp2b is not built; this section then names the timing fix for the record and for whatever bitstream
follows. A reflash still waits on the lead's own word to the board lane.

**Outcome (2026-10-18): vp2 is resident and its canary passed (dev 987bf254ee40), so vp2b is NOT built.** This section is
the record. The attribution (above) named the growth as the DYN unit's unregistered result chain and refuted all three
pre-registered branches as written; the change it leaves on the table, collapsing the unit's sends onto one result
register, is deferred, not scheduled, and belongs to whatever bitstream follows. A's WNS, at its seal, is recorded above.

**Which change, decided by the seals against the section above's keys** (issue_read_operands' own-cell LUT first, routed
WNS second; 1.5 ns of WNS alone decides nothing, the one-sample null band is 3.74 ns):

- **B names Q-15 (B's own cells near vp1's 17,505, A's near 27,845): the mask test replaces the round trip.** ex_stage.sv
  902-915's `dyn_child_back = decompress_bounds(compress_bounds(dyn_child_q, 5'd0, 1'b0), dyn_child_q.start)` and its two
  64-bit compares go; `dyn_child_repr_q` is computed from the encoder's own exactness condition on the latched child
  (start, len = base - start): E from the encoder's leading-zero window (ariane_pkg.sv compress_bounds), exact iff
  `(E == 0 && !len[12]) || ((start | len) & ((1 << (E + 3)) - 1)) == 0`. With both ends on the granule the residues sr and lr
  are zero, so the residue-form step cannot fire, and q = len >> (E + 3) < 1024 by E's choice; with start = cursor for a
  fresh child the cursor-relative reconstruction adds no term. The same children are admitted and refused, so NO reading
  changes: va-q15 and every D arm identical. Validation before any build: bounds-model.py (verif/tests/custom/capstone/)
  gets the mask form as a third function and asserts equality with the round trip at hardware width over its existing
  corpus (400,000 random regions, three cursors each, and every region below 2^16); one disagreement and the mask form is
  NOT taken: the fallback is the round trip pipelined (compress's bounds_t registered, decompress and compare the next
  cycle; no stall, the DYN unit reaches its test after the liveness round trip). Positive control: a child one byte off its
  granule reads exact = 0 in the model and refuses 29 in va-q15 (as it does today). Predictions: issue_read_operands own
  cells within ~1,000 of B's seal reading, TOP within ~1,000 of B's, routed WNS within the band of B's with B's worst
  endpoint, lint by name at 52/3/0/0/40/0/2/25/1395/2/100 with the 40 UNOPTFLAT names identical (UNUSEDSIGNAL may drop by
  dyn_child_back's one), loops as B's. A reading worse than B's by more than the band means the test added logic the pruned
  cone did not have, and names the pipelined form for the next build.
- **A names D3 (the mirror): the switcher's FP access through a dedicated narrow port.** The context's FP save and restore
  (D3: 32 registers plus fcsr at frame +656 / +912, zeroed at entry) move one register per cycle through one read port and
  one write port of the FP register file with the id registered a cycle ahead, instead of steering the operand ports'
  muxes (`fp_dom_sel`, issue_read_operands.sv). Semantics unchanged; the FP state arm va-runv3 keeps every value and may
  move in retired-cycle count only (recorded, not registered). Validation: the D list below, and, because the change is in
  the issue stage, the FP coverage set rv64ud-p-* and s18-fp-* identical to bf3db3175's run (the 95-test sweep has no FP
  coverage). Predictions: own cells within ~1,000 of A's seal reading, TOP within ~1,000 of A's, WNS within the band of A's
  with A's endpoint, lint by name identical, loops as A's.
- **Neither (both near 27,845): the registered two-level id decode for the switcher's register access.** The frame walk's
  108 register ids on the read and write ports (D1/D2) are decoded in two registered levels instead of one flat level.
  COSTED FIRST, no build without it: a design note deriving the predicted LUT reduction from the synth lane's per-instance
  utilization of vp2 (the same report that showed the own-cell growth), and the predicted WNS; if the costing cannot
  predict a recovery above the band, vp2b is not built and the flash decision rests on the canary alone.
  **Amendment 2026-10-18, after B's seal and BEFORE A's: the mechanism this branch names is refuted at the source.**
  Between vp1 (9ed9a2594, the merge base) and vp2 (bf3db3175) exactly two commits touch issue_read_operands.sv: D3
  8e8b69693 (35 lines added, 6 removed: the FP port steal, `fp_dom_sel` and its read/write steering) and D4 41fa120bf (one
  line: CSNODE and CSRETIRE added to the serialisation list). D1 and D2 did not touch the module; the switcher's register
  access through it (the 108 ids) predates vp1 unchanged, so "D1/D2's path" cannot be a vp1-to-vp2 growth. If A reads near
  27,845 too, the ten thousand is NOT a source change inside the module: it is the flow's cross-boundary retiming or
  absorption of logic that belongs to a neighbour (ex_stage, the scoreboard, the switcher), and B's seal shows the
  per-module own-cell counts move by thousands on a one-line change elsewhere (scoreboard 14,451 -> 16,727). Then the next
  instrument is not an RTL redesign but a netlist attribution: the synth lane lists issue_read_operands' own cells by the
  driving nets' hierarchical origin on vp2 and vp1 (the loop-block query already does this per block), and the RTL change
  is named from that listing, costed, and only then built. The two-level id decode is withdrawn as the "neither" action;
  no vp2b build is handed off on a "neither" reading until the attribution names the cone.
  **The attribution ran and named the cone (2026-10-18, the synth lane's read-only queries on the routed checkpoints of
  vp1, vp2 and B; LUT cells by PARENT, dual-packed counted separately: IRO own 21,084 / 32,878 / 33,885).** 96% of
  issue_read_operands' growth is cells named after the scoreboard entries' cap_result fields: `mem_q` stems 11,370 ->
  22,741, by field [sbe][cap_result][dom_switch_req] 18 -> 3,165, rs1_metadata 1,662 -> 4,754, result_metadata 2,866 ->
  5,711, rs1_cursor 1,036 -> 1,946, [sbe][result] 4,437 -> 5,119, [sbe][ex][tval] 265 -> 761. The control fires: in
  ex_stage, B minus vp2 is the Q-15 stem alone (dyn_child_repr_q 2,343 -> 0). What the cells are: 100% end in `_i_<n>`,
  the INPUT cones of the scoreboard's entry flops; 0 carry a replication suffix (the matcher finds 2,844 / 2,801 replicas
  elsewhere in the design); the scoreboard's own cells for those fields are tiny on both trees (51 -> 3, 35 -> 10, 4 -> 1),
  so it is not relocation; the flop counts are unchanged (dom_switch_req 2,616 / 2,628, rs1_metadata 530 / 520,
  result_metadata 520 / 520). The cones per flop bit deepened: dom_switch_req 0.007 -> 1.20 LUT per bit (a wire became a
  select), rs1_metadata 3.1 -> 9.1, result_metadata 5.5 -> 11.0. **The source, with both trees quoted:** the DYN unit's
  result endpoint `_ep_res_0` (910 bits) is generated as ONE combinational priority chain over every Anvil `send ep.res`
  site: vp1's generated core/capstone_dyn_unit.anvil.sv:11153 has 90 ternaries, vp2's :12820 has 105 (the .anvil's send
  sites 31 -> 37 for CSRUNV, CSNODE and CSRETIRE; all in the one handler thread), and `create_result_pack_runv`
  (capstone_unit.anvilh:454-458) is a fourth distinct value for the switch-request fields, frame / action / cnull, where
  vp1 had zero / CALL / RETURN. The chain feeds cva6.sv's `cap_wbdata_id_ex[1]` and the scoreboard's cap_result flops with
  no register between (cva6.sv's assembly and scoreboard.sv are identical on both trees), so its per-bit logic is counted
  at those flops' D cones, which the flow homes under issue_read_operands. The FLU's chain grew too, 55 -> 58. **Reading:
  real logic from item D's result sends, not relocation, not tool replication, not a defect; the module-level "own
  cells" of a unit whose output is unregistered are the flow's to place.** No vp2b build follows from it. Two fix shapes,
  costed before any is named: (a) register the DYN result pack at the unit boundary: the chain's logic gets its own flops
  and the critical cone shortens, but every DYN op retires one cycle later, so every retired-count reading in the sweep
  and the sup list moves and the baselines are re-recorded (the sweep compares retired counts); (b) collapse the 37 sends
  onto one result register written on each path and sent once at the handler's end: cycle-neutral, no reading changes,
  but a restructuring of the dyn unit's control flow in Anvil with the generated assignment checked for every path
  (Anvil drops a set silently in two known shapes). (b) is the one to take if a build is ever needed for this; neither is
  needed for a working vp2, whose WNS difference to vp1 (3.404 ns) is inside the one-sample null band.
- **Mixed (both partial, the sum near 10,000): both changes, one build.**

**Arms that re-run on the vp2b tree, every pre-registered value unchanged** (the exception named above: va-runv3's
cycle count under the D3 change): item D's va-runv, va-runv-quantum, va-runv2, va-runv3, va-runv4, va-runv5, va-runv6,
va-runv7, va-barrier, va-q15, va-csmint-emu, va-csmint-mon, and the board lane's va-csmint-entry; item A's va-flat,
va-flat-off, va-alloc, va-alloc-off, va-alloc-corrupt, va-alloc-exhaust, va-ns-collide, va-ns-collide-off, va-straddle;
vb-trans, vb-trans-plain, vb-trans-vp2, vb-trans-plain-vp2, vb-trans-off. **Gates**: lint by name against bf3db3175's
52/3/0/0/40/0/2/25/1395/2/100 with the 40 UNOPTFLAT names identical (a changed name goes to the auditor before the handoff);
the 95-test sweep 92 identical / 0 differ / 3 sourceless against bf3db3175's run; the sup list 152 of 152 identical to
bf3db3175's (its two quantum-timed moves against the D4 tree are now the baseline); the vp2 board kit (dev 4be3dfe15509, 23
images plus q17-order and vb-trans-off) re-simulated on the vp2b tree with vectors identical to the bf3db3175 run, so the
board's staged images need no rebuild. The handoff carries the hash, the lint numbers and this section's predictions for the
chosen variant; the grade is written at seal against them.

## vp2c, pre-registered 2026-10-18: Q-24 alone (the medeleg mask gains causes 24-30), for whatever bitstream follows; NOT built now

vp2's first virtual boot (board lane, c1fa1a863d09) exposed Q-23 and Q-24 (ISSUES.md). The semantics owner ruled Q-23
closed with the RTL as it is (trusted S performs no capability memory access; QEMU is tightened; the module takes the
CSMINT descriptor v7 route), so vp2 PROCEEDS WITHOUT A BITSTREAM and the LDC-side RTL edits that were drafted for the
other outcome are discarded (a scratch patch, never committed). What stays pre-registered, for the next bitstream of
the virtual profile, is Q-24 and nothing else, on capstone-ariane lane branch `virtual-profile-2-c` from fc679dc68 (RTL
bf3db3175): `csr_regfile.sv` CSR_MEDELEG's write mask gains `UNEXPECTED_OPERAND_TYPE` 24 through
`INSUFFICIENT_SYSTEM_RESOURCES` 30 under `CVA6Cfg.CAPSTONE_EXT`, as QEMU's `CAPSTONE_DELEGABLE_EXCPS`. One file, one
expression; medeleg resets to 0 and nothing writes those bits today, so every existing reading is unchanged; the switcher's
unmasked restore at `:2023` is untouched (M-owned, see Q-24).
- **Arms:** va-medeleg: in M, `csrw medeleg` with bit 24 set reads back with bit 24 set (today it reads 0: the positive
  control on the mask); from S with bit 24 set, a scalar STC through an integer base (24 on both profiles, Q-23's rule)
  lands at stvec with scause 24, stval the integer, sepc the STC (the S handler prints them and steps over); with the
  bit clear the same STC lands in M (mcause 24, the trap handler prints it); bit 25 the same way with a load through a
  revoked capability (25); a trap from U under the bit lands at stvec with SPP = 0. Every other arm unchanged: the D list
  (va-runv..va-runv7, va-barrier, va-q15, va-csmint-emu, va-csmint-mon, the board lane's va-csmint-entry), item A's nine,
  the five vb-trans arms.
- **Gates:** lint by name against bf3db3175's 52/3/0/0/40/0/2/25/1395/2/100 with the 40 UNOPTFLAT names identical
  (the mask is a constant; nothing in a loop cone); the 95-test sweep 92 identical / 0 differ / 3 sourceless against
  bf3db3175's run; the sup list 152 of 152 identical.
- **Synthesis prediction, with its key:** the change is seven constant bits in a CSR write mask: own cells of
  csr_regfile within the netlist band of vp2's (28,405 LUT cells; B read 28,133 on a change elsewhere), TOP within the
  band of vp2's 178,406, routed WNS within the one-sample band (3.74 ns) of vp2's -19.019 with vp2's worst endpoint
  (cpmp_q -> issue_read_operands fu_data_q cap_metadata), loops and LUTLP-1 as vp2's. Refutation key: a worst path
  ending at medeleg_q or a csr_regfile endpoint, or csr_regfile's own cells moving by more than the band. Routability
  caveat: at ~178k LUT a build of this size may fail to route on hierarchy alone (A did); a failure is recorded as such,
  never as a WNS. The grade is written at seal against this section.
- **If the next bitstream also carries other changes** (vp2b's deferred fix, the Q-15 mask form), each is its own
  section and this one is cited by hash; the arms and gates above still apply.

## N saved virtual contexts, costed 2026-10-18 (the remedy for Q-26's one slot), NOT scheduled

vp2 keeps ONE saved context, so a paused context blocks the next entry (commit_stage.sv:271-276, mcause 29) and
re-ENTRY is not a resume (ISSUES.md Q-26): a join or a shared-memory wait is "not available on a one-slot bitstream" and
the module returns -EBUSY where a second live context is required. The semantics owner asked for this costing so the
lead can buy the remedy once the compiler lane says how many Tier 3 bundles need a second live context; the points are
N = 2, 4, 8.

- **What changes.** The switcher: nothing (every context's architectural state already lives in its frame; ENTRY,
  ESCAPE and RESUME act on the frame named by the request, capstone_dom_switcher.anvil:157-161, :295-345). The CSR file:
  the one slot `{used, paused, terminal, kind[2:0], frame[63:0], options[1:0], caller_priv[1:0]}` (74 flops,
  csr_regfile.sv:387-391) becomes an array of N plus an active-slot index (log2 N bits), set at entry and resume
  (:2116-2119), written by the escape (:2127-2130) and by the bad-frame id 74 (:2046-2056), cleared by the discard of the
  matched slot (:2123); the frame match is N 64-bit comparators of dom_switch_req.base_addr against the N frames, done in
  the CSR file, which returns match / paused / kind / free to commit. Commit: the three predicates (same frame, resume ok,
  service ok; :271-276) read the matched slot, and the refusal for a NEW frame becomes "no free slot"; the escape's rd
  target, the quantum counter and the csup* record stay single (one context runs at a time). QEMU's twin is the existing
  property `x-capstone-supervisor-slots` at the same N.
- **Predictions, keyed as the attribution builds were, with the design total AND the flop count as controls.** Flops:
  exactly 74·N + log2 N (N = 2: +149, 4: +298, 8: +595), the one sharp key, since a flop count is not moved by the flow.
  LUT: about 85·N (64 per comparator plus the muxes; 2: ~170, 4: ~340, 8: ~680), BELOW the per-module netlist band
  measured on the attribution builds (csr_regfile moved 272 on B, the scoreboard 2,276), so the LUT reading can neither
  confirm nor refute at any N and must not be graded as if it could; TOP within its band. WNS: within the one-sample
  band of the base with the worst endpoint unchanged (cpmp_q -> issue_read_operands cap_metadata); refutation key: a
  worst path ending in csr_regfile's slot match or at commit's dom_switch_req valid. **The WNS premise, read by the
  synth lane on vp2's routed checkpoint (2026-10-18):** today's compare survives as `commit_stage_i/runv_same_frame0`
  (the three predicates absorbed into it); its combinational ACCEPT path (:603) from `runv_frame_q` into the switcher,
  endpoint `dom_switcher/cur_base_q_reg[14]/D`, has slack +20.712 ns over 41 levels (the next four cur_base_q bits
  +20.789..+20.884), and it is the worst path from runv_frame_q to any endpoint; the registered refusal path into
  `runv_refuse_q_reg/D` has +36.130 ns over 9 levels. N comparators and one mux level (about two LUT levels, 1-2 ns) fit
  in the +20.712 with room, so the one-cycle registered fallback (every CSRUNV one cycle later, every D arm's
  retired-count reading moving) is NOT needed at N <= 8 on today's numbers; it stays written down as the fallback if the
  routed build says otherwise. Routability at ~178k LUT is the risk A showed, independent of N.
- **Arms.** va-runv8 on N = 2: context A enters and escapes with a quantum (kind 1) holding dirty FP registers and a
  non-default fcsr; B enters a second frame, runs to a terminal escape and is discarded; A is RESUMED with action 0 and
  continues at the interrupted pc with its integer, FP and fcsr state exact (the readings re-ENTRY cannot produce, Q-26);
  a third frame while two are paused reads the 29 trap ("no free slot"); a discard of the non-active slot by frame; a
  resume of the OLDEST slot. Every existing arm unchanged (N >= 1 keeps today's semantics): the D list, item A's nine, the
  five vb-trans arms, the board kit's 25 images re-simulated identical. Gates: lint by name with the 40 loop names, the
  95-test sweep 92/0/3, the sup list 152/152 against the base.
- **Effort and recommendation.** RTL about a day (CSR file, commit, cva6 wiring), the arm a day, the gates half a day,
  then the usual synthesis cycle on the lead's buy. N = 4 unless the largest Tier 3 bundle has more live threads; 8 costs
  nothing measurable and buys nothing the module's queue needs.

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
| B translation in C mode | RTL lane | ALL GATES GREEN 2026-10-17 on capstone-ariane `virtual-profile` 9ed9a2594 (vp1's RTL): seven arms, the flag-off control and the Q-17 arm as pre-registered; lint 52/3/0/0/40/0/2/25/896/2/100 with the 40 loop names identical; the flag-off sweep 92/95 identical (3 sourceless), the full sup list 152/152 identical; synthesis handed off as `caplifive_vp1_9ed9a2594.bit`; the flash waits for synthesis inside the prediction and the lead's own word to the board lane |
| A lifetime table | RTL lane | STEP 2 GREEN 2026-10-17 on capstone-ariane `virtual-profile-2` 724e8d393 (step 1 fbc67c26b): the unit allocates from the table's header as QEMU's table_alloc (free list, then the bump; the open-time header checks at every allocation, status 3 -> 25, exhausted -> 30), retires without a push (a kill nulls the retired record's links as cap_rev_retire), ids without a generation, the scaffold and the index shield gone; the rev channel's namespace (flag, table base) is HELD while a handler runs, and the three revocation trackers forget on every namespace change (the audit's arm read a stale ALLOW across the physical/virtual boundary before that, 25 after). Nine arms as pre-registered (va-flat, va-alloc with the flag-off controls, corrupt 25, exhaust 30, ns-collide, straddle); lint by name at baseline (40 loops, UNUSEDSIGNAL 896 -> 926 all generated-unit wires); sweep 92 identical / 0 differ / 3 sourceless (SWEEP-RAN 95); sup list 152 of 152 identical. Recorded divergences, not closed: DROP retires the shared node where QEMU's is register-only (Q-18, item E); no visit bound on the REVOKE walk (a corrupt cyclic table hangs where QEMU refuses: item D's fail-closed work, with PAGED and non-RECYCLE tables refused at CSRUNV); the derive handler does not check the parent's next points back; QEMU's open validates one list entry beyond the popped one; capacity and next saturate at 65,536; the bump writes root+16 back unchanged beside root+24; depth 17 bits and parent 16 where QEMU's are 30. Next: D (CSRUNV front end, one context, CSMINT/CSRETIRE, FP state) |
| D `CSRUNV`, `CSMINT`/`CSRETIRE` | RTL lane | GREEN 2026-10-17 on capstone-ariane `virtual-profile-2-d` bf3db3175 (D1 441f2aa9e, D2 87a5e633b, D3 8e8b69693, D5/D6/E/H aca8c4929, D4 41fa120bf, Q-15 c4d80699b, the decoder and CSR-write fixes bf3db3175; from step 2's 724e8d393). CSRUNV (0x24) from scalar S or scalar M enters a context on the existing switcher with the frame as its storage (4 KiB: the caller's integer snapshot at +1024 and its FP state at +2048..+2304, the context's x_i at +64+16i, fpr at +656, fcsr +912, the views +576..+639 when options are set); kinds 1 (quantum), 2 (terminal), 3 (service ECALL, action 2 consumes +224 and advances), 4 (page fault, action 0 retries), 5 (node pressure: the allocator's 256-id reserve inside a context, action 0 re-executes after a software collection); action 1 forgets, action 3 is illegal (collection is software); a bad frame is a terminal event {2, 29} written into the frame with rd = 2 (the only non-trap 29); a refused action (0 while the one slot holds a context that is not a resume of the same frame, 2 without a paused kind 3, a misaligned frame) TRAPS with mcause 29 at commit (commit_stage.sv:274-278, :1021-1027; reworded 2026-10-18 after vp2's boot 2, where the kit read "refused actions read 29" as rd; QEMU traps the same, capstone_supervisor.c:878-893). One saved context. FP state per context (zeroed at entry, saved and restored by the switcher). Item E: SEAL, RETURN, CCSRRW, SFENCE.VMA and CSR 0x803 are illegal in a context, and #25's untagged CINCOFFSET/CINCOFFSETIMM/SCC give integers there. Item H: csupctl bit 2 is the collection strobe (the trackers forget); Q-18: DROP is register-only under the table. Decision 3: the table namespace is a context, S outside one with srevroot set, or M with csupctl bit 3; a corrupt table reads 29 there (Q-20). D4 (the semantics owner's split): CSMINT traps in S and the monitor emulates it; CSRETIRE (0x53) is hardware (next-unused bounds the id and a read-only pass over the run refuses a cycle or an over-long run before any write, then the run loses VALID and the operand is unlinked; rd 0/1, never a fault); CSNODE (0x54, RTL-only) mints QEMU's child-and-ancestor pair for the emulation (rd = the ancestor id, rs1 re-noded; 30 below two identities); both admitted to trusted S and M under the table bit, outside a context, never U. Q-15: an inexact CDERIVE child is refused with 29 under the table namespace by the encoder's own round trip (va-q15; the physical profile keeps rounding). Found by the board lane's CSMINT-emulation arm (va-csmint-emu, 32 readings as pre-registered on this tree): an unknown custom encoding is now illegal at decode (mtval = the bits) and a csupctl/srevroot write flushes the pipeline; the board lane's monitor-path arm va-csmint-mon (46 readings, the monitor's own routine pinned at caplifive-sbi 868dd5f) and its entry arm va-csmint-entry (the shipped _cap_trap_entry, cb91d7d) both read as pre-registered on the same RTL (cab4ee891 carries va-csmint-mon as a regression). Arms as pre-registered: va-runv, va-runv-quantum, va-runv2 (53 readings), va-runv3, va-runv4, va-runv5, va-runv6 (50), va-runv7, va-barrier, va-q15, va-csmint-emu, va-alloc-exhaust. Gates on the D3 tree 8e8b69693 (D1-D3 alone): the 95-test sweep 92 identical / 0 differ / 3 sourceless against step 2, the sup list 152 of 152 identical; on the D4 tree 41fa120bf the sweep again 92 / 0 / 3. Gates on the final tree bf3db3175: lint by name 52/3/0/0/40/0/2/25/1395/2/100 with the 40 UNOPTFLAT names identical; sweep 92 identical / 0 differ / 3 sourceless; sup list 150 of 152 identical, 2 differ by timing only (sup-stcsb-ev-q2k retired 19874 -> 19898; sup-s16-esc-n32 retired 566 -> 539 with its resume count 0x17 -> 0x14, a count the arm registers as '>= 1'): both values reproduced exactly by the D4 tree plus the csupctl-write flush alone, so the flush's cycles per csupctl write move where a 200- or 2000-cycle quantum lands; every registered reading unchanged. Recorded divergences, not closed: ids 0/1 and >= 65536 answer CSRETIRE 1 without a read (QEMU: > UINT32_MAX); the reserve test counts next-unused saturated at 65536; in M under the table bit the LSU's probes read the table, so the monitor's emulation window must hold no capability data access (the trackers assume their capability valid across the namespace change). vp2 SEALED 2026-10-18 as caplifive_vp2_bf3db3175.bit (sha256 776cdd8c...7475; routed WNS -19.019, LUTLP-1 0), graded in its section below: the WNS prediction holds on the endpoint rule with the metadata-cone caveat, loop membership refuted without consequence, issue_read_operands +9,196 LUT pending the vp1 split. vp1 SEALED 2026-10-18 as caplifive_vp1_9ed9a2594.bit (sha256 01072623...2905; routed WNS -15.615, LUTLP-1 0), graded in its section: WNS holds, loop membership refuted in shape by R-56's gate, the module-level area rows refuted; vp2's grade amended at the split (issue_read_operands' own cells +10,340 LUT, the ranking refuted). Next: the flashes on the lead's own word to the board lane, vp1 first; the attribution builds are the lead's to buy |
| E policy list | RTL lane | GREEN 2026-10-17 with D (aca8c4929): the virtual context's refusals and #25, va-runv5 |
| board acceptance (done-means 3, 4) | board lane | vp1 flashed and its first session clean against d93a8d66b with its four arms as pre-registered (dev c5e696ac7328); vp2 `caplifive_vp2_bf3db3175.bit` RESIDENT 2026-10-17 on the lead's own word: canary 9 of 9, all 25 arms equal to their simulated vectors, corpus set 7 33 of 33 (dev 987bf254ee40) |
| synthesis vp1 / vp2 | synth lane | SEALED 2026-10-18 and graded above: vp1 WNS -15.615, vp2 -19.019, LUTLP-1 0 on both; the WNS predictions hold on the endpoint rule, the area rows were refuted and the growth attributed (below) |
| attribution A / B | RTL + synth lanes | B sealed: Q-15's cone is not the growth; A's area read before its seal: D3 moves the cones' hierarchy home, it is not the logic; the growth is the DYN unit's unregistered result chain (90 -> 105 sources), real logic from item D; the per-module own-cell key is refuted as an instrument. A FAILED TO ROUTE after ~5 h (congestion: 242 signals, 171 overlaps; no bitstream, no routed WNS); routability at ~178k LUT is marginal and netlist-dependent |
| vp2b timing contingency | RTL lane | NOT BUILT: vp2's canary passed. The section stands as the record; the fix it names (collapse the DYN unit's sends onto one result register, cycle-neutral) is DEFERRED, not scheduled, for whatever bitstream follows |
