# Exact bounds on silicon: (i) a wider capability against (ii) a side store. Costed comparison for the lead (2026-10-16)

*RTL lane, deliverable 2 of `2026-10-16-virtual-profile-on-silicon.md`. Numbers come from the RTL at d93a8d66b
(capstone-sublet b7e3287e4), the FPGA configuration (`capstone_cv64a6_imafdc_sv39_config_pkg.sv`: AXI data 64 bits, user
sideband OFF, D-cache line 128 bits = one granule, write-through), QEMU's exact-bounds prototype at d35af47b163c, and the
compiler's datalayout. Where a figure is an estimate it says so and names what it rests on.*

## What both options share, and what the 2026-10-10 audit missed

Exact bounds have to be IN THE CORE under either option: the LSU compares decoded 64-bit base and end today
(`load_store_unit.sv:336, 1339-1340`, fed by `decompress_cap_tagged`), and an exact design replaces the decode with the
stored values. So the register-side cost is common to (i) and (ii):
- the capability metadata register file, scoreboard lanes and forwarding (`issue_read_operands.sv:1748`, `cva6.sv:111-134`:
  a 65-bit {tag, metadata} lane beside the 64-bit cursor) grow by 128 bits (base, end) per entry: 32 registers x 128
  = 4 Kbit of flops in the file plus every scoreboard and bypass mux widened from 65 to 193 bits;
- the CPMP's 16 entries (`pmp_data_if.sv:44`, 128 bits each, decompressed every cycle at `:125-130`) store exact bounds
  instead and lose the decoders: a net simplification;
- the domain switcher's save slot (129 bits logical, `capstone_dom_switcher.anvil:21-24`) and the switcher's save area;
- every compress site that produces a capability (6 in core/) keeps producing the 27-bit field for the in-memory word's
  physical-profile meaning and additionally carries the exact pair; every decompress site (9) is bypassed under the flag.
This common part is one bitstream cycle on its own, sits in issue/forwarding (the cone that set WNS on every 2026-10
build) and is the first thing to synthesise, before either memory format.

The audit's "a quarter of memory" for (ii) was wrong: QEMU's map is `capboundsfat_t bounds[4*64]` per 4 KiB page
(`cap_mem_map.h:19-20`), 24 bytes per 16-byte granule, i.e. 1.5x of memory if provisioned densely, 1.0x for base+end
without the cursor. Its "3x traffic" is re-derived below from the adapter.

## Option (i): a 32-byte capability in memory

Format: granule 0 = {cursor 64 | metadata 64} as today, granule 1 = {base 64 | end 64}; the tag covers the pair.
- **Units.** D-cache line 128 -> 256 bits or a two-line capability access (`wt_dcache_mem.sv`, `wt_dcache_ctrl.sv`,
  the wbuffer entry and its byte-enable overlay `wt_dcache_wbuffer.sv`, the refill user byte at `wt_dcache_mem.sv:358/501`
  that R-10 and R-29 live in); the AXI adapter's burst from 2 data beats to 4 (`wt_axi_adapter.sv:217-221`); the LSU
  request/response and the store buffer's forwarding (R-29's window) for a 256-bit datum; the tag SRAM unchanged (one bit
  per pair, or one bit per granule with the second granule's tag forced equal). The rev-node unit is untouched.
- **UNOPTFLAT cones.** Enters all 8 of the 40 loops in load_store_unit / load_unit / wt_dcache_ctrl / wbuffer /
  wt_dcache, including `rd_ctag` (`wt_dcache.sv:144`), the tag ring, by widening the data it carries.
- **Memory traffic.** STC: 4 data beats + 1 tag transaction (today 2 + 1). LDC miss: 4 + 1 (today 2 + 1). LDC hit: 0.
  Per capability access on AXI: 33 bytes against 17 today, 1.9x.
- **Compiler and runtime.** `sizeof(void *)` 16 -> 32 in AS200: the datalayout `-p200:128:128:128:64`
  (`TargetDataLayout.cpp:325`) becomes `p200:256:256:256:64`; every struct with a pointer changes size and alignment.
  The compiler lane measured the compiler half on dev's clang (2026-10-16): the carrier of `__intcap` goes from i128
  to i256, which legalises but costs 27 instructions per add against 6 (4.5x, llc -O2), and i128 appears 114 times in
  8 backend files; `Align(16)` has 27 hits, `ldc` 79 and `stc` 57 (a capability is no longer one granule, so the
  tag-preserving memcpy fast path is rebuilt on ldc/stc pairs); in the ports, `aligned(16)` 27 hits in 21 files,
  `(+15)&~15` 8, `ROUND8` 24, each load-bearing (one such fix landed the same day, 423d13f5b81f). musl adapts through
  `sizeof` conditionals and is not the bill. mallocng's slot geometry (UNIT 16, `struct meta` bounded to a page, patch
  0009:278,307), the ABI v5 headers' size assertions (`runtime/common/*.c`) and the CSRUNV frame (576/640/656) all
  change, and the virtual QEMU needs the same format to stay the reference. The compiler lane's effort figure is three
  to five weeks of compiler, runtime and ports, dominated by the ports' alignment audit and re-measuring every carve
  count. This re-ports the stack that was just landed and doubles the footprint of every pointer-rich structure,
  which changes the slot geometry exact bounds were adopted to protect.
- **Bitstream cycles.** 2 for the memory path (line width or two-line access, then the forwarding/wbuffer residuals the
  R-29 history shows this path produces) + the common 1 = 3, before any fault found on the board.

## Option (ii): a side store of exact bounds per capability-bearing granule

Format: the 16-byte word unchanged; a shadow region holds {base 64 | end 64} per granule (16 bytes; QEMU also stores the
cursor, which is redundant with the word). Authoritative only when the virtual-context flag is set; the physical
profile reads the compressed field as today. This is QEMU's own prototype (`x-capstone-exact-bounds`, scapctl bit 8).
- **Units.** The tag path is the template: the shadow TAG region already has a serialised TAG_WR/TAG_RD transaction per
  AXI access (`wt_axi_adapter.sv:156-157, 445-480`) and an L1 copy (`cap_tag_q`, `wt_dcache_mem.sv:146`). The side store
  adds a 128-bit L1 array beside `cap_tag_q` (same indexing, 128x the bits: for the FPGA D-cache that is the size of the
  data array itself), a BOUNDS_WR/BOUNDS_RD leg in the adapter's tag state machine (2 beats), its base register beside
  `CAP_TAG_MEM_BASE`, and the wbuffer carrying the pair with a pending capability store so a younger LDC forwards exact
  bounds (the R-29 class again, on a new field). The LSU reads the pair with the data on an LDC and writes it on an STC.
  Rev-node unit untouched. Line width and the data beats untouched.
- **UNOPTFLAT cones.** Enters `rd_ctag` and the wbuffer/load_unit loops (the forwarding of the new field), not the
  line-width logic; fewer of the 8 than (i), but the tag ring is the one that broke synthesis before (2026-08-21).
- **Memory traffic.** STC: 2 data + 1 tag + 2 bounds beats = 5 transactions (today 3), 33 bytes against 17, 1.9x,
  the same as (i). LDC miss: 5. LDC hit: 0 if the L1 bounds array is present, else 2 beats per hit. Provisioning: 1.0x
  of RAM dense (the tag region is 1/128); or per capability-bearing page with a directory, as QEMU's entry list does,
  which is a two-level access on a miss.
- **Compiler and runtime.** Nothing in the compiler: datalayout, i128 carrier, Align(16), ldc/stc and the ports'
  alignment all stay (the compiler lane's reading, conditional on the point below); the runtime reserves and maps the
  region as it does the tag region; the acceptance images run by sha256. **The side store is maintained by hardware
  with no software participation**: the LSU writes the pair on every capability store and reads it on every capability
  load, the adapter moves it with the tag beats, firmware reserves the region at boot exactly as `CAP_TAG_MEM_BASE` is
  reserved today. No allocator tracks it and the compiler emits nothing for it. (The compiler lane asked this question
  explicitly, since a software-maintained side store would invert the comparison; the answer is hardware.)
- **Bitstream cycles.** 1 for the adapter leg + L1 array + LSU read/write, 1 for the forwarding residuals = 2, + the
  common 1 = 3. Equal to (i) in cycles; the difference is where the risk sits (cache data path and every software
  structure for (i); the tag ring and a new shadow region for (ii)).

## Recommendation

**(ii), the side store, authoritative under the virtual-context flag, 16 bytes per granule, with the L1 bounds array.**
The hardware cost in cycles and traffic is the same as (i) to within the estimate's resolution; (i) additionally re-ports
the entire landed virtual stack and changes mallocng's slot geometry, while (ii) keeps the in-memory word, the physical
profile and every acceptance image unchanged and is the format the reference emulator already implements. The thing
(ii) must get right that (i) does not face is coherence of a second shadow region: the tag region's serialised
transaction already solves it for one bit, and the same ordering (data, then tag, then bounds, one AXI id) is the design.

What the lead decides: (i) or (ii); dense provisioning (1.0x RAM) against a per-page directory (QEMU's shape, cheaper in
RAM, a two-level miss). My recommendation is dense, because the FPGA has the RAM and the directory is a second
dependent walk on the exact path that carries the WNS.

## Format freeze proposal

Proposed, for the lead to freeze and the model's owner to hold:
- **The in-memory word: {revnode_id 30 | perm 4 (MANAGE, R, W, X) | cap_type 3 | bounds 27 (iE 1, T 12, B 14)}**, bit for
  bit as both sides have it today (`ariane_pkg.sv:638-642`, `cap_compress.c:30-37`). Nothing to change; freeze it.
- **27 bounds bits, not 28.** The 28th bit was the cursorless flag, gone from both encoders; the RTL's d34c26002 made
  room for MANAGE with it. The alignment rule the guide states for the compact profile (L >= 4 KiB needs base and end
  aligned to 2^(msb(L) - 9)) is the 27-bit field's rule and stays the physical profile's contract.
- **30 node bits, meaning a flat index under the virtual-context flag** (QEMU's 2^30 namespace, item A); the physical
  profile keeps packing {generation 14 | index 16} into the same field internally, which no software reads. **No
  generation bits in the word on either profile.** Under the flag the ABA defence is the collection sweep (H), as the
  guide specifies.
- **The encoding function**: the RTL's compress_bounds WITH renormalisation (2a3a87dfc / afcd19bea) is the reference
  until QEMU lands 4a21b485b01f's block (Q-16); after that the two are to be proven bit-identical by the shared model
  (`bounds-model.py` against QEMU's `cap_compress` over the 2,001,820-case set the helper lane already ran).
- **The exact pair under (ii)**: {base 64 | end 64}, cursor NOT stored (it is in the word), authoritative when the
  virtual-context flag is set, ignored by the physical profile; written on every capability store, read on every
  capability load; base region reserved by firmware.
- **The cursor stays in the word: a compiler requirement, not a density result** (the compiler lane, 2026-10-16). The
  compiler reads a capability's cursor with `PseudoTRUNC_CAP`, an `ADDI rd, rs, 0` that works because the cursor is
  the low 64 bits of the capability in the register file and the word (`CapstoneInstrInfo.td:2582-2585`); it does NOT
  use `lcc rd, rs, 2`, which traps on an untagged operand and brought back C-19 (a null-pointer comparison killing the
  domain at entry). Base and end are read with `lcc` selectors 3 and 4, so they can live behind the LSU; the cursor
  cannot. Any later revision tempted to move the cursor out of the word to free bits breaks this. The register-side
  tag-plus-metadata lane is unchanged under (ii), so reading a tagged register as an integer still raises
  UNEXPECTED_OPERAND, which the pseudo exists to avoid.
- **The record (item A)**: QEMU's word0 prev[31:0] / next[63:32], word1 depth[29:0] / parent[59:30] / flags[62:60]
  (`cap_rev_table_abi.h:8-30`) as the virtual-profile layout; the physical pool's 110-bit record stays as is.

## Proposed owner of the model's semantics

The plan asks for a proposed owner. Proposal: **the lane that owns capstone-qemu's `virtual-capstone` branch and the
guide (today the helper lane) owns the model's semantics**; a semantic change lands in QEMU, the guide and a Q-entry
before the RTL follows, and the RTL lane owns conformance (the Q-entries and the directed tests that read each one on
both sides). The format freeze above is the first entry in that owner's ledger. The lead names the owner.
