# Shadow-tag traffic only where tags live: a page bitmap in the AXI adapter (plan, 2026-11-12)

## Why

Against ordinary RISC-V, the same Plain binary runs slower on the Capstone SoC. That SoC factor is nab 1.21, imagick
1.51, lbm 1.62 on V1f stock (train PREREG Addendum 46), and mcf about 1.6-1.8. Capstone's own runs inherit it:
Capstone / riscv is xz 1.58, mcf 2.08, lbm 1.50, nab 1.31 and imagick 1.72 on tonight's clean C.5a runs.

The measurement bitstream with the shadow-tag FSM tied off (capstone-ariane measure-notag-v1f 6806732fb,
`needs_tag = 1'b0`) answers where the factor comes from:

| Plain test | riscv | notag | V1f | Share of the gap the FSM explains |
|---|---|---|---|---|
| imagick | 11.17 s | 11.18 s | 16.89 s | 99.8 % |
| nab | 664.14 s | 664.07 s | 802.37 s | 100.0 % |
| lbm | 1,098.43 s | 1,098.38 s | 1,774.28 s | 100.0 % |
| getppid (cycles) | 1,018 | 1,054 | 1,891 | 95.9 % |

So for ordinary code the factor is the shadow-tag FSM in `core/cache_subsystem/wt_axi_adapter.sv`, and only that.

## What the FSM does today (1fc2cd5aa; rtl-oracle read, checked by the rtl lane)

- `needs_tag` holds for every D$ LOAD_REQ and STORE_REQ in [0x8000_0000, 0xBC2D_2D2D) (`:156-158`). The PTW's misses
  are included: they reach the adapter as D$ loads (port 0).
- **Store.** The D$ is write-through, so every store is followed by a 1-byte AXI tag write (ID 0) to
  `0xBC2D_2D2D + ((paddr - 0x8000_0000) >> 4)` (`:445-460`).
- **Load miss.** It takes a second 1-byte AXI read (`:467-482`), and the data return is held until that tag beat
  arrives (`:828-867`).
- **Before the tag read**, the load waits in TAG_WAIT until every outstanding tag write has its B response
  (`:425`, `:441`). Only one tag read may be in flight (`tag_rd_inflight_q`).
- `arb_gnt` is withheld in TAG_WAIT and in TAG_IDLE with `needs_tag`, so I$ fills wait too (`:183-188`).
- A load with `tag_pend_data_in = 0` already returns tag 0 without any tag beat (`:861-867`). That is the path every
  non-data-region load uses.

## Design v1: a "page may hold tags" bitmap

**State.**
- One bit per 4 KiB page of the data region: 246,483 bits, about 7 BRAM36.
- A reset sequencer clears it, about 3.9 k cycles at 64 bits per cycle. The adapter holds its grant until the clear
  is done.

**Invariant.** Bit 0 means every tag of the page reads as 0. The DRAM tag bytes of such a page may hold anything.

**Load miss, page bit 0.** No tag read. `tag_pend_data_in = 0`, so the existing zero-tag return leg serves it, with
no TAG_WAIT drain and no tag beat.

**Store, page bit 0, ctag = 0.** No tag write.

**Store, page bit 0, ctag = 1** (STC, the dom-switch store, a rev-node write under the virtual-context flag):
1. ZERO_FILL: write 0 to the page's 256 tag bytes. That is 33 single-beat, byte-enabled 64-bit writes, since the
   tag base is 0xBC2D_2D2D and the run starts at byte 5 of a word. They are single beats, so the 1-in-16 pages whose
   tag run crosses a 4 KiB boundary need no burst split. Each write is ID 0 and is counted in `tag_wr_pend`. The
   counter is widened, or the fill drains in chunks, so it cannot wrap.
2. Set the bit only after every fill write is counted.
3. Then the normal TAG_WR for the store's granule. It is ID 0 too, so AXI orders it after the fill.

**Page bit 1.** Today's behaviour, unchanged: tag write on every store; drain plus tag read on every load miss.

**Lookup.**
- The bitmap is read from the FIFO head's paddr into a registered lookup. It is valid for one head and cleared when
  that head is granted.
- In TAG_IDLE, `arb_gnt`, the FSM branch and the `tag_pend` push all use `needs_tag & page_bit_q`, which is the
  registered value. No combinational BRAM output reaches `arb_gnt`.
- Cost: one cycle per D$ memory request in the data region. The lookup is dropped on every grant, so each new FIFO
  head waits for its own read: measured +1.3 % on Plain lbm (train PREREG Addendum 56). Next revision: the lookup is
  valid without a wait when the head's page lies in the registered word (64 pages) and no bitmap write happened in
  the previous cycle.

**What it changes besides speed (deliberate, recorded).**
- Tag bytes left in DRAM by an earlier boot, or written into the tag region by a JTAG or software store, are invisible
  on a page until its first tagged store. That page's fill then zeroes them.
- Today such stale or forged bytes are read as tags. So this is stricter than today.
- AMOs stay outside the tag path. The S-06 P6 residual is unchanged on bit-1 pages and gone on bit-0 pages.
- Bits never clear during a boot. A page that once held a capability stays on the slow path until reset.

## Expected effect, and the open question that decides v2

- **Plain on the new bitstream ≈ notag.** Ordinary programs write no tags, so the only bit-1 pages hold the monitor's
  and module's structures.
- **Capstone code pays only on pages that ever held a capability.** That leaves array data (lbm, nab, xz buffers,
  imagick pixels) free, and heaps of pointer-linked objects slow:
  - mcf's node and arc arrays carry pointers, so they are capabilities;
  - gcc and perlbench heaps.
- So v1 should take most of the factor from lbm, nab, xz and imagick, and little from mcf.
- **v2 is for tagged pages.** It needs no ordering change:
  - a pending tag-write address list, retired in order on each ID-0 B, instead of the global drain;
  - more than one tag read in flight, ordered by `tag_pend`;
  - a small tag-word cache: one 64-bit tag word covers 8 granules, 128 B of data.
- **Decision input before synthesis.** A QEMU census per benchmark: the share of data accesses that land on pages
  which have held a capability. If mcf, gcc and perlbench are mostly on such pages, v2 goes into the same bitstream
  (CLAUDE.md: batch).

## Verification, in order

1. A directed RTL sim test (rtl-sim skill), with every condition created, not just present:
   - (a) In M-mode, store a non-zero byte into the tag region for granule G of page P. A load of G reads tag 0, with
     the bit 0.
   - (b) STC into P, then a plain load of a neighbouring granule, back to back. The neighbour reads 0, the STC'd
     granule reads 1, and the fill's B responses are counted before the read.
   - (c) A first tagged store into a page whose tag run crosses a 4 KiB boundary (the 0xF2D page).
   - (d) An untagged load of a page whose DRAM tag bytes were set before reset reads 0.
   - (e) Plain store and load stream: zero tag AXI transactions, counted through `tag_debug_o`.
   - (f) Positive control: the bitmap forced to all ones. (a) then reads tag 1, as today's silicon does.
2. The full Capstone sim sweep. It must stay at its classification, with no PASS/TIMEOUT flips.
3. `verif/sim/rtl-lint-gate.sh` PASS. Grep the UNOPTFLAT list for wt_axi_adapter before any edit to the `arb_gnt`
   cone.
4. claim-auditor on the invariant and on the lookup and ordering.
5. Synthesis-only run under a 40 GB guard. The per-module LUT and BRAM delta travels with the hash.
6. Full build. Seal, then a Tier 1 boot, then the Capstone control rungs.
7. Plain imagick/nab/lbm against notag, then the Capstone test set.

**Predictions** (each is stated before its run):
- Plain on the new bitstream is within 2 % of notag on imagick, nab and lbm.
- Capstone in-context cycles fall on lbm, nab, xz and imagick.
- mcf in-context falls by less than 15 % with v1 alone.

## v2: a tag-word cache for marked pages (design, 12 Nov, before any RTL; revised after claim-auditor)

**Why.** On silicon v1 took 25-34 % off Capstone imagick and lbm but only 5.6 % off nab (train PREREG Addendum 57).
The Plain bitmap removed 138 s of nab's tag path; Capstone nab kept most of it. The working hypothesis: nab's
pointer-holding records make its data pages marked. On a marked page each store still writes its tag byte, and each
load miss still drains every outstanding tag write and then makes a second, serialised memory round trip.

**State.** A direct-mapped tag-word cache (TWC):
- 256 entries, indexed by the tag word address bits [10:3]. One tag word is 8 tag bytes, 128 B of data, so the cache
  covers 32 KiB, the D$ size.
- Each entry: a valid bit (flops), the word address bits [31:11], and the 8 tag bytes whole. The D$ receives the whole
  byte (`dcache_rd_shift_user_d[0]`), so a cached bit would not be equivalent.
- A sticky `twc_off` bit, cleared only by reset.

**Invariant.** A valid entry holds the bytes DRAM will hold once every tag write the adapter has issued has completed.

**Lookup.** Address and data are read every cycle at the D$ FIFO head's tag word (BRAM, registered, like the bitmap).
The registered result serves the head only if:
- the head's granule (paddr[31:4]) is the one it was read for;
- no TWC change (install, invalidation, `twc_off`) happened in the read cycle.

So a head waits one cycle whenever its granule changes, and only on a marked page. A tag word does not line up with
128 B of data (the base is at byte 5 of a word), and comparing word addresses would put an adder into the arb_gnt cone.
No decision about a head is taken without a valid lookup: the head waits, as with the bitmap.

**Load miss on a marked page.**
- **Hit:** no tag read, no TAG_WAIT. The cached byte is pushed with the request into the tag-pending FIFO, which becomes
  {beat, override, byte}. At the data's last beat the byte goes into the user lane, as a tag beat would.
  - The byte is taken at grant. Every older store's TAG_WR was issued (the store head is popped only on its TAG_WR
    grant), so the byte includes it. No younger store's write is included.
- **Miss:** today's path: drain, then the tag read. The read is the whole aligned 8-byte word (size 3) at the same bus
  cost. Its beat installs the entry unless the read was poisoned.

**Poison.** A fill read is not installed if any of these happened from its issue up to and including the cycle its beat
is consumed (the check in that cycle is combinational):
- a TAG_WR to the same word;
- any TAG_FILL write, to any word. 1 tag word in 32 holds bytes of two pages, so a fill can rewrite part of a word that a
  marked page's read is fetching (claim-auditor 2a);
- `twc_off` being set.

**Store on a marked page.**
- With a hit and the cached byte equal to {7'b0, ctag}: no TAG_WR.
- Otherwise TAG_WR as today, and its issue **invalidates** the entry at that index. There is no byte merge, so a stale
  hit can never overwrite another word's entry (claim-auditor 5i/5ii). An invalidation and an install in the same cycle
  at the same index: the invalidation wins.

**Writers of tag bytes, enumerated.** The invariant holds only against writers the adapter sees:
1. TAG_WR: invalidates its entry and poisons a same-word fill in flight.
2. TAG_FILL: clears every valid bit and poisons any fill in flight.
3. **Software writes into the tag region** set `twc_off` for the rest of the boot and poison any fill in flight. These
   are D$ stores, AMOs or SC (ATOMIC_REQ) whose paddr lies in [0xBC2D_2D20, 0xBFF0_0000).
   - The lower bound is the 16-byte granule that holds the first tag byte. A store at 0xBC2D_2D20..2F is a
     data-region store that writes tag bytes (claim-auditor 3).
   - Their B responses are not counted in tag_wr_pend, so TAG_WAIT does not order them before a tag read (claim-auditor
     2b). Turning the cache off makes that today's behaviour again, not a lasting stale entry.
   - No firmware or module source writes the tag region (train PREREG Addendum 56). Only tests forge tags this way, and
     a boot that does it falls back to v1.
4. **Not seen:** the debug module's system-bus master (`dm_axi_m_req`, ariane_xilinx.sv:398). The TWC is cleared only
   by reset. A debug-bus write into the tag region while the core runs can leave a stale entry; a fence does not flush
   the TWC. The board flow loads over JTAG before the core runs; the wedge diagnostic only reads. Recorded, not handled.
5. There is no other writer. The bitstream's top is ariane_xilinx (the synthesis report's design name), and its
   crossbar has two master ports (`NrSlaves = 2`, corev_apu/tb/ariane_soc_pkg.sv:20, "actually masters"):
   - slave[0], the core;
   - slave[1], the debug module (ariane_xilinx.sv:659-672).

   Revocation reads and writes use D$ ports 2 and 3 (cva6.sv:2421-2423), so they pass through this adapter.

**Expected effect.**
- Load misses that walk an array hit 7 of 8 times.
- Untagged stores to words whose tags are known to be 0 write nothing.
- With fewer outstanding tag writes, a miss's drain is shorter.
- The targets are nab, mcf, gcc and perl.

**Verification.** The same order as v1.
- Directed test tag-word-cache, each case created:
  - a hit with tag 1, and a hit with tag 0;
  - a store skip, and a store that changes the tag (the entry invalidated, DRAM checked after a conflict eviction);
  - an older tagged store in the write buffer racing a younger load's fill read on the same word;
  - the same race on a boundary word, against a page's first tagged store;
  - a software write into the tag region (last, since it turns the TWC off).
- `+define+TWC_TRACE` prints every tag read and write, so the counts tell the models apart, not only the verdict.
- The 95-test sweep against v1.
- The lint gate, and claim-auditor on the RTL.
- A synthesis-only gate, then the full build.

It ships in one bitstream with the same-word lookup bypass (perf/tag-page-bitmap 7b929ac71).

## Fairness

- This changes the Capstone SoC only. The riscv baseline is untouched.
- It is a hardware implementation of the existing tag semantics, and it reads stricter than today.
- It is reported as the bitstream it is: a new sealed hash, with the per-module resource delta beside the
  performance numbers.
