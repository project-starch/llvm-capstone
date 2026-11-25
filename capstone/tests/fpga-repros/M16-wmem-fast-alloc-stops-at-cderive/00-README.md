# M-16: a protected wmem BLOCK_FAST allocation loop stops the core at its CDERIVE (silicon only, P1)

**What it is.** M-16 (`docs/ref/ISSUES.md`) is the Sublet-protected tshark not returning on silicon. It is
intermittent on bitstreams F and P1, and returns on the QEMU twin. This folder holds a 74-line program,
`src/tsapp-stress.c`, that runs only tshark's wmem allocation path. It stops the core in the same way, at the same
instruction, after about 2·10^7 allocations: about 11 minutes on P1, against 40-68 minutes per sample with tshark.

**Wrong symptom? Siblings, so a reader with a neighbouring symptom is redirected:**
- `../R27-revnode-orphan-response/`: a silent deadlock after a pipeline flush during the DYN unit's revocation-node
  query, fixed in RTL at `66c4e7517`. M-16 shares the shape: silent, no trap, a CAPSTONE_DYN op valid at the bypass
  head. It was observed on P1, which descends from that fix, so it is not that instance unless the fix is incomplete.
  Nothing here shows which.
- `../S13-o1-dyn-rev-node-hang/`: a hang in the DYN/rev-node path at `-O1` with no exception. It is not reproducible
  from its folder, so it cannot be compared with this one.
- A domain that faults with a cause, or prints a wrong answer, is not this issue. Here nothing is printed after the
  stop, and the twin's run of the same image returns.

## The program

`src/tsapp-stress.c` (sha256 `42b7ad10…`). It is not a port file and it is not tshark. It uses tshark's two wmem scope
kinds with wmem patch 0001 (Sublet lifetimes for `block_fast`):
- a packet scope (`WMEM_ALLOCATOR_BLOCK_FAST`) with K allocations per packet, then `wmem_free_all`;
- a file scope (`WMEM_ALLOCATOR_BLOCK`). One allocation in 12 goes there, half of those are freed again with
  `wmem_free`, and the scope is reset every 1,000 packets.

Sizes come from a fixed LCG: 70 % 8-64 B, 25 % 65-512 B, 5 % 513-4096 B. Each allocation's first and last bytes are
written and read back into a checksum. The output is a function of argv only, so the twin's lines are the reference
for the board's. The arguments are `packets K report seed`. A `STRESS pkt=…` heartbeat every `report` packets places a
stop to within `report` packets.

**Build (native route, P1).** The program is compiled with the protected tshark tree's own `tshark.c.o` compile line.
It is linked exactly as `tshark_m5.dom` is linked, with `tsapp_stress.o` in `tshark.c.o`'s place: the same wmem 0001
objects (dev's Q-15-repaired revision), glib, `tsapp-heap.o`, the SDK's libc and `-Wl,--no-relax`.
- Control: the same link of the tree's own `tshark_m5.o` reproduces `tshark_m5.dom` byte for byte, in both the
  protected and the unprotected tree.
- Recipe: `<hdd>/capstone/nested/tshark/stress/build-stress.sh` (`ad719e29…`).
- Image `tsapp_stress.dom` `c93cc972…`; the board runs its strip, `84d347d8…` (20,208,504 B).
  - Both stay outside the repo for their size, under `<hdd>/capstone/nested/tshark/stress/sublet/`.
- Against the unprotected build: +36 CDERIVE and +5 CREVOKE.

**The site.** `wmem_block_fast_sublet_alloc` holds the same word at the same offset as in tshark.
- tshark's stop pc `0x3ff6780f04` is offset 0x45c into it, word `0xa283955b`: the CDERIVE of the inlined
  `capstone_cap_derive`.
- In this image that word is at vaddr `0x77c9ee`, which loads at `0x3ff676c9ee`.

## How it was run on P1

- Bitstream P1 `caplifive_ids55-…-tagoverlap6c-pagedtbl_d83d3a633`, read back by name before and after. Board 1,
  2026-11-25 11:41-14:49 (+0800).
- Kit `t3-smoke/bkit91-k2n-native-p1-tree-cscfix` (2^18 records), M-16's own.
- One stage, two commands, both protected: s0 = `100 212 10 1`, s1 = `400000 212 10000 1`.
- Budget 10,800 s, then the runner read the apertures (`BAKED_WEDGE_APERTURES=1`).
- **The unprotected image did not ride along.** Two images put the firmware at 30,051,848 B, over the kit's 26,000,000 B
  cap. The unprotected control is the twin, where both commands return, together with M-16's board record, where the
  unprotected tshark returned in every boot.
- Pre-registration: nested-workloads `6659e74b24ca`. Result: `832faa980e3b` (`docs/ref/nested-allocator-workloads.md`,
  "M-16: a stress form of the allocation path").

## What it read

Result lines are in `results/p1-board1-stress1.txt` and the twin's reference in `results/twin-native-stress1.txt`. The
reader is `src/read-stress.py` (`0384cfba…`); usage `read-stress.py <board.txt> <twin serial.log>`.

- **s0 returned.** All 12 STRESS lines equal the twin's.
- **s1 stopped.** All 11 of its STRESS lines equal the twin's. The last is `pkt=100000 fast=19432859`, and the twin's
  next is `pkt=110000 fast=21375862`.
  - So the stop lies between 1.943·10^7 and 2.138·10^7 fast allocations, 10.2-11.2 tshark runs' worth.
  - The heartbeats had arrived 60-75 s apart. After the last one the console was silent for the remaining ~10,000 s
    of the budget.
- **Apertures** after the budget, verbatim in the results file:

| aperture | stress (P1) | tshark tsu-r1 (P1) | tshark tsu-p1 / tsu-r2 (P1) |
|---|---|---|---|
| commit pc (slot 0) | `0x3ff676c9ee` (this image's CDERIVE) | `0x3ff6780f04` (tshark's CDERIVE) | `0x3ff6780f04` |
| trap log (sw=255) | `0x89` | `0x89` | `0x89` |
| sw=220 `{lsu_ctrl.valid,bypass_empty,0,0,fu}` | `0x8c` (CAPSTONE_DYN at the bypass head) | `0x8c` | `0x8c` |
| sw=225 `{tbe,wstore,wload,wrev,domsw,stall,memwr,memwait}` | `0x92` | `0x92` | `0x96` (`stall` set) |
| sw=227 | `0x03` | `0x07` | `0x07` |
| sw=228 `{1,dom_switch_idx}` | `0xb8` | `0x9f` | `0x9f` |
| sw=224, 226, 229, 192-195, 219, 221, 222 | `0x05`, `0x00`, `0x00`, all `0x00`, `0xd7`, `0x02`, `0x00` | same | same |
| mepc / rev-node head | `0xffffffff80007376` / `0x0071` | same | same |
| serving index | `0x1381` | `0x14b9` | `0x11de` / `0x19fb` |

- sw=227's differing bit is `data_req.write_en`.
- sw=228 is the domain-switch index; the stress boot's domain set is not tshark's.
- The serving index has differed between every pair of non-returns so far.
- **Caution on the pc:** it is scoreboard slot 0, which is exact only shortly after a flush. A hot allocator would also
  leave residue there. The pc names the site the allocation loop occupies; it does not prove the stop is that
  instruction.

## What it shows, and what it does not

- **The stop needs nothing of tshark beyond this allocation path.** There is no dissection, no capture input, a far
  smaller live heap, and no I/O beyond the heartbeat. It holds the same function, the same pc offset and M-16's
  aperture signature.
- **Hazard.** One stop after about 2.0·10^7 allocations gives about 4.9·10^-8 per allocation. The 95 % interval from
  this single draw is 1.2·10^-9 to 1.9·10^-7.
  - This lies at the low end of M-16's tshark band (4.7·10^-8 to 7.4·10^-7).
  - It is one draw. The interval is the result, not the point.
- **Not shown:**
  - which instruction stops: see the caution on the pc;
  - whether this is a deadlock or a livelock: the console was silent, and the apertures were read once;
  - which bitstream change introduced it, and whether the unprotected image can stop: the unprotected image did not
    run on the board.

## Next reductions

Each step keeps the heartbeat and the twin reference:
1. BLOCK_FAST alone: no file scope.
2. A fixed allocation size.
3. A loop of `capstone_cap_derive` on a fixed parent, outside wmem.

Each one that still stops halves what the hardware side has to look at. The first one that returns names the
ingredient. After that, a directed `.S` for simulation, which is CLAUDE.md's rule 8.

## Files

- `src/tsapp-stress.c`: the program.
- `src/read-stress.py`: the P1 boot's reader.
- `results/p1-board1-stress1.txt`: the P1 boot's result lines and aperture block.
- `results/twin-native-stress1.txt`: the twin's reference lines.
- `SHA256SUMS`: these files.
