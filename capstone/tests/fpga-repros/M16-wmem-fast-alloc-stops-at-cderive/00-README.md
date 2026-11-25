# M-16: a protected wmem BLOCK_FAST allocation loop stops the core at its CDERIVE (silicon only, P1)

**What it is.** M-16 (`docs/ref/ISSUES.md`) is the Sublet-protected tshark not returning on silicon. It is
intermittent on bitstreams F and P1, and returns on the QEMU twin. This folder holds a 74-line program,
`src/tsapp-stress.c`, that runs only tshark's wmem allocation path. It stops the core in the same way, at the same
instruction, after about 2·10^7 allocations: about 11 minutes on P1, against 40-68 minutes per sample with tshark.
On bitstream B5b-2 (`…-r75d-b5b2n_904cd9c6b`), with tshark's one-operand allocator revision, it ran 1.17·10^8
allocations in one boot on 2026-11-25 without stopping ("On B5b-2" below).

**Wrong symptom? Siblings, so a reader with a neighbouring symptom is redirected:**
- `../R27-revnode-orphan-response/`: a silent deadlock after a pipeline flush during the DYN unit's revocation-node
  query, fixed in RTL at `66c4e7517`. M-16 shares the shape: silent, no trap, a CAPSTONE_DYN op valid at the LSU
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
| sw=220 `{lsu_ctrl.valid,lsu_ready_o,0,0,fu}` | `0x8c` (a valid CAPSTONE_DYN op at the LSU head, `lsu_ready_o` = 0) | `0x8c` | `0x8c` |
| sw=225 `{tbe,wstore,wload,wrev,domsw,stall,memwr,memwait}` | `0x92` | `0x92` | `0x96` (`stall` set) |
| sw=227 | `0x03` | `0x07` | `0x07` |
| sw=228 `{1,dom_switch_idx}` | `0xb8` | `0x9f` | `0x9f` |
| sw=224, 226, 229, 192-195, 219, 221, 222 | `0x05`, `0x00`, `0x00`, all `0x00`, `0xd7`, `0x02`, `0x00` | same | same |
| mepc / rev-node head | `0xffffffff80007376` / `0x0071` | same | same |
| serving index | `0x1381` | `0x14b9` | `0x11de` / `0x19fb` |

- **sw=220's bit 6 is `lsu_ready_o`, not "bypass empty".** The runner's label and the port comment say bypass empty;
  the driver is `assign lsu_head_debug_o = {lsu_ctrl.valid, lsu_ready_o, 2'b00, 4'(lsu_ctrl.fu)}`
  (`core/load_store_unit.sv:341` at `715bdd1fe`; `:412` at `904cd9c6b`, per the RTL lane). So `0x8c` is a valid op of
  unit 12 (CAPSTONE_DYN) at the LSU head with the LSU not ready. sw=219 `0xd7` = 215 is that op's `fu_op`; the RTL
  lane decodes it as LDC from `ariane_pkg`'s enum with `CAPSTONE_SUBLET_FMT` defined (not re-derived here).
- The other apertures are driven at `715bdd1fe`: 193 `store_buffer.sv:101`, 194 `store_unit.sv:118`, 221
  `store_unit.sv:414`, 219 `load_store_unit.sv:342`, 222 the WT adapter's `tag_debug_o` (`cva6.sv:2378`).
- **Not traced to a driver:** sw=224-228, the trap log (255), and the multi-byte rows (commit pc, mepc, tval, rev-node
  words). At `715bdd1fe` each of their signals is connected to a submodule output port in `core/cva6.sv`, but they
  are not traced to an assignment inside that module. sw=195's net was connected the same way and is dead, so read
  these rows' labels as unverified until they are. sw=227's bit 2 is `dom_switch_data_req.write_en`, the domain
  switcher's request, not the LSU's (`core/cva6.sv` 224-bank case `5'b00011`).
- **sw=195 and sw=229 carry no information.** Both show `load_state`, which comes from `load_unit.sv`'s `state_o`.
  That output is declared (`core/load_unit.sv:102` at `715bdd1fe`, `:109` at `904cd9c6b`) and never assigned, so
  both apertures read 0 on every boot, wedged or not. Found by the RTL lane on 2026-11-25 and checked here with
  `git show <rev>:core/load_unit.sv` and the `load_state` mux lines in `core/cva6.sv` (`5'b00011` in the 192 bank, `5'b00101` in the 224 bank).
  Read nothing into their 0x00.
- sw=227's differing bit is bit 2, `dom_switch_data_req.write_en` (see above).
- sw=228 is the domain-switch index; the stress boot's domain set is not tshark's.
- The serving index has differed between every pair of non-returns so far.
- **Caution on the pc:** it is scoreboard slot 0, which is exact only shortly after a flush. A hot allocator would also
  leave residue there. The pc names the site the allocation loop occupies; it does not prove the stop is that
  instruction.

## On B5b-2: no stop in 1.17·10^8 allocations (board 2, 2026-11-25)

**What differs from P1.** The bitstream is B5b-2, `…-r75d-b5b2n_904cd9c6b`. Its trap route makes the two-operand CREVOKE
illegal, so the program links tshark's one-operand wmem 0001 (Fig. 10's revision), not the revision P1 ran. A
difference between the two boots therefore cannot be assigned to the hardware alone.

**Build.** `tsapp-stress.c` is unchanged. It is compiled with the B5b-2 protected tshark tree's own `tshark.c.o`
compile line, and linked with that tree's recorded image link, with `tsapp_stress.o` in `tshark.c.o`'s place.
- Controls: the recorded compile line reproduces `tshark.c.o` byte for byte, and the recorded link reproduces the
  tree's `tshark.dom` byte for byte.
- Recipe: `<hdd>/capstone/nested/b5b2/build-stress-trap.sh` (`f928ae1c…`).
- Image `118ac0e7…`; the board runs its strip, `12c63983…` (20,982,240 B). No illegal encoding (`scan_illegal.py`).
- The CDERIVE word `0xa283955b` of `wmem_block_fast_sublet_alloc` sits at offset 0x4f0 here, so it runs at
  `0x3ff67bc234`.

**Run.** One boot, firmware `2b19fc2a…` (21,657,608 B), read back by name before and after.
- /init (the trap route's) runs two stages, each under a 3,600 s watchdog, and prints a tick every 120 s:
  - p0 = `100 212 10 1`, then p1 = `300000 212 10000 1`;
  - p2 = `300000 212 10000 1`.
- The overlay is `<hdd>/capstone/nested/b5b2/make-boardS.py` (`dc44c9f2…`).
- The unprotected image did not fit under the 26,000,000 B firmware cap, so it ran only on the twin.
- Pre-registration: nested-workloads `9b7c370ba5c7`. At P1's point estimate it predicted p1 would not return, with
  probability 1 − 0.0033.

**What it read.** Board lines: `results/b5b2-board2-stressS.txt`. Twin lines: `results/twin-trap-stressS.txt`. Reader:
`src/read-boardS.py` (`d22d1262…`), run as `read-boardS.py <board.txt> <twin.txt>`.
- Control `smoke.dom rc=7`.
- **p0, p1 and p2 all returned with rc 0**, and every STRESS line equals the twin's (12, 32, 32). 116,616,408 fast
  allocations were completed.
- There was no watchdog line, and 27 ticks. p1 took 1,432.4 s, 47.7 s per 10,000 packets; P1's native route took
  about 65 s.
- No apertures were read, since nothing stopped.

**What it says.**
- The hazard on B5b-2 with this allocator revision is **below 2.6·10^-8 per allocation (95 %)**. That is about half
  P1's point estimate, and inside P1's single-draw interval. It is a bound, not an absence.
- **The two boots do not show different rates.** If P1's boot (one stop in about 2.0·10^7) and this one (none in
  1.17·10^8) shared one rate, the single stop would land in P1's boot with probability 0.15. This test was computed
  after the boot.
- So the reductions below need a bitstream on which the stop has been seen. As of 2026-11-25 that is P1.

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
- `src/read-boardS.py`: the B5b-2 boot's reader.
- `results/b5b2-board2-stressS.txt`: the B5b-2 boot's result lines.
- `results/twin-trap-stressS.txt`: its trap-route twin's lines.
- `SHA256SUMS`: these files.
