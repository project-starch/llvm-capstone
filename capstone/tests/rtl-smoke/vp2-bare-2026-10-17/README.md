# vp2's bare arms for `caplifive_vp2_bf3db3175.bit`: pre-registered board vectors (2026-10-17, before any vp2 run)

The readings the vp2 plan section (`docs/plans/2026-10-16-virtual-profile-on-silicon.md`, "Readings the bitstream must
answer") lists, as bare JTAG arms:
- item D's arms: va-runv..va-runv7, va-barrier, va-q15, va-csmint-emu, va-csmint-mon;
- item A's arms: va-flat, va-alloc with its three controls, va-ns-collide with its control, va-straddle;
- each with the controls its header names.

q17-order (bare M-mode) runs as `vp1-bare-2026-10-17`'s image, byte for byte.

**vb-trans and vb-trans-plain run on vp2 as their `VB_TABLE` builds** (`vb-trans-vp2`, `vb-trans-plain-vp2`;
capstone-ariane 1bcef8712). The flag-off build of the edited file reproduces vp1's two vb-trans hashes exactly. They
were held out of vp2 until that variant landed:
- On bf3db3175 every vb-trans arm reads csupstatus 5 / csupcause 25 where vp1 reads its pre-registered values.
- Since item A, a domain under `csupctl.virt` runs in the lifetime-table namespace at srevroot, so its capabilities'
  node 2 must be a valid record in a table. vb-trans sets no srevroot. The va-* arms each build such a table, which is
  why they read right.
- The plan's line "vp1's vb-trans arms and Q-17's" for vp2 was written without a run (the RTL lane, 2026-10-17).
- On vp1 (no item A) vb-trans is correct as pre-registered and stays in vp1's batch unchanged.

**Run on 2026-10-17, on the lead's own word** ("Flash authorisation" below), after vp1's first session ended as
pre-registered: the canary passed 9 of 9, all 25 arms read their vectors, set 7 matched d93a8d66b's 33 records
(`results/board-vp2-bf3db3175.result-lines.txt`).

**Build** (`build.sh`):
- The tests are copies of capstone-ariane `virtual-profile-2-d` fc679dc68, whose RTL is bf3db3175's (the five commits
  above it touch `verif/` only, 0 files elsewhere). They are checked equal at build time when `VP_SRC` is given; they
  were, from `git archive fc679dc68`.
- The wrapper, env and node-aware header are vp1-bare's, unchanged.
- The recorder's three scratch registers per test are chosen by `pick-regs.py` from the GPRs the test never names.
  va-csmint-mon names all 32; its choice and the reason are in `build.sh`.
- Toolchain check: this recipe with vp1's id characters reproduces vp1-bare's three hashes exactly.
- Results: `results/build.result-lines.txt`; `images/SHA256SUMS` (23 images).

**The board vectors are not the headers' simulation vectors.** The recorder's CAPPRINT creates its own capability with
CAPCREATE, and a CAPPRINT inside a domain cannot run as it does in simulation. Both can change a reading, and the node
counts in item A's arms are the obvious place. So each vector is taken from simulating THESE images (the ELFs
`out/*.elf`, `out/SHA256SUMS`, staged in `/tmp/capstone/vp2-bare-elfs/`) on bf3db3175's RTL, through
`make veri-testharness elf=...` (the RTL lane, 2026-10-17). An image whose simulation fails, or whose vector does not
express its header's claim, is rebuilt (a `BOARD_TEST_CH` guard, as vb-trans has) before it is pre-registered.

**The board vectors are complete:** all 23 images and vp1's two reused ones. They come from simulating each image as
built on bf3db3175's RTL (`results/rtl-sim-vectors.txt`). In va-csmint-mon's vector, -1 means 0xffffffffffffffff.
Three images were rebuilt from tests-only commits before their vectors were taken:
- va-runv6, from c55e3ff31: an arm fault in the test.
- va-csmint-emu and va-csmint-mon, from fc679dc68: `PHYS()` measured from `RAM_BASE` while their RAM capability's base
  is `_start`. On the board, `_start` sits 0x1000 higher, behind the wrapper's page.

**Flash authorisation (2026-10-17, the lead's own word in the board lane's session).** The board lane asked about
`caplifive_vp2_bf3db3175.bit`: sha256 776cdd8c40d985708a09d4d43ab1c5106414ae1347cc4436a711381d2c197475, routed WNS
-19.019, inside the plan's band but 3.2 ns worse than any bitstream that has run here. Its recommended option was to
flash after vp1's first session with a canary first, and to go back to vp1 if the canary differs. The lead answered
"Go", and then, asked for an explicit word, "Go with reflash, if u need, I confirm". That covers the vp2 flash and the
reflash back to vp1 on a canary difference. Then the lead gave the explicit form, naming the bitstream and the hash (in
Russian; translated): "Flash vp2 caplifive_vp2_bf3db3175 after vp1's first session, after the preflight on the sealed
sha256 776cdd8c40d985708a09d4d43ab1c5106414ae1347cc4436a711381d2c197475. First in the session, the control: S-16, the
Sublet floor and s1gap, twice each, stop on any difference." Asked two follow-ups, the lead chose:
- the functional rule for s1gap: its causes, validity, traps, GAP and count must match exactly; the eight cycle counts
  vary run to run on d93a8d66b itself, and are reported;
- "if vp1 as expected, go with vp2": vp2 goes ahead only if vp1's first session ends as pre-registered. Otherwise vp1
  stays resident and the chain holds.
- **When:** after vp1's first session completes (`AFTER-FLASH DONE`, not a STOP), and the preflight passes on the
  sealed sha256.
- **The canary comes first, before any vp2 arm:** the S-16 set, the Sublet floor and s1gap, each run twice. It stops on
  any difference from d93a8d66b's records, or between the two runs.
  - On a difference the board goes back to `caplifive_vp1_9ed9a2594.bit` and the session stops. The next step would be a
    vp2 respin.
- **Then:** this table's 25 arms in table order, then corpus set 7, which must be identical to d93a8d66b.

**The session, pre-registered before it runs (2026-10-17).** These are the board lane's private scripts; this is
what they do.
- **Start:** only after vp1's first session ends clean (its `AFTER-FLASH DONE`, no STOP, vp1 read back as resident).
- **Preflight:** on the sealed sha256 and WNS (-19.019, inside the band floor -19.596), with the DTS pairing.
- **Flash:** the readback must name caplifive_vp2_bf3db3175.bit.
- **The canary, nine comparisons, all of which must pass:**
  - The S-16 set and the Sublet floor, in each of two passes, exactly equal to d93a8d66b's records.
  - s1gap in each pass, by vp1's amended rule (88d670de65bc): functional readings exactly, the eight cycle counts
    reported.
  - Pass 1 against pass 2 by the same rules. So a run-to-run variation in any functional reading fails it; the cycle
    counts vary run to run on d93a8d66b itself.
- **If the canary does not pass:** a preflight on vp1's sealed sha256, a reflash to vp1 with readback, and a stop.
- **Otherwise:** this table's 25 arms in table order, one power cycle each. Each is compared with its board vector,
  count and values as 64-bit integers. Then corpus set 7 against d93a8d66b's record.


**Amended before the run (2026-10-17, 20:24): an image that never started is re-run once.**

On vp1's first session a JTAG `load_image` timed out before `continue` (s16st-esc-n8), so that arm produced no reading.
vp1's PREREG amendment of the same time gives the rule and its precedent. Here every bare run of the session goes
through `infra-retry.py`: the canary's S-16, floor and s1gap arms, and the 25 arms.
- A run that exits non-zero with no `] continue` line and no reading is run once more, with the same image. Its first
  attempt is kept beside the run, as `<run>.infra-fail-1`.
- A run that started is returned as it is.
- A second non-start is a missing reading, so the canary's "stop on any difference" fails, and the board goes back to
  vp1 as pre-registered.
- The 25 arms' outer timeout is 1200 s, up from 600 s: a load_image hang plus one retry needs about 10 minutes.
- The arms phase fails (rc 2) unless the pre-registered table gives exactly 25 arms. A dry run had read an empty
  table as done.

Dry runs, with only the innermost runner stubbed (the real run-capstl.sh and wrapper run):
`results/dryrun-infra-retry.txt`.
- One non-start injected in each kind of run: the canary passes 9 of 9, all 25 arms read as pre-registered, and each
  injected arm shows exactly one retry.
- A double non-start: the canary fails.

**Still open before a vp2 board run:**
- (Not needed: resident controls for the physical-pool builds. The `-off` builds run in vp2's own batch beside their
  flag-on arms. On vp1 or d93a8d66b their readings are not predicted: srevroot and the item-D CSRs do not exist there.)

| image | sha256 (16) | board_rec | board vector (from the simulation of this image) |
|---|---|---|---|
| `va-runv` | f764c436cdb52687 | 0x8000c000 | SR 15: `2 2 0x1234 2 0xb 0 0 0x55 0 0x77 0x77 5 0xb 0 9` |
| `va-runv-quantum` | 34934ab8b2271c8c | 0x8000c000 | SR 15: `2 1 0x1234 1 0 0 0 0 0 0 0 3 0x8000000000000010 0 9` |
| `va-runv2` | 7b8ab66e23607d31 | 0x8000d000 | SR 53: `2 4 4 0xd 0 0 0 0 5 0x1d 3 3 0xb 0 7 7 0 0 0x42 0x1d 2 2 2 0 0x99 0x1d 0x1d 0 0 2 0x80003ec0 2 0x1d 0 0 2 0x80003ec0 2 0x1d 0 0 2 0x80003ec0 2 0x1d 0 0 2 0x1234 2 0x1d 0x1234 0` |
| `va-runv3` | d967dcdea54e3a86 | 0x8000c000 | SR 16: `2 3 0 0 0 0 0x3ff8000000000000 0x20 0x3ff8000000000000 1 2 0x4004000000000000 0x40 0x3ff8000000000000 1 0` |
| `va-runv4` | e8ac8002db62b1e8 | 0x8000d000 | SR 16: `2 5 5 0x1e 0 0x33 7 1 2 2 2 0x44 8 2 0 0` |
| `va-runv5` | a5c00ebb2d79ed1b | 0x8000c000 | SR 33: `2 2 2 2 0 0 2 2 2 0 0 2 2 2 0 0 2 2 2 0 0 2 2 2 0 0 2 2 2 0xc 0xf 9 0` |
| `va-runv6` | 7922462bf7b1a21d | 0x80009000 | SR 50: `2 2 8 4 5 2 4 0x3000000100000001 0x300000000 0x3000000000000000 0x55 0x1000000100000001 6 3 3 0x10000000c0000002 0x500000004 0x66 0 0 0x2000000000000000 0x500000004 0x100000001 0xc0000002 0x19 0x19 1 1 1 1 1 1 1 1 0x3000000000000000 1 0x3000000000000000 0 0x180000001 0x2000000000000000 0x1e 0x3f 0x40 5 0x3000000fc0000001 0x3e00000000 0 0x2000000000000000 0x2000000fc0000001 2` |
| `va-runv7` | 8edbfa78d1ec7acf | 0x8000c000 | SR 11: `2 2 2 2 0 0 2 2 2 0 0` |
| `va-barrier` | 28202d0d516170e8 | 0x80007000 | SR 5: `4 0x1000000100000001 0x1000000100000001 5 6` |
| `va-q15` | ee0959d19d8f0a82 | 0x80010000 | SR 10: `0 0x1018 2 2 2 0x1d 0 0x77 4 0` |
| `va-csmint-emu` | c2defd4da8304921 | 0x8000e000 | SR 32: `0 4 5 0x3000000100000001 0x3000000000000000 4 0x8000a004 6 0x8000b018 0x1d 0 0x1d 0 0x1d 0 0xf 0x8000c000 0 6 7 0x3000000180000001 0x3000000000000000 0x1e 0 0 0x8000d200 0x8000d300 0x8000d200 4 0 0 0` |
| `va-csmint-mon` | 600b48c94fabddc3 | 0x80011000 | SR 46: `4 -1 -1 -1 4 0x8000b004 -1 6 0x8000c018 -1 0x1d 0 -1 0x1d 0 -1 0x1d 0 -1 0xf 0x8000d000 -1 0xd 0x8000f000 -1 2 0xa0b1165b 6 -1 -1 -1 0x1e 0 -1 0x1d 0 0 0x8000e200 0x8000e300 0x8000e200 4 0 0 0 0x3f 0x100000000` |
| `va-straddle` | f7ea04e9d24ea60d | 0x8000a800 | SR 7: `0 5 0xd 6 2 0 0` |
| `va-flat` | 08b1ba902363901d | 0x8000a800 | SR 14: `0 0 2 0xe 0x77 0x19 5 1 0 0 0 2 1 0` |
| `va-flat-off` | 407e1a8a13293e7f | 0x80009800 | SR 13: `0 0 0xe 0x77 0x19 1 1 0 0 0 0 0 0` |
| `va-alloc` | 4f346db72ed8337b | 0x8000a800 | SR 23: `0 0 2 0 0x40 0x80 0x19 5 0x8000000000000400 7 0 3 1 2 1 1 2 1 0 0 2 1 0` |
| `va-alloc-off` | cbef3379e9a8b4ad | 0x8000a800 | SR 22: `0 0 0 0x40 0x80 0x19 1 0x8000000000000400 6 0x200000005 0 4 0 0 4 0 0 0 0 0 0 0` |
| `va-alloc-corrupt` | 123ce8c012c7e7f9 | 0x8000a800 | SR 23: `0 0 0x1d 0 0 0 0 5 0x8000000000000400 6 0x100000004 1 1 2 1 4 0 1 0 0 0 0 0` |
| `va-alloc-exhaust` | 3ed79f75b2f92377 | 0x8000a800 | SR 23: `0 0 0x1e 0 0 0 0 5 0x8000000000000107 7 0 1 0 0 0 0 0 0 2 1 2 1 0` |
| `va-ns-collide` | 47605a54f9682c0b | 0x80009800 | SR 6: `0x77 0 5 0x19 0 0` |
| `va-ns-collide-off` | 90e3d70e14fd406a | 0x80009800 | SR 6: `0x77 0 1 0 0x77 0` |
| `vb-trans-vp2` | 0c325b29ba5cc640 | 0x80013000 | SR 36: `0 5 2 0x28 0x420590db 0x77 0x1234 0 5 2 0x5a5a 0 0 5 0xd 0 0 5 0x99 0 5 0xf 0 0 0 5 0xf 0 5 0 5 0x1c 0 0 5 0` |
| `vb-trans-plain-vp2` | 38508a4220203bf1 | 0x8000b000 | SR 5: `0 5 0x66 0 0` |
| `q17-order` (vp1's image) | ee2adf9b373e5028 | 0x80004000 | SR 3: `0x19 0x19 2` |
| `vb-trans-off` (vp1's image) | 79e670cc77352ee4 | 0x80011800 | SR 13: `0 1 0 0x28 0 0x77 0x1234 0 1 0 0 0x5a5a 0` |
