# vp2's bare arms for `caplifive_vp2_bf3db3175.bit` (staged 2026-10-17; the board vectors are pending)

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

**Not authorised to run.** A flash of bf3db3175 waits on the lead's own word naming it, after vp1's first session.

**Build** (`build.sh`):
- The tests are copies of capstone-ariane `virtual-profile-2-d` 1bcef8712, whose RTL is bf3db3175's (the three commits
  above it touch `verif/` only, 0 files elsewhere). They are checked equal at build time when `VP_SRC` is given; they
  were, from `git archive 1bcef8712`.
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

**Pending, before any vp2 board run:**
1. The RTL lane's vectors for the 23 images, and for vp1's vb-trans-off and q17-order on bf3db3175.
2. The table below filled with them, and pushed on the lane before the run.
3. Positive controls on the resident bitstream, where an arm's control is defined (the `-off` builds and
   va-ns-collide-off). The physical-pool controls should read the same on d93a8d66b.

| image | sha256 (16) | board_rec | board vector (from the simulation of this image) |
|---|---|---|---|
| `va-runv` | f764c436cdb52687 | 0x8000c000 | pending |
| `va-runv-quantum` | 34934ab8b2271c8c | 0x8000c000 | pending |
| `va-runv2` | 7b8ab66e23607d31 | 0x8000d000 | pending |
| `va-runv3` | d967dcdea54e3a86 | 0x8000c000 | pending |
| `va-runv4` | e8ac8002db62b1e8 | 0x8000d000 | pending |
| `va-runv5` | a5c00ebb2d79ed1b | 0x8000c000 | pending |
| `va-runv6` | 5861aa1557e59c21 | 0x80009000 | pending |
| `va-runv7` | 8edbfa78d1ec7acf | 0x8000c000 | pending |
| `va-barrier` | 28202d0d516170e8 | 0x80007000 | pending |
| `va-q15` | ee0959d19d8f0a82 | 0x80010000 | pending |
| `va-csmint-emu` | 3975c4af5495f488 | 0x8000e000 | pending |
| `va-csmint-mon` | cb5834d86bcb2f0c | 0x80011000 | pending |
| `va-straddle` | f7ea04e9d24ea60d | 0x8000a800 | pending |
| `va-flat` | 08b1ba902363901d | 0x8000a800 | pending |
| `va-flat-off` | 407e1a8a13293e7f | 0x80009800 | pending |
| `va-alloc` | 4f346db72ed8337b | 0x8000a800 | pending |
| `va-alloc-off` | cbef3379e9a8b4ad | 0x8000a800 | pending |
| `va-alloc-corrupt` | 123ce8c012c7e7f9 | 0x8000a800 | pending |
| `va-alloc-exhaust` | 3ed79f75b2f92377 | 0x8000a800 | pending |
| `va-ns-collide` | 47605a54f9682c0b | 0x80009800 | pending |
| `va-ns-collide-off` | 90e3d70e14fd406a | 0x80009800 | pending |
| `vb-trans-vp2` | 0c325b29ba5cc640 | 0x80013000 | pending |
| `vb-trans-plain-vp2` | 38508a4220203bf1 | 0x8000b000 | pending |
