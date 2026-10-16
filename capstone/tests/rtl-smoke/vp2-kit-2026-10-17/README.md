# vp2's Tier 1 and Tier 2 kit: the pipeline and its gates, pre-registered (2026-10-17, before any image of it is final)

For the second virtual-profile bitstream (`docs/plans/2026-10-16-virtual-profile-on-silicon.md`, "The bitstream
sequence", bitstream 2). Its board readings are Tier 1's launcher smoke, with `contract.dom` reading
`VIRTUAL_APPLICATION_OK` and a stale access denied with 25 (exit 139), and the first virtual runtime suite, the
`run.py` gate. This folder pre-registers how that kit is built and gated, so the build does not wait when vp2
arrives. The images themselves are built from the compiler lane's ABI v6 revision when it lands. No image hash here
is final until then.

## The pipeline (`build-kit.sh <runtime rev> <label>`, the board lane's private script)

From ONE runtime revision, in a detached worktree whose `capstone/capstone-qemu` is the reference QEMU tree
(capstone-qemu `virtual-capstone` d35af47b163c). That tree supplies `cap_rev_table_abi.h` to the module build, so the
module and the emulator agree.
- **Compiler:** an existing virtual-capable LLVM build, `KIT_LLVM`, gated on `-capstone-image-gp` being present
  (5d077ae20a20). dev147, 612b3ec514c0, predates it and cannot build the virtual SDK. The clang hash is recorded.
- **musl:** `prepare-musl-capstone.sh` from a private copy of the 1.2.5 tarball, then the virtual SDK
  (`runtime/virtual/build-sdk.sh`).
- **The adapter (`capstone_vm.ko` + `capstone-vexec` + `capstone-job`), twice:**
  - for the QEMU guest's kernel (caplifive-buildroot linux-custom 6.1.0), the prediction's kit;
  - for the board's kernel (caplifive-system build-fpga linux-6.4.14), with buildroot's FPGA module flags
    `KCFLAGS="-march=rv64g -mabi=lp64d"`. A module with compressed instructions hangs the board in `insmod`.
- **`contract.dom`** (`runtime/virtual/contract.c`, -O1).
- **Tier 2's two application images:** SQLite's `sqlite3.dom` (`ports/sqlite/app`) and mruby 4.0.0-rc2
  (`ports/mruby/app`).

## The gates (each must pass before an image is staged on the board)

1. **The board module has no 16-bit instruction:** a count of 2-byte encodings in `objdump -d capstone_vm.ko` reads 0.
2. **The board module's `vermagic` names the board kernel's release, 6.4.14.** The QEMU module's names 6.1.0.
3. **The launcher is a static RISC-V executable.**
4. **The QEMU prediction:** `runtime/virtual/run.py` on the QEMU kit (the reference QEMU, the guest's own kernel and
   adapter) writes its record. Each test's expected exit status is `run.py`'s own table: normal 0, nested-stale 139,
   vm-ro-write 139, vm-none-read 139, preempt 143, sqlite_write/read 0, mruby 0 and the rest. That record is the
   board's prediction, test by test.
   - Under option (iii) the run is with exact bounds off, which `run.py` at ABI v6 is expected to select. At dev's
     4798dcab2d52 it still passes `x-capstone-exact-bounds=true`.

## The board run (written when vp2 is resident; recorded here so it is not improvised)

`run.py` drives QEMU only. On the board the same steps run under Linux through the stage driver:
- **Staging:** the module, launcher, `contract.dom`, `sqlite3.dom` and mruby go into the initramfs.
- **The arm:** a script that loads `capstone_vm.ko` and runs each `run.py` step through `capstone-vexec`. For each step
  it prints `VM_EXIT:<name>:<status>`, then `rmmod`.
- **The arm's last line:** `BASELINE-PROBE vm_steps = <n>`, the form the stage driver accepts for a host-binary arm
  (board-run skill, after the PHANTOM guard).
- **The comparison:** each step's status against the QEMU record.
- **Settled from the sources (2026-10-17): the board's `capstone.ko` (#3), which the drivers `insmod` before the
  first rung, can be loaded beside `capstone_vm.ko`.**
  - They register different misc devices: `capstone` (modcapstone `module/capstone.c:654-656`) and `capstone-vm`
    (`runtime/virtual/module/capstone_vm.c:1156-1157`).
  - `capstone_vm.ko` exports no symbol, so the two cannot collide at load.
  - The virtual CSRs (`0x5c0` profile, `0x5c1` the rev-node root, `0xcc0` the available-node count) are touched only
    by `capstone_vm.ko`. It saves and restores `0x5c1` around each use, with interrupts off.
  - Neither `capstone.c` nor the classic monitor (capstone-sbi 93440d5) names those CSRs. The positive control: the
    same `git grep` over the monitor finds its 26 `csrr`/`csrw` lines (sbi_capstone.S 18, sbi_capstone.c 7, sbi_capstone.h 1).
  - This is a reading of the sources. The first QEMU kit run loads both modules and shows it.

## The page-table walker's PMP entry: no monitor change

The plan asks the monitor to program an all-permitting PMP entry before the first virtual context. On the board that
entry already exists. OpenSBI's Domain0 (this platform's boot log, 2026-10-16) programs:
- Region05, the whole address space with S/U (R,W,X), at the lowest priority;
- the firmware, 0x80000000-0x8009ffff, M-only (Regions 03 and 04);
- the CLINT, closed to S/U (Regions 01 and 02).

The RTL lane confirms that the walker reads only the virtual context's page tables, which the kernel module keeps
in Linux RAM. The rev-node and tag paths do not pass the LSU's PMP. The simulation's entry 0, NAPOT over all memory
with RWX, is the bare-board shorthand. It must not be copied to the board, because it would expose the M-mode
firmware to S-mode.

## The capability PMP over Linux's own code: already granted by the classic monitor

The RTL lane's CSRUNV simulation found that S-mode code is confined by the capability PMP in capability mode. So the
module's path that issues CSRUNV needs a CPMP entry over itself; the directed arm grants CPMP0 over its image. On the
board that coverage already exists. Read 2026-10-17 from the classic line, opensbi df0e9a8 and capstone-sbi 93440d5:
- **At boot:** `sbi_capstone_init.S` enters capability mode with `CAPENTER(_cap_text_start, _cap_text_end)` and splits
  out the monitor's stack. `cap_env_init` (`sbi_capstone_dom.c:13-15`) then writes the three genesis regions into
  `cpmp(0..2)`:
  - `[0, _cap_text_start)`, which holds the UART and the CLINT;
  - `[_cap_text_end, dom_stack)`;
  - `[dom_stack_end, top)`, which holds Linux's RAM and so the module's code.
- **After boot:** regions carved later are loaded into a CPMP slot on demand. Any load, store or fetch access fault
  (causes 5, 7 and 1) enters `swap_cpmp` (`sbi_capstone.c:1975`), which installs the live region covering the
  faulting address.

So no monitor change is needed for coverage, as with the walker's PMP entry above. Every board boot already runs
Linux under it.

**One thing to confirm with the RTL lane before vp2:** `swap_cpmp` runs only on those three standard access-fault
causes. If a CPMP miss in capability mode raises a different cause, a miss on a carved region would not be swapped
in and would be fatal. The genesis entry covering Linux's RAM is resident from boot. Whether Linux's code stays
covered by a resident entry once the monitor carves regions out of it is not read here.

## Status of the pipeline proof

- On dev's runtime 4798dcab2d52 with the October 10 compiler (bd372e25f5fb), the SDK step fails: 101 of 1,361 musl
  files hit a backend assertion (`Res.getValueType() == Node->getValueType(0) || …`). The same SDK built clean with
  that compiler on 2026-10-10, so today's runtime needs a newer compiler.
- The proof reruns with the compiler lane's dev-current build when it lands. Its numbers go into this folder's RESULTS
  then.
