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

**The RTL lane's answer (2026-10-17):** a capability-PMP miss in capability mode raises exactly the three standard
causes.
- A fetch raises 1 (`core/pmp/src/pmp_data_if.sv:224,246`, through the frontend's FE_INSTR_ACCESS_FAULT).
- A load raises 5 and a store 7 (`pmp_data_if.sv:291-293, 310-311`).
- Observed in simulation too: the D1 arm's S-mode code faulted with cause 1 while CPMP0 was empty, and ran once
  CPMP0 covered its image.
- So `swap_cpmp`'s trigger set covers every miss. The CSRUNV context itself runs at M privilege, so no CPMP check
  applies to it; its protection is the page tables.

**The monitor's bookkeeping when it carves** (`create_region`'s split, `sbi_capstone.c:745-830`): the CPMP slot of the
region being split is rewritten with the piece that remains. The carved piece becomes a new, non-resident region,
which `swap_cpmp` loads on its first access fault. So Linux's code is always either under the resident slot or one
fault away from being swapped in.

## Pre-registered (2026-10-17, before the revision exists): the v6 module's `add()` change

> **WITHDRAWN (2026-10-17, the same morning, before any run against it).** The predicate below is refuted by the
> compiler lane's next measurement. It added the refused mapping's size to the arm, and `malloc(2029)`'s refusal
> was an **8192**-byte mapping, a power of two, refused with errno 22 at 4096 alignment. So `is_power_of_2` is not
> the operative condition by itself. The arm's NULL came from a different mapping than the 12288 it was designed
> around.
> - The live candidate is the natural-alignment disjunct of `add()`:466, `r->address & (r->bytes - 1)`. That
>   would make refusals depend on the address mmap returns, not only on size. It is not established, because
>   errno 22 is shared by every EINVAL in `add()`. The lane is recording the refused base next.
> - So the change below, and the prediction downstream of it, are withdrawn. A new prediction will be
>   registered against the revision that follows. The rerun plan (bundle 5; bundles 13 and 14 as controls;
>   run.py at `--timeout 1500`) stays.

**The change, from the compiler lane** (narrowed at lane 2691b60cbb69). It is not written yet: it changes what the
silicon enforces, so it waits for the project lead.
- `add()`: in the exact-bounds-off branch, `capstone_repr_ok(r->address, r->bytes)` replaces the test for a
  naturally aligned power of two. The page checks, CV_MAX_REGION_BYTES, the total-bytes limit and the overlap test
  are kept.
- `representable.h` is shared, and `build-adapter.sh` copies it from the tree being built.
- `remap_begin()` is left alone.

**Why `add()`:**
- The arena predicate today is `is_power_of_2`, measured in the lane's malloc-contract predicate arms (exact bounds
  off): `malloc(9341)` reads ALLOCATED with `last_ok_bytes=32768`, a power of two above 16384, admitted while the gate
  was armed. That refutes a size threshold by measurement.
- `malloc(2029)` reads NULL. Its 12288-byte group is the lane's model, not a measurement.

**The prediction, verbatim from the compiler lane, for the `add()`-only change:**
- the five allocation markers clear (aligned allocation, population array, remap-align setup, remap stale setup,
  precise allocation);
- your NULLs at 6144, 34816 and 35840 clear, and cases 00 and 10 read CAUGHT at cause 28;
- malloc(2029) becomes ALLOCATED where it is NULL today; malloc(9341) stays ALLOCATED;
- heap growth/zero and real_linux_mremap_growth_and_shrink STILL FAIL — not addressed by this change;
- bounds still fails, because the request rounding puts +200003 inside the object.

**What the board lane runs on that revision, and reads against the prediction.** Same kit script, QEMU
`40264f5de09ef153`, the same guest images:
1. **Tier 3 bundle 5** (`ffmpeg/carved-repros` `virtual-nested-pools`, the recorded cc_args), unchanged from the
   third pilot:
   - no NULL at 6144, 34816 or 35840;
   - cases 00 and 10 CAUGHT at cause 28 at their labelled probes;
   - 13 of 13 equal to the record.
2. **Bundles 13 and 14 again as controls:** CAUGHT 9/9 and 3/3, cell by cell as before. The change must not move a
   cell that was already right.
3. **run.py with `--timeout 1500`:**
   - `normal` still exits 1 on `VM_CONTRACT_FAIL:heap growth/zero` (the growth row above);
   - every other step is reported, not predicted.
   If growth clears, the model is wrong somewhere, and that is reported as such.

## Pre-registered again (2026-10-17, before the revision exists): the `add()` change, both halves

This section replaces the withdrawn one above. It is based on the compiler lane's lane commit fe1a12fe7c74.
- **The refusal has two operative halves.** The launcher, instrumented, prints `CAPSTONE_VM_ADD_REFUSED` with
  `add()`'s disjuncts at each CV_ADD refusal.
  - **The size half:** every printed refusal has `pow2=0`, at 28672 and 49152 bytes, including two that satisfy
    alignment. So non-power-of-two arenas are refused wherever they land.
  - **The alignment half:** the same predicate arm reads NULL on one run, with its refusal at 8192, and ALLOCATED on
    the next, with 8192 admitted. 8192 is a power of two, so only the natural-alignment disjunct remains. The
    launcher requests only max(granule, 4096) alignment.
  - The board lane's four boots of bundle 5 confirm this independently. malloc(6144), group 8192, was admitted in 1
    of 4 boots; calloc(1, 34816) and calloc(1, 35840) were refused in all 4.
- **The change:** `capstone_repr_ok(r->address, r->bytes)` in `add()`'s exact-bounds-off branch. It loosens the length
  to a granule multiple and the base to granule alignment. heap.c already asks for max(granule, 4096), so this also
  removes the nondeterminism. `remap_begin()` is left alone. It waits for the project lead, because it reaches
  silicon.

**The prediction, verbatim from the compiler lane:**
- the five allocation markers clear;
- non-power-of-two groups (34816, 35840, 200003, the population array, 6 MiB) clear DETERMINISTICALLY;
- malloc(6144), group 8192, clears on EVERY boot rather than 1 in 4 — run it at least 3 boots, because 1 of 4 would
  otherwise read as a pass;
- CAPSTONE_VM_ADD_REFUSED absent from the logs;
- heap growth/zero and real_linux_mremap_growth_and_shrink STILL FAIL, not addressed;
- bounds still fails, request rounding puts +200003 inside the object.
- **Added by the compiler lane before the revision exists** (lane ea37cbd801e6): its align-probe mode reads
  attempted=600 ok=187 failed=413 today. 420 refusal rows split pow2=0 aligned=1: 109, pow2=1 aligned=0: 87,
  both failing: 224, and pow2=1 aligned=1: 0. On the fixed revision it should read **failed=0 of 600**. Any
  surviving pow2=1 aligned=0 row means the alignment half still bites; any surviving pow2=0 row means the size
  half does.

- **Note on the growth row, added before the revision exists** (compiler lane ad72c3583446): the growth failure now
  has a known cause. On the exact-bounds-on arm, the only profile whose 6 MiB setup succeeds, the launcher
  refuses the 12 MiB growth with `CAPSTONE_VM_REMAP_REFUSED stage=granule bytes=12587008 align=16384`. realloc
  rounds `needed` to a page, not to the granule, and the lane's own representability check in remap_mapping
  refuses the remainder. That check is a lane-only change, not on dev. So the growth row still predicts FAIL for
  the `add()` change, and a surviving growth failure is that check, not a second site. Its own fix is domain-side,
  in realloc.c. `real_linux_mremap_growth_and_shrink` reaches neither a refusal nor a remap, and stays open.

- **Amended again before the revision exists** (compiler lane b61f56b10c06, the growth fix):
  - **The finding:** a granule-sized group header makes in-place growth across a granule boundary unsound. The
    header grows with the mapping while mremap keeps bytes in place, so the payload moves. It showed as
    corrupted contents with the growth reporting success. realloc now refuses such a remap and copies instead.
  - **So `CAPSTONE_VM_REMAP_OK` must read 0.** Both of the suite's growths cross a granule boundary. A non-zero
    count would mean the unsound path was taken.
  - `real_linux_mremap_growth_and_shrink` counts completed remaps (`run-malloc.py:153`), so it fails by design
    now. It is a check to re-scope, not a defect.
  - **The fix run's revision will carry b61f56b10c06 as well as the `add()` change.** So the growth rows above
    were stated for an earlier base. The prediction is to be re-stated, row by row, against that exact revision
    before the run.

**The board lane's runs on that revision.** The kit is rebuilt from the revision with the same toolchain, QEMU and
guest images.
0. **Amended before the revision exists (compiler lane 561b8b2cd692): the refusal line now prints unconditionally.**
   The runs below therefore use the unmodified harness, without `VIRTUALVM_TRACE`. The positive control (run 1)
   is repeated on a kit built from 561b8b2cd692 without the switch, and it must print the line on case 00. The
   paragraph that follows describes the earlier, traced arrangement. It is kept for the record.
   **Earlier: every run below sets `VIRTUALVM_TRACE=1`.** The launcher prints `CAPSTONE_VM_ADD_REFUSED` only under
   `CAPSTONE_VM_TRACE` (`exec.c:203`, `:763`), which the corpus harness never sets. A first control run without it
   read 0 lines while case 00 was refused, so it proved nothing. `virtualvm-trace.patch` (3 lines, the board
   lane's private copy of `tools/virtualvm.py`) exports it when `VIRTUALVM_TRACE` is set. The control and the
   checks run through the same patched harness.
1. **Positive control for the absence check, first:** cases 00 and 10 of bundle 5 on a kit built from fe1a12fe7c74,
   the instrumented pre-fix launcher. The serial must show at least one `CAPSTONE_VM_ADD_REFUSED` line, because case
   00 is refused on every boot. If it shows none, the absence below means nothing and the instrument is fixed first.
2. **Bundle 5 in full, once:** 13 of 13 equal to the record, with cases 00 and 10 CAUGHT at cause 28 at their
   labelled probes, and no `CAPSTONE_VM_ADD_REFUSED` line.
3. **Cases 00 and 10 in three more fresh boots** (`--only 00,10`): both CAUGHT in every boot, and no refusal line.
   One NULL in any boot means the alignment half still bites.
4. **Bundles 13 and 14 as must-not-move controls:** 9/9 and 3/3, cell by cell as before.
5. **run.py `--timeout 1500`:** `normal` still exits 1 on `heap growth/zero`. Every other step is reported, not
   predicted.

## Status of the pipeline proof

- On dev's runtime 4798dcab2d52 with the October 10 compiler (bd372e25f5fb), the SDK step fails: 101 of 1,361 musl
  files hit a backend assertion (`Res.getValueType() == Node->getValueType(0) || …`). The same SDK built clean with
  that compiler on 2026-10-10, so today's runtime needs a newer compiler.
- The proof reruns with the compiler lane's dev-current build when it lands. Its numbers go into this folder's RESULTS
  then.
