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

## Monitor item for vp2, decided by the runtime and RTL lanes: CSMINT emulated in the monitor

**This one is a monitor change**, unlike the walker PMP and the CPMP coverage above, which needed none. It is not for
the current bitstream.

**What the monitor does when an S-mode CSMINT traps to it.** QEMU's `helper_csmint`, `capstone-qemu
target/riscv/capstone_table.c:218-283` at d35af47b163c, is the reference for every step and every refusal:
1. Decode rd, rs1 (the slot) and rs2 (the descriptor) from the trapping instruction:
   `csmint 1010000 rs2 rs1 001 rd 1011011` (`insn32.decode:982`).
2. Read the descriptor `{u64 base; u64 end; u64 perms}` through S's translation (MPRV), as three aligned
   doublewords.
   - desc & 7 raises load-misaligned.
   - A slot that is not 16-byte aligned raises store-misaligned.
   - A slot with no writable translation raises a store access fault.
3. Refuse before any write, where QEMU does:
   - base >= end or perms > ALL: 29;
   - a descriptor that is not representable (option (iii); the round trip through the encoder must give back base,
     end and cursor): 29;
   - fewer than two free identities: 30.
4. Mint the LINEAR capability [base, end), with cursor base and the descriptor's perms, as the monitor mints for
   domains today.
5. Run the new table op CSNODE under the S table namespace. It creates the child node and its MREV ancestor, as
   `cap_rev_create` plus `cap_rev_mrev` do.
6. Store the capability, with its tag, to the slot through S's translation.
7. Set rd to the ancestor's id and advance mepc.

**The RTL lane's answers (2026-10-17; the Q-entry id follows when the runtime lane files it):**
- **CSNODE rd, rs1:** opcode 0x5b, funct3 1, funct7 0x54 (0x50 stays CSMINT, which traps; 0x53 is CSRETIRE),
  R-type with rs2 = x0.
  - rs1 is a capability register, the LINEAR capability just minted.
  - With fewer than two identities left it raises 30 before any write.
  - Otherwise it allocates the child (depth 0, valid, linear), then an MREV ancestor, and re-parents the child under
    the ancestor at depth 1.
  - It re-nodes rs1 onto the child and returns the ANCESTOR id in rd.
  - It is legal from S outside a context, and from M when `csupctl` bit 3 (the "table" bit) is set. The emulation
    sets that bit around it. It is illegal from U and inside a context.
- **The trap:** an S-mode CSMINT raises illegal instruction (2) to M, with mtval = the instruction bits. So the
  monitor decodes from mtval and needs no read at mepc.
- **The store: not STC under MPRV.** An STC's address is a physically bounded capability, and only the DYN unit
  checks it, which knows nothing of MPRV.
  - The descriptor's three words are read as scalar loads under MPRV. CPMP at S must cover that page, and Linux's
    RAM entry does.
  - The slot's VA is translated in software, by reading the page tables as the walker does.
  - The minted capability is then stored with STC through a physical capability over the slot's page, as the
    monitor stores into a domain's context today. No CPMP condition applies to that M-mode store.
  - Slot alignment (cause 6), descriptor alignment (cause 4) and the unrepresentable descriptor (29) are the
    emulation's checks, made before CSNODE runs.

**The three settled by the RTL lane (2026-10-17):**
- **The authority to mint** is the monitor's existing capability-mode mint primitive. CAPCREATE, CAPTYPE, CAPBOUND
  and CAPPERM (opcode 0x7b) are legal from M and illegal only under supervision.
  - Checked at d93a8d66b: `core/decoder.sv`, the funct3 000 block at about line 1137 sets `illegal_instr` only when
    `sup_active_i`.
  - CAPBOUND(base, end) over the descriptor's VA range gives the VA-bounded LINEAR capability. CAPCREATE gives it
    node 2, which CSNODE re-nodes onto the child.
  - No genesis capability is needed. The monitor must never dereference that capability itself, because its cursor
    is a VA.
- **The slot walk** faults rather than set A or D, as the RTL's walker does (the RTL lane cites `cva6_ptw.sv:453,
  468, 478` and `cva6_mmu.sv:142` on the virtual-profile branch).
  - It requires V, W and R, U with SUM, A and D.
  - If any is missing it raises store page fault 15 to S with tval = the slot VA, as QEMU's `probe_access_full`
    does. It never sets A or D on the module's behalf.
- **The test path is the RTL simulation's D4 arm**, which runs the whole monitor sequence as its M-mode trap handler.
  - The sequence: the CSMINT trap, the MPRV descriptor read, the mint, CSNODE under `csupctl.table`, the physical
    walk and the physical STC.
  - So the arm is the executable specification. The monitor code is written against it, and reads the rd and the
    tagged slot that QEMU's helper gives.
  - QEMU is unchanged, and the board's prediction for CSMINT comes from QEMU's own CSMINT.
  - D4 follows the RTL lane's E and D5 arms. The monitor code waits for it.

**D4 is in the RTL** (capstone-ariane virtual-profile-2-d 41fa120bf, 2026-10-17).
- CSNODE (0x54) and CSRETIRE (0x53) are admitted to trusted S and to M under `csupctl.table`, outside a context. Both
  are illegal from U, and escape with cause 2 inside a context.
- CSNODE allocates QEMU's pair, child first and then the ancestor. It re-nodes rs1 onto the child and returns the
  ancestor.
- CSRETIRE returns 0 on success and 1 on rejection, and never faults.
- Arms: va-runv6 (CSNODE and CSRETIRE from M and trusted S) and va-runv7 (both illegal inside a context).
- The arms exercise CSNODE, not the whole emulation path from the trapped S-mode CSMINT through the MPRV read, the
  walk and the physical STC. That handler is the board lane's to write, and is tested as its own arm.

**Two vp2 facts that move readings, from the RTL lane:**
- **Item E refuses RETURN inside a virtual context**, as QEMU's virtual-C filter does. A `csupctl.virt` domain that
  ends with RETURN escapes with cause 2 (csupstatus 5) instead of returning. CSRUNV contexts are unaffected.
- **Every allocation inside a context keeps a reserve of 256 identities** (CDERIVE, SPLIT, MREV, INIT), as QEMU's
  `capstone_require_node`. A context's table needs more than 256 free identities before its first derive, or it
  reads node pressure (kind 5, cause 30). Trusted S and M outside a context allocate down to the last identity.
- Both are QEMU's behaviour, so the Tier 1 and Tier 2 predictions already carry them. Only a reading taken from a
  vp1-era image would move.

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

**THE PREDICTION, re-stated by the compiler lane against b61f56b10c06 plus the `add()` relaxation, verbatim. It
supersedes every earlier row in this section.** The lane marks each row computed or inferred. The constraint behind
the REMAP_OK row is recorded at the compiler lane's 67e4368f6331, `capstone/docs/plans/2026-10-17-vp2-launcher-items.md`.

run-malloc, exact bounds OFF:
- the five allocation markers (aligned allocation, population array, remap-align setup, remap stale setup, precise allocation): CLEAR
- align-probe: failed=0 of 600, where it is 413 of 600 today
- CAPSTONE_VM_ADD_REFUSED: absent, and now genuinely falsifiable since the print is unconditional
- malloc(6144), group 8192: ALLOCATED on EVERY boot, not 1 in 4 — please run at least 3 boots, since 1 of 4 would otherwise read as a pass
- CAPSTONE_VM_REMAP_OK: 0
- remap_align_contents_survive, remap_align_reported, basic_inplace_mremap_failure: True
- real_linux_mremap_growth_and_shrink: STILL FALSE — a check to re-scope, not a defect
- bounds / MALLOC_FAIL:denial survived: STILL FALSE — request rounding puts +200003 inside the object
- the suite overall: the OFF arm should converge on what the ON arm already reads at b61f56b10c06, namely 21 true / 2 false, with the same two checks failing for the same two reasons. If OFF lands anywhere short of 21, something other than add() is still biting.

run.py, --timeout 1500:
- heap growth/zero, on normal and both parallel steps: CLEARS. This row is GROUNDED rather than inferred: contract.c:312 is require(p[i] && !p[i][131071], "heap growth/zero"), so it allocates at least 131072 bytes, which is above MMAP_THRESHOLD and takes the large path. Its mapping computes to 135168 bytes, which is NOT a power of two, so add() refuses it on every boot today. Despite its name it is an allocation refusal, not a growth failure.
- heap-threads child malloc: expected to clear, INFERRED not grounded — I have not computed its allocation sizes, so if it survives treat it as a second site.
- sqlite_write exit 7 NOMEM: expected to clear, also inferred. Your watch on it stands, and it now has a third candidate: a growth that crossed a granule boundary and was refused before b61f56b10c06 would also have shown as NOMEM. If it clears at b61f56b10c06 WITHOUT the add() change, that was it.
- sqlite_read and mruby exit 1: expected to clear if they are downstream of an allocation refusal; otherwise second sites. Inferred.

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

## Pre-registered (2026-10-17): run.py on b61f56b10c06 WITHOUT the `add()` change, to separate sqlite_write's NOMEM

The question: does sqlite_write's exit 7 (NOMEM) come from a growth refused before the growth fix, or from an
`add()` refusal? This run has the growth fix and not the `add()` relaxation, with exact bounds off and
`--timeout 1500`.
- **Computed:** `normal`, `parallel_one` and `parallel_two` still exit 1 on `heap growth/zero`. Their 135168-byte
  large-path mapping is not a power of two, and `add()` refuses it on every boot.
- **The reading:**
  - sqlite_write exits 0 here: its NOMEM was the refused growth, which b61f56b10c06 fixes;
  - it still exits 7: it is an `add()` refusal, or a further site.
- Every other step is reported, not predicted. The pre-fix run on fe1a12fe7c74, above in RESULTS.md, is the
  comparison.

## Status of the pipeline proof

- On dev's runtime 4798dcab2d52 with the October 10 compiler (bd372e25f5fb), the SDK step fails: 101 of 1,361 musl
  files hit a backend assertion (`Res.getValueType() == Node->getValueType(0) || …`). The same SDK built clean with
  that compiler on 2026-10-10, so today's runtime needs a newer compiler.
- The proof reruns with the compiler lane's dev-current build when it lands. Its numbers go into this folder's RESULTS
  then.

## Pre-registered (2026-10-17, before the boot): Tier 1/2 on vp2 with the flat-table module

One boot of a private image on `caplifive_vp2_bf3db3175.bit`, resident since its first session (987bf254ee40,
83be6830551d); no flash. The files are in `board-tier12/`: the arm's scripts, the judge, the boot script and the
stage's `SHA256SUMS`.

**The kit (c14933904de4, the compiler lane's flat kit).** Its run.py record on QEMU is the prediction: flat arm PASS
41/41, 22 `VM_EXIT` lines; the paged arm also read 41/41, as a null control. The board runs the bytes that record ran:
- `contract.dom` `bf433e107032fa8a`, `sqlite3.dom` `a6ee743522518c02`, mruby `5fee8d8f604229ea`, taken from run.py's
  own stage.

**The board adapter** is built by build-kit.sh's adapter block with `KIT_FLAT=1` (e8b55b5dfdfc), from c14933904de4's
runtime. All 32 sources in the record's `source_sha256` are identical in that tree.
- `capstone_vm.ko` `0d62a1bcdeecc13f`. It has vermagic 6.4.14, no 16-bit instructions (the same counter finds 1,612
  in the QEMU module), and no `csrr` of scapctl or urevavail. The compiler lane's own board build is byte-identical.
- `capstone-vexec` `b27afefc09c6cae6` and `capstone-job` `c501da210412e23e`, static RISC-V executables.
- QEMU module hashes are not comparable across builds: that kernel embeds the module's build path
  (DEBUG_BUGVERBOSE). The kit's `bb6d1aec…` and a build-kit.sh build at another path have identical code and differ
  only in the path string and the build-id. The board module carries no path, so its hash is a code identity.

**The firmware.**
- `fw_payload.bin` `2882836388198297`, 19,565,064 B; Image `ce8a2ba31fbe5ce3`.
- BL's recipe (wrapper 1269c77, monitor cb91d7d) plus `CAPSTONE_VP2_CSMINT`; `csmint_dispatch` is in the ELF.
- The bake checked by hash that the private initramfs carries every staged file, and that the restored shared image
  carries none.
- It also packs about 2.5 MB of the shared tree's standing files (speedtest1*, k800, sqlite_host*). None of them
  runs.
- Of the standalone preflight's checks, the DTS-against-bitstream, single-FDT and 20 MB budget checks pass. Its
  overlay checks look for each rung as a `.dom` in the shared overlay. They do not apply here: the vm rungs are
  scripts in a private image, and the bake's hash check stands in for them.

**The rungs, in order:** b0-stats and b0-hello (BL's controls; both read 0 on vp2 in tonight's set 7), then
vm-contract (Tier 1), then vm-suite (Tier 2, run.py's steps; the longest, last).

**Predictions.**
- **b0-stats, b0-hello:** retval 0.
- **vm-contract:** retval 0. contract.dom's `ok` prints `VIRTUAL_APPLICATION_OK` and exits 0. Its `stale` access is
  denied with cause 25, exit 139, which is the reading that the run is virtual.
  - The layout branch, written before the boot at the runtime lane's request: a 29 at contract.dom's FIRST allocation,
    before any `VIRTUAL_APPLICATION_OK`, means "header or layout, kit side". It goes to the RTL lane with the table
    dump and is not a silicon finding. This boot is the first time silicon reads a table the module wrote, and the RTL
    lane's cross-check of a module-written table has not run yet.
- **vm-suite:** 22 `VM_EXIT` lines, each with the flat record's status:
  - 0: normal, threads, vm, nested, heap-threads, sparse-churn, parallel_one, parallel_two, sqlite_write, sqlite_read,
    mruby;
  - 139: nested-stale, stale, bounds, retired, vm-ro-write, vm-none-read, vm-guard, vm-padding, sparse-stale;
  - 143: preempt;
  - except `cleanup`: the board kernel cannot unload a module, so `rmmod` fails and cleanup is pre-registered false.
  - `board-check.py`, which runs run.py's own analysis block from c14933904de4, must read 40 of 41 tests true, with
    cleanup the only false one.
- **A reading that differs from the record** is reported with both values, and goes to the RTL lane before anything is
  called silicon.

**The judge, checked before the boot** (`board-tier12/results/dryrun-tier12.txt`).
- On the flat QEMU serial it reads 41/41.
- One altered status flips exactly that test.
- `cleanup` is flagged when the board expectation is applied to the QEMU serial.
- A board-format dry run found a bug, now fixed. The driver's echoed command line carries both rung markers literally,
  and the unanchored judge matched that echo and read "NO DATA" on a clean transcript. The markers are now anchored at
  line start, with a numeric rc on the END marker. The anchored judge reads the same transcript correctly, and refuses
  a duplicated segment.
- The boot script was dry-run with only the driver stubbed. A clean transcript, the layout branch, and no transcript
  each produce the predicted lines.

## Pre-registered again (2026-10-18, before boot 2): Tier 1/2 on vp2 with CSMINT v7, the v7 module and the monitor's S-trap injection

Boot 1 (results/board-vp2-tier12-1.result-lines.txt) wedged in vm-contract. The module's `ldc` on an integer base was
refused by the RTL with cause 24 (Q-23). The monitor then wedged, because it had no path to hand an S-origin trap to
the kernel (Q-24). Both are addressed, and nothing else changes: same bitstream, rungs, judge and stage layout.

**What changed since boot 1:**
- **The module:** the v7 board adapter at runtime 179686bcf198, the compiler lane's kit. It is built by build-kit.sh
  KIT_FLAT=1 inside the kit, and a second build in another directory gives the same hash.
  - Since 2f15ed14b4f6 only two files changed: capstone_vm.c gained the node_flat_records parameter (default and
    maximum 65,536), and a tests README.
  - `capstone_vm.ko` ed24f827b5beb373. My reading of it: vermagic 6.4.14, no 16-bit instructions, no ldc or stc, and
    no csrr of 0x5c0 or 0xcc0. The controls fire: boot 1's 0d62a1bcdeecc13f reads 2 ldc and 2 stc, and the paged QEMU
    module 69a7016b reads one of each csrr.
  - Its custom instructions are all register-form: run (0x24) x4, CSMINT (0x50) x2, csretire (0x53) x2.
  - `capstone-vexec` b27afefc09c6cae6 and `capstone-job` c501da210412e23e are boot 1's bytes.
  - The kit loads the parameter unset. The geometry comes from the stats line (a readback of the live table header),
    not from the invocation: node_bytes 1048576, node_capacity 65536.
- **The prediction:** run.py's record at 179686bcf198 on QEMU virtual-capstone-v7 f2f15f8fc70c, the tightened QEMU:
  trusted S makes no capability memory access.
  - The record is the kit's `out/result-flat-179686bc.json` (serial `tmp/virtual-app.a9a4el5a/serial.log`; the compiler
    lane's pre-registration 189b170683ca): the flat arm reads PASS 41 / 41 with 22 VM_EXIT lines. The paged arm is the
    null control.
  - It is re-run rather than carried over from 2f15ed14b4f6: the parameter sits outside the flat #ifdef, so both
    arms' modules changed.
  - It runs the board's bytes: contract.dom bf433e107032fa8a and mruby 5fee8d8f604229ea (both boot 1's), and
    sqlite3.dom a4b2c0f05809f433. sqlite3.dom does not reproduce across builds, so it is staged by its sha256.
  - From boot 1's stage two files change: capstone_vm.ko and sqlite3.dom.
- **The monitor:** capstone-sbi vp2/tier1-monitor 1e9451a, which merges two lines:
  - the S-trap injection with its review fixes (vp2/os-trap-inject f8bfde5: NODM on the FPGA host only, the INJL
    loop guard) and ccf9afc, which makes the domain-call counter and swap_cpmp's return value FPGA-only. The QEMU monitor
    is then byte-identical to capstone-bootstrap c3293b3, and the FPGA output is byte-identical to f8bfde5's;
  - the v7 routine, gated to CSMINT from S (vp2/csmint-v7 67ed67f).
  - Its base is capstone-bootstrap c3293b3, whose tree equals boot 1's monitor cb91d7d (`git diff cb91d7d c3293b3`
    is empty). From boot 1 the monitor changes in three files, +239/−20: sbi_capstone.c, sbi_capstone.S and
    capstone_csmint.S. The wrapper (1269c77) and the defines are boot 1's.
  - Pinned: sbi_capstone.c c8172e5abc09687e…, sbi_capstone.S 7b23b6a67540fd45…, capstone_csmint.S 4600d04680158edf…,
    sbi_capstone.h 2c6f6bcab4ce180e…. Each was checked in the firmware build's extracted source.
  - The injection is validated on the RTL in two runs. The first (osinject/results/rtl-bf3db3175.result-lines.txt)
    read 14 of 14 as pre-registered, with the control failing as predicted. Revision 2 adds the U-CSMINT row on
    fba7812's files: 23 of 23 as predicted, the control as predicted
    (osinject/results/rtl-bf3db3175-rev2.result-lines.txt, lane f077f70d1e57).
  - The v7 routine is validated by va-csmint-mon (81/81 on v7) and by va-csmint-entry v7 on fba7812:
    mon's 81 values at the entry arm's layout (csmint-emu/results/rtl-bf3db3175-v7-entry.result-lines.txt, lane
    f077f70d1e57).
- **The firmware:** BL's recipe + CAPSTONE_VP2_CSMINT on that monitor (FW_MON=/tmp/capstone/csbi-t1mon).
  - The pre-bake check build is fw_payload ae517ef354830e70, gate 25712/32768. It is the same hash from fba7812 and
    from 1e9451a, so the RTL arms run on fba7812's files hold for 1e9451a.
  - The baked payload, with the v7 module in its initramfs, is fw_payload 41fb669c9f77462d (19,565,064 B; Image
    2e7c65b3df7d76d7), gate PASS (25712 of 32768 B). csmint_dispatch and _inject_s_trap are in its ELF.
  - The bake checked by hash that the private initramfs carries all 14 staged files, and that the restored
    shared image carries none. The shared buildroot is d04bd83b13cd, its HEAD since 2026-10-16 04:22 by the
    reflog, so boot 1's too.
  - The standalone preflight reads exactly as it did for boot 1. The DTS-against-bitstream, single-FDT and
    20 MB checks pass; its overlay checks do not apply to a private image.
- **The drivers:** they classify the injection tags (lane 0b8813639710). An EXCX or ILLX followed by INJS is a trap
  handed to the kernel, not a monitor spin, and NODM and INJL are monitor stops.

**Predictions** (rungs as boot 1: b0-stats, b0-hello, vm-contract, vm-suite):
- b0-stats, b0-hello: 0, as on boot 1 (results/board-vp2-tier12-1.result-lines.txt).
- vm-contract, the identity branch: `ok` prints VIRTUAL_APPLICATION_OK and exits 0; `stale` is denied with cause 25,
  exit 139. This is the reading that the run is virtual. Two other branches are written down:
  - 29 at contract.dom's first allocation, before VIRTUAL_APPLICATION_OK: header or layout, kit side. It goes to the
    RTL lane with the table dump.
  - 24 from S: a remaining integer-based access, which the module's 0-ldc/stc gate says cannot exist. Its address
    names the instruction.
  - With the injection, a 24 or any other unhandled trap from S no longer wedges. The monitor prints INJS after its
    EXCX report, and the kernel takes the trap: an oops, or a signal to the process. If the kernel survives,
    vm-contract.sh returns a 92x retval (921-925, its header) and the boot reaches vm-suite, whose reading after such
    a failure is reported, not judged. If the oops hangs the kernel, the oops text is the reading. Either way the
    driver classifies the rung as injected-trap, not as a monitor spin.
  - NODM or INJL: the monitor's own stop. NODM means a domain fault return with no domain call outstanding; INJL
    means S's stvec is the faulting pc. The core is lost and later rungs are collateral.
- vm-suite: 22 steps with the record's statuses, except cleanup (no module unload on the board kernel).
  board-check.py with run.py's analysis block (run.py d03ea74cc714f17a, unchanged since boot 1) must read
  40 of 41 true, with cleanup the only false one.
- A reading that differs from the record is reported with both values, and goes to the RTL lane before anything
  is called silicon.

**The injection's M-path half** (a classic domain's fault still returns its code) is not a board rung. On this
silicon a classic domain fault wedges today (s1-sqlite-silicon-2026-10-16: sbp1 and sbp2 at cause 25, sbp6 at 28),
so no board run can show the return. The patch's only M-path change is the NODM stop when no domain call is
outstanding. The compiled monitor brackets 8 of 8 domain calls. In capstone-c's FPGA output at 1e9451a (BL's
defines + CAPSTONE_VP2_CSMINT), the counter at gp+208 has 34 references: an increment and a decrement around each call
in call_domain, call_domain_with_cap, schedule_domain, share_child_region, share_region (two), shared_region_annotated
and supervised_invoke, the test in fault_return_from_domain, and the carve in dom_init. That and the RTL lane's review
are its evidence.

**The scripts, checked before the boot** (`board-tier12/results/dryrun-tier12-boot2.txt`). The boot runs
`board-tier12/run-tier1-2.sh 41fb669c9f77462d` on the stage in `board-tier12/stage.SHA256SUMS`, which from here on is
boot 2's. Boot 1's manifest, the one its pre-registration above names, is kept as `board-tier12/stage-boot1.SHA256SUMS`. vm-contract's branch
is read by `board-tier12/vm-contract-branch.py`, and vm-suite by board-check.py as on boot 1.
- On the record's own serial it reads 41 of 41. With the board's expectation applied, it flags only cleanup, which is
  true on QEMU, as it must be. One altered status (vm-none-read) flips exactly that test.
- Wrapped in the driver's framing, with cleanup failing as the board kernel makes it fail, the serial reads 40 of 41,
  with cleanup the only false test.
- run-tier1-2.sh ran with every pin real (firmware, stage, run.py, llvm-nm, driver) and only the driver stubbed. A clean
  transcript, an injected trap and a NODM stop each produce their predicted lines.
- The gates refuse before the driver starts:
  - an unfilled placeholder;
  - a wrong firmware, driver or run.py hash;
  - an existing output directory;
  - an llvm dir without llvm-nm. The first version checked for llvm-objdump, and the dry run caught it.

### Boot 2, result (2026-10-18 00:38-00:50): the S-trap injection works on silicon; 18 virtual runs stop at the image-gp `delin` (C-27), and parallel_one's launch traps in CSRUNV

Result lines: `board-tier12/results/board-vp2-tier12-2.result-lines.txt`.
- **b0-stats, b0-hello:** 0, as predicted.
- **vm-contract:** retval 922, NOT as predicted, and on no pre-registered branch.
  - Both `ok` and `stale` exit 139 with `domain fault cause=26`, at pc = slot 0 base + 0x1f60 and address = slot 0
    base + 0x21ff0.
  - There is no VIRTUAL_APPLICATION_OK, and no monitor tag.
  - vexec's five `CAPSTONE_VM_ADD_OK` lines (its `maps[]` index 0-4) print `linear=0`, gp's (map 0, reg=3) included.
    That is vexec echoing its own request (exec.c:220-221), not a readback of the minted type. The QEMU record's serial
    prints the same five lines.
- **vm-suite** ran after the first failure, so it is reported, not judged.
  - 16 steps ran, normal through preempt. All exit 139 with the same fault: cause 26, pc offset 0x1f60, address
    offset 0x21ff0.
  - In the next step (parallel_one) the monitor printed `EXCX MCAU 1D MEPC 011137F4 MTVL 0 MSTA 820`, then `INJS 1D`.
    The kernel printed `Oops - unknown exception` with `epc vm_ioctl+0xb04/0x1310 [capstone_vm]`, cause 0x1d and
    status SPP=1, then panicked ("Fatal exception in interrupt").
  - The oops's Code bytes `(17db) 48b7` are word 0x48b717db, funct7 0x24: the module's `run a5, a4, a1`, with
    a4 = 0x8159b000 and a1 = 0.
  - Per the runtime lane, the 29 is the refusal of a second concurrent launch. vp2 has one context slot and QEMU has
    32, so run.py's parallel steps pass only on QEMU. The module is to serialize contexts instead of being trapped.
  - Whether the RTL should trap to M or return 29 in rd is with the RTL lane.

**What this boot settles.** The monitor's S-trap injection is validated on silicon. An S-origin capability trap was
reported (EXCX), handed to the kernel (INJS), and delivered with the right epc, cause and SPP. The kernel then printed
its own oops, where boot 1's monitor wedged. A virtual application's fault is delivered to the module as a domain
fault and becomes SIGSEGV (139), 18 times in this boot.

**The fault, located** (claim-auditor: supported, with the wording below; a localization from source and exclusion,
N = 1 boot, 18 of 18 faults identical).
- **The mapping is observed, not assumed.** vexec loads the image at `p_vaddr - low` into map 0 (exec.c:413-441), and
  contract.dom's single LOAD is at VA 0x10000. On QEMU every entry reads map 0 base + 0x974, and domain_main is
  0x10974. So pc offset 0x1f60 is VA 0x11f60: `delin a0` in `__capstone_cap_init`, right after `scc a0, gp, a0`.
- **The non-circular fingerprint is tval − pc = 0x20090.** It follows from the instruction bytes alone (`auipc a0,
  0x20` at pc − 0x14, then `addi 0xa4`), and it holds on all 18 faults. The function's next delin would read 0x24574.
  For DELIN the decoder takes the operand from the rd field (decoder.sv:1235). tval is the DYN op's operand_a
  (cva6.sv:1522, ex_stage.sv:1184), and the commit stage latches pc, cause and tval from the same instruction.
- **a0 is NONLIN.**
  - The module requests NONLIN for gp and refuses LINEAR (capstone_vm.c:580, :628).
  - On silicon the type is set by the monitor's CSMINT emulation (capstone_csmint.S:258-264 in fw 41fb669c9f77); on
    QEMU by helper_csmint.
  - SCC keeps rs1's type (capstone_flu_unit.anvil:122).
  - A LINEAR gp is excluded by behaviour. The earlier `scc t2, gp, t0` at 0x10030 would have nulled it, and DELIN of
    an untagged operand raises 24, not 26.
- **The RTL's DELIN raises UNEXPECTED_CAP_TYPE (26) for any tagged non-LINEAR capability**, and 24 for an untagged one
  (capstone_dyn_unit.anvil:797-801 at bf3db3175, with no virtual-context branch).
- **QEMU's helper_csdelin returns early for NONLIN** (op_helper.c:1569 at f2f15f8fc70c). So the delin is a no-op on
  QEMU and a fault on the RTL. The QEMU record read 41/41 on the same bytes.
  - That QEMU took the early return is deduced, not observed: a LINEAR gp would also fault there on QEMU.
  - The observation is the compiler lane's control: the old images on the strict-DELIN QEMU must reproduce the board
    line exactly, `cause=26` at base + 0x1f60, address base + 0x21ff0, `entry=unknown`.
- **Not explained by this localization:** the CSRUNV trap in parallel_one (vm-suite, above), a second signature.
- This is **C-27** ("`delin` of a NONLIN gp is a spec/RTL trap that QEMU hides"). Earlier instances of the class:
  history 20-08-2026 S-02/S-05 (lines 146-156) and 27-07-2026 RESULTS-delin-wedges…. Silicon wedged on it then (R-5);
  vp2 traps cleanly with 26. The mechanism above is cited from source; the docs are cited only as prior instances.
  - ISSUES-ARCHIVE.md:3328 records C-27 as `FIXED 2026-09-05 — the DELIN is dropped under the NONLIN-gp contract`. That
    status is false. CapstoneExpandPseudoInsts.cpp:381 still emits the DELIN for every `PseudoCapGlobalBase`, and
    CapstoneISASemantics.md:52 still lists it under the default ABI ("never under `-capstone-gp-captable`").
  - The false FIXED is why the boot was spent. Its retraction is the compiler lane's, as its own commit.
- It is not one site. Counting each `delin` whose rd was set from gp (cincoffset, cincoffsetimm or scc) within the 5
  preceding instructions gives contract.dom 525 of 530, sqlite3.dom 5,983 of 5,988 and mruby.dom 9,859 of 9,863.
  Sampled sites are bare `scc rd, gp, x; delin rd`.

**Ruled by the runtime lane** (the semantics owner), as Q-23 was: the RTL is the reference.
- QEMU's default DELIN raises 26 on a non-LINEAR rs1. The RTL lane implements it on virtual-capstone-v7 and
  re-measures Tier 0.
- There is no strict-DELIN property for prediction runs: the default is strict, and a test-only property exists only
  where a suite's bootstrap needs the old behaviour, as with x-capstone-test-s-ldc.
- The fix is (b): the backend's image-gp materialization drops the delin. The image gp is NONLIN by ABI, and a
  LINEAR gp would make every global access write the shared node's linear bit on the RTL. The backend change is
  reserved to the project lead (CapstoneISASemantics.md:52, "the DELIN decision is the project lead's").
- The gate on the rebuilt images is `delin-census.py` (this folder): 0 gp-derived delins. The images above are its
  positive control, at 525 / 5,983 / 9,859.
- Every QEMU record before the new pin is blind to this class, so the Tier 3 predictions wait for it.
- Boot 3 is this pre-registration with the images' hashes moved, plus a delin branch: 26 at +0x1f60 is the unfixed
  glue, kit side (C-27).
- The board stays idle until all of these are in:
  - the lead's word on the backend change;
  - the strict-DELIN, one-slot QEMU pin;
  - the rebuilt SDK and images, with the census at 0;
  - a from-scratch run.py record, in which the old images fail at the delin on the new QEMU as the control;
  - the module's context serialization;
  - the RTL lane's answer on CSRUNV's 29.

## Pre-registered (2026-10-18, before boot 3-T1): Tier 1 on the C-27-fixed images

**The runtime lane's ruling.** Boot 3 is held for Tier 2: vp2 has one saved-context slot (Q-26), and the module does
not serialize yet. Tier 1 runs now, because vm-contract runs `ok` and then `stale`, each a single context. This boot
is boot 2 with the three images rebuilt by the C-27 fix and with no vm-suite.

**The images,** from the compiler lane's v8 kit: compiler built at 01b62884c0c7, which contains C-27's fix 299465232cc3 (the
DELIN dropped under the image-gp ABI).
- contract.dom f2017064bea77dd2, sqlite3.dom 233640f7193b034c, mruby bba891bc16576d98.
- `delin-census.py` reads 0 gp-derived delins on all three. The totals fell by exactly the old counts: 530 → 5,
  5,988 → 5, 9,863 → 4. The instruction counts fell by the same amounts (33,870 → 33,345, and so on), so the
  instructions were removed, not moved out of the window.
- The residual 5/5/4 are the allocator glue's `ldc rd; delin rd`. Their operand's type is only known at run time,
  and the strict-DELIN pin would name any that is not LINEAR.

**Unchanged from boot 2,** checked by hash:
- module ed24f827b5beb373, capstone-vexec b27afefc09c6cae6, capstone-job c501da210412e23e;
- the monitor 1e9451a and the firmware recipe;
- the bitstream.

The stage changes in exactly the three images (`board-tier12/stage.SHA256SUMS`; boot 2's is
`stage-boot2.SHA256SUMS`). The firmware is fw_payload 2df129f4643a64fe (Image d38b07ac077e598a), gate PASS. The bake
checked the initramfs by hash, and csmint_dispatch and _inject_s_trap are in the ELF.

**The prediction:** the compiler lane's Tier 1 record on QEMU virtual-capstone-v7 852c5bc76774, which has strict DELIN
and one context slot by default. Serial: `<v8-kit>/t1-contract/guest/serial.log`; status: `<v8-kit>/status-t1`.
- It reads `T1_EXIT:ok:0` with VIRTUAL_APPLICATION_OK, then `T1_EXIT:stale:139` with one `domain fault cause=25`.
- There is no cause 24 or 26 and no delin. The geometry is the flat 1 MiB, 65,536 records.
- It ran the board's contract.dom bytes. Its module c4440adde0d02347 and launcher a405840dce789345 are the QEMU-kernel
  builds of the same sources as the board's ed24f827 and b27afefc, not a different revision.

**Rungs:** b0-stats, b0-hello, vm-contract (`board-tier12/boot3/run-tier1-3t1.sh 2df129f4643a64fe`).

**Predictions:**
- b0-stats 0 and b0-hello 0.
- **vm-contract 0, the identity branch:** `ok` prints VIRTUAL_APPLICATION_OK and exits 0; `stale` is denied with cause
  25 and exits 139. This is Tier 1's headline: a virtual application runs on silicon, and its stale access is
  denied.
- **The other branches, written down by the runtime lane:**
  - 26 anywhere: C-27 not fixed in the staged bytes. That is kit side, and the census was wrong. A 26 at map 0 base
    + 0x1f60 reads as the `delin` branch.
  - 24 from S: a remaining integer-based access, which the 0-ldc/stc gate says cannot exist.
  - 29 before VIRTUAL_APPLICATION_OK: header or layout, kit side.
  - Anything else goes to the RTL lane.

**Two gaps, named:**
- The prediction for `stale` comes from this new-pin record alone. No earlier board reading exists for it.
- The ADD_OK geometry on the board comes from boot 2's module, unchanged.

**The judge:** `boot3/vm-contract-branch3.py`, boot 2's branches plus `delin`. It reads boot 2's real transcript as
`delin`, and a fault 4 bytes off as `other`. The script was dry-run with every pin real and the driver stubbed
(`boot3/dryrun-boot3t1.txt`).

### Boot 3-T1, result (2026-10-18 03:17-03:23): the C-27 fix holds on silicon; vm-contract then faults at `malloc`'s read of a QEMU-only CSR (Q-21)

Result lines: `board-tier12/results/board-vp2-tier12-3t1.result-lines.txt`.
- **b0-stats, b0-hello:** 0.
- **vm-contract:** retval 922, on none of the pre-registered branches.
  - Both `ok` and `stale` exit 139 with `domain fault cause=2` at map 0 base + 0x10ccc, address 0xcc002573.
  - There is no VIRTUAL_APPLICATION_OK, no cause 24 or 26, and no monitor tag.
- **The C-27 fix holds on silicon.** Every virtual run passed `__capstone_cap_init` for the first time, where boot 2
  stopped at its delin.

**Located.**
- **The instruction:** VA 0x20ccc in contract.dom f2017064 is `csrr a0, 0xcc0` in `malloc`, and 0xcc002573 is that
  instruction's own encoding.
- **The cause:** CSR 0xcc0 (urevavail) does not exist on the RTL (ISSUES Q-21, `csr_regfile.sv:1079`), so a read is an
  illegal instruction. QEMU implements it, so every QEMU record passed.
- **The gap:** Q-21's rule ("never read 0xcc0") reached the module but not the application runtime. All three images
  read it twice, in `malloc` and `realloc`. The images boot 2 ran had the same two reads; the delin faulted first.

**Ruled by the runtime lane:**
- The compiler lane removes the reads from the runtime, whose availability signal is the hardware's pressure event, and
  rebuilds the images.
- The RTL lane makes 0xcc0 and 0x5c0 illegal by default in QEMU and re-pins.
- The Tier 1 and Tier 2 app-step records are re-made on that pin before anything is called a pass.
- Boot 3-T1 repeats on the rebuilt images with this pre-registration, the hashes moved and a `qemu-csr` branch added:
  cause 2 whose tval is a Zicsr instruction on 0xcc0 or 0x5c0 means the read was not removed, kit side.

**The gate, from here on.** Every staged image goes through `delin-census.py` and `qemu-only-csr-census.py` before it
is staged. `boot3/stage-boot3.sh` refuses otherwise, and it refuses today's images.
- The CSR census's positive controls: today's three images read 2 each, and the paged QEMU module 69a7016b reads 2
  in `vm_open`.
- Its negative controls: the board module ed24f827 and the QEMU flat module c4440add read 0.
- A non-ELF input reads NO DATA.
- The runtime lane has asked for one instrument covering every QEMU-only behaviour: a decode whitelist against the
  RTL's decoder. These two censuses are its first rules.

**What would have caught it before the boot.** A memory written earlier the same day named the habit: before a boot
predicted by a QEMU record, list the QEMU-only paths the images take. Q-21 was on dev, and the census is a two-minute
count, but it was not run for this boot. A habit in memory did not fire under time pressure; a gate in the staging
script does.

## The vp2 silicon gate (2026-10-18): every staged file against the RTL lane's census table

`vp2-silicon-gate.py` reads `capstone/docs/plans/2026-10-18-vp2-silicon-census.tsv` at run time. A change to the
table therefore changes the gate.
- **What it checks:**
  - every custom-2 and custom-3 instruction, SFENCE.VMA, MRET and SRET, by its row;
  - every CSR access, by its row, or by the table's CSR-privilege rule when the CSR has no row;
  - images as `in_context`, modules as `trusted_S`.
- **Two operand rules:**
  - a DELIN of a gp-derived operand;
  - an LDC/STC whose base an integer instruction wrote.
- **Trusted S, stricter:** an LDC/STC in a module is refused unless a capability instruction wrote its base in the
  same function, which is v7's contract (Q-23).
- `delin-census.py` and `qemu-only-csr-census.py` stay as its first two rules.
- **Controls, all firing:**
  - boot 2's images: 525/5,983/9,859 gp-derived delins and 2 reads of 0xcc0 each;
  - boot 3-T1's images: 2 reads each;
  - boot 1's module 0d62a1bc: 4 integer-base LDC/STC, the first at 0x27d4, boot 1's fault;
  - the paged module 69a7016b: 0xcc0 and 0x5c0;
  - the board module ed24f827: 0.
- **Two false positives were found and fixed before it was trusted:**
  - the operand scan crossed function boundaries (setjmp read `li a0, 0` from the function before it);
  - `unimp` (0xc0001073, a deliberate trap on both machines) was read as a CSR write. It is counted apart.
- **The rebuilt images** (contract 125705371837ea92, sqlite a5942a1add489597, mruby a902e740fc5d3b0f) read 0 refused
  and 1 candidate each: `libc_exit_fini`'s LDC through a base made by `addi`/`mv` from a capability. The table has no
  row for an integer ALU instruction on a tagged operand. The RTL lane is asked for one. Until it answers, the
  candidate holds boot 3-T1's repeat, because a non-empty report does not boot.

## Pre-registered (2026-10-18, before boot 3-T1's repeat): Tier 1 on the Q-21-fixed images

Boot 3-T1 with its images rebuilt once more. The runtime's malloc and realloc no longer read CSR 0xcc0 (the compiler
lane's fix f6dfd19d9396). Everything else is boot 3-T1's: module ed24f827, launcher, monitor 1e9451a, recipe,
bitstream, rungs and judge.

**The images** (`<hdd>/v9-kit`, where `<hdd>` is the run host's scratch volume): contract.dom 125705371837ea92, sqlite3.dom a5942a1add489597, mruby
a902e740fc5d3b0f. Each is exactly 8 instructions shorter than boot 3-T1's: two guarded reads at two sites.

**The gates, run on these exact hashes by `boot3/stage-boot3r.sh` before staging:**
- `delin-census.py` reads 0/0/0.
- `qemu-only-csr-census.py` reads 0/0/0 on the images and 0 on the module.
- `vp2-silicon-gate.py`, against the RTL lane's census TSV, reads 0 refused on all four files.
  - Each image has one candidate, `libc_exit_fini`'s LDC through an `addi`/`mv`-derived base. On both machines
    that is cause 24 (the RTL lane: commit_stage.sv:453/878; QEMU translate.c gen_set_gpr).
  - Each candidate is attributed in `boot3/silicon-gate-attributions.tsv` (the runtime lane's ruling): unreachable,
    because `.fini_array` has size 0 and `bgeu` at libc_exit_fini+0x44 skips the loop.
  - The gate verifies that evidence against each image's section header. A row keyed to another hash, or whose
    evidence fails (the control: /bin/ls's 8-byte .fini_array), is rejected.
  - The latent defect behind it, pointer arithmetic on a capability lowered to `addi`, goes to the compiler lane.

**The stage** changes in exactly the three images (`stage.SHA256SUMS`; boot 3-T1's is `stage-boot3t1.SHA256SUMS`).
The firmware is fw_payload ed7c599a132508f5 (Image 92fbb0e2ca4aa45b), gate PASS, with csmint_dispatch and
_inject_s_trap in the ELF.

**The prediction:** the compiler lane's re-made record on QEMU virtual-capstone-v7 3b8040d5ff6e, binary
809d1fc107e64515, on which 0xcc0 and 0x5c0 are illegal by default. The fixed arm's serial is
`<hdd>/csr-pair/fixed/guest/serial.log`, record cbfa22b4c957.
- `ok` exits 0 with VIRTUAL_APPLICATION_OK.
- `stale` exits 139 with one `domain fault cause=25` on contract.dom 125705371837ea92.
- No cause 2 and no "requires capability".
- Its control arm, on boot 3-T1's images, reproduced boot 3-T1's board line exactly: five faults at cause 2 with
  address 0xcc002573.

**Rungs:** b0-stats, b0-hello, vm-contract (`boot3/run-tier1-3r.sh ed7c599a132508f5`).

**Predictions:**
- b0-stats 0 and b0-hello 0.
- **vm-contract 0, the identity branch.** VIRTUAL_APPLICATION_OK, exit 0, then the stale access denied with cause
  25, exit 139. This would be the first virtual application on silicon.
- **The branches** (`boot3/vm-contract-branch3.py`):
  - `qemu-csr`: cause 2 whose tval is a 0xcc0 or 0x5c0 instruction. The read was not removed, kit side.
  - `delin`: 26 at map 0 base + 0x1f60, the C-27 glue, kit side.
  - 24 from S: a remaining integer-based access.
  - 29 before OK: layout.
  - Anything else goes to the RTL lane.

The script was dry-run with every pin real and the driver stubbed. It reads boot 3-T1's real transcript as
`qemu-csr`, a clean one as `identity`, and stops on a wrong firmware.

**Correction (2026-10-18, after the pre-registration above, before its boot's result):** the `libc_exit_fini` sites are
musl's source doing integer arithmetic through `(uintptr_t)`, not a codegen defect. More to the point, musl's
`libc_exit_fini` is never called in these images: the runtime's strong `__libc_exit_fini` overrides musl's weak alias,
and it is capability-correct. The compiler lane's control proved it (C-82, 6ec2d7b380c2). c82-destructor.dom
a250d387ddfd352d carries the pattern and a one-entry .fini_array, and it exits 0 after running its destructor. The
board lane's earlier claim to the runtime lane, that any image with a destructor faults at exit, is retracted.
- The gate is unchanged in what it reports: the instructions are present.
- Under the RTL lane's TSV row "integer ALU op on a tagged register" (dev 35b14e455880, b29410a7fe1a), the gate now
  REFUSES them rather than listing them as candidates.
- The attribution rows carry them, with the evidence checked. Their reason understates the case: the function has no
  caller at all.
- A gate refusal is a static pattern, not a predicted fault.
- Plain JALR through an untagged register is admitted (the TSV's new row): a plain jump, which faults only outside
  the PCC, and that is dynamic.

### Boot 3-T1's repeat, result (2026-10-18 03:49-03:55): C-27 and Q-21 fixed on silicon; the run reached the first reply and stopped at the adapter's QEMU-shaped empty-slot check (Q-30)

N = 1 boot, both contract runs identical. fw_payload ed7c599a132508f5, module ed24f827, contract.dom 125705371837ea92;
result lines in `board-tier12/results/board-vp2-tier12-3t1r.result-lines.txt`.

- b0-stats 0 and b0-hello 0, as predicted.
- **vm-contract 922, branch `other`** (not predicted): `ok` and `stale` both exit 139 with cause 24.
- **No cause 2 anywhere, so Q-21 is fixed on silicon.** malloc ran past its former 0xcc0 read. The first heap grant
  went through and the domain resumed, so C-27 holds too.

What the run reads, in order:
1. Five grants succeed (`CAPSTONE_VM_ADD_OK` slots 0-4). All are made before the first CSRUNV.
2. The sixth is the first grant made while the domain is paused at a service ECALL (kind 3): reg 10, LINEAR, 8192
   bytes (on QEMU this grant reads perms 7). It is refused: `CAPSTONE_VM_ADD_REFUSED ... errno=14` (EFAULT).
   - QEMU 3b8040d5ff6e reads `ADD_OK slot=5` at the same point (record cbfa22b4c957).
3. The domain's mmap returns failure. Its errno store faults with cause 24 at VA 0x1207c, `sw s1, 0(a0)`.
   - a0 comes from `__errno_location` (`movc a0, tp; cincoffsetimm a0, a0, -0x124`).
   - The address 0 - 0x124 says tp's cursor was 0. So the 24 is downstream of the refusal, not a cause in its own right.

**Located.** In the module (ed24f827's source, `<v9-kit>/out/adapter-board/module/capstone_vm.c`) a LINEAR add pins no
pages, so its -EFAULT returns are:
- the copies to and from user space;
- add() :622-623, `slot = t->frame + 8 + r->reg * 2; if (slot[0] || slot[1]) goto unpin;`.

The post-success copy marks the thread terminal. The next CV_STEP would then fail with ESRCH and the launcher would
die() on it. Instead the board printed `domain fault`, which needs a successful step, so that branch is excluded.
remap_end() :715-716 carries the same check. It is unreachable in this flat build: `exact_bounds` is false, so
remap_begin refuses first.

The slot is the frame's x10 at +224, and on silicon it is not empty at a service pause:
- **RTL (bf3db3175):** the frame IS the paused context's storage (vp2 item D, decision 1). Every escape saves x1..x31,
  with tags, into the frame's own slots (capstone_dom_switcher.anvil:155-161). The reply resume loads them and zeroes
  +224.
- **QEMU:** zeroes the slots at entry and keeps the context in its own state (capstone_supervisor.c:950-969).
- **The board read +224 itself, indirectly.**
  - The service's size argument, `args[0]`, is the module's `frame[72]` (+576).
  - On silicon +576 is written only by the switcher's copy from +224 (dom_switcher.anvil:492).
  - The launcher reached the grant, so that size was non-zero (exec.c:267-270).
  - So +224's low word was non-zero at this escape.
- The QEMU record ran the flat adapter c4440adde0d02347, built from the same source with the same define for the
  QEMU kernel.
- The claim-auditor confirmed this localization from source, with the corrections now in this text.

So an adapter that requires +224 to be empty passes only on QEMU. The item D plan asked for this divergence to be
recorded as a Q-entry (:71-73), and it was not. It is Q-30, filed by the RTL lane.

**Ruled by the semantics owner the same morning:**
- the RTL's frame is the reference;
- the adapter overwrites +224 instead of requiring it empty (the compiler lane's module revision);
- QEMU gains the frame write on escape (the RTL lane, the next pin).

The next run's pre-registration follows when the revised module's hash exists.

## The kit's QEMU firmware with the S-trap injection (capstone-sbi vp2/qemu-kit-inject c170e49, 2026-10-18)

**Why.** Since capstone-sbi 83966bc the FPGA monitor injects a trap from S or U that it does not service into S
(medeleg cannot route causes 24-30 on this RTL, Q-24). The kit's QEMU firmware did not inject, so the kit's QEMU
reference answered those traps differently from the board. c170e49 gives the QEMU monitor the same injection, behind
`CAPSTONE_S_TRAP_INJECT`. Without the define, the QEMU build is byte-identical to 4674ab6, and so are FPGA builds with
or without it.

**It returns through CEPC, not mepc.** Under cap_mem, QEMU's trap from S or U writes only CEPC
(cpu_helper.c:1986-1991), and mret returns through CEPC (op_helper.c:352-356). On the RTL, CEPC's cursor and mepc are
one register (Q-28). So the QEMU block and the FPGA block are paired by behaviour, not by bytes.
- The first port, f5f1bbe, read and wrote mepc. The claim-auditor refuted it, and it is the control below.

**Recipe:** `qemu-kit-fw/build-kit-qemu-fw.sh <capstone-sbi commit> inject <fresh dir>`.
- OpenSBI cf344cf3, PLATFORM=generic.
- The monitor C generated by capstone-c and checked by buildroot 8fd1ea12's check-monitor-asm.py.
- The define reaches capstone-c and all three of the platform's flag lines (cpp, c, as).
- fw_jump.elf: c170e49 inject = 2bc1dbbad1cc50d7; f5f1bbe inject = 34bb30a7dcdfb9d1.

**The behavioural pair** (`qemu-kit-fw/run-pair.sh`): run.py on QEMU 3b8040d5ff6e (binary 809d1fc107e64515, one
context slot) with the v9 images (contract.dom 125705371837ea92) and the flat adapter c4440adde0d02347. Predictions were
written into the script before it ran.

| arm | prediction at `threads` (the second CSRUNV, refused 29) | reading |
|---|---|---|
| c170e49 | INJS 0x1d, then Linux takes the trap at stvec and reports it itself; no INJL, no spin | `INJS` `0x1d`, then `[4.790252] Oops - unkno...`. run.py stops the guest at the first "Oops" (run.py:186-190), so the line is cut by the harness, not by a crash |
| f5f1bbe | INJS, then INJL with the stale mepc, and a spin to the timeout | `INJS` `0x1d` `INJL` `0x1d` `0xffffffff80003344`, then silence until the 180 s timeout (03:51:16 to 03:54:19) |

Both arms read `normal` as VIRTUAL_APPLICATION_OK first, and both run.py records are FAIL from `threads` on, because
QEMU now refuses the second context as silicon does (Q-26). The pair shows the injection reaching Linux. It says nothing
about the virtual tests after `threads`.

## Pre-registered (2026-10-18, before boot 3-T1's third run): Tier 1 with the Q-26 + Q-30 module

The repeat with ONE file changed: the board module, ed24f827 → **0cc4760d21fde17a**, built from llvm-capstone
457b78f27e10 by the kit's ADAPTER-board step (the compiler lane). Two changes from ed24f827:
- **Q-26:** contexts serialized on one slot. CV_THREAD_CREATE refuses with -EBUSY while a frame is live. Tier 1 creates
  no threads.
- **Q-30:** add() and remap_end() overwrite the reply slot instead of requiring it empty. The emptiness check stays for
  pre-start grants.

**The pins:**
- **Module.** In the image by hash (the bake's cpio check). It carries a string absent from ed24f827 ("thread create
  refused"), and its vermagic is the board kernel's.
- **Kit gates, on the module.** `qemu-only-csr-census.py` 0, and `vp2-silicon-gate.py` 0 refused (2 conditional, as
  ed24f827). The images, launcher and scripts are the repeat's, by `stage.SHA256SUMS`, which differs from
  `stage-boot3r.SHA256SUMS` in that one line.
- **Firmware.** fw_payload 31abc60d3d27d4be (Image 21e730020b2f6a1d). The monitor is unchanged: 1e9451a, wrapper
  1269c77, the repeat's defines, and the extracted monitor sources hash as the repeat's.
- **Preflight.** It reads as for boots 1-3: the DTS-against-bitstream, single-FDT and size-budget checks pass. Its
  overlay and control checks do not apply to a private baked image.
- **Scripts.** `boot3/run-tier1-3r3.sh 31abc60d3d27d4be`, with the judge `boot3/vm-contract-branch3r3.py`. The judge is
  the repeat's plus `reply-slot`. Both were dry-run with every pin real and the driver stubbed
  (`boot3/dryrun-boot3r3.txt`).

**The QEMU reference:** QEMU 809d1fc107e64515, one slot, the same source's flat module, contract.dom
125705371837ea92 (the compiler lane's q30-run, run.py 34/41).
- `ok`: 24 `CAPSTONE_VM_ADD_OK` lines, then VIRTUAL_APPLICATION_OK, exit 0. As (bytes reg perms linear):
  - pre-start: 524288 3 7 0, 1048576 2 6 0, 16384 10 6 0, 262144 11 6 0, 65536 12 4 0;
  - then 8192 10 7 1, 16384 10 6 1, 16384 10 7 1, 49152 10 6 1;
  - then 135168 10 6 1 three times, 32768 10 7 1;
  - then 135168 10 6 1 seven times, 65536 10 7 1;
  - then 135168 10 6 1 twice, 8192 10 7 1.
- `stale`: 12 ADD_OK lines (the same five pre-start; then 8192, 16384, 16384, 49152, 32768, 4096, 4096, all reg 10,
  linear), then one `domain fault cause=25`, exit 139.

**Rungs:** b0-stats, b0-hello, vm-contract.

**Predictions:**
- b0-stats 0 and b0-hello 0.
- **vm-contract 0, the identity branch:** VIRTUAL_APPLICATION_OK with exit 0, then the stale access denied with cause
  25 and exit 139. The first virtual application on silicon.
- **The Q-30 reading, in either branch:** the sixth grant, the first made at a service pause, reads
  `CAPSTONE_VM_ADD_OK slot=5 ... bytes=8192 reg=10 perms=7 linear=1`. With the old module that line cannot appear on
  silicon.
- **Recorded beside the verdict:** the grant sequence against QEMU's above. A differing sequence with an identity verdict
  is still recorded, as a divergence to read. This is the first silicon run of the heap's growth, retirement and mmap
  paths, and of a page-fault resume, so a stop there is a first reading, not a regression.

**The branches** (`boot3/vm-contract-branch3r3.py`):
- `reply-slot`: errno 14 at a grant again. The Q-30 change is not in the staged module, kit side.
- `qemu-csr` or `delin`: kit-side regressions. The images are unchanged, so either would be an instrument fault.
- `layout`: 29 before OK, kit side.
- `injected`: a trap from S or U handed to the kernel. Its cause and pc are the reading, for the module (compiler lane)
  or the RTL lane by what faulted.
- `monitor-stop`: NODM or INJL.
- Anything else goes to the RTL lane.

### Boot 3-T1's third run, result (2026-10-18 04:42-04:54): the kernel takes an instruction access fault in the new module's own text, at the rung's first ioctl

Pre-registered at d7a7b6206cf5, N = 1. Result lines in `board-tier12/results/board-vp2-tier12-3t1r3.result-lines.txt`.
- b0-stats 0 and b0-hello 0.
- **vm-contract never returned: branch `injected`.** The reading, before any grant:
  - MTVL:81D2E940, CPMX, INJS:2;
  - then Linux's own "Oops - instruction access fault" at `vm_ioctl+0xb60 [capstone_vm]`, badaddr 0x81d2e940;
  - then a kernel panic.
- **The Q-30 reading was not reached.** The rung stopped before the launcher's first grant.

Located, not attributed:
- epc ffffffff01113940 and badaddr 0x81d2e940 share their low 12 bits. The fault is the instruction FETCH of the
  module's own text page, physical 0x81d2e000.
- The kernel wrote that page at insmod, and its oops dump read the faulting instruction back from it. Data loads and
  stores reach the page; only the fetch faults.
- swap_cpmp found no live region covering mtval that was not already loaded in a CPMP entry. Its lookup skips loaded
  regions, so the reading cannot distinguish:
  - a hole in the region table;
  - a loaded entry that covers the page but refused the fetch.
- On the FPGA, print_regions and print_cpmps print nothing.
- This boot's carve tags are identical to the repeat's. The repeat's smaller module (ed24f827) ran the same path with
  its text on other pages.
- Modelled from R-55's exact-pieces rule (not read): the remainder below the CMA carves is one exact piece
  [0x80100000, 0xAC000000), which also covers the kernel's own text, so a covering region is predicted to exist.

## Pre-registered (2026-10-18, before the CPMX diagnostic run): the third run again, with the region table reported at a CPMX

**The one change:** the monitor, 1e9451a → capstone-sbi vp2/cpmx-diag 0c772b0.
- `CAPSTONE_CPMX_DIAG` makes swap_cpmp's CPMX path (FPGA) call `report_cpmx_diag(mtval)`. It reports:
  - every live region: id, CPMP slot, decoded base and end, permissions (lcc selector 5);
  - every loaded CPMP entry: id|region, bounds, permissions;
  - CPHT for a loaded entry that covers mtval.
- It is read-only (each capability is read and written back, as print_regions does).
- Without the define, `qemu-kit-fw/monitor-variant-gate.sh 1e9451a vp2/cpmx-diag` reads 0 differing lines on all three
  configurations.

**The pins:**
- Firmware e56ec94f185ad5ee: the third run's Image 21e730020b2f6a1d, wrapper 1269c77, the board's four defines plus
  CAPSTONE_CPMX_DIAG.
- The dom_stack gate passes (25712 B of 32768 B). `report_cpmx_diag` is in the ELF and swap_cpmp calls it.
- The stage is the third run's. Rungs b0-stats, b0-hello, vm-contract.
- Run by `boot3/run-tier1-3r4.sh e56ec94f185ad5ee`, judged by the third run's judge plus `boot3/cpmx-diag-read.py`.
  Both were dry-run (`boot3/dryrun-boot3r4.txt`).

**Predictions:**
- b0-stats 0, b0-hello 0.
- **If the fault recurs** (the same module, the same boot sequence, so likely the same placement), the dump names one
  branch:
  - `covered-loaded` with X in the permissions: a loaded entry covers mtval and the fetch was refused anyway, so the
    RTL's fetch-side check (the RTL lane). The model above predicts this, or the next.
  - `covered-loaded` without X: the monitor gave Linux a non-executable piece (the monitor's carve).
  - `covered-unloaded`: swap_cpmp's lookup missed a live covering region (the monitor).
  - `uncovered`: a hole in the region table (the monitor's carves; R-55's replay is then pointed at this boot).
- **If it does not recur:** vm-contract reads by the third run's branches, and the run is recorded as placement-dependent
  with the dump armed.

### The CPMX diagnostic run, result (2026-10-18 04:58-05:09): a LOADED, executable CPMP entry covers the fetch the silicon refused

Pre-registered at 4ed92db89028. N = 2 for the fault. Result lines in
`board-tier12/results/board-vp2-tier12-3t1r4.result-lines.txt`.
- **The fault.** b0-stats 0 and b0-hello 0, then the same fault at the same instruction as the third run:
  - Oops at `vm_ioctl+0xb60`, the instruction right AFTER the module's CSRUNV (`.4byte 0x48b717db` at +0xb5c);
  - mtval 0x81d2c940 (the module two pages lower, same page offset).
- **The dump (branch `covered-loaded`, with X).** CPMP entry 3 holds region 14, decoded [0x80100000, 0xAC000000),
  permissions 0x7, and covers mtval. All 11 loaded entries carry 0x7, and region 14 is the exact piece R-55's model
  predicted. So the silicon refused an S-mode fetch that, by the time M looked, a loaded executable entry covered.
  Pre-registered owner: the RTL's fetch side.
- **Why it depends on placement.** The S caller's FIRST fetch after a CSRUNV escape returned is what faults.
  - ed24f827: CSRUNV at .text 0x37f4, so the next instruction shares its 64-byte line. Every return worked on boot
    3-T1 and the repeat.
  - 0cc4760d: CSRUNV at 0x393c, the last word of its line, so the next instruction (0x3940) starts a new line. It
    faulted on both boots.
- **The working hypothesis, sent to the RTL lane with a directed test.** The frontend restarts at the restored pc during
  the ESCAPE's RESTORE walk (id 0, the pc, first; the CPMP ids later). A fetch that needs a fresh line is then checked
  against the CPMP state of that moment, and the fault rides the instruction to commit. The test is a CSRUNV at a line
  end against one mid-line.

## Pre-registered (2026-10-18, before boot r5): Tier 1 and Tier 2 in one boot, the module with CSRUNV line-aligned

**The change.** The module is 82250f627aba01f1: the compiler lane's 457b78f27e10 plus `.balign 64` before CSRUNV in
run() (`boot3/csrunv-align.patch`). This is a board-lane experiment, not the module's revision.
- An unmodified rebuild by the same recipe reproduced 0cc4760d21fde17a, the identity control.
- The variant's CSRUNV sits at .text 0x39c0 (64-aligned), with .text aligned to 64.
- Both kit gates pass (0 QEMU-only CSRs; 0 refused, 2 conditional).
- The CPMX diagnostic monitor stays armed: fw_payload 8129241bb6debad2, Image 551f48a9834bbbcd, monitor 0c772b0 with
  the board's defines plus CAPSTONE_CPMX_DIAG.

**Rungs:** b0-stats, b0-hello, vm-contract, then **vm-suite last** (the image already carries run.py's suite).

**Scripts.** `boot3/run-tier12-r5.sh 8129241bb6debad2`. It runs the judges:
- vm-contract-branch3r3.py;
- board-check3.py over the vm-suite rung, with run.py 85461f0e25ab66bf (the module revision's, identical to the kit's)
  and these eight tests pre-registered false: threads, vm, heap-threads, threads_output, vm_contract, heap_threads,
  shared_heap_pages_released, cleanup;
- cpmx-diag-read.py.

The dry run is in `boot3/dryrun-r5.txt`.

**Predictions:**
- b0-stats 0 and b0-hello 0.
- **Under the hypothesis:** the return fetch now shares CSRUNV's line, so vm-contract reads the identity branch (exit
  0, then 139/25). Its sixth grant reads `ADD_OK slot=5 ... bytes=8192 reg=10 perms=7 linear=1` (the Q-30 reading)
  and its grant sequence is QEMU's 24.
- **vm-suite: 22 exits and board-check 33 of 41.** Those are QEMU's 34 minus `cleanup`, the compiler lane's boot 4
  pre-registration (39cf6763d3a6):
  - threads, vm and heap-threads exit 1 (refused at create);
  - parallel_one and parallel_two 0;
  - preempt 143, every deliberate fault 139, sqlite and mruby 0;
  - no cause 29, no oops.
- **What refutes the hypothesis:** the same fault at CSRUNV+4.
- **What narrows it:** a fetch fault at the next line (CSRUNV+0x40) or later inside vm_ioctl. That would mean the window
  covers more than the restart fetch.
- Any other stop is read by its branch, with the dump if a CPMX appears.

### Boot r5, result (2026-10-18 05:11-05:22): the line-boundary hypothesis is RETRACTED; the fault is at CSRUNV+4 wherever CSRUNV sits

Pre-registered at ef4747aab249. Result lines in `board-tier12/results/board-vp2-tier12-r5.result-lines.txt`.
- **The fault, unchanged.** b0-stats 0, b0-hello 0, then the same instruction access fault and panic. epc is
  vm_ioctl+0xba4 = .text 0x39c4 = CSRUNV + 4, where CSRUNV now starts a 64-byte line: the faulting instruction shares
  CSRUNV's line and even its 8-byte block. The dump again reads CPMP 3 = [0x80100000, 0xAC000000) RWX covering mtval.
- **RETRACTED: the explanation the previous section offered** ("only a return fetch that needs a new line faults"). It
  was a lane-side working hypothesis, sent to the RTL lane as one, and never on a shared branch.
  - What would have caught it before the boot: the differential it rested on had two modules that differ in many
    respects (their Q-26/Q-30 code, their sizes, their physical placement), and the line was only the first
    difference noticed. One matched pair is not a localisation when the pair differs by more than one thing.
- **What stands, N = 3 boots of two builds of the compiler lane's revision.** The S caller's FIRST fetch after a CSRUNV
  escape returns (always CSRUNV + 4) is refused, while a loaded RWX CPMP entry covers it when M looks.
- **Unexplained:** ed24f827's returns (CSRUNV at .text 0x37f4) did not fault, on two boots and many steps.

## Pre-registered (2026-10-18, before boot r6): the compiler lane's module with a monitor that retries the refused fetch

**The change** (from the third run: monitor only). Module 0cc4760d21fde17a again (the stage is the third run's files,
by hash). The monitor is capstone-sbi vp2/fetch-retry 007d81d: the CPMX diagnostic plus `CAPSTONE_VP2_FETCH_RETRY`.
- An S/U instruction-fetch access fault whose mtval a LOADED CPMP entry with X covers is not injected. The monitor
  reports `FRTY:<mepc>`, executes sfence.vma and fence.i, and returns to the faulting pc.
- A second refusal at the same pc with fewer than 4096 instructions retired since is injected, as before.
- Define-off byte-identical in every configuration (monitor-variant-gate).

**The pins.** Firmware bbc3cd7e9e1d6b56 (Image 5a3b6462bc2cc6e4, wrapper 1269c77, the board's defines plus
CAPSTONE_CPMX_DIAG and CAPSTONE_VP2_FETCH_RETRY). The dom_stack gate passes, and swap_cpmp calls
fetch_retry_covered. Run by `boot3/run-tier12-r6.sh bbc3cd7e9e1d6b56`, dry-run in `boot3/dryrun-r6.txt`. Rungs
b0-stats, b0-hello, vm-contract, then vm-suite last.

**Predictions:**
- **If the refusal is transient** (the CPMP state the fetch was checked against at the escape is not what M reads
  later), the retry succeeds:
  - FRTY lines at the CSRUNV+4 pc, one per faulting return, and no Oops;
  - vm-contract identity (exit 0, then 139/25), its sixth grant `ADD_OK slot=5 ... bytes=8192 reg=10 perms=7 linear=1`;
  - vm-suite 22 exits and board-check 33 of 41 (the eight pre-registered false);
  - the first virtual applications on silicon.
- **If it is persistent** (a verdict that sfence.vma and fence.i do not clear), there is one FRTY at that pc, then the
  guard injects. The CPMX dump and Oops follow as in r4/r5, and the mechanism is the RTL lane's.
- **The FRTY count is a reading in either case:** how many returns fault.

### Boot r6, result (2026-10-18 05:23-05:55): TIER 1 PASSES ON SILICON, with the fetch retry; Tier 2 stops in `vm`

Pre-registered at 452c1611a52e. Result lines in `board-tier12/results/board-vp2-tier12-r6.result-lines.txt`. The run
was ended by SIGINT after ~25 minutes without progress. The driver released the board but wrote no board.txt, so the
lines come from driver.log, scoped to this run's load_image.

**Tier 1: the identity branch, the first virtual application on silicon.** b0-stats 0, b0-hello 0, vm-contract 0.
- contract.dom `ok` exits 0 with `VIRTUAL_APPLICATION_OK argv env files heap_growth heap_retirement mmap`. Its 24 grants
  equal QEMU's 24, checked by script, and the sixth is the pre-registered Q-30 line, `ADD_OK slot=5 ... bytes=8192
  reg=10 perms=7 linear=1`.
- `stale` is denied with cause 25, exit 139.

**The retry.** It fired twice in the whole boot (FRTY 2), both at retire+0xdc, the instruction after a CSRETIRE, before
any grant. After that, none of the hundreds of CSRUNV returns faulted.
- So the refusal is not per escape, and the retry turns it from a kernel panic into a continue.
- Its mechanism stays with the RTL lane. An rtl-oracle reading of bf3db3175: ENTRY's init strobe zeroes the CPMP bank,
  the ESCAPE restores the pc first and the CPMPs later, and a kill_s1 guard is meant to cover the window.

**Tier 2 (vm-suite).**
- normal 0 and threads 1, as pre-registered.
- `vm`'s first launch read as QEMU's. Its second launch made QEMU's grants up to vm_contract()'s `mmap(8193,
  PROT_READ)` (16 KiB), then nothing but heartbeats.
- Between that grant and QEMU's next, vm_contract() runs only `require(p[8192] == 0, "read-only first touch")`.

## Pre-registered (2026-10-18, before boot r7): r6 with a module that reports a repeated page fault

**The change:** only the module, 8fdfa127cf13a7c2 = 0cc4760d21fde17a plus a diagnostic (`boot3/resolve-diag.patch`, a
board-lane experiment). resolve(), the module's in-step page-fault resolution, prints `capstone_vm: DIAG fault
repeat=N cause= addr= pc= pte=` when it sees the same pc and address again (at repeats 2, 16, 256 and 65536). The pte is
the live PTE value via follow_pte.
- Both kit gates pass.
- Firmware 684684692449f302, whose monitor sources hash as r6's (007d81d, the retry and CPMX diagnostic armed).
- Rungs as r6. Run by `boot3/run-tier12-r7.sh 684684692449f302`, dry-run in `boot3/dryrun-r7.txt`.
- If `vm` stops again, the board lane ends the run by SIGINT once DIAG lines have appeared, or after 10 minutes without
  progress.

**Predictions:**
- Tier 1 as r6 (identity), and normal 0 and threads 1.
- **If `vm` is a fault loop** at the read-only first touch: DIAG lines for one address in the third page of the
  16 KiB arena, cause 13, pc in vm_contract. The PTE names the reason:
  - present with A clear: CVA6 faults on A = 0, which QEMU sets in hardware;
  - not present: the resolution did not reach the page table the domain walks;
  - present and complete: the domain's view is stale, which goes to the RTL lane with the PTE.
- **If no DIAG line appears** and `vm` still stalls: the stall is not a repeated fault in resolve(). It is then a wait
  elsewhere (the slot queue, a lock), read from what the module prints.

### Boot r7, result (2026-10-18 06:00-06:31): Tier 1 identity again (N = 2); `vm` stalls at the same point; the diagnostic was VOID

Result lines in `board-tier12/results/board-vp2-tier12-r7.result-lines.txt`.
- **Tier 1:** b0-stats 0, b0-hello 0, vm-contract 0 (identity), one retry before the first grant.
- **vm-suite:** normal 0 and threads 1, then `vm` stalled after the mmap(8193, PROT_READ) grant, as in r6.
- **No DIAG line, which reads nothing.** ~~The board's console does not show KERN_INFO: the module's own pr_info "thread
  create refused" is absent here and in r6, while the kernel's pr_warn lines show. The diagnostic was pr_info.~~
  - **Corrected 2026-10-18:** the conclusion stands, but the mechanism was wrong.
    - The baked driver sets `/proc/sys/kernel/printk` to 1 before the first rung (`fpga_driver/run_ladder_perf_fpga.py:201`), so during rungs the console shows only KERN_EMERG. pr_warn is hidden too.
    - The pr_warn lines cited above as showing were printed during boot, before that setup. In boot r11 there are
      695 kernel lines before it and 0 after it.

## Pre-registered (2026-10-18, before boot r8): S-24's -ELOOP module, with the PTE printed at pr_warn

**The change: only the module, 4131e4d8763e86dd.** It is the compiler lane's b04983b244e1 (S-24: a resolve repeated past
64 times for one address and cause fails with -ELOOP and a pr_warn line) plus `boot3/eloop-pte-diag.patch`. The patch
prints `capstone_vm: DIAG fault repeat=N cause= addr= pc= pte=` at pr_warn at repeats 2, 16 and 65; the PTE comes from
follow_pte.
- The unmodified b04983b244e1, rebuilt by the same recipe, reproduces the compiler lane's cc075f9cc5433275.
- Firmware 7b3f3e9a50de3e58: the monitor is unchanged (007d81d, its sources hash as r6/r7's).
- Rungs as r7, run by `boot3/run-tier12-r8.sh 7b3f3e9a50de3e58`.

**Predictions:**
- Tier 1 identity, normal 0, threads 1.
- At `vm`: DIAG lines with cause 13, an address in the third page of the 16 KiB arena, and the PTE.
  - **Expected (the A/D candidate, S-24):** the PTE is valid with A (0x40) clear, the same PTE at each repeat. Then the
    -ELOOP line, `vm` exits non-zero, and the suite continues.
  - **If the PTE has A set:** the walker's refusal is something else, which goes to the RTL lane with the PTE.
- The steps after `vm` are read against QEMU's record. Any other read-only first touch is predicted to end the same way.

### Boot r8, result (2026-10-18 06:34-06:46): Tier 1 identity (third boot); vm-suite runs seven steps, `vm` ends on a terminal store fault, and sparse-churn wedges the core

Pre-registered at bc776696ec59. Result lines in `board-tier12/results/board-vp2-tier12-r8.result-lines.txt`.
- **Tier 1:** b0-stats 0, b0-hello 0, vm-contract 0 (identity), two retries as before.
- **vm-suite**, in order: normal 0, threads 1, vm 139, nested 0, nested-stale 139/25, heap-threads 1, then a wedge in
  sparse-churn.
- **`vm`: RETRACTED 2026-10-18. The A/D candidate is NOT refuted; it is unresolved.**
  - **The original bullet** said the candidate was refuted, because the fault was "TERMINAL on its first occurrence"
    and there was "no DIAG and no -ELOOP line anywhere in the boot, so no fault repeated".
  - **Why that reads nothing:** those lines were pr_warn, and during rungs the console shows only KERN_EMERG (r7's
    corrected bullet above). "Terminal on its first occurrence" had no other support, because the launcher prints
    the same fault line whether the fault ended the step at once or after the module's 64-repeat -ELOOP bound.
  - **What stands:**
    - the store after `mprotect(p, 12288, RW)` faulted (cause 15) at the arena's base + 8192;
    - `vm` exits 139, with the same launcher line in r8, r9, r10 and r11;
    - QEMU passes this store.
  - **A reading, not a conclusion:**
    - r6 and r7 had no repeat bound. `vm` made no progress for 25 minutes after its last printed line, the
      mmap(8193, PROT_READ) grant, and nothing prints between that grant and this store.
    - r8-r11 had the bound, and `vm` exits 139 within seconds.
    - A fault loop ended by the bound reads this way, and so does a terminal fault in r8 beside an unrelated r6/r7
      stall.
  - **Not a reading:** resolve()'s `if (!same)` is a deduction from the source.
- **sparse-churn wedged.** The transcript stops mid-line. The apertures show M, a stall, a pending store, and the domain
  switcher at index 56 with a register write in flight. The trap log names a pc six instructions after the module's
  CSRUNV. Sent to the RTL lane.

## Pre-registered (2026-10-18, before boot r9): every vm-suite step except sparse-churn, which goes last

**The changes** from r8, two stage files:
- **The module:** the compiler lane's production ce3a542d00acce86 (69db2f133b93: S-24's -ELOOP bound, the
  thread-create refusal at pr_warn, S-21's child-frame retire).
- **The suite:** `board-tier12/vm-suite-inner-r9.sh`, run.py's guest script with sparse-churn moved to the end, before
  rmmod, because it wedged the core in r8. Everything else is in run.py's order.

**The pins.** Firmware b47f82487c6030f6 (the monitor unchanged, 007d81d). board-check reads to the end of the
transcript when vm-suite has no END marker. Run by `boot3/run-tier12-r9.sh b47f82487c6030f6`, dry-run in
`boot3/dryrun-r9.txt`.

**Predictions:**
- **Tier 1:** identity.
- **vm-suite, against QEMU's record (q30-run),** step by step:
  - normal 0, threads 1, vm **139** (r8's terminal store fault, the module's !same path, which this module keeps;
    QEMU 1), nested 0, nested-stale 139, heap-threads 1;
  - stale, bounds, retired, vm-ro-write, vm-none-read, vm-guard, vm-padding and sparse-stale each 139, with their
    pre-registered faults;
  - preempt 143, parallel_one 0, parallel_two 0, sqlite_write 0, sqlite_read 0, mruby 0;
  - then sparse-churn: a **wedge** as in r8, or 0 if r8's wedge was not deterministic.
- Every step from stale to mruby is a first silicon reading. A differing exit is named with its fault line.

### Boot r9, result (2026-10-18 06:51-07:04): Tier 1 identity (fourth boot, the production module); vm-suite reads 13 steps as pre-registered, then sparse-stale wedges the core exactly where sparse-churn did in r8

Pre-registered at 2727936779cd. Result lines in `board-tier12/results/board-vp2-tier12-r9.result-lines.txt`.
- **Tier 1:** identity on the compiler lane's ce3a542d00acce86.
- **vm-suite, all as pre-registered:**
  - normal 0, threads 1, vm 139 (r8's store fault), nested 0, nested-stale 139, heap-threads 1;
  - stale, bounds, retired, vm-ro-write, vm-none-read, vm-guard and vm-padding each 139.
- **sparse-stale wedged.** It ran in run.py's place, before the relocated sparse-churn. The transcript stops mid-way
  through the launcher's slot-12 grant line, as r8's sparse-churn did. The apertures are r8's in every field but the
  trap log.
- **One reproducer, the RTL lane's.** On both boots the domain's commit pc is contract.dom VA 0x20e98, `free+4`, free()'s
  first store. Both sparse modes begin `*slot = malloc(64); free(*slot);`.

## Pre-registered (2026-10-18, before boot r10): both sparse steps last, so preempt, the parallel pair, sqlite and mruby are read

**The change:** only the suite script, `board-tier12/vm-suite-inner-r10.sh`: sparse-stale, then sparse-churn, after
every other step. The module (ce3a542d00acce86) and the monitor (007d81d) are r9's. Firmware 9b3eb92a963acb7c. Run by
`boot3/run-tier12-r10.sh 9b3eb92a963acb7c`.

**Predictions:**
- Tier 1 identity; then r9's thirteen exits.
- **Then the first silicon readings, from QEMU's record (svadu=false reads the same, the compiler lane's
  fe8c4283f8ad):** preempt 143, parallel_one 0, parallel_two 0 (serialized by the queue), sqlite_write 0 with
  SQLITE_SUM=5050, sqlite_read 0 with SQLITE_PERSIST=100, mruby 0 with MRUBY_SUM=5050 and MRUBY_FILE=virtual.
- **Then sparse-stale:** a wedge at r8/r9's point, with sparse-churn behind it as collateral.

### Boot r10, result (2026-10-18 07:06-07:19): preempt, the parallel pair and every earlier step read as QEMU's; sqlite_write wedges the core in M

Pre-registered at bb4981135515. Result lines in `board-tier12/results/board-vp2-tier12-r10.result-lines.txt`.
- **Tier 1:** identity, the fifth boot.
- **vm-suite, through parallel_two, as pre-registered:**
  - r9's thirteen exits;
  - then the first silicon readings of **preempt 143, parallel_one 0 and parallel_two 0**. Both serialized processes
    print VIRTUAL_APPLICATION_OK with QEMU's 24 grants.
- **sqlite_write wedged.** Its output goes to a file, so the transcript shows nothing of it. The apertures differ from
  the sparse wedge: M, not stalled, an instruction valid at commit, the LSU idle, no monitor tag.
- **Silicon count so far:** 16 of vm-suite's 22 steps read. 15 match QEMU, and `vm` diverges (the module's store-fault
  path). sqlite_write wedges, and sqlite_read, mruby and the two sparse steps (a wedge of their own) are not yet read.

## Pre-registered (2026-10-18, before boot r11): which resolve() refusal ends `vm`, and does a fence before every CSRUNV remove the sparse wedge

**The changes:**
- **The module:** f02ca82ea9b2dd61 (the compiler lane's d66030133aa6):
  - every terminal return of resolve() prints `capstone_vm: resolve refused at <site>: ...` at pr_warn, at most one
    line per step;
  - a banner at insmod (`capstone_vm: loaded, diag resolve-sites-3, supervisor_slots=1`);
  - the S-24 -ELOOP bound;
  - `fence_before_entry` (default 0, 0644: a `fence rw,rw` before each of the four CSRUNV sites when set).
- **The suite:** `board-tier12/vm-suite-inner-r11.sh`.
  - sqlite is deferred to r12: r10's wedge is a second unknown, and it would cost the control.
  - The sparse pair runs with the fence set through sysfs, then sparse-stale runs with it cleared, last.
  - Each flip echoes the value read back.
- **The pins.** Firmware 4c2685def8817a87; monitor 007d81d unchanged. Run by `boot3/run-tier12-r11.sh
  4c2685def8817a87`, dry-run in `boot3/dryrun-r11.txt`.

**Predictions:**
- **Tier 1:** identity.
- **The banner must appear.** If it does not, every absent site line in this boot is void.
- **`vm`:** the compiler lane's per-site table (`vp2-flat-table-2026-10-17/PREREGISTRATION-resolve-refusal-sites.md` on
  lane/compiler-exact-bounds-iii, pushed before the run).
  - **`remap`** with cause 15, pfn_old != pfn_new, and prev(the same address, cause 13): the COW-on-write mechanism.
  - Another site names its own mechanism.
  - No site line with the banner present means the store does not fail in resolve() at all, and the !same deduction
    is refuted.
  - vm-ro-write and vm-none-read are predicted to print `vma` lines, as on QEMU (the compiler lane's control).
- **The other returning steps:** as r9/r10, plus mruby 0 with MRUBY_SUM=5050 and MRUBY_FILE=virtual (the first silicon
  reading).
- **The fence, read as four outcomes** (the semantics owner's table):
  - **fence=1 returns, fence=0 wedges:** the fence orders the race, which is a kit-side fix.
  - **Both wedge:** not the fence.
  - **Both return:** the race is timing-sensitive and the print or the order moved it. That is its own finding, not
    a fence pass.
  - **fence=1 wedges and fence=0 returns:** a different variable.

### Boot r11, result (2026-10-18 07:49-08:02): mruby wedges the core like sqlite_write; the module's console lines were unreadable, and a claim behind r8 is retracted

Pre-registered at b6521767bae0. Result lines are in `board-tier12/results/board-vp2-tier12-r11.result-lines.txt`.
- **Tier 1:** b0-stats 0, b0-hello 0, vm-contract 0 (identity, the sixth boot), and 4 FRTY retries.
- **vm-suite through parallel_two:** as r10. The 16 exits and the 9 launcher fault lines equal r10's in cause and in
  code-relative pc, compared by script.
- **mruby, its first silicon attempt: the core WEDGED.**
  - All 15 apertures equal r10's sqlite_write wedge: M, not stalled, an instruction valid at commit, the LSU idle,
    store state 0, switcher index 56.
  - Two large applications now wedge the same way. The sparse wedge's signature is different: stalled, with a pending
    store.
  - The fence flips and the sparse steps were not reached, so r11 has no fence reading.
- **The banner and the refusal lines are VOID, not absent.**
  - The baked driver's cold_boot sets `/proc/sys/kernel/printk` to 1 before rung 1
    (`fpga_driver/run_ladder_perf_fpga.py:201`, called at :272). Its DEVOK marker is in the stream.
  - From then on the console prints only KERN_EMERG. The reconstructed UART stream has 695 kernel lines before that
    setup and 0 after it.
- **RETRACTED with this boot** (dev 9e645777a5fb):
  - the board lane's claim that the console shows pr_warn: its evidence was boot-time lines;
  - r8's "the A/D candidate is refuted", which rested on absent pr_warn DIAG and -ELOOP lines. The candidate is
    unresolved. The r8 section above is corrected.
- **What would have caught it:** the banner was designed as the must-fire control, and it did its job: its absence
  is how this was found. The wrong part was the claim that chose the level. A same-level line had been checked only
  at boot, never inside a rung.
- **The ordering slip:** the suite put sparse-churn, a second expected wedge, between the two halves of the fence
  pair, so a churn wedge would have cost the fence=0 arm. The batching rule already says this (at most one expected
  wedge, last). r12 takes the compiler lane's order.

## Pre-registered (2026-10-18, before boot r12): resolve()'s refusal site read from the ring buffer, and the fence pair alone

**The changes from r11:**
- **The module** is 06d663f30bf29825 (the compiler lane's 0b963e269b4e).
  - It is r11's code: the llvm-objdump disassembly is equal line for line (5296 lines) and the section tables match.
  - The one change is the load banner, now at KERN_EMERG (level 0). It is the only module line the rung console can
    show. The five site lines and "thread create refused" stay at level 4. Diag tag resolve-sites-4.
- **The suite** is `board-tier12/vm-suite-inner-r12.sh`.
  - **The module's lines** are read from the ring buffer, `dmesg | grep capstone_vm:`, prefixed KMSG-:
    - right after `vm`;
    - again after the returning steps, with the buffer's line count.
    - There is no `-c`: busybox 1.36.1's dmesg supports it, but the whole-buffer read needs nothing more.
  - **mruby and sqlite are out:** r10 and r11 wedge on them alike.
  - **The fence pair goes in the compiler lane's order:** echo 1 with its read-back, sparse-stale, echo 0 with its
    read-back, sparse-stale, then sparse-churn LAST. Each read-back is labelled with the step it precedes.
- **Unchanged:** the firmware is 778836000a52d476, monitor 007d81d; the extracted monitor sources equal r11's.
- **The run:** `boot3/run-tier12-r12.sh 778836000a52d476`, dry-run in `boot3/dryrun-r12.txt`.

**Predictions:**
- **Tier 1:** identity.
- **The console banner, exactly once:** `capstone_vm: loaded, diag resolve-sites-4, supervisor_slots=1`, at the
  vm-contract insmod. If it is absent, the module did not load, and the boot is a different failure.
- **The vm-suite steps through parallel_two:** as r10 and r11.
- **The KMSG window after `vm`:**
  - **The must-fire:** threads' `capstone_vm: thread create refused`. If it is absent, the window is VOID: neither a
    site line nor its absence reads.
  - **`vm`'s site line,** per the compiler lane's table (PREREGISTRATION-resolve-refusal-sites.md on
    lane/compiler-exact-bounds-iii):
    - `remap` with cause 15, pfn_old != pfn_new, and prev = (the same address, cause 13): the COW-on-write mechanism;
    - `vma`, `anon`, `pin` or `arena`: each names its own mechanism;
    - must-fire present and no site line: the store does not fail in resolve(), and the !same deduction is refuted.
  - **What the A/D candidate predicts:** an `-ELOOP` line, "fault at ... resolved N times with no change".
- **The KMSG window after the returning steps:** the compiler lane's QEMU control predicts `vma` lines for vm-ro-write
  and vm-none-read.
- **The fence pair** (the runtime lane's four outcomes):
  - fence=1 sparse-stale returns and fence=0 wedges: the fence orders the race.
  - Both wedge: not the fence.
  - Both return: timing-sensitive, its own finding; sparse-churn then runs as the last unknown.
  - fence=1 wedges and fence=0 would have returned: unreadable in this order. That is a different variable, and it
    would need its own boot.
- **sparse-churn:** the expected wedge (r8), last. It is reached only if both sparse-stale arms return.

### Boot r12, result (2026-10-18 08:21-08:33): `vm` is a store fault loop that S-24's bound ends; the fence does not remove the sparse wedge

Pre-registered at 585f4391f32e. Result lines are in `board-tier12/results/board-vp2-tier12-r12.result-lines.txt`.
Every pre-registered reading held.
- **Tier 1:** identity, the seventh boot; 4 FRTY retries.
- **The console banner,** once at the vm-contract insmod: `capstone_vm: loaded, diag resolve-sites-4,
  supervisor_slots=1`, at KERN_EMERG. It also confirms the retraction's mechanism: level 0 passes the quieted
  console.
- **Through parallel_two:** the 16 exits and 9 launcher fault lines equal r11's, compared by script.
- **`vm`: a loop, not a single fault.**
  - The dmesg window after `vm` holds the must-fire (threads' "thread create refused") and two -ELOOP lines: `fault
    at 0x3faf4ba000 cause 15 resolved 65 times with no change; the page is pinned and the PTE is untouched by this
    module` (and 66, the launcher's repeat).
  - 0x3faf4ba000 is the launcher's own fault address: the store `p[8192] = 23` after `mprotect(p, 12288, RW)`.
  - No resolve() refusal site fired, so the `!same` deduction is refuted.
  - The A/D-loop candidate's predicted reading, an -ELOOP line, is what printed. r8's refutation of it was retracted
    this morning, and it was wrong in substance as well as in method.
  - What makes the store fault again after each resolution is open. It goes to the compiler lane, which owns the
    module, and to the RTL lane for the walker's side.
- **The vma lines:** vm-ro-write, vm-none-read and vm-guard print `vma` lines, need 0x2 against flags without
  VM_WRITE and need 0x1 against flags without VM_READ, as the compiler lane's QEMU control predicts.
- **The fence:**
  - fence_before_entry read back `Y`, and sparse-stale WEDGED.
  - All 15 switch apertures equal r8's and r9's fence-less sparse wedges, compared by script. The commit pc's code
    offset is also the same.
  - The fence does not remove the sparse wedge. The fence-off arms come from other boots and module builds.
  - This is also the RTL lane's prediction for its ghost-store candidate, an STC whose translation faults retiring
    without the exception. That candidate is in simulation.
- **A runner defect, caught by the live monitor:** the "FENCE flips" extraction matched digits, but the bool
  parameter reads back `Y`, so the extraction was empty. The result lines take the flip from the transcript. Later
  runners match `[0-9YN]`.

## Pre-registered (2026-10-18, before boot r13): S-24 two-sided, the `vm` loop with resolve_dirty off and on (a diagnostic, not a Tier 2 record)

- **The module:** a56fc415ad7133aa (the compiler lane's resolve-sites-10).
  - It passed its QEMU control 34/41, byte-identical to the baseline, with resolve_dirty=1 and resolve_pte=1.
  - 4683 instructions, 0 compressed.
  - Two new 0644 parameters, both bool, reading back Y/N:
    - `resolve_dirty`: for a cause-15 fault, fault the page in through the kernel's write path before the pin;
    - `resolve_pte`: print the faulting PTE in the bound line.
  - `fault_repeat_max` stays at its default, 64, so the bound is comparable with r12's.
- **The suite:** `board-tier12/vm-suite-inner-r13.sh`.
  1. normal and threads.
  2. `resolve_pte`=Y, then `vm` with resolve_dirty=N, then a dmesg read.
  3. `resolve_dirty`=Y.
  4. threads again: the post-flip window's own must-fire.
  5. `vm` again, then a dmesg read.
  6. The other returning steps, with resolve_dirty=Y.
  - No sparse steps (S-25) and no kit applications.
- **Unchanged:** firmware 3df8df7da5c9ea30, monitor 007d81d; the extracted sources equal r12's. Run by
  `boot3/run-tier12-r13.sh 3df8df7da5c9ea30`, dry-run in `boot3/dryrun-r13.txt`.

**Predictions:**
- **Tier 1:** identity.
- **The console banner,** once, with tag resolve-sites-10.
- **`vm` with resolve_dirty=N:** exit 139, as r8-r12.
  - The VM0 window holds threads' must-fire and the bound line at the launcher's fault address, cause 15, now with
    the PTE.
  - The runtime lane's account predicts `present=1 write=1 young=1 dirty=0`. A `dirty=1` refutes the account,
    including the fix's premise.
- **`vm` with resolve_dirty=Y:** exit 1, QEMU's status, with the cause left as a reading.
  - The VM1 window holds threads2's must-fire and no bound line. The PTE print belongs to the bound, so its absence
    is expected.
  - A resolve() site line there would name a different failure.
- **The other returning steps:** as r12. The fix covers cause 15 only, so the cause-13 steps (vm-none-read,
  vm-guard) keep their `vma` refusals.

### Boot r13, result (2026-10-18 09:34-09:42): S-24 is read on silicon and fixed. The PTE has D clear, and faulting the page in for write before the pin makes `vm` read QEMU's status

Pre-registered at ad7ef6004ffc. Result lines are in `board-tier12/results/board-vp2-tier12-r13.result-lines.txt`.
Every pre-registered reading held, compared by script.
- **Tier 1:** identity, the eighth boot.
- **The console banner,** once: resolve-sites-10.
- **`vm` with resolve_dirty=N:** exit 139. The bound line sits at the launcher's fault address and now carries the
  PTE: `val=0x20331857 present=1 write=1 young=1 dirty=0`.
  - The low byte 0x57 decodes to V, R, W, U and A set, with D clear: the runtime lane's account, read on silicon.
  - The RTL raises a store page fault while D is clear and never sets it.
  - GUP's `follow_page_pte` marks the struct page dirty, not the PTE (board tree `mm/gup.c:624-627`).
- **`vm` with resolve_dirty=Y:** exit 1, with `VM_CONTRACT_FAIL:execute through mapping PCC`, QEMU's own status and
  failure point. No new bound line after the flip; threads2's refusal is the window's must-fire.
- **The other returning steps:** equal to r12's in exits and in every fault line's cause and code offset. The three
  `vma` refusals hold the predicted relation. The fix covers cause 15, and a store to a mapping without VM_WRITE still
  refuses at `vma`.
- **The runner's per-window KMSG counts are cumulative.** dmesg is not cleared, so later windows repeat earlier
  lines. The result lines take each window's new lines by timestamp.

## Pre-registered (2026-10-18, before Tier 2 boot 4): the Tier 2 record on the module queue's revision, S-25's steps not run

- **The module:** 29c8e54e969e04a7 (the compiler lane's resolve-sites-12).
  - S-26's escape line (raw event words, budget 16).
  - resolve_dirty on by default.
  - The Q-26 wait fix: its QEMU control has 0 kernel WARNINGs where the baseline had 1, and a suite verdict
    byte-identical to the baseline.
  - Verified here: sums, 4684 instructions with 0 compressed, the banner at KERN_EMERG, vermagic 6.4.14.
  - The arena pin is NOT in it.
- **The suite:** `board-tier12/vm-suite-inner-b4.sh`, run.py's guest script.
  - **NOT RUN (S-25, R-57),** each printing `VM_NOT_RUN:<step>:S-25`: sparse-stale, sparse-churn, sqlite_write,
    sqlite_read and mruby. A capability store as a page's first touch wedges the core on bf3db3175, and the arena pin
    is not in this module.
  - resolve_dirty is read back before `vm`, as a reading rather than an assumed default.
  - The module's lines are read from dmesg.
- **Firmware:** 8b8d946fadcb4b13; the monitor (007d81d) is unchanged and its extracted sources equal r13's.
- **The run:** `boot3/run-tier12-b4.sh 8b8d946fadcb4b13`, dry-run in `boot3/dryrun-b4.txt`, with its own oracle of 17
  exits.

**The prediction:** the compiler lane's QEMU run on the reference pin, capstone-qemu b290f5656600 (binary
7a4f3a64776f0c88) with svadu=false, QEMU module arm 4397583b38a54a22.
- Its first run booted completely, and its 22 VM_EXIT lines were read here from its serial.log. Its judge then
  crashed on a missing CAPSTONE_LLVM_BIN, and that run's directory was replaced by the re-run. The judged record
  (pass count and false set) follows from the re-run and is added beside the result.
- **For the 17 steps this boot runs:** normal 0, threads 1, vm 1, nested 0, nested-stale 139, heap-threads 1,
  stale 139, bounds 139, retired 139, vm-ro-write 139, vm-none-read 139, vm-guard 139, vm-padding 139, preempt 143,
  parallel_one 0, parallel_two 0, cleanup 0.
  - The board's cleanup reads 1, because its kernel cannot unload the module, as in every board boot.
- **The fault lines:** each step's cause and code offset as r12 and r13 read them on the board, compared by script.
- **The board-specific readings:**
  - Tier 1 identity.
  - The console banner once (resolve-sites-12).
  - resolve_dirty reads back Y.
  - The VM window holds threads' must-fire and no bound line.
  - The returning window holds the `vma` refusals with their need/flags relation.
  - No wedge.

### Tier 2 boot 4, result (2026-10-18 11:09-11:22): VOID as a record. Module 29c8e54e's board arm faults every virtual context at startup

Pre-registered at f7487e0f1cd9. Result lines are in `board-tier12/results/board-vp2-tier12-b4.result-lines.txt`.
- **b0-stats 0 and b0-hello 0.** The console banner printed (resolve-sites-12), and resolve_dirty read back Y.
- **Every virtual context faulted before its HELLO,** at the same place, Tier 1's own vm-contract included
  (retval 922, the first Tier 1 failure in nine boots): `cause=24` at code offset 0x1d94, `address=0x2`,
  `entry=unknown`.
  - Every vm-suite step through preempt exits 139.
  - No S-23 retry fired.
  - The dmesg window after `vm` holds only the banner.
- **The rung then stopped returning in the parallel pair.** The driver's aperture read shows a LIVE core (the LED
  stretcher's OR: 224 = 0xff, 193 = 7, index 107), so it is a hang with Linux running, not a reading.
- **The attribution:** everything but the module is byte-identical to r13: contract.dom, the launcher, the monitor
  sources and vm-contract.sh. The variable is the module, a56fc415 to 29c8e54e.
  - Its QEMU arm passed the control, so the defect lives in the board arm or in a path only silicon exercises.
  - The verbatim lines went to the compiler lane.
  - Tier 2's record waits for a corrected module. r13's a56fc415 remains the last module read good on silicon.

## Pre-registered (2026-10-18, before Tier 2 boot 4b): the record on module a00109e9 (resolve_dirty off by default, on for `vm` alone)

- **The module:** a00109e9738a4b95 (the compiler lane's resolve-sites-13, commit 8bc0c4dd88cc). It is 29c8e54e with
  resolve_dirty defaulting OFF.
  - Verified here: the llvm-objdump text is identical, 5730 lines; `.sdata` is 8 bytes shorter, the initialised-true
    variable gone; the only other difference is the tag string.
  - The lane's source diff isolates boot 4's regression to that default.
- **The suite:** `vm-suite-inner-b4b.sh`, boot 4's except that resolve_dirty is read back at its default, set to Y
  for `vm` alone, and cleared right after, each with a read-back. This is how r13's dirty=Y arm read it.
- **Firmware** dea932c3051aeebf; the monitor is unchanged.
- **The run:** `boot3/run-tier12-b4b.sh dea932c3051aeebf`, dry-run in `boot3/dryrun-b4b.txt`.
- **Predicted:**
  - Tier 1 identity, which boot 4 lost.
  - The banner (resolve-sites-13).
  - The read-backs N, then Y, then N.
  - The 17 exits as boot 4's pre-registration lists them, from the reference-pin QEMU run: normal 0, threads 1,
    vm 1, nested 0, nested-stale 139, heap-threads 1, the seven 139s, preempt 143, the parallel pair 0, cleanup
    (0 on QEMU, 1 on the board).
  - Fault lines as r12 and r13.
  - No cause-24 startup fault.
  - The five S-25 steps NOT RUN.
  - No wedge.

## Pre-registered (2026-10-18, before Tier 2 boot 4c): r13's module with boot 4b's suite. It is the control for the module attribution, and the Tier 2 record

- **Boot 4b** (module a00109e9, resolve_dirty read back N at its default) faulted every context exactly as boot 4
  did: cause 24 at code offset 0x1d94, address 0x2. So resolve_dirty's default was not the variable.
- **The staged files differ from r13's only in the module and the suite,** compared by script. The first failing
  context, vm-contract's, uses no suite.
- **Boot 4c puts r13's module back:** a56fc415ad7133aa (resolve-sites-10, from `diag-r13/`, sums ok). Everything
  else is boot 4b's.
  - The suite `vm-suite-inner-b4b.sh`: resolve_dirty read back at its default, Y for `vm` alone, then N.
  - The five S-25 steps are not run.
  - Firmware baa132ee2fbd535e; the monitor is unchanged.
- **The run:** `boot3/run-tier12-b4c.sh baa132ee2fbd535e`.
- **Predicted:**
  - Tier 1 identity, as in every boot with a56fc415 (r11's code base through r13).
  - No cause-24 fault.
  - The read-backs N, then Y, then N.
  - The 17 exits as the reference-pin QEMU run reads them: vm 1 under the explicit Y, and cleanup 1 on the board.
  - Fault lines as r12 and r13.
  - No wedge.
- **What it decides:** if 4c reads identity, the regression is in what changed from a56fc415 to 29c8e54e/a00109e9
  other than the default: the Q-26 wait rewrite or the escape-print code. If 4c fails too, the module is not the
  variable, and something else in the boot changed.

### Tier 2 boot 4b, result (2026-10-18 11:26-11:38): VOID as a record. resolve_dirty's default was not the variable

Pre-registered at 5143ae4e0601. Result lines are in `board-tier12/results/board-vp2-tier12-b4b.result-lines.txt`.
- resolve_dirty read back N at its default, Y before `vm` and N after it.
- **Every context faulted as in boot 4,** Tier 1 included (retval 922): cause 24 at code offset 0x1d94, address 0x2.
  - 0x1d94 is contract.dom VA 0x11d94, memset+0x2c, `sb a1, 0x0(a3)` with a3 the destination.
  - So memset was handed a pointer whose cursor is 2, a value read out of memory before HELLO.
- **The pre-registered "no cause-24 fault" failed.** The compiler lane's attribution to the resolve_dirty default
  is withdrawn by this boot.
- **The module is still the variable:** only it and the suite differ from r13, and the first failing context uses
  no suite. Boot 4c is the control.

### Tier 2 boot 4c, result (2026-10-18 11:38-11:46): THE TIER 2 RECORD. Every run.py step outside S-25 reads QEMU's result on silicon

Pre-registered at 8ac6cc3143d8. Result lines are in `board-tier12/results/board-vp2-tier12-b4c.result-lines.txt`.
Every pre-registered reading held, compared by script.
- **Tier 1:** identity.
- **The 17 exits equal the reference-pin QEMU prediction** (capstone-qemu b290f5656600, svadu=false):
  - `vm` reads 1 under resolve_dirty=Y for it alone, read back N, then Y, then N. This is S-24's fix.
  - cleanup reads 1 because the board kernel cannot unload the module.
- **The 16 steps shared with r13** equal it in exit and in every fault line.
- **S-25's five steps were not run.** They wait on the RTL fix (R-57) or the module's arena pin.
- **No wedge.**
- **The module control:** with 4b's suite, image, launcher and monitor, only the module changed (a00109e9 there,
  a56fc415 here). 4b failed every context and 4c reads the record, so boot 4's regression lies in the Q-26 wait
  rewrite or the escape-print code. It is bisected next.
