# Whole-corpus audit and cross-program comparison, 2026-10-10

The second audit of the day. The first (`10-10-2026_17-20-00_bug-corpus-audit-three-programs.md`)
traced every cell of FFmpeg's, tshark's and memcached's tables to its record. This one reads all 28
corpora, compares the other six programs' bugs and methods with those three, and fixes what the
comparison showed needed fixing. It also adds virtual-Capstone arms to the three programs and hunts for
the nested-allocator defect shapes they lack. Lane branch `audit2-corpora`.

## Pre-registered runs (each section committed and pushed before its run)

### R1. memcached allocator-repros on stock CheriBSD: a fixed arm for every case, and case 8 attributed

The runner (`runners/cheribsd/run-defects.py`) had no fixed run, so case 8's CHERI catch -- the whole
of "memcached nested spatial CHERI 1/4" -- had no control and no attributed site. The fault was
recorded only as `pc=0x105530`. The runner now runs FIXED then buggy on the same program in one boot
(`corpus.h`: event value 1 selects the fixed sequence), and judges a buggy arm with declared
`fault_sites` by resolving the fault pc against the program's own symbols (`llvm-nm` from the SDK,
the supervisor's resolved `mc_defect_read` as the load anchor). Case 8 declares `mc_case_body` (its
`case.json`, `fault_sites_why`, from the source).

Platform: stock CheriBSD purecap, image `~/cheri/output` (0cb16209), libc revocation on, all 9
cases, one boot, with the revocation control and the platform controls in the same boot.

Predictions:
- fixed arm, all 9: completes (exit 0, completed=1, no fault).
- buggy 00-07: completes, as recorded on 2026-10-06 (the cache keeps the object; nothing reaches free()).
- buggy 08: SIGPROT, bounds (si_code 1), at a pc inside `mc_case_body`.
- revocation control: faults at its labelled load.
If 08's fault is elsewhere, or its fixed arm does not complete, the cell becomes NO-READING and
memcached's nested spatial CHERI cell drops from 1/4 to 0/3.

### R2. FFmpeg subobject-repros, capstone-subobject arm: the accept-anywhere waiver replaced

`tools/run-capstone-domain.py` accepted a fault ANYWHERE as a catch on this arm (`or subobject`).
It now accepts a fault only at the labelled probe or in a function the case declared before the
run (`fault_sites`). Its probe pattern also failed to recognise the unprefixed `write_probe` of
cases 05 and 08. Declared sites, from the source:
- `ff2_case_run` for 00-03, where the case body makes the defective member access;
- `memcpy` for 07 (`ff2_strlcpy`'s copy). This is the weaker form, stated as such in its case.json.

Platform: the 2026-10-10 audit's physical Capstone VM (capstone-qemu 32e7c9754f9a, kernel e58613598c89,
firmware 6f2b082cb677, module a88ed2159b43, capstone-exec 0752aa7c49c9), SDK level0, with
`-Xclang -fcapstone-subobject-bounds`, all 9 cases, one boot.

Predictions:
- controls: clean RETURNED, oob CAUGHT, uaf RETURNED, subobj CAUGHT.
- 00, 01, 02, 03: CAUGHT in `ff2_case_run`, a declared site.
- 05, 08: CAUGHT at `write_probe`, the labelled probe.
- 07: CAUGHT in `memcpy`, a declared site.
- 04, 06: NOT CAUGHT (the defect reproduces).
- Every fixed arm FIXED.

### R3. CheriBSD subobject and plane carve-bounds arms, supervised

Until now both runners ran the buggy arm directly, so their catches carried an exit status (162) and
no pc. The buggy arm now runs under `supervise`, and `tools/attribute-cheribsd-faults.py` names the
function each fault landed in (`attribution.tsv`). Platform: stock CheriBSD 0cb16209, one boot per
corpus.
- R3a `ffmpeg/subobject-repros`, `CHERI_EXTRA_CFLAGS="-Xclang -cheri-bounds=subobject-safe"`
  (cheribsd-subobject):
  - fixed: all 9 FIXED;
  - buggy: SIGPROT for 00, 01, 02, 03, 05, 06, 07 and 08, and 04 completes;
  - attribution: 00-03 in `ff2_case_run`, 05/06/08 at `write_probe`, 07 in `memcpy`.
- R3b `ffmpeg/plane-repros`, `CHERI_EXTRA_CFLAGS=-DFFP_CARVE_BOUNDS` (cheribsd-carve-bounds): fixed
  FIXED; buggy SIGPROT at `ffp_read_probe`.

### R4. wmem PoisonCap: the fixed runner, as a validation run

`ports/wireshark/wmem/host/cheribsd/poisoncap/run.py` now resolves each case's own label, with a
`supervise-wm_defect_write` for the write cases. The 2026-10-09 run attributed 16, 17, 18, 20 and 21 by
hand. Run: cases 13-21 (the spatial cases), modes 0 and 1, PoisonCap image c8df9e17, revocation on.

Prediction:
- every arm SIGPROT, bounds, in both modes (as on 2026-10-09);
- the runner itself reports "at probe" for all nine: 13, 14, 15 and 19 at `wm_defect_probe`, and 16,
  17, 18, 20 and 21 at `wm_defect_write`;
- 0 faulted_elsewhere.

### R5. FFmpeg pool-repros column 2 (`poolstock`) with an in-boot Sublet-heap control

Only the build tied the `poolstock` arm (column 2: FFmpeg's pools stock on the Sublet heap) to the
Sublet heap. A level0 heap would also let the four cases complete. Run in one physical Capstone VM
boot:
- app safety fixtures 4 and 5 on `poolstock` (`ports/ffmpeg/app/host/safety-expect.txt`, rows added
  today), predicted FAULT temporal -- what the Sublet heap does and level0 does not;
- then the four cases (`runners/run-sublet-port.sh poolstock`), predicted to complete as recorded
  (the pool keeps the buffer; nothing reaches free()).

Images are the existing `domain-sublet-poolstock` build; their hashes go in the record.

### R3c. subobject-repros on cheribsd-subobject again, with the supervisor reporting the faulting object

In R3a, case 07's fault pc lay outside the program image: libc is a shared library on CheriBSD, and
`attribute-cheribsd-faults.py` resolved only the program's own symbols. So it read "outside every
function". `supervise.c` now prints, after a fault, the mapped object that holds the pc and the one
that holds the return address, each with its load base. The tool resolves both against that object's
ELF (the program, or the library under the sysroot).

Same platform and flags as R3a. Predictions are as R3a for all 9, and for 07: SIGPROT in libc's
`memcpy`, the declared site, with its return address in the program (`ff2_strlcpy` or
`ff2_case_run`).

### Phase 0: the virtual platform, rebuilt on apollo, and its gate (done before any case below ran)

The CPython, PostgreSQL and SQLite virtual bundles name a platform (compiler 4c6135fe, qemu c676fd49,
kernel e833dfb1, launcher 5240ec2c, module 2394a455) that exists on no host this lane can reach, and
whose builder recorded no host. So the platform was rebuilt from dev and the virtual-capstone QEMU
branch:
- compiler: Release LLVM at dev bd372e25f5fb (`capstone-image-gp` present);
- qemu: capstone-qemu `virtual-capstone` 1a6dd2073206, qemu-system-riscv64 172111865d9a,
  `x-capstone-exact-bounds` present;
- kernel e58613598c89, firmware 6f2b082cb677 and rootfs 9903242c37a7 (the physical platform's images,
  copied into the kit);
- adapter built against that kernel's prepared tree, with the QEMU tree's revocation-table ABI
  header: capstone-vexec 5ba594e58775, capstone_vm.ko d36fab7edce1.

It is a DIFFERENT platform from the other programs' bundles, digest by digest, and every bundle
written on it says so.

Gate, on this kit:
- the runtime's own contract program: VIRTUAL_APPLICATION_OK (argv env files heap_growth
  heap_retirement mmap);
- postgres/c-repros on `virtual-malloc`, through its own runner: 5 of 5 cases agree with the committed
  2026-10-10 bundle on verdict, reason, fault cause and faulting function (all CAUGHT, cause 28);
- the comparison itself fires: a copy of that bundle with one verdict flipped reads 1 of 5 differing;
- with exact bounds OFF (a kit whose qemu strips the flag), 5 of 5 differ -- the launcher does not run
  an image at all without exact bounds, so every case reads NO-READING infra.

`tools/run-virtual-cases.py` passed a synthetic smoke case:
- both controls faulted;
- a synthetic over-read was caught at `ffh_read_probe` with cause 28;
- a physical SDK and a physical configuration were refused with exit 75.

### R6. FFmpeg, tshark and memcached on virtual Capstone: plain-heap, plain-temporal, subobject, carved

Arm `virtual-malloc` (configuration `virtual-mallocng`) on all eight corpora, and `virtual-nested-pools`
(`virtual-ffmpeg-carve`, `-DFFC_SUBLET_CARVE`) on carved-repros. Runner `tools/run-virtual-cases.py`,
one boot per corpus on the kit above. The controls bounds-malloc and uaf-malloc run in each boot, and
every fixed arm runs on the same image.

Predictions:
- controls fault; every fixed arm prints VERDICT FIXED and exits 0.
- plain-heap, 46 cases (25 FFmpeg, 12 tshark, 9 memcached): CAUGHT at the labelled probe, cause 28.
  This includes the five cases CheriBSD's size classes absorb, since mallocng bounds each object
  exactly. A case at a size whose padding the virtual heap leaves accessible would read MISSED, and
  would be the representability question the virtual README leaves open.
- plain-temporal, 26 cases: CAUGHT at the labelled probe, cause 24 or 25 (mallocng retires the freed
  object's lifetime).
- subobject, 9: MISSED. The member crossing stays inside one allocation, and this arm has no field
  bounds.
- carved on virtual-malloc, 13: MISSED. The crossing stays inside the one av_calloc block.
- carved on virtual-nested-pools, 13: CAUGHT at the labelled probe, as the physical Sublet carve read
  (13/13).

### R7. wmem-repros on virtual Capstone: both columns, with controls that can fire both ways

**The port.** The changes come from the unmerged prior-art branch `virtual-capstone-bug-corpora`,
commits 319f385c2d3e and 6483372d3ffa. Only the hunks the corpus needs were taken:
- a `capstone-application` preset;
- a `WM_SUBLET` option, which puts the Sublet region and chunk layers into a hosted capability build;
- `WM_DOMAIN` split into `WM_DOMAIN` (freestanding) and `WM_CAPABILITY` (Sublet handles).

The branch's line-shifting hunks were dropped (comments in `backing.c`, the `port.h` payload override,
`main.c`), because the CheriBSD and PoisonCap builds are Debug builds and their line tables moved. With
those hunks gone, 185 of 185 physical images are byte-identical before and after the change:
- the domain builds: chunks, region, reference and control 90;
- native libc and native chunks;
- CheriBSD libc;
- PoisonCap.

A rebuild with no change also reproduces every hash, so the comparison can fire.

**The corpus.** `driver.c` is untouched. `shared/driver-virtual.c` includes it and replaces only the
hosted `main()`:
- both modes run;
- the payload comes from `__capstone_sublet_malloc_linear`;
- the fix differential runs in either mode.

The new control `controls/virtual/91_ctl_chunk_freed_in_block` frees a BLOCK chunk individually, then
reads it. `controls/virtual/90` is a link to the existing jumbo control.

**The runner.** `tools/run-virtual-cases.py --prebuilt` runs the port's own builds:
- `virtual-malloc`: `-DWM_LIBC_SYSTEM=ON -DWM_CHUNKS=OFF`, the build the CheriBSD libc arm ran.
- `virtual-nested-pools`: `-DWM_SUBLET=ON -DWM_CHUNKS=ON`.

It refuses a build whose CMakeCache or SDK is not the arm's. A catch is a fault in the case's own probe
function (`wm_probe` for a read, `wm_write_probe` for a write, from its case.c), after its
`WM_DEFECT case=N ready` line. Its tests include a planted off-probe fault (NO-READING) and a mutant
that accepts any fault (2 tests fail).

Platform: the Phase 0 kit, the virtual SDK and LLVM bd372e25f5fb. One boot per arm. The controls run
in the same boot.

Predictions:
- Controls on `virtual-malloc`: bounds-malloc, uaf-malloc and 90 fault; 91 completes. On
  `virtual-nested-pools`: all four fault.
- `virtual-malloc`, 22 MISSED. Each buggy differential prints DEFECT-REPRODUCED and exits 0, as all 22
  did on the CheriBSD libc arm:
  - the 13 temporal cases' stale chunk is reoccupied inside a block wmem kept;
  - the 9 spatial crossings stay inside one block;
  - virtual mallocng sees neither.
- `virtual-nested-pools`, 22 CAUGHT in the case's probe function, as the physical sublet-chunks arm read
  (22/22):
  - temporal: cause 24 or 25;
  - spatial: cause 28.
- Every fixed arm: VERDICT FIXED, exit 0.
- Hazards, written down before the run:
  - Several cases faulting at one pc outside their probe is a harness fault, never a catch. The rule
    reads it NO-READING (unattributed). The prior-art branch's memcached run showed 6 such faults.
  - A 384 MiB payload the VM cannot give reads CONTROL-FAILED, and every case would be NO-READING; the
    fix would then be a smaller payload, recorded.

### R8. memcached allocator-repros on virtual Capstone: both columns

**The port.** From the prior-art branch (6483372d3ffa) come the `capstone-application` preset and
`MCP_SUBLET`: the shared ledger on the Sublet authority, in a process. Its `main.c` hunk is not taken,
because the corpus has its own driver. A guard refuses `MCP_SUBLET` together with `MCP_STOCK_MALLOC`.
134 of 134 physical images are byte-identical before and after: native, native stock, domain,
CheriBSD and PoisonCap.

**The corpus.** `shared/build-cases.sh capstone-application` builds the port on the virtual SDK and links
each case with its `capstone-cc`. `shared/driver-virtual.c` (MCP_SUBLET builds only) includes
`driver.c` unchanged, and replaces `main()`:
- the mode is an argument;
- the payload is lent by `capstone_borrow_aligned_block`.

The Sublet build also compiles the cases with `MC_CARVE_BOUNDS`, because the physical column 3 reads
`sublet-carve` for 06 and 07. For every other case the flag leaves the code identical; case 02's and
case 05's disassembly was compared. The new control `controls/virtual/90_ctl_chunk_freed_to_slab`
frees a slab chunk to its class with `slabs_free`, then reads it.

**The arms:**
- `virtual-malloc` maps to `virtual-memcached-stock`: `MCP_STOCK_MALLOC`, the ASan arm's ledger, run as
  `buggy N`.
- `virtual-nested-pools` maps to `virtual-memcached-ledger`: `MCP_SUBLET`, mode 1, run as `buggy N 1`.

Both arms take the runner's `--prebuilt` mode with the corpus's own result lines. A catch is a fault in
`read_probe`/`write_probe`, or in a site declared before the run: case 08's `mc_case_body`, declared
for R1.

Predictions:
- Controls: bounds-malloc and uaf-malloc fault on both arms. 90 completes on `virtual-malloc` and faults
  at `read_probe` on `virtual-nested-pools`.
- `virtual-malloc`, 1 CAUGHT and 8 MISSED, the physical `sublet-malloc` reading (1/9):
  - 00-04 (temporal): MISSED. A chunk goes back to its class's free list and an object to cache.c's;
    neither reaches `free()`, and ASan saw none of them (0/5).
  - 05, 06, 07 (crossings inside one page): MISSED.
  - 08: CAUGHT, cause 28, in `mc_case_body`. The scan runs past the 16 KiB rbuf object, which is its own
    virtual-mallocng object.
- `virtual-nested-pools`, 9 CAUGHT, the physical column 3 (5/5 temporal, 4/4 spatial):
  - 00-04: cause 24 at `read_probe`;
  - 05: cause 28 at `write_probe`;
  - 06: cause 28 at `write_probe`, the carve's suffix bound;
  - 07: cause 28 at `read_probe`, the carve's key bound;
  - 08: cause 28 in `mc_case_body`.
- Every fixed arm: VERDICT FIXED, exit 0.
- Hazard: the prior-art branch's stock run faulted 6 cases at one pc with cause 24, unattributed. If that
  recurs here, those cells are NO-READING (unattributed or setup-fault), never catches. It would be a
  port defect to find, not a reading.

### R9. wmem-repros 22 (c702b44a01, USB HID double free) on every arm

**The case.** It comes from the nested-shape hunt (`docs/ref/nested-shape-hunt-2026-10.md`). It is a fix
reversal, like cases 16 and 17. The record is `22_c702b44a01_usbhid_output_usages_freed_twice/`:
case.c, case.json and PROVENANCE.md, which quotes `c702b44a01^` by line.

**Construction.** The native fix differential was run while building the case (the case's own
gate 8), so that arm's outcome was seen before this pre-registration:
- the first reduction's observable (two allocations sharing storage) did not occur in the faithful
  sequence;
- the recorded observable is the free list losing the INPUT field's freed array;
- buggy DEFECT-REPRODUCED and fixed FIXED, 3 of 3 each.

No protected arm has run.

**Runner changes, each tested both ways before this run:**
- `poisoncap/run.py`: a case with no labelled probe and declared `fault_sites` uses `wm_defect_probe`
  only as the load-base anchor, and is caught only by a fault in a declared function. In a stubbed
  test, a fault in `wmem_block_free` and an unresolved pc both read unpaired.
- `run-virtual-cases.py`: a declared site that is a zero-size label (`wm_widen_probe`) matches only
  its own instruction. A pc 4 bytes off reads NO-READING (unattributed).

**Sites, declared from source before any protected run (case.json):**
- `wm_widen`: PoisonCap's handback load;
- `wm_widen_probe`: the chunk port's `wm_chunk_of` handback probe on Capstone.

Predictions, with the control that must fire in the same session:
- **native-fix-differential** (`runners/run-native.sh`): two-sided.
- **native-detect** (`runners/run-asan.sh`): no ASan report; the fixed arm FIXED; the ASan controls
  fire.
- **cheribsd-revocation** (stock CheriBSD 0cb16209, `WM_LIBC_SYSTEM`, differential under
  supervise): complete, DEFECT-REPRODUCED, NOT CAUGHT. The revocation control faults at its label.
- **poisoncap-spatial** (mode 0): complete. **poisoncap-protected** (mode 1): SIGPROT si_code 2 in
  `wm_widen`.
- **Capstone domain, region build** (`WM_CHUNKS=OFF`): `spatial` and `sublet` both complete. Case 11
  in the same runs must fault at the read probe in sublet mode, cause 24.
- **Capstone domain, chunk-port build** (`WM_CHUNKS=ON`): `sublet-chunks` (mode 1) faults at the
  labelled allocator probe `wm_widen_probe`, cause 24.
  - Mode 0 on this build is predicted to stop at the port's own double-free check (`wm_chunk_of`,
    `wm_fail(260)`). That is the port's bookkeeping, not a capability, and it is run and recorded
    as an observation, not as a cell.
- **Capstone domain, reference build** (`sublet-malloc`, mode 1): complete. Control 90, built with the
  same options, faults at the read probe, cause 24.
- **virtual-malloc**: MISSED, DEFECT-REPRODUCED. **virtual-nested-pools**: CAUGHT at
  `wm_widen_probe`, cause 24. Controls 90 and 91 behave as in R7b.

## Outcomes (written after each run)

- **R1** (`memcached/allocator-repros/results/2026-10-10-cheribsd-fixed-buggy/`): as predicted, 18 of 18 arms.
  - Case 08's buggy arm faults with SIGPROT, si_code 1, at `mc_case_body+0x1e8`. That is the declared
    site: the defect's own scan, before the probe. Its fixed arm completes.
  - Cases 00-07: fixed and buggy both complete.
  - The revocation control faulted at its labelled load.
  - memcached's nested spatial CHERI 1/4 stands, now with a control and an attributed site.
- **R2, first attempt**: no reading. My chain script passed only the subobject flag, not the buffer-pool
  sources that the 2026-10-09 record's command names, so all 9 cases were BUILD-FAILED (the controls ran
  and passed). It is re-run as R2b with the recorded arguments, under the same pre-registration, since
  this attempt read nothing.
- **R3a** (cheribsd-subobject, supervised): 8 of 9 as predicted.
  - Attributed as predicted: 00-03 in `ff2_case_run`; 05, 06 and 08 at `write_probe`; 04 completes.
  - Case 07 faulted at a pc outside the program image, in libc, a shared library here. The attribution
    tool knew only the program's symbols, so it read "outside every function": an instrument limit, not a
    refutation. It is fixed and re-run as R3c.
- **R3b** (`plane-repros/results/2026-10-10-cheribsd-carve-bounds-supervised/`): as predicted. The fixed
  arm completes; the buggy arm faults with SIGPROT si_code 1 AT `ffp_read_probe`.
- **R4** (`wmem-repros/results/2026-10-10-poisoncap-supervised-13-21/`): as predicted. All 18 arms
  (cases 13-21, modes 0 and 1) fault with SIGPROT, bounds, AT their case's own label:
  - `wm_defect_probe` for 13, 14, 15 and 19;
  - `wm_defect_write` for 16, 17, 18, 20 and 21.
  The runner now produces what the 2026-10-09 run needed a hand attribution for.
- **R5** (`pool-repros/results/2026-10-10-poolstock-control/`): as predicted.
  - App fixtures 4 and 5 FAULT temporal on the poolstock image, so its heap is the Sublet heap.
  - In the same boot the four cases complete: fixed FIXED, buggy DEFECT-REPRODUCED.
  - Column 2's four pool misses now stand on an in-boot control.
- **R2b**: no reading, my error again. It linked `sdk-level0` (runtime 3c72f36acd5b); the 2026-10-09 record
  names runtime afa624ec6387, which is `sdk-level0-cap`. That heap could not hold the corpus's payload, so
  every fixed and buggy run stopped at CONTROL-FAILED 604 (`driver.c`: `aligned_alloc` returned NULL) before
  any case code ran. The controls passed. What would have caught it: matching the SDK's runtime hash
  against the record before the run, not after.
- **R2c** (`subobject-repros/results/2026-10-10-capstone-subobject/`): as predicted, 9 of 9.
  - 00-03 CAUGHT in `ff2_case_run`, the declared site; 05 and 08 CAUGHT AT `write_probe`, which the
    runner now recognises as the labelled probe; 07 CAUGHT in `memcpy`, declared; 04 and 06 reproduce.
  - Every image is byte-identical to the 2026-10-09 run's, at the same offsets. So the stricter rule
    keeps all seven catches: the waiver was unsound, but nothing it admitted was wrong.
  - 07 stays the weaker form on Capstone, which records no return address; R3c names its caller.
- **R3c** (`subobject-repros/results/2026-10-10-cheribsd-subobject-supervised/`): as predicted, 9 of 9.
  - 00-03 in `ff2_case_run`; 05, 06 and 08 AT `write_probe`; 04 completes.
  - 07 in libc's `memcpy+0x80`, with return address `ff2_strlcpy+0x7e`, the case's own copy. The
    extended supervisor reports the mapped object and the return address, and the tool resolves both
    against the sysroot's libc.
  - The suite's exit 1 is its stale expectation that buggy arms complete; the attribution exit is 0.
- **R6** (`<corpus>/results/2026-10-10-virtual/<arm>/`, derived into case.json by `derive-verdicts.py`):
  as predicted, with no cell off its prediction.
  - plain-heap: 46 of 46 CAUGHT at the probe, cause 28. That includes the five cases CheriBSD's size
    classes absorb.
  - plain-temporal: 26 of 26 CAUGHT at the probe.
  - carved: 13 MISSED on `virtual-malloc`, and 13 CAUGHT at the probe on `virtual-nested-pools`.
  - Every control faulted and every fixed arm printed VERDICT FIXED with exit 0.
  - The subobject run built nothing, from the same chain omission as R2's first attempt; it is R6b.
- **R6b** (`subobject-repros/results/2026-10-10-virtual/virtual-malloc/`): as predicted, 9 of 9 MISSED.
  Each buggy arm printed DEFECT-REPRODUCED and completed; each fixed arm printed FIXED.
- **R7, first attempt**: no reading, on the hazard the pre-registration named. The virtual heap caps one
  allocation at 256 MiB (`runtime/virtual/vm.h`, `CAP_VM_MAX_BYTES`), so the 384 MiB payload failed:
  - every case and both wmem controls exited 75 in both columns;
  - column 3 printed `CONTROL-FAILED the virtual heap lent no 402653184-byte payload`;
  - column 2's driver returned 75 silently at `aligned_alloc`.

  The mallocng controls faulted as required. The fix: the capstone-application builds take a 128 MiB
  payload (`WM_VIRTUAL_PAYLOAD_MIB`, appended after `port.h`'s include guard so that no line moves). The
  physical images are again 185/185 byte-identical. R7b reruns it under the same pre-registration.
- **R7b** (`wmem-repros/results/2026-10-10-virtual/<arm>/`, derived): as predicted, 44 of 44 cells.
  - Controls, `virtual-malloc`: bounds-malloc, uaf-malloc and 90 fault; 91 completes. Controls,
    `virtual-nested-pools`: all four fault. So both columns showed, in their own boot, that they can
    read either way.
  - `virtual-malloc`: 22 MISSED. Every buggy differential printed DEFECT-REPRODUCED and exited 0,
    including the 13 temporal cases' reoccupation assertions.
  - `virtual-nested-pools`: 22 CAUGHT.
    - The 13 temporal cases are cause 24 in `wm_probe`.
    - The 9 spatial cases are cause 28 in `wm_probe` (13, 14, 15, 19) or `wm_write_probe` (16, 17, 18,
      20, 21). These are the labels R4 resolved on PoisonCap.
    - Every fault is at the probe's own access instruction: `wm_probe+0x14` is its `lbu`, and
      `wm_write_probe+0x18` its `sb`.
  - Every fixed arm printed VERDICT FIXED with exit 0.
  - The virtual columns agree with the physical ones cell for cell: sublet-malloc 0/22 and
    sublet-chunks 22/22.
- **R8, first attempt**: the runner wrote no bundle. Both boots ran and printed every case, but the
  runner then crashed: in `run_hosted` a loop variable named `argv` shadowed the helper of the same
  name. The unit tests cover the observers, not `run_hosted` end to end, so they could not see it. The
  helper is now `run_argv`. R8b reruns under the same pre-registration; the bundle records the runner's
  hash.
- **R8b** (`allocator-repros/results/2026-10-10-virtual/<arm>/`, derived): as predicted, 18 of 18 cells.
  - Controls: bounds-malloc and uaf-malloc fault on both arms. 90 completes on `virtual-malloc` and
    faults on `virtual-nested-pools`.
  - `virtual-malloc`: 00-07 MISSED, each with DEFECT-REPRODUCED. 08 is CAUGHT, cause 28, at
    `mc_case_body+436`, which is `lbu` at `case.c:66`: the defective `while (*ptr == ' ')` scan
    stepping past the rbuf object.
  - `virtual-nested-pools`: 9 CAUGHT.
    - 00-04: cause 24 at `read_probe+0x14`.
    - 05 and 06: cause 28 at `write_probe+0x18`.
    - 07: cause 28 at `read_probe`.
    - 08: cause 28 at the same scan load.
  - The prior-art branch's one-pc fault did not recur. That branch did not build the stock ledger.
  - Both columns agree with the physical ones: 1/9, and 9/9 (counting `sublet-carve` for 06 and 07).
