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
