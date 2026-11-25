# B5b-2 SPEC test-size images: built on the trap route, 0 illegal encodings by B5b-2's decoder (2026-11-25)

The runtime lane asked for SPEC test-size images for the first B5b-2 Linux kit (bitstream 904cd9c6b, the trap route).
They were to be built with the `-capstone-image-gp` compiler, with a DELIN count by encoding for each image. The B5b-2
plan (`docs/plans/2026-11-24-b5b2-trap-route.md`, the integer-M monitor section) records why: a userland built
without that flag executed DELIN at 876 sites, and DELIN is illegal under the Sublet format. Train inputs are not
built; the lead's rule keeps train for the final campaign.

**Status (amended the same day): one benchmark has executed on QEMU, the rest only on the boards to come.** See "QEMU smoke" below.

## The scan

`capstone/tests/b5b2-illegal-scan/scan_illegal.py` counts raw encodings. It sweeps every executable section
linearly, aware of 16-bit instructions, and uses no disassembler. Each word is classified by capstone-ariane
904cd9c6b `core/decoder.sv` under `CAPSTONE_SUBLET_FMT`, for code running in protected U:
- **custom-2, illegal:** REVOKE, TIGHTEN, DELIN, SPLIT, SEAL, MREV, INIT, DROP, CAPENTER, CALL, RETURN, CSSUPERVISE,
  CSRUNV, CSRETIRE, CSMINT (illegal in U), every unassigned funct7, CJALR and CCSRRW;
- **custom-2, conditional:** CREVOKE is legal only with rd = rs2 = x0, and CRENEW only with rs2 = x0;
- **other illegal:** custom-0, custom-1 and custom-3, and csr ops on 0x7c3, 0x7c4 and 0xfc0-0xfc4;
- **legal:** SHRINKTO, SHRINK, LCC, SCC, MOVC, CINCOFFSET(IMM), CDERIVE, CBNZ, and LDC/STC (protected U only).

It exits 2 on an image with no executable bytes (no data is an error), 1 if any encoding is illegal, else 0.

Controls, made by `make-controls.sh` (the 1s are the control firing):

| control | reading | exit |
|---|---|---|
| ctl.o: 24 planted illegal words (6 of them 2 bytes off, after a c.nop) and 14 legal | ILLEGAL=24, every legal word legal | 1 |
| empty.o: no code | NO-DATA | 2 |
| a global access built by clang 29027126f39b without `-capstone-image-gp` | DELIN=2 | 1 |
| the same with `-capstone-image-gp` | DELIN=0 | 0 |

On linked images the scanner also fires on the measured Full arm FRCDKMSa3. Each of its 11 images carries two to
six CREVOKEs with rs2 nonzero (the two-operand form), which B5b-2 makes illegal; the scan finds no DELIN in them. So
the paper's Full images cannot run on this bitstream.

## The build

- **Runtime:** llvm-capstone origin/trap/eval ed8cc375c2b1, whose `runtime/virtual/build-sdk.sh` builds the trap
  route's SDK:
  - application_abi 3 and `CAPSTONE_RUNTIME_TRAP=1`;
  - ports/musl-capstone `linux-start.c`, `native-syscall.S` and `delegate.c`, and musl patch 0011 (linux trap
    startup);
  - no `.capstone_virtual` launcher handshake.
- **SDK options:** those of the measured Full SDK (perf/alean sdkm13b): heap lifetimes and object bounds on, scrub
  off, native I/O, fault recovery off, and the same arena, contexts, data, exchange, grant and stack sizes.
  `MUSL_CAPSTONE_EXTRA_CFLAGS=-fomit-frame-pointer`.
- **Not in the SDK:** `HEAP_MAPPING_REUSE` and `MEMSET_CAP_ZERO`, which trap/eval does not have, and alloc-lean's
  mallocng patches 0011-0013.
- **Compiler:** clang 29027126f39b (llvm-c9), the measured Full arm's compiler, for both the SDK and the benchmarks.
- **Benchmarks:** the Full arm's compile flags. `-capstone-shrinkto-globals`, `-capstone-shrinkto-end-adjust=1`,
  `-capstone-direct-calls`, `-fomit-frame-pointer`, `-capstone-drop-redundant-shrink(-declarations)`,
  `-fjump-tables`, `-capstone-fold-global-lo`, `-capstone-disable-cost-per-use` and `-capstone-memcpy-align-dispatch`.
  The SDK codegen adds `+relax +c +cap-base-rvc` (`+cap-base-rvc` uses standard C encodings and is not an ISA change).
  The compiles replay the recorded `make.out` lines, and images are stripped as the Full arm's are.
- **Difference from the Full arm's sdk.json:** application_abi 3 against 2, and `-DCAPSTONE_RUNTIME_TRAP=1` in
  codegen.

Rows measured with these images are not comparable to the paper's Full rows, because the allocator and runtime
differ. mcf's text, for example, is 0x159fc against 0x23e60.

The section and segment layout matches trap/eval's own `build-trap-smoke.sh` output built with the same libc:
- the same sections;
- entry 0x10000;
- an executable LOAD at 0x1000, a read-write LOAD and a TLS segment.

The `.capstone.size_assume` section, written by the SDK's compiler wrapper (its post-link size-assumption check),
is removed by the strip. The smoke images contain no 16-bit instructions, and these do.

## Result lines

The SDK's archives scan clean: runtime 19 members, builtins 55 and libc 1,349, all ILLEGAL=0. 32 members have no
code and read NO-DATA.

| image | sha256 (stripped .dom) | words32 | ILLEGAL | DELIN |
|---|---|---|---|---|
| perlbench_r | 38ce666fe711d65a5a5f7fe1809bbecb03b16d8a09633f52f3cae3deb2585bf6 | 391,784 | 0 | 0 |
| cpugcc_r | 10800cd3ddfd0db35b641d4a5190d7a08096b44997ae0466c11af6ba85f5ff75 | 1,749,466 | 0 | 0 |
| mcf_r | 75bbd17e327644361b49c3c3fd9725e79b694f1d7e5293534172ca75bb9bace2 | 21,445 | 0 | 0 |
| lbm_r | 906d7c735fa2988f04c7523bb1891e2475a88d63baff50785e4006d292ef0e41 | 19,974 | 0 | 0 |
| x264_r | 95d93fae37aa287183a4c4f6872606bba432b0cdbc38a04d5f208b88c98ef9ad | 113,884 | 0 | 0 |
| ldecod_r | 33ea45308caeb588020a5235cd387737c5cfb970cb44351ff47b4166e4f2b3a6 | 110,392 | 0 | 0 |
| imagick_r | 0aa16141bd09027220dfe858fd032a3b08e3b937009a25f8ac61178824352c5e | 382,759 | 0 | 0 |
| imagevalidate_525 = _538 | 1d33c357fdf46e7479d17fd64b953aae48efd5bf68115137ea00c8b1be203fb8 | 20,120 | 0 | 0 |
| nab_r | 82f8fd370ce6a6b550d5a9f44377d4ca7248af9e56ec955075a30b5c301186b0 | 58,684 | 0 | 0 |
| xz_r | 2e4bd52cf3d29bd056bb662eccf486f0dda650a410b68a3955dac4c5caa23e57 | 33,004 | 0 | 0 |

The unstripped images read the same counts. The capability ops present in each are CDERIVE (4; 2 in
imagevalidate), the one-operand CREVOKE (3), LCC, SCC, SHRINK, SHRINKTO, MOVC, CINCOFFSET(IMM), LDC and STC.

## The lean arms (added the same day)

The runtime lane asked for the measured allocator on the trap route, as a second build kept beside B5T, so that rows
can be compared with the Full arm. Lane runtime/trap-alloc-lean carries it, on trap/eval:
- alloc-lean's mallocng patches, renumbered 0012-0014 after 0011-linux-trap-startup. The grow-in-place patch is
  ported to the trap route's realloc order: the new object is derived first, and can fail with ENOMEM.
- memset-cap-zero.
- The whole-mapping cache (HEAP_MAPPING_REUSE), ported as owner' = CRENEW(owner). Its QEMU qualification is
  tests/trap-mapping-reuse/RESULTS.md: every pre-registered row as predicted. Residuals: the mprotect hook never
  fired, because the trap kernel refuses mprotect through an object reference; the out-of-place renewal is
  unobserved; the alias row's same_address read 0.

Arms built with the same compiler and flags:
- B5TL0 (no mapping reuse, ba87c530a8f6);
- B5TL1 (with it, afeace3b2e0d).
All 22 images of the two arms read ILLEGAL 0 / DELIN 0.

## QEMU smoke (added the same day)

The platform: capstone-qemu da49602d76, the trap-route-keep-tags firmware, the runtime lane's trap kernel
(`sublet.dom_binfmt=1`) and 963 MiB, the board's DTS memory. Each arm's 519.lbm_r test stage ran unchanged from a
tmpfs cwd:

| arm | image | rc | specdiff |
|---|---|---|---|
| B5T | 906d7c735fa2988f | 0 | 1/1 PASS |
| B5TL0 | 5dedb373ee975e94 | 0 | 1/1 PASS |
| B5TL1 | 0d9105dab2bccdfe | 0 | 1/1 PASS |

- The control, trap/eval's smoke.dom, exited 7, as its source returns.
- The grader's control: the same text with lbm.out corrupted reads 0/1.
- What this shows: the startup ABI and the loader path work for each SDK. The other seven benchmarks run only on the
  B5b-2 boards (the runtime lane's kit), not on QEMU, to keep the shared QEMU lock free.

The first boot attempt was void because of a harness defect: the initramfs was repacked as a normal user, so the
setuid busybox ran as that user.

## Open

- **Seven of eight not run.** perlbench, gcc, mcf, x264, imagick, nab and xz run on the boards.
- **Scope of the scan:** it reads encodings, so it says nothing about an instruction's run-time legality beyond
  decode, and assumes protected U.
