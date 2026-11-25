# Line B's Plain arms: SPEC test images paired with B5TL1 (2026-11-25)

Two Plain arms, built for the Line B campaign (docs/plans/2026-11-25-line-b-final-campaign.md, "the matching rules").
- **Plain-matched** is the campaign's Plain.
- **Plain-paper** is the measured Plain recipe, unchanged, as the bridge to the paper's rows.

The lead's rule (PREREG 2026-11-08-train): every generic compiler option is the same in the Capstone build and in
riscv-base; Capstone-only options are exempt, and ISA/ABI limits are disclosed.

## Images and pairing

Each Plain stage is a copy of B5TL1's board stage with only bin/native/<bench>_base.native replaced. run.sh, inputs,
plan.json and the plain helpers (ldecod, natperl, imagevalidate_538) are byte-identical to the B5TL1 stage: the
staging script asserts it file by file (558 files for perlbench, 3 to 6 for the rest). MANIFEST.json lists every
image's sha256, its source and its B5TL1 partner's sha256.

| benchmark | B5TL1 | Plain-matched | Plain-paper |
|---|---|---|---|
| 500.perlbench_r | f3c8c80d71844564 | b91c1c2fbfe816b1 | 779a8688f58c649b |
| 502.gcc_r | f800e015d161c911 | d11c11ae6c4a2727 | 6385edb2ab2b20fe |
| 505.mcf_r | a1daec395a9e7260 | d1dafbfa6715e115 | 2e1b522831e9020c |
| 519.lbm_r | 0d9105dab2bccdfe | e20167555db0c00e | bd72cb22753b5c60 |
| 525.x264_r | f686fe9679b04d69 | fe7092b29993f6e5 | 5adf7cac02e0a8ed |
| 538.imagick_r | a666e331f10d73cc | b461c489b84ebfbc | f7f3e78a5a494659 |
| 544.nab_r | da57e5473a33e6fd | 1d8f590910692234 | d0c47f5c4a14b28a |
| 557.xz_r | ecb70755f9ee1690 | d6df063ae6ba293a | b889d0f57fcd13b3 |

## Plain-matched: what is equal, and why the rest differs

| | B5TL1 (Full) | Plain-matched | status |
|---|---|---|---|
| compiler | llvm-c9 clang 29027126f39b | the same binary (sha256 ecd6d8c49efb...), --target=riscv64-unknown-linux-musl | equal |
| application optimisation | -O2 | -O2 (the measured Plain config's OPTIMIZE, PORTABILITY and src.alt) | equal |
| application ISA | capstone64 +m +a +f +d +c, +relax, +cap-base-rvc | rv64gc lp64d (C included), relax (the riscv default) | C equal; +cap-base-rvc Capstone-only |
| -fomit-frame-pointer, -fjump-tables | on | on | equal |
| relocation model | static (capstone-cc) | -fno-pie, applications and libc (the riscv64-linux-musl default is PIE: `-mrelocation-model pic -pic-is-pie`) | equal |
| the other FRCDKMSa3 -mllvm flags (shrinkto-globals, direct-calls, fold-global-lo, ...) and the C5 redundant-bounds pair | on | n/a | Capstone-only |
| libc | musl 1.2.5 + musl-capstone 0001-0014 | musl 1.2.5 (sha256 a9a118bb...) + 0001-0003 + 0014's always-inline get_meta + the fairness rebuild's round-evict mapping cache (libc.a 25c745633fa2962a) | see below |
| libc optimisation | -O1 -fomit-frame-pointer | -O1 -fomit-frame-pointer -fno-pie | equal |
| libc ISA | capstone64 +m +a, no C | rv64imafd lp64d, no C | equal: a BUILD OPTION, not a limit |
| support library | the SDK's capability builtins | buildroot gcc 12.3 libgcc.a (sha256 4dd50737...), the measured Plain recipe's | differs: a capability build cannot serve plain RISC-V, disclosed |
| mapping cache | HEAP_MAPPING_REUSE: 512 entries, 64 MiB, heap_round lengths, mr_make_room eviction (3e70344d8c3f) | perf/plainevict's round-evict cache: 512, 64 MiB, page-rounded lengths, the same eviction rule | equal, by the fairness precedent (below) |
| runtime | the trap runtime (heap ownership, memset-cap-zero) | none (musl's own startup) | Capstone-only |

**Why B5TL1's libc has no C.** It is a build option. The SDK compiles musl with `-target capstone64-unknown-elf`
and target features +m and +a (survey-musl-capstone.py, compile_flags); +c is never requested. The same compiler
compiles B5TL1's applications with +c +cap-base-rvc, so this target can use RVC in capability code. The archive has
0 16-bit instructions (counted in malloc, memcpy and fwrite of libc-capstone-virtual.a), and Plain-matched's libc
is built without C to equal it.

**The musl patches, one decision each:**
- 0001 (BUFSIZ 8 KiB), 0002 (dirent buffer 32 KiB), 0003 (no fcntl after an O_CLOEXEC open): generic, applied. The
  measured Plain musl (perf/T0b) has them too.
- 0014 (get_meta always inline): generic, ported. Its read-once of sizeclass and maplen is a capability-codegen detail
  and is not ported.
- 0013 (one-call malloc): it undoes 0009's split of malloc into __malloc_alloc plus heap_alloc_issue. Stock musl's
  malloc is already one call, so there is nothing to port.
- 0012 (grow mmapped chunks in place): NOT ported. Its own text says it exists because "the capability heap has no
  mremap", and that riscv-base's musl moves the chunks with mremap "for nothing". Stock musl's realloc is the match;
  porting it would add copies Plain does not pay.
- 0010 (meta area at once): `#ifdef CAPSTONE_MUSL_MALLOC`, a no-op on plain, which has brk.
- 0004-0009 and 0011: capability layouts and the trap-route startup. Capstone-only.
- The mapping cache: free.c, glue.h, malloc.c and plain_reuse.c from perf/plainevict's source, the one built into
  musl-rv64gc-reuse-round-evict (plain_reuse.c rounds a length to 4096 and carries mr_make_room's eviction rule).

## Correction (2026-11-25): the first Plain-matched build is withdrawn

The first build (lane c8d1b1038305: images d963f725..., board images g1-g4 60efd53f.../ffb7d365.../572d3f19.../
232d5925..., then 9909baba.../67df9178.../c00100d1.../44fe8af6... with the 3,600 s bound) is WITHDRAWN. It had two
divergences from the matching precedent, both found in prior art the first build missed (perf/alean/plain,
perf/plainevict):
- **It was PIE.** The fairness wrapper (perf/plainevict/plain-cc) passes -fno-pie "as the Capstone arm's static
  relocation model". llvm-c9 defaults to PIE for riscv64-unknown-linux-musl. Control: an extern-global load compiles
  to 1 GOT relocation by default and 0 with -fno-pie.
- **It had no mapping cache.** The train PREREG's fairness rulings (spec-silicon, Addenda 67 and 82-84, as ruled by
  the semantics owner): Plain carries a mapping cache when the Capstone arm does, in the round-evict variant.
  - Addendum 82: the un-rounded plain cache never hits.
  - Addendum 83: the rounding fix plus the eviction rule, matching heap_round and mr_make_room.
  - Large-realloc growth stays matched as Addendum 82 states: B5TL1's grow-in-place against Plain's mremap.
The images and stages above are the rebuild; the withdrawn ones are kept locally as *-pie-nocache.

## Plain-paper

spec/tree-plain's plain-rv64 images, unchanged: spec-silicon's ports/spec-cpu2017/plain/build-plain.sh and
build-plain-gcc.sh.
- The compiler is cc-prmerged clang fca004e7e063 (sha256 b90e13006a01...), and the libc is perf/T0b's musl-rv64gc
  (libc.a 2d16947e1ca7b8fc, rv64gc, -O1, 0001-0003).
- The SPEC options are -O2, with the same config.
- Every image was re-hashed against its build log, and all are unchanged.
- Generic options that differ from B5TL1: the compiler revision, no explicit -fomit-frame-pointer/-fjump-tables, PIE
  code (its plain-cc passes no -fno-pie), and no mapping cache.
  So it is the bridge to the paper, not the campaign's Plain.

## Board images (rvbase bitstream 8db8f9cda only)

Built by trapk/plain/make-payload-plain.sh: the board's rootfs.cpio unchanged plus an overlay, the Plain kernel (Linux
6.1, NO_HZ_IDLE at HZ=100, no Capstone series), stock OpenSBI 1.3.1, and caplifive-plain.dtb. Each is under the 26 MB cap.
- The overlay is board-init (here) and plain-smoke (plain-smoke.c here, built with plain-cc-matched: one small and
  one mmapped allocation, "ok", exit 7), plus the Plain-matched stages.
- board-init reads the kernel log by offset (no dmesg -c) and bounds each stage at 3,600 s (STAGE_BOUND).

| image | stages | fw_payload.bin sha256 | bytes |
|---|---|---|---|
| g1 | lbm, nab, xz, mcf | 50c135eb29a9948e | 19,564,552 |
| g2 | x264, imagick | f716db4ae042ceda | 21,661,704 |
| g3 | gcc | 9a16de8eebde78ad | 21,661,704 |
| g4 | perlbench | 32ca390b88110fe0 | 21,661,704 |

## Plain-twin smoke (2026-11-25, before any board session)

A functional smoke of the Plain-matched images on the QEMU twin, one boot, 22:28 to 22:39.
- **The kernel is the trap-twin kernel, not the Plain kernel the boards run.** That is trapk linux-6.1, Image
  749fed2606e3ebca (built 21:35, before the boot), with sbi-keep-run fw-y, -m 963M and sublet.dom_binfmt=1. It is
  enough for a functional smoke of static plain processes, which use none of its Capstone series. It says nothing
  about the Plain kernel or the Plain firmware, which only a board session runs.
- **What ran:** the board's board-init (84aa5d1c4c91ef97, behind a wrapper setting STAGE_BOUND=1500), then
  plain-smoke (42e1f6d1b289fa93), then six Plain-matched stages: lbm, imagick, mcf, gcc, xz, nab. The images are the
  table's hashes, checked against the initramfs. board-init and plain-smoke are the four board images' /init and
  /plain-smoke, by hash. x264 and perlbench were not in this boot.

Results:
- plain-smoke printed "ok" and exited 7, its pass value.
- Every stage ran every command with rc 0, and the guest reached "done".
- run-stage.py read, on each stage's text cut from the boot log, gave specdiff PASS everywhere: lbm 1/1, imagick 1/1,
  mcf 2/2, gcc 1/1, xz 12/12, nab 1/1.
- mcf and gcc keep plan.json beside the stage, not inside it, as their B5TL1 stages do, so the copied stage has none.
  They were read with B5TL1's plan.json, whose commands are the same; their run.sh is byte-identical to B5TL1's apart
  from the image hashes.
- **The grader's control:** mcf's inp.out with one number changed in the decoded text (1998 to 1999) read specdiff
  1/2, FAIL, exit 1.
