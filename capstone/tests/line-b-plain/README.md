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
| 500.perlbench_r | f3c8c80d71844564 | d963f7251e5db89e | 779a8688f58c649b |
| 502.gcc_r | f800e015d161c911 | 2863bebea5da1826 | 6385edb2ab2b20fe |
| 505.mcf_r | a1daec395a9e7260 | 8367f8f2490bf2bf | 2e1b522831e9020c |
| 519.lbm_r | 0d9105dab2bccdfe | a96c5f98765390cf | bd72cb22753b5c60 |
| 525.x264_r | f686fe9679b04d69 | 094fcadb698de376 | 5adf7cac02e0a8ed |
| 538.imagick_r | a666e331f10d73cc | 5e12cf9d63dfac6a | f7f3e78a5a494659 |
| 544.nab_r | da57e5473a33e6fd | 5c3e40db600da518 | d0c47f5c4a14b28a |
| 557.xz_r | ecb70755f9ee1690 | 8e3a31f855b6d0e3 | b889d0f57fcd13b3 |

## Plain-matched: what is equal, and why the rest differs

| | B5TL1 (Full) | Plain-matched | status |
|---|---|---|---|
| compiler | llvm-c9 clang 29027126f39b | the same binary (sha256 ecd6d8c49efb...), --target=riscv64-unknown-linux-musl | equal |
| application optimisation | -O2 | -O2 (the measured Plain config's OPTIMIZE, PORTABILITY and src.alt) | equal |
| application ISA | capstone64 +m +a +f +d +c, +relax, +cap-base-rvc | rv64gc lp64d (C included), relax (the riscv default) | C equal; +cap-base-rvc Capstone-only |
| -fomit-frame-pointer, -fjump-tables | on | on | equal |
| the other FRCDKMSa3 -mllvm flags (shrinkto-globals, direct-calls, fold-global-lo, ...) and the C5 redundant-bounds pair | on | n/a | Capstone-only |
| libc | musl 1.2.5 + musl-capstone 0001-0014 | musl 1.2.5 (sha256 a9a118bb...) + 0001-0003 + 0014's always-inline get_meta (libc.a fca13d2d0ff0db47) | see below |
| libc optimisation | -O1 -fomit-frame-pointer | -O1 -fomit-frame-pointer | equal |
| libc ISA | capstone64 +m +a, no C | rv64imafd lp64d, no C | equal: a BUILD OPTION, not a limit |
| support library | the SDK's capability builtins | buildroot gcc 12.3 libgcc.a (sha256 4dd50737...), the measured Plain recipe's | differs: a capability build cannot serve plain RISC-V, disclosed |
| runtime | the trap runtime (heap ownership, mapping-reuse cache, memset-cap-zero) | none (musl's own startup) | Capstone-only |

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

## Plain-paper

spec/tree-plain's plain-rv64 images, unchanged: spec-silicon's ports/spec-cpu2017/plain/build-plain.sh and
build-plain-gcc.sh.
- The compiler is cc-prmerged clang fca004e7e063 (sha256 b90e13006a01...), and the libc is perf/T0b's musl-rv64gc
  (libc.a 2d16947e1ca7b8fc, rv64gc, -O1, 0001-0003).
- The SPEC options are -O2, with the same config.
- Every image was re-hashed against its build log, and all are unchanged.
- Generic options that differ from B5TL1: the compiler revision, and no explicit -fomit-frame-pointer/-fjump-tables.
  So it is the bridge to the paper, not the campaign's Plain.

## Recorded at build time

None of these images had run when this record was written. A functional smoke on the Plain twin (the Plain kernel,
trapk/plain) comes before any board session.
