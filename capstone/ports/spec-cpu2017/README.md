# Capstone SPEC CPU2017 kit

Everything needed to go from a SPEC CPU2017 1.0.2 archive to validated runs of its C benchmarks on
the Capstone virtual profile (128-bit capability pointers, RISC-V, QEMU), and to compare the profile's
arms, spatial-only against spatial-plus-temporal: the runcpu config, the toolchain and platform pins,
the scripts, and the records. SPEC's sources are licensed, so the SPEC-derived parts are not here: the
`src.alt` packages (the source changes, built by `tools/make-srcalt.sh`) live in a private directory
that `tools/build.sh` reads through `CAPSTONE_SPEC_PACKAGES`; `manifest.lock.json` pins them by hash
and `srcalt-notes/` says what each one changes and why. `artifacts/` (source archives, images, the
adapter) is local and unversioned; the manifest pins every file in it too. Results built this way are
research results and never reportable.

The measurement plan, with the pre-registered readings of its controls, is
`capstone/docs/plans/spec-spatial-vs-temporal.md`; the heap switches it measures are PR #279.

## What is pinned

`manifest.lock.json` names every input by hash. The scripts check it before they use anything.

| Input | Pin | Built by |
|---|---|---|
| SPEC CPU2017 1.0.2 | archive sha256; every file against SPEC's MANIFEST | `tools/spec-extract.py` |
| Source changes | 3 src.alt packages, sha256 each (`packages/`, explained in `srcalt/*/README`) | `tools/make-srcalt.sh` |
| Toolchain (clang, lld, musl, virtual SDK) | llvm-capstone commit + tree; source archive sha256; musl 1.2.5 sha256 | `recipes/toolchain.sh` |
| Its integration of open PRs | public base commit + PR head commits, merge order, tree | `recipes/integration.sh` |
| QEMU | capstone-qemu commit, archive sha256, configure options, binary sha256 | `recipes/qemu.sh` |
| Guest kernel, firmware, rootfs | sha256 (`artifacts/platform/images`); provenance in the manifest | not rebuilt (see Limits) |
| Adapter (kernel module, launcher) | sha256 | `recipes/adapter.sh` |
| runcpu config | `config/capstone-virtual.cfg`, used byte for byte | -- |

`artifacts/` (source archives, images, adapter, optionally a bundle of the PR heads) is not in git;
the manifest names every file in it by hash.

## Source changes

| Benchmark | Change | Why |
|---|---|---|
| 500.perlbench_r | 13 files (src.alt) | pointer provenance and alignment on 128-bit pointers, musl differences; `srcalt/500.perlbench_r/README` |
| 502.gcc_r | 9 files (src.alt) | 7 for 128-bit pointers (qsort swap unit, page-group, obstack, pool and bitmap alignment, splay-tree keys), 2 out-of-bounds reads in GCC 4.5 itself that later GCC releases fixed; `srcalt/502.gcc_r/README` |
| 505.mcf_r | 1 file (src.alt) | spec_qsort swaps `long` words and drops pointer tags |
| 500.perlbench_r test input | 4 of pack.t's checks stubbed, in the run directory only | each rebuilds a pointer from bytes, which carries no tag; `tools/perl-pack-stub.py`, applied and recorded by `tools/capstone-run.sh` |
| 525.x264_r input generator | ldecod runs on the host | it reads past its own table; it only produces x264's input. x264 runs on Capstone |
| all | `-DSPEC_LP64`, per-benchmark portability flags | `config/capstone-virtual.cfg` says why for each |

The installed SPEC sources and inputs are never changed; `tools/build.sh` checks the tree against
SPEC's MANIFEST after the build.

## From the archive to results

```bash
K=<llvm-capstone>/capstone/ports/spec-cpu2017 W=<work directory>
export CAPSTONE_SPEC_PACKAGES=<private kit with packages/>
bash $K/recipes/integration.sh $K <llvm-capstone git dir>  # recompute the pinned integration tree from the PR heads
bash $K/recipes/toolchain.sh $K $W/toolchain             # clang, SDK (checks the source archive)
bash $K/recipes/qemu.sh $K $W/qemu
bash $K/recipes/platform-dir.sh $K $W/qemu $W/platform  # checks every platform file
python3 $K/tools/spec-extract.py SPEC17.tar.gz $W/q/tree ALL
bash $K/tools/build.sh $K $W/q/tree/SPEC17 $W/toolchain/sdk $W/q/build capstone-virtual
bash $K/tools/run.sh $K $W/q/tree/SPEC17 $W/toolchain $W/platform $W/q/build test 505.mcf_r $W/q/test/505.mcf_r
```

### The arms: spatial-only against spatial-plus-temporal

`manifest.lock.json` `runtime` names a second llvm-capstone commit: the toolchain commit plus PR #279's
heap switches (`CAPSTONE_HEAP_LIFETIMES`, `CAPSTONE_HEAP_SCRUB_FREE`), cherry-picked, which leaves the
compiler, lld, compiler-rt and the musl port unchanged (`unchanged_subtrees`, checked by
`recipes/integration.sh`). `recipes/sdks.sh` builds one SDK per arm (`arms`) from that tree against the
pinned compiler and libc archive and proves they differ in `heap.c.obj` only. `tools/build.sh` builds
every arm in one tree under its own label and checks that every benchmark object is byte-identical across
labels; `tools/run.sh` takes the label and records SPEC's reported seconds, the launcher's counters
(`CAPSTONE_VM_STATS`, from the run directory's `.err` files) and the guest QEMU's CPU time;
`tools/compare.sh` runs the arms of each benchmark together, repeated with rotated CPU assignment, and
`tools/compare-report.py` tabulates them with the host-load flags. The plan with the pre-registered
readings is llvm-capstone's `capstone/docs/plans/spec-spatial-vs-temporal.md`.

```bash
bash $K/recipes/sdks.sh $K $W/toolchain $W/sdks
python3 $K/tools/spec-extract.py SPEC17.tar.gz $W/c/tree ALL
bash $K/tools/build.sh $K $W/c/tree/SPEC17 $W/sdks $W/c/build          # all three arms, integrity gate
bash $K/tools/compare.sh $K $W/c/tree/SPEC17 $W/toolchain $W/platform $W/c/build train $W/c/train 3 40,41,42,...
```

`run.sh` boots a fresh guest for one benchmark at one size, runs `runcpu --action=validate`, and
writes `result.json`. runcpu generates every command and validates every output with SPEC's own
tools (specdiff with each benchmark's tolerances, imagevalidate for x264 and imagick).
`tools/capstone-run.sh` is runcpu's `submit` wrapper: Capstone images run in the guest, SPEC's
tools and the host ldecod on the host, and anything else is refused. Each invocation leaves a
record (route, argv, cwd, executable sha256, guest VM identity, guest exit or fault).
`tools/verdict.py` passes a run only if runcpu succeeded, nothing was refused, the benchmark's
own executable ran in the guest, and every guest invocation exited 0.

`recipes/integration.sh` recomputes the toolchain tree from public commits and the bundle.

## Qualification

Of the previous toolchain (manifest `toolchain_previous`, 193d4495): run q4 on the build host, 2026-10-17
10:05-17:33 UTC (`tools/qualify.sh`, private kit commit ddf3d0b,
platform as pinned, guest memory 8 GiB, share model mapped-xattr): **QUALIFIED**. A fresh
extraction of the archive, one build, both negative controls, then all 20 runs in parallel,
each in its own guest on its own host core. Records in `qualification/` (`qualification.json`
holds every run's `result.json`; home-directory paths rewritten to `~`, originals in p12 `~/spec-cap/kit2/q4`).

| Benchmark | test | train | executable sha256 |
|---|---|---|---|
| 500.perlbench_r | PASS (203 min) | PASS (307 min) | `abc7fe2ed9b1` |
| 502.gcc_r | PASS (1 min) | PASS (440 min) | `61ddd59ce371` |
| 505.mcf_r | PASS (53 min) | PASS (189 min) | `6f887e52a6e3` |
| 519.lbm_r | PASS (6 min) | PASS (72 min) | `481a0491e71e` |
| 525.x264_r | PASS (219 min) | PASS (200 min) | `ea31d45e77a3` |
| 538.imagick_r | PASS (1 min) | PASS (180 min) | `9e7f0ed93ee3` |
| 544.nab_r | PASS (9 min) | PASS (304 min) | `c848ea9cc291` |
| 557.xz_r | PASS (44 min) | PASS (119 min) | `9f2b387cd3c9` |
| 997.specrand_fr | PASS (0 min) | PASS (0 min) | `717c60b9ec0e` |
| 999.specrand_ir | PASS (0 min) | PASS (0 min) | `717c60b9ec0e` |

Negative controls: a wrapper that appends one line to every guest output makes runcpu reject
997.specrand_fr (verdict FAIL); the host's `/bin/true` is refused (exit 125, route refused).

The executables are byte-identical to those of run q3 (fresh tree, same kit sources, toolchain
built in another directory from compiler-identical sources). Minutes are wall time under QEMU
and are not performance figures: for the first ~3 hours other load on p12 slowed the guests
about 2.3x (mcf test took 53 min here, 23 min in q3).

## Limits

- The guest kernel Image has artifact-only provenance: its source tree had no git metadata. The
  firmware and rootfs come from a Buildroot build whose commit is recorded.
- The guest sees the run directories through 9p with `security_model=mapped-xattr` (capstone_vm
  `--share-model`, PR #252): with the default model perl's `stat.t` test 33 (`-o`, file owner)
  fails because the guest user does not own the host's files.
- clang's sha256 depends on the toolchain WORKDIR (the assertions build embeds source paths);
  the manifest says so. The libc, SDK file set and SPEC executables do not.
- The QEMU binary is dynamically linked; its hash pins the build host's build.
- capstone-run.sh and run.sh supersede the config and runner of the never-published local branch
  `ports/spec-cpu2017`.
- The toolchain of the arms comparison (manifest `toolchain`, 954b77c1: dev of 2026-10-20 plus 23 PRs) has
  not been through `tools/qualify.sh`; every comparison run is validated by runcpu, and `results/` records
  what ran. The quick test of 2026-10-20 is in `results/2026-10-20-quick-test/`.
