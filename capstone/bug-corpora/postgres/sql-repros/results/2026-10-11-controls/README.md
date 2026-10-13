# PostgreSQL 17.5: parent, sibling and successor controls, 2026-10-11

The PostgreSQL row of `capstone/docs/plans/2026-10-11-nested-lifetime-controls.md` (committed
53325bba0d5c before anything was built). Nested allocator: memory-context chunks (`palloc`/`pfree`)
of a dedicated **AllocSet**. Protected = `PGSU_NESTED=sublet`, memory-contexts patch 0003, as in the
security table; release = the same source without it. The controls and how they run:
[`../../controls/README.md`](../../controls/README.md). `results.json` holds every session's
result lines, addresses, accesses and fault.

**Outcome: every prediction holds, on both arms, in 3 of 3 repetitions.** The positive control
also holds. The stale read faults with cause 25 at the probe on protected. On release it
completes and returns C's data.

## Per control (repetition 1; repetitions 2 and 3 give the same outcomes, faults and offsets)

Probes are in `capstone/bug-corpora/postgres/sql-repros/controls/module/pgcorpus_lifetime.c`.

| control | arm | outcome | addresses | accesses (probe line) | checks |
|---|---|---|---|---|---|
| parent | protected | completed | ctx 0x3fb8a65bb0, block 0x3fb8a65d10, A 0x3fb8a65e00 (block+240), D 0x3fb8a65da0; header 32 B | read link :126, read header :132, write :139, read :148 | link = D's chunk 0x3fb8a65d80; header 0x34000000043 unchanged; 48 B read back; next palloc = A |
| parent | release | completed | ctx 0x3f8a76cb30, block 0x3f8a76cc90, A 0x3f8a76cd40 (block+176), D 0x3f8a76ccf0; header 16 B | same | same four checks pass |
| sibling | protected | completed | A 0x3fb7feada0, B 0x3fb7feae00, block 0x3fb7fead10 | read :158, write :166, read :158 | B intact after pfree(A); read-back passes |
| sibling | release | completed | A 0x3fb5994cf0, B 0x3fb5994d40, block 0x3fb5994c90 | same | same |
| successor | protected | completed | A = C = 0x3f9d574da0 (first palloc) | write :173, read :182 | read-back passes |
| successor | release | completed | A = C = 0x3f8f80ccf0 (first palloc) | same | same |
| successor-stale | protected | **fault cause 25**, pc 0x3f99044e7c = `lc_probe_stale_read`+20 | A = C = 0x3f9aa21da0; fault address 0x3f9aa21da0 | write :173, read :182, stale read :189 | — |
| successor-stale | release | completed | A = C = 0x3fba93dcf0 | write :173, read :182, stale read :189 | stale read 0xc2c3c0c1c6c7c4c5 = C's data |

`lc_probe_stale_read`+20 is link address 0x54e7c, `ld a0, 0x0(a0)`: the load through A, the same
in both images. Faults in repetitions 2 and 3 are at pc 0x3fa9044e7c and 0x3fa3044e7c. That is the
same offset, with a different load base.

On protected the chunk bounds show the patch at work. A live is [A-32, A+64), so header plus
request. C is a new child over the same range. The block reference spans the whole context
allocation ([ctx, ctx+8192)). On release, A's and C's bounds are the whole context allocation.

## Image identity checks (run in the same boot, before the controls)

| arm | trivial SELECT | `corpus_control('uaf-palloc')` |
|---|---|---|
| protected | answered | fault, cause 25, in `corpus_control_read` (expected) |
| release | answered | complete (expected) |

The verdict check can fire, which was tested by applying each arm's expectations to the other
arm's `successor-stale` records. Both come back DIFF: on release, "expected a fault at
lc_probe_stale_read, outcome completed"; on protected, "expected completion, outcome fault
(cause 25 in lc_probe_stale_read)".

## What ran

- **Source.** llvm-capstone HEAD 53325bba0d5c, **plus this session's uncommitted
  `sql-repros/controls/`**. Nothing else was taken from the working tree. The snapshot is
  `git archive` of 53325bba `capstone/` and `compiler-rt/lib/builtins`, committed in a scratch
  repository on p12 as 347afd2de162 so `build.py` can rev-parse it. The controls were copied in
  untracked, so both manifests record `runtime_dirty: true`. File hashes of the controls are in
  `results.json`. `controls/README.md` was written after the run; it is not an input.
- **Builds** (p12, JOBS=12 each, run concurrently). They used `controls/build-image.sh`, which
  calls the copied `controls/build-domain.sh`; the README says what that copy changes. The kit
  toolchain was clang 4c6135fe…, with the pinned 17.5 archive fcb7ab38…. The protected build
  applied patch 0003 (8b36b576…). Both images link 7 modules with 502 functions: the 6 of the
  security-table images plus `pgcorpus_lifetime`. The runtime archive is e5693f5c… and the
  builtins archive is a72ed0bf…, the same in both images.

| | release | protected |
|---|---|---|
| image sha256 | a4329a433eb6130e8304dcaf919162dd2ed189d9c5e8d6bc59732b3080386b6a | 3197202a6d4798b262255a883a9a5a9396abce7c7589c94c2f6f0c029872e173 |
| manifest `nested` | none | postgres |
| build wall time | 183 s | 192 s |
| one boot, 14 sessions | 673 s | 632 s |

- **Platform.** The kit `kit-sublet`: QEMU af37cc32…, launcher 3975314a…, module a1c6cb6b…,
  Image e833dfb1…, fw_jump 44cd6cbc…, rootfs 62049800…. Machine: `virt-capstone`, one hart, 2 GiB,
  U-mode, exact bounds (`run-staged.py` d6770b8b… defaults). Each arm used its own QEMU lock file.
- **Inputs.** The kit fixture cluster, tree 0353a5b9… (the same tree the same-answers run and the
  corpus runs used). The share is `fixture/install/share/postgresql` (tree a4bb87d5…) plus the
  image's own staged extension files. Each session ran on a fresh `cp -a` of the cluster as
  `nobody`, with the `--single` arguments of `../../shared/run-arm.py`.
- **N.** 3 repetitions per control per arm, each in its own backend session, in one boot per arm.

## Deviations

- The image is not byte-identical to the security table's: it also links the controls' module.
  Patch 0003 and the backend objects come from the same recipe, but the image hash differs.
- `build-domain.sh` is a copy. The port's script hard-codes its module list and the source
  directory of its reach module, and this lane may write only under `bug-corpora/postgres/`.
- The first build attempt failed at configure, because the kit's `ld.lld` needs the toolchain's
  `hostlib` on `LD_LIBRARY_PATH`. The failed build directories were removed and rebuilt from
  scratch. No run used them.
- aset.c's `AllocSetContext` and `AllocBlockData` are private, so the module mirrors their
  leading fields: patch 0002's 10 freelists, and `blocks` right after the header. Each run checks
  that mirror before any access: the block owner, the containment, and `freelist[2]` = A's chunk
  after the pfree.
