# Tier 3 rebuild manifest: the bug corpus's virtual arms

Selected from the bundles at `44d82a9eb3f3`, as the plan's acceptance item 3.4 asks
(`capstone/docs/plans/2026-10-16-virtual-profile-on-silicon.md`, lines 49-56 at origin/dev `a0516df1a290`).
Everything here is read from git objects. Nothing was built, run, committed or pushed.

| file | what it is |
|---|---|
| `enumerate.py` | the rule, as code: rev + repo in, counts out, manifest written |
| `enumerate-run.txt` | the output of the run that wrote the manifest (exit status on the last line) |
| `manifest.json` | the rule, counts, the 26 bundles (recorded inputs verbatim, runner blob, recipe, landing commit, drift), the 382 cells (verdict verbatim plus the full observation), the 304 images, survivors, problems |
| `manifest.tsv` | one `kind=cell` row per cell (382), then one `kind=image` row per distinct image (304) |

Command (from any directory; `--expect` turns a count mismatch into exit 4):

    python3 /tmp/capstone/vp2-prep/tier3/enumerate.py --expect 26,382,304 --expect-survivors 91

Defaults: `--repo /tmp/capstone/wt-land-b0 --rev 44d82a9eb3f3`. `--survivors` defaults to
`/mnt/hdd/<user>/capstone/virtual-corpus-images-2026-10-16`, with `<user>` taken from `$USER` when it runs. The
outputs replace the login with `<user>`. Exit codes: 0 clean, 2 no data, 3 consistency problem, 4 expectation mismatch.

## Result

| | count | expected (plan) |
|---|---|---|
| bundles | **26** | 26 |
| cells | **382** | 382 |
| distinct images | **304** | 304 |
| survivors matched to a recorded image | **91** of 91 listed, 0 unmatched | 91 |

- **Survivor files.** All 91 files hash to their SHA256SUMS line. This was checked twice: by `sha256sum -c`, and by
  the script re-hashing each file.
- **Survivor names.** Each survivor's path `<program>__<corpus>/<arm>/<case>.dom` names the cell that recorded its
  hash. There are 0 mismatches.
- **No shared images across bundles.** No image is used by more than one bundle: 304 images, each used by exactly 1.
- **Verdicts agree with `case.json`.** All 382 cells have a `case.json` verdict, reason and `verdict_from` that match
  the bundle's own record (382/382).
- **Verdict tally over the 382 cells:**

  | verdict | reason | cells |
  |---|---|---|
  | `CAUGHT` | `null` | 225 |
  | `MISSED` | `null` | 136 |
  | `NO-READING` | `unattributed` | 15 |
  | `NO-READING` | `inconclusive` | 2 |
  | `NO-READING` | `not-reached` | 2 |
  | `NO-READING` | `infra` | 1 |
  | `NO-READING` | `control-failed` | 1 |

## The rule as implemented

**"verdicts-v1 bundle"** is the contract that `tools/verdicts.py` writes into every bundle's `inputs.json` as
`"contract": "verdicts-v1"`. The bundle is the directory `results/<stamp>/<arm>/`, which holds `verdicts.jsonl` and
`inputs.json` (`capstone/bug-corpora/SCHEMA.md:259-261` at the rev: "The bundle is the only original ...
`verdict_bundles` names the current bundle per arm").

**"Configuration's target"** comes from three records:
- the corpus maps each arm name to a configuration in `corpus.json` `arm_configurations`;
- the configuration's `target` is in `capstone/bug-corpora/tools/arms.json` `arms.<configuration>.target`;
- the value selected is `capstone-virtual`. 18 of the 37 configurations have it.

The steps, all read at the rev:

1. Read every `capstone/bug-corpora/**/corpus.json` (24 files).
2. For each entry `arm -> dir` of `verdict_bundles` (the **current** bundle per arm), take
   `cfg = arm_configurations[arm]`.
3. Keep the bundle when `arms.json` gives `arms[cfg].target == "capstone-virtual"`.
4. Require the bundle's `inputs.json` to say `contract == "verdicts-v1"`, `configuration == cfg`, `arm == arm` and
   `corpus == <corpus path>`. A mismatch is an error, not a skip. All 26 bundles pass.
5. A **cell** is one line of `verdicts.jsonl`, and its key is `observation.case`. That key is unique within each
   bundle, and `inputs.json` `cases` equals the line count in every bundle.
6. An **image** is a distinct `observation.image_sha256`. All 382 values are 64 hex digits.
7. The **verdict** of a cell is the record's top-level `verdict` and `reason`, verbatim. In the TSV they are written
   as `verdict=...` / `reason=...` so the field name travels with the value.

Six corpora declare no `verdict_bundles` and contribute nothing: httpd/apr-pool-repros, httpd/bucket-repros,
mruby/gc-slot-repros, mruby/release-differential, perl/release-differential and sqlite/capi-repros.

Sample, `ffmpeg/pool-repros` at the rev:

- **`corpus.json`.**
  - Lines 51-53: `"arm_configurations": { "virtual-malloc": "virtual-ffmpeg-pool-stock", "virtual-nested-pools": "virtual-ffmpeg-pool-sublet" ...`
  - Lines 57-59: `"verdict_bundles": { "virtual-malloc": "results/2026-10-11-virtual/virtual-malloc", ...`
- **`tools/arms.json`.** Lines 246-247: `"virtual-ffmpeg-pool-stock": { "target": "capstone-virtual",`
- **`results/2026-10-11-virtual/virtual-malloc/inputs.json`.** Lines 2 and 29-31: `"arm": "virtual-malloc"`,
  `"configuration": "virtual-ffmpeg-pool-stock"`, `"contract": "verdicts-v1"`, `"corpus": "ffmpeg/pool-repros"`.
- **The first line of `verdicts.jsonl`, abridged:**

      {"verdict": "MISSED", "reason": null,
       "evidence": "reached (the case printed VERDICT DEFECT-REPRODUCED) and completed with no fault",
       "observation": {"case": "00_461fb22053_af_join_dedup_bound", "arm": "virtual-malloc",
                       "image_sha256": "b738dfd069c96fa1e112a5a8b3a34ebe0d6fe10d6591961cc448e373037dcac3",
                       "reached": true, "completed": true, "fault": null, "attribution": null,
                       "controls": [{"name": "bounds-malloc", "observed": "fault", ...}, ...], ...}}

- **The cross-check.** `00_461fb22053_af_join_dedup_bound/case.json` lines 67-69 carry `"verdict": "MISSED"` and
  `"verdict_from": "results/2026-10-11-virtual/virtual-malloc"`.

## Ambiguity 1: the plan's sentence, read literally, selects 57 bundles, not 26

"Every verdicts-v1 bundle whose configuration's target is `capstone-virtual`" can also be read as "every
`inputs.json` at the rev with that contract, whatever corpus.json says". Read that way:

- **100 verdicts-v1 bundles** exist at the rev. By target: capstone-virtual 57, capstone-silicon 31, capstone-domain 2,
  and 10 whose `configuration` is absent from `arms.json`, so they have no target at all.
- **The 10 have six configuration names:** `app-level0-pg-nested-sublet`, `app-sublet`, `replay-arena`,
  `replay-sublet`, `virtual-wmem-chunks` and `virtual-wmem-libc`. Two of them, `virtual-wmem-libc` and
  `virtual-wmem-chunks`, are wmem's earlier virtual bundles, so the literal count of virtual bundles depends on how
  those are treated.
- **Of the 57 capstone-virtual bundles,** 26 are current. The other 31 are:
  - 27 superseded runs, such as `cpython/allocator-boundary-repros/results/2026-10-11-system/*`,
    `postgres/*/results/2026-10-11-exact/*` and the `2026-10-10-virtual` runs of temporal corpora that were re-run on
    2026-10-11;
  - 4 under `capstone/runtime/virtual/results/managed-mallocng/*`, outside any corpus.

  `manifest.json` `literal_reading.virtual_not_selected` lists all 31.

The current-bundle reading is the one that reproduces the plan's numbers. It is also the one the plan's author used.
The commit that wrote the "91 of 304" sentence, `4d9b3d652ce7`, says in its message: "Of the 304 distinct images in
the corpus's 26 virtual bundles". It is also the set `tools/derive-verdicts.py` writes `case.json` verdicts from.
The script implements that reading and prints the literal count as a side line.

## The survivors

- **Coverage.** The 91 are exactly every image of seven bundles, and no image of any other bundle:

  | corpus | bundle | images |
  |---|---|---|
  | ffmpeg/carved-repros | virtual-malloc | 13 |
  | ffmpeg/carved-repros | virtual-nested-pools | 13 |
  | ffmpeg/plain-heap-repros | | 25 |
  | ffmpeg/subobject-repros | | 9 |
  | memcached/plain-heap-repros | | 9 |
  | wireshark/plain-heap-repros | | 12 |
  | wireshark/plain-temporal-repros | | 10 |

  These are the seven `2026-10-10-virtual` bundles written by `tools/run-virtual-cases.py` blob `9e2a8000e2d6`.
- **Independent check against the original copy.** I hashed every ELF under `/tmp/capstone/a2/r6-raw`, the source
  the plan's commit names. It holds exactly 91 distinct current images, each twice (`bin/` and `stage/images/`).
  - Its `ffmpeg-plain-temporal-repros-*` and `memcached-plain-temporal-repros-*` directories hold only images that are
    not current: those are the superseded 2026-10-10 bundles.
  - It also holds one `controls.dom` (sha256 prefix `e60cc469afb0`) in ten directories. That is plausibly the controls
    image of those runs, but no bundle records its hash, so it is a candidate reference, not a match.

## What a rebuild needs, and what the bundles record

For every bundle, `manifest.json` gives:
- `recorded_build`: every non-bookkeeping key of `inputs.json`, verbatim;
- `platform`: the hashes of the compiler, QEMU, kernel, firmware, rootfs, launcher, module, runner and judge;
- `runner`: the script whose blob sha256 equals `platform.runner`. All 26 resolve to a blob, but only 5 come from a
  commit on the rev's ancestry (see below);
- `recipe`: what that runner revision does to produce the hashed image, with the line in that blob.

| # | bundle (under capstone/bug-corpora/) | configuration | cells | images | survive | recorded tally | runner blob, recipe line | landed | build keys recorded |
|---|---|---|---|---|---|---|---|---|---|
| 0 | `cpython/allocator-boundary-repros/results/2026-10-10-sublet-isa/virtual-malloc` | `virtual-cpython` | 32 | 1 | 0 | CAUGHT 11, MISSED 11, NO-READING:inconclusive 1, NO-READING:infra 1, NO-READING:not-reached 1, NO-READING:unattributed 7 | `cpython/allocator-boundary-repros/runners/virtual/run-virtual.py` @ `76836244c3eb` l.155 | `4b7ba20df7f3` | build{application,image_sha256,nested,profile,runtime_dirty,runtime_revision} |
| 1 | `cpython/allocator-boundary-repros/results/2026-10-10-sublet-isa/virtual-nested-pools` | `virtual-cpython-pools` | 32 | 1 | 0 | CAUGHT 24, MISSED 3, NO-READING:not-reached 1, NO-READING:unattributed 4 | same | `4b7ba20df7f3` | same |
| 2 | `cpython/pymalloc-repros/results/2026-10-10-patch0003/virtual-malloc` | `virtual-pymalloc` | 20 | 20 | 0 | MISSED 20 | `cpython/pymalloc-repros/runners/virtual/run-virtual.py` @ `e52a8bf5d0c5` l.116 | `4b7ba20df7f3` | **none** |
| 3 | `cpython/pymalloc-repros/results/2026-10-10-patch0003/virtual-nested-pools` | `virtual-pymalloc-pools` | 20 | 20 | 0 | CAUGHT 20 | same | `4b7ba20df7f3` | **none** |
| 4 | `ffmpeg/carved-repros/results/2026-10-10-virtual/virtual-malloc` | `virtual-mallocng` | 13 | 13 | 13 | MISSED 13 | `tools/run-virtual-cases.py` @ `9e2a8000e2d6` l.159 | `df906f3dcaa0` | build{cc_args,flags,sdk_identity} |
| 5 | `ffmpeg/carved-repros/results/2026-10-10-virtual/virtual-nested-pools` | `virtual-ffmpeg-carve` | 13 | 13 | 13 | CAUGHT 13 | same | `df906f3dcaa0` | same (cc_args `-DFFC_SUBLET_CARVE -I<scratch>/w2/capstone/runtime/include`) |
| 6 | `ffmpeg/plain-heap-repros/results/2026-10-10-virtual/virtual-malloc` | `virtual-mallocng` | 25 | 25 | 25 | CAUGHT 25 | same | `df906f3dcaa0` | build{cc_args,flags,sdk_identity} |
| 7 | `ffmpeg/plain-temporal-repros/results/2026-10-11-virtual/virtual-malloc` | `virtual-mallocng` | 13 | 13 | 0 | CAUGHT 13 | `tools/run-virtual-cases.py` @ `227de8fdecda` l.177 | `4fdcdceae6c9` | build{cc_args,flags,sdk_identity} |
| 8 | `ffmpeg/pool-repros/results/2026-10-11-virtual/virtual-malloc` | `virtual-ffmpeg-pool-stock` | 4 | 4 | 0 | MISSED 4 | `tools/run-virtual-cases.py` @ `227de8fdecda` (prebuilt mode) | `4fdcdceae6c9` | build{argv,cache,controls,prebuilt,sdk_identity} |
| 9 | `ffmpeg/pool-repros/results/2026-10-11-virtual/virtual-nested-pools` | `virtual-ffmpeg-pool-sublet` | 4 | 4 | 0 | CAUGHT 4 | same | `4fdcdceae6c9` | same |
| 10 | `ffmpeg/subobject-repros/results/2026-10-10-virtual/virtual-malloc` | `virtual-mallocng` | 9 | 9 | 9 | MISSED 9 | `tools/run-virtual-cases.py` @ `9e2a8000e2d6` l.159 | `df906f3dcaa0` | build{cc_args,flags,sdk_identity} (cc_args name 6 port sources and 4 include dirs) |
| 11 | `memcached/allocator-repros/results/2026-10-11-virtual/virtual-malloc` | `virtual-memcached-stock` | 9 | 9 | 0 | CAUGHT 1, MISSED 8 | `tools/run-virtual-cases.py` @ `d9ad57011cb1` (prebuilt mode) | `382ec2d31385` | build{argv,cache,controls,prebuilt,sdk_identity} |
| 12 | `memcached/allocator-repros/results/2026-10-11-virtual/virtual-nested-pools` | `virtual-memcached-sublet` | 9 | 9 | 0 | CAUGHT 8, NO-READING:control-failed 1 | same | `382ec2d31385` | same |
| 13 | `memcached/plain-heap-repros/results/2026-10-10-virtual/virtual-malloc` | `virtual-mallocng` | 9 | 9 | 9 | CAUGHT 9 | `tools/run-virtual-cases.py` @ `9e2a8000e2d6` l.159 | `df906f3dcaa0` | build{cc_args,flags,sdk_identity} |
| 14 | `memcached/plain-temporal-repros/results/2026-10-11-virtual/virtual-malloc` | `virtual-mallocng` | 3 | 3 | 0 | CAUGHT 3 | `tools/run-virtual-cases.py` @ `d9ad57011cb1` l.177 | `382ec2d31385` | build{cc_args,flags,sdk_identity} |
| 15 | `postgres/c-repros/results/2026-10-10-virtual/virtual-malloc` | `virtual-mallocng` | 5 | 5 | 0 | CAUGHT 5 | `postgres/c-repros/shared/run-arm.py` @ `369ad6a8ab49` l.165 | `0f257e5af3f0` | build{heap,opt,runtime_sha256,sdk,virtual} |
| 16 | `postgres/mmgr-repros/results/2026-10-11-virtual/virtual-malloc` | `virtual-mallocng-replay` | 5 | 5 | 0 | MISSED 5 | `postgres/mmgr-repros/shared/run-defects.py` @ `b6a4d66832c2` l.126 | `c5d247874078` | **none** |
| 17 | `postgres/mmgr-repros/results/2026-10-11-virtual/virtual-pg-pools` | `virtual-mallocng-replay-pools` | 5 | 5 | 0 | CAUGHT 5 | same | `c5d247874078` | **none** |
| 18 | `postgres/sql-repros/results/2026-10-11-virtual/virtual-malloc` | `virtual-mallocng` | 9 | 1 | 0 | CAUGHT 2, MISSED 7 | `postgres/sql-repros/shared/run-arm.py` @ `f89b5fb21a89` l.426 | `c5d247874078` | extensions, fixture, image{heap,nested,sha256} |
| 19 | `postgres/sql-repros/results/2026-10-11-virtual/virtual-pg-pools` | `virtual-mallocng-pg-pools` | 9 | 1 | 0 | CAUGHT 5, MISSED 2, NO-READING:unattributed 2 | same | `c5d247874078` | same |
| 20 | `sqlite/engine-repros/results/2026-10-11-virtual-request-bounds/virtual-malloc` | `virtual-sqlite-memsys5` | 33 | 33 | 0 | CAUGHT 5, MISSED 25, NO-READING:inconclusive 1, NO-READING:unattributed 2 | `sqlite/engine-repros/shared/run-virtual.py` @ `a2cb5d025df8` l.118 | `3f1fe3a3d8f5` | amalgamation_sha256, build{arm,groups,opt,sdk} |
| 21 | `sqlite/engine-repros/results/2026-10-11-virtual-request-bounds/virtual-nested-pools` | `virtual-sqlite-memsys5-pools` | 33 | 33 | 0 | CAUGHT 27, MISSED 6 | same | `3f1fe3a3d8f5` | same |
| 22 | `wireshark/plain-heap-repros/results/2026-10-10-virtual/virtual-malloc` | `virtual-mallocng` | 12 | 12 | 12 | CAUGHT 12 | `tools/run-virtual-cases.py` @ `9e2a8000e2d6` l.159 | `df906f3dcaa0` | build{cc_args,flags,sdk_identity} |
| 23 | `wireshark/plain-temporal-repros/results/2026-10-10-virtual/virtual-malloc` | `virtual-mallocng` | 10 | 10 | 10 | CAUGHT 10 | same | `df906f3dcaa0` | same |
| 24 | `wireshark/wmem-repros/results/2026-10-11-virtual-23/virtual-malloc` | `virtual-wmem` | 23 | 23 | 0 | MISSED 23 | `wireshark/wmem-repros/shared/run-defects.py` @ `4ff964e88606` l.128 | `31b69a0516f2` | **none** |
| 25 | `wireshark/wmem-repros/results/2026-10-11-virtual-23/virtual-nested-pools` | `virtual-wmem-pools` | 23 | 23 | 0 | CAUGHT 23 | same | `31b69a0516f2` | **none** |
| | **total** | | **382** | **304** | **91** | | | | |

"landed" is the commit at or before the rev that added the bundle's `verdicts.jsonl`. Only the two sqlite and the two
wmem bundles were written by the runner blob that is still at the rev. The other 22 were written by an older revision
of their runner.

**21 of the 26 bundles were written by a runner revision that exists only on lane branches.** The runner's commit is
NOT an ancestor of `44d82a9eb3f3`; the bundles reached dev squashed, so the runner content that wrote them never sat
on dev in that form. The runner commit is the newest commit whose diff set the runner to the recorded blob:

| bundles | runner blob | commit | on rev ancestry | branches containing it |
|---|---|---|---|---|
| 0, 1 | `76836244c3eb` | `d9cd67661fc6` | no | origin/cleanup/no-sublet-heap, origin/ffmpeg/app-sublet-pools, and 5 more |
| 2, 3 | `e52a8bf5d0c5` | `599bd9d9b886` | no | the same 7 |
| 4-6, 10, 13, 22, 23 (all 91 survivors) | `9e2a8000e2d6` | `43a233efbc66` | no | audit2-corpora, origin/audit2-corpora |
| 7-9 | `227de8fdecda` | `fadca9662bcb` | no | the same 7 as bundles 0, 1 |
| 11, 12, 14 | `d9ad57011cb1` | `7637fec12edf` | no | origin/nested/memcached-lifetimes |
| 15 | `369ad6a8ab49` | `0f257e5af3f0` | yes | |
| 16-19 | `b6a4d66832c2`, `f89b5fb21a89` | `852ce78cafb1` | no | the same 7 as bundles 0, 1 |
| 20, 21 | `a2cb5d025df8` | `65d46bbc49d1` | yes | |
| 24, 25 | `4ff964e88606` | `31b69a0516f2` | yes | |

`manifest.json` `runner.refs_containing_commit` lists up to eight refs for each.

The recipes, read from the recorded runner blob:

- **`tools/run-virtual-cases.py`, case.c mode** (bundles 4-7, 10, 13, 14, 22, 23). The command is in blob
  `9e2a8000e2d6` lines 159 and 222, and at line 177 of blobs `227de8fdecda` and `d9ad57011cb1`. Each image is built as

      <SDK>/capstone-cc -O0 -DCORPUS_VIRTUAL -I<corpus>/shared <case>/case.c <corpus>/shared/driver.c <build.cc_args> -o <case>.dom

  One image per case, run as `fixed N` and then `buggy N`. The controls image is `capstone-cc -O0
  postgres/c-repros/shared/controls.c` (line 167), and its hash is not recorded.
- **`tools/run-virtual-cases.py`, prebuilt mode** (bundles 8, 9, 11, 12; `run_hosted`, blob `227de8fdecda` line 472 for bundles 8 and 9, blob `d9ad57011cb1` line 455 for 11 and 12).
  The image is `<build.prebuilt>/bin/defect-NN`, built by the corpus's `shared/build-cases.sh capstone-application`
  with the `build.cache` options. The runner refuses a CMakeCache that does not say them, nor `CAPSTONE_SDK` and
  `PORT_PLATFORM=capstone-application` (`cache_says`). The controls' hashes are recorded in `build.controls`.
- **cpython allocator-boundary** (bundles 0, 1). One interpreter image, `<build>/image/cpython.dom`, from
  `build-virtual.sh cpython OUT`; its manifest says `nested` is `none` or `cpython`. Each of the 32 cells is a
  `trigger.py` run by that one image, and the stdlib is staged from the same build. `build.image_sha256` equals all 32
  cells' `image_sha256`, which the script checked.
- **cpython pymalloc** (bundles 2, 3). The image is `<build>/bin/defect-NN` from `build-cases.sh capstone-application`.
  The runner (lines 84-92) refuses a build whose `work/00/CMakeCache.txt` does not have `PYMALLOC_SUBLET`
  OFF for virtual-malloc, or ON for virtual-nested-pools, and a virtual `CAPSTONE_SDK`. The controls come from
  `--controls-build`.
- **postgres c-repros** (bundle 15). The image is `<bindir>/<case>.dom` from `shared/build-domain.sh`, whose
  `build.json` is recorded as `build`. Controls are in `<bindir>/controls.dom`.
- **postgres mmgr, wireshark wmem** (bundles 16, 17, 24, 25). The image is `<hosted build>/bin/<NN-slug>`, the port's
  hosted program (preset capstone-application, virtual SDK). The CMake option is `PG_SUBLET` or `WM_SUBLET`, OFF or ON
  by arm, and the runner refuses a mismatch. Controls come from `--controls-hosted-build`.
- **postgres sql-repros** (bundles 18, 19). One server image, `image/postgres.dom`, from `build-virtual.sh postgres`
  with `PGSU_NESTED=sublet` for pools. `image.sha256` equals all 9 cells' `image_sha256`, which the script checked.
  Each cell is a `trigger.sql` against a copy of the initdb'd fixture.
- **sqlite engine-repros** (bundles 20, 21). The image is `<build>/<case>.dom` from `ports/sqlite/repro322/build-virtual.py`,
  with `build.arm` `memsys5` or `memsys5-sublet`, the per-group `-D/-U` flags in `build.groups`, `-O0`, and
  `amalgamation_sha256`.

### What the bundles do NOT record, which a rebuild needs

1. **The source revision of every image** (case.c, driver.c, port sources, patches). No `inputs.json` has such a field.
   - There are two proxies, and neither is proven: the landing commit, and the runner's commit (table above). The
     runner's commit is the closer one, since it is where the run's own script lived. It still only marks one point in
     a window: the runner kept that blob until its next change, and the sources may have moved inside that window.
   - **Since the runner's commit**, the case directories differ from the rev in 27 cells (computed from the manifest):
     - postgres c-repros: 5, all 5 `case.c`;
     - postgres mmgr: 10, all 5 `case.c` in each of its two bundles, from `06691aa99b2f`, which is on dev but not on
       the lane branch;
     - sqlite: 10, 5 `case.c` (10, 14, 15, 22, 28) in each of its two bundles;
     - cpython allocator-boundary: 2, case 19 in each of its two bundles, whose probe and negative-control scripts
       were added.
   - **Since landing**, only the 5 postgres c-repros cells differ.
   - **`shared/` also differs from the runner's commit** for the case.c corpora:
     - ffmpeg carved: `driver.c`, `corpus.h` and `carve-recarve-control.c`;
     - `driver.c` alone for ffmpeg plain-heap, memcached plain-heap and wireshark plain-heap. FFmpeg plain-heap's
       changed after the run, in `510126308bb6` (2026-10-16).

     So for 6 of the 7 survivor bundles, a rebuild at the rev compiles sources that differ from those at the runner's
     commit:
     - five through `shared/`, as listed above;
     - FFmpeg subobject through files its `cc_args` name: `ports/ffmpeg/buffer-pool/src/shared/pool-allocator.c` and
       `payload-backend.h`, and `runtime/include/capstone/capability.h` and `sublet/sublet.h` on the `-I` path.

     Only wireshark plain-temporal shows no drift. `manifest.json` carries both drift lists per bundle and per cell:
     - `shared_and_port_files_changed_since_landing` and `..._since_runner_commit`;
     - `case_sources_changed_since_landing` and `..._since_runner_commit`.
   - Since landing, a corpus's `shared/` and its `upstream.port` changed for some bundles (the counts are file paths
     changed between the landing commit and the rev, in `manifest.json` `shared_and_port_files_changed_since_landing`):
     - ffmpeg carved: 21;
     - ffmpeg plain-heap: 19;
     - ffmpeg subobject: 18;
     - memcached plain-heap: 37;
     - wireshark plain-heap: 22;
     - sqlite: 4;
     - postgres mmgr: 1 (`shared/run-defects.py`);
     - postgres c-repros: 3.
   - The postgres c-repros and mmgr `case.c` change is `06691aa99b2f`: each gained a `#ifdef PGCLIENT_NEGATIVE_CONTROL`
     (or the mmgr equivalent) branch. Its default path reads unchanged on the c-repros diff I read, but the file is not
     byte-identical. I did not read the sqlite or mmgr case.c diffs line by line.
2. **A build recipe for 6 bundles with no `build` key at all.**
   - These are pymalloc (2, 3), mmgr (16, 17) and wmem (24, 25).
   - The arm's CMake option is implied by the runner's refusal check, quoted above. Nothing else of the build is
     recorded: no CMake cache, no compiler flags and no SDK path.
   - postgres sql-repros records the image's `nested` and heap, but not how the image was built.
3. **SDK identity.**
   - The run-virtual-cases bundles record only `sdk_identity: ["virtual-mallocng", "virtual"]`.
   - postgres c-repros and sqlite record a path, `/tmp/capstone/pgv-none/image/sdk`, which is now absent.
   - `platform.compiler` is a sha256 of `clang`: `4c6135fe...` in 19 bundles and `7977ad4b...` in 7. It matches no blob
     in the repo, so it cannot be resolved to a revision (UNRESOLVED; it is an identity check, not a recipe).
   - The rebuild uses the option-(iii) SDK by design, so this matters only for interpreting a moved verdict.
4. **Controls images.** None is among the 304, but a MISSED verdict needs the configuration's controls in the same
   invocation (`arms.json` `controls_for_missed`, copied into each bundle record). Their hashes are recorded only for
   the port controls of ffmpeg pool (`pool-return-90`) and memcached allocator (`chunk-free-90`), in `build.controls`.
   - Elsewhere the control evidence string is cut at 160 characters (`run-virtual-cases.py` blob `9e2a8000e2d6` l.193),
     leaving at most a 4-hex prefix: `sha256=4302` for the mallocng controls of the 2026-10-11 bundles, `ee80`/`af1e`
     for sqlite pools.
   - That reading of `sha256=...` as the image hash is supported by `pool-return-90 ... 2bc1` and `chunk-free-90 ... a0e1`
     matching their recorded full hashes.
   - The 2026-10-10 bundles carry no hash at all.
5. **Prebuilt trees and kits that no longer exist here.**
   - `/tmp/capstone/ffpool-cases-OFF` and `-ON` (ffmpeg pool) are absent.
   - `~/pg-virtual-kit/mc/corpus-cases-OFF` and `-ON` (memcached allocator) are absent from this host's home.
     `CMAKE_C_FLAGS` names `<p13-kit>`, which suggests the a-p13 setup, now gone.
   - `/tmp/capstone/pgv-none/image/sdk` is absent.
   - Present: `/tmp/capstone/virt-stack` and `/tmp/capstone/ffmpeg-buffer-pool/build/native/sources/ffmpeg-ported`,
     both named in subobject's `cc_args`.
   - `<scratch>` in several `cc_args` is a placeholder the runner's `local_path()` substituted, so the absolute
     include roots it stood for are not recorded.
6. **cpython allocator-boundary.** `build.runtime_revision` `a58a424b92eb908875682d118e75299312a07445` is not an
   object in this repo's store, nor in the capstone-qemu, caplifive-buildroot or capstone-ariane checkouts
   (UNRESOLVED). `runtime_dirty: true` says the tree had uncommitted changes, so even that revision would not
   reproduce the image. The stdlib staging (`stage_stdlib`) is part of the run's inputs and is not hashed.
7. **postgres sql-repros.** The fixture is recorded only as `fixture.cluster_tree_sha256` `0353a5b9...`. The initdb'd
   cluster itself is not stored, and has to be re-made and matched against that hash.
8. **The exact-bounds setting of 13 bundles.**
   - `platform.exact_bounds` is `true` in 13 bundles, the ones run through `appvm`.
   - It is absent from the 13 run through `tools/virtualvm.py`, whose `platform()` does not record it (rev, lines
     32-42): all of the run-virtual-cases bundles, including every survivor bundle.
   - The plan's statement that today's images need the exact-bounds bit is therefore not checkable from those bundles'
     records.

9. **The compiler of 7 bundles is the October 10 build, and it predates C-80's fix.**
   - `platform.compiler` `7977ad4b...` is the sha256 of `/tmp/capstone/virt-stack/llvm-build/bin/clang`, which is
     bd372e25f5fb, Release with assertions on. Read 2026-10-17 by hashing the clang binaries on this host.
     `4c6135fe...` matches none of the four found, so it stays UNRESOLVED.
   - C-80 (fixed on dev by 7ce213020f87) passes a type mismatch silently in a release build without assertions,
     where an assertions build stops. An assertions build that compiled an image did not take that path. A build
     without assertions may have miscompiled non-immediate FP constants under the virtual `+f`/`+d`.
   - The rebuild uses a compiler carrying the fix in every bundle, so this is a reason to rebuild rather than
     relink. It is also one more reason a moved verdict is not attributable to silicon alone.

## Checks that were made to fire

- **Zero bundles.** A scratch repo whose one corpus.json has no `verdict_bundles` exits 2 with "NO DATA: zero bundles
  selected ... looked in 1 corpus.json files".
- **Bogus repo and pre-contract rev.** `--repo /nonexistent` exits 2. The parent of `0f257e5af3f0`, the first commit
  with `verdict_bundles`, exits 2 on its absent `arms.json`.
- **Consistency checks.** A scratch repo with four planted faults exits 3 and names each one:
  - a `case.json` verdict changed;
  - an `inputs.json` `configuration` changed;
  - a duplicated jsonl line, which tripped three checks: duplicate case, `cases` count and tally;
  - a non-hex image hash.
- **Runner matching.** In that same scratch repo the runner match fails: its only `run-virtual-cases.py` blob is the
  rev's, not the recorded one, and it is reported as UNRESOLVED.
- **Expectations.** `--expect 26,382,305` and `--expect-survivors 90` each exit 4.
- **Survivor checks.**
  - A SHA256SUMS with one altered hash makes `sha256sum -c` exit 1.
  - The script reports `0/4 files hash` for a SHA256SUMS whose files are absent, and `matched 3, unmatched 1` for a
    zero hash.
  - A SHA256SUMS whose hash is filed under a sibling case's name reports 1 name mismatch.

## Ambiguities and open points

- **Literal reading vs current-bundle reading.** Covered above: 57 vs 26. The 26 reproduces the plan's numbers, and
  the plan's own commit says "26 virtual bundles".
- **A cell is a (bundle, case) pair.** Fixed and buggy runs of a case-c case are one image, one cell. A trigger-script
  corpus (cpython allocator-boundary, postgres sql) has one image shared by all its cells. So "rebuild the image" means
  rebuilding the interpreter or server once per bundle, and each cell's case input is its trigger file. Every TSV cell
  row lists the case directory's files with their blob ids at the rev.
- **The verdict to predict against** is the bundle record's `verdict`/`reason`. It agrees with `case.json` for all 382
  cells at the rev, so the choice does not move anything today.
- **Recipe text is my reading of the runner.** It was checked against the cited line of each recorded runner blob, but
  it is a summary, not a script that has been run. No rebuild was attempted.
- **The `r6-raw` `controls.dom` (`e60cc469afb0...`)** is a candidate controls reference for the seven survivor bundles.
  It is not proven, because no bundle records a controls hash for them.

## The pilot: one bundle rebuilt and run on the ABI v6 kit (2026-10-17, not a final image)

**What ran:** bundle 13, `memcached/plain-heap-repros` `virtual-malloc`.
- The runner is the compiler lane's revision of `tools/run-virtual-cases.py` (`lane/compiler-exact-bounds-iii`
  85b1daddf230; runner sha256 `5ff1997e...`).
- The SDK is the kit's ABI v6 SDK (`../RESULTS.md`), byte-identical in its built files to the compiler lane's.
  The compiler lane's status line for it is: "builds and produces v6 domains; not yet verified to work".
- The kit is the v6 QEMU adapter (launcher `32f79ca8...`, module `97334f0f...`), the reference QEMU
  (`40264f5d...`) and the guest images. The firmware, kernel and rootfs equal the hashes the original bundle recorded.
- The output went to `/tmp/capstone/vp2-prep/tier3-pilot`, not to the corpus's `results/`.

**What it read:**
- **CAUGHT 9 of 9, the recorded tally.**
- All 9 images were rebuilt; every image hash differs from the recorded one.
- For each case, the verdict, the fault cause (28), the faulting symbol (`mch_write_probe` / `mch_read_probe`) and
  the attribution (`probe`) equal the recorded cell's.
- Both configuration controls, bounds-malloc and uaf-malloc, faulted as expected.

**What this shows and what it does not:**
- The case.c rebuild path (9 of the 26 bundles, 107 of the 304 images) works end to end on the v6 SDK. That covers
  build, one boot, the judge, and a comparison cell by cell.
- It does not show the v6 heap is sound. run.py's contract FAILS on the same kit (`../RESULTS.md`), and these 9
  cases are spatial.
- **The second pilot, a temporal bundle (bundle 14, `memcached/plain-temporal-repros`, 3 cases, same kit and
  SDK):** CAUGHT 3 of 3, the recorded tally. Every cell reads cause 25 at its labelled probe (`mct_read_probe` /
  `mct_write_probe`), attribution `probe`, reached, equal to the record. The controls faulted as expected.
- **The third pilot, a Sublet bundle (bundle 5, `ffmpeg/carved-repros` `virtual-nested-pools`, 13 cases, the
  recorded `cc_args` `-DFFC_SUBLET_CARVE` and the runtime include):** 11 of 13 cells equal the record (CAUGHT, cause 28
  at the labelled probe). The four controls read as expected, including the live re-carve (complete) and the stale
  carve (25). **Cases 00 and 10 read NO-READING control-failed instead of CAUGHT.** In both, the v6 heap returned
  NULL to the case's allocation:
  - malloc(6144), case 10's fixed arm;
  - calloc(1, 34816) and calloc(1, 35840), both arms of case 00.
  Sizes from 64 to 12288 B succeeded in the same boot, 3072 and 12288 among them, so it is not a size threshold.
  This is the same v6 heap that fails run.py's contract (`../RESULTS.md`). There, each measured refusal is the
  module's `add()` rejecting an arena that is not a power of two. This pilot's own group sizes (20480 and 36864 by
  the compiler lane's model) were not measured. These two cells wait on the module fix. (Corrected 2026-10-17
  05:00: the first version stated the modelled sizes as fact, and called the cause confirmed two-sided; see
  `../RESULTS.md`.)
- **Cases 00 and 10 rerun in three more fresh boots** (same kit, images and binaries; `--only 00,10`):
  - case 10 fixed, malloc(6144), was admitted in 1 of 4 boots (block 6144, VERDICT FIXED; its buggy arm CAUGHT at
    cause 28, as recorded) and NULL in the other 3;
  - case 00, calloc(1, 34816) and calloc(1, 35840), was NULL in all 4 boots.
  The same binary and input give both outcomes for case 10, so that refusal depends on something that varies per
  boot. This fits `add()`:466 refusing a power-of-two arena that is not naturally aligned. The compiler lane measured
  a refused 8192-byte mapping at 4096 alignment. Case 00's group is not a power of two, so it is refused at any
  address. That the refusal is line 466 rather than another EINVAL is the compiler lane's next reading (the
  refused base).
- **Tool note:** the compiler lane's toolchain copy has `llvm-readobj` but no `llvm-readelf`, which the judge
  calls. The pilot used a bin directory of symlinks to that toolchain, with `llvm-readelf` pointing at
  `llvm-readobj`. So the recorded compiler hash is still that clang's (`ba603a57...`).

## The Sublet pilot again, on the fixed v6 revision: pre-registered (2026-10-17, before the run)

Bundle 5, `ffmpeg/carved-repros` `virtual-nested-pools`, rerun on the kit built from the compiler lane's 89ee98be99e5.
That kit carries the module's `add()` fix, whose results are in `../RESULTS.md`.
- **Unchanged from the third pilot:** the runner (sha256 `5ff1997e...`), the corpus files (`diff -r` empty), the
  recorded `cc_args`, the reference QEMU and the guest images.
- **Changed:** the runtime revision moved from 85b1daddf230 to 89ee98be99e5. The adapter module is `d24d036e...` (was
  `97334f0f...`), the launcher `8234e6a1...` (was `32f79ca8...`), and the SDK is rebuilt. So this is a reading of the
  bundle on the candidate revision, not a one-variable test of `add()`.

**Predictions:**
- **13 of 13 CAUGHT, the recorded tally.** Each case reads cause 28 at its labelled probe, attribution `probe`.
- **Cases 00 and 10, the two NO-READING cells, now read CAUGHT,** with their fixed arms printing VERDICT FIXED.
  - **Case 10:** its malloc(6144) group is a power of two, and the old rule refused it only when mmap placed it
    unaligned (1 of 4 boots admitted). The new rule accepts any representable arena, and a page-aligned power of two
    is always representable.
  - **Case 00:** calloc(1, 34816) and calloc(1, 35840). The old rule refused their group at any address, because it
    is not a power of two. The compiler lane's modelled group, 36864 B, has granule 2^(15-9) = 64 under option
    (iii), so a page-aligned base and that length are representable.
- **The other 11 cases equal the third pilot's cells.** The four controls read as before: bounds-malloc fault,
  uaf-malloc fault, live-carve complete, uaf-carve fault.

**Open threads a failure would land on:**
- **A launch fault before any case output, cause 29 at address 0 in `capstone_launch_unpack`,** is the compiler lane's
  open thread on this revision (`../RESULTS.md`, run.py's `basic`), not the case.
- **A NULL allocation again in case 00 or 10** means `add()` still refuses those arenas. The guest's refusal lines
  name the base and length.

**Result (2026-10-17, one boot): every prediction held. 13 of 13 CAUGHT, every cell equal to the record.**
- **Compared by script:** verdict, fault cause, faulting probe symbol, attribution and reach, against the bundle's
  recorded cells (`manifest.json`). 0 mismatches.
- **The positive control:** the same comparison over the third pilot's verdicts finds exactly 2 mismatches, cases 00
  and 10.
- **Cases 00 and 10's fixed arms now allocate and pass:**
  - case 00: `block mallocng bytes=35840`, `VERDICT FIXED`, exit 0;
  - case 10: `block mallocng bytes=6144`, `VERDICT FIXED`, exit 0.
  Their buggy arms fault with cause 28 at `ffc_write_probe_u8`, as recorded.
- **The four controls read as expected:** bounds-malloc fault (28), uaf-malloc fault (25), live-carve complete,
  uaf-carve fault (25).
- **No launch fault and no refusal line** in the guest's transcript.
- The output is in `/tmp/capstone/vp2-prep/tier3-pilot/ffmpeg-carved-fix`, not in the corpus's `results/`.

**What this shows:** the third pilot's two NO-READING cells were the module's arena rule. On the candidate revision
the Sublet bundle reproduces its recorded verdicts in full. The rebuild of the other 23 bundles is still to run.

## The plain case.c bundles on the fixed v6 revision: pre-registered (2026-10-17, before the runs)

Next are the five bundles whose recorded build is the plain case.c recipe: no `cc_args`, flags `-O0 -DCORPUS_VIRTUAL`,
runner `tools/run-virtual-cases.py`. They run on the same kit, SDK, runner and toolchain as the fixed Sublet pilot
above. Each gets its own guest boot, serially under the QEMU lock, with output outside the corpus's `results/`.

| manifest # | corpus | arm | cases | recorded tally |
|---|---|---|---|---|
| 5 | `ffmpeg/carved-repros` | `virtual-malloc` | 13 | MISSED 13 |
| 7 | `ffmpeg/plain-heap-repros` | `virtual-malloc` | 25 | CAUGHT 25 |
| 8 | `ffmpeg/plain-temporal-repros` | `virtual-malloc` | 13 | CAUGHT 13 |
| 23 | `wireshark/plain-heap-repros` | `virtual-malloc` | 12 | CAUGHT 12 |
| 24 | `wireshark/plain-temporal-repros` | `virtual-malloc` | 10 | CAUGHT 10 |

**Predictions:**
- **Every one of the 73 cells equals its recorded cell** in verdict, fault cause, faulting probe symbol, attribution
  and reach. The same script as the Sublet pilot's checks this.
- **Each bundle's configuration controls read as the runner expects.**

**Why bundle 5's MISSED is the right reading:** without the Sublet carve, a crossing between carved sub-regions of one
malloc block stays inside the block's bounds. Its record says MISSED 13, the same cases the nested-pools arm catches.

**What a mismatch would say:** a CAUGHT where MISSED was recorded, or the reverse, is a change of verdict on the
candidate revision. A NO-READING is a harness or heap problem, to be read from the guest transcript before anything
else.

**Result (2026-10-17 09:20-09:21, one guest boot per bundle):**
- **Every verdict holds:** MISSED 13, CAUGHT 25, CAUGHT 13, CAUGHT 12 and CAUGHT 10, the recorded tallies.
- **The cell-by-cell prediction held for four of the five bundles,** 63 cells with 0 mismatches.
  - The check is `compare-cells.py` (committed here), the Sublet pilot's comparison as a tool.
  - Its positive control, over the third pilot's verdicts, finds exactly cases 00 and 10.
- **It missed for the fifth, `wireshark/plain-temporal-repros`.** All 10 cells are CAUGHT at the same probe
  (`wst_read_probe`), attribution `probe`, reached. But the fault cause is **25** where the record says **24**.
  - The guest's line for each is QEMU's `Cap mem access on revoked capability`.
  - This bundle was recorded on 2026-10-10. The other two temporal bundles were recorded on 2026-10-11 and read 25 for
    the same access shape: `ffmpeg/plain-temporal-repros` 13 x 25, `memcached/plain-temporal-repros` 3 x 25.
  - So the record that differs is the older one, and the new runs agree with the newer records.
  - Why the 2026-10-10 runtime read 24 is not established here.
  - **Not a verdict change.** The schema (`bug-corpora/SCHEMA.md:254`) makes CAUGHT need reached, a fault and an
    attribution, not a particular cause. Both readings are CAUGHT at the labelled probe, so a fold over verdicts sees
    no flip. The cause number is recorded beside the verdict, not in it.

## The prebuilt bundles on the fixed v6 revision: pre-registered (2026-10-17, before the builds)

These four bundles run in the runner's prebuilt mode. Each corpus's `shared/build-cases.sh capstone-application` is
rebuilt against the kit's v6 SDK with the recorded `build.cache` options, and so is its `controls/virtual` copy. The
runner refuses a CMakeCache that does not state those options and that SDK (`cache_says`).
- **The archives are local copies,** so nothing is downloaded: FFmpeg 9.0.1 from `/tmp/capstone/ffmpeg-buffer-pool`
  through `FFPOOL_ARCHIVE`, and memcached 1.6.45 from `/tmp/capstone/dl`, the port's default.
- **The rebuilt images and controls cannot match the recorded hashes,** because the SDK is different. The cells are
  compared as above.

| manifest # | corpus | arm | cache | recorded tally |
|---|---|---|---|---|
| 9 | `ffmpeg/pool-repros` | `virtual-malloc` | `FFPOOL_SUBLET=OFF` | MISSED 4 |
| 10 | `ffmpeg/pool-repros` | `virtual-nested-pools` | `FFPOOL_SUBLET=ON` | CAUGHT 4 |
| 12 | `memcached/allocator-repros` | `virtual-malloc` | `MCP_STOCK_MALLOC=ON MCP_SUBLET=OFF` | CAUGHT 1, MISSED 8 |
| 13 | `memcached/allocator-repros` | `virtual-nested-pools` | `MCP_STOCK_MALLOC=OFF MCP_SUBLET=ON` | CAUGHT 8, NO-READING:control-failed 1 |

All four also take `CMAKE_BUILD_TYPE=Debug` and `-isystem <the kit's clang>/lib/clang/22/include`.

**Predictions:**
- **Every cell equals its recorded cell,** except one.
- **That one is bundle 13's case 05 (`05_2d61f18_item_data_one_past`), predicted CAUGHT.** Its recorded fixed arm
  faulted with cause 24 at address 0, a NULL dereference. That is the shape the module's `add()` refusals produced in
  the Sublet pilot. **The hypothesis behind the prediction:** that NULL was an arena refusal the fix removes.
  - If the fixed arm still faults 24 at 0, the hypothesis is wrong for this case, and the cell is read from its
    transcript.
- **Bundle 13's tally, if the prediction holds:** CAUGHT 9.

**Result (2026-10-17 09:23-09:25): one bundle of four as predicted; the other three FAIL their controls.**

The cases and controls were built with `CAPSTONE_LLVM_BUILD_DIR` set to the kit's own LLVM, the one the SDK was built
with. The preset's default, `<src>/llvm/cmake-build-debug`, does not exist in this source tree, and the first attempt
failed at configure on its missing `llvm-ar`. The runner accepted every CMakeCache (`cache_says`).

| # | bundle | read | recorded | cells equal to the record |
|---|---|---|---|---|
| 12 | memcached allocator, `virtual-malloc` | CAUGHT 1, MISSED 8 | CAUGHT 1, MISSED 8 | **9 of 9** |
| 9 | ffmpeg pool, `virtual-malloc` | NO-READING:control-failed 4 | MISSED 4 | 0 of 4 |
| 10 | ffmpeg pool, `virtual-nested-pools` | NO-READING:control-failed 4 | CAUGHT 4 | 0 of 4 |
| 13 | memcached allocator, `virtual-nested-pools` | NO-READING:control-failed 9 | CAUGHT 8, control-failed 1 | 1 of 9 (case 05, control-failed as recorded) |

**Two signatures:**
- **ffmpeg pool, both arms.** Every program, the `pool-return-90` control included, faults with cause 24 at
  `cap_malloc_validate`.
  - That is `free()`'s probe load (`runtime/virtual/heap.c:333`, `lbu zero, 0(p)`), so `free` is reached with a
    pointer that is not a capability.
  - The generic controls, bounds-malloc (28) and uaf-malloc (25), read as expected.
  - Not C-72: the kit's compiler defines `__SIZEOF_INTCAP__`, and the port has no intcap guard.
  - The caller is not identified.
- **memcached allocator, Sublet arm only.** The control and seven of the cases fault with cause 29 inside
  `capstone_cap_derive`, at its CDERIVE (word `0xa2e395db`, funct7 0x51), after its own window test has passed.
  - The stock arm of the same corpus runs clean.
  - Case 01's buggy arm still faults 25 at `read_probe`, a correct catch.
  - A 29 from CDERIVE with the window inside the parent is consistent with Q-15's refusal of an inexact child on
    QEMU's virtual profile. The child's bounds are not logged, so **that is not established**.

**Prediction for bundle 13's case 05: not confirmed.** It reads control-failed again, so the NULL-from-`add()`
hypothesis is not supported for that case either. Its fixed arm now fails before reaching that point, inside
`capstone_cap_derive`.

**What is not excluded:** a defect in this rebuild rather than in the revision. Prebuilt mode builds each port from
the CURRENT corpus scripts at 89ee98be99e5, with the cache options as recorded. The recorded bundles came from older
script and runner revisions, with another toolchain (`~/pg-virtual-kit`). The plain case.c bundles, built by the
runner itself, all ran.

**The two memcached pilots rerun on the fixed revision (2026-10-17).** `memcached/plain-heap-repros` and
`plain-temporal-repros` had run only on 85b1daddf230's kit, the first two pilots above. On 89ee98be99e5 they read
CAUGHT 9 and CAUGHT 3, every cell equal to the record (`compare-cells.py`, 0 mismatches).
- So **nine** of the thirteen `run-virtual-cases.py` bundles hold on the fixed revision, the Sublet carve included.
- **Correction to dev 2df55c358ccb's message.** It already said nine. At that commit two of the nine had run only on
  the older kit, so the true count was seven. The count of nine holds from this rerun on.

## `ffmpeg/subobject-repros` `virtual-malloc` on the fixed revision: pre-registered (2026-10-17, before the run)

The recorded build compiles FFmpeg's pool sources directly into each case: `buffer.c`, `refstruct.c`, and the port's
`pool-allocator.c`, `metadata-allocator.c`, `observe-pool-events.c` and `payload-pointers.c`. It also carries
`-Daligned_alloc(a,n)=malloc(n)`. The same pool code faulted in `free()` with cause 24 in every program of the two
pool bundles above, so this bundle discriminates.
- **The recorded `cc_args`, with paths mapped:** `<scratch>/wt-merge` becomes the kit's source tree, and the
  `-isystem` becomes the kit's clang include. The `ffmpeg-ported` sources are the recorded
  `/tmp/capstone/ffmpeg-buffer-pool` copy.
- **Recorded tally:** MISSED 9.

**Both outcomes, written before the run:**
- **(a) MISSED 9, every cell equal to the record.** The pool code itself does not hand `free()` a non-capability on v6.
  The pool bundles' failure then lies in what differs: their CMake library build (`-DFF2_SYSTEM_MEMORY`, the
  `ffmpeg-traced` sources) and the `aligned_alloc` path, which this build maps to `malloc`.
- **(b) control-failed with cause 24 at `cap_malloc_validate`, as in the pool bundles.** The failure lies in code
  common to both builds, not in `aligned_alloc`.

No prediction between them: this run decides.

**Result: outcome (b), in class. NO-READING:control-failed 9, against a recorded MISSED 9; 0 of 9 cells equal.**
- The generic controls read as expected: bounds-malloc 28, uaf-malloc 25.
- **Every case program faults with cause 24 in the pool code, in both arms.**
- **Case 00** faults at `refstruct_pool_uninit+0x78`:
  - `ldc a2, 0(a0)` loads a capability from memory;
  - the next store through it, `sw a1, 0x60(a2)`, faults 24.
  So a pointer that the pool code stored in memory comes back without its tag.
- The site differs from the pool bundles' (`free()`'s probe), but the class is the same: a value used as a capability
  is not one.
- **What this settles:** the `aligned_alloc` path is not needed for the failure, because this build maps
  `aligned_alloc` to `malloc`. The failure is in code common to both builds: the FFmpeg pool sources, the port's
  shared allocator files, the v6 runtime, or the compiler.
- **Not established:** where the tag is lost. The record ran the same sources on the 2026-10-10 SDK with
  `/tmp/capstone/virt-stack`'s toolchain; this run uses the v6 kit's SDK and `llvm-dev-current-20261017`.

## Two discriminator runs for the failing bundles: pre-registered (2026-10-17, before the runs)

**Fact first: the kit's QEMU decodes CDERIVE.** The compiler lane asked whether this QEMU pin decodes funct7 0x51.
d35af47b16 has `csderive 1010001 ..... ..... 001 ..... 1011011` (`target/riscv/insn32.decode:984`), implemented by
`helper_csderive` (`op_helper.c:1212`). That helper raises 29 (`ILLEGAL_OP_VAL`) in exactly two places:
- **(a)** size 0, or a window outside the parent, tested from the cursor;
- **(b)** Q-15, when the child does not survive `cap_compress`/`cap_uncompress` and `capstone_exact_bounds` is off.

`capstone_cap_derive`'s own pre-test bounds the offset, not offset + size, so it does not exclude (a).

**Run 1: memcached allocator, Sublet arm (#13), on a kit whose QEMU runs with `x-capstone-exact-bounds=true`.** The
binaries, controls, SDK and module are the same. Only the QEMU `-cpu` string differs, through a wrapper.
- If the 29s are Q-15's (b), they vanish. The arm then reads as recorded, CAUGHT 8, with case 05 read from its
  transcript.
- If they are (a), they stay.

**Run 2: ffmpeg pool, stock arm (#9), the same binaries, on the PRE-fix module kit** (`v6-pilot`, module `97334f0f`,
launcher `32f79ca8`). This is the compiler lane's single-variable test for the module change.
- **(i)** Still cause 24 at `cap_malloc_validate`: the module change is exonerated.
- **(ii)** The 24s stop: the module change is implicated.
- **(iii)** A NULL from a refused arena instead: the old module's `add()` rule decides first, and the test does not
  separate.

The binaries carry the v6-fix SDK's runtime, so `--sdk` stays the v6-fix SDK (the runner checks the CMakeCache
against it).

**Run 1 result: Q-15's (b).**
- With `x-capstone-exact-bounds=true` the memcached Sublet arm reads CAUGHT 8 and NO-READING:control-failed 1, every
  cell equal to the record (`compare-cells.py`, 0 mismatches). With the property off it read 8 mismatches.
- The wrapper's log shows the property on the run's one boot:
  `-cpu rv64,...,x-capstone-u-mode=true,x-capstone-exact-bounds=true`.
- **So under option (iii) the memcached port's Sublet carve derives children that the compact format cannot hold,
  and QEMU's Q-15 refuses them with 29.** The 2026-10-11 record ran with exact bounds held in the metadata.
- **This is option (iii)'s recorded cost, confirmed, not a new finding.** The virtual-profile plan
  (`2026-10-16-virtual-profile-on-silicon.md`, "The cost omits the nested allocators") lists memcached 0006's slab
  chunks among the twenty derive sites that must place children representably under (iii). It says "the Tier 3 nested
  arms depend on them". This run shows that on 89ee98be99e5 the memcached carve is not yet repaired. The fix stays
  with the compiler lane, as port owner.
- **Corrected the same day:** the first version of this line called it "a vp2 Tier 3 blocker". The runtime lane routed
  it to the plan's heading, which already covers it.
- **Not established:** which derive and which size. The compiler lane's cause-29 fault had the same instruction and
  cause through the v6 heap's unrounded `aligned_alloc`. It is fixed at its d4c8d191bcd0, and does not touch this
  corpus.

**Run 2 result: (i), the module change is exonerated for the pool's cause 24.**
- On the pre-fix module kit the same binaries read NO-READING:control-failed 4.
- Seven programs fault with cause 24 at `cap_malloc_validate`, as on the fixed kit.
- Case 03's two programs fault 24 at `memset+0x2c` instead, VA 0x11a08.
- The tag loss does not depend on `add()`.

## Run 3: compiler or runtime, for the FFmpeg pool tag loss: pre-registered (2026-10-17, before the run)

`ffmpeg/subobject-repros` case 00, built twice; asked by the compiler lane. The SDK is the v6-fix SDK both times
(runtime, libc, builtins, linker and codegen flags), and only `sdk.json`'s `cc` differs:

| arm | clang | identity (`--version`) | binary sha256/16 |
|---|---|---|---|
| kit (as run above) | `/tmp/capstone/vp2-prep/v6-fix/llvm-build/bin/clang` | llvm-capstone 85b1daddf230 | ba603a574ce08ea4 |
| old | `/tmp/capstone/virt-stack/llvm-build/bin/clang`, the 2026-10-10 record's | llvm-capstone bd372e25f5fb | 7977ad4b53d445e8 |

**Outcomes:**
- **(i) The old clang's build also faults 24:** the compiler is exonerated, and the loss is in the runtime or the
  sources.
- **(ii) It does not fault, and case 00 reads as recorded:** a compiler regression in bd372e25f5fb..85b1daddf230.
- **(iii) `capstone-cc`'s qualification probes refuse the old clang** (C-46, intcap, C-80): no reading.

**Run 3 result: (i), the compiler is exonerated for this case.**
- **The old arm ran the old clang.** A verbose compile through the copied SDK's `capstone-cc` invokes
  `/tmp/capstone/virt-stack/llvm-build/bin/clang-22` (bd372e25f5fb). The kit's runs `llvm-dev-current-20261017`
  (85b1daddf230).
- **The record's own compiler.** The 2026-10-10 record's `platform.compiler` is
  7977ad4b53d445e89114468b57371c4c9ecbb412b0b575db474fe169b85dcf16, exactly the old clang's binary.
- **The image is byte-identical** to the kit-clang build (`bdf538f4859482f3`), and its fixed arm faults the same way:
  cause 24 at code+0x16eec, `refstruct_pool_uninit+0x78`.
- **So for this program the two compilers emit identical code.** Built with the record's compiler, the v6 SDK still
  loses the tag. The loss is in what the SDK brings (runtime, libc), or in the sources against it, not in the
  compiler change.
- **Prior art the runtime lane points to:** C-69, FF_FIELD_AT's align-1 capability load lowered to a byte copy. A byte
  copy would show the same way, but it is a compiler-side mechanism, and this result makes it identical under both
  compilers. Which of the SDK's pieces drops the tag is not established.

## wireshark/wmem-repros, both arms, on the fixed revision: pre-registered (2026-10-17, before the runs)

**The runner and its builds.**
- The runner is the corpus's `shared/run-defects.py`, on a `capstone_vm` VM booted from the v6-fix kit:
  - its QEMU, Image, fw_jump and rootfs (`-snapshot`);
  - `--profile virtual`, the launcher `capstone-vexec` and the module `capstone_vm.ko`.
- **The hosted builds** use the port `ports/wireshark/wmem` with the `capstone-application` preset, against the v6-fix
  SDK, with `CAPSTONE_LLVM_BUILD_DIR` = the kit's LLVM.
  - `WM_CORPUS_DIR` = the corpus root for the cases, and `controls/` for the controls.
  - `WM_SUBLET` is OFF for `virtual-malloc` and ON for `virtual-nested-pools`.
  - The archive is the local `wireshark-4.6.8.tar.xz`.
- **This revision's patch 0001 has no `repr_round`.** The virtual-profile plan lists "Wireshark's wmem jumbo blocks
  (0001)" among the twenty derive sites that need representable placement under (iii).

**Predictions:**
- **`virtual-malloc` (#25):** MISSED 23, every cell equal to the record. The controls read as the runner requires:
  chunk use-after-free and overread complete, jumbo use-after-free faults.
- **`virtual-nested-pools` (#26):** two outcomes, written before the run.
  - **(a) CAUGHT 23, every cell equal to the record:** no case derives a child that is inexact under (iii).
  - **(b) Some cells fault 29 at a CDERIVE:** the jumbo-block derive is refused by Q-15, as memcached 0006 was. The
    same exact-bounds rerun as Run 1 then separates it: with `x-capstone-exact-bounds=true` the cells return to the
    record.

The VM holds the host-wide QEMU lock while it is up, so the session runs as one bounded block and is brought down
after.

## Run 4: `+movc-keeps-integer-source`, the one SDK flag arm that links: pre-registered (2026-10-17, before the run)

**Why only one arm.** The compiler lane cut five single-variable SDK arms (`/tmp/capstone/vp2-prep/sdk-arm-{1..5}`,
`SDK-ARMS.md`). Only arm 1 links:
- the 10-10 runtime and libc archives are soft-float, and `capstone-cc` refuses them on the virtual profile, which
  needs F and D;
- dropping `+f/+d` or the ABI fails `capstone-cc`'s codegen gate.
So the 10-10 record and the v6 kit differ by profile, not by one flag.

**Arm 1:** the v6-fix SDK with `+movc-keeps-integer-source` dropped from `sdk.json` codegen, one key different
(diffed), the kit's clang. Subobject case 00 is built as in run 3.

**Outcomes:**
- **(i) cause 24 at `refstruct_pool_uninit+0x78` again:** the flag is exonerated, and what remains is the archives'
  contents.
- **(ii) the fixed arm passes:** the flag is implicated.
- **The image hash is recorded beside the reading,** so a silently identical rebuild is visible.

**Run 4 result: (i), the flag is exonerated.**
- Arm 1's image is a different build, `561c29fdc447ebae` against the kit's `bdf538f4859482f3`, so it is not a silent
  identical rebuild.
- It faults the same way: cause 24 at `refstruct_pool_uninit+0x78`, the same `ldc a2, 0(a0)` and then `sw a1,
  0x60(a2)`.
- **Taken together with run 3, the tag loss is in the v6 profile's own runtime or libc contents, or in the sources
  against them.** It is not the compiler change, and not `+movc-keeps-integer-source`.
- **The archives cannot be swapped**, because the 10-10 ones are soft-float. So the next separation is a rebuild of
  those archives' sources with v6 flags, a scope for the compiler lane to set.

**wmem results (2026-10-17 10:11-10:13).**
- **The VM.** `capstone_vm up --profile virtual` on the v6-fix kit needs two things the kit lacked:
  - the job helper, which build-adapter.sh already builds (`out/adapter-qemu/capstone-job`, f49e70ff…);
    `build-kit.sh` now copies it into `adapter/`;
  - a guest SSH server, because the kit's rootfs (9903242c…, the record's own) has no dropbear. It is passed as
    `--ssh-server`, the static dropbear the B0 sessions used (`/tmp/capstone/b0/platform/guest/ssh_server`).
- **`virtual-malloc` (#25): MISSED 23, every cell equal to the record** (`compare-cells.py`, 0 mismatches), as
  predicted. The controls behaved: the jumbo use-after-free faulted 25 at the probe.
- **`virtual-nested-pools` (#26): neither pre-registered outcome as written.** NO-READING:control-failed 23.
  - The arm's jumbo use-after-free control faults with **cause 29 during setup**: pc +0x18b20, before its ready mark,
    `last=0xc0de0001`. So the runner reads "expected fault, observed none" at the probe and voids every cell.
  - That is outcome (b)'s mechanism arriving through the control rather than the cases.
- **The exact-bounds rerun separates it.** With `capstone_vm up --exact-bounds` (recorded in the VM's config as
  `"exact_bounds": true`):
  - the controls pass;
  - the arm reads **CAUGHT 22 + NO-READING:unattributed 1, 22 of 23 cells equal to the record**.
  - So wmem patch 0001's jumbo-block derive is refused by Q-15 under (iii), the plan's recorded cost, as for memcached
    0006.
- **The one remaining cell is case 22** (`usbhid_output_usages_freed_twice`, a double free).
  - **Recorded:** CAUGHT, cause 25 in `wmem_block_sublet_free`, the case's declared fault site.
  - **On v6:** the same cause 25, from the same double free, one call deeper, in the runtime's
    `capstone_cap_revoke_child`. The declared site no longer names the frame that faults.
  - Same detection, different attribution. It is a case.json `fault_sites` question for the corpus owner, not a
    behaviour change.

## Run 5: `-fno-builtin` (SDK arm 6): pre-registered (2026-10-17, before the arm exists)

**The defect behind it.** The compiler lane's hosted-contract probe (`runtime/virtual/hosted-contract.c`) finds the v6
SDK losing a capability tag on a constant-size `memcpy` of unknown alignment (`memcpy_constant_size_unknown_alignment`
FALSE).
- Rebuilding only the probe with `-fno-builtin` flips it to TRUE.
- The images differ (683a6b6f against d6de4abc), so it is two-sided.
- The 10-10 SDK lacks the flag as well.

**Arm 6:** the v6-fix SDK with `-fno-builtin` added to `sdk.json` codegen, everything else equal, the kit's clang.
Subobject case 00 as in runs 3 and 4.

**Outcomes:**
- **(i) the fixed arm prints VERDICT FIXED, and the buggy arm reads as recorded (MISSED):** the pool fault is that
  copy shape.
- **(ii) cause 24 at `refstruct_pool_uninit+0x78` again:** a separate shape, and the `memcpy` finding is a second,
  independent defect.
- **Either way open:** why the 10-10 record, whose SDK had the same exposure, passed. Either its codegen did not
  inline that copy, or the copy shape is new in v6's sources. The image hash goes beside the reading.

## postgres/c-repros `virtual-malloc` (#16) on the fixed revision: pre-registered (2026-10-17, before the run)

- **Build:** `shared/build-domain.sh <v6-fix SDK> <out>`, one domain program per case.
- **Run:** `shared/run-arm.py --virtual-kit <v6-fix kit>`, the kit's direct boot, so no VM is needed.
- **Prediction:** CAUGHT 5, every cell equal to the record (`compare-cells.py`).
- **Recorded:** CAUGHT 5, on the 2026-10-10 SDK at `/tmp/capstone/pgv-none/image/sdk`, runtime e5693f5c….

**Result: CAUGHT 5, every cell equal to the record** (0 mismatches). All five fault with cause 28 at the case's declared
function. The build made 10 case images and the controls with heap virtual-mallocng, and the uaf-malloc control
faulted as expected.

## postgres/mmgr-repros, both arms (#17, #18), on the fixed revision: pre-registered (2026-10-17, before the runs)

- **Hosted builds:** `ports/postgres/memory-contexts`, `capstone-application` preset, against the v6-fix SDK, with
  `PG_SUBLET` OFF or ON.
  - `PG_CORPUS_DIR` = the corpus root for the cases, and `controls/` for the controls.
  - The archive is the local `postgresql-17.0.tar.bz2`.
- **Run:** `run-defects.py --virtual-kit`.

**Predictions:**
- **`virtual-malloc` (#17):** MISSED 5, every cell equal to the record.
- **`virtual-pg-pools` (#18):** two outcomes.
  - **(a) CAUGHT 5, every cell equal to the record.**
  - **(b) 29s at a CDERIVE in the cases or controls.** Patch 0003's large chunks are on the plan's list of (iii) derive
    sites. The exact-bounds wrapper kit (Run 1) then separates it.

**Run 5 result: (i).** On arm 6, case 00's fixed arm prints `VERDICT FIXED the fix's bound of 599 keeps every write
inside its own member`. Its buggy arm prints `VERDICT DEFECT-REPRODUCED` and completes: MISSED, as recorded.
- The image is a different build, `ffd2170b30b57e5c` against the kit's `bdf538f4859482f3`.
- The two generic controls fault as expected (28, 25).
- **So `-fno-builtin` clears case 00's tag loss.**
- **The mechanism is not established.** The compiler lane withdrew its explanation: in its reduction both compilers
  tail-call `memcpy` with or without the flag. Why the 10-10 record passed without the flag is also open.

## Run 6: the three FFmpeg pool bundles on arm 6: pre-registered (2026-10-17, before the runs)

`ffmpeg/subobject-repros` (all 9) and `ffmpeg/pool-repros` (both arms, prebuilt rebuilt with arm 6's SDK; the runner's
`cache_says` checks the SDK path), on the v6-fix kit.
- **Prediction:** every cell equal to the record: MISSED 9, MISSED 4, CAUGHT 4.
- **A cell that still faults 24 in pool code is a second shape** that the flag does not cover.

**mmgr result (#17, #18): both arms as recorded.** `virtual-malloc` reads MISSED 5 and `virtual-pg-pools` CAUGHT 5,
every cell equal to the record, which is outcome (a). Patch 0003's derives on these five cases are representable under
(iii). The hosted builds made 17 case programs and 8 controls per arm.

**Run 6 result: every cell equal to the record on arm 6.** Subobject 9/9 (MISSED 9), pool stock 4/4 (MISSED 4), pool
Sublet 4/4 (CAUGHT 4). So **`-fno-builtin` resolves the FFmpeg pool tag loss in all three bundles.** The mechanism and
the 10-10 question stay open, as above. Adding the flag to the shipped SDK is the compiler lane's landing, with its
hosted-contract check as the gate.

## Status on the fixed v6 revision (2026-10-17 10:30), computed by `status.py` from the verdict files

```
ffmpeg/carved-repros | virtual-nested-pools | kit 89ee98be | recorded 13 new 13 mismatches 0 missing 0
ffmpeg/carved-repros | virtual-malloc | kit 89ee98be | recorded 13 new 13 mismatches 0 missing 0
ffmpeg/plain-heap-repros | virtual-malloc | kit 89ee98be | recorded 25 new 25 mismatches 0 missing 0
ffmpeg/plain-temporal-repros | virtual-malloc | kit 89ee98be | recorded 13 new 13 mismatches 0 missing 0
wireshark/plain-heap-repros | virtual-malloc | kit 89ee98be | recorded 12 new 12 mismatches 0 missing 0
wireshark/plain-temporal-repros | virtual-malloc | kit 89ee98be | recorded 10 new 10 mismatches 10 missing 0
memcached/allocator-repros | virtual-malloc | kit 89ee98be | recorded 9 new 9 mismatches 0 missing 0
memcached/plain-heap-repros | virtual-malloc | kit 89ee98be | recorded 9 new 9 mismatches 0 missing 0
memcached/plain-temporal-repros | virtual-malloc | kit 89ee98be | recorded 3 new 3 mismatches 0 missing 0
postgres/c-repros | virtual-malloc | kit 89ee98be | recorded 5 new 5 mismatches 0 missing 0
postgres/mmgr-repros | virtual-malloc | kit 89ee98be | recorded 5 new 5 mismatches 0 missing 0
postgres/mmgr-repros | virtual-pg-pools | kit 89ee98be | recorded 5 new 5 mismatches 0 missing 0
wireshark/wmem-repros | virtual-malloc | kit 89ee98be (capstone_vm) | recorded 23 new 23 mismatches 0 missing 0
ffmpeg/subobject-repros | virtual-malloc | SDK arm 6 (-fno-builtin) | recorded 9 new 9 mismatches 0 missing 0
ffmpeg/pool-repros | virtual-malloc | SDK arm 6 (-fno-builtin) | recorded 4 new 4 mismatches 0 missing 0
ffmpeg/pool-repros | virtual-nested-pools | SDK arm 6 (-fno-builtin) | recorded 4 new 4 mismatches 0 missing 0
memcached/allocator-repros | virtual-nested-pools | kit + QEMU exact bounds | recorded 9 new 9 mismatches 0 missing 0
wireshark/wmem-repros | virtual-nested-pools | kit + QEMU exact bounds (capstone_vm) | recorded 23 new 23 mismatches 1 missing 0
bundles run: 18 of 26; not run: [('cpython/allocator-boundary-repros', 'virtual-malloc'), ('cpython/allocator-boundary-repros', 'virtual-nested-pools'), ('cpython/pymalloc-repros', 'virtual-malloc'), ('cpython/pymalloc-repros', 'virtual-nested-pools'), ('postgres/sql-repros', 'virtual-malloc'), ('postgres/sql-repros', 'virtual-pg-pools'), ('sqlite/engine-repros', 'virtual-malloc'), ('sqlite/engine-repros', 'virtual-nested-pools')]
```

**Reading the table:**
- **12 bundles hold cell for cell on the kit as-is.** One more, wireshark plain-temporal, holds every verdict; its
  cause is 25 where its older record has 24.
- **3 bundles (the FFmpeg pool code) need `-fno-builtin` in the SDK.**
- **2 Sublet arms need option (iii)'s recorded port repairs,** shown by QEMU's exact-bounds control: memcached 0006
  and wmem 0001. wmem's case 22 differs in attribution only.
- **8 bundles have not run yet:** CPython (4) and PostgreSQL sql-repros (2) need interpreter or server rebuilds, and
  SQLite engine-repros (2) needs build-virtual.py over 33 cases.

## sqlite/engine-repros, both arms (#21, #22), on the fixed revision: pre-registered (2026-10-17, before the runs)

- **Build:** `ports/sqlite/repro322/build-virtual.py --sdk <v6-fix SDK> --amalgamation <the kit's sqlite-source>`
  (sqlite3.c 7fc92e19…, the record's input), with `--memsys5` for `virtual-malloc` and `--memsys5 --sublet` for
  `virtual-nested-pools`.
- **Run:** `shared/run-virtual.py` on a `capstone_vm` VM from the v6-fix kit, without `--exact-bounds`, which is
  option (iii)'s production configuration.
- **The record's VM ran `--exact-bounds`** (the runner's docstring, 2026-10-11). So this run differs from the record
  in the bounds model as well as the revision.

**Predictions:**
- **`virtual-malloc` (#21): every cell equal to the record** (CAUGHT 5, MISSED 25, NO-READING 3), except where (iii)
  rounds.
  - A CAUGHT that becomes MISSED is allowed only for a spatial overrun of an allocation of 4 KiB or more whose
    accessed byte lies inside its granule's rounding slack. That is (iii)'s documented cost.
  - Each such flip is checked against the allocation's size from its transcript. Any other flip is a finding.
- **`virtual-nested-pools` (#22):** two outcomes.
  - **(a) every cell equal to the record** (CAUGHT 27, MISSED 6), up to the same rounding exception;
  - **(b) Q-15 29s at the memsys5-block CDERIVE,** which is on the plan's list of (iii) derive sites. The exact-bounds
    rerun then separates it.

**Result: not runnable at 89ee98be99e5. NO-READING:build-failed 33 on both arms.** Every case fails to compile at
`ports/sqlite/repro322/repro322_common.h:71`.
- **The cause is the tree's own (iii) change, 99aa4350bcfb** ("memsys5's static arena granule-aligned, with a
  compile-time granule macro").
- The virtual memsys5 branch (lines 66-74) uses `CAPSTONE_REPR_ASSERT_POW2` and `CAPSTONE_REPR_GRANULE_POW2`.
- They come from `<capstone/representable.h>`, which the header includes only in its freestanding branch (line 153).
- `build-virtual.py` adds `-I capstone/runtime/include` only for `--sublet` (line 207).
- So the stock arm finds neither the include nor its path, and the Sublet arm finds the path but not the include.
- The repair belongs to the port's owner, the compiler lane, which has been told. Neither prediction is read.

**Rerun with the build fix (2026-10-17 10:46-11:02).**
- **The tree.** The kit's source tree carries only b0d93849ceba's two-file fix (`build-virtual.py`,
  `repro322_common.h`), applied as a diff onto 89ee98be99e5. The SDK and the kit are unchanged.
- **The controls.** The configuration's controls are built with `--observe controls/control_{uaf,bounds}_mem5.c`. My
  first rerun omitted them, and the runner read every cell control-failed: an error in my recipe.
- **Cases 06 and 22 read build-failed:** they need `--ext-src`, a SQLite 3.22.0 full source tree, and none is on this
  host. The only full tree here is 3.53.3, which is the wrong version.

| arm | without exact bounds ((iii), production) | with `--exact-bounds` | cells equal to the record (exact) |
|---|---|---|---|
| `virtual-malloc` | CAUGHT 3, MISSED 27, unattributed 1, build-failed 2 | the same | 29 of 33 |
| `virtual-nested-pools` | CAUGHT 1, unattributed 30 (cause 29 in `capstone_cap_derive`), build-failed 2 | CAUGHT 25, MISSED 6, build-failed 2 | 31 of 33 |

- **`virtual-nested-pools`: outcome (b), confirmed by the exact-bounds control.**
  - Without exact bounds, 30 cells die with 29 at the memsys5 Sublet derive.
  - With exact bounds, every built cell equals the record.
  - That is the plan's recorded (iii) cost for SQLite's memsys5 blocks.
- **`virtual-malloc`: two real differences, cases 12 and 24, and neither depends on (iii)** (they read the same with
  exact bounds).
  - Both had recorded cause-24 faults (an operand that is not a capability): 12 in `sqlite3Fts5IterTerm`
    (unattributed), and 24 in `sqlite3OpenTable` (CAUGHT under its fault site).
  - On v6 both complete with no fault, so they read MISSED.
  - For 24 this matches the corpus's silicon stock arm, which also completes. The campaign plan records "24 completes
    where virtual-malloc faults", with the emulator agreeing.
  - So the record's cause-24 fault there was a property of the 10-11 virtual build, not of the defect. What dropped
    the tag in that build is not established.
  - **C-81 is a candidate explanation (a stripped tag reads 24), and it cannot be tested: the record's images are
    gone** (checked 2026-10-17).
    - No file under /tmp, /home or /mnt/hdd has either record sha256 (12 dc55fe7c6c2460d0, 24 6fbbd27ef45c7ac8).
      The search's control, the v6 case-12 image, is found 3 times.
    - The build SDK the record names (`build.sdk`) no longer exists.
    - The record's `platform.compiler` hash is the runner's symboliser (SCHEMA.md), and no clang on disk matches
      it anyway. The search's control, dev147's clang-22, is found 4 times.
    - So the rows stay "differ, unexplained".
- **The pre-registered (iii) rounding exception was not needed:** no CAUGHT 28 became MISSED.

**Cases 06 and 22, with the 3.22.0 ext source (2026-10-17 13:27-13:34, `tier3-sqlite-ext.sh`).**
- **The source.** The kit's own cached `sqlite-src-3220000.zip`, at the SHA3-256 that `ports/sqlite/app/prepare-sources.sh`
  pins (f8695d69...; VERSION 3.22.0), unpacked and passed as `--ext-src`. Kit, tree, controls and runner are the
  10:46 rerun's.
- **The 31 cells already read are the check:**
  - all 33 images of each arm's earlier build are byte-identical in this one, and the two new images are 06 and 22;
  - all 93 earlier readings (31 cells by three runs) repeat exactly.
- **Predictions, written before any reading** (`sqlite-ext-PREDICTION.txt`):
  - virtual-malloc 06 inconclusive and 22 CAUGHT 24 in `editDist3FindLang`, as the record;
  - nested-pools without exact bounds, cause 29 in `capstone_cap_derive` for both;
  - with exact bounds, the record: 06 CAUGHT 25 in `idxRemFunc`, 22 CAUGHT 25 in `editDist3FindLang`.

| arm | 06 expert | 22 spellfix | all 33 against the record |
|---|---|---|---|
| `virtual-malloc` | inconclusive (= record) | CAUGHT 24 `editDist3FindLang` (= record) | 31 of 33; the two differences are 12 and 24, as above |
| `virtual-nested-pools`, (iii) | 29 in `capstone_cap_derive` (as predicted) | **CAUGHT 25 `editDist3FindLang` (= record; predicted 29)** | 2 of 33 (18 and 22); the other 31 are cause 29 at the derive |
| `virtual-nested-pools`, `--exact-bounds` | CAUGHT 25 `idxRemFunc` (= record) | CAUGHT 25 `editDist3FindLang` (= record) | **33 of 33** |

- **One prediction missed:** case 22 under (iii). Its run reaches the stale-vtab fault without first attempting an
  inexact derive: an attempted one would have stopped it with 29 at the derive, as it stops the other 31.
  - The prediction applied the other 30 cells' outcome without checking 22's allocation order.
  - The reading equals the record, so the miss changes no conclusion.
- **SQLite is now fully read on v6:**
  - virtual-malloc 31 of 33, with 12 and 24 independent of (iii);
  - nested-pools 33 of 33 under exact bounds, and under (iii) the plan's recorded Q-15 cost on every cell that
    derives an inexact child before it faults.

**Status after SQLite (computed by `status.py`, its SQLite rows now the 33-cell runs): 20 of 26 bundles run; not run: cpython (4) and postgres/sql-repros (2), blocked on the per-port SDK build (soft-float builtins in a hard-float link), which the compiler lane has taken.**

**Scope of the `-fno-builtin` result (2026-10-17, from the compiler lane's measured control; now ISSUES C-81).**
- **The flag diverts only EXPLICIT `memcpy` calls to musl's run-time copy.** `__builtin_memcpy`, and every copy the
  compiler synthesises (struct assignment, by-value aggregates, array initialisation), still expand bytewise with the
  flag on when their alignment is not provable, and drop the tags.
- **So run 6's 17/17 shows that the FFmpeg pool's tag-losing copies were explicit `memcpy` calls. It does not show
  that the images are tag-safe.** Where this README says the flag "resolves" or "clears" the pool tag loss, read: the
  three bundles read as recorded with it.
- **The defect is long-standing, not a regression.** The two compiler revisions emit byte-identical bodies. The
  compiler lane localises it to `findOptimalMemOpLowering`'s scalar path, and claim-auditor is checking that.
