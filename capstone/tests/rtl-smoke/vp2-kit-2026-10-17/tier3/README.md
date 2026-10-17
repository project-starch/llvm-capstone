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
