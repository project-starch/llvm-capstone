# PostgreSQL 17.5: parent, sibling and successor controls, 2026-11-11

The controls of [`../../controls/README.md`](../../controls/README.md), run again after the
lifetime instructions changed (`CREVOKE` takes the object alone, `CRENEW` exists) and patch 0003 was
converted to them. Same pre-registration as
[`../2026-10-11-controls`](../2026-10-11-controls/README.md), same four controls, same N.
`results.json` holds every session's result lines, addresses, accesses and fault.

**Outcome: every prediction holds, on both arms, in 3 of 3 repetitions**, as on 2026-10-11. The
positive control also holds: `corpus_control('uaf-palloc')` faults with cause 25 in
`corpus_control_read` on protected and completes on release.

| control | protected | release |
|---|---|---|
| parent | completed 3/3: link = D's chunk, header unchanged, 48 B read back, next palloc = A | completed 3/3, same four checks |
| sibling | completed 3/3, B intact after `pfree(A)` | completed 3/3 |
| successor | completed 3/3, C at A's address, read-back passes | completed 3/3 |
| successor-stale | **fault cause 25** in `lc_probe_stale_read` 3/3 (`pgcorpus_lifetime.c:189`, link address 0x4c874) | completed 3/3, the stale read returns C's data |

Differences from the 2026-10-11 record, none in an outcome:

- The stale read faults at `lc_probe_stale_read`+0, where the earlier image faulted at +20: this
  compiler emits the load as the function's first instruction. The probe, the cause and the source
  line are the same.
- On protected the chunk header is 32 bytes and A lies at block+240, as on 2026-10-11. The block
  header again holds two capabilities: the generation and the pointer `malloc` returned for the
  block (see the port README), where the earlier patch held a lifetime and a generation.
- A live chunk's bounds are [A-32, A+64), header plus request, and C is a new child over the same
  range, as before.

## What ran

- **Source.** Branch `one-operand-crevoke-crenew` at 59201bdf2 as a plain copy without `.git`
  (the manifests record `runtime_dirty: true`), with patch 0003 at sha256 `29b94c5a026e…` and
  `controls/build-domain.sh` as in this tree. That copy of the port's build script had not followed
  the port: it compiled `static_modules.o` without the F and D target features the virtual profile
  requires, and `capstone-cc` refused the link. It now passes the same flags as
  `ports/postgres/app/build-domain.sh`. File hashes of the controls are in `results.json`.
- **Builds.** `controls/build-image.sh`, once without and once with `PGSU_NESTED=sublet`, with
  `CAPSTONE_SDK_CORPUS=1` and `CAPSTONE_SDK_EXTRA_CODEGEN=-fno-builtin` so that the relink SDK
  records the corpus setting the runner's gate reads. Compiler sha256 `0bbb1f88bc90…`, the pinned
  17.5 archive `fcb7ab38…`. Both images link 7 modules with 502 functions.

| | release | protected |
|---|---|---|
| image sha256 | 4684708a84ff9dde808a518f20d12fea37876210c40784e098fadcadb6fc8c06 | a94b8c21c090cb257dbbb4cdcef4b90a4b10e46a9dd19426a19580bf301c6231 |
| manifest `nested` | none | postgres |
| one boot, 14 sessions | 495 s | 517 s |

- **Platform.** The virtual kit built from this tree: QEMU `2d5a7107890e…` (the model with the
  one-operand `CREVOKE` and `CRENEW`), launcher `0f8e0f903725…`, module `e47bca62addf…`, Image
  `48eb725a40e4…`, fw_jump `5b5c047a0030…`, rootfs `62049800ed0f…`. Machine: `virt-capstone`, one
  hart, 2 GiB, U-mode, pinned mode with the module's defaults, `-cpu` extras
  `x-capstone-test-csrs=true,x-capstone-exact-bounds=true`.
- **Inputs.** The kit fixture cluster, tree `0353a5b9…`, the tree the 2026-10-11 run used; the share
  is the fixture's `install/share/postgresql` plus each image's own staged extension files. Each
  session ran on a fresh `cp -a` of the cluster as `nobody`.
- **N.** 3 repetitions per control per arm, each in its own backend session, in one boot per arm.
  The two boots ran side by side, each under its own QEMU lock file.

Paths in `results.json` are written with `~` in place of the build account's home directory;
nothing else in the runner's records was edited.
