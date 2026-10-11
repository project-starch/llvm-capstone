# PostgreSQL 17.5: protected build gives the same answers, 2026-10-11

This is the PostgreSQL row of the pre-registration
`capstone/docs/plans/2026-10-11-protected-same-answers.md` (commit 9bc6f9adf1c9). The prediction
was that on `work.sql` in the single-user backend, the protected build (`PGSU_NESTED=sublet`,
memory-contexts patch 0003) gives the same 22 result rows and final count as the stock virtual
build and as native PostgreSQL 17.5.

**Outcome: the prediction holds.** Both Capstone arms exit 0 without a domain fault. Their 22 rows
equal native 17.5 exactly, and so the protected rows equal the stock rows. The final count is
`1500` in all three. The comparison does fail when the reference is altered (see the positive
control below). N = 1 per arm.

| Arm | Image sha256 | Exit | Domain faults | Rows | Final count | Equal to native | Equal to stock |
|---|---|---|---|---|---|---|---|
| native 17.5 (x86_64, 16-byte MAXALIGN) | postgres 3e43659f… | 0 | n/a | 22 | 1500 | (oracle) | yes |
| stock (`PGSU_NESTED` unset, nested=none) | f03bcc1f… | 0 | 0 | 22 | 1500 | yes | (itself) |
| protected (`PGSU_NESTED=sublet`, nested=postgres) | 9c2098f0… | 0 | 0 | 22 | 1500 | yes | yes |

The 22 rows hash to `280000084d6bc8493f3c91444aed0fc7d3d09d0c38e05030deb4ec809d25789c` on every arm
(sha256 over the rows joined with `\n`). `work-compare.py` prints `IDENTICAL: 22 result rows` for
both arms. Apart from the rows, the two guest transcripts differ only in timestamps, the PID and
checkpoint timings, and the `CAPSTONE_VM_STATS` line. The shutdown checkpoint wrote 86 buffers and
ended at `lsn=0/15F3700` in all three runs, native included.

The protected arm did exercise the lifetime machinery. Its per-process statistics show
`nodes=65233 collections=5 reclaimed=53787`, against `nodes=6203 collections=0 reclaimed=0` for
stock. After both runs, `capstone-vexec --stats` reported zero live domains, regions, bytes and tag
pages.

## Positive control

`run-ports.py` has no positive control for PostgreSQL; FFmpeg's flipped input is its only one.
This run uses three altered copies of the native reference instead. In each copy, a different kind
of change was made:

- the final count `1500` changed to `1501`;
- the join row `name-3` changed to `name-4`;
- the last row (`----`) dropped.

The same comparison, with both run-ports' `pg_rows` and `work-compare.py`, then reports "not equal"
for both arms and all three copies. `work-compare.py` exits 1 each time.

## What ran

**Source.** The snapshot is `git archive` of 9bc6f9adf1c9 `capstone/`, plus llvm-capstone
`compiler-rt/lib/builtins` at 9bc6f9adf1c9. On p13 it was committed as `adc27e25d667` ("snapshot of
9bc6f9adf1c9"), and both manifests record that commit as `runtime_revision` and
`recipe_revision`, with `runtime_dirty: false`. Against 9bc6f9adf1c9, the snapshot tree differs
only in the absent submodule gitlinks and in one CPython results `.tsv`, which git normalised from
CRLF. The builtins tree (283d17fc) is identical.

**Builds.** Both builds ran on p13 with `JOBS=24`, the kit toolchain (clang sha256 4c6135fe…) and
the pinned 17.5 archive (fcb7ab38…), using `CAPSTONE_TMP_ROOT` under the run directory:

```
env -u PGSU_NESTED bash capstone/ports/common/application/build-virtual.sh postgres $OUT/stock-clean
PGSU_NESTED=sublet bash capstone/ports/common/application/build-virtual.sh postgres $OUT/protected-clean
```

Both builds applied the port patches 0001-0016. The stock build also applied 0017, the gap
observer. It is applied as source, but `PGSU_GAP_OBSERVER` was not set, so the observer was not
compiled in. The protected build skipped 0017 and applied
`postgresql-17.0-0003-memory-contexts-sublet-lifetimes.patch` (sha256 8b36b576…). Each image links
6 modules with 499 functions. The runtime archive is e5693f5c… and the builtins archive is
a72ed0bf…, the same in both images.

**Native reference.** The committed virtual port results use `pg_reference` sha256
`fee37cc7adaf0c2c4e221ad07d150e5ed7f727b6d76903ee3285776f4e0f5329`. That is the existing reference,
`native-work.stdout` from the 2026-09-28 `build-native16-fixture.py` root. Its manifest records:

- archive fcb7ab38;
- builder 9229ef45;
- postgres 3391b933;
- cluster tree ac378cd8.

The manifest does not record the command that ran `work.sql`, so the native reference was also
produced again in this session. `build-native16-fixture.py` was run unchanged (builder 9229ef45)
on p13: x86_64, gcc 15.2.0, `-O1`, 16-byte MAXALIGN, and the same archive. Then:

```
postgres --single -D <copy> -c shared_buffers=4MB -c max_connections=10 -c timezone=GMT \
  -c log_timezone=GMT -c dynamic_shared_memory_type=sysv -c exit_on_error=true postgres < work.sql
```

This ran once on a copy of the kit's pristine cluster, the cluster the Capstone arms used. It ran
once more on a copy of the fresh cluster from its own initdb. Both stdout files are byte-identical
to the existing reference (fee37cc7…). The rows are compared against the in-session run on the kit
cluster.

**Pristine cluster.** This is the kit fixture cluster: builder tree digest `ac378cd8…`, the
manifest's value, and run-ports `tree_sha` `0353a5b9…`. Its share tree is run-ports `tree_sha`
`9daf98e4…`. These are the same values as in the committed run-ports results.

**One boot.** `run-staged.py` (d6770b8b…) booted with the kit's Sublet-lifetime platform:
`virt-capstone`, one hart, 2 GiB, U-mode and exact bounds (its defaults), and a private QEMU lock.
The stage held:

- `stock/bin/postgres` and `protected/bin/postgres`, each beside its own copy of the share tree;
- one pristine cluster;
- `work.sql` (f4310b75…);
- the launcher and the module.

Each arm got its own `cp -a` of the cluster, so neither saw the other's writes. Each arm's guest
command is run-ports.py's PostgreSQL command, with only the cluster path and the working directory
changed. The gate script (sha256 a6ad0de0…):

```sh
#!/bin/sh
cd /mnt/vm || exit 1
dmesg -n 1
insmod capstone_vm.ko || exit 1
export CAPSTONE_EXEC_DIAGNOSTICS=1 CAPSTONE_VM_STATS=1
chmod 666 /dev/capstone-vm
for arm in stock protected; do
  echo PORT_BEGIN:postgres-$arm
  cp -a pgcluster /tmp/pgcluster-$arm
  chown -R nobody /tmp/pgcluster-$arm
  su nobody -s /bin/sh -c "cd /mnt/vm/$arm; /mnt/vm/capstone-vexec ./bin/postgres --single -D /tmp/pgcluster-$arm -c shared_buffers=4MB -c max_connections=10 -c timezone=GMT -c log_timezone=GMT -c dynamic_shared_memory_type=sysv -c exit_on_error=true postgres < /mnt/vm/work.sql"
  echo PORT_END:postgres-$arm:$?
done
echo PORT_BEGIN:cleanup
./capstone-vexec --stats
rmmod capstone_vm
echo PORT_END:cleanup:$?
echo VIRTUAL_STAGED_DONE
```

**Checks.** Rows are extracted with run-ports.py's `pg_rows`, copied verbatim. An arm is accepted
only if all of these hold:

- its exit status is 0;
- `domain fault cause=` does not appear;
- it has exactly 22 rows, and they equal the native rows.

`su` alone can return 0 after a SIGPROT; the README says so.
`work-compare.py` was run as a second, independent check.

## Platform

| Component | sha256 |
|---|---|
| qemu-system-riscv64 (Sublet-lifetime QEMU) | af37cc3269d1cfb47a72eb79cc3dfd2397848c613ca703e43625c68ab50ad27d |
| capstone-vexec (launcher) | 3975314a7a6da119c4bff6a0ab0c601fcd96d305f59a0a68fe5b36b9bd544ad4 |
| capstone_vm.ko (module) | a1c6cb6b79a98cf1f09e55a37adf2320e2b9787922b7e10bf4088c27e396e512 |
| Image | e833dfb12bccf8a39bd6a81e8e42e4622f0e76bcc6cc09624a1fbe7198c18a4b |
| fw_jump.elf | 44cd6cbc2d225c2fea6c6000ea0b9eab83bc34e769a7c95c036cf17b6fe0f116 |
| rootfs.ext2 | 62049800ed0fa8827959c0060bbd74f0435aacb577dd521e14a14a099d167845 |

These match the platform the pre-registration names: QEMU af37cc32, launcher 3975314a, module
a1c6cb6b.

## Images

| | stock | protected |
|---|---|---|
| sha256 | f03bcc1fc5a912f63a1df0326c2234a67e0f3fb1ba80f1c82e1a443ec5e4f4ba | 9c2098f09cc6d6affc3e391158769cc49e2087afc5305626bebffde3d5bf73a5 |
| manifest `nested` | none | postgres |
| manifest `runtime_revision` | adc27e25d667a5c5702f8ed62c05d9a4ce4908b0 (snapshot of 9bc6f9adf1c9) | same |
| manifest `runtime_dirty` | false | false |

## Wall times (p13, one measurement each)

| Step | Seconds |
|---|---|
| first build: stock / protected / native fixture, run concurrently | 123 / 122 / 61 |
| clean rebuild: stock / protected, run concurrently | 125 / 125 |
| one boot, both arms, from QEMU start to `VIRTUAL_STAGED_DONE` | 89 |
| guest `elapsed_ns`: stock / protected | 29.0 / 31.2 |
| native `work.sql` | 0.14 |

The guest times are one sample each, taken from `CAPSTONE_VM_STATS`. They are not a performance
measurement.

## Deviations from the plan

- **Builds ran twice.** The first two builds recorded `runtime_dirty: true`. The cause was a
  `__pycache__` directory that a hashing probe from this session left in the snapshot tree while the builds ran; it
  was not a source change. It was removed, the tree was confirmed clean, and both images were rebuilt
  into fresh directories. The rebuilt images are byte-identical to the first ones (f03bcc1f… and
  9c2098f0…) and record `runtime_dirty: false`. The run used the rebuilt images.
- **Native reference also reproduced.** The existing reference was used, and it was also produced
  again in this session, because its `work.sql` command was not recorded. Both are byte-identical.
- **Stage layout.** Each arm has its own `bin/` and `share/` under `/mnt/vm/<arm>`, so that two
  images can run in one boot. run-ports.py uses one `/mnt/vm/bin/postgres`.

`result.json` holds the per-row hashes, the verdicts, the controls and all the input hashes.
