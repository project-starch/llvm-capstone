# SQLite 3.22.0: protected memsys5 gives the same answers (2026-10-11)

The SQLite row of the pre-registration
[`docs/plans/2026-10-11-protected-same-answers.md`](../../../../../docs/plans/2026-10-11-protected-same-answers.md)
(9bc6f9adf1c9): speedtest1 `main --size 1`, the row count and FNV-1a hash of each of its 32
phases, on three arms.

- **native**: `build-native.sh`, the release as distributed with SQLite's default allocator, on the host.
- **stock**: virtual Capstone. The engine is built with `SQLITE_ENABLE_MEMSYS5`, and speedtest1
  configures a memsys5 heap with `--heap`.
- **protected**: the same, with `../../../sublet/sublet-3220000-memsys5.patch` (memsys5 blocks and
  lookaside slots as CDERIVE/CREVOKE lifetimes) applied to the engine.

**Outcome: as predicted.** With an 8 MiB heap, stock and protected both exit 0 and give the same
row count and hash as native on all 32 phases. Neither arm logs a domain fault.

## Wiring

Before this run, the protected memsys5 existed only in the corpus builder,
`repro322/build-virtual.py --sublet`. `build-domain.sh` now takes `SQLITE_HEAP`:
- `memsys5` compiles a copy of the prepared amalgamation, `src/sqlite3-capstone.c`, with
  `-DSQLITE_ENABLE_MEMSYS5`;
- `memsys5-sublet` makes the same copy and applies the patch with `patch -s -F0 -p1`, as
  build-virtual.py does.

Both arms compile with the same flags (`-O2`, `HAVE_MALLOC_H`, `HAVE_MALLOC_USABLE_SIZE`,
`SQLITE_ENABLE_MEMSYS5`, `-I capstone/runtime/include`), so their engine sources differ only by
the patch.
- The stock copy is byte-identical to the prepared `sqlite3.c` (`7fc92e19...`, the patch's
  `Upstream-SHA256`).
- `shell.c` and `speedtest1.c` are unchanged.
- The heap is configured at run time through speedtest1's own `--heap <bytes> <min>` option, which
  `malloc`s the heap from the virtual runtime (musl mallocng). No source change was needed.

Checks that the patch is live in the protected image, from `llvm-objdump -d` of `speedtest1.dom`:
- protected: 14 CDERIVE (`.insn r 0x5b,1,0x51`) and 5 CREVOKE (`0x52`);
- stock: none.

That memsys5 is the allocator in both Capstone arms is shown by the 1 MiB boot below. Both arms
run out of memory at the same phase as native memsys5, which the platform allocator would not do.

## Heap size: a deviation from the instruction

The corpus heap is 1 MiB with a 64-byte minimum (`SQLITE_HEAP_SIZE` in
`repro322/repro322_common.h`, used by build-virtual.py), and it was run first. That run went:
- Native speedtest1 built with memsys5 over that heap runs out of memory at phase 200
  (`VACUUM`, "SQL error: out of memory"), after 13 phases, and exits 1.
- **Both Capstone arms do exactly the same**, in boot 1: 13 phases equal to native memsys5, then
  the same error, exit 1.

That is a capacity limit of the configuration and the same on all three memsys5 arms, not a
difference the patch makes.

Native probes with `--heap N 64` and `--stats` for N = 2, 4, 8, 16 and 32 MiB:
- every size gives all 32 phases equal to native;
- the memsys5 high-water is 1,062,976 bytes, the largest single allocation 120,000 bytes.

The patch costs 4 bytes per atom (`aReq`), so the protected pool has about 6% fewer atoms
(nByte/69 against nByte/65). So that capacity is not a variable between the arms, the measured
configuration is **`--heap 8388608 64`**, about 8x the high-water, in boot 2. The pre-registration
names no heap size.

This took two boots, not the one planned: boot 1 at 1 MiB (both arms), then boot 2 at 8 MiB (both
arms). The images did not change between them.

## Platform

The pre-registered platform: Sublet-lifetime QEMU, launcher and module as in the 2026-10-11 corpus
runs; the pre-registration's af37cc32, 3975314a and a1c6cb6b are the sha256 prefixes below. One
hart, `x-capstone-exact-bounds=true` (run-staged.py's default), `-m 2G`.

| file | sha256 |
|---|---|
| stock/speedtest1.dom | `385a7fbb38fcbbd6d2558e9491b4d5e0609ef658132853c0ee280c53c18a37a0` |
| protected/speedtest1.dom | `9981f0bef468e8002dcb021ee65be2cd75ad414673532768c47a46c41760d405` |
| stock/sqlite3.dom | `e7d426ee261b89327bd0118c5449aacc5a51b2b4104b5eb3c180691d47d3f1b8` |
| protected/sqlite3.dom | `4410cf28edbb50fa6a123901e0035b8b3adc9dc417f87d6464bccbd83efec478` |
| stock/src/sqlite3-capstone.c | `7fc92e19a3af61f3961b83447a91f1e1dec06646e343037dfd96a397167700b2` |
| protected/src/sqlite3-capstone.c | `b297fc3c83c69af8cd9409ff5c3faec230b4c9b719eadb3984465099782cbb40` |
| prepared sqlite3.c | `7fc92e19a3af61f3961b83447a91f1e1dec06646e343037dfd96a397167700b2` |
| prepared speedtest1.c | `16750ede41d01935a1f423298b7676027048ddc0ce7951cc828b9f1760407291` |
| sublet-3220000-memsys5.patch | `3f16cf52933c20a93cbc860e90bd42f388710fc4c60f407227146ab60b5f9dd2` |
| native/speedtest1 | `19a766546cd1a1b6aaceb71b430559fedee0c7ef323a90b320b180ab2086d1be` |
| qemu-system-riscv64 | `af37cc3269d1cfb47a72eb79cc3dfd2397848c613ca703e43625c68ab50ad27d` |
| capstone-vexec | `3975314a7a6da119c4bff6a0ab0c601fcd96d305f59a0a68fe5b36b9bd544ad4` |
| capstone_vm.ko | `a1c6cb6b79a98cf1f09e55a37adf2320e2b9787922b7e10bf4088c27e396e512` |
| Image | `e833dfb12bccf8a39bd6a81e8e42e4622f0e76bcc6cc09624a1fbe7198c18a4b` |
| fw_jump.elf | `44cd6cbc2d225c2fea6c6000ea0b9eab83bc34e769a7c95c036cf17b6fe0f116` |
| rootfs.ext2 | `62049800ed0fa8827959c0060bbd74f0435aacb577dd521e14a14a099d167845` |
| clang | `4c6135feafae20c23edf01a4964d51459d7acb85469cd50166704e1f9a3365d4` |
| libapplication-runtime.a | `4748adbad8826b8d2f3b344d84b5951a0ed6db8bd92885e41f9c6469f62c438f` |
| libc-capstone-virtual.a | `a0f5c573ac5822c7276d31708a2ced5979cd30bb22af501abed5ab97812223a9` |
| gate.sh (8 MiB) | `8be861eb9751ea14691dbecec8100f8a396cde6f8e07851e636a6929524b6667` |
| gate.sh (1 MiB) | `778eec33284e89ee18a4aad737e54b3e298fed9b22a7685af1bbdb7c617bdd96` |

- Source: a snapshot of 9bc6f9adf1c9 with this directory's wiring edits (`app/build-domain.sh`,
  `app/README.md`).
- SDK: `runtime/virtual/build-sdk.sh` (musl 1.2.5 with the port's patches, virtual libc).
- The engine `.c` hashes are of the compiled copies.

## Commands

`$S` is the run directory on the build host and `$K` the platform kit. The environment was
`LD_LIBRARY_PATH=$K/toolchain/hostlib CAPSTONE_LLVM_BUILD_DIR=$K/toolchain CAPSTONE_REPO_ROOT=$S/repo
CAPSTONE_TMP_ROOT=$S/tmp`.

```sh
SQLITE_322_CACHE_ROOT=$S/cache bash capstone/ports/sqlite/app/prepare-sources.sh $S/src
bash capstone/ports/sqlite/app/build-native.sh $S/src $S/native
MUSL=$(MUSL_CACHE_ROOT=$S/tmp/musl-src bash capstone/ports/musl-capstone/prepare-musl-capstone.sh | tail -1)
bash capstone/runtime/virtual/build-sdk.sh $S/vrt "$MUSL"
SQLITE_HEAP=memsys5        bash capstone/ports/sqlite/app/build-domain.sh $S/vrt/sdk $S/src $S/stock
SQLITE_HEAP=memsys5-sublet bash capstone/ports/sqlite/app/build-domain.sh $S/vrt/sdk $S/src $S/protected
# native oracle, in a scratch directory
$S/native/speedtest1 --size 1 st.db
# one boot per heap size: python3 capstone/runtime/virtual/run-staged.py --qemu $K/kit-sublet/qemu-system-riscv64 \
#   --images $K/kit-sublet/images --stage <stage> --work <new dir> --timeout 7200
# gate.sh, per arm in order stock then protected, after insmod capstone_vm.ko:
cd /tmp && rm -f st.db st.db-journal
CAPSTONE_EXEC_DIAGNOSTICS=1 /mnt/vm/capstone-vexec /mnt/vm/speedtest1-$arm.dom --heap 8388608 64 --size 1 st.db
```

A pre-check, not an arm: native speedtest1 built from `sqlite3-stock.c` with
`-DSQLITE_ENABLE_MEMSYS5`, run with the same `--heap`.

## Results

| arm | heap | exit | phases | equal to native | equal to stock | speedtest1 TOTAL | guest wall |
|---|---|---|---|---|---|---|---|
| native | default allocator | 0 | 32 | n/a | n/a | 0.057 s | n/a |
| stock | 8 MiB | 0 | 32 | **yes** | n/a | 19.474 s | 23 s |
| protected | 8 MiB | 0 | 32 | **yes** | **yes** | 19.725 s | 23 s |
| native memsys5 (pre-check) | 8 MiB | 0 | 32 | yes | yes | 0.082 s | n/a |
| stock | 1 MiB | 1 (OOM at 200) | 13 | no | n/a | n/a | 18 s |
| protected | 1 MiB | 1 (OOM at 200) | 13 | no | yes | n/a | 9 s |
| native memsys5 (pre-check) | 1 MiB | 1 (OOM at 200) | 13 | no | yes | n/a | n/a |

How the times were measured:
- **speedtest1 TOTAL** is speedtest1's own total in guest time.
- **guest wall** is the guest's `date +%s` around the launcher call, to whole seconds, launcher
  start included.
- **Boot wall** on the host, including the boot: 52 s at 8 MiB, 35 s at 1 MiB.
- The 1 MiB guest walls (18 s against 9 s) are runs that stop at the out-of-memory error, and are
  not a performance comparison.

After both arms in each boot:
- the launcher reports 0 live domains, regions, bytes and tag pages;
- `capstone_vm` unloads with status 0.

Per phase, 8 MiB heap: rows and hash.

| phase | native | stock | protected | all equal |
|---|---|---|---|---|
| 100 | 0 `cbf29ce484222325` | 0 `cbf29ce484222325` | 0 `cbf29ce484222325` | yes |
| 110 | 0 `cbf29ce484222325` | 0 `cbf29ce484222325` | 0 `cbf29ce484222325` | yes |
| 120 | 0 `cbf29ce484222325` | 0 `cbf29ce484222325` | 0 `cbf29ce484222325` | yes |
| 130 | 25 `86f7643d251c40ef` | 25 `86f7643d251c40ef` | 25 `86f7643d251c40ef` | yes |
| 140 | 10 `0c7b7e747ae91a62` | 10 `0c7b7e747ae91a62` | 10 `0c7b7e747ae91a62` | yes |
| 142 | 1050 `359b7f1ecbeaaff8` | 1050 `359b7f1ecbeaaff8` | 1050 `359b7f1ecbeaaff8` | yes |
| 145 | 100 `a0d680adf09a4a75` | 100 `a0d680adf09a4a75` | 100 `a0d680adf09a4a75` | yes |
| 150 | 0 `cbf29ce484222325` | 0 `cbf29ce484222325` | 0 `cbf29ce484222325` | yes |
| 160 | 100 `df2ea006eed925a5` | 100 `df2ea006eed925a5` | 100 `df2ea006eed925a5` | yes |
| 161 | 100 `24d182f082f52e7c` | 100 `24d182f082f52e7c` | 100 `24d182f082f52e7c` | yes |
| 170 | 100 `7e14902b1948dce8` | 100 `7e14902b1948dce8` | 100 `7e14902b1948dce8` | yes |
| 180 | 0 `cbf29ce484222325` | 0 `cbf29ce484222325` | 0 `cbf29ce484222325` | yes |
| 190 | 0 `cbf29ce484222325` | 0 `cbf29ce484222325` | 0 `cbf29ce484222325` | yes |
| 200 | 0 `cbf29ce484222325` | 0 `cbf29ce484222325` | 0 `cbf29ce484222325` | yes |
| 210 | 1 `78a5f1150135b920` | 1 `78a5f1150135b920` | 1 `78a5f1150135b920` | yes |
| 230 | 0 `cbf29ce484222325` | 0 `cbf29ce484222325` | 0 `cbf29ce484222325` | yes |
| 240 | 0 `cbf29ce484222325` | 0 `cbf29ce484222325` | 0 `cbf29ce484222325` | yes |
| 250 | 0 `cbf29ce484222325` | 0 `cbf29ce484222325` | 0 `cbf29ce484222325` | yes |
| 260 | 1 `0c3a0c3404ffd068` | 1 `0c3a0c3404ffd068` | 1 `0c3a0c3404ffd068` | yes |
| 270 | 0 `cbf29ce484222325` | 0 `cbf29ce484222325` | 0 `cbf29ce484222325` | yes |
| 280 | 0 `cbf29ce484222325` | 0 `cbf29ce484222325` | 0 `cbf29ce484222325` | yes |
| 290 | 0 `cbf29ce484222325` | 0 `cbf29ce484222325` | 0 `cbf29ce484222325` | yes |
| 300 | 0 `cbf29ce484222325` | 0 `cbf29ce484222325` | 0 `cbf29ce484222325` | yes |
| 310 | 900 `5f60bc2cdcb89e99` | 900 `5f60bc2cdcb89e99` | 900 `5f60bc2cdcb89e99` | yes |
| 320 | 1 `9b7e48527050edfe` | 1 `9b7e48527050edfe` | 1 `9b7e48527050edfe` | yes |
| 400 | 0 `cbf29ce484222325` | 0 `cbf29ce484222325` | 0 `cbf29ce484222325` | yes |
| 410 | 700 `649813fe025d510b` | 700 `649813fe025d510b` | 700 `649813fe025d510b` | yes |
| 500 | 0 `cbf29ce484222325` | 0 `cbf29ce484222325` | 0 `cbf29ce484222325` | yes |
| 510 | 700 `7e30b42e2404cc90` | 700 `7e30b42e2404cc90` | 700 `7e30b42e2404cc90` | yes |
| 520 | 512 `07aa5d31c6064a19` | 512 `07aa5d31c6064a19` | 512 `07aa5d31c6064a19` | yes |
| 980 | 1 `b8835f20872ee7fa` | 1 `b8835f20872ee7fa` | 1 `b8835f20872ee7fa` | yes |
| 990 | 0 `cbf29ce484222325` | 0 `cbf29ce484222325` | 0 `cbf29ce484222325` | yes |

Every phase's values for every arm and both heaps are in [`phases.json`](phases.json). N = 1 per arm.
