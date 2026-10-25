# tshark: protected wmem gives the same answers (2026-10-11)

Pre-registered in `capstone/docs/plans/2026-10-11-protected-same-answers.md` (9bc6f9adf1c9,
pushed before any build or boot). Wireshark row: wmem patch 0001 (block, block_fast), the port's
capture set with `-V -n`, stdout byte for byte against native stock tshark.

**Verdict: as predicted.** On all ten captures, the stock virtual image and the protected image
(`TSAPP_SUBLET=1`) each print stdout byte-identical to native stock tshark. The exception is ntp,
which must differ as the negative control, and differs identically on both arms. Each arm's
stderr is byte-identical to the native minimal build's. Every run exits 0, and no run faults.
Protected equals stock on every capture: same stdout hash, stderr hash and exit status. N = 1 per
arm, both arms in one boot.

## Wiring (uncommitted in this worktree; the run used exactly it)

Before this change, patch 0001 was applied only in the wmem component build (`ports/wireshark/wmem`,
`WM_SUBLET=ON`), and the tshark application ran wmem unprotected.

- `app/host/cross-build.sh`: `TSAPP_SUBLET=0|1` (default 0). With 1, it requires
  `CAPSTONE_APPLICATION_PROFILE=virtual`. After this port's own patches it applies
  `../wmem/patches/wireshark-4.6.8-0001-wmem-sublet-lifetimes.patch` with
  `patch --batch --forward --fuzz=0` (any failure stops the build), and appends
  `-I$CAPSTONE_REPO_ROOT/capstone/runtime/include` to `CMAKE_C_FLAGS` for `capstone/capability.h`.
- `common/application/build-virtual.sh` (tshark branch): `TSAPP_SUBLET=1` records `NESTED=wireshark`.
- `common/application/build.py`: `wireshark` was added to `--nested` and to the "no objects, no
  regions" tuple (as FFmpeg's 265e4eb47677 did). The program check now accepts the nested name
  that belongs to the app (`tshark` -> `wireshark`): with `--app tshark --nested wireshark` the old
  `app != nested` check would raise.
- `app/README.md`: one paragraph on `TSAPP_SUBLET=1`.

**Fuzz-0 check.** Done locally on the pinned tarball (sha256 `c0f1ccf2...79d0`, verified):
- the six app patches 0001-0006 each apply at `--fuzz=0`;
- then wmem 0001 applies at `--fuzz=0` with no offsets (`wmem_allocator.h`,
  `wmem_allocator_block.c`, `wmem_allocator_block_fast.c`).

The build used the same check. Seven upstream units include `wmem_allocator.h`, all of them
inside `wsutil/wmem`.

**The protection is in the protected image only:**
- the protected `xsrc/wsutil/wmem/wmem_allocator.h` includes `capstone/capability.h`, and its
  `CMAKE_C_FLAGS` carries `runtime/include`; stock has neither;
- `llvm-nm` finds 7 `*sublet*` wmem symbols in the protected `libwsutil.a` and 0 in stock's;
- the protected `tshark.dom` defines `wmem_block_sublet_*` and `wmem_block_fast_sublet_*`, and
  stock's does not;
- the disassembly of the `wmem_allocator_block*.o` objects shows 41 undecoded custom-opcode
  lines (the `.insn` CDERIVE/CREVOKE encodings) in protected and 0 in stock.

This run did not measure that the wrappers execute at run time.

## Inputs and platform

- **Source:** p12 snapshot commit `2cfdee027f83`. It is the `capstone/` tree of 9bc6f9adf1c9,
  plus the four wired files above (their local diff's sha256 is `72875b38...b82a`), plus
  `compiler-rt/lib/builtins` from 9bc6f9adf1c9. Both manifests: `runtime_dirty: false`.
- **Compiler:** the kit's clang 22 (`cda7e335ff9f`), sha256 `4c6135fe...65d4`.
- **Platform** (kit-sublet, sha256):

  | part | sha256 |
  |---|---|
  | qemu | `af37cc3269d1cfb47a72eb79cc3dfd2397848c613ca703e43625c68ab50ad27d` |
  | kernel | `e833dfb12bccf8a39bd6a81e8e42e4622f0e76bcc6cc09624a1fbe7198c18a4b` |
  | firmware | `44cd6cbc2d225c2fea6c6000ea0b9eab83bc34e769a7c95c036cf17b6fe0f116` |
  | rootfs | `62049800ed0fa8827959c0060bbd74f0435aacb577dd521e14a14a099d167845` |
  | launcher | `3975314a7a6da119c4bff6a0ab0c601fcd96d305f59a0a68fe5b36b9bd544ad4` |
  | module | `a1c6cb6b79a98cf1f09e55a37adf2320e2b9787922b7e10bf4088c27e396e512` |

  QEMU 8.1.1, one hart, exact bounds on (`run-staged.py` default), 2 GiB.

- **Images** (`build-virtual.sh tshark`, manifest fields):

  | arm | TSAPP_SUBLET | nested | image sha256 |
  |---|---|---|---|
  | stock | unset | none | `9fd099f23376ab05acd3a68c380b6259a52ea95a76d0a9527fe3cb8be5dfec2e` |
  | protected | 1 | wireshark | `be27d346309a9a1e1b460f69b51703a17b1996eb5444e9aa4f425740d2ab6d36` |

  Both share these manifest fields:
  - profile and heap `virtual`;
  - `recipe_revision` and `runtime_revision` `2cfdee027f83`;
  - runtime archive `e5693f5c...e545`, builtins archive `a72ed0bf...f56e`;
  - stack 1 MiB.

- **Native oracle** (built on p12 from the same verified tarball, Release, the option set of the
  earlier `native-stock` build: tools off except tshark; Lua, pcap, cap, GnuTLS, netlink,
  minizip-ng and xxhash off; libraries found: GLib 2.88, libgcrypt 1.12.0, c-ares 1.34.6, PCRE2
  10.46, libxml2 2.15.2, zlib):
  - stock tshark (stdout oracle), sha256 `65429f75...daaa`;
  - native minimal `native-min-pa` (stderr reference): app patches 0001-0006,
    `capstone-stubs.c`, the dissector whitelist, `-DCAPSTONE_HF_PREALLOC=4096`, sha256
    `88cb507b...eae1`.

  Controls on the oracle itself:
  - `host/oracle.sh` native-min-pa vs native stock: ORACLE PASS;
  - stock against itself: ORACLE FAIL, on ntp only, as it must;
  - the earlier native stock on the local host (sha256 `6424476e...`, built 2026-09 with more
    optional libraries) prints the same stdout hashes on all six original captures.

## Commands

```sh
# p12: environment for both builds (build-arm.sh)
export LD_LIBRARY_PATH=$K/toolchain/hostlib CAPSTONE_LLVM_BUILD_DIR=$K/toolchain \
       CAPSTONE_REPO_ROOT=~/sa-ws/repo CAPSTONE_TMP_ROOT=~/sa-ws/tmp JOBS=36 \
       PATH=~/sa-ws/hostcc:$PATH TZDIR=~/sa-ws/zoneinfo-2404        # see Deviations
bash capstone/ports/common/application/build-virtual.sh tshark ~/sa-ws/build-stock
TSAPP_SUBLET=1 bash capstone/ports/common/application/build-virtual.sh tshark ~/sa-ws/build-protected
# one boot, both arms (runner in this directory, sha256 a106e3a8...b442)
LD_LIBRARY_PATH=$K/toolchain/hostlib CAPSTONE_QEMU_LOCK=~/sa-ws/qemu.lock \
python3 run-same-answers.py --kit $K/kit-sublet --repo ~/sa-ws/repo \
  --stock ~/sa-ws/build-stock/image --protected ~/sa-ws/build-protected/image \
  --native-stock <native-stock>/run/tshark --native-minimal <native-min-pa>/run/tshark \
  --captures <pristine 4.6.8>/test/captures --work ~/sa-ws/boot-1
```

**How the runner works:**
- **Staging:** the captures are staged on a read-only disk. A flip is a capture with byte
  len-8 xor 1, as `host/run.py` makes it.
- **Gate:** the gate (sha256 `91ca9d29...3c61`) runs `capstone-vexec ./<arm>.dom -r C -V -n`
  with `TZ=UTC`, `HOME` and `WIRESHARK_CONFIG_DIR` set to a fresh empty directory, and
  `CAPSTONE_EXEC_DIAGNOSTICS=1`. It runs the stock arm's ten captures first, then the protected
  arm's.
- **Guest report:** for each run, the guest prints the exit status, and the sha256 and length of
  stdout and of stderr. It drops `TSAPP-HEAP` lines from stderr first, which is `host/run.py`'s
  rule; the build.py image links no `tsapp-heap.o`, so there were none. It also prints the
  result of `cmp` against the native references staged beside the captures, and any launcher
  `domain fault` line.
- **Judging:** the host judges both the hash and the guest's `cmp`.

## Results

Every run exits 0 and reports no fault. Every stderr is 10,398 bytes,
sha256 `ebd622b6...18a8`, equal to native minimal's.

| capture | native stock stdout (sha256, bytes) | stock arm | protected arm | protected = stock |
|---|---|---|---|---|
| dhcp | `df1713eb...06cf`, 15,645 | MATCH | MATCH | yes |
| dns_port | `c74b3491...90b8`, 27,319 | MATCH | MATCH | yes |
| http | `8b55e1c9...495dac`, 4,301 | MATCH | MATCH | yes |
| arp | `d04154a7...52df`, 1,598 | MATCH | MATCH | yes |
| dhcp.flip | `891f584d...60bb`, 15,645 | MATCH | MATCH | yes |
| dns_port.flip | `37e34f91...4747`, 27,318 | MATCH | MATCH | yes |
| http.flip | `73c486b3...3fbd3`, 4,301 | MATCH | MATCH | yes |
| arp.flip | `54ee4b92...aa87`, 1,598 | MATCH | MATCH | yes |
| ntp (must differ) | `f6508560...3418`, 3,128 | DIFFERS (`336f591d...6623`, 2,729) | DIFFERS (same) | yes |
| dns-ooo | `8e6d3060...87c3`, 18,516 | MATCH | MATCH | yes |

**Controls:**
- **Flips:** each flip changes native stock's stdout from the original's (2 diff lines each), so
  the comparison is input-sensitive.
- **ntp:** both Capstone arms print exactly native minimal's stdout (`336f591d...`), which is what
  the whitelist predicts: NTP is not a whitelisted dissector.
- **Per-run JSON:** every per-run hash, length, `cmp` result and verdict is in `verdicts.json`.

## Wall times (p12, 80 cores, shared with another lane's CPython builds during the image builds)

| step | time |
|---|---|
| native stock tshark (configure + build, -j32) | 72 s |
| native minimal tshark (in parallel with stock) | 35 s |
| stock image, `build-virtual.sh` end to end (7 deps with their native gates, cross build, `build-domain.sh`, `build.py`) | 1,369 s |
| protected image, the same, run in parallel with stock | 1,374 s |
| one boot, both arms: staging, native references, 20 tshark runs | 903 s |

The dependency builds were not shared: `build-virtual.sh` requires a fresh output directory and
builds the dependencies into it. So they ran twice, in parallel.

## Deviations and findings about the host, none about the program

- **Host packages:** p12 lacked the host tools the recipes call. These packages were installed
  with apt: `cmake ninja-build clang meson flex bison libc-ares-dev libxml2-dev python3-packaging
  tzdata-legacy`. `/usr/bin/clang` is clang 21, which builds only lemon.
- **Compiler wrapper:** p12's gcc 15 makes implicit function declarations an error in every C
  mode. Because of that, libgpg-error's native gate failed (3 of 13 tests built: `t-lock.c` calls
  `time()` without `<time.h>`). The recorded gates ran under gcc 13. The builds put a `gcc`/`cc`
  wrapper (`/usr/bin/gcc -std=gnu17 -fpermissive`) first on `PATH`. That affects only host-side
  compiles; the cross side uses `capstone-cc`. The gate then passed 13 of 13.
- **Timezone data:** GLib 2.80.5's native suite failed `/GTimeZone/posix-parse`.
  - Ubuntu 26.04's tzdata 2026c makes `PST8PDT` a link to America/Los_Angeles, whose
    abbreviation at t=0 is LMT; the test expects PST.
  - The recording host's 24.04 package of the same 2026c release ships `PST8PDT` as its own zone.
  - The builds set `TZDIR` to a copy of that host's `/usr/share/zoneinfo`. GLib then reported
    133 ok, as recorded.
  - The images do not read zoneinfo, and every tshark run sets `TZ=UTC`.
- **Failed attempts:** two build attempts failed on those two gates before any cross build. Their
  directories were moved aside, and the reported images come from fresh directories.

Files here:
- `verdicts.json`, the runner's full record;
- `run-same-answers.py`, the runner.

No serial log is kept.
