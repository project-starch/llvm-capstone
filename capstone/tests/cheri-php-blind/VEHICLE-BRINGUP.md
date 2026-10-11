# Bringing up the CHERI vehicle on a host with no root access

`capstone/tests/cheri-baseline/provision-cheri-vehicle.sh` is the verified
recipe and is unchanged. Its preflight assumes a few host tools are already
installed, and on a host where they are not — and where there is no `sudo` —
each one is supplied the same way the script already supplies `makefs`:
`apt-get download` + `dpkg-deb -x` into a local prefix. `vehicle-env-no-sudo.sh`
points the toolchain at that prefix.

```bash
bash capstone/tests/cheri-php-blind/fetch-host-deps.sh     # no root needed
. capstone/tests/cheri-php-blind/vehicle-env-no-sudo.sh
CHERI_ROOT=$HOME/cheri CHERI_WS=$HOME/cheri-ws JOBS=6 \
  bash capstone/tests/cheri-baseline/provision-cheri-vehicle.sh
```

`fetch-host-deps.sh` ends in a verification block that compiles and links against
every dependency, and exits non-zero if any is unusable. Satisfying
`pkg-config --modversion` is **not** sufficient — see (4) below.

## What had to be supplied, and why

| package | needed by | symptom if absent |
|---|---|---|
| `clang-19`, `libclang-rt-19-dev` | cheribuild's host compiler | the script's `host-cc-wrappers` loop probes `/usr/bin/clang` and `continue`s when absent, leaving `--clang-path` pointing at a file that does not exist |
| `lld-19` | host linker | `clang does not support -fuse-ld=lld, using slower bfd instead` — BFD linking LLVM is slow and memory-hungry |
| `libtool` | the QEMU build | `Fatal error: Dependency for qemu missing: Required program libtoolize is missing!` |
| `libattr1-dev`, `libcap-ng-dev` | QEMU 9pfs | cheribuild's `check_required_pkg_config` for `libattr` / `libcap-ng` |
| `libslirp-dev`, `libslirp0` | QEMU user networking | `Required pkg-config file for slirp is missing` |
| `makefs` | disk-image assembly | the script fetches this one itself |

`pixman-1` and `glib-2.0` were already present on this host.

## The one thing that stops the vehicle booting

**`-cpu rv64,sstc=false` is required**, and neither `provision-cheri-vehicle.sh`
nor `cheribuild run-riscv64-purecap` passes it. Without it the purecap kernel
panics early in boot:

```
--- exception 2, tval = 0x14d51073
panic: ...                       KDB: enter: panic
db>
```

`0x14d51073` is a CSR access to `0x14D` = `stimecmp`, the Sstc extension.
CheriBSD 26.07's kernel uses Sstc; QEMU 7.2.0 implements it and advertises it,
but S-mode access to `stimecmp` requires M-mode firmware to set `menvcfg.STCE`,
and the bbl purecap firmware does not — so the kernel detects Sstc, writes
`stimecmp`, and takes an illegal instruction inside `DELAY()` /
`cpu_idleclock()`. On the console this looks like a hang.

Turning Sstc off makes QEMU stop advertising it and the kernel falls back to the
SBI timer, after which it boots to multi-user normally. This is **version skew**
— the published kernel (July 2026) is newer than this QEMU fork (7.2.0) — not a
misconfiguration. CHERI is unaffected: purecap kernel and purecap userland both
run, which `cheri_status` verifies in-guest. `make-qemu-argv.sh` sets it, and
`QEMU_CPU` overrides it.

Pinning an older `REL=` to match the QEMU fork would be the alternative; keeping
the current release and disabling one timer extension is the smaller change.

## Six details that are easy to lose

1. **Pre-create the compiler wrappers.** The script writes
   `$CHERI_ROOT/host-cc-wrappers/{clang,clang++}` only if `/usr/bin/clang`
   exists, and `continue`s otherwise *without* overwriting — so wrappers placed
   there beforehand survive, and no edit to the script is needed. Put `ld.lld`
   beside them or cheribuild falls back to BFD.
2. **`libtoolize` bakes in its data directories.** `_lt_pkgdatadir` is the
   documented override, but it expects `build-aux/ libltdl/ m4/`, which is *not*
   Debian's layout: Debian uses `/usr/share/libtool` for `pkgltdldir`,
   `/usr/share/aclocal` for the macros, and ships **no `libltdl` tree at all**.
   `vehicle-env-no-sudo.sh` points at a symlink shim onto Debian's layout.
   Separately, `build-aux/config.{guess,sub}` are *relative* symlinks
   (`../../misc/...`) and therefore dangle inside a relocated prefix; they are
   linked back to the host copies.
3. **`JOBS` defaults to `nproc - 8`**, which clamps to **1** on an 8-core host.
   Set it explicitly. CHERI-LLVM took ~66 minutes at `JOBS=6` here.
4. **`pkg-config` files bake in `prefix=/usr`**, so a relocated package makes
   them emit `-I/usr/include/...` and `-L/usr/lib/...` paths that do not contain
   it. `pkg-config --modversion` still succeeds, so cheribuild's dependency
   check passes and the build then fails much later with
   `'libslirp.h' file not found`. `CPATH` does **not** rescue this: meson takes
   its `-I` flags from `pkg-config`, not from the environment. `fetch-host-deps.sh`
   repoints `prefix=` in `slirp.pc`, `libattr.pc` and `libcap-ng.pc`, and then
   *compiles and links* a probe against all three rather than trusting
   `--modversion`.
5. **meson caches its configure results.** After fixing the `.pc` files the build
   failed identically, because `qemu-build/build.ninja` still held the old
   `-I/usr/include/slirp` and ninja reported `no work to do` for the
   reconfigure. cheribuild's `-f` does not force a meson reconfigure. Delete
   `$CHERI_ROOT/build/qemu-build` to make it re-run.
6. **Never re-extract the toolchain while a build is in flight.** `dpkg-deb -x`
   overwrites `clang` in place, and doing so while cheribuild is executing it
   fails the build with `exec: .../clang: Text file busy` — which reads like a
   compiler bug and is not one. It killed the QEMU build at object 405 of 5212.
   `fetch-host-deps.sh` now refuses to run while a `cheribuild.py` process
   exists.

`libslirp.so.0` exists only in the local prefix, so the QEMU built against it has
no rpath and **will not start** without `LD_LIBRARY_PATH`. Anything that
*executes* `qemu-system-riscv64cheri` — including `cheri-run.py` — must source
`vehicle-env-no-sudo.sh`, not just the build.

The full dependency list is worth taking from cheribuild rather than discovering
one failure at a time:

```bash
grep -nE 'check_required_pkg_config\(|check_required_system_tool\(' \
  $CHERI_WS/cheribuild/pycheribuild/projects/build_qemu.py
```

LTO for QEMU stays off: it needs `llvm-{ar,ranlib,nm}` beside the wrapper clang,
and on a 31 GB host an LTO link of QEMU is a worse trade than a marginally
faster emulator.

## A non-problem worth not chasing

The QEMU target is renamed `riscv64cheri-softmmu` → `riscv64xcheri-softmmu`
(`build_qemu.py:427`), so the emulator installs as `qemu-system-riscv64xcheri`
while `provision-cheri-vehicle.sh` refers to `qemu-system-riscv64cheri`. That
looks like it would emit an argv naming a nonexistent binary, but **cheribuild
also installs a compat symlink** `qemu-system-riscv64cheri ->
qemu-system-riscv64xcheri`, so the script's argv does work. `make-qemu-argv.sh`
prefers the real x-form and asserts each artifact exists, which is still the more
robust path, but the naming difference is not a bug to fix.

## Vehicle identity

Record this with any result; it is the configuration-reality evidence.

| component | version |
|---|---|
| CheriBSD kernel + purecap world | FreeBSD 15.0-CURRENT `releng/26.07-88f39900c329`, `CHERI-PURECAP-QEMU` (downloaded dist sets, not built) |
| revocation knobs | `runtime_revocation_default`, `..._every_free_default`, `..._async` — all present, verified by `preflight-kernel.sh` |
| host compiler | Debian clang 19.1.7 |
| CHERI-LLVM | clang 17.0.0 (CTSRD-CHERI), built from source, ~66 min at `JOBS=6` |
| CHERI-QEMU | 7.2.0 (CHERI-Alliance fork), `qemu-system-riscv64xcheri` |
| firmware | `bbl-riscv64cheri-virt-fw_jump.bin` (built; stock OpenSBI is not CHERI-aware and yields no console output at all) |
| sysroot | the downloaded rootfs itself — no CheriBSD world is built, so `$SDK/sysroot-riscv64-purecap` does not exist |
| purecap triple | `--target=riscv64-unknown-freebsd -march=rv64gcxcheri -mabi=l64pc128d` |
| required boot flag | `-cpu rv64,sstc=false` (see above) |

CHERI-LLVM and CHERI-QEMU versions come from the provisioning log; fill them in
from `$CHERI_ROOT/logs/` once the run completes.
