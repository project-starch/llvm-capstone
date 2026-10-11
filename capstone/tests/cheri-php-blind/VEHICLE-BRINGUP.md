# Bringing up the CHERI vehicle on a host with no root access

`capstone/tests/cheri-baseline/provision-cheri-vehicle.sh` is the verified
recipe and is unchanged. Its preflight assumes a few host tools are already
installed, and on a host where they are not — and where there is no `sudo` —
each one is supplied the same way the script already supplies `makefs`:
`apt-get download` + `dpkg-deb -x` into a local prefix. `vehicle-env-no-sudo.sh`
points the toolchain at that prefix.

```bash
. capstone/tests/cheri-php-blind/vehicle-env-no-sudo.sh
CHERI_ROOT=$HOME/cheri CHERI_WS=$HOME/cheri-ws JOBS=6 \
  bash capstone/tests/cheri-baseline/provision-cheri-vehicle.sh
```

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

## Three details that are easy to lose

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

## Vehicle identity

Record this with any result; it is the configuration-reality evidence.

| component | version |
|---|---|
| CheriBSD kernel + purecap world | FreeBSD 15.0-CURRENT `releng/26.07-88f39900c329`, `CHERI-PURECAP-QEMU` (downloaded dist sets, not built) |
| revocation knobs | `runtime_revocation_default`, `..._every_free_default`, `..._async` — all present, verified by `preflight-kernel.sh` |
| host compiler | Debian clang 19.1.7 |

CHERI-LLVM and CHERI-QEMU versions come from the provisioning log; fill them in
from `$CHERI_ROOT/logs/` once the run completes.
