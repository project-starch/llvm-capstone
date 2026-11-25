# ioctl(FIOCLEX/FIONCLEX), uname and getrandom on the native route: directed test

The native route's `native_supported` list refused all three with `-ENOSYS`, although the runtime's table already
routes each one `NS_FLAT`:
- `ioctl` is `{I, I, I}`, with fd argument 0;
- `uname` is `OUT_FIX(390)`;
- `getrandom` is `OUT_ARG(1)`.

Fig. 10's CPython stopped on two of them:
- **`ioctl(FIOCLEX)`.** CPython's run-file path calls `_Py_set_inheritable`, which does not fall back on ENOSYS.
- **`uname`.** pyperf's `platform.machine()` calls it.

tshark's runtime report lists `getrandom` (278) among the calls the route did not serve.

**The change** (module, three files):
- **`ioctl`** serves FIOCLEX and FIONCLEX as `fcntl(F_SETFD)` on the authorised descriptor, which is what Linux's
  `ioctl_fioclex` does.
  - The request is compared as an unsigned int, because musl may sign-extend it.
  - Every other request still answers `-ENOSYS`, as on the delegated route: a request with a pointer argument
    needs its own marshalling.
- **`uname`** is `newuname`: the task's UTS namespace, copied under `uts_sem`. The personality overrides (UNAME26,
  the compat architecture) are not applied; a native task never sets them.
- **`getrandom`** follows `drivers/char/random.c`:
  - the same flag check;
  - unless GRND_NONBLOCK or GRND_INSECURE is given, it waits for the CRNG, and a signal reads `-EINTR`;
  - then it returns `len` bytes. `len` is bounded by the route's buffer.

**What it checks** (`fu.c`, one line per check, then `FU_DONE PASS|FAIL <n>`):
- **ioctl:**
  - FIOCLEX sets FD_CLOEXEC, read back with `F_GETFD`, and FIONCLEX clears it;
  - a bad fd reads EBADF;
  - a pointer request (TCGETS) is still refused with ENOSYS.
- **uname:** returns 0 with sysname `Linux` and machine `riscv64`.
- **getrandom:**
  - 64 bytes, not all zero;
  - GRND_NONBLOCK gives 16 bytes;
  - an unknown flag reads EINVAL.

**How it ran** (2026-11-25, twin):
- The image was built with arm FRCDKMS's SDK and flags (`perf/fioclex/build-test.sh`, image `cf29d092…`).
- It ran from `/tmp/nt/sr` with `capstone_vm.dom_grant_root=/tmp/nt/sr` and 2^18 records, slot 28.
- **Control kernel:** the twin Image `dcb97386…` of every nested twin run.
- **New kernel:** `e468843a…`, a `cp -a` of that tree with the same edits on its three module files
  (`perf/fioclex/kbuild.sh`, `module.diff`).

| kernel | result |
|---|---|
| control `dcb97386` | `FU_DONE FAIL 11`: FIOCLEX and FIONCLEX refused; the bad fd reads 38, not EBADF; uname and getrandom refused with 38 |
| new `e468843a` | `FU_DONE PASS 0` |

**Also checked:**
- The control's checks that must hold on both kernels did hold on both: the initial F_GETFD, TCGETS refused with 38,
  and the getrandom bad-flag call returning −1.
- **Compile check against the P1 kits' 6.4 tree** (`perf/k2n/linux-board-k2n-native-hpm2tree`):
  - `capstone_vm.o` was compiled with the tree's own recorded command, base (`native-faccessat`) and this branch;
  - both give rc 0 with the same 11 warnings, and the new object references `uts_sem` and the random API;
  - no file of the tree was written.

**Withdrawn on the way:**
- The first `fu.c` printed the `utsname` struct even when `uname` failed.
- On the control kernel the struct kept its 0x5b fill and had no NUL, so the printf read past its bounds. The
  capability check stopped it (cause 28, rc 139) before the getrandom checks ran.
- The print is now conditional and the compares are bounded. Both kernels were run again, as in the table.

**Not done here:**
- No board run.
- No kit carries the change yet. A kit takes it by rebuilding its kernel tree's adapter.
