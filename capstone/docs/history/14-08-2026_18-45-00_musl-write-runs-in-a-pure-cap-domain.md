# A POSIX program runs against musl inside a pure-capability domain, 2026-08-14

**Result, QEMU, verified.** `write()` from musl, called by ordinary C in a
`capstone64-unknown-elf` domain, is serviced by the host and RETURNS.

```
S1: hostcall.c direct write ok
musl-hello: write #1 through musl in a domain
S2: musl write() RETURNED
musl-hello: write #2, so write() RETURNED
__CAPSTONE_HOSTCALL_HOST_DONE__ status=0 serviced=4
```

Harness: `capstone/musl-capstone/musl-hello/run-musl-hello.sh`. The program is
`#include <unistd.h>` and two `write()` calls; nothing in it is Capstone-aware.
The path is musl `write()` → `__syscall_cp` → `__capstone_hostcall` →
`__capstone_yield` → host → real `write(2)`.

`S2` is the load-bearing marker: one write only proves the outward path, and it
can only be reached by RETURNING from the first.

## Three defects found on the way, in the order they were peeled

**1. `syscall_arg_t` must be `void *`.** It was `__UINTPTR_TYPE__`, on the
CHERI-shaped assumption that uintptr_t is capability-width. Measured on this
target: `sizeof(void*) == 16`, `sizeof(__UINTPTR_TYPE__) == 8`, and neither
`__uintcap_t` nor `__intcap_t` exists. **There is no capability-carrying integer
type here; the only type that carries a capability is a pointer.** The symptom
was musl's `write` emitting `mv a2, a1` (integer move) for the buffer while
using `movc` for the others, so the tag was stripped and the domain faulted in
`helper_cscincoffset` when the buffer was indexed.

Cost: 1270 → 1242 compilable files (93.3 % → 91.3 %), because musl does
arithmetic on `syscall_arg_t` in places. The survey baseline was lowered
deliberately and the reason recorded in the file. It also needs
`-Wno-int-conversion`: int→pointer is an *error* in current clang, not a
warning, and this ABI requires it — integer arguments ride in the cursor of an
untagged capability. Without that flag 497 of 1361 files fail on that
diagnostic alone.

**2. `tp` was never set, so errno could not exist.** `__errno_location` compiles
to `mv a0, tp; addi a0, a0, -0x124`, and the shared entry glue zeroes `tp`. Any
failing syscall therefore faulted inside `__syscall_ret` instead of returning
-1, hiding the real error behind a capability fault. A domain has no threads, so
`tp` does not need a TLS image — it needs addressable space below it. 1024 bytes
with `tp` in the middle, derived from `gp`, and carried across the yield.

**3. C-19: a call in return position loses its epilogue.** See ISSUES.md. This
was the one that mattered: `write()` performed its write correctly and the
caller then fell through into its own `default:` case and returned `-ENOSYS`.
`-fno-optimize-sibling-calls` works around it; `build-sqlite-capstone.sh`
already carried the flag, undocumented, so any domain built without it is
exposed.

## Two instrument failures, both mine, both worth the space

**A diagnostic that becomes the defect is worse than none.** The "which syscall
was refused" reporter was written twice and broke twice: first a 32-byte buffer
taking 34 bytes (the capability bounds caught it on the exact byte —
`bounds = (…e60, …e80)`, `addr = …e80`), then a degenerate SHRINK. It was
deleted rather than fixed a third time: the value now travels as a NUMBER in the
shared metadata and the host does the printing, so the domain formats nothing.

**Filtering by line drops data that shares a line with noise.** Kernel printk
and console output interleave, so `grep -v "remote fence"` discarded init
messages that happened to share a line with a fence message, and made two boots
look like they stalled at different places. Stripping the noise as a SUBSTRING
showed both stalling at the same point.

## Infrastructure: the console flood, and what it did and did not explain

The image prints `remote fence extension is not available in SBI v1.0` 842 times
during boot — **61 % of all console traffic** — and under TCG with an emulated
8250 the console is the bottleneck. Booting with `loglevel=1` (a second
`-append` overrides the driver's) cut console traffic from 71125 to 2708 bytes
and time-to-login from 8-16 minutes to under 2.5.

**It did not eliminate the intermittent stall**, and an earlier claim here that
"the flood WAS the flake" is retracted. Characterisation of what remains: the
vCPU thread spins at 99.9 %, so it is a guest-side busy loop; it is not the
domain (not yet loaded) and not the qemu arguments (the identical command line
reaches the login prompt when its output goes to a file instead of a pexpect
pty). Most likely a getty/respawn race in the buildroot image. Survived rather
than fixed: `CAPSTONE_QEMU_LOGIN_TIMEOUT=300` so a stall costs five minutes
instead of sixteen, plus three retries on exit 75.

## State

| | |
|---|---|
| musl sources the compiler accepts | 1242 / 1361 (91.3 %) |
| `libc-capstone.a` | links; `write(1,…)` pulls in only `__capstone_hostcall` |
| resumable hostcall from a pure-cap domain | works (`yield-probe/`) |
| musl `write()` end to end | works (`musl-hello/`) |
| syscalls implemented | `write` (stdout/stderr), `exit`, `exit_group`; everything else `-ENOSYS` |

Next: the file opcodes (`FILE_OPEN`/`READ`/`WRITE`/`CLOSE`) already exist in
HostCall v0 and map almost 1:1 onto `openat`/`read`/`write`/`close`, so `stdio`
on a real file is the next rung. Then the lending path, for which this
copy-based transport is the matched control.
