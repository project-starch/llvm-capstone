# Reference Lua 5.4 runs against musl inside a pure-capability domain, 2026-08-14

**Result, QEMU, verified.** The reference interpreter — 22 core translation units,
unmodified — compiled against musl instead of a hand-written libc, running in a
`capstone64-unknown-elf` domain:

```
LUA S1: entered
LUA S2: newstate ok
LUA S3: base library opened
LUA S4: chunk compiled
LUA S5: pcall ok
LUA OK: t[20] == 400, real interpreter on musl
__CAPSTONE_LUA_PROBE_PASSED__
__CAPSTONE_HOSTCALL_HOST_DONE__ status=0 serviced=7
```

Harness: `capstone/musl-capstone/lua-probe/run-lua-probe.sh`. The chunk is the
one the cjalr bring-up uses (`local t={} for i=1,20 do t[i]=i*i end return t[20]`),
so the result is directly comparable with the hand-written-libc build. Every
stage is a required marker, not just the last: reaching `400` needs the parser,
the VM, the GC and a realloc that MOVES the table's array part.

**What it replaces.** `xlang/lua-cdp/capstone-lua/` carries **1008 lines** of
hand-written libc (`capstone_lua_libc.h` + `lua_libc.c`) for exactly this.

## Getting from "compiles" to "links": four steps, each measured

| step | symbols resolved | why |
|---|---:|---|
| application include path without `src/include` and `src/internal` | 18 of 22 TUs | musl defines `weak` and `hidden` as MACROS there; Lua has a field `GCObject *weak`, and the collision reads as "expected member name or ';'" |
| `-O0` instead of `-O1` | 5 TUs | the i128-on-a-capability family. `-O0` is what the working Lua recipe already uses |
| our own `strtod`/`strtof`/`strtold` | **15** | musl's `floatscan.c` converts through `long double` |
| our own allocator | 5 | every file under `src/malloc` fails on `sizeof(void*)` static asserts |

`src/include` and `src/internal` are musl's own build headers. They belong on the
path when compiling musl and nowhere else — a distinction the survey needs and an
application must not inherit. `build-musl-hello.sh` had them too, harmlessly, and
has been corrected.

## `long double` is unusable on capstone64, in compiler-rt as well as musl

Cutting `strtod` removed 15 symbols at once, and the reason is worth stating
separately: **every** 128-bit long-double builtin fails to compile for this
target, with the same three backend assertions as musl's own long-double files.

```
comparetf2.c  addtf3.c  subtf3.c  multf3.c  divtf3.c
extenddftf2.c extendsftf2.c floatsitf.c floatunsitf.c
trunctfdf2.c  trunctfsf2.c        -- all FAIL
```

`getActiveBits() <= 64`, `getSignificantBits() <= 64`, `VT.isVector() && "Unable
to legalize non-vector shift"`, and `Cannot materialize arbitrary >64-bit
constants`. So this is not a musl porting gap that a better port would close: it
is the i128 representation serving as both a capability and a `long double`. Any
program reaching `strtod`, `printf("%a")` or `long double` arithmetic is blocked
until the backend separates the two.

## Stubs that fail loudly

`fopen`, `vfprintf`, `strtod` are stubbed, and each prints a line saying so. A
silent partial `strtod` was the tempting option and is the wrong one: Lua's lexer
uses it for FLOAT literals, so a subtly wrong parser would make a chunk compute a
wrong number and still report success. The probe's chunk uses integer literals,
which Lua reads with its own `l_str2int`, so the path is linked but not taken.

## Two syscalls were refused, and it matters

```
hc-host: kind=0xe0 nr/val=113 arg0=0     SYS_clock_gettime
hc-host: kind=0xe0 nr/val=169 arg0=0     SYS_gettimeofday
```

Both come from Lua 5.4's `luai_makeseed`, which calls `time()`. They return
`-ENOSYS`, so **Lua's string-hash seed is not seeded**. Harmless for this chunk,
but it is exactly the condition hash-flooding attacks want, and it should not
survive into anything measured for security. Wiring a time opcode into HostCall
v0 is the fix.

## A 795 KB domain needs a longer timeout, not a diagnosis

The first run failed with no fault, no message, and a log that stopped between
libcapstone's "Segment size" and "Loadable size" lines — i.e. inside the guest's
ELF loader, which mmaps, zeroes and copies the whole image before the ioctl. That
reads like a dead domain and was merely a slow one: 795 KB against musl-hello's
4 KB, all of it under TCG. It completes at `--timeout-multiplier 20`; the default
in `run-lua-probe.sh` is now 20 with that reason recorded.

## What this changes for the CDP corpus

`xlang/lua-cdp/WHY-SHIM.md` gives as its first reason for measuring through
distilled C shims: *"Capstone cannot run the real thing … no OS, no syscalls, no
libc."* That reason is now false for the libc half. The 13 corpus cases still use
shims and one (`09-luaossl-124`) already runs through real Lua on the
hand-written libc; this removes the libc obstacle for the rest. It does **not**
remove the other obstacle WHY-SHIM.md names — the real native libraries
(OpenSSL, SDL2, libuv) are still stubbed, and nothing here changes that.
