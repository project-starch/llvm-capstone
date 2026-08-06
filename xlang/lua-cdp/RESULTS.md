# Lua-CDP corpus — CHERI vs Capstone, security results

**The fair, measured comparison of CHERI and Capstone on 13 real Lua↔C
cross-domain-pointer use-after-frees.** The same distilled shim source runs on
both platforms; only the capability mechanism differs, and each runs in its
native deployment mode. This file is the security half — compatibility and
performance are separate axes (see the bottom). Performance + memory on the
official Lua binary-trees benchmark (the same 13-case Lua on both platforms)
are now measured — see `PERF-MEMORY.md`.

## The one-sentence result

**The CHERI configuration that is actually deployed — async quarantine — prevents
0 of 13 of these cross-domain use-after-frees at the moment of the stale access;
Capstone's synchronous revoke-on-free prevents all 13** (as does CHERI's
aggressive, non-default `eager`).

## The metric

Each cell answers one question: **was the stale cross-domain access prevented at
the contract point** (the instant the freed C object is dereferenced through the
still-live language handle)?

- **prevented** — a capability fault fires *on the stale access itself*.
- **not prevented** — the stale access completes (the freed object is read/written).

This is deliberately *not* "did the process survive": a program can also die
later for an unrelated reason (see the openssl double-free footnote), which is a
different event from the CDP being caught.

## The corpus

13 pure Lua↔C cross-domain-pointer UAFs, each a two-allocation coupling (a Lua-GC
handle ⟷ a separately allocated native C resource), distilled from a real
upstream bug. C++ cases (sol2, LuaBridge, wxLua) and LuaJIT cases (xmlua,
tarantool) are excluded — Capstone has no libc++/STL for `capstone64` and no JIT
backend, so they cannot run on both sides. See `README.md`.

Each shim is **fidelity-gated**: it reproduces its case's `heap-use-after-free`
natively under ASan before it is trusted (e.g. openssl: `READ of size 8` at
offset 0 of the freed `EVP_CIPHER_CTX`, matching the upstream trace).

## Vehicle

| | **Capstone** | **CHERI** |
|---|---|---|
| Compiler | `capstone64-unknown-elf` clang (`llvm/cmake-build-debug`) | CHERI-clang (`~/cheri/output/sdk`) |
| Emulator | Capstone-QEMU (`capstone/capstone-qemu`) | `qemu-system-riscv64xcheri` |
| Runtime | freestanding **domain**, loaded by a Buildroot-Linux host controller | ordinary **CheriBSD purecap process** |
| Allocator | `xlang/common/revoke_arena_domain.c` on `revoke_on_free_alloc.h` — every allocation independently revocable | CheriBSD `malloc` + kernel revocation |
| Opt level | `-O0` (both — at `-O1`+ the dangling access is hoisted/elided) | `-O0` |
| Workload | **identical shim source** `xlang/lua-cdp/shims/*.c` | **identical shim source** |

## Fairness: one shim, two mechanisms

Both columns compile the **byte-identical file** `shims/<case>.c` — one directory,
no per-platform copies (both build scripts resolve `../shims/<case>.c`). The shim
only *declares* `malloc` / `free` / `memcpy` / `abort` / `mock_report`; each
platform links different *definitions*, and that difference **is** the mechanism
under test:

| | shim (identical) | + companion it links against (the mechanism) |
|---|---|---|
| **Capstone** | `luac_<case>.c` | `xlang/common/revoke_arena_domain.c` (revoke-on-free) + freestanding domain driver |
| **CHERI** | `luac_<case>.c` | `cheri/mock_report.c` + CheriBSD **libc** (`malloc` + kernel revocation) |

So the workload is held constant and only the capability/revocation system varies.
That is the entire comparison.

## Configs

| Config | Meaning |
|---|---|
| CHERI **spatial** | revocation OFF — spatial safety only |
| CHERI **async** (temporal) | revocation ON, async quarantine sweep — **the deployed default** |
| CHERI **eager** | revocation ON, revoke on every `free` — aggressive, synchronous |
| Capstone **revoke** | revoke-on-free — maps onto CHERI `eager` |
| Capstone **control** | identical program minus the one REVOKE — the attribution test |

## Results — was the CDP access prevented?

| # | Case | CHERI spatial | CHERI **async** (default) | CHERI eager | **Capstone** |
|---|---|:---:|:---:|:---:|:---:|
| 1 | lua-openssl #141 | no | no † | **yes** | **yes** |
| 2 | ldbus #20 | no | no | **yes** | **yes** |
| 3 | cffi-lua #57 | no | no | **yes** | **yes** |
| 4 | lua-sdl2 #75 | no | no | **yes** | **yes** |
| 5 | wireshark #16807 | no | no | **yes** | **yes** |
| 6 | luv #696 | no | no | **yes** | **yes** |
| 7 | lgi #122 | no | no | **yes** | **yes** |
| 8 | lgi #65 | no | no | **yes** | **yes** |
| 9 | luaossl #124 | no | no | **yes** | **yes** |
| 10 | luadbi #35 | no | no | **yes** | **yes** |
| 11 | luv #503 | no | no | **yes** | **yes** |
| 12 | lua-curl #80 | no | no | **yes** | **yes** |
| 13 | lmdb value-after-txn | no | no | **yes** | **yes** |
| | **prevented** | **0/13** | **0/13** | **13/13** | **13/13** |

† **openssl double-free footnote.** openssl #141 is the only double-free in the
corpus (`__gc` reads the freed ctx, then frees it again). Under async the stale
read still **completes** (not prevented — same as the other 12); the process then
aborts at the *second* `free` via the allocator's double-free detector
(`SIGABRT`), which is a different event from revocation catching the read. The
raw per-config outcomes are in `cheri/expected-results.tsv`.

## Why the numbers are trustworthy

1. **Fairness** — byte-identical shim source on both sides (`shims/*.c`); only the
   mechanism varies, each in its native mode.
2. **Attribution** — the Capstone `control` (no-revoke) MISSes on all 13, so each
   FAULT is caused by the revoke, not an `-O0` spill artifact. CHERI's `spatial`
   column (0/13) is the same control.
3. **Config reality** — the CHERI run records `REVOKE_ENABLED=0/1/1` across the
   three passes, so a config that silently failed to apply would be caught.
4. **Predict-then-measure** — `rows.tsv` in each column was committed before the
   runs.

## How to reproduce

```bash
# both columns, rebuild + re-measure + check against expected-results.tsv:
bash xlang/lua-cdp/reproduce.sh            # ~1 h (26 Capstone boots + 1 CHERI boot)

# one column at a time:
bash xlang/lua-cdp/reproduce.sh capstone   # Capstone: 13 rows x {revoke,control}
CHERI_ROOT=$HOME/cheri \
  bash xlang/lua-cdp/reproduce.sh cheri     # CHERI: 13 rows x {spatial,async,eager}, one boot
```

`reproduce.sh` rebuilds every shim from source, re-runs under QEMU, and **exits
non-zero on any disagreement** with `capstone/expected-results.tsv` /
`cheri/expected-results.tsv`. A column nobody can re-measure is a claim, not a
result.

## What this is NOT (limitations)

- **Distilled shims, not the real libraries** — the real OpenSSL/SDL2/… cannot run
  on Capstone at all (no OS/syscalls). That gap *is* the compatibility axis.
- **QEMU, not silicon** — Capstone absolute costs need RTL/FPGA.
- **Security only** — compatibility (coverage/porting cost) is a separate,
  still-open axis. Performance + memory are now measured on real Lua
  (`binary-trees`, see `PERF-MEMORY.md`), alongside the earlier SQLite
  revocation-overhead result — this file stays security-only by design.
- **"async 0/13" = not caught *synchronously*** — a later sweep would revoke, but
  not at the instant of the stale access. That window is the security-relevant
  one.
