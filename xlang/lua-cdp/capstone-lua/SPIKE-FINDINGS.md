# Lua-on-Capstone feasibility spike — findings

**Question.** Can reference Lua (the interpreter) actually be compiled and run as a
Capstone domain, so the xlang CDP cases could use *real Lua + a C stand-in
resource* instead of a pure C shim (what the collaborator asked for)?

**Verdict.** Yes, and the blocker is **not** Lua and **not** libc — it is
**Capstone-compiler maturity**. Reference **Lua 5.4.7 source needs 0 portability
patches**; the libc surface is mostly already provided by the domain runtime; the
two things that stop a full compile today are **two Capstone LLVM backend codegen
gaps**, one of which is the already-tracked C-2 family.

## What was proven (measured, not estimated)

Compiling all 26 core+lib TUs of reference Lua 5.4.7 for `capstone64` with the
freestanding recipe below:

| Result | Count | Files |
|---|---|---|
| **Compile clean** | **21 / 26** | all core except ltable; all libs except lstrlib/lbaselib/ltablib/lauxlib |
| Backend codegen gap | 2 | `ltable.c`, `lstrlib.c` |
| Trivial libc-decl gap | 3 | `lauxlib.c`, `lbaselib.c`, `ltablib.c` |

The 19/20 **core** files compile with the best flag set; only `ltable.c` fails on
a backend gap.

### 128-bit-pointer portability of Lua's source: CLEAN (0 patches)

Confirmed both by an exhaustive source audit and by compilation: every
pointer→integer cast in Lua is a benign hash/seed/opaque-print token
(`point2uint`, `luai_makeseed`, `lua_topointer`'s LCF case); the stack is
relocated the CHERI-safe way (byte offsets via `savestack`/`restorestack`, rebased
on a fresh base capability); light userdata round-trips the full 128-bit host
pointer losslessly through union member `.p` (the load-bearing path for the xlang
host-pointer use). No pointer alignment-masking anywhere. `uintptr_t` here is
64-bit, so `point2uint` is clean i64. One footprint note (not a fix): `TValue`
grows 16→32 bytes (16-byte capability), so stack slots / table parts / Nodes
roughly double — a memory-overhead datum for the paper.

## The two backend gaps (the real blocker)

Both are the Capstone LLVM backend failing to select a 128-bit (capability-width)
DAG node that Lua's ordinary C generates. Neither is a Lua bug.

1. **`ltable.c` — `Cannot select: i128 = or t78, undef:i128`.** Lua's table code
   indexes arrays and takes pointer differences; on capstone64 the GEP index /
   pointer-diff arithmetic is capability-width, and legalizing the `zext i32 …
   to i128` used for indexing produces an `or i128 X, undef` the backend has no
   pattern for. Fails at -O0/-O1/-O2, all flag sets. Under `-fno-jump-tables`
   the *same* node also appears in lapi/ldo/lstate/lstring (that flag shifts
   codegen enough to expose it more widely).

2. **`lstrlib.c` — `Cannot select: i128 = CapstoneISD::SELECT_CC …,
   Constant:i64<-9223372036854775808>, …`.** A select/ternary whose result is
   capability-width. **This is the C-2 family** already tracked in this project
   (the same `i128 = SELECT_CC` that crashes the C++ `new`-expr and that the
   mruby domain comment warns about: "selects between two capabilities and
   crashes the backend").

**Fix path:** compiler patches (add the missing i128 isel patterns / a DAGCombine
folding `or X, undef`, and resolve C-2 `SELECT_CC`), *or* narrow source rewrites
of the two hot patterns (against the "0 patches" goal, and fragile). The compiler
route is the right one and shares directly with existing Capstone backend work.

## The trivial libc-decl gaps (mechanical)

All three are in the overridable/strippable surface, not the core:
`lauxlib` (`errno`, `strerror` — the file-loader / fileresult paths, strippable via
`luaL_loadbufferx` instead of `luaL_loadfilex`); `lbaselib` (`fwrite`, `stdout` —
`print`, overridable via the `lua_writestring` macro → console writer); `ltablib`
(`clock_t` — sort-pivot randomization, overridable via `l_randomizePivot`). None
touch the interpreter core.

## Reproduce

```bash
CLANG=llvm/cmake-build-debug/bin/clang
LUA=xlang/lua-cdp/_toolchain/.work/lua54          # reference Lua 5.4.7 source
SHIM=xlang/lua-cdp/capstone-lua/capstone_lua_libc.h
STUBS=xlang/lua-cdp/capstone-lua/include          # empty stubs for hosted headers
$CLANG -target capstone64-unknown-elf -Xclang -target-feature -Xclang +m \
  -ffreestanding -nostdlibinc -fno-builtin -ffunction-sections -fdata-sections -O0 \
  -include "$SHIM" -I"$STUBS" -I"$LUA" -c "$LUA/<file>.c" -o /dev/null
```

Key recipe facts, learned the hard way:
- **`-nostdlibinc`** (not just `-ffreestanding`) is required: `-ffreestanding`
  alone still lets clang's `<stdint.h>`/`<limits.h>` `#include_next` into glibc's
  16-byte-pointer-incompatible `bits/types.h`. `-nostdlibinc` drops `/usr/include`
  while keeping clang's builtin `stddef`/`stdint`/`limits`/`float`/`stdarg`.
- The **hosted headers** Lua includes (`string.h`, `math.h`, `setjmp.h`,
  `locale.h`, `ctype.h`, `stdio.h`, `time.h`, `signal.h`, …) are provided as empty
  `#pragma once` stubs in `include/`; all decls come from the force-included
  `capstone_lua_libc.h`. `signal.h` must supply `sig_atomic_t`.
- **Do NOT pass `-fno-jump-tables`** — it triggers the ltable i128-or gap in 4 more
  files. (It may be needed at link/runtime for the domain; revisit then.)

## Remaining work to a *running* Lua domain (revised estimate)

1. **Fix the 2 backend gaps** — the dominant cost, and genuine compiler work.
   The `SELECT_CC` one is C-2 (already on the radar); the `or i128 undef` one is a
   fresh, likely-small isel/combine gap. This is the real bottleneck.
2. **libc gaps** (small): `setjmp`/`longjmp` (capability-aware asm — see below),
   `snprintf`/`vsnprintf` (the biggest single TU: full printf conversions incl
   `%a`), `fmod`/`modf`/`frexp`/`ldexp` + `__divdf3` (soft-float/libm additions),
   and the trivial decl stubs / macro overrides for errno/strerror/fwrite/stdout/
   clock_t and `luai_makeseed`.
3. **Reuse** verbatim: `start.S`/`link.ld`, the revoking allocator
   (`xlang/common/revoke_arena_domain.c`), the beebs/sqlite string+libm+ctype, the
   header-shadow pattern.
4. **Wire** `domain_main` + a C stand-in binding (userdata + `__gc` modelling the
   native resource) → run the real `trigger.lua` from memory (`luaL_loadbufferx`).

### setjmp/longjmp ABI (verified against primary source)

The QEMU domain build uses `my_first_domain/start.S` (the **cjalr ABI**) with **no**
`-capstone-gp-captable` flag, so **`ra` is a capability here** (saved with `stc`,
returned via `cjalr`) — *not* a scalar. So all 14 saved regs (ra, sp, s0–s11) are
capabilities → all `stc`/`ldc`, `jmp_buf` = 14×16 = 224 B, 16-byte aligned (tag
preservation). The setjmp-spec agent assumed gp-captable (ra scalar); that is
wrong for our build. Verify by experiment before trusting the final asm.

## Bottom line for the paper / the collaborator

Real Lua on Capstone is feasible and **Lua itself is ready** (0 source patches,
21/26 TUs compile). What is not ready is the **Capstone compiler**: two i128
lowerings — including the already-known C-2 `SELECT_CC` — block a full build. That
is the same compiler-maturity gap the whole project is closing, now with a
concrete, reproducible Lua-scale test case for it. This is exactly the
compatibility axis in evidence: not "Lua can't", but "the Capstone toolchain
can't *yet*".
