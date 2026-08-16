# MicroPython as a freestanding Capstone domain — the compilation plan

**Status:** Stage 0 is DONE (2026-08-16) and its numbers are below. Stages 1-6 are PROPOSED, none started.

**Scope:** getting the interpreter to *compile and link* as a `.dom`. Conformance and performance
methodology is a separate document. Nothing here touches the board, the paper, or the RTL.

## Why this target

SQLite proved that a large C program runs in a pure-capability domain. MicroPython adds the thing
SQLite does not have: a language runtime with **its own allocator and garbage collector layered on
top of the domain heap**, which is the nested-allocator question this branch exists for. It is also
five times smaller in code than SQLite, so it is a cheaper vehicle for that question than any
further SQLite work.

## Reference numbers

Measured 2026-08-16, RV64 `riscv64-linux-gcc -Os -fdata-sections`, MicroPython master `2e3304a`,
SQLite amalgamation 3.53.04 built with no `SQLITE_OMIT_*`:

| | MicroPython (py core, one TU) | SQLite (amalgamation) |
|---|---:|---:|
| `.text` | 166 KiB | 696 KiB |
| static pointer slots (abs64 relocs in data/rodata) | 960 | 970 |
| defined data symbols | 346 | 182 |
| external libc symbols | 10 | ~28 |

Each static pointer slot becomes one straight-line store in `__capstone_cap_init`
(`CapstoneCapGlobalInit.cpp`), so MicroPython needs that machinery at the same scale SQLite already
proved, at a fifth of the code.

Further measured facts that the plan leans on:

- `py/` is 133 `.c` files; the `minimal` port builds 132 of them. **Those 132 amalgamate into one
  translation unit with exactly one fix** (`#include <stdarg.h>` before the rest; without it
  `mp_obj_new_exception_msg_vlist` re-declares with a different `va_list`). The gp-captable ABI
  requires one module because `getGpCaptableIndex` numbers globals per module, and for SQLite that
  requirement cost real work. Here it is close to free.
- External libc surface of the whole minimal port: `memcpy memmove memset memcmp strlen strcmp
  strncmp strchr` plus `read`/`write` (and `__stack_chk_*`, disabled with
  `-fno-stack-protector`). All eight already exist in
  `benchmarks/beebs/adapted/beebs_freestanding_string.c`; `read`/`write` are the existing HostCall
  path. **No `malloc`**: allocation is the GC over a static array (minimal port default 25 KB;
  the SQLite domain already runs a 256 KB static heap, so heap size is not a constraint).
- Linked `minimal` firmware for RV64: `.text` 137 KB, `.data` 10.7 KB, `.bss` 26.5 KB.

## Stage 0 — the compilation census. DONE 2026-08-16.

Reproduce (13 minutes, no board, no QEMU). Per file, with the SQLite silicon flag set:

```
clang -target capstone64-unknown-elf -Xclang -target-feature -Xclang +m \
  -ffreestanding -fno-builtin -fno-optimize-sibling-calls -fno-jump-tables -std=c99 -O0 -w \
  -mllvm -capstone-gp-captable -mllvm -capstone-shrink-stack=false \
  -mllvm -capstone-shrink-globals=false -mllvm -capstone-merge-string-constants=true \
  -DCAPSTONE_GP_CAPTABLE_ABI=1 -I<shim> -I. -Iports/minimal -Iports/minimal/build \
  -c py/<file>.c
```

`<shim>` is a throwaway freestanding include directory (`assert.h`, `string.h`, `stdio.h`,
`stdlib.h`, `errno.h`, `setjmp.h`, `unistd.h`, `sys/types.h`, `math.h`) declaring only what the core
references. Building it properly is stage 4.

### Result 1: the frontend has nothing to say

**Zero errors** over the whole amalgamated core with `-fsyntax-only`. Clang for `capstone64` accepts
MicroPython's tagged-object macros and pointer arithmetic unchanged, with `void*` at 128 bit. The
representation problem is not a frontend problem.

### Result 2: 119 of 133 files compile, and the 14 that do not fall into four signatures

| files | signature | first site | belongs to |
|---:|---|---|---|
| 7 | `MCStreamer::emitIntValue` assert ``1 <= Size && Size <= 8`` via `emitGlobalConstantStruct` | module-level global emission (no function) | **backend**: cannot emit a 16-byte integer in static data |
| 2 | `APInt::getSExtValue()` assert in `CapstoneDAGToDAGISel::tryShrinkShlLogicImm`, `CapstoneISelDAGToDAG.cpp:697` | `gc_init`, `mp_pairheap_delete` | **backend**: missing width guard |
| 2 | `Cannot select: i128 = and` / `i128 = xor` | `mp_obj_get_type`, `bound_meth_unary_op` | **backend**: i128 bitwise lowering |
| 3 | `Cannot materialize arbitrary >64-bit constants as capabilities`; `CIncOffset displacement must fit in signed 64-bits` | `list_pop`, `str_finder`, `mp_execute_bytecode` | **source**: the object representation |

Failing files: `gc modbuiltins modsys obj objboundmeth objdict objgenerator objint objlist
objmodule objstr objtuple pairheap vm`.

**11 of the 14 are toolchain gaps; 3 are the object-representation question.** That inverts the
original scoping, which assumed the representation was the whole job. The seven-file group is the
clearest example: `mp_rom_obj_t` is a union, so `MP_ROM_INT(x)` in a const table asks the AsmPrinter
to emit a pointer-sized *integer* in static data, and the Capstone streamer only handles up to 8
bytes. SQLite never triggers this because it has no integer-in-pointer-slot unions.

### A trap found while measuring

Without assertions, `APInt::getSExtValue()` returns `int64_t(U.pVal[0])`, i.e. the low 64 bits,
**silently** (`llvm/include/llvm/ADT/APInt.h:1565`). A Release clang does not abort on `gc.c` and
`pairheap.c`, it emits a wrong immediate. Until stage 1b lands, build MicroPython domains only with
an assertions-enabled clang, and treat any domain built with a NoAsserts clang as void.

### What the census does NOT say

- **A file that compiles is not a file that is correct.** The frontend accepted every tag
  manipulation, so each `(mp_int_t)obj & 3` in the 119 clean files lowered to *something*. Whether
  that something preserves the capability tag is a runtime question no compile can answer. Stage 3
  exists for exactly this, and it is the part most likely to be underestimated.
- `-O0` only. The SQLite domain builds `-O0` for the same reasons; whole-image `-O1` is separately
  blocked by C-17.
- **First failure per file**, so the counts are lower bounds: a fixed file can fail again further in.
- Per-file, not a domain build. A real domain is one TU, which changes cap-table numbering and can
  surface failures this census cannot.

## Stage 1 — close the three backend gaps

Unblocks 11 of the 14 files and touches the shared compiler, so it carries the heavier gate.

- **1a. 16-byte integers in static data.** `emitGlobalConstantImpl` reaches
  `CapstoneELFStreamer::emitValueImpl` with size 16. Emit as two 8-byte halves in the right order.
- **1b. Width guard in `tryShrinkShlLogicImm`** (`CapstoneISelDAGToDAG.cpp:697`): check
  `getSignificantBits() <= 64` before `getSExtValue()`. The same class of bug was already fixed once
  in `SelectionDAGAddressAnalysis` (see the three codegen fixes of 2026-07-27); this is the second
  instance, which suggests a sweep for other unguarded `getSExtValue()` calls in the target is worth
  one grep.
- **1c. i128 `and`/`xor` lowering.** A previous fix covered the constant-mask path and left the
  general one; the census hits both a constant (`and t4, Constant:i128<15>`) and a register form
  (`xor t7, t11`). This is the open-ended item of the three.

Each fix ships with:

- **a lit test that fails without it.** A fix with no failing test is unproven, and this project has
  paid for gates that could never fire.
- **byte-identity for everything else**: hash a known `.dom` (e.g. the SQLite domain) before and
  after; a backend change that moves unrelated codegen invalidates every measurement on file.
- the standard regression gate: lit, BEEBS, RV8, authority, SQLite QEMU rows.

**Gate:** 130 of 133 files compile; regression suites unchanged; reference `.dom` hashes identical.

## Stage 2 — REPR_CAP: decide the representation, then fix the three loud sites

The three remaining failures are the compiler refusing to treat an object word as an integer. The
fix is a fifth object representation beside upstream's REPR_A..D.

**Proposed shape, to be validated in code and not in prose:**

- `mp_obj_t` stays `void*` (a capability).
- **The capability tag becomes the type test.** Tagged means "pointer to an object"; untagged means
  everything else. `cap_get_tag` replaces the `(x & 3) == 0` test.
- Inside the untagged case the existing REPR_A bit layout is kept **verbatim** (`...1` small int,
  `...010` qstr, `...110` immediate), so no other part of the interpreter changes meaning.
- `mp_int_t`/`mp_uint_t` are pinned to `int64_t`/`uint64_t` through the upstream `MP_INT_TYPE_OTHER`
  hook, so they stop being pointer-width. This is a config hook, not a patch.
- Constructing a non-pointer object becomes "materialise an untagged 128-bit value", never "cast a
  pointer".

Why this shape rather than upstream's low-bit tagging carried over: it uses what the hardware
already provides, it keeps the GC's root scan **exact** instead of conservative, and it never puts a
capability with a deliberately misaligned cursor into circulation, which is the shape most exposed
to the open silicon defects (S-06, S-07).

The open question it must answer first: can the backend materialise an arbitrary untagged 128-bit
value cheaply? Today it refuses (`Cannot materialize arbitrary >64-bit constants as capabilities`),
which is the correct default and may need one narrow, explicit escape hatch rather than a general
relaxation.

**Gate:** 133 of 133 files compile, **and** a representation self-test runs as a domain under QEMU:
small-int round trip, qstr round trip, pointer round trip, tag survives a store and reload. The
self-test must FAIL when one macro is deliberately broken, demonstrated before the pass is believed.

## Stage 3 — the silent half

Turn "it compiles" into "there are no unintended integer/pointer conversions left". The census
proves the compiler is quiet about most of them, which is the danger, not the reassurance.

- Count and list every `ptrtoint`/`inttoptr` on AS200 in the module IR and judge each one. There are
  132 casts to `mp_int_t`/`mp_uint_t` in `py/*.c`, roughly 20 of them applied to object-typed
  expressions.
- Prefer a mechanical list over reading: a pass or a script over the emitted IR, so the result is a
  number that can be re-checked after every change, not an opinion.

**Gate:** a list with a verdict per site, recorded as a follow-up note in `history/`.

## Stage 4 — the freestanding runtime

- `capstone_micropython_libc.{c,h}` modelled on `benchmarks/sqlite/adapted/capstone_sqlite_libc.*`,
  replacing the throwaway shim. The eight mem/str functions come from the existing BEEBS
  freestanding string file; `read`/`write` go through HostCall.
- **NLR.** `py/nlrrv64.c` is 81 lines and saves `ra`, `s0-s11`, `sp` with `sd`. Here `ra` and `sp`
  are capabilities, so this becomes `stc`/`ldc` with 16-byte slots in `nlr_buf_t`. Extend the arch
  detection in `py/nlr.h`, otherwise it silently selects `MICROPY_NLR_SETJMP` and asks for a
  `setjmp` the domain does not have. The census hit exactly that.
- **GC root spill**: the same machinery, capability-aware, so spilled registers keep their tags and
  the collector can see roots.
- `mpconfigport.h` for the domain: `MICROPY_FLOAT_IMPL_NONE`, no native/viper emitter, GC on, static
  heap array, REPL off for the first build.

**Gate:** links to a `.dom`.

## Stage 5 — the domain build

`build-micropython-silicon.sh` modelled on `build-sqlite-silicon.sh`. One TU. Keep the link-time
gate of **exactly one `.capstone_gp_table` header**, which turns a whole class of silent
wrong-answer into a build error. Run `benchmarks/sqlite/check-capinit-slots.py` over the result.

Measure and record, do not assume: carve count, cap-init slot count, `.text`.

**Known risk, with a number:** 346 defined data symbols against SQLite's 179 working carves, and
rungs at roughly 170 cap-table entries have entry-stalled historically. If the carve count lands
high this is a real bring-up problem, and it is far cheaper to meet it here than on the board.

**Gate:** `micropython.dom` builds; static gates green (`cjalr=0`, single descriptor); cap-init slot
check clean.

## Stage 6 — one line of Python

`print(1+1)` from frozen bytecode, in a domain, under QEMU.

**Gate:** `2`.

Nothing before this proves the build is real. A `.dom` that has never run is an artifact, not a
result, and this project has spent board sessions on images that were correct and inert.

## Effort, as estimates and labelled as such

| Stage | Estimate | Confidence |
|---|---|---|
| 1a, 1b | 1-2 days together | high, both are contained |
| 1c | 2-5 days | low, previous fix covered only part of the space |
| 2 | 1-2 weeks | medium, the design is cheap and the fallout is not |
| 3 | 2-3 days | medium |
| 4 | 3-5 days | medium, NLR under linear capabilities is the unknown |
| 5 | 2-4 days | medium, carve count is the risk |
| 6 | 1 day if 1-5 hold | low, this is where everything unmeasured surfaces |

## Out of scope for this plan

Board runs. Conformance and performance methodology. The allocator study's later arms (per-object
`shrink` at `gc_alloc`, revoke on GC sweep). Any change to the paper.
