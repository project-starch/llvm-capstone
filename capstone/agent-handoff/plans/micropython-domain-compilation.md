# MicroPython as a freestanding Capstone domain — the compilation plan

**Status:** 2026-08-16. **The MicroPython core AND a Capstone port compile as one translation unit.** Stages 0 and 1 are
done, and closing the last five took four more backend items plus one source patch, recorded below
as stage 1b. Stages 2-6 are PROPOSED, and stage 2 has changed character: it is no longer what stands
between us and a build, it is what stands between a build and a *correct* one.

Reproduce: `benchmarks/micropython/{fetch-micropython.sh,census-capstone.sh}`.

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
| 2 | `Cannot select: i128 = and` / `i128 = xor` | `mp_obj_get_type`, `bound_meth_unary_op` | **source** (see the correction below) |
| 3 | `Cannot materialize arbitrary >64-bit constants as capabilities`; `CIncOffset displacement must fit in signed 64-bits` | `list_pop`, `str_finder`, `mp_execute_bytecode` | **source**: the object representation |

Failing files: `gc modbuiltins modsys obj objboundmeth objdict objgenerator objint objlist
objmodule objstr objtuple pairheap vm`.

**9 of the 14 are toolchain gaps; 5 are the object-representation question.** The seven-file group
is the clearest toolchain case: `mp_rom_obj_t` is a union, so `MP_ROM_INT(x)` in a const table asks
the AsmPrinter to emit a pointer-sized *integer* in static data, and the streamer only handles up to
8 bytes. SQLite never triggers this because it has no integer-in-pointer-slot unions.

### Correction, same day: the i128 `and`/`xor` group is NOT a backend gap

It was filed as one first. Reading the bail path refutes that. `lowerScalarI128Logical`
(`CapstoneISelLowering.cpp:8297`) lowers an i128 logical op by narrowing both operands to XLen and
re-extending, and it returns `SDValue()` when an operand is **not** an extension of a 64-bit value.
Here the operands are real capabilities, so it bails, and the bail is correct in the same sense the
mixed-extend bail documented ten lines below it is correct: masking a capability's address bits and
handing back an untagged result is exactly the C-16 failure mode, where a truncated pointer lost its
tag and was then used as a base. Teaching the backend to do it silently would buy two files and
reintroduce a bug class this project has already paid for.

Where those i128 values come from, read out of the IR rather than guessed:

```
py/obj.h:102   #define MP_OBJ_NEW_IMMEDIATE_OBJ(val) ((mp_obj_t)(((val) << 3) | 6))
obj.ll:334     %cmp = icmp eq ptr addrspace(200) %0, inttoptr (i128 14 to ptr addrspace(200))
```

so a comparison against `mp_const_none` and friends materialises an integer constant **as a
capability**, and the DAG then folds a run of such comparisons into `(x & 15) == …`. Same root as
the three "loud" files: the object word is being *constructed* from an integer.

### The direction that already works, and it is the more common one

`(mp_int_t)(o) & 7`, i.e. **reading** a tag, compiles today: clang emits `ptrtoint ptr
addrspace(200) to i64` followed by an ordinary 64-bit `and` (`obj.ll:91`, `obj.ll:111`). Only the
**construction** direction (integer to object word) fails. That materially narrows stage 2: the
representation work is about how non-pointer objects are built, not about every tag test in the
interpreter.

### Refuted the same day: pinning `mp_int_t` to 64 bit is not a shortcut

The obvious cheap fix is upstream's own `MP_INT_TYPE_INT64`, which makes `mp_int_t`/`mp_uint_t`
`int64_t`/`uint64_t` instead of pointer-width, with no patch at all. Re-ran the whole census with
`-DMP_INT_TYPE=1`: **identical result, 119 pass, the same 14 files fail with the same signatures.**
The i128 values come from `mp_obj_t` being a pointer type, not from `mp_int_t`, so this changes
nothing on its own. Worth doing anyway as part of stage 2, but it buys zero files by itself and
should not be presented as progress.

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

## Stage 1 — close the two backend gaps

Unblocks 9 of the 14 files. Both are changes to our LLVM fork, so this carries the heavier gate.

- **1a. 16-byte integers in static data (7 files).** `emitGlobalConstantImpl`
  (`llvm/lib/CodeGen/AsmPrinter/AsmPrinter.cpp:4260`) already has a Capstone-specific carve-out for
  `Size > 8`: when the expression is **relocatable** (a capability pointer) it emits the symbol in
  the low word and zeroes the high word. The case that aborts is the neighbouring one, an
  **absolute** value, which falls through to `emitValue(ME, 16)` and trips
  `MCStreamer::emitIntValue`'s `1 <= Size && Size <= 8`. Extend the same block: emit the low 8 bytes
  and zero-extend, in data-layout order. Shared LLVM file, but a block that is already Capstone's
  and is guarded by `Size > 8`, which other targets do not reach through this path.
- **1b. Width guard in `tryShrinkShlLogicImm` (2 files).** `CapstoneISelDAGToDAG.cpp:697` calls
  `getSExtValue()` on a constant that can be 128 bits. Check `getSignificantBits() <= 64` first.
  Capstone-only file, one condition. The same class was already fixed once in
  `SelectionDAGAddressAnalysis` (three codegen fixes, 2026-07-27), so this is the second instance and
  a grep for other unguarded `getSExtValue()` calls in the target is worth doing in the same pass.

Each fix ships with:

- **a lit test that fails without it.** A fix with no failing test is unproven, and this project has
  paid for gates that could never fire.
- **byte-identity for everything else**: hash a known `.dom` (e.g. the SQLite domain) before and
  after; a backend change that moves unrelated codegen invalidates every measurement on file.
- the standard regression gate: lit, BEEBS, RV8, authority, SQLite QEMU rows.

**Gate:** 128 of 133 files compile; regression suites unchanged; reference `.dom` hashes identical.

### Stage 1 outcome, 2026-08-16: DONE, and 1b was five times bigger than it looked

Commit `67a7f60599df`. **126 of 133 files compile**, up from 119.

1a landed as described. 1b did not: guarding `tryShrinkShlLogicImm` only moved the crash into a
TableGen-generated predicate, because **45 immediate predicates in this target read constants with
`getSExtValue()`** and every one of them is written in terms of `int64_t` while i128 is legal here.
Patching them one at a time would have been 45 diffs and a re-audit on every upstream merge, so the
guard went where those predicates are generated: `CodeGenDAGPatterns.cpp` now emits a width check
ahead of both forms that convert a constant to int64, the `ImmLeaf` prologue and the
arbitrary-predicate form whose class is `ConstantSDNode`. The APInt form is left alone on purpose,
because taking an APInt is how a predicate declares that it handles arbitrary widths; the single
Capstone predicate that takes an APInt and then calls `getSExtValue()` on it (`TrailingOnesMask`) is
guarded in place.

Gates run, not assumed: Capstone lit 50/50; RISCV + TableGen 2629 tests; X86 5269 tests; only the
six known `emutls`/`tls-android` failures, and `emutls.ll` was re-run against a stashed rebuild to
confirm it fails identically **without** these changes rather than trusting the documented history.
Byte-identity: every Capstone lit input compiles to a bit-identical object before and after, the
only difference being the new test that previously produced no output at all.

Worth carrying forward: without assertions `getSExtValue()` returns the low 64 bits silently, so
this whole class was a **miscompile** in a release compiler and merely a crash in ours. The two
files it hit here, `gc.c` and `pairheap.c`, now fail with a readable `Cannot select` instead.

## Stage 1b — the last five, 2026-08-16 (commit `aa469bf76dd5`)

Four backend items and one source patch. Everything here has a test that was run against the
pre-fix binaries and failed there.

- **Zero-extended 64-bit constants.** `MP_OBJ_NEW_SMALL_INT(-1)` arrives as
  `inttoptr (i128 0xFFFFFFFFFFFFFFFF)`, because C widens the cast to the pointer's index type. The
  i128 constant path took only the sign-extended spelling, while `inttoptr i64 -1` already compiled
  to `li a0, -1`. Both name one register value; bits above the low 64 stay refused, which is the
  part that would fabricate metadata.
- **`collectStaticCapReducedObject`** guarded the holder but not the target it points at, and a
  table entry pointing at an `extern` object is a declaration.
- **The `AND` case of `Select`** is XLen peepholes reading the mask as `uint64`; it now leaves i128
  alone, the same guard `tryShrinkShlLogicImm` already carries.
- **Bitwise arithmetic on a capability, lowered rather than refused.** `gc_init` aligning a pointer
  down, `pairheap` stealing a low bit, `bound_meth_unary_op` hashing two pointers. The address is
  read with the same `lcc rd, rs, 2` a pointer difference uses, the operation happens at XLen, and
  the result is **untagged** — which is what the C asked for, since a value built out of `uintptr_t`
  bits cannot carry a tag here.
- **`vm.c`, in the source**, because this one the compiler must not paper over:
  `-MP_OBJ_ITER_BUF_NSLOTS + 1` is a sizeof-derived, therefore unsigned, index that is correct only
  because it wraps at pointer width. Where a pointer is wider than `size_t` the scaled displacement
  leaves the address space — the C-16 signature exactly.

### Two attempts that failed, recorded because they cost the time

**Declaring `i128 -> i64` truncation free** stops DAGCombiner widening the arithmetic, and also
silently reroutes pointer subtraction away from the `lcc` sequence `ptr-arith.ll` pins on purpose.
Whether the integer view of a capability register *is* its cursor is true in the QEMU model by union
aliasing (`cap.h`: `capboundsfat_t` leads with `cursor`) and unverified on the RTL. Reverted.

**Rewriting the align-down in the source** as `p - (p & (N-1))`, to keep it in the pointer domain and
preserve the tag, does not survive: DAGCombiner folds it straight back into `p & ~(N-1)`. Measured,
not assumed — the census is identical with and without that patch. This is why the case is lowered
rather than diagnosed.

### The correction this forces to what stage 2 says below

An earlier revision of this plan argued that making these compile "moves the failure from build time
to run time, which is the wrong direction". That was wrong twice over. The compiler is not hiding
anything: it now implements what the C says, and `(void *)(uintptr_t)x` is untagged on any capability
machine. And the crash it replaced was not a canary but an accident — the identical idiom at 64-bit
width has always compiled silently, so which files failed depended on whether DAGCombiner happened
to widen them. An inconsistent alarm is worse than none. Finding these sites is a systematic job and
it is stage 3.

### AND THE CENSUS WAS MEASURING THE WRONG THING for four rounds

Clang searches `/usr/include` even for a bare-metal triple, so every `#include <string.h>` in the
first four rounds resolved to the **host glibc header** and `adapted/include/` was never read.
Caught by deleting `adapted/string.h` and watching nothing change. With `-nostdlibinc` the shims
became load-bearing, one header (`alloca.h`) was missing, and the result is 133 of 133 for real. The
harness is now negative-tested in both directions: without `string.h` it reports 7 of 133, without
`assert.h` 13 of 133.

The earlier failure counts survive this — they were about MicroPython's own code and the backend,
neither of which the libc headers touch — but "compiles freestanding" did not mean what it said
until now.

**The same gap exists in `build-sqlite-silicon.sh` and the ladder builds**, which set `-ffreestanding`
without `-nostdlibinc`. SQLite is mostly insulated (`SQLITE_OS_OTHER` includes very little) and
`-include capstone_sqlite_libc.h` gets there first, so this is a latent hazard rather than a known
defect. It is not changed here: touching the flag set of the builds every board measurement rests on
is its own change, with its own gate.

## Stage 1c — the whole port compiles, and what the study found. 2026-08-16.

A six-way study (five investigations returned; the sixth and the verification pass died on an
account limit) plus verification of every load-bearing claim against the primary source. What
follows is only what was re-checked in this session, not what the study asserted.

### The port exists and compiles as ONE translation unit

`benchmarks/micropython/port/` — `mpconfigport.h`, `mphalport.h`, `qstrdefsport.h`, `mpy_domain.c`.
The whole core plus the port, compiled as one TU with the silicon flags:

| | |
|---|---:|
| `.text` at `-O0` | 321 KiB (328,912 B) |
| carves in `.capstone_gp_initdesc` | **232** |
| `__capstone_cap_init` | 11,900 B |
| `.bss` | 97 KiB (the 96 KiB GC heap) |
| undefined symbols | **11** |

The eleven are `__gpfree_globals_base`, `setjmp`/`longjmp`, and the eight mem/str functions. All
eleven have a supplier already in the tree. `MICROPY_NLR_SETJMP=1` means the capability setjmp
built in stage 1b IS MicroPython's exception mechanism, so no `nlrcapstone.c` is needed and
`py/nlr.h` is not touched.

### The generated glue cannot build this domain, and the reason is not size

`gen-gp-captable-glue.py` **aborts** on global 230: `.L.L__capstone_merged_strs.0`, 4080 B of
initialized data, `.L`-private so not copy-eligible, overflowing the 12-bit store offset.

That blob is the product of `-capstone-merge-string-constants=true`, so the obvious move is to drop
the flag. Measured, both ways:

| | carves | `.text` | generator |
|---|---:|---:|---|
| merging ON | **232** | 321 KiB | ABORTS at global 230 |
| merging OFF | **633** | 343 KiB | succeeds, 8,315 lines of glue |

So merging is what keeps the carve count near SQLite's proven 179; without it 633 carves is far
outside anything this project has run. **`-capstone-merge-string-constants=true` stays on, and the
generated glue is therefore not an option.**

`build-sqlite-silicon.sh:1550` uses `start-gp-captable-interp.S` — the descriptor-driven glue is
what the one large domain that works already uses, and its `.text` is O(1) in the carve count.

### But `DOMAIN_GLUE=interp` faults in this checkout, for known-good rungs too

Measured: `beebs_prime` with `DOMAIN_GLUE=interp` faults immediately (`cause = 7`, an out-of-bounds
capability access in the entry path); the same rung with the default generated glue returns its
oracle. So this is not about MicroPython.

The study's diagnosis is that nothing in this checkout copies the globals template into `dom_data`,
because `caplifive-buildroot` is pinned at `6912474`, one commit before `8c7b973 "modcapstone:
deliver the gp-captable init descriptor into dom_data"` on branch `xlang-gp-captable-delivery`.

Verified here: the pin and the branch are as described, `8c7b973` is the direct child of the pinned
commit, it touches only three files (+54/-1), and its diff adds the `memcpy` of the globals template
plus a `pr_info("gp-captable: copied ...")`. Verified further: the **staged** `capstone.ko` and
`capstone-test.user` are from 2026-07-29 and contain neither string, so they predate it.

**NOT verified, and it is the load-bearing part:** that this is the whole reason interp faults. The
discriminating experiment is a SQLite QEMU run, since SQLite uses interp and is reported to work —
if it passes today, the missing delivery cannot be the explanation and something else is going on.
That run has not been made. Do it before bumping the submodule.

### Two runtime tag losses in the GC, both patched and both proven in the emitted code

These are stage 2's problem showing up early, and they are the reason a compiling interpreter is
not a working one.

`patches/0002` — `PTR_FROM_BLOCK` (`py/gc.c:96`) reconstructs a heap address as
`block * BYTES_PER_BLOCK + (uintptr_t)pool_start`, an integer round trip, and all six uses
dereference the result. `gc_alloc` already computes the same address the tag-preserving way at
`py/gc.c:1007`, so the patch makes the macro agree with the function it is supposed to reproduce.
Proven: one integer `add` becomes one `cincoffset`, and the function it changes is
**`gc_mark_subtree`** — the site that walks the object graph, i.e. the first collection would have
faulted.

`patches/0003` — `gc_init`'s align-down (`py/gc.c:239`) masks the address. Subtracting the
misalignment instead keeps it a pointer, **but only with a `volatile`**: without one the optimizer
folds `p - (p & (N-1))` straight back into `p & ~(N-1)`. Proven: `andi ..., -32` disappears from
`gc_init` and a `cincoffset` appears.

### The idiom study, done here because that agent hit the limit

Six spellings compiled and read at `-O0` and `-O1`:

| idiom | -O0 | -O1 |
|---|---|---|
| `block * N + (uintptr_t)p` | untagged (`slli` only) | untagged |
| `p + block * N` | **tag kept** (`cincoffset`) | **tag kept** |
| `(uintptr_t)p & ~M` | untagged (`lcc`+`andi`) | untagged |
| `p - (p & M)` | untagged (folded back) | untagged |
| `p - asm_launder(p & M)` | tag kept | **MISCOMPILES** |
| `p - volatile(p & M)` | **tag kept** | **tag kept** |

The asm-laundered row is the one to know about: at `-O1` it emits `lcc a1, a1, 2` on a register
that came from `andi`, i.e. it reads the cursor of a non-capability, which QEMU's `helper_cslcc`
raises `UNEXPECTED_OPERAND` on. Checked whether this is a regression from stage 1b: it is **not**.
`lowerSUB`'s cursor path predates those commits (`git show 5bfcbd91ba72` has the same two
`getCapstoneCapabilityCursor` calls); stage 1b added only a sixth call site elsewhere. It is a
pre-existing hazard for any integer offset that `isCapstoneIntegerOffset` does not recognise, and
it argues for the `volatile` spelling rather than the asm one regardless.

### An instrument defect, of the class this project keeps paying for

`benchmarks/sqlite/check-capinit-slots.py:36-37` locates the descriptor by splitting the `readelf`
line and taking field 5. That is the file offset only while the section index is a single digit:
`line.replace("]", " ")` turns `[ 2]` into two tokens but `[12]` into one, shifting every field by
one, so the script would read the section SIZE as an offset. SQLite's descriptor happens to be
section 2, which is why it has never misbehaved. Verified by reading the line it parses. Fix it and
negative-test it before its OK is trusted on any other image.

### Stage 1 of that path, EXECUTED 2026-08-16: interp is genuinely blocked, and the fix is in-tree

The experiment was "run SQLite under QEMU, since it uses the interp glue and is reported to work".
Prediction made before the run, from the artifact rather than the source: it must FAIL, because the
firmware QEMU boots (`caplifive-buildroot/build/images/fw_jump.elf`, 2026-07-29) contains none of
the strings `gpoff`, `gp-captable` or `capstone_gp_initdesc`, and the source it was built from has
zero occurrences of `gpoff`.

**It failed, with the same signature as `beebs_prime`:**

```
beebs_prime  Cap mem access OOB: pc=1015a0250 rs1=x2 cursor=1015a1600 imm=48 size=16 bounds=(1015a1660, 1015c0000)
sqlite       Cap mem access OOB: pc=101600250 rs1=x2 cursor=101750b50 imm=48 size=16 bounds=(101750bb0, 101800000)
```

Same instruction offset (`...250`), same register (`x2` = sp), same displacement and width, and in
both cases **sp's cursor sits below its own bounds base** — 0x60 in the SQLite case. One cause, both
domains. So this is not about MicroPython and not about SQLite; the interp glue does not work in
this checkout at all.

**SQLite does not currently run under QEMU here.** That is not new: `ISSUES.md:4482` records the
QEMU core tier going red on 2026-08-14 with a monitor `create_domain` diagnosis, fixed by
`capstone-sbi 1a926b0` + `caplifive-buildroot b098a39` — and `current-next-step.md:39-57` records
that those commits could not be pushed (403 on two inner repos). **Neither commit exists in this
checkout**, and the nested `capstone-sbi` here is `2f772bb`, which has zero `gpoff`. So this tree
predates the gp-carve work entirely rather than carrying its bug.

**The fix that IS available here is one in-tree commit**, `caplifive-buildroot 8c7b973` on branch
`xlang-gp-captable-delivery`, the direct child of the pinned `6912474`, +54/-1 over three files. Read
in full: it does the copy in the **kernel module**, not the monitor —

```c
if (m_args.gp_offset > 0 && m_args.gp_offset < m_args.code_len) {
    unsigned long code_size   = (((m_args.code_len - 1) >> 4) + 1) << 4;
    unsigned long dom_data_off = code_size + MONITOR_SEAL_SIZE;   /* 16 * 96 */
    memcpy((void*)(dom_vaddr + dom_data_off), (void*)(dom_vaddr + m_args.gp_offset), tmpl_len);
}
```

— replicating the monitor's own carve arithmetic instead of changing it, guarded on a new
`gp_offset` ioctl field so every existing domain is untouched. **No monitor change and no vendor
patch is required**, which corrects what was written here before the run.

Two consequences worth carrying:

* **There is no PCC code-window limit in this checkout.** `sbi_capstone.c:301` splits at
  `base + code_size`, so PCC covers the whole image. The 4 KiB truncation that `ISSUES.md:4482`
  describes is a property of the gp-carve monitor, which is not here. MicroPython's 321 KiB of
  `.text` is therefore not a link-time problem at all.
* Applying `8c7b973` means rebuilding `capstone.ko`, `capstone-test.user` and the rootfs. The
  staged copies are from 2026-07-29. Before rebuilding, note that
  `build/build/modcapstone-1.0/module/capstone.c` carries local edits absent from `package/`, and
  buildroot rsyncs `package/` over it.

**Negative-test the delivery on the day it lands**, and do it with `beebs_prime` under
`DOMAIN_GLUE=interp`: it fails today and must pass after, with the generated-glue build of the same
rung as the control. A delivery that silently does nothing looks exactly like one that works, right
up until the descriptor is read.

### The fix is APPLIED and VERIFIED, 2026-08-16

`caplifive-buildroot` checked out on `xlang-gp-captable-delivery` (`8c7b973`), then
`make build A=modcapstone-rebuild`. The rebuild was checked on the ARTIFACTS, not the source, since
this project has had a "fix" leave the firmware byte-identical: `capstone.ko` (14:35) now carries
the `gp-captable` string, `capstone-test.user` carries `gp_offset`, `rootfs.ext2` carries both.

Then the negative test, control first:

| arm | before the rebuild | after |
|---|---|---|
| `beebs_prime`, `DOMAIN_GLUE=generated` (control) | PASS, retval 582955588 | **PASS, retval 582955588** |
| `beebs_prime`, `DOMAIN_GLUE=interp` | FAULT, cause 7 | **PASS, retval 582955588** |

The control is what makes the second row mean something: the rootfs was rebuilt between the two
measurements, and an unchanged control says the rebuild did not break or bless anything by itself.

**Direct evidence that the delivery actually fired**, rather than the run merely passing:

```
Loadable size = 4280, gp_offset = 1000
```

That line is the new `libcapstone`, which locates `.capstone_gp_initdesc` **by section name** and
reports its image offset. A non-zero `gp_offset` is exactly the condition the module's copy is
guarded on. The module's own `pr_info` is not on the console because it is a kernel-log message; the
userspace line is the one that reaches the transcript, and it is sufficient.

**The parent's submodule pointer is deliberately NOT bumped.** `git push` of
`xlang-gp-captable-delivery` returns 403, the same access blocker `current-next-step.md:39-57`
records for `capstone-sbi` and `capstone-opensbi`. Bumping the gitlink to a commit that does not
exist remotely is precisely the failure that file describes as already having stranded
`caplifive-buildroot`'s `components/opensbi` pointer. So the parent still records `6912474`, and a
fresh clone needs the branch checked out by hand until the push access exists:

    cd capstone/caplifive-buildroot && git checkout xlang-gp-captable-delivery
    make build CAPSTONE_CC_PATH="$(realpath ../capstone-c)" A=modcapstone-rebuild

**Local edits that the rebuild discarded**, saved as a diff before it ran: a
`DOMAIN_MIN_FREE (512 KB)` order bump in `build/build/modcapstone-1.0/module/capstone.c` plus
`pr_emerg` tracing, neither of them in `package/`. The tracing is noise. The order bump is
functional and may matter for MicroPython's larger image, but it is a separate decision from this
one and was not bundled into it.

### What is now the shortest path to a running interpreter

1. ~~Settle the interp question, apply the fix, negative-test it.~~ **DONE. The interp glue works:
   `beebs_prime` under `DOMAIN_GLUE=interp` went FAULT to PASS across the rebuild with an unchanged
   control.**
2. Link with the globals offset sized to `.text`, copying `build-sqlite-silicon.sh:1539-1586`.
3. Provide `setjmp`/`longjmp` from `nlrjmp_kernel.h` and the eight mem/str functions from
   `beebs_freestanding_string.c`.
4. Run `print(1+1)`.

## Stage 2 — REPR_CAP: decide the representation, then fix the five loud sites

**Nothing here blocks the build any more.** This stage is now entirely about whether the built
interpreter is *correct*, and it should be read that way: every site below compiles today and
produces an untagged value where MicroPython expects a usable object reference.

The seven sites that stage 1 was chasing are the compiler refusing to treat a capability as an
integer. The fix is a fifth object representation beside upstream's REPR_A..D. Note what is NOT in
that list: reading a tag already works, so the scope is construction and pointer-bit arithmetic.

| file | site | what the source does |
|---|---|---|
| `objlist`, `objstr`, `vm` | `list_pop`, `str_finder`, `mp_execute_bytecode` | build an object word out of an integer constant or offset |
| `obj` | `mp_obj_get_type` | compares against immediate objects, which ARE integer constants cast to pointers; the DAG folds the run of comparisons into `(x & 15) == N` |
| `objboundmeth` | `bound_meth_unary_op` | `xor` of two object words |
| `gc` | `gc_init` | `(uintptr_t)p & ~31`, an ordinary align-down |
| `pairheap` | `mp_pairheap_delete` | `(uintptr_t)p & ~1`, `pairheap.c:36` steals a pointer's low bit as a flag |

The last two look like a toolchain problem and are half of one: the C is ordinary and portable, and
it reaches the backend as an i128 `and` only because DAGCombiner sinks the `zext` into it. Stage 1b
lowered that case rather than refusing it, so both compile now — and both still produce an untagged
value, which is what `(void *)(uintptr_t)x` means here. `gc_init` and `NEXT_GET_RIGHTMOST_PARENT`
will fault on first dereference until the source stops round-tripping pointers through integers.
That is this stage's work, and it is the same work as the object word.

Also worth knowing before designing REPR_CAP: `pairheap.c:34-36` shows the low-bit tagging idiom is
**not confined to `mp_obj_t`**. Any subsystem may steal a pointer bit, so the representation work
needs a way to find those rather than a list of the ones in `obj.h`. That is what stage 3 is for.

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
| 1a, 1b | 1-2 days together | high, both are contained and both have a named site |
| 2 | 1-2 weeks | medium, the design is cheap and the fallout is not |
| 3 | 2-3 days | medium |
| 4 | 3-5 days | medium, NLR under linear capabilities is the unknown |
| 5 | 2-4 days | medium, carve count is the risk |
| 6 | 1 day if 1-5 hold | low, this is where everything unmeasured surfaces |

## Out of scope for this plan

Board runs. Conformance and performance methodology. The allocator study's later arms (per-object
`shrink` at `gc_alloc`, revoke on GC sweep). Any change to the paper.
