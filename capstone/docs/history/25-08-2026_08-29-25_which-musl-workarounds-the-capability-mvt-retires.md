# Which musl workarounds the capability MVT retires, and which it does not

Every workaround in `capstone/musl-capstone/` was written around a numbered
compiler defect. With a capability now being `MVT::c128` in its own register
class, most of those defects are gone. This is the measured list, so the port
work is a sequence of deletions with evidence rather than a re-survey by feel.

**Every row was reproduced against the compiler on `capstone-musl`.** Not
inferred from the issue text: the shape named in each workaround's own comment
was compiled, at the optimisation levels the issue names.

## Retired

| Workaround | Its stated reason | Evidence it is gone |
|---|---|---|
| `mruby-probe/patch-parser.py` | `Cannot select: i128 = xor t103, Constant:i128<27>` from `nint(pass?NODE_CALL:NODE_SCALL)` | the exact `new_call` shape compiles at -O2 |
| `libc-ext/string.c`, `strlen.c`, `memcpy.c`, `memmove.c` byte loops | C-28: InstCombine re-widens a pointer alignment test to i128, `Cannot select: i128 = and` | all four spellings the issue names, plus musl's own `memcpy` word loop, compile at -O0/-O1/-O2 |
| `libc-ext/gen-vfprintf-double.py` | C-20: `long double` is 128-bit and so is a capability, so every long-double operation hits the i128 wall | musl's `fmt_fp` shapes compile, `frexpl`/`fmodl`/`LDBL_*`/`va_arg(long double)` included |
| the C-21 clamp in `ExprConstant.cpp` | `(void *)-100` aborts clang, and `AT_FDCWD` is -100 | fixed on this branch, bounded by the address width |
| C-26 vararg address space | indirect vararg pointer built in AS 0, tag lost | fixed on this branch, `va_arg ... ptr addrspace(200)` |
| C-22 at -O1 | an integer selected as a `cincoffsetimm` BASE, `helper_cscincoffsetimm` asserts `rs1_v->tag` | not representable: `CIncOffsetImm` takes `GPCR`. musl's `fmt_u` digit loop compiles at -O1 and -O2, and the base is `s0` with the index in its own integer register |
| C-25, and "a pointer difference is an integer even when neither side is recognised" | the heuristics that guessed which operand was the pointer | the heuristics are deleted; the type says which |
| C-14 reaching-def rule | a live capability proved scalar, `mv` drops its tag | the whole pass is deleted; an integer copy is `ADDI` on `GPR`, a capability copy is `MOVC` on `GPCR`, and nothing rewrites between them |

`locks.c` sits between the columns. Its compiler reason -- `src/thread/__lock.c`
hitting the "cannot materialize arbitrary >64-bit constants" wall -- is gone, the
shape compiles. The stub itself is a **design decision** the file states plainly:
a domain is single-threaded, and this becomes silently wrong the day it gains
threads. Removing it is a separate call from removing the workaround.

## NOT retired

| Issue | Status | What still happens |
|---|---|---|
| **C-19** | **still open**, workaround still required | `return callee(...)` at -O2 compiles to `cjalr ra` and then *nothing* -- no reload of `ra`, no stack teardown, no return. The function falls through. `-fno-optimize-sibling-calls` restores the epilogue, verified both ways |
| **C-23** | still open | `&weak_undefined != 0` is always true: the address is `cincoffset gp, off`, which is never null |
| **C-20 runtime half** | tractable, not free | the seven soft-float symbols `long double` needs -- `__addtf3 __subtf3 __multf3 __eqtf2 __netf2 __fixunstfsi __floatunsitf` -- must come from somewhere |

The C-20 runtime half has an answer the README did not have. It said the
builtins were "unusable on this target in compiler-rt as well as musl: every
128-bit builtin fails with the same backend assertions, because i128 is both a
capability and a `long double`". **All eleven of compiler-rt's 128-bit soft-float
builtins now compile for capstone64** -- `addtf3 subtf3 multf3 divtf3 comparetf2
extenddftf2 trunctfdf2 floatsitf floatunsitf fixtfsi fixunstfsi`, 11 of 11. So
`long double` can have a runtime, which unblocks `strtod` as well.

## `cap-copy.c` is not a workaround and must stay

Its byte-loop prohibition is an architectural fact, not a compiler limitation: a
capability is 16 bytes plus a tag beside the memory, and only a capability-wide
load/store carries the tag. Sixteen byte moves deliver the right bytes with the
tag cleared, and the pointer then faults arbitrarily far away. Nothing about the
capability MVT changes that.

## Order this suggests

1. Delete `patch-parser.py` and let mruby's real parser compile.
2. Drop the byte-loop string members and let musl's own word-at-a-time files in.
   This is the largest single gain: it is the real reason nine `src/string` files
   were excluded, and `cap-copy.c` says so.
3. Build compiler-rt's soft-float for the target, then retire
   `gen-vfprintf-double.py` and un-stub `strtod`.
4. Re-run the survey. Its `CONTROL_MUST_FAIL`, `src/math/fmodl.c`, is expected to
   flip, which makes the harness report ERROR by design; pick a new must-fail
   file at that point rather than before.
5. C-19 and C-23 stay on the compiler side. C-19 is the more serious of the two:
   a silent fall-through, worked around by a flag that every build must remember.

Step 4 needs a musl tree. `musl.libc.org` was unreachable from this machine, and
substituting another source would bypass the SHA-256 pin `fetch-musl.sh` exists
to enforce.
