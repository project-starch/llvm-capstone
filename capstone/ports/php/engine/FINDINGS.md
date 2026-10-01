# PHP engine in a Capstone domain — state and findings

Phase 0 complete and gated. Phase 2 rung A passes. **Rung B (`zend_startup`) does not yet
complete**; it exhausts the domain stack. Details below so the next session does not
re-derive any of it.

## Measured

| fact | value | how |
|---|---|---|
| Zend required TUs that compile for capstone64 | **34 / 34** | `compile-sweep.sh` |
| `.text` of those | **896,196** | same |
| x86-64 `-O0` baseline for the identical set | 458,460 | corpus `.o` files |
| capstone64 inflation factor | **1.95x** | ratio of the two |
| linked engine image (rung A) MemSize | **1,204,464** (1.15 MB) | `llvm-readobj` |
| module's undeclared ceiling | 2,097,152 | `capstone.c:152-165` |
| largest domain that has ever run in this tree | 1,621,624 | `docs/plans/speedtest1-on-silicon.md:920` |
| usable stack in the engine image | **~2,655 KB** | `rung_B_domain.c -DRUNG_B_MEASURE_STACK` |
| usable stack in a *small* probe image | 124 KB | `probes/stack-depth.c` |
| `sizeof(zend_scanner_globals)` on this target | **176** (0xb0) | `_Static_assert` probe |

**The image-size ceiling is not the blocker.** 1.15 MB fits order 10 with 417 KB spare.
Phase 1 (two-region) is therefore *not* needed for size. It may still be needed for stack.

## Two compiler bugs, both new as far as I can tell

**1. `-fcommon` aborts the backend.** `Unknown section kind` →
`llvm_unreachable` at `llvm/lib/CodeGen/TargetLoweringObjectFileImpl.cpp:631`.
`getSectionPrefixForGlobal` handles Text/ReadOnly/BSS/ThreadData/ThreadBSS/Data/
ReadOnlyWithRel but **not `Common`**. Measured: 6 of 6 affected TUs fail with `-fcommon`,
6 of 6 compile with `-fno-common`. PHP's own `CFLAGS_CLEAN` passes `-fcommon`, so this will
hit any 2000s-era C codebase — `-fcommon` was clang's default until 11.

**2. `-finstrument-functions` crashes clang.**
`Assertion '(i >= FTy->getNumParams() || FTy->getParamType(i) == Args[i]->getType()) && "Calling a function with a bad signature!"'`
at `llvm/lib/IR/Instructions.cpp:759`. The `__cyg_profile_func_enter/exit` hooks' `void*`
parameters do not match the capability pointer type. This removes the obvious tool for
finding deep-stack culprits.

Both deserve `C-nn` entries in `docs/ref/ISSUES.md`.

## Capability-specific hazards hit while porting

- **Implicit declarations truncate capabilities.** PHP builds with
  `-Wno-implicit-function-declaration`; on a normal target an undeclared `dlopen()` merely
  warns and returns `int`, here it truncates a 128-bit capability. Surfaced as
  `zend_extensions.c:34: incompatible integer to pointer conversion ... from 'int'`. Every
  pointer-returning libc function is declared in `stubinc/` instead. **Do not copy PHP's
  flag.**
- **Never size a PHP global by eye.** An opaque `unsigned long[16]` (128 B) stub for
  `ini_scanner_globals` was overrun by `scanner_globals_ctor` (`zend.c:634`) because the
  real struct is **176 B** — eight pointers at 16 bytes each. It cost a debugging cycle and
  the symptom was a stack fault in an unrelated function. Stubs now take the TYPE from
  PHP's headers (`libc/php_capstone_php_stubs.c`).
- **`setjmp`/`longjmp` are ABI-specific.** This build makes `ra` a **capability**
  (`stc ra` / `cjalr zero,0(ra)`), verified from a compiled frame. The existing
  `xlang/lua-cdp/capstone-lua/capstone_setjmp.S` is for the gp-captable ABI where `ra` is a
  scalar; using it would store a capability with `sd` and lose the tag. Ours is
  `libc/capstone_setjmp.S`; `probes/setjmp-probe.c` verifies all five properties including
  that a capability survives the unwind with its tag intact (expect `retval = 207`).
- **`alloca` in a loop.** `zend_API.c:1300` `do_alloca()` inside
  `zend_register_functions`' loop (`:1221`); `zend.h:180` makes that `alloca`. Stack taken
  in a loop is not reclaimed until the function returns. Routed off-stack via
  `stubinc/alloca.h`, which is upstream-sanctioned — `zend.h:183` uses `emalloc` where
  `HAVE_ALLOCA` is absent. **Verified routed** (`U php_capstone_alloca` in `zend_API.o`).
  It did **not** fix the overflow, so it is not the cause.

## SOLVED: the 2.6 MB stack exhaustion was a missing `-target-feature +m`

**Root cause.** The build did not enable the RISC-V **M** extension, so there is no hardware
multiply and *every* multiply becomes a libcall to `__muldi3` -- including the two 32-bit
multiplies inside compiler-rt's own `__muldi3`:

    /* compiler-rt/lib/builtins/muldi3.c */
    r.s.high += x.s.high * y.s.low + x.s.low * y.s.high;

so `__muldi3` calls itself, unboundedly. `ports/sqlite/build-sqlite-capstone.sh` passes
`-Xclang -target-feature -Xclang +m` for exactly this reason; this build did not.

**Evidence, with a positive control.**

| | without `+m` | with `+m` |
|---|---|---|
| calls inside `__muldi3` | 1 to `__muldsi3` + **2 to itself** (verified: `__muldsi3` exists as its own symbol, so this is not a nearest-symbol artifact) | **0** |
| standalone repro | faults, stack exhausted | `retval = 126` -- returned AND `nTableSize == 128` |
| `zend_startup` stages 1-8 | stage 3 fails | **all pass** |

**A 127 KB standalone reproducer exists** and is worth keeping as a compiler-repro:
`probes/hashinit-repro.c` + `probes/hashinit-stubs.c` + `Zend/zend_hash.c` + the freestanding
libc. `zend_hash.c` needs only eight symbols (`_emalloc _efree _ecalloc _erealloc _estrndup
zend_error zend_block_interruptions zend_unblock_interruptions`), so it links without the
engine. One call -- `zend_hash_init_ex(ht, 100, NULL, NULL, 1, 0)` -- exhausts the stack.
**Control that makes it conclusive:** declaring a 2 MB `dom_data` gives ~3.87 MB of stack and
it *still* fails, so it was unbounded recursion and not a stack that was merely too small.

### Why this took so long, recorded so the next bug goes faster

- **`env->pc` is worthless here and misled every early attempt.** QEMU writes it at
  translation-block boundaries. It variously named `zend_extensions.c:90`, `domain_main`, and
  `zif_get_loaded_extensions + 0x40` -- the last of which is `li a6, 0x10`, not a store and
  not illegal. Only `rs1`/`imm`/`size` (bounds faults) or `tval` (illegal instruction)
  describe the real instruction. `fault-locate.py` says this; I did not believe it enough.
- **Depth-triggered watchdogs cannot see a shallow-entry recursion.** Watchdogs in `malloc`,
  `calloc` and `alloca` all had floors of `top - 64KB`, and every one of those functions is
  entered only ~4 frames deep, so none could ever fire. Three separate "it did not trip,
  therefore X is not called" inferences were wrong for this reason.
- **What finally worked was the stage bisect**, which needed no instrumentation: run
  `zend_startup`'s steps one at a time and see which fails. It went from "somewhere in a
  1.4 MB image" to one line in two runs. It should have been the first move, not the last.
- **Symbol lookups need the `.L` labels filtered out but the spans taken from the symbol
  table**, not from `objdump` headers -- `objdump` prints `.Lpcrel_hi*` as function headers,
  so scanning to the "next function" truncates a body at its first local label.

## SOLVED: rung B passes. `zend_startup` runs in a domain

`./build-engine.sh B` then one boot returns **retval = 31** -- `ENTERED|SINKED|STARTED|
TABLES|ALLOC`, nothing missing, no failure bits. `CG(function_table)->nNumOfElements > 0`,
so startup really registered the builtins, and an `emalloc`/`efree` round-trip works after
it. The stage bisect passes **all 10 stages** (`run-stage-bisect.sh 10`).

Neither remaining fault was a capability or codegen defect. Both were ours.

### Stage 9's cause 24 was HEAP EXHAUSTION, not a lost tag

`php_capstone_malloc.c` includes `zend-alloc/zend_capstone_alloc.h`, and inherited its
defaults: **a 16 KB arena and 64 slots**. Those are right for the standalone CRASH-008 and
UAF probes and nowhere near enough for the engine -- `zend_startup` registers hundreds of
hash entries. `zend_arena_carve` then returns NULL, and the first caller that does not check
is `zend_ini_startup`, whose `malloc(sizeof(HashTable))` feeds straight into
`zend_hash_init_ex`. `_zend_hash_init` stores through the NULL `ht` at
`zend_hash.c` -- `sw a0, 0x0(a3)` with `a3 = 0`, **cause 24**.

The dump said so plainly and was misread for a long time: `rs1 = x13, value = 0,
value_hi = 0` is a NULL base register, and the prologue's `stc a0, 0x0(a5)` spilling the
`ht` parameter to the slot the fault reloads means `ht` was NULL **on entry**. A null
pointer, not a stripped tag. "Cause 24" reads as a capability failure and it is also just
what dereferencing NULL looks like.

Now 512 KB / 4096 slots, funded by halving the stack (1 MB -> 512 KB, safe since the `+m`
fix removed the `__muldi3` recursion; measured alloca high-water on the startup path is
under 1 KB). The 2 MB declaration is UNCHANGED on purpose: at `data=3 MB` the **monitor**
faults inside `create_domain` (cause 5 at `pc = 0x8002164c`, `pcc_base = 0x8001a1d0`, a
16-byte read at `0x8008ffd0` against bounds starting `0x8008ffe0` -- one grain below its own
base). That is a monitor-side defect worth filing separately; until then the total must stay
on the geometry that works.

### Stage 8's cause 5 was `ctype.h` evaluating its argument three times

`stubinc/ctype.h` defined the ctype predicates as macros, each naming its argument two to
four times:

    #define isupper(c)  ((c) >= 'A' && (c) <= 'Z')
    #define tolower(c)  (isupper(c) ? (c) + 32 : (c))

`Zend/zend_operators.c:1755` is `*result++ = tolower((int)*str++);`, so `str` advanced
**three times per iteration** and the loop ran off the end. Caught precisely: a 1-byte read
at offset 14 of `"func_num_args"` (13 chars + NUL, bounded to exactly 14 bytes) during
`zend_startup_builtin_functions`. The disassembly of `zend_str_tolower_copy` showed three
`stc` of the incremented pointer for one `lbu` that mattered.

They are single-evaluation `static __inline__` functions now, which is what a real libc
provides -- C requires these to be callable, so the corpus assumes it. Audited: no other
multi-evaluation macro with a side-effecting argument remains in `stubinc/`.

**Worth noting what this was.** The capability bound caught a genuine 1-byte over-read
that no redzone would have flagged, in a 14-byte `.rodata` object, with the exact object
and offset in the fault dump. The bug was in our header rather than PHP, but the mechanism
is the one the whole port exists to demonstrate.

### Three upstream compiler fixes are applied, and none of them was the cause

Cherry-picked from `origin/dev` onto `llvm/lib/Target/Capstone/` only (so the rebuild is a
few objects, not 382 commits of LLVM), and verified live -- machine verifier clean at
`-O0/-O1/-O2`, zero `ADDI` on a frame index, C-46's test passing at `-O0` and `-O2`:

- **C-46** `563e0765953e` -- a direct call's target built LINEAR; a `movc` copy nulls the
  source and the next `cjalr` raises cause 24.
- **C-52** `fc987bb99d8d` -- the local-stack-slot base register was a GPR; spilling it drops
  the tag. Reaches `-O0`: `TargetPassConfig.cpp:1126` runs LocalStackSlotAllocation in the
  `else` branch, i.e. specifically at `-O0`.
- **C-50** `4c407f9456d4` -- a byval copy's frame index built in address space 0. Also
  reaches `-O0`: only the `+8` half needs instcombine's `or disjoint`; the offset-0 half
  emits `ADDI %stack.N, 0` at every level and the verifier rejects it.

They were adopted because C-46 and C-52 matched the cause-24 symptom exactly. They did not
fix it. Keeping them anyway: C-50's shape was genuinely present in this image at `-O0`, and
both zend-alloc suites stay green with them in (CRASH-008 control 200 / CAUGHT cause 7;
UAF control 213 / CAUGHT cause 24), so they cost nothing.

`-capstone-shrink-stack` was also tested OFF and was **not** the cause. It is left ON
(the default): turning it off saves ~33% of `.text` (960,396 -> 645,412) but costs
stack-object bounds, and every corpus bug targeted here is in an `emalloc`'d buffer.

### The harness was wrong too, and that cost more time than either bug

`run-stage-bisect.sh` interleaved a clang compile and a 66-object lld link with each guest
boot, and `exit 1`'d on the first failure. Result: four runs reported four different "first
failing stages" (2, 2, 8, 2), every one of them a **boot-phase** death -- one in the OpenSBI
banner, before Linux -- on a stage that passes by hand in 0.57 s. A passing stage 2 was
reported as the first failure three times.

It is now two phases (build everything, then boot) with one retry per stage and no early
exit, plus a `CAPSTONE_TIMEOUT_MULTIPLIER`. Stages are cumulative, so a spurious FAIL is
detectable: stage 8 passing means stage 7's code ran.

A lost-boot mode **remains unexplained**. It dies right after the loader prints
`Segment size` and before `Domain requirement`, it is not a timeout budget problem (20x
failed where 6x got further), and it is not rootfs corruption (QEMU runs `-snapshot`) or
resource exhaustion. The same image passes standalone and 10/10 in a loop. Retry masks it;
it should be root-caused before these numbers are quoted as stable.

### Available upstream, not taken

- **The 4 MiB domain wall is already solved.** `caplifive-buildroot` `2b8ad05` serves any
  block past the buddy allocator's MAX_ORDER from CMA via `dma_alloc_pages`, tested to
  32 MiB under QEMU with `cma=256M`. This retires the two-region plan entirely. It also
  corrects the earlier objection that `GFP_HIGHUSER` carries no `__GFP_MOVABLE` -- true of
  the buddy path, irrelevant to this one.
- **A QEMU store-address watch**, `origin/diag/store-address-watch`, **+2 commits** from our
  pin: stores reported by address at any privilege with no tag-map query. The direct answer
  to `env->pc` being a translation-block boundary.

No pins were changed.

## Rung C PASSES: the scanner, parser and compiler run in a domain

`./build-engine.sh C` then one boot. Matched pair, both arms on the same image except the
`-DRUNG_C_BAD` source select:

| arm | source | result |
|---|---|---|
| good | `1;` | retval 2503870719: ENTERED SINKED STARTED ACTIVATED COMPILED OPS RETURN EVALTYPE, **2 opcodes**, first = 62 `ZEND_RETURN`, last = 149 `ZEND_HANDLE_EXCEPTION`, 0 errors, no bailout |
| bad | `1+` | retval 527: ENTERED SINKED STARTED ACTIVATED **ERRORS**, `compile_string` returned NULL, no fault |

Two opcodes is exactly right and worth stating, because it is the check that the COMPILER ran
rather than just the scanner: `1;` frees nothing (a constant needs no `ZEND_FREE`), then
`compile_string` appends `zend_do_return` -> `ZEND_RETURN` and `zend_do_handle_exception` ->
`ZEND_HANDLE_EXCEPTION`. `pass_two` also ran, which is the pass that rewrites every opcode
operand. `zendparse`'s 16,028-byte frame -- the largest in the image -- fits the 512 KB stack.

**NO "<?php" PREFIX.** `compile_string` does `BEGIN(ST_IN_SCRIPTING)`, so the scanner starts
already inside script mode, exactly as `eval()` does. The plan says "`<?php 1;`"; for this
entry point that would scan `<?php` as code and fail. The correct input is `1;`.

### What rung C needed that zend_startup does not do: init_compiler()

First attempt faulted **cause 24 at `zend_hash.c:852`** (`p = ht->arBuckets[nIndex]`) with the
7-byte `.rodata` literal `"rung-C\0"` -- the filename argument -- still live in a register.
`zend_startup` builds the PERSISTENT tables; `CG(filenames_table)` is not one of them. It is
created by `init_compiler` (`zend_compile.c:153`), which a SAPI reaches once per request via
`zend_activate()`. `compile_string` -> `zend_prepare_string_for_scanning` ->
`zend_set_compiled_filename` does `zend_hash_find(&CG(filenames_table), ...)`, so without it
the first lookup runs against a zeroed HashTable and `arBuckets` is untagged.

`init_compiler` alone, not the full `zend_activate`: this rung only compiles, and
`init_executor`'s per-request executor state belongs to the rung that runs opcodes.

`compile_string` also COPIES its argument (`tmp = *source_string; zval_copy_ctor(&tmp)`) and
only then lets `zend_prepare_string_for_scanning` `erealloc` the COPY to `len+2` for flex's
doubled NUL -- so a `.rodata` literal is safe to pass, which is what makes baking the script
into the image work.

### longjmp is STILL not exercised, and the reason is a real gap for rung D

I expected the bad arm to unwind through `EG(bailout)`. It does not, and both reasons matter:

1. `zend_error` for `E_PARSE` deliberately does not bail -- it sets `EG(exit_status)` and
   calls `zend_init_compiler_data_structures`, letting the parser return failure
   (`zend.c`, end of `zend_error`). So the bad arm is correct PHP behaviour.
2. More importantly, **`zend_error` never calls `zend_bailout` at all** (grep count: 0). The
   SAPI's error callback does, at `main/main.c:779` in `php_error_cb`. Our `ub_error` only
   counts.

Consequence, and it is not cosmetic: a genuine `E_ERROR` or `E_COMPILE_ERROR` currently does
NOT stop the engine -- it reports and returns, and execution continues past a fatal error.
Harmless while we only compile; wrong as soon as opcodes run. **Rung D's error callback must
call `zend_bailout()` for the fatal types, mirroring `php_error_cb`.** That is also what will
finally exercise `setjmp`/`longjmp` in anger, which remains unverified beyond its probe.

## RUNG D PASSES: PHP EXECUTES

`./build-engine.sh D` then one boot. This is the milestone the plan names "PHP executes".

| arm | source | result |
|---|---|---|
| good | `6*7` | retval 2753791: ENTERED SINKED STARTED COMPILER EXECUTOR EVALED ISLONG **IS42**, zval type 1 `IS_LONG`, **value 42**, no errors, no bailout |
| fatal | `0;function f(){}function f(){}` | retval 799: ... COMPILER EXECUTOR **BAILED ERRORS**, `evaled` false, no fault |

The good arm went through on the first attempt. `6*7 = 42` coming back as an `IS_LONG` means the
opcode dispatch table (populated at runtime by `zend_init_opcodes_handlers`, since 5.0.0 predates
`zend_vm_execute.h` and uses a plain indirect call), the `EX_T` temp-variable ABI, `ZEND_MUL` over
two constants and `ZEND_RETURN` copying a zval out all work in a domain.

Driven through `zend_eval_string` (`zend_execute_API.c:948`) rather than a hand-rolled execute
sequence, so the executor is entered exactly as the engine enters it. With a non-NULL
`retval_ptr` that function wraps the source as `return <src> ;`, which is why the source is an
EXPRESSION and not a statement.

### setjmp/longjmp is now verified IN ANGER, and it needed a fix first

The fatal arm is the first time this port has taken a real `longjmp`. It works: a duplicate
function declaration (`E_COMPILE_ERROR`, deliberately not `E_PARSE`) reached `zend_bailout`,
unwound through `EG(bailout)` and our capability-aware `capstone_setjmp.S`, and landed in
`zend_catch` -- `evaled` false, no fault, no wedge.

That required fixing the gap rung C exposed: **`zend_error` never calls `zend_bailout`** (grep
count: 0); `php_error_cb` does, at `main/main.c:773-781`. Rung D's `ub_error` therefore mirrors
php_error_cb -- bail on `E_CORE_ERROR`/`E_ERROR`/`E_COMPILE_ERROR`/`E_USER_ERROR`, and NOT on
`E_PARSE`, which reports failure by return value instead. Without this a fatal error would
report and return, and execution would continue past it.

### What rung D needed beyond rung C

`init_executor()` (`zend_execute_API.c:117`) as well as `init_compiler()`. `zend_startup` builds
the persistent tables; the per-request executor state -- `EG(symbol_table)`, the argument and
arg_types stacks, the symtable cache, the `EG(function_table)`/`EG(class_table)` aliases -- comes
from `init_executor`. A SAPI reaches both through `zend_activate()`. `zend_execute` itself is
already wired by `zend_startup` (`zend.c:582`, `zend_execute = execute`).

### Budget after rung D, because Phase 3 is the next thing to spend it

    .text 647,684 + .rodata 93,272 + .data 7,508 + .bss 1,088,912 = 1,837,376 loadable
    + 2,097,152 declared                                          = 3,934,528
    ceiling                                                         4,194,304
    SPARE                                                             259,776

Phase 3 needs `ext/standard/url.c` and `ext/standard/var.c`, which are NOT among the 34 Zend TUs
and are new image bytes. At `-O0` on this target they plausibly fit in ~260 KB, so Phase 3 looks
reachable WITHOUT landing the CMA commit -- but it is the first thing that will run out of room,
and `-capstone-shrink-stack=false` is already spent. Excising `PHP_FUNCTION(get_headers)`
(`ext/standard/url.c:592`) is the free reduction: it is the only thing tying `url.o` to the
streams subsystem.

## Phase 3: the triggers RUN but there is NO VERDICT -- blocked on the domain size ceiling

`run-triggers.sh [CRASH-110|CRASH-073|both]` is written and works. What it reports today is
**no verdict**, and that is the correct outcome rather than a failure of the harness: the
CONTROL arm does not complete, and a fault in the fault arm without a passing control proves
nothing (capstone/bug-corpora/README.md).

### What is built and working

* **`ext/standard/url.c` compiles BYTE-IDENTICAL from the corpus tree** against the real
  `php.h` -- no source edit at all. It needed eight more stub headers, which php.h reaches
  for through `main/php_streams.h` and TSRM: `sys/stat.h`, `sys/socket.h`, `netinet/in.h`,
  `arpa/inet.h`, `netdb.h`, `dirent.h`, `utime.h`, `time.h`. 23,264 bytes of `.text`.
* **The plan's prediction about `get_headers` was exactly right.** url.o's only undefined
  symbols outside the engine are `_php_stream_open_wrapper_ex` and `_php_stream_free`, both
  reached only from `PHP_FUNCTION(get_headers)` (url.c:592-645), plus `php_error_docref1`.
  All three are supplied by `libc/php_capstone_ext_stubs.c` rather than by patching url.c,
  so the file under test stays pristine. Excising get_headers is unnecessary for compiling;
  it was only ever about linking.
* **`var_dump` is not needed and is not linked.** Both over-reads are inside
  `php_url_parse`, before anything formats the result, so `ext/standard/var.c` stays out.
* `parse_url` registers through a one-entry `zend_function_entry` and
  `zend_register_functions`; rung E reaches REGISTERED cleanly.

### The control arm faults, and why it is not yet explained

Cause 24 in `memcpy`, called from `_estrndup` (`Zend/zend_alloc.c:403`,
`memcpy(p, s, length)`), with `a0` -- the destination -- an untagged bare integer
(`x10 = 101d9fd60`, inside `.bss`). So `_emalloc` handed back a pointer whose tag was gone
by the time `_estrndup` used it.

Ruled out, each by measurement rather than argument:

* **Not our arena allocator.** A `ZEND_CAP_TAG_GUARD` added to both return paths of the
  allocator in `zend-alloc/zend_capstone_alloc.h` (off by default, so the CRASH-008 and UAF
  suites are untouched) never fired.
* **Not PHP's `_emalloc` for ordinary sizes.** A sweep of `emalloc(1..64)`, then free-all and
  re-request to exercise PHP's own `AG(cache)` path, returns a tagged pointer every time in
  BOTH arms.
* **Not codegen differing between arms.** `_estrndup` disassembles byte-identically in the
  control and fault images, and its prologue uses `stc`/`ldc` correctly.
* **Not the control configuration.** Rung D passes identically in both arms
  (retval 2753791, IS42 set), so `-DZEND_CAP_BOUNDS_REAL_SIZE` is sound in general.

**HEAP EXHAUSTION IS REFUTED.** It was the leading hypothesis and it is wrong.

`ZEND_CAP_ARENA_TRACE` (diagnostic, off by default) reports each exhaustion mode separately
through the fault channel -- `0x5E......` when `zend_arena_carve` cannot carve the request,
`0x51......` when the slot table is full -- because they are different bugs and a shared
marker could not tell them apart. On the failing control run at the default 512 KB arena,
**neither marker fires**: the only `badaddr` in the log is the cause-24 default. The arena
never runs out, and the slot table never fills.

That negative is only worth something with a positive control, so there is one: the same
image built with a 16 KB arena fires `arena bytes exhausted, remaining KB = 0` and then
faults with cause **7**, a different cause entirely. The instrumentation works and the
signal is detectable, so its absence at 512 KB is a real result.

Two consequences:

1. The rung E control fault is **something else** -- see the elimination list below.
2. **The earlier claim that Phase 3 is blocked on the 4 MiB ceiling is WITHDRAWN.** That
   rested on needing a bigger arena to test exhaustion. Since exhaustion is not the cause, a
   bigger arena would not have helped, and the ceiling is not what is stopping Phase 3. The
   CMA commit remains worth landing -- it would let `-capstone-shrink-stack` go back on and
   restore the stack bounds this port gave up -- but it is not the unblock for the triggers.

### Everything static has now been eliminated

Each of these was tested, not argued:

| candidate | how tested | result |
|---|---|---|
| our arena allocator returns untagged | `ZEND_CAP_TAG_GUARD` on both return paths | never fires |
| arena byte exhaustion | `ZEND_CAP_ARENA_TRACE` marker + 16 KB positive control | never fires (control does) |
| slot table full | separate `0x51` marker | never fires |
| PHP's `AG(cache)` holds untagged pointers | rung walks `alloc_globals.cache` after startup+compile | 2 live entries, **0 untagged** |
| `_estrndup` spills `p` at the wrong width | disassembly of its memcpy call site | correct: `stc` store, **`ldc`** into a0 at `7ead8` |
| PHP's `_emalloc` spills `p` at the wrong width | register-to-slot width scan over its whole body | correct: `p` at `s0-0x50` is `stc`/`ldc`; its one `ld` (`7d7fc`) feeds `bnez`, the `if (!p)` null test |

A caution about that last row, recorded because it nearly became a false finding: the first
version of the width scan reported `s0-0x50` as mixed because the tracker did not clear a
register binding on `shrink rd, ...`, so it attributed a *global* scalar load
(`auipc`/`addi`/`cincoffset gp`/`delin`/`shrink`/`ld`/`beqz`, a shape that recurs all through
this image) to the frame slot. An integer `ld` of a pointer is ALSO legitimate on its own when
it only feeds a zero test, which is what both real cases turn out to be. Mixed width is a
lead, not a verdict.

**So the tag is destroyed IN MEMORY.** Both functions store the capability correctly and load
it correctly, the value is right when stored and wrong when loaded, and between the `stc` and
the `ldc` there are intervening calls. A tag that dies in the slot because something else
wrote over it is invisible to disassembly, which is why static means are now exhausted.

### The tag watch was already at our pin, and it gave the exact faulting instruction

No submodule move was needed: `CAPSTONE_TAGWATCH` (with `_LO`/`_HI`/`_VICTIM`/`_GRANULE`/`_MAX`)
already exists in the pinned `capstone-qemu` at `target/riscv/op_helper.c`, and its own comment
describes this situation verbatim -- "the tag goes and the pointer only faults later, somewhere
else, as `requires capability` with no clue who did it". It only reports at `env->priv == PRV_C`,
which is exactly where a domain runs. **Pins unmoved.**

With it armed over the domain stack, QEMU named the faulting instruction outright:

    cincoffset with an UNTAGGED rs1 -- pc=0x101c97198 rd=x10 rs1=x10 val=0x101d9fd90 priv=3

So cause 24 is the `cincoffset` that `memcpy` does on `dst` (`d + i`), not the byte store -- a
correction to the earlier reading of this fault. `val` is the correct block address; only the
tag is gone.

**What the watch rules out.** With `_VICTIM` set to that block, every reported kill has
`victim bounds = (101c00000, 101dcc000)` -- the whole image, i.e. gp-derived pointers spilled
to the stack and later overwritten by scalars. Two of the culprit pcs are PHP's `_emalloc`
storing its own scalar `size` argument (`0x7d62c` is the `sd a1, 0x0(a0)` at the top of that
function). All benign stack reuse. **Not one kill of a NARROW, allocation-bounded capability
covering that block was reported.** If the block's own pointer had been tagged in memory and
then clobbered, that is what we would have seen.

That points away from "a store killed a live capability" and toward "the slot never held a
tagged capability in the first place" -- a 16-byte `ldc` over a granule the capability map does
not record, whose low 8 bytes happen to hold the right address. Which is a different bug shape
from the one being hunted.

**The filter needs to be `_GRANULE`, not `_VICTIM`**, because `_VICTIM` matches any capability
whose bounds span the address and the whole-image capability always does. `_GRANULE` names the
one 16-byte slot, which requires `p`'s stack address in `_estrndup` (`s0-0x60` in its frame) --
obtainable, but it needs another pass, and the kills cluster in only ~672 bytes of stack
(`0x101fff700`-`0x101fff9a0`), so the window is already small.

## ROOT CAUSE CLASS FOUND: an 8-byte union write kills a 16-byte capability in the same slot

`zvalue_value` is a union. Two of its members matter here:

    long  lval;                             8 bytes
    struct { char *val; int len; } str;     val is a CAPABILITY -- 16 bytes

On x86-64 `lval` and `str.val` are BOTH 8 bytes, so writing one and reading the other is the same
bits and PHP's `zval.type` discriminates. **Under 16-byte capabilities they are different sizes.**
An 8-byte store to `lval` overwrites only half of `str.val` and, because the tag lives in a
side table keyed by 16-byte granule, it also DESTROYS THE TAG of the capability sharing that slot.

Observed, with the pc and the store width from the tag watch:

| source | store | effect |
|---|---|---|
| `zend_language_parser.c:2671` -- `yyval.u.constant.value.lval = 1` | **size 8** (`sd`) | kills the granule's tag |
| `zend_language_parser.c:3475` -- `*++yyvsp = yyval` | **size 16** (`stc`) | correct capability copy |

Confirmed at the instruction: `1fd98: li a0, 0x1` then **`1fd9c: sd a0, 0x10(a1)`** -- an 8-byte
store of the literal into the union at znode offset 0x10, which is where `value` begins. Line
3475's copy is `ldc`/`stc` pairs, i.e. correct.

So the parser's own semantic-value handling writes `lval` into a union that elsewhere carries a
capability, and the capability's tag does not survive it.

**ONE LOOSE END, stated rather than smoothed over.** An `sd` at granule+0 would clobber the low 8
bytes too, and our dead pointer kept its CORRECT address. So the copy that was actually read did
not have that store land on its low half. The logs show why that is possible: many kills are
`granule = ...f0` with `store addr = ...f8`, i.e. an 8-byte store at granule+8.
`cap_mem_map_remove_range` rounds DOWN to the granule, so such a store removes the whole granule's
tag while leaving bytes [granule, granule+8) -- the address -- untouched. Address preserved, tag
gone, which is exactly the observed value. Pinning down which copy took which store is a refinement
on the mechanism, not a question about whether the mechanism is real. Every link in the chain is individually
correct, which is why so much inspection found nothing:

  * the scanner stores the string with `movc` + **`stc`** at offset 0 (`17ca0`, `190d4`), and the
    length with `sw` at offset 0x10 -- tag registered;
  * `_zval_copy_ctor` reads it back with **`ldc`** (`6d014`) -- right instruction;
  * `zval` is 48 bytes, 16-aligned, `value` at offset 0 -- no misalignment;
  * `compile_string` uses only `ldc`/`stc` (23/22, zero `ld`/`sd`);
  * `_estrndup`, PHP's `_emalloc`, our `malloc` and `memcpy` all handle the pointer correctly.

The defect is not in any one of them. It is the **union aliasing a capability with a scalar**, which
the plan named as the main porting hazard ("unions containing pointers (`zvalue_value`, `znode.u`,
`temp_variable` -- ~28 union declarations in the required headers)") and which was the one thing
never checked, because every search was for a mishandled POINTER rather than a correctly-handled
SCALAR landing on top of one.

**This also explains the state-dependence.** It fires only when a reduction writes `lval` into a
semantic value whose slot currently holds a string capability, which depends on the grammar path
the input takes -- so no isolated allocate/free/reallocate sweep could ever reproduce it, and two
of the three `_estrndup` callers in the same run pass a perfectly good tagged pointer.

### Why this is NOT the corpus bug, and what it means for the experiment

`parse_url`'s over-read is a Phase 3 result about PHP. This is a PORTING defect in the control
arm: it fires on the stock-bounds build, which is supposed to be the boring baseline, so it blocks
the matched pair rather than being a finding about PHP. It needs fixing before either trigger can
return a verdict.

The fix is not a compiler change. Candidate directions, in order of fidelity:
  * widen the union's scalar members so a write covers the whole 16-byte granule -- cheap, but it
    edits PHP's headers and so weakens "byte-identical from the corpus tree";
  * make the capability-bearing member not share storage (move `str.val` out of the union) -- a
    bigger edit, same fidelity cost;
  * accept it and bound the experiment's claim to paths that do not cross this hazard -- honest,
    but it is exactly the Zend VM paths the corpus triggers use.

That is a design decision about the port's fidelity, not a bug to patch blind, so it belongs with
the project lead.

### LOCALISED to one field read: _zval_copy_ctor's `zvalue->value.str.val`

A three-print probe per `_estrndup` call (p, then s, then `__builtin_return_address(0)`) settles
which operand and which caller. The failing triple:

    Print = Cap(1, 0x7, 0x101d9fea0, 0x101d9fe40, 0x101d9feb0)   p  -- destination, TAGGED
    Print = Scalar(0x101d9fdc0)                                  s  -- SOURCE, UNTAGGED
    Print = Scalar(0x101c5d030)                                  ra -- the caller
    cincoffset with an UNTAGGED rs1 -- val=0x101d9fdc0           the fault, same value

and the triple BEFORE it shows that same address handed out tagged:

    Print = Cap(1, 0x7, 0x101d9fdc0, 0x101d9fd60, 0x101d9fdd0)   p of an earlier call

So a string `_estrndup` allocated and returned TAGGED comes back later as an UNTAGGED `s`. The
tag dies while the caller holds it. Over the run: 17 Cap prints, 10 Scalar.

**The caller is `Zend/zend_variables.c:137`** -- `_zval_copy_ctor`'s
`zvalue->value.str.val = estrndup(zvalue->value.str.val, zvalue->value.str.len)`. So the untagged
pointer is read out of a **`zval`'s `value` union** (`zvalue_value`), which is one of the ~28
pointer-bearing unions the plan flagged as the main porting hazard.

Two earlier callers in the same run pass a TAGGED `s` (`zend_language_scanner.c:4643` and
`:4788`), so this is not every call -- it is this path.

### The static-alignment hypothesis is MEASURED AND DEAD

`cap_mem_map_add` silently no-ops on an unaligned address (`if (addr_is_aligned(addr))` with no
else), so a capability stored to an under-aligned slot registers no tag and reads back untagged
with its address intact -- every symptom here, including the tag watch's silence. Measured on
this target, though, the layout gives it no opening:

    sizeof(zval) 48    _Alignof(zval) 16    offsetof(zval, value) 0
    sizeof(zvalue_value) 32   _Alignof 16   sizeof(char *) 16
    sizeof(Bucket) 128        _Alignof 16

48 is a multiple of 16, `value` sits at offset 0, and a heap zval from our allocator starts at
`base + 96`, itself 16-aligned. So no zval array stride or field offset misaligns the union.
`-Wcapstone-capability-alignment` (on dev, postdating our clang) would still be the systematic
check for the rest of the port, but it does not explain this fault.

### What is left, stated narrowly

The field is 16-aligned and the reader uses the right instruction, so the remaining question is
**who writes `value.str.val`, and with what store**. If some path writes that union member with
an 8-byte store -- plausible where a union is also written as `lval`, a `long` -- then no map
entry is ever added, nothing is killed, and the next read is untagged with the correct address.
That is consistent with every observation and with the tag watch reporting nothing.

The next step is therefore to disassemble the writers of `value.str.val` on the scanner path that
produced this string, rather than any further runtime watching.

### RETRACTED: "p arrives untagged" was WRONG -- csdebugprint disproves it

The section below deduced, from a whole-STACK tag watch finding no kill at `_estrndup`'s `p`
slot, that `p` must already be untagged on arrival. **That deduction is wrong**, and it was
wrong for a scoping reason: the watch window was `0x101dcd000-0x102000000`, the stack only, and
the reasoning treated "no kill in the window" as "no kill anywhere".

`csdebugprint` (`.insn r 0x5b, 0x1, 0x43`, spliced into a diagnostic COPY of `zend_alloc.c` by
`PHP_DIAG_ALLOC=1`) settles it by direct observation instead. Over a full trigger run
`_estrndup` is reached 9 times and **every one receives a TAGGED capability -- 9 Cap, 0
Scalar.** And the faulting value is the very pointer one of them received:

    line 2762  Print = Cap(1, 0x7, cursor 0x101d9fda0, base 0x101d9fd40, end 0x101d9fdb0)
    line 2914  cincoffset with an UNTAGGED rs1 -- val=0x101d9fda0

Same cursor, tagged on receipt, untagged at use. So the tag is lost AFTER `_estrndup` has it.

### Localised: the tag dies inside memcpy, on memcpy's OWN spill of `d`

The faulting pc `0x101c971a0` is `memcpy`, and the instruction before it is
`a7194: ldc a0, 0x0(a0)` -- `d` reloaded from memcpy's own frame slot **with a correct,
tag-preserving `ldc`, returning an untagged value**. memcpy's prologue is also correct:
`movc` the arguments, `stc a1` to `s0-0x30`, `ldc` it back at `a6fec`, and the one `sd` at
`a6fe8` stores `n`, a scalar. A slot-width scan of the whole body shows `d` at `s0-0x60` and `s`
at `s0-0x70`, each `stc`/`ldc` plus one `ld` that feeds the `(uintptr_t)d & 15` alignment
arithmetic -- legitimate, the same shape as a null test.

So within ONE memcpy invocation: `stc` of `d` into `s0-0x60` (which must add a map entry),
then `ldc` of `d` from `s0-0x60` returning tag 0.

A granule-targeted watch (`CAPSTONE_TAGWATCH_GRANULE=0x101fff7f0`, that slot at the faulting
frame's `s0-0x60`) shows the granule is heavily reused and killed repeatedly by 4- and 8-byte
stores -- but from the ALLOCATOR's own frames (`_emalloc`, `zend_arena_carve`), which run and
return BEFORE memcpy's frame exists. Nothing is reported between memcpy's `stc` and its `ldc`.

### What is needed next, and why the current instrument cannot answer it

**The tag watch reports REMOVALS only.** It cannot show whether the `stc` ever ADDED the entry.
The two remaining possibilities are exactly (a) the add never happened, and (b) a removal on a
path the watch does not report -- and they are indistinguishable with a removal-only probe.

So the next instrument must log **both** `cap_mem_map_add` and `cap_mem_map_remove` for a single
granule, in order, which is a few lines in `cap_mem_map.c`. That is a QEMU change rather than an
env var, but it is small, local, and answers (a) vs (b) in one run. The store-address watch
remains the alternative for (b) specifically, since it reports at any privilege.

### Superseded reasoning, kept for the record
### SETTLED: the tag was never killed in p's slot -- p arrives untagged

The granule pass answers it. `_estrndup`'s `p` slot is computable from the fault registers:
memcpy's `sp = 0x101fff7b0` with a `-0xa0` frame puts `_estrndup`'s `sp` at `0x101fff850`, its
`s0` at `sp + 0x80 = 0x101fff8d0`, and `p` at `s0 - 0x60` = **`0x101fff870`**.

A whole-stack tag watch (`LO=0x101dcd000 HI=0x102000000 MAX=40000`) recorded **1751 kills,
COMPLETE, not truncated**. Not one of them is at `0x101fff870`. A tag watch reports a scalar
store only when the map HELD a capability at that granule, so:

**No capability tag was ever destroyed in the slot `_estrndup` reads. The value was already
untagged when it was stored there.** The tag died upstream of `_estrndup`, and NOT by an
in-memory clobber -- which retires the "something wrote over the live pointer" reading for good.

### What the same run does show, and where it points

Filtering the 1751 kills to narrow capabilities (span < 4 KiB) whose bounds CONTAIN the dead
pointer `0x101d9fd90` gives exactly six, all of one object:

    bounds = (0x101d9fd30, 0x101d9fda0)   span 112 bytes,  dead pointer at base + 0x60

    pc 101c96fe0  size 8   memcpy                     (beebs_freestanding_string.c:152)
    pc 101c976ac  size 8   memset                     (beebs_freestanding_string.c:336)
    pc 101c9bc8c  size 8   _emalloc                   (zend_capstone_alloc.h:269)
    pc 101c0fd9c  size 8   zend_language_parser.c:2671
    pc 101c14318  size 16  zend_language_parser.c:3475   (16-byte: REPLACES, does not destroy)
    pc 101c0cfcc  size 4   zend_language_scanner.c:5349

So a 112-byte capability over that block DID exist and was tagged. Every one of these kills is
at `0x101ffb5xx`-`0x101ffc7xx` -- roughly 0x3000 BELOW `_estrndup`'s frame, i.e. in deeper
frames, during the parse. They are stale stack copies being overwritten by ordinary slot reuse,
which is benign.

Read together: the block was allocated and used during parsing, its capability copied around
the stack, then the block was FREED and HANDED OUT AGAIN to `_estrndup` -- and the pointer
returned the second time had no tag. **The suspect is now the free-and-reuse path, not the
fresh-carve path.**

### The MECHANISM, from reading QEMU: the tag is not in memory at all

`capstone_helper.c`: `store_capregval` writes the compressed 128 bits to memory and separately
calls `cap_mem_map_add`; `load_capregval` then takes the tag **entirely** from
`cap_mem_map_query`. So the tag lives only in `env->cm_map`, a side table keyed by 16-byte
granule -- NOT in the stored bytes.

That is exactly the observed symptom: **drop a map entry and `ldc` returns the same 128 bits with
`tag = false` -- the address preserved bit-for-bit, the tag gone.** It also explains why every
static inspection of the generated code came back clean: the codegen is correct; nothing in it
loses a tag.

Everything that could drop an entry was then checked, and each is excluded:

| path | verdict |
|---|---|
| `helper_remove_cap_mem_map` (any plain store) | calls `tagwatch_report` BEFORE removing -- watched |
| untagged `stc` over a tagged granule | also calls `tagwatch_report` ("untagged stc") -- watched |
| `cap_mem_map_clear` | only from `helper_csdebugclearcmmap`, a debug instruction this port never executes |
| map eviction under pressure | none: `add_entry` DOUBLES the heap array, and the header records that the old fixed 512-entry ceiling with its abort was removed for exactly this reason |

And no capability op silently untags: `helper_csshrink` raises `UNEXP_OP_TYPE` /
`UNEXP_CAP_TYPE` / `ILLEGAL_OP_VAL` and otherwise only mutates bounds and cursor, keeping the
tag; `csdelin`, `cssplit`, `csmrev` and `cstighten` assert rather than degrade. A `movc` of a
non-copyable source nulls it, but that yields ZERO, and our dead value carries the correct
address.

### Compiler output: the whole producer chain is clean

`_estrndup` -> PHP's `_emalloc` (`0x7d610`) -> our `malloc` (`0xaba50`) -> our static `_emalloc`
(`0xabc70`) -> arena. All four inspected:

  * `_estrndup` stores `p` with `stc` and loads it with **`ldc`** into a0 at `7ead8` for the memcpy;
  * PHP's `_emalloc` keeps `p` at `s0-0x50` with `stc`/`ldc` (eleven `ldc`s); its single `ld` there
    feeds `bnez`, the `if (!p)` null test, which needs no tag;
  * our `malloc` is a tail call -- `return _emalloc(n ? n : 1)` -- and a0 passes through
    UNTOUCHED to `cjalr zero, 0(ra)`; the `sd`/`ld` on `-0x38(s0)` is the ternary, a scalar;
  * our static `_emalloc` is covered by `ZEND_CAP_TAG_GUARD`, which never fires.

### Where that leaves it, stated as a deduction

Every in-domain removal path is watched and reported nothing at `p`'s granule; the map cannot
evict; no op silently untags; and no instruction in the producer chain drops a tag. By
elimination the `stc` that wrote `p`'s slot **wrote an already-untagged value**, so the tag was
missing one step earlier still.

The one link never DIRECTLY observed is the tag of a0 at the instant PHP's `_emalloc` returns.
Two ways to see it, both cheap:
  * `helper_csdebugprint` exists as a debug instruction -- a print placed around the call
    boundary reports a register's tag and type without touching PHP's source;
  * `tagwatch_report` returns early unless `env->priv == PRV_C`, so anything at another
    privilege is invisible to it by construction. The store-address watch
    (`origin/diag/store-address-watch`, +2 commits) reports stores BY ADDRESS with no map query
    and at any privilege, which is the one instrument that covers what is left.

### The geometry, and why no sweep reproduces it

The 112 bytes is NOT the request. The dead pointer sits at `base + 0x60`, so the layout is a
**16-byte payload behind 96 bytes of header + MEM_HEADER_PADDING**, and the capability spans
`96 + REAL_SIZE(16) = 112` -- exactly what the control arm's bound should be. (A 96-byte header
is worth a look in its own right.) So the failing allocation is `_emalloc(16)`, i.e. an
`estrndup` of a 15-character string, which is one of url.c's path/scheme copies.

Payload 16 was ALREADY covered, clean, by the fresh-then-cached sweep over 1..64. A second sweep
over 64..512 in step 4, allocating / freeing / re-allocating and checking the tag of both the
fresh and the reused pointer, is **also clean -- no size returns untagged**.

**So the fault is STATE-DEPENDENT and no isolated allocate/free/re-allocate sequence reproduces
it.** It needs the real history: the block allocated and used during the parse, its capability
copied across several frames, freed, and handed out again. That is a different line of attack
from sweeping sizes, and the cheap static and sweep-based options are now exhausted.

Two candidates for it, in order of cost:
  * a ring buffer in .bss recording (address, tag) for every pointer our malloc returns, dumped
    through php_fault_report on the fault -- localises the producing call without a pin move;
  * the store-address watch (`origin/diag/store-address-watch`, +2 commits), which reports
    stores BY ADDRESS with no map query and so can see a write to a granule the capability map
    does not track -- the one thing a tag watch cannot do by construction.

**Still available if the granule pass does not settle it:** the store-address watch on
`origin/diag/store-address-watch`, +2 commits from our pin. It reports stores BY ADDRESS with
no map query, so it catches a write to a granule the map does not track -- which is precisely
the shape the tag-watch result above now points at, and which a tag watch by construction
cannot see.

### It could not be tested, and that is the real blocker

Raising the arena costs DOUBLE, because it lives in `.bss`: the model the measurements fit is

    code(non-bss) + .bss + declared_dom_data  <=  ~4 MiB

and the declaration must itself be at least `.bss + .data + stack + cap table`. So every byte
of arena is charged once to the image and once to the declaration. Measured attempts:

| arena | slots | stack | declared | loadable | result |
|---|---|---|---|---|---|
| 512 KB | 4096 | 512 KB | 2,097,152 | 1,837,376 | boots; control faults cause 24 |
| 1 MB | 8192 | 256 KB | 2,097,152 | 2,572,712 | MONITOR fault, both arms |
| 864 KB | 8192 | 256 KB | 1,900,544 | 2,438,032 | MONITOR fault |
| 864 KB | 4096 | 256 KB | 1,949,696 | 2,241,424 | MONITOR fault |

The last row sums to 4,191,120 -- UNDER the nominal 4,194,304 -- and still faults, so the
effective limit is lower once page rounding and the cap table are counted. The monitor's
failure mode is the same internal cause-5 documented above (`pc = 0x8002164c`, a 16-byte read
one grain below its own bounds), not a clean "failed to allocate", which is worth filing.

These ceiling measurements stand as a record of what fits, and the monitor's failure mode
(an internal cause-5 rather than a clean allocation failure) is still worth filing. But see
the section above: **the conclusion originally drawn here -- that the ceiling blocks Phase 3
-- is WITHDRAWN.** It assumed a bigger arena was needed; exhaustion turned out not to be the
cause, so arena size is not what stands between us and a verdict.

## Layout

    stubinc/            ~20 freestanding headers; sys/types.h must supply BSD uint/ulong
                        because zend_hash.h:25 includes it expecting them
    libc/               malloc seam onto the capability allocator, string/ctype/conversion,
                        printf family, OS stubs, PHP stubs, capstone_setjmp.S
    probes/             setjmp-probe.c, stack-depth.c
    compile-sweep.sh    the Phase 0 gate
    build-engine.sh     builds a rung; declares .capstone_domreq and asserts it moved no
                        loaded byte

`container/run.sh` now bind-mounts the corpus read-only at `/corpus` when present.

**The malloc seam is the point of the whole exercise** and must not be "fixed": `ZEND_MM`
stays undefined so `ZEND_DO_MALLOC` is plain `malloc` (`zend_alloc.c:55`), which we supply
from the capability allocator, so every `emalloc` block gets its own bounded capability.
Backing PHP's real `zend_mm` arena instead would put every allocation inside one capability
and catch nothing.
