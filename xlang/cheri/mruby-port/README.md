# purecap mruby — IT RUNS

Kept because the result is reportable either way, and so the next attempt
starts from here instead of rediscovering it.

**Status (2026-08-01): purecap mruby EXECUTES RUBY under CheriBSD.**

    B_open_full=ok
    C_arith_ok
    HELLO_FROM_PURECAP_MRUBY      <- real Ruby, real interpreter, purecap
    D_puts_ok
    [2, 4, 6]                     <- blocks, procs, arrays
    E_blocks_ok
    F_trigger_ran                 <- xlang/repro/10/trigger.rb executed
    G_all_ok  exit 0

Four changes got there. Two are ABI flags, two are upstream-supported config
switches, and only one is a source edit:

| Change | Kind | Fixes |
|---|---|---|
| `MRB_STR_EMBED_LEN_BIT` 5 -> 6 | one-line source | `_Static_assert` at build time |
| `-ftls-model=initial-exec` + `-cheri-tgot-tls` | ABI flags | `ld-elf.so.1: Traditional TLS not supported` |
| `-DMRB_USE_METHOD_T_STRUCT` | upstream config | `PROT_CHERI_TAG` at `vm.c:561` |
| `-DPOOL_ALIGNMENT=16` | upstream config | `BUS_ADRALN` at `parse.y:125` |

`build_config_purecap.rb` here carries all of them. `probe_run_ruby.c` is the
proof: it opens a VM, evaluates arithmetic, `puts`, blocks, and loads a corpus
trigger, with SIGPROT and SIGBUS handlers installed so any fault is reported
with its cause and address rather than a bare signal.

## The second root cause: parser pool aligned to 8

After the method fix, startup completed and `mrb_load_string` faulted with
**`BUS_ADRALN`** at `parse.y:125`, `c->car = car;` — storing a capability into
an AST cons cell. `src/pool.c` explains it:

    #define POOL_ALIGNMENT 8      /* capabilities need 16 */

The parser's pool allocator hands out 8-aligned cells whose `node*` fields are
16-byte capabilities under purecap. `#ifndef`-guarded, so `-DPOOL_ALIGNMENT=16`
fixes it with no source change.

Note this one the compiler DID point at: the `parse.y` provenance warnings
(`a->cdr = (node*)newlen`) are in exactly this file. Warnings missed the method
bug but would have aimed a porting effort at the parser.

## IMPORTANT: running a corpus trigger is not yet a MEASUREMENT

`F_trigger_ran` with exit 0 under all three revocation configs is **not** a
CHERI MISS verdict, and must not be quoted as one. `xlang/repro/10/run.sh` records
that this trigger "runs to completion, exit 0" without ASan on ordinary
hardware too, because the stale write lands on still-mapped memory. So exit 0
cannot distinguish "CHERI did not catch it" from "the defect did not occur".

To turn real-mruby runs into measurements each trigger needs an oracle that is
observable WITHOUT a sanitizer — the role the shims' deterministic geometry
plays today. Options: instrument `stack_extend` to report whether the stack
actually moved, or choose corpus rows whose defect produces a visibly wrong
value rather than a silent stale access.

## Bring it up

```bash
./build-purecap-mruby.sh            # fetch/patch/build, then VERIFY
./build-purecap-mruby.sh --probe    # also build the smoke test
MRUBY_SRC=/path/to/mruby ./build-purecap-mruby.sh   # use an existing tree
```

Needs the CHERI SDK and purecap sysroot from
`capstone/tests/cheri-baseline/provision-cheri-vehicle.sh`, plus ruby and rake.
Verified end-to-end from a clean, unpatched mruby tree: it applies the source
change, builds, and checks the result is `cheriabi` with zero TPREL
relocations — a binary can build and still be hybrid, and TPREL is what rtld
refuses at load time, so neither is assumed. Re-running is idempotent.

It builds; it does not run. Running needs a CheriBSD guest — the script prints
the staging and `cheri-run.py` commands.

## Reproducing the diagnosis

`probe_init_stages.c` is the stage-4 probe: it replicates `mrb_init_core`'s
body with a marker after each subsystem init, so the last marker printed names
the faulting one. Build it against the purecap libmruby and run it in the
image:

```bash
$SDK/bin/clang --target=riscv64-unknown-freebsd -march=rv64gcxcheri \
  -mabi=l64pc128d --sysroot=$ROOTFS -mno-relax -ftls-model=initial-exec \
  -cheri-tgot-tls -O0 -g -I<mruby>/include -I<mruby>/build/purecap/include \
  probe_init_stages.c <mruby>/build/purecap/lib/libmruby.a -lm -o probe
```

## Build

```bash
CHERI_SDK=$HOME/cheri/output/sdk \
CHERI_SYSROOT=$HOME/cheri/rootfs-purecap \
  rake MRUBY_CONFIG=build_config_purecap.rb
```

`build_config_purecap.rb` cross-builds against the CheriBSD distribution
rootfs as sysroot (the same one the shims use). `mruby-purecap-embed-len.patch`
is the single source change.

## The three changes, all ABI-level

| Change | Why |
|---|---|
| `MRB_STR_EMBED_LEN_BIT` 5 → 6 | `RSTRING_EMBED_LEN_MAX` is `4*sizeof(void*) - 5` — 27 at 8-byte pointers, **59** at 16-byte capabilities, which no longer fits a 5-bit length field. Fails as a `mrb_static_assert`: "pointer size too big for embedded string". |
| `-ftls-model=initial-exec` | purecap CheriBSD requires it. |
| `-cheri-tgot-tls` | purecap uses capability TGOT TLS. Without it the binary keeps `R_RISCV_TLS_TPREL64` relocations and `ld-elf.so.1` refuses it: "Traditional TLS not supported". Check with `llvm-readelf -r … \| grep TPREL` — it must be empty. |

None of these touch the allocator, the GC, or object lifetime, which is what
keeps this a **port** rather than a rewrite of the thing under measurement.

mruby's default value representation needed no change: it already holds a real
`void *` instead of boxing pointers into a word.

## Where it stops — DIAGNOSED (2026-08-01)

Exit 162 = SIGPROT, a capability fault. Bisected in four boots, each probe
printing markers between stages so the last marker names the failing step.
Control throughout: the purecap C shims run clean in the same image on the
same kernel (`sanity_mock` rc=0), so the vehicle is sound and mruby is what
faults.

| Probe | Result |
|---|---|
| 1 | faults inside `mrb_open()` — never reaches parse, codegen, exec or I/O. "Faults on `puts 1`" was misleading: no Ruby is ever executed. |
| 2 | inside `mrb_open_core`, **not** `init_mrbgems` — so a smaller gembox is no workaround. `sizeof(void*)`=16 and `malloc(sizeof(mrb_state))`=18144 both fine. |
| 3 | `mrb_gc_init` **ok**, context alloc **ok** → fault is in `mrb_init_core`. The GC initialises cleanly, which rules out the packed-GC-struct theory. |
| 4 | 16 of 17 subsystems ok — including `mrb_init_string`, `mrb_init_hash`, `mrb_init_class`, the ones carrying the pointer-size assertions. Fault is in the last: **`mrb_init_mrblib`**. |

`mrb_init_mrblib` is `mrblib_proc_init_syms()` then **`mrb_load_proc(mrb,
mrblib_proc)`** — it executes the Ruby standard library, embedded as
`static const mrb_irep` structs holding real C pointers. Nothing is parsed, so
this is not a binary-format or pointer-reconstruction problem.

**Conclusion: mruby's object model ports cleanly to purecap; the fault is on
FIRST BYTECODE EXECUTION.**

## ROOT CAUSE (probe 5): method pointers packed into an integer

A `SIGPROT` handler reported **`si_code=2` — `PROT_CHERI_TAG`**, with the
faulting value `0x1dc784 [rxR,0x100800-0x2a5000]`: a plausible code address
carrying **no tag**. Symbolized, that is `src/vm.c:561`

    val = MRB_METHOD_CFUNC(m)(mrb, self);

an indirect call through the method table. `include/mruby/proc.h` explains it:

    #define MRB_METHOD_FROM_FUNC(m,fn) ((m)=(mrb_method_t)((((uintptr_t)(fn))<<2)|MRB_METHOD_FUNC_FL))
    #define MRB_METHOD_FUNC(m)         ((mrb_func_t)((uintptr_t)(m)>>2))

mruby stores a C function pointer **shifted left 2 with flag bits in the low
bits**, then shifts back to call it. Classic interpreter pointer-tagging,
invisible on x86. On purecap `uintptr_t` is `__uintcap_t`, so the cast itself
is harmless — but shifting the address far outside the capability's bounds
makes it unrepresentable and the hardware **clears the tag irreversibly**.
Shifting back restores the address, not the authority.

**Fix, upstream-supported and config-only:** build with
`-DMRB_USE_METHOD_T_STRUCT`. The `#else` branch of that same header keeps the
function pointer and the flags in separate struct fields, so no pointer is
ever round-tripped through arithmetic. Now in `build_config_purecap.rb`.

### Would compiler warnings have caught it? No — tested, not assumed.

`why_warnings_miss_it.c` reduces mruby's pattern to seven lines. Compiled with
**every** CHERI diagnostic plus `-Wall -Wextra`:

    clang --target=riscv64-unknown-freebsd -march=rv64gcxcheri \
          -mabi=l64pc128d --sysroot=$ROOTFS -Wcheri -Wall -Wextra -fsyntax-only

the compiler is **completely silent**. So `-Werror` would not have helped.

The reason is structural: on purecap `uintptr_t` IS `__uintcap_t`, so
`(uintptr_t)f << 2` is legal capability arithmetic. Whether the shifted
address becomes unrepresentable — and the tag is cleared — depends on the
RUNTIME address, and the value round-trips through a typedef'd integer, into a
struct field, across functions, so there is no local expression to condemn.
The same holds for alignment faults: whether a field is 16-aligned depends on
runtime layout.

CHERI clang did warn elsewhere — `cast from provenance-free integer type to
pointer` (3x in `parse.y`) and `binary expression on capability types ... not
clear which should be used as the source of provenance` (in `class.c`'s method
cache). Those are real hazards worth fixing, just not the one that fired.

Practical guidance: enable `-Wcheri` (chiefly `-Wcheri-provenance` and
`-Wcheri-capability-misuse`; `-Wcheri-inefficient` is performance noise), fix
every hit because the list is finite and each is a genuine hazard — and then
still expect to debug at runtime. "Compiles purecap" and "runs purecap" are
separate milestones, which is why the signal-handler + llvm-symbolizer loop
below matters more than the compiler: it localised both faults in about one
boot each.

## NEXT FAULT (after the method fix): SIGBUS

With `MRB_USE_METHOD_T_STRUCT` the tag fault is gone and startup now dies with
**SIGBUS (138)** instead — the misaligned-capability class, the same wall
upstream SQLite hits inside `sqlite3_open`. Expect capability-bearing fields
sitting at 8-aligned offsets; the sqlite corpus fixed its instances by
16-aligning the offending structs.

## Next step if this is picked up

Chase the SIGBUS the same way the SIGPROT was chased — the technique works
and is cheap:

1. Catch SIGBUS in-process (`probe_init_stages.c` pattern plus a handler),
   print `si_code` and `si_addr`, symbolize the address against the binary.
   `BUS_ADRALN` confirms misalignment and `si_addr` names the field.
2. 16-align the struct that owns it. Alignment fixes are local and do not
   change semantics, which is why they were acceptable for SQLite too.
3. Repeat. Both corpora suggest these come in small numbers, not hundreds.

Getting this to run unblocks the corpus's **worst-case** CHERI number: real
mruby recycles dead objects on a per-page free list without returning them to
the allocator, which is exactly the case revocation cannot observe.

