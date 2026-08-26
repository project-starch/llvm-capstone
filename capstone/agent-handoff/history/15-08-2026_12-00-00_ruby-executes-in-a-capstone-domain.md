# Ruby executes in a pure-capability Capstone domain, 2026-08-15

**Result, QEMU, verified.** Reference mruby, unmodified except for one line, running
precompiled Ruby bytecode in a `capstone64-unknown-elf` domain. Six constructs of
increasing complexity, each checked against its expected value in one boot:

```
MRUBY RUNG 1 nil:          returned -1 (want -1)
MRUBY RUNG 2 integer:      returned 1 (want 1)
MRUBY RUNG 3 add:          returned 3 (want 3)        method dispatch on Integer
MRUBY RUNG 4 empty array:  returned 0 (want 0)        object allocation
MRUBY RUNG 5 array store:  returned 7 (want 7)        array store and load
MRUBY RUNG 6 while loop:   returned 210 (want 210)    jumps, counter, arithmetic
MRUBY STAGE 7 DONE: allocs=275 peak=161953
```

Harness: `capstone/musl-capstone/mruby-probe/run-mruby-stages.sh`.

**COMPLETE, after C-26 was fixed the same day.** The full interpreter runs:

```
MRUBY S2: mrb_open ok                     <- mrblib loads, the whole Ruby-level stdlib
MRUBY S3: irep executed
MRUBY S4: t[19] == 400
MRUBY S6: parsing Ruby source             <- mrb_load_string, the PARSER
MRUBY S7: parsed source produced 400
MRUBY S5: state closed                    <- GC teardown, no fault
MRUBY MEM at-exit: requested=178750 peak=178750 calls=755 fails=0
```

So: bytecode AND source, standard library AND core, with a clean teardown. 755 allocations,
179 KB peak, no allocation failure. The bytecode route is no longer a requirement -- it was
proposed to get around a parser that would not compile, and it stays only as the cheaper
image (1.91 MB against 2.46 MB).

## What it took

Four config macros, all upstream-supported, and **one** source line.

| macro | why |
|---|---|
| `MRB_NO_BOXING` | `boxing_word.h` packs an `mrb_value` into one 64-bit word by tagging the pointer; its static assert fails outright at 16-byte pointers. **Ours** -- the CHERI port did not need it. |
| `MRB_NO_DIRECT_THREADING` | the VM's computed-goto dispatch is a second absolute-addressed jump table, the same mechanism Lua needed `LUA_USE_JUMPTABLE=0` for |
| `POOL_ALIGNMENT=16` | from the CHERI port: the parser pool hands out 8-aligned cells holding capabilities |
| `MRB_USE_METHOD_T_STRUCT` | from the CHERI port: the method table otherwise stores a tagged pointer |

The one line is in the parser, `parse.y:504`: `nint(pass?NODE_CALL:NODE_SCALL)` folds to
`27 ^ (pass != 0)` and, because the result is cast to `node *`, the xor lands at i128.
Routing it through an `enum node_type` first keeps the fold at integer width.
`mruby-probe/patch-parser.py` applies that into the build directory and fails loudly if it
ever stops matching; the mruby tree stays byte-identical.

**The parser RUNS** (S6/S7 above): `mrb_load_string` over the same four-line chunk, so
bytecode and source differ in exactly one thing -- who turned the text into an irep. Both
produce 400.

## Six defects, all ours, all fixed

| | |
|---|---|
| `jmp_buf` 208 bytes and 8-byte aligned, where `capstone_setjmp.S` writes 224 and needs 16 | fixed |
| the libc archive built without `-fno-jump-tables`, 15 members with switch tables | fixed |
| `.bss` inside the loaded segment: 262 KB of zeros copied over 9p | fixed |
| **C-25** pointer difference required tagged operands, so `NULL - NULL` faulted | fixed |
| **C-26** `va_arg` of a by-reference struct loads the reference with `ld` | fixed |

Plus QEMU: `helper_cslcc` asserted on an untagged operand, killing the machine, so the monitor
never reported cause/pc. It now raises Unexpected operand type (24) per the spec. Without that
change C-25 could not have been localised past "somewhere in mrb_top_run".

## Two system properties nobody had written down

**Domain images are capped near 4 MB**, not by hardware and not by the monitor -- which asks
only for 16-byte alignment -- but by the module: `__get_free_pages(GFP_HIGHUSER,
dom_pages_log2)` is a buddy allocation, and `MAX_ORDER` cuts in at order 10. A 6 MB heap made
it order 11 and the allocation failed silently: `pr_alert` into a ring buffer the driver had
already muted with `dmesg -n 1`, and `dom_id` left unset, so the run looked like it simply did
nothing.

**9p demand paging cannot load a large domain.** The guest loader mmaps the `.dom` and memcpys
out of it, so read straight from the share every 4 KiB page is an RPC under TCG with
`cache=none`. A 1.35 MB image did not finish in 900 s, nor in 1800 s. Copied to `/tmp` first
and loaded from there: **under one second**. Every domain run should copy first; the run
scripts here now do, and stamp T0/T1/T2 so the phases stay separable.

## How it was localised, because the method mattered more than any guess

Every one of the following was refuted by a measurement after being the most plausible
explanation at the time: the image is too big, the heap is too small, our first-fit allocator
is quadratic, mrblib's bytecode is too large. The eventual causes were in none of those places.

The chain that worked, each arm differing from the last in one thing:

1. `cp` to local storage vs. loading over 9p -- refuted image size, and fixed the load.
2. 256 KiB vs 768 KiB heap -- identical fault pc, refuted heap exhaustion.
3. First-fit vs an O(1) bump arena -- identical wedge, exonerated our allocator.
4. Staged arms returning markers: `mrb_state`, `mrb_gc_init`, then sixteen of seventeen
   `mrb_init_*` all return. mruby's **whole C object system works here.**
5. Our own four-line irep instead of mrblib's -- same fault, exonerating mrblib.
6. `mrb_read_irep` and `mrb_proc_new` return, `mrb_top_run` faults: the VM, not the loader.
7. The bytecode ladder above, once C-25 was fixed.
8. `mrb_funcall_id` argc=0 returns, argc=1 faults; then the same shape in ten lines without
   mruby at all.

**Nine boots were spent before step 4, one hypothesis at a time.** That is exactly what
CLAUDE.md's "BATCH VARIANTS, and make every run RETURN" section exists to prevent, and it was
written after the same mistake cost six board sessions. After switching to batched arms, one
boot did what four had.

## Three instrument failures, all mine, all worth the space

**A harness that hid build errors.** `run-mruby-stages.sh` sent build output to `/dev/null` and
left the previous `.dom` in the share directory, so a failed build ran the STALE image and
reported as if it were the new arm -- three times. Fixed with three checks: remove the `.dom`
first, do not hide the build log, and verify the file exists afterwards. The third is not
redundant: a build can exit 0 and produce nothing.

**A circular base derivation.** A fault pc was mapped to an instruction by assuming which
instruction faulted and deriving the load address from that. The staged arms now report their
own load address.

**Two wrong drafts of the QEMU change**, both caught before building: raising 25 from memory
instead of 24 from the spec -- the mislabelling ISSUES.md blames for three misdirected
investigations -- and carving out `imm == 0` to answer "is this valid?" with 0, which would
have moved QEMU away from the specification to make one program work.

## What this changes for the corpus

`xlang/RESULTS.md` gives as the first reason for measuring through shims that purecap mruby
took four changes and only one of nine pinned versions is proven. That reason is unchanged.
What has changed is that the **libc obstacle is gone** and mruby is now a build rather than a
port on our column too. Upgrading the twelve Ruby rows from shim to real interpreter no longer
needs anything from the interpreter. What remains is: the nine pinned versions ported (our
config is validated on one), `fstat`/`lseek` for the gem rows, and the revoke arena scaled
from 64 KiB and 64 slots to what a real VM does -- now a measured number, 755 allocations and
179 KB. That last one is the question the shims cannot answer at all, and it is the one worth
doing next.


## The corpus mechanism runs; an actual corpus ROW does not, and the reason is the emulator

Six of the twelve Ruby rows share one mechanism -- a cached interior pointer into the VM stack
across a re-entrant Ruby callback that reallocates it. Written against mruby itself, on the
revoking arena, it behaves exactly as the corpus predicts for that class:

```
control (xlang_set_no_revoke)   CDP 3: stale read COMPLETED -- not blocked      MISS
revoke-on-free                  [CAPSTONE] capability fault: cause = 24         BLOCKED
```

The control is the load-bearing arm: it shows the stack really moved and the stale read was
reachable, so the other arm's fault is about revocation rather than about a pointer that was
never stale.

**An actual row is a different thing and does not run yet.** Row 10 (CVE-2022-1106) was the
candidate, because its pinned mruby commit is the tree this port was validated on and its
trigger is pure Ruby, so `embed-ruby.py` can carry `xlang/repro/10/trigger.rb` into the domain
verbatim. Two arena sizes, two different walls:

* **512 KiB** -- the CONTROL arm faults in `mrb_gc_mark`. The trigger recurses 150 deep
  allocating a string and an array per frame, and the revoking allocator never reclaims, so the
  arena runs out and mruby walks into the result.
* **4 MiB** -- the control gets much further (the allocation heartbeat reaches 576) and then
  **QEMU** asserts in `cap_mem_map.c`: it can track capability tags across at most 2 MiB, in a
  linearly-scanned array. That is **I-1**, and it is an emulator limit, not a defect in the
  compiler, the monitor or mruby.

So the row is blocked between two walls: an allocator that never reclaims needs a big arena, and
the emulator cannot track a big arena. Silicon has neither limit.

**A retraction on the way there.** "Identical pc and badaddr at eight times the arena, so
exhaustion is refuted" was stated from a run that never happened -- the driver was invoked with
a relative path from the wrong directory, `rc=2`, and the lines read back were the PREVIOUS
run's. The identical addresses were not a finding; they were the proof that it was the same
data. This is the second instance of the same shape in one session, the first being a stale
`.dom` executed three times. Both are "output that looks like a result but came from an earlier
state", and both were caught only by looking at a timestamp.

## Reducing the trigger's depth does NOT unblock row 10, and that is now measured

The six "template" rows all call `recurse(150)`, which a non-reclaiming allocator cannot
afford. The obvious move was to find the smallest depth that still arms the bug, with the
matched pair as its own validation: revoke faults + control completes at the same depth would
mean the stale access happened and revocation stopped it.

`MRUBY_PROBE_ROW_DEPTH` builds that ladder. **It did not work, and the reason is not the
depth.** At depth 3 the CONTROL arm -- revocation DISABLED -- faults in `mrb_gc_mark + 0x50`,
the same symbol the very first 512 KiB attempt hit. So:

* arena exhaustion is not the only blocker (depth 3 exhausts nothing);
* the depth is not the variable;
* and the control arm is not clean, which invalidates any verdict from the revoke arm.

**A hypothesis, marked as one, and then REFUTED -- the paragraph that stood here is withdrawn.**
It read that `xlang_set_no_revoke()` turns off revocation only, so the control still SPLITs per
allocation and keeps exact bounds; that the control was therefore "exact bounds, no revocation"
rather than "ordinary malloc, no revocation"; and it concluded that the two configurations the
corpus compares "are not separable with this allocator on this workload". **That conclusion was
wrong and is retracted.**

What refuted it was a THIRD arm with neither property -- row 10's trigger against the plain libc
allocator, no per-allocation bounds and no revocation at all. It faults too, at `mrb_class +
0x254`. A symptom that does not move across three allocators is not about the allocator.

**The actual cause was in our libc**, and it is now ISSUES.md **C-29**: `memcpy` copied byte at
a time, and a byte loop strips the TAG off every capability it moves. mruby grows its VM stack
by memcpying the old stack into a larger allocation (`stack_copy`, `src/vm.c`), so after any
growth every object pointer on the stack was untagged -- hence `mrb_class` on the next method
call and `mrb_gc_mark` on the next collection. Small chunks never grow the stack, which is why
everything above this line passed.

**The design error worth keeping.** The two arms differed from each other in exactly one thing,
which is the rule, but they SHARED a property that neither of them isolated: both used the
revoking allocator. A matched pair can only separate the variable it varies; it says nothing
about anything both arms have in common. The third arm cost one build and settled it.

## Row 10 now RUNS, and the trigger fully arms

With C-29 fixed, `xlang/repro/10/trigger.rb` runs verbatim to completion in every allocator arm.
The readout that says so is `$arr.size`, asked in Ruby after the fact so the corpus's file stays
untouched: the trigger pushes two objects per recursion level, and

```
ROW: $arr.size = 302        libc allocator -- all 151 levels, the recursion completed
ROW: trigger COMPLETED without a capability fault
```

**A clean completion here is the CORRECT result for that arm, not a null one**, and the corpus
says so itself in `xlang/repro/10/run.sh`:

> `=== Running under RISC-V QEMU (observed: exit 0) ===`
> Without ASan the stale OP_RANGE write lands inside the old, still-mapped stack allocation.

So the libc arm reproduces the corpus's own documented RISC-V behaviour: mechanism armed, stale
write performed, nothing catches it. That is a MISS in the corpus's sense.

**The stale write is present in OUR generated code**, which was checked rather than assumed.
`regs` is a macro (`#define regs (mrb->c->ci->stack)`, vm.c:1241), so the C question is whether
the compiler takes the destination address before or after the call in

```c
regs[a] = mrb_range_new(mrb, regs[a], regs[a+1], FALSE);   /* vm.c:2822 */
```

In our -O0 build of the real vm.c it takes it BEFORE and spills it across the call:

```
cincoffset a3, a0, a3     # &regs[a], from the pre-call stack
stc        a3, -704(a5)   # saved across the call
cjalr      ra, 0(a5)      # mrb_range_new -> <=> -> recurse(150) -> stack_extend
ldc        a2, -704(a0)   # the STALE pointer back
```

A ten-line program in the same shape says the same thing for host clang at -O0, -O1 and -O2, so
the optimisation level is not the variable. `r_check` -> `mrb_cmp` -> `mrb_funcall_id(<=>)`
confirms the re-entrant callback is reached for two non-numeric operands.

## On the REVOKING arena the workload does not fit, and "COMPLETED" was hiding it

The same trigger, same everything, on the revoke-on-free arena:

```
ROW: the trigger raised a Ruby exception: <no message>
ROW: $arr.size = 104                     <- 52 of 151 levels
MRUBY ARENA after-row: carved=4122752    <- of 4194304
```

`stack_extend_alloc` gets NULL from `mrb_realloc_simple` and raises `mrb->stack_err`, the
preallocated `SystemStackError`, which carries no message -- so it prints as `<no message>`, and
before the `$arr.size` readout existed it printed as **nothing at all**. Every earlier run of
this arm reported `ROW: trigger COMPLETED without a capability fault` having executed a third of
the workload. **A completion marker that cannot tell "finished" from "gave up" is not a result**,
and this one had already been read as one.

It also explains why the revoke arm did not fault: the exception is raised INSIDE `<=>`, so
`mrb_range_new` never returns, so the stale write at vm.c:2822 is never executed. There was
nothing for revocation to catch.

## Why recycling the arena would not have helped this workload

The obvious fix is to make `rof_free` keep the handle REVOKE hands back -- the spec's revocation
section confirms the type transitions, so the block is recoverable. It would not help here, and
the reason is architectural rather than a matter of effort:

**`SPLIT` has no inverse.** It is in the instruction list (`insn-list.adoc:31`); there is no
join, merge, coalesce or combine anywhere in the specification. An allocator that makes each
allocation independently revocable by SPLITting it off a linear arena therefore carves that arena
irreversibly: dead blocks can be handed out again, but two adjacent dead blocks can never become
one. `revoke_on_free_alloc.h` says as much about itself ("the arena only ever shrinks from the
top ... NEVER coalescing").

mruby grows its VM stack MONOTONICALLY, so every freed stack is smaller than the next request and
a non-coalescing free list matches none of them. Recycling is still worth building for workloads
that free and reallocate similar sizes -- SQLite and the shims do -- but it is not the fix for
this one.

What does bound the churn is the growth policy. mruby's default is LINEAR
(`MRB_STACK_EXTEND_DOUBLING` is the non-default branch, vm.c:165), with the comment that linear
"saves memory on small devices" -- true of the live set, and exactly backwards for an allocator
that cannot reclaim, where the cumulative carve is the sum of every stack ever allocated and
linear growth makes that sum quadratic. It is an upstream option, so turning it on is a build
configuration rather than a patch, but it is NOT the stock configuration and must be reported
wherever a number taken with it appears.

Measured, and it is the whole difference: **carve 4,122,752 linear -> 1,556,800 doubling**, same
workload.

## ROW 10 IS MEASURED: control MISS, revoke-on-free BLOCKED

With doubling making the workload fit, all three arms in one boot,
`musl-capstone/mruby-probe/run-corpus-rows.sh`:

| arm | $arr.size | outcome | verdict |
|---|---|---|---|
| libc allocator | 302 | completed | MISS |
| rof, revocation **OFF** | 302 | completed, carved 1,556,800 | **MISS** |
| rof, revoke-on-free | — | `capability fault: cause = 24` | **BLOCKED** |

**The control is what makes this a measurement.** It is the same build, the same workload and the
same allocator, differing from the revoke arm in one call to `xlang_set_no_revoke()`. It runs all
151 recursion levels and returns, so `mrb_range_new` returned, so the stale write at vm.c:2822
was executed and nothing stopped it. That is the corpus's MISS, and it agrees with what
`xlang/repro/10/run.sh` records for plain RISC-V.

**The fault is at the CVE's own instruction, and it reproduces.** Two independent builds in two
boots both fault at domain vaddr `0xc2284` = `mrb_vm_exec + 0x198e0`, which the assembly puts
immediately at the return from the `mrb_range_new` call in `OP_RANGE_INC` (`.Lpcrel_hi466`, the
call site's own label). The reported pc is accurate to about one instruction: the printf-probe
byte-copy control, where the faulting instruction is known by construction, reports the `ldc` of
the copied pointer with the `lw` that dereferences it one slot later.

**The caveat that must travel with this number.** `MRB_STACK_EXTEND_DOUBLING` is NOT mruby's
default. It is needed because the revoking arena cannot hold the stock configuration's churn, and
that is itself a result about the allocator rather than a detail of the harness: at 4 MiB and
linear growth the interpreter gets 52 of 151 levels and dies with `SystemStackError`. Both arms
carry the same option, so the comparison between them is sound; a claim about "mruby as shipped"
is not.

## SIX ROWS MEASURED, and each faults where the corpus says it should

| row | our fault site | corpus's documented site | control | revoke |
|---|---|---|---|---|
| 4 | `mrb_vm_exec + 0x3d9c` | `mrb_vm_exec` | 302, completed | fault |
| 5 | `hash_new_from_values + 0x13c` | `hash_new_from_values` | 604, completed | fault |
| 8 | `hash_values_at + 0x1c0` | `hash_values_at` | 604, completed | fault |
| 10 | `mrb_vm_exec + 0x198e0` | `mrb_vm_exec`, vm.c:2822 | 302, completed | fault |
| 12 | **`io_get_open_fptr + 0x80`** | `File#initialize_copy` dangling `DATA_PTR` | completed | fault |
| 13 | `hash_slice + 0x180` | `hash_slice` | 1208, completed | fault |

**Row 12 is the first from outside the VM/hash family** and the first needing a GEM: its trigger
opens a real file (`File.new("/dev/null")` through HostCall v0), frees the `mrb_io` behind the
object, and then closes it. Both non-revoking arms complete with a Ruby `closed stream.` -- the
freed struct reads as closed rather than faulting, which is the MISS -- and they report identical
allocation counts (2950 calls, 554,829 bytes), so the two rof arms differ only in the one call to
`xlang_set_no_revoke()`. The revoke arm faults in the function the trigger's own comment names.

**Recorded here because it was mis-stated once:** row 12 was reported as "not a measured row" on
the strength of its libc arm alone, before the revoke arm had reported. The libc arm completing
is half of the result, not the result.

### The older five-row table

| row | CVE / issue | our fault site | corpus's documented site | control | revoke |
|---|---|---|---|---|---|
| 4 | CVE-2022-1071 | `mrb_vm_exec + 0x3d9c`, return from the `const_missing` callback | `mrb_vm_exec` | 302, completed | fault |
| 5 | CVE-2022-1934 | `hash_new_from_values + 0x13c` | `hash_new_from_values` | 604, completed | fault |
| 8 | mruby hash | `hash_values_at + 0x1c0` | `hash_values_at` | 604, completed | fault |
| 10 | CVE-2022-1106 | `mrb_vm_exec + 0x198e0`, return from `mrb_range_new` in `OP_RANGE_INC` | `mrb_vm_exec`, vm.c:2822 | 302, completed | fault |
| 13 | mruby hash | `hash_slice + 0x180` | `hash_slice` | 1208, completed | fault |

**Five independent pinned trees, five different functions, and in every case the one the corpus
names.** That the localisation reproduces the corpus's own crash site five times over is worth
more than any single fault: it is the check that the port is running the defect the row is about
rather than some other defect of ours.

Rows 8 and 13 also carry the allocation readout added the same day, which shows the trigger doing
real work rather than merely not raising: `requested` grows from 376 KB to 1.80 MB (row 8) and to
2.07 MB (row 13) across the trigger, over roughly 700 allocations.

### Three more rows ran, and two of them fault SOMEWHERE ELSE

| row | outcome | where |
|---|---|---|
| 9 | revoke arm faults | `__capstone_cap_copy_fwd + 0x3c0` -- our own memcpy, not `mrb_gc_mark` |
| 15 | revoke arm faults | `envadjust + 0x208` -- mruby's stack-adjust bookkeeping, not `mrb_str_format` |
| 14 | NEITHER rof arm faults | both complete identically, 3227 allocations |

**These are not counted as measured rows.** A fault proves revocation stopped something; it does
not prove it stopped the defect the row is about, and in both cases the site is demonstrably not
the one `target.md` names.

**Row 15's site is a finding in its own right.** `envadjust` runs after every VM-stack
reallocation and tests `oldbase <= st && st < oldbase+size`, where `oldbase` is the block
`mrb_realloc` has just freed -- and freed means REVOKED. `oldbase+size` is a `cincoffset` on a
revoked capability, which the specification makes an Unexpected-operand-type exception. So
**revoke-on-free faults on pointer ARITHMETIC over freed memory, not only on dereferences**. That
is strictly stronger than ASan, which reports only accesses, and it means revocation can preempt
a row's own defect with an earlier, benign use. It is short-circuited behind `e &&
MRB_ENV_STACK_SHARED_P(e)`, which is why it does not fire in every row.

**Row 14 is unresolved, with the trigger PROVEN to arm elsewhere.** The corpus's own x86 ASan
build reproduces it on demand:

```
ERROR: AddressSanitizer: heap-use-after-free ... READ of size 4
    #0 mark_context_stack  src/gc.c:556
    #1 mark_context        src/gc.c:573
    #2 root_scan_phase     src/gc.c:874
```

**Our port configuration does NOT disarm it, and that is now measured rather than assumed.**
The corpus's own x86 ASan build was rebuilt carrying our macros -- `MRB_METHOD_T_STRUCT` and
`POOL_ALIGNMENT=16` on top of its own `MRB_GC_STRESS` -- and it still reports the identical
use-after-free at `mark_context_stack`, gc.c:556. So the four config macros are not the
explanation, which removes the whole "our build is a different program" hypothesis class. The
reference build was restored afterwards and re-checked to still reproduce.

What is left is the ALLOCATOR and the GC's page dynamics: mruby carves objects out of heap PAGES
and only calls `mrb_free` when a page falls entirely empty, so whether the revoking allocator
ever sees a free at all depends on page occupancy, which differs with object layout. That is the
thread to pull, and it is a GC-behaviour question rather than a capability one.

The corpus's `build_config.rb` for this row defines **`MRB_GC_STRESS`**, a full collection on
every allocation, which is what makes it deterministic. That define was missing from our build --
found by reading the row's build_config rather than assuming our four macros were the whole
configuration -- and adding it changes the run (3192 to 3227 allocations, and `gc.o` grows by 296
bytes) but still produces no fault in either rof arm. So the trigger arms on x86 and not here,
and the reason is not yet known.

### The older table, kept because the localisation method is the point

| row | CVE / issue | fault site | corpus's documented site | control | revoke |
|---|---|---|---|---|---|
| 10 | CVE-2022-1106 | `mrb_vm_exec + 0x198e0`, return from `mrb_range_new` in `OP_RANGE_INC` | `mrb_vm_exec`, vm.c:2822 | 302, completed | fault |
| 4 | CVE-2022-1071 | `mrb_vm_exec + 0x3d9c`, return from the `const_missing` callback | `mrb_vm_exec` | 302, completed | fault |
| 5 | CVE-2022-1934 | **`hash_new_from_values + 0x13c`** | `hash_new_from_values` | 604, completed | fault |

Row 5 lands in a different FUNCTION, not merely a different offset in the VM loop, and it is the
function `target.md` names. That the localisation independently reproduces the corpus's own crash
site in three cases is worth more than any single fault.

**Row 5's arms do not all run the trigger the same number of times**, and that is stated rather
than smoothed over: the libc arm reports `$arr.size = 302` and the control 604, so the recursion
fired once in one and twice in the other. The trigger passes three `Bad` objects as hash keys and
the callback fires per key comparison, which depends on hash-bucket collisions, which depend on
object identity and therefore on ADDRESSES -- so a different allocator plausibly produces a
different number of comparisons. That explanation is a hypothesis and has not been verified.
**It does not affect the verdict**, because the load-bearing pair is control against revoke, and
those two share the allocator and therefore the addresses: their arena statistics are identical
to the byte (carved 1,645,152, live 959, peak 968) across two separate boots. The libc arm is a
reference, not the matched control.

Row 4's trigger is
worth reading: `M.const_missing` runs `recurse(150)` and returns 42, so the constant lookup is
what re-enters the VM, and the interpreter then writes the 42 through the stale `regs`. **No Ruby
exception is raised** -- `const_missing` is the handler -- which is why the probe naming
exceptions matters: silence there is the correct outcome for this row and a red flag for row 5.

## Porting a row to another pinned tree is ONE line

`mruby-purecap` differs from its own pinned upstream commit by exactly one line in one file:

```c
#define MRB_STR_EMBED_LEN_BIT 6    /* was 5 */
```

mruby keeps a short string inside the object header with the length in a bitfield;
`RSTRING_EMBED_LEN_MAX` derives from `sizeof(void*)`, so at 16-byte capabilities it is 59 and no
longer fits five bits. `src/string.c` has a static assert for exactly this. **Building against
the ported tree hid the requirement completely** -- row 4 is what found it, by failing with
"pointer size too big for embedded string".

`build-mruby-probe.sh` now SHADOWS the header into the build directory rather than editing the
corpus tree, which must stay byte-identical (same reason `patch-parser.py` writes into the build
directory). The whole `include/mruby` directory is copied, not just `string.h`, because mruby's
headers include each other by bare name and a lone shadowed header finds no siblings. `-D` cannot
do this job: the header defines the macro unconditionally.

Six of the nine pinned trees carry `MRB_STR_EMBED_LEN_BIT 5` and need the shadow; the five older
ones have no such macro. The presym prerequisite is now conditional on the tree having
`include/mruby/presym.h` at all -- mruby gained presym in 3.0 and most pinned trees predate it,
so demanding the generated table rejected buildable trees for lacking something they have no
concept of.