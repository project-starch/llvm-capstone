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