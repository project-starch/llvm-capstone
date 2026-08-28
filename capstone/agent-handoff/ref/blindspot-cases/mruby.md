# mruby: 36 blind-spot cases

The most fruitful source. Criterion and class definitions: [README.md](README.md).

## Why every heap bug here is a candidate by construction

Read at the line in current upstream `src/gc.c`:

```c
/* init_heap_page: the free list is threaded THROUGH the objects */
for (p = page->objects, e = p+MRB_HEAP_PAGE_SIZE; p<e; p++) {
  p->as.free.tt = MRB_TT_FREE;
  p->as.free.next = prev;
  prev = p;
}
page->freelist = prev;

/* incremental_sweep_phase: a swept slot goes straight back to the page */
p->as.free.next = page->freelist;
page->freelist = p;
```

No `malloc` and no `free` happens per object. A dangling reference to a swept RVALUE
therefore points at memory that is tagged, in bounds, and never returned to the
system allocator.

**Consequence for the harness: a sanitizer cannot see it.** ASAN observes only
`malloc`/`free`. The signatures that DO show are: an assertion that a marked object
is `MRB_TT_FREE`; a 4-byte read of `obj->tt`; or **a wrong answer handed back to
Ruby**. Issue 3596 has it under a debugger -- a live `mrb_value` tagged
`MRB_TT_STRING` whose target's header reads `MRB_TT_FREE`.

## The lever that makes class B permanent

`MRB_API int mrb_gc_add_region(mrb_state *mrb, void *start, size_t size);`
(`include/mruby/gc.h:118`, verified) carves a caller-supplied buffer into heap pages.
And `src/gc.c:1508` reads:

```c
if (dead_slot && !page->region) {   /* ... mrb_free(mrb, page) */
```

**Region pages are never returned to the allocator.** With a region-backed heap the
whole engine heap is one capability, no page ever reaches `free()`, and every class
B case below becomes class A.

*Caveat with teeth:* when the region fills, mruby falls back to `malloc` for new
pages. Size it so it cannot, and assert on the page count `mrb_gc_add_region`
returns, or the fallback silently reintroduces malloc'd pages and the measurement
changes underneath.

## Port prerequisites

`xlang/cheri/mruby-port/` already runs purecap mruby under CheriBSD (2026-08-01).
Required everywhere:

| flag | why |
|---|---|
| `-DMRB_NO_BOXING` | `include/mrbconf.h:62-65` defaults to `MRB_WORD_BOXING`, which packs a pointer into an integer word. A static size assertion catches it, which is the good case. |
| `-DMRB_USE_METHOD_T_STRUCT` | otherwise `proc.h` packs a C function pointer as `(uintptr_t)fn << 2 \| flag` and clears the tag |
| `-DPOOL_ALIGNMENT=16` | `src/pool.c` picks 8; the parser's AST cons cells hold capabilities |
| `MRB_STR_EMBED_LEN_BIT` 5 -> 6 | one-line source edit; the embedded-string length field is too narrow for a 16-byte pointer |

For reproduction, steal the configuration from case B8: `MRB_HEAP_PAGE_SIZE=169`
plus `MRB_GC_STRESS`. The small page makes whole-page frees frequent, which is how
latent cases become observable.

---

## Class A: the bad access stays inside a live GC page

Invisible to purecap **and** to revocation, as-is.

| # | issue | fix commit | affected | component | class | script |
|---|---|---|---|---|---|---|
| A1 | 6339 | `0972c8477` (2024-09-09), `0955539cf` | **master only, NO release** | `array.c` `mrb_ary_delete` | UAF -> slot reuse as another type | **CHERI: MISS, all 3 configs** |
| A2 | 5534 | `e323cd0c6ebd` | 3.0.0 | `class.c` `mrb_alias_method` | **type confusion `REnv*` -> `RProc*`** via free-list reuse | no |
| A3 | 3542 | UNKNOWN | ~1.2-1.3 | `gc.c` `mrb_gc_mark` | GC lifetime, `tt == MRB_TT_FREE` | yes |
| A4 | 3550 | **UNKNOWN, `15fba69710c7` REFUTED** | unknown | `gc.c`, Fiber | GC lifetime | yes |
| A5 | 3720 | `b200c7475ae6` | <= 1.3.0 | `gc.c` `mrb_gc_mark` | terminated-fiber stacks | yes |
| A6 | 4000 | `135b4773e3e5` | <= 1.4.1 | `gc.c` | generational GC lifetime | yes |
| A7 | 3699 | `c6736357a720` (UNVERIFIED) | <= 1.3.0 | `gc.c` `mrb_gc_mark` | GC lifetime | yes |
| A8 | 3385 | UNKNOWN | ~1.2 | `gc.c` `mrb_gc_mark` | GC lifetime | yes, 4 lines |
| A9 | 3689 | `c9a4f8a63bef` | <= 1.3.0 | `gc.c` `mrb_write_barrier` | write barrier | yes, 3 lines |
| A10 | 3596 = **CVE-2017-9527** | `5c114c91d4ff` | <= 1.2.0 | `gc.c` `mark_context_stack` | UAF on the RVALUE header | yes |
| A11 | 6316 | doc-only `322642364af2` | 3.3.0+ | `gc.c` `obj_free` twice on one RVALUE | **double free / free-list corruption** | no, C-side `dfree` |
| A12 | 6317 | `3324773f5696` | <= 3.3.0 | `gc.c` `mrb_gc_register` | GC lifetime | no |
| A13 | 4164 | `0925a3281033` | <= 2.0.0 | `hash.c` `sg_shift` | GC lifetime -> freed RVALUE | yes, 3 lines |
| A14 | 2525 | `1114a9042ebc` (UNVERIFIED) | ~1.0 | `gc.c` `atomic_gray_list` | write barrier | partial |
| A15 | 2996 | `9bb552fdb022` (UNVERIFIED) | ~1.2 | `gc.c` `mrb_field_write_barrier` | write-barrier assertion | no |
| A16 | 506 | UNKNOWN | 2012 | `gc.c` `mrb_obj_alloc` | **RVALUE type confusion by construction** | no |
| A17 | 6870 | `036ab85df6f4`, `be36b67a128e` | master 2026 | `mruby-task`, `gc.c` | GC lifetime | no |
| A18 | 6886 | `456a8687af65` | master 2026 | `task.c` `mrb_task_mark_all` | GC lifetime | no |
| A19 | 6872 | UNKNOWN | master 2026 | `gc.c` `mrb_gc_mark` | corrupt RVALUE from the gray stack | no |
| A20 | OSS-Fuzz 56406 | endpoint is a **bisection point, not a fix** | 3.2.0 | `mrb_gc_mark_iv` | UAF read of `obj->tt` | restricted |
| A21 | OSS-Fuzz 56991 | `8d1192f8` | 3.2.0 | `mrb_gc_mark` | UAF read 4 | restricted |
| A22 | OSS-Fuzz 57703 | `b47c8b738ae3` | post-3.2 | `mrb_gc_mark_iv` | UAF read 4 | restricted |
| A23 | OSS-Fuzz 58577 | `b47c8b738ae3` | 3.2.0 | `gc_mark_children` | UAF read 4 | restricted |
| A24 | OSS-Fuzz 59931 | endpoint is a **bisection point, not a fix** | post-3.2 | `obj_free` | UAF read 4 | restricted |
| A25 | OSS-Fuzz 57108 / 57672 / 58723 | `93648fc9` for one of them | 3.2 era | `mrb_str_hash_m` on a swept `RString` | UAF read 4 | restricted |
| A26 | 1301 | UNKNOWN | ~1.0 | `gc.c` | write barrier | no |

## Class B: the whole page was freed to `malloc`

Blind to plain purecap now; **also blind to revocation once the heap is
region-backed**. The ASAN freed-region size is the tell that it was a whole page.

| # | issue | fix commit | component | region | script |
|---|---|---|---|---|---|
| B1 | 3486 = **CVE-2017-9527** | `5c114c91d4ff` | `gc.c` `mark_context_stack` | 49200 | no |
| B2 | 3616 | `51e0e690c270` | `gc.c` `gc_each_objects` | 49200 | yes, 1 line |
| B3 | 3681 | `51e0e690c270` | `gc.c` `gc_each_objects` | 49200 | yes, 1 line |
| B4 | 3674 | `a6a4b76393fa` | `gc.c` `mrb_gc_mark` | 49200 | yes |
| B5 | 4154 | `0fc6b563602d`, `d68da042b366` | `gc.c` `obj_free` | 49200 | yes |
| B6 | 3793 | `15d48efa4bf6`, `c08224983867` | `gc.c` `mrb_gc_mark` | 57392 | yes |
| B7 | 3804 | `3acaa44a70a4` | `gc.c` `mrb_gc_mark` | 57392 | yes |
| B8 | 6326 | `1c5839fb01bc` and two more | `gc.c:782` `obj_free`; `array.c` `sort_cmp` | 8144 | build config |
| B9 | 6662 | `2135088ada98` | `mruby-array-ext` `ary_intersect_p` | 49184 | yes |
| B10 | 3829 | `e4662d77e75d` | `gc.c` `mrb_gc_mark` | 48 | no |

## Class C: rejected, and why that matters

About 45 further mruby bugs were enumerated and **rejected**: they overflow a
standalone `malloc` buffer -- the VM stack `stbase`, `ary->as.heap.ptr`,
`str->as.heap.ptr`, khash tables, irep -- which CHERI catches trivially. Among them
are most of mruby's CVEs (2018-10191, 2018-10199, 2020-6838/6839/6840, 2020-15866,
2021-46020/46023, 2022-0080/0570/0631/1071/1106/1212/1934, 2025-7207, 2025-12875,
2025-13120, 2026-1979, 2020-36401).

**CVE-2022-1071 in particular already has a full case study in-tree** at
`/home/diego/cheribsd-26.07/cases/mruby-regs/`, which correctly calls itself a
window case rather than an invisible one: the VM stack is a plain `realloc`'d array,
so revocation closes it. Do not re-derive it.

## The lead case, MEASURED

**A1, issue 6339.** `mrb_ary_delete` keeps the removed element in a local `ret`
the GC does not know about. The element's `==` runs Ruby, which runs the GC, the
object is swept while `delete` is still running, and its slot comes back off the
page free list as a `String`.

The runnable specimen is `benchmarks/mruby/cases/a1-6339.rb`. Two things about it
were learned by building both sides and running them, and neither was obvious:

**The version range in this table used to say `<= 3.3.0`, and that was wrong in the
most misleading direction.** No release carries the C `mrb_ary_delete` at all --
3.2.0, 3.3.0 and 3.4.0 all still have the Ruby-level `Array#delete`. The bug lives
only in a master window ending at `0972c8477` (2024-09-09). Building the purecap
tree that was already on disk, mruby 3.0.0, and running the script produced a clean
answer that meant nothing: the function under test did not exist there.

**The oracle needs an allocation burst, and without it there is no oracle.** After
`delete` and `GC.start` the freed slot has not been recycled yet, and

| oracle | affected | fixed |
|---|---|---|
| `x.is_a?(C)` | 1 | 1 |
| `x.class == C` | 1 | 1 |
| `x.class.to_s == "C"` | 1 | 1 |
| `x.instance_of?(C)` | 1 | 1 |

every one of them answers 1 on both builds. An earlier version of the specimen
appeared to work only because it evaluated five oracle expressions in a row, and
their own string building allocated enough to recycle the slot between them -- so
the later ones separated and the earlier ones did not. **An oracle that depends on
its own evaluation order is not an oracle.** With `200.times { |k| "filler#{k}" }`
and a second `GC.start` inserted, `instance_of?` and `inspect` separate 10 runs out
of 10:

```
affected (0972c8477^)   2222222222     x.class is String, x.inspect is "1"
fixed    (master)       1111111111     x.class is C
```

`is_a?`, `x.class == C` and `x.class.to_s` still answer 1 on both even with the
burst, and must not be used.

## What the scripts are actually worth, measured

Eleven of the 36 carry a transcribed script. Running them on builds from their own
era against a reference is cheap, and it says the count overstates the yield
badly. **Scripted is not usable.**

| case | on an affected build | verdict |
|---|---|---|
| **A1** | wrong answer, `class=String` | **usable, and MEASURED under CHERI** |
| A4 | `mrb_gc_mark: Assertion (obj)->tt != MRB_TT_FREE` | fires, but its FIX ATTRIBUTION IS WRONG (below) |
| A3, A5, A10, B2 | no observable difference from the reference | no oracle as transcribed |
| A8, A9, B3 | `NoMethodError` on one or both builds | script does not run as transcribed |
| B9 | identical result on both builds | no oracle (below) |

## Why A1 is the only usable one, and it is structural

The rule below said to pick cases whose corrupted slot comes back to Ruby. **A13
satisfies it and still produced nothing** -- `Hash#shift` returns the pair, and at
1, 4, 20 and 64 entries the pair is intact on 2.0.0, 3.0.0 and master alike. So the
rule is necessary and not sufficient, and guessing per case is a poor use of a
build.

Scanning instead is better. The A1 fix is `mrb_gc_arena_save` / `mrb_gc_protect` /
`mrb_gc_arena_restore` around a loop that calls `mrb_equal`, so the hunt is: a C
function that calls the equality family inside a loop over elements and returns
one, without that protection. In current master that is

| file | function | returns |
|---|---|---|
| `src/array.c` | `mrb_ary_index_m`, `mrb_ary_rindex_m` | an index |
| `src/array.c` | `mrb_ary_splat` | a new array |
| `src/hash.c` | `obj_eql`, `mrb_hash_has_value`, `mrb_hash_equal` | a boolean |
| `src/range.c` | `range_eq` | a boolean |
| `mrbgems/mruby-array-ext` | `ary_include` | a boolean |

**Not one of them returns the element.** That is the whole answer to why A1 stands
alone: upstream has protected every site that hands an element back, and what is
left returns integers and booleans, which cannot carry a type confusion into a Ruby
expression. `Array#index` was tested against an array cleared underneath it by a
`==` override, on master and on 2.0.0, and both answer `nil` with the array empty.

The implication for the corpus is not comfortable. A usable class-A specimen needs
a defect that is BOTH invisible to the allocator AND able to hand the recycled slot
back as a value, and mruby's maintainers fix that second half promptly because it
is the half users notice. The blind spot is real; the supply of Ruby-visible
instances of it is small.

## The selection rule these failures teach

A4, B9 and A2 were each pursued to a build and a trigger, and each is a real defect
that produced NOTHING observable from Ruby:

* **A4** marks a freed RVALUE -- `mrb_gc_mark: Assertion (obj)->tt != MRB_TT_FREE`
  fires on a build that carries the assertion. With assertions compiled out the
  marking happens silently and the Proc the script can reach survives intact.
* **B9** iterates a hash set over objects being freed underneath it. The trigger
  fires (5 `eql?` calls, 2780 with hashes forced to collide) and the answer is the
  same on both sides.
* **A2** omits a write barrier when `mrb_alias_method` hangs a fresh `REnv` on an
  already-black `RProc`. Reached via `super`, with `step_ratio` and
  `interval_ratio` driven to 1 to hold the GC in incremental marking across the
  alias, the answer is `:from_base` either way.

**What A1 has and these do not: the corrupted object is RETURNED to Ruby.**
`Array#delete` hands back the very object whose slot was recycled, so a script can
ask it what it is and get a `String`. A4's freed object lives in a fiber's saved
stack, B9's in a temporary set, A2's in a proc's env -- all interpreter-internal,
all invisible to a script no matter how the GC is driven.

So the rule for picking the next case out of the remaining 26, before spending a
build on it: **does the defect put the reused slot somewhere a Ruby expression can
name?** A UAF on a return value, on an array element, on a hash value or on an ivar
qualifies. One on a callinfo, an env, a fiber stack or a temporary set does not,
and no amount of GC coaxing will change that.

Three traps, each caught by a control, each of which would otherwise have produced
a published number:

**The reference build cannot fail the way the old one does.** `mrb_assert` is
`((void)0)` without `MRB_DEBUG`, and the modern builds are compiled without it --
`strings` finds `tt != MRB_TT_FREE` in the 2017 binary and NOT in the modern ones.
So "the reference build printed nothing" says only that it could not print that.
Every era comparison has to be built the same way, which is why A4 was re-run
against `15fba69710c7` rather than against master.

**A4's fix commit in this table is refuted.** `15fba69710c7` is "Revert ae4217e81;
fix #3619" -- issue 3619, not 3550 -- and the assertion still fires on a build AT
that commit, with the assertion verified present in the binary. A4's real fix is
unknown, and so is its affected range.

**B9 has no oracle.** Our 2024-09 build is not even affected: `Array#intersect?`
is still Ruby-level there and the C `ary_set_t` machinery arrives later. Built at
`2135088ada98^` (2026-03-21) instead, the transcribed script answers `false` on
both sides. The trigger does fire -- instrumenting `eql?` counts 5 calls, and 2780
once hashes are forced to collide -- so this is not a case of the condition never
being created. The defect simply does not change the answer, which is the one
thing the criterion says an oracle must do.

**Building anything from 2017 needs a one-line patch first.** Modern Ruby rejects
`FileUtils.mkdir_p path, { :verbose => $verbose }`, a positional hash where
keywords are now required, and mruby's Rakefile fails at load. One substitution in
one file, and it is build-system only, so it cannot affect what is being measured.

Other short ones, verbatim from their issues:

```ruby
# A13, issue 4164
a = {'a' => 'A'}
b = a.shift
printf b[1]

# A9, issue 3689
GC.start
ObjectSpace.each_object{ GC.generational_mode = nil }
a

# A8, issue 3385
a = []
a[0] = a
a.to_s
b

# A6, issue 4000
b=*'$'..'0000'

# A3, issue 3542   (needs MRB_GC_STRESS)
def foo(*)
end
puts foo('a', 'b', 'c', 'd', 'e')

# A10, issue 3596 = CVE-2017-9527   (needs MRB_GC_STRESS)
i = 0
hash = {}
while i < 256
  hash['%d' % i] = i.to_s
  i += 1
end

# A4, issue 3550
f = Fiber.new do
    m = Fiber.current
    Fiber.yield Proc.new {}
end
f = f.resume
GC.start

# A5 (issue 3720) and A7 (issue 3699) have scripts too, but they are long
# fuzzer-derived listings. Copy them from the issue text rather than from here:
# a transcribed script that does not reproduce costs more than a missing one.

# B2, issue 3616
ObjectSpace.each_object { GC.start }

# B3, issue 3681
ObjectSpace.each_object { GC.generational_mode = nil }

# B9, issue 6662
$a = Array.new(50) { Object.new }
$b = Array.new(40) { Object.new }
class << $b.last
  def eql?(o)
    $b.map! { nil }
    GC.start
    super
  end
end
$a.intersect?($b)
```

## What is verified and what is not

Fix commits were re-fetched and their subjects read back. **Not verified:** A3, A8,
A16, A19, A26 have no linked commit at all; A4, A7, A14, A15 have referenced commits
whose subjects do not say "fix N". Three OSV "fixed" fields are **bisection
endpoints, not fixes** and must not be cited as such. Affected-version ranges are
derived from fix-commit dates against release-tag dates, not read off an advisory.
OSS-Fuzz testcases for A20-A25 are view-restricted, so those have no recoverable
PoC; their value is that they name the call site and show the class is live and
recurring in 3.2 and 3.3.

## Next

The cases are collected; nothing has been run on Capstone. The port is the gate, and
`history/28-08-2026_00-30-00_mruby-is-portable-jerryscript-is-not.md` puts it at
eleven census errors, most of them the four flags above.
