# Candidates for this corpus, measured at the pin

This corpus was declared empty with the note that the xlang mruby rows would be
re-run inside the complete interpreter. That note is wrong about what is worth
measuring here, and this file is the measurement that says so.

**The xlang rows cannot exercise this corpus's boundary.** All eleven of them
reach a real `free` or `realloc` at the domain allocator -- `xlang/capstone/rows.tsv`
says so itself in its `alloc_route` column (`stack realloc`, `gc sweep -> free`,
`explicit mrb_free`, or `none` for the two spatial rows), and every one was found
*by* ASan, which can only report a release the allocator saw. Revoke-on-free at
`MRBD_HEAP=sublet` answers all of them; the third arm, `sublet-gc`
(`patches/4.0.0-rc2/0008`, every GC object slot issued and revoked on its own),
is not touched by a single one.

What this corpus needs is the class ASan is structurally blind to: memory the
interpreter hands out again without the domain allocator ever seeing a release.
**Nine defects reproduce at the pin, four of them invisible to ASan.**

## How the search was run

The pin is `4.0.0-rc2` = `9d523e2f74f2e63ca02840937523de61398a617d` (2026-03-12).
The port's other pin, `head` = `ad98f216eb472202c8e5deece5ea13655d9f7969`
(2026-09-17), is **2471 commits** ahead; upstream `master` is 2543 ahead.
`build-mruby-domain.sh` already states why that window is the interesting one: rc2
is *"the Sublet evaluation's pin: temporal defects on mruby's own allocators, fixed
later"*.

So `rc2..master` was swept rather than the history before rc2. Three filters found
everything below, in ascending order of yield:

1. the advisories named in the window -- 7 GHSAs and 1 CVE;
2. commit **bodies** naming a sanitizer report, an `MRB_TT_FREE` assertion or a
   fault, which is a far better filter than subjects;
3. the house vocabulary for a lifetime defect: *keep X alive*, *keep X rooted*,
   *off freed memory*, *while X allocates*, *vacated*, *carry X out of*.

**A fix in the window does not by itself mean the defect is live at the pin.** The
defect may have been *introduced* after it. `92b085a1a`
(*keep a returning frame's env rooted while its close allocates*) is the
counterexample and is excluded here: `svar_env_adopt_owner` and the whole
special-variable container machinery it repairs are absent at rc2, having landed
2026-08-30. Every row claimed below was checked to exist at the pin and then
*reproduced* there; none rests on the window alone.

Upstream has **no open issues**, so there is no unfixed-and-therefore-live seam to
mine; everything has to come from the fixed window.

## The builds

Nine of the candidates shipped an upstream test with their fix. Those tests are the
reproduction: extracted from the fix commit's own diff, run against the pin.

| build | flags | what it answers |
|---|---|---|
| `host` | `MRB_DEBUG` | does mruby's own `MRB_TT_FREE` assertion fire, or the answer come back wrong? |
| `stress` | `MRB_DEBUG MRB_GC_STRESS` | does it need a collection at every allocation? |
| `asan` | `-fsanitize=address`, assertions off | **is the defect visible to ASan at all?** |
| `asan-page1` | ASan + `MRB_HEAP_PAGE_SIZE=1` | **how deep is the nesting?** (below) |
| control | pin + the fix hunks that apply | does the fix flip the outcome? |

`asan-page1` is the instrument that makes nesting depth measurable rather than
inferred, and `92b085a1a` is where the idea comes from -- it observes that its own
defect is invisible *"only while its heap page still stands: with
`MRB_HEAP_PAGE_SIZE=1`, or whenever the env was the page's last object, the page is
gone and ASan reports the read"*. One object per GC page turns every slot release
into a page release, so a defect that is ASan-silent at the default 1024 slots and
ASan-visible at one object per page is reusing a **GC slot**; a defect silent at
both is nested *below* the GC, in a block the interpreter never releases at all.

The control carries six fixes (`5e8a65457`, `4663fef45`, `a54353ecf`, `08a0432d1`,
`fb4974528`, `859288c19`), and a second control carries `1c57532b2` alone. Three
could not be applied onto rc2 -- `39aecc143`, `eb7693857` and `606d9a6b2` each
depend on an intermediate commit -- so for those rows the flip is unproven and only
the pin's own failure is reported.

## What reproduces at the pin

| # | upstream fix | advisory | client code | pin | control | asan | asan-page1 |
|---|---|---|---|---|---|---|---|
| 1 | `1c57532b2` | -- | `String#lstrip!`, `#rstrip!`, `#strip!` | wrong answer **+ parent corrupted** | PASS | silent | silent |
| 2 | `a54353ecf` | -- | `Hash#[]`, `#delete`, `#key?` | 9 failures | PASS | silent | silent |
| 3 | `08a0432d1` | -- | `Hash#assoc`, `#rassoc`, `#==`, `#eql?` | 4 failures | PASS | silent | silent |
| 4 | `eb7693857` | -- | `Hash#inspect`, `#__except`, `#rehash` | 4 failures | fix n/a | silent | silent |
| 5 | `4663fef45` | `GHSA-2778-fvwg-5m8w` | hash scans, `Hash#shift` | 4 failures | PASS | buffer-overflow | buffer-overflow |
| 6 | `fb4974528` | -- | `Hash#key`, `#slice`, `#slice!` | **SIGSEGV** | PASS | buffer-overflow | use-after-free |
| 7 | `606d9a6b2` | -- | `Hash#merge`, pattern `**rest` | **SIGABRT** | fix n/a | use-after-free | use-after-free |
| 8 | `39aecc143` | -- | `OP_ENTER` short argument list | **SIGSEGV** | fix n/a | use-after-free | use-after-free |
| 9 | `0cf969a2b` | `GHSA-j6fq-xj4w-877x` | `shaped_iv_foreach` under `#inspect` | test passes, **ASan faults** | -- | use-after-free | use-after-free |

Rows 7 and 8 abort inside mruby's own barriers, at exactly the assertion each fix
commit names:

    606d9a6b2 -> src/gc.c:1411  mrb_field_write_barrier: (value)->tt == MRB_TT_FREE
    39aecc143 -> src/gc.c:846   mrb_gc_mark: Assertion `(obj)->tt != MRB_TT_FREE' failed

`MRB_GC_STRESS` changes two rows and no verdict: it turns row 5 into an abort as
well, and row 8 into a plain failure rather than a fault. Every other row answers
the same with a collection at every allocation as without one, so none of the nine
depends on stress to reproduce -- which is what makes them usable as cases.

### Temporal or spatial: not the same answer for all nine

Every one of the nine has a temporal **cause** -- something changes underneath a
reference the client is still holding. That is what makes them one family. But only
three **manifest** as a use-after-free, and the distinction decides which mechanism
could enforce against each, so it is measured here rather than assumed from the
family name.

| row | what ASan reports at the pin | enforcement handle |
|---|---|---|
| 7 `606d9a6b2`, 8 `39aecc143`, 9 `0cf969a2b` | `heap-use-after-free` | **revoke-on-free** answers these |
| 5 `4663fef45` | `heap-buffer-overflow`, READ 0 bytes after the 80-byte entry array from `ea_resize` | **bounds**, not revocation -- nothing is freed |
| 6 `fb4974528` | overflow 0 bytes past a 30-byte region at 1024 slots per page, use-after-free at one | either, depending on which it reaches first |
| 1 `1c57532b2`, 2 `a54353ecf`, 3 `08a0432d1`, 4 `eb7693857` | nothing, at either page size | **neither** -- see below |

So: three use-after-free, one purely spatial, one that is both depending on the page
size, and four with no allocator event to enforce on at all.

Row 5 is the one to be careful about. `GHSA-2778-fvwg-5m8w` reads like a temporal
defect and is caused by one -- a delete inside an `eql?` callback leaves the scan
walking for more entries than the hash still holds -- but the violation it commits is
reading one element past the end of the entry array. A revoke-on-free arm cannot see
it, because the array was never freed; bounds can. Filing it as temporal would
predict the wrong arm.

Rows 1-4 are the corpus's reason to exist precisely because they fall in neither
column: the reference outlives what it named, so the shape is temporal, and yet
nothing is released and nothing goes out of range, so neither a bounds check nor a
free-triggered revocation has an event to fire on. Rows 2-4 need per-entry identity
inside the hash's array; row 1 needs revocation at an ownership transfer.

Row 9 is the one whose oracle is not its own test: the extracted test **passes** in
the `host` and `stress` builds because the stale read does not change the answer,
while ASan faults at the pin on exactly the path the fix describes --
`shaped_iv_foreach` (`src/variable.c:464`) reading a block that `shaped_iv_set`
(`:416`) freed to allocate a wider one, reached through `inspect_i`
(`src/kernel.c:83`). A case built from it has to name ASan or a capability fault as
its oracle, not an assertion on the answer.

### Rows 1-4 are this corpus's material

They are the only four in the set that **ASan cannot see at either page size**. That
is the measurement, not an inference: their reuse never reaches the GC slot layer,
so shrinking a GC page to one object does not expose them either. They are nested
*below* the collector.

**Row 1 is the clearest case found, and the cheapest to arm.** Four lines, no GC
stress, no callback:

    base = ".        abcdefghijklmnopqrstuvwxyz0123456789"
    view = base[1..-1]
    p view.lstrip!
    p base

    pin:     "        abcdefghijklmnopqrstuvwxyz01"
             ".abcdefghijklmnopqrstuvwxyz0123456789\x003456789"
    control: "abcdefghijklmnopqrstuvwxyz0123456789"
             ".        abcdefghijklmnopqrstuvwxyz0123456789"

`base[1..-1]` is a shared-buffer view: one `mrb_malloc`'d buffer with an
`mrb_shared_string` refcount and two `RString` heads into it. `str_lstrip_bang`
cached `RSTR_PTR(s)` *before* `mrb_str_modify()`, and `mrb_str_modify()` is
precisely what un-shares -- it allocates the view its own buffer and copies. Every
write after it goes through the stale pointer into the **parent's** buffer, which is
why `base` comes back with a NUL in the middle of it. The whole fix is moving the
`char *ptr = RSTR_PTR(s);` line to after the modify, at three sites.

ASan is silent because no allocator event happens at all. `str_decref`
(`src/string.c:264`) frees the shared buffer only when the last reference goes; here
the parent still holds it, so the refcount drops 2 to 1 and nothing is released.

**And the write is in bounds at every granularity**, which is worth stating exactly
because it decides which mechanism could catch it. Measured on the reproducer above:
the parent is 45 bytes, the view occupies parent offsets 1 to 44, and the corruption
spans offsets **1 to 37** -- inside the parent's allocation, and inside the view's own
extent as well. So neither malloc-granularity bounds nor per-view bounds see it. The
pointer is not out of range; it names the wrong buffer. The view was handed a new
buffer by `mrb_str_modify()` and the write went to the old one.

The only handle is **revoking the view's alias into the shared buffer at the
un-share**, which is revocation on an ownership transfer rather than on a free. That
is a third enforcement point, distinct from both arms the port has, and this row is
the one that argues for it.

**Rows 2-4** share one shape at three sets of sites. The nested allocator is the
hash's own entry array: `ar_delete` and `ht_delete` (`src/hash.c:586`, `:951` at the
pin) vacate a slot by marking its key undef and decrementing the count,
`ea_compress` and `ea_resize` (`:454`, `:448`) move entries inside the block, and a
later store reuses a vacated slot. All of it happens inside one `mrb_malloc`'d
block, so no release reaches the domain allocator -- the same shape as the GC's own
`page->freelist` recycling at `src/gc.c:1188-1191`, one level further down.

The client is hash.c's own lookups and scans. Every one can re-enter Ruby through a
key's `eql?` or `hash`, and `H_CHECK_MODIFIED` watches the capacity, the flag bits
and the two pointers. A delete followed by an insert that restores the size moves
none of those, so the guard does not fire and the scan answers from an entry that
has been vacated and refilled. The oracle is a wrong answer where the fixed
interpreter raises `hash modified`, so a case can report beside itself without a
fault, unlike the corpora whose cases end the domain.

### Rows 5-9 belong with the xlang material

ASan sees all five, so they bottom out in a real release and `sublet` answers them
at the malloc layer. Row 6 is the one that moves under the page knob -- a
buffer-overflow at 1024 slots per page, a use-after-free at one -- which places it
at the GC slot layer with a page release underneath. The rest are worth having as
the ladder's lower rungs but do not exercise `sublet-gc` any more than the existing
xlang rows do.

## What did not reproduce, and what is untested

| candidate | status at the pin |
|---|---|
| `5e8a65457` / `GHSA-jfmr-44fc-gfhg` | its own upstream test **passes**, in all four builds |
| `17d124b00` (`mrb_ary_splice` self-aset) | its own upstream test **passes** |
| `eb701fa5b` (incremental GC driven from a realloc) | **not measurable**: the test needs `GC.malloc_threshold`, which the pin does not have |
| `7d626242a` (constant cache on a freed irep's address) | wrong answer one iteration earlier than the commit describes; not counted |
| `92b085a1a` | **excluded**: the machinery it repairs postdates the pin |
| `0bfdb9c18` (UAF when GC runs during a stack realloc) | **untested**: shipped no test; `src/vm.c` only |
| `456a8687a` (re-mark task stacks in `final_marking_phase`) | **untested**: shipped no test, and names the `MRB_TT_FREE` assertion -- but `mruby-task` is in none of the port's gemboxes |

`GHSA-jfmr-44fc-gfhg` is an out-of-bounds *read* whose test asserts an exception
rather than a fault, so a plain build has nothing to report. It is not established
that the defect is absent at the pin -- only that this test does not show it.

`7d626242a` is the most interesting unsettled candidate: the cache behind
`OP_GETCONST` is keyed by the **irep's address**, and nothing dropped an irep's
entries when it was freed, so a later irep landing at the same address is answered
from the stale entry -- address reuse with no dereference of freed memory at all,
which is the purest form of what this corpus is about. But the commit's own
reproducer gives `[:top, :top]` from the **first** iteration at the pin where the
commit describes `[:foo, :top]` first. Settling it needs the C-side helper the fix
added to `mrbgems/mruby-eval/test/eval.c`, which needs a build with tests enabled;
the fix does not apply onto rc2 unaided.

The two untested leads are the remaining work with the highest expected yield:
both name this corpus's mechanism and neither can be dismissed without writing a
reproducer, since neither shipped one.

## How much of this depends on the pin

The question "would another version give us these?" is not an estimate -- the cases
run, so it is a measurement. Four versions were built and the nine cases run against
each, `host` plus `asan`:

| version | date | of the nine, live | what changes |
|---|---|---:|---|
| `3.4.0` | 2025-04-20 | 6, one of them ambiguous | rows 1 and 9 **do not exist yet**; row 8 crashes but ASan is silent there, so it is not claimed |
| `4.0.0-rc2` | 2026-03-12 | **9** | the current pin |
| `4.0.0` | 2026-04-20 | **9** | identical to the pin, case for case |
| `4.1.0-rc2` | 2026-09-11 | 6 | rows 1, 5 and 8 fixed; rows 2, 3, 4, 6, 7, 9 still live |
| `head` (`ad98f216e`) | 2026-09-17 | 1 | only row 9 survives |

`probe/versions.sh` prints that matrix. Row 6 is worth one note: at `3.4.0` it reports
a use-after-free where the 4.0 line reports a buffer-overflow, so a row's *class* can
move between versions even when it reproduces in both -- another reason the table is
run rather than derived.

**The pin is already the best available choice, and `4.0.0` is the same choice with a
release number.** Every one of the nine reproduces identically at `4.0.0-rc2` and at
`4.0.0` -- same verdicts, same ASan reports -- so moving the pin from the release
candidate to the release costs nothing and stops the corpus resting on a
pre-release. That is worth doing for its own sake; `port.json` and
`build-mruby-domain.sh` would both follow.

Going **older** loses the two best-understood rows outright rather than merely
fixing them: `str_lstrip_bang` and `shaped_iv_foreach` are absent before the 4.0
line, so rows 1 and 9 have no code to be live in at `3.4.0`. Row 1 is the cheapest
case in the set, so an older pin is strictly worse. The hash machinery
(`H_CHECK_MODIFIED`, `ea_resize`, `mrb_hash_merge`) does reach back to `3.2.0`
(2023), so rows 2-7 are roughly three years old and are not an artefact of a recent
rewrite.

Going **newer** is where the cliff is, and it is a cliff rather than a slope: five
of the nine were fixed in a single week, 2026-09-08 to 09-12. `4.1.0-rc2` was tagged
2026-09-11, one day before the hash cluster landed, which is why six of the nine are
still live in it and only one is at `head` six days later. The port's `head` pin is
therefore the wrong one for this corpus, which `build-mruby-domain.sh` already says
in its own words.

**Ancestry of the named fix commit is only a proxy, and it was wrong once here.**
`merge-base --is-ancestor` puts row 5's fix (`4663fef45`) outside `4.1.0-rc2`, so the
defect should still be live there; the case **passes** at `4.1.0-rc2` when run. An
equivalent change reached that tag by another route. So this table is built from runs,
not from the ancestry matrix, and the matrix is kept only as the thing that pointed at
which versions were worth building.

## Gems, and what is out of reach

The port's gemboxes carry `mruby-string-ext`, `mruby-hash-ext`, `mruby-array-ext`,
`mruby-method` and `mruby-set`, so rows 1-9 are all reachable in the domain build.

`GHSA-f3mm-x76x-jmcv` (`859288c19`) is the same shape at three sites -- a value
taken out of the structure that held it and published on the arena afterwards,
where `mrb_gc_protect()` itself can allocate and collect it. Two of the three,
`Hash#shift` and `mruby-method`'s argument shift, are in the port. The third,
`Task::Queue#__pop_try`, is in `mruby-task`, which exists at the pin but is in none
of the port's gemboxes -- as is `456a8687a`'s site.

## Measured in a domain: the control arm, and what blocks the other two

A clang was built from `7d01722aab88` ("SROA: do not split an alloca so that a
capability is cut or misaligned") -- `LLVM_TARGETS_TO_BUILD=Capstone`, Release with
assertions, through `capstone/tests/build-toolchain.sh` so it took the machine-wide
memory lock; `ninja rc=0`, scope peak 11 GiB, no OOM events. With it the port's own
`scripts/smoke.rb` reaches `SMOKE_DONE` and `LT-RESULT mruby.dom status=0 rounds=158
PASS` in the `level0` arm. The cause-24 fault in `mrb_packed_int_decode` reported above
is gone, so that fix was indeed the blocker.

**`level0` is the matched pair's control arm** -- free only marks, so nothing is
revoked -- and `HOW-TO-RUN-ON-QEMU.md` is explicit that this is what a "caught" claim
needs: a cause-24 fault *"looks identical to a caught use-after-free until the control
shows the same program completing when the revoke is removed"*. So nothing below is a
catch, by construction; this is the row that has to MISS before a fault in `sublet`
means anything.

| case | `level0` domain | reading |
|---|---|---|
| 1 `1c57532b2`, 117-byte variant | completes, `status=0`, parent corrupted | **reproduces, uncaught** |
| 2 `a54353ecf` | completes, 9 wrong answers | **reproduces, uncaught** |
| 3 `08a0432d1` | completes, 4 wrong answers | **reproduces, uncaught** |
| 5 `4663fef45` | completes, 4 wrong answers | **reproduces, uncaught** |
| 6 `fb4974528` | completes, 2 wrong answers | reproduces as a wrong answer, not the native SIGSEGV |
| 7 `606d9a6b2` | completes, 2 wrong answers | reproduces as a wrong answer, not the native SIGABRT |
| 9 `0cf969a2b` | completes, PASS | expected: its oracle is ASan, and this arm revokes nothing |
| 4 `eb7693857` | cause-24 fault, `mrb_vformat +0x7c0` | **not a catch** -- no revocation in this arm |
| 8 `39aecc143` | cause-24 fault, `mrb_vm_exec +0x488` | **not a catch**, same reason |

Rows 4 and 8 are the trap the documentation names, met in practice: a cause-24 fault in
an arm that cannot revoke anything. Read without the control they would have been two
"caught" results.

### Row 1 needs a longer string in a domain than natively

The case as extracted **passes** in the domain, and not because anything caught it.
`patches/4.0.0-rc2/0001` states the reason itself: `RSTRING_EMBED_LEN_MAX` is
*"27 at 8-byte pointers, 59 at 16-byte capabilities"*. The upstream test's 45-byte
subject is a heap string natively, so `base[1..-1]` becomes a shared view and the
defect fires; in a domain 45 is under the embed limit, the subject is embedded in its
`RString`, and there is no shared buffer for a stale pointer to point into.

A 117-byte subject reproduces in both: natively the parent comes back corrupted and the
fix turns it green, and in the `level0` domain the parent comes back corrupted with
`status=0`. That is a fidelity condition a case must carry, and it generalises -- any
case in this corpus whose trigger depends on a size class has to be re-derived for
16-byte pointers rather than copied from upstream.

### What blocks `sublet` and `sublet-gc`

Both arms build, and both fail their **own** control: `scripts/smoke.rb` halts at
`cause = 24` in `stack_extend_alloc +0x194`, the VM stack's `mrb_realloc`. So the cases
were not run in either arm -- an arm whose control fails cannot report anything about a
defect.

It is not the runtime or the grants: `RUNTIME_REPO` was `b338c156c8d1`, which sits on
`9704639b4a3a`, exactly the pair `results/2026-09-26` records, and the grants were the
documented 134217728 and 67108864. The port's own recorded `sublet` and `sublet-gc`
mrbtest runs used compiler `7d01722aab88`, the same commit built here, so the
difference is narrower than the toolchain: candidates are the QEMU
(`movc-merge d621df553f` was used, as recorded), `CAPSTONE_REV_NODES`, or the buddy
heap's behaviour on the VM stack's repeated realloc at this pool size. That is the next
thing to chase, and it is one fault at one site rather than a category.

One infra note, because it cost a wrong reading: the first `level0` control with the new
toolchain stalled after `stty columns 29999` with no fault and no result. Re-run
unchanged, it passed. A run that neither finishes nor faults is an infra flake, not a
result, and `measure2.sh` now retries such a run up to three times instead of recording
it.

## Earlier attempt: the apparatus works, the compiler was the blocker

This was attempted, not just reasoned about. What works:

* All three arm images build from this branch at the pin, in about 45 seconds each --
  `MRBD_HEAP=level0`, `sublet` and `sublet-gc`, the last carrying patch 0008 (12
  `sublet` symbols in the image, none in `level0`).
* The `sublet-gc` arm needs the host's **second grant**, `LT_GC_REGION_BYTES`, which
  is **not on dev**: `run-mruby-domain.sh` passes the define but no source on this
  branch consumes it. It is on the unmerged `runtime/libc-test-second-grant`
  (`b338c156c8d1`), which works as `RUNTIME_REPO` once its `caplifive-buildroot`
  submodule is initialised.
* QEMU runs the domain and reports capability faults with a cause, a pc and the
  register file, and the pc symbolizes against the unstripped image by adding the
  image's `0x10000` base to the offset from the `pc_cap` region base.

What blocks it: **no toolchain on this host can run the port's own smoke script.**
With the newest available clang, `scripts/smoke.rb` halts at
`cause = 24, pc = 0xd8242b2c`, which resolves to `mrb_packed_int_decode` +0x28 --
mruby's packed-integer bytecode reader. The port's recorded runs
(`results/2026-09-26`) name compiler `7d01722aab88`, which is

    SROA: do not split an alloca so that a capability is cut or misaligned (Capstone)

on the unmerged branch `compiler/sroa-keep-capability-whole`. It is not on dev, and
**none of the ten prebuilt toolchains under `/tmp/capstone` contains it** -- checked
with `merge-base --is-ancestor` against each one's embedded source commit. A function
that decodes an integer out of a byte stream through a local is exactly what that fix
is about, so the fault and the missing fix agree.

So the measurement needs a clang built from that commit, after which the runs are
cheap: an arm image is 45 seconds and a case is one bounded QEMU boot.

**The control is what makes this an honest negative.** The first case run produced a
cause-24 fault, which read like a result -- the mechanism catching the defect. Running
the port's own `smoke.rb` in the same arm faulted at the *identical* pc, which says
the instrument is broken rather than the defect caught. A fault is no more
self-evidently a finding than a clean run is.

### The recipe, so it is not rediscovered

    PATH=$HOME/.venvs/capstone/bin:$PATH             # run-domain-smoke.py needs pexpect
    CAPSTONE_LLVM_BUILD_DIR=<a clang with 7d01722aab88>
    RUNTIME_REPO=<a worktree of runtime/libc-test-second-grant, submodule initialised>
    CAPSTONE_QEMU_BINARY=<capstone-qemu movc-merge d621df553f>
    CAPSTONE_GP_NONLIN=1                             # else gp is refabricated LINEAR at every cjalr
    CAPSTONE_REV_NODES=16777216                      # as the port's recorded runs used
    MRBD_HEAP_REGION_BYTES=134217728                 # sublet and sublet-gc
    MRBD_GC_REGION_BYTES=67108864                    # sublet-gc only

    bash run-mruby-domain.sh <image> <work> 180 <case>.rb -- /mnt/host/files/<case>.rb

A worktree created off this branch inherits a sparse checkout of `capstone` and
`xlang` only, and `capstone-test-env.sh` refuses a root without `llvm/`, so
`git sparse-checkout disable` comes first. Redirect the run script's output to a file
rather than piping it: a pipe replaces its exit status, and this cost one wrong "rc=0"
reading here before it was caught.

## Limits of this measurement

Everything above is a **native x86-64** measurement of vanilla rc2. It establishes
that the defects are live at the pin, which instrument sees each one, and at what
depth each is nested. It does not say what the Capstone domain does with them: no
clang, no QEMU and no board were used, the port's patches 0001-0003 and 0008 were
not applied, and no arm of `MRBD_HEAP` was run. Turning these into cases means
writing each as the corpus schema wants, then arming `spatial`, `sublet` and
`sublet-gc` and measuring.

The prediction worth committing before that runs, in the spirit of
`xlang/capstone/rows.tsv`:

* **rows 2-4** MISS under `level0` and under `sublet`, and FAULT only under
  `sublet-gc` *if* the arm is extended to sub-let hash entry slots. As
  `patches/4.0.0-rc2/0008` stands it sub-lets GC object slots only, so these three
  are predicted to MISS in all three arms -- they are the rows that say what the
  port does not yet cover.
* **row 1** MISSes in all three arms, and a fourth arm that merely bounds each
  `RString` view would not change that: the write is in bounds for the view as well,
  as measured above. The arm it needs issues each view its own alias into the shared
  buffer and **revokes that alias in `mrb_str_modify()`**, at the un-share -- an
  ownership transfer, not a free. Cheap to site, because `str_unshare_buffer`
  (`src/string.c:274`) is the one place it happens.
* **row 5** MISSes under `level0`, whose pointers carry the whole arena's bounds, and
  FAULTs under both sublet arms -- but by **bounds**, since its read runs past an
  entry array that was never freed. If it faults under an arm with revocation and no
  per-block bounds, that reading is wrong.
* **rows 6-9** MISS under `level0` and FAULT under both `sublet` and `sublet-gc`.

If rows 2-4 fault under `sublet` as it stands, the entry array is being reallocated
where this reading says it is reused, and the reading is wrong.
