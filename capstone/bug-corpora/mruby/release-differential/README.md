# Defects live in the ported mruby release, and what each arm sees

The port pins mruby at `4.0.0-rc2` (`9d523e2f74f2`, 2026-03-12). This corpus is
the answer to a plain question — *which memory-safety and correctness defects are
live in that build, and which of them does a Capstone domain catch?* — asked over
the whole release rather than over a hand-picked list.

**165 defects are live at the pin.** Each is live because its own upstream
regression test fails at the pin and passes at master; none of the 165 rests on a
commit date or on a fix's wording. [ledger.json](ledger.json) is the result, one
row per defect, with what four native arms and one domain arm see.

## Why not the advisories

The obvious route yields almost nothing here. Of the **44** CVE/GHSA entries
published for mruby, **43 are already fixed in 4.0.0-rc2**; the one that is not
(`CVE-2026-79590`, a NULL passed to `memcpy` in Prism) is not a reuse defect. Of
the **78** OSS-Fuzz OSV advisories, **73** are fixed at the pin. Zero published
use-after-free CVEs survive in this release.

That is not a gap in the search, it is a property of the pin: `4.0.0-rc2` is
recent, so the defects still in it are the ones found *after* it. The material
therefore has to come from the window `rc2..master` — 2571 commits — and
upstream's own regression tests are the instrument that reaches it.

## How the 165 were found

`probe/extract.py` walks the window and keeps every commit that touches **both** a
source file and a Ruby test file — the shape of "a fix and the test that proves
it", 674 of them. For each, the lines the commit *added* to the test become a
standalone case: `probe/harness.rb` (which reimplements mrbtest's assertions so a
case runs under a plain `mruby`), the added lines, and `probe/footer.rb`.

`probe/run-sweep.py` then runs all 674 against the pin and against master:

| | at the pin | at master |
|---|---:|---:|
| PASS | 80 | 394 |
| FAIL (wrong answer) | 189 | 70 |
| CRASH | 17 | 0 |
| TIMEOUT | 3 | 0 |
| FEATURE_GAP (the test needs an API the pin lacks) | 211 | 83 |
| NORUN (the case does not parse; new syntax, or a partial extraction) | 174 | 127 |

**A row is a defect only if it fails at the pin and passes at master**, which is
the 165. The other two outcomes are separated automatically: a test that raises
`NoMethodError` or `NameError` is asking for a feature the pin does not have, not
reporting a defect, and a case that does not parse is a bad extraction. Both are
counted above rather than quietly dropped.

The discovery step is checked against known ground truth: the twelve rows the
[GC-slot corpus](../gc-slot-repros) found by hand are all 12 rediscovered by this
sweep. One of them, `0cf969a2b`, is only found once the window runs to master
rather than to the port's `head` pin — its fix landed the day after that pin was
taken, so it is live in *both* of the port's pins.

## What each arm sees

Four native builds of the pin (`probe/corpus_config.rb`) and one Capstone domain.
The gem set is the port's own, so `mruby-io`, `mruby-pack` and `mruby-task`
defects are in range.

| arm | what it answers |
|---|---|
| `host` (`MRB_DEBUG`) | does the answer come back wrong, or an assertion fire? |
| `stress` (`+ MRB_GC_STRESS`) | does it need a collection at every allocation? |
| `asan` | **is the defect visible to a malloc-layer sanitizer at all?** |
| `asan-page1` (`+ MRB_HEAP_PAGE_SIZE=1`) | one object per GC page, so a slot release becomes a page release: **is the reuse inside a GC page?** |
| `level0` domain | the port's default heap: **the arena's** bounds, tag integrity, and `free` only marks. The revocation control. |

Over the 165:

- **ASan sees 10.** Six use-after-free, four heap-buffer-overflow.
- **Two more appear only at one object per GC page** — `39aecc143`
  (`OP_ENTER` and a short argument list) and `628ccec60` (`mruby-task` not
  marking the main task's values). Those two reuse a GC object slot, which is
  the class a malloc-layer tool is structurally blind to.
- **153 are invisible to ASan at either page size.** Their reuse never reaches
  an allocator event: the reference outlives what it named, and nothing is
  freed and nothing goes out of range.
- `stress` turns 17 rows into crashes and changes no verdict, so no row here
  depends on a collection at every allocation to reproduce.

## In a Capstone domain

The domain arm is `level0`: every pointer carries its bounds and its tag, and
`free` only marks. It is the **control** for the revoking arms, so a fault here
is not a catch by revocation — it is the defect itself reaching a capability
check. Its own control is the port's `scripts/smoke.rb`, which completes
(`SMOKE_DONE`).

**17 of the 165 fault in the domain**: 13 with SIGSEGV and 4 killed by the
per-case watchdog (three of those loop forever at the pin natively too).
148 complete.

The interesting part is the overlap with ASan:

| | count | rows |
|---|---:|---|
| domain faults, ASan sees it too | 7 | `13e017c2f` `1737589f0` `4663fef45` `84cc5aa60` `93eb74a59` `af6f23ddb` `bef45e223` |
| domain faults, ASan silent at the default page size | 10 | `03f242d09` `0654bd470` `39aecc143` `727ae28ad` `77d1d928f` `94993eede` `a5f6411b1` `d8911416c` `eb7693857` `ec89364c4` |
| ASan sees it, the domain does not fault | 4 | `59552ecb8` `606d9a6b2` `628ccec60` `fb4974528` |

Six rows are worth naming separately, because for them the native build returns a
**wrong answer and no diagnostic** while the domain **traps**: `4663fef45`,
`84cc5aa60`, `93eb74a59`, `bef45e223`, `eb7693857`, `ec89364c4`. That is silent
corruption converted into a fault with no revocation in the arm. Note what does
the work: `CAPSTONE_LEVEL0_SHRINK` is opt-in and **off** in these builds, so
level0 hands out the whole arena's bounds, not each object's. These six do not
fault because an object's bounds were exceeded — they fault because a
capability that was rebuilt from an integer or derived from a dead one carries
no tag, or because the access left the arena altogether.

The four in the last row are the honest other direction: `606d9a6b2` and
`fb4974528` abort natively and only answer wrongly in the domain, and
`628ccec60` is the `mruby-task` GC-slot row, whose reuse a bounds check has no
event to fire on.

## The revoking arms: what revocation catches

`sublet` (revoke on free) and `sublet-gc` (every GC object slot issued and
revoked on its own) were both **blocked** when this corpus was first measured:
each failed its own control, faulting at ~40 frames of Ruby recursion — shallow
enough that the port's own `scripts/smoke.rb` stopped at M8. `probe/depth-probe.sh`
bracketed it, one depth per run in all three arms, and `level0` completed at every
depth.

That was mruby's own defect, not the heap's. `stack_extend_alloc()` hands the old
VM stack to `mrb_realloc()` and then calls `envadjust()`, which moved every
frame's pointer with `ci->stack += delta` — pointer arithmetic on a pointer
`realloc` has already freed. On an ordinary allocator it computes the right
address; where `free` revokes, the result is derived from a revoked capability,
carries no tag, and the next write through it faults. Upstream fixed exactly this
in `e5c82761f` (2026-07-24) after another memory-safe C implementation trapped on
the same write, and the port backports it as
[patch 0010](../../../ports/mruby/app/patches/4.0.0-rc2/0010-envadjust-rebases-off-the-new-stack.patch).
Only the `4.0.0-rc2` pin needs it; the `head` pin already carries the fix.

With that patch **all three arms complete their control**, and the 165 run in each:

| | completes | faults | watchdog |
|---|---:|---:|---:|
| `level0` (control: arena bounds, tags, no revocation) | 148 | 13 | 4 |
| `sublet` (revoke on free) | 147 | 15 | 3 |
| `sublet-gc` (per GC object slot) | 144 | 18 | 3 |

A **catch** is a case the control runs to completion and a revoking arm faults
on. `level0` revokes nothing, so its own 13 faults are the defect reaching a
bounds or tag check, and are not catches. Four cases are caught:

| case | level0 | sublet | sublet-gc | ASan | what it is |
|---|---|---|---|---|---|
| `59552ecb8` | **completes, silently** | **fault** | **fault** | use-after-free | `mruby-sprintf`, format string mutated during the call |
| `606d9a6b2` | wrong answer | wrong answer | **fault** | use-after-free | `hash.c`, the pair a scan carries into a set |
| `628ccec60` | wrong answer | wrong answer | **fault** | **silent** | `mruby-task` does not mark the main task's values |
| `fb4974528` | wrong answer | wrong answer | **fault** | heap-buffer-overflow | `mruby-hash-ext`, what a walk carries out of a hash |

Three of the four are caught **only** by `sublet-gc`: revoke-on-free never sees
them, because the reuse happens inside a GC page the allocator still owns. That
is the case for per-slot revocation, measured on real defects rather than argued.

`628ccec60` is the one to look at hardest. ASan is blind to it at the default
page size, revoke-on-free misses it, and per-GC-slot revocation faults on it —
and it is reachable at all only because the port now builds `mruby-task`.

## Reproducing

    # the window, the cases and the native verdicts
    python3 probe/extract.py                       # needs a full mruby clone; see the script
    python3 probe/run-sweep.py <arm>/bin/mruby verdicts.json

    # the domain arms, one boot each; the revoking arms need the node pool
    bash probe/run-arm.sh level0
    CAPSTONE_REV_NODES=16777216 bash probe/run-arm.sh sublet
    bash probe/depth-probe.sh

The node figure is this kit's. `sublet_heap.c` records that capstone-qemu
recycles retired identities since `22aec7ee0f`, which makes the 65,536-identity
pool bound live identities rather than allocations per boot; that commit is not
in the emulator this kit was built from, and on it the 65,536 default is not
enough for these runs.

The harness (`probe/harness.rb`, `probe/footer.rb`) and the four-arm build config
are taken from the GC-slot corpus's own probe on `corpus/mruby-gc-slot-reuse`,
where they were written; the extraction, the sweep and the domain runner are new
here. What that corpus does for twelve hand-picked rows, `extract.py` does for
every commit in the window.

## What this does not say

- **No case is cut to the [schema](../../SCHEMA.md) yet.** `corpus.json` says
  `triaged` for that reason, and `cases` is 0. The ledger is the measurement;
  promoting rows to `case.c` + `case.json` + `PROVENANCE.md` is the next step,
  and the rows in the first two table columns above are the ones worth promoting.
- A domain fault is **not** attributed to a specific defect until its matched
  pair runs: the pin plus that one fix, which must stop faulting. That is done
  for none of the 17 yet.
- Of the 153 ASan-blind rows, exactly **one** is caught by a mechanism here
  (`628ccec60`, by `sublet-gc`). The other 152 are still caught by nothing in
  this set: not ASan, not level0's tags and arena bounds, not revocation at
  either layer.
  That is the honest size of the remaining gap.
- 24 further defects are live at the pin with a reproducer in their issue rather
  than in a test, so `extract.py` cannot see them; the whole `mruby-task`
  assertion lane (mruby #6862, #6863, #6868, #6870, #6886, #6887) is among them.
  They are not in the 165.
