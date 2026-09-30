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
| `level0` domain | capability bounds and tags, `free` only marks. The revocation control. |

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
corruption converted into a fault, by bounds and tags alone, with no revocation
in the arm.

The four in the last row are the honest other direction: `606d9a6b2` and
`fb4974528` abort natively and only answer wrongly in the domain, and
`628ccec60` is the `mruby-task` GC-slot row, whose reuse a bounds check has no
event to fire on.

## The revoking arms are blocked, and by what

`sublet` (revoke on free) and `sublet-gc` (every GC object slot issued and
revoked on its own) both build, and **neither can report anything about this
corpus, because both fail their own control.** `probe/depth-probe.sh` brackets
it — one recursion depth per run, the same script in all three arms:

| depth | `level0` | `sublet` | `sublet-gc` |
|---:|---|---|---|
| 20 | ok | ok | ok |
| 40 | ok | **fault** | **fault** |
| 60, 100, 200, 500 | ok | fault | fault |

So the wall is the VM stack's repeated growth under the buddy heap, at somewhere
between 20 and 40 frames of Ruby recursion, and it is independent of this corpus:
`level0` runs every depth. Until it is fixed, no revocation claim can be made
about these 165 — which is the point of running the control first. The GC-slot
corpus records the same wall at `stack_extend_alloc`, so this brackets a fault
that was already known rather than finding a new one.

## Reproducing

    # the window, the cases and the native verdicts
    python3 probe/extract.py                       # needs a full mruby clone; see the script
    python3 probe/run-sweep.py <arm>/bin/mruby verdicts.json

    # the domain arms, one boot each
    bash probe/run-arm.sh level0
    bash probe/depth-probe.sh

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
- The 153 ASan-blind rows are the reason a per-slot mechanism is interesting,
  but this corpus has not shown that any mechanism catches them. It has shown
  that neither ASan nor capability bounds do.
- 24 further defects are live at the pin with a reproducer in their issue rather
  than in a test, so `extract.py` cannot see them; the whole `mruby-task`
  assertion lane (mruby #6862, #6863, #6868, #6870, #6886, #6887) is among them.
  They are not in the 165.
