# C-32 after design A: a local reproducer of the live silicon site

`python3 capstone/tests/c32-sinkfold-repro/check.py` — run it from the repo root.
Sub-second. Exit 0 as recorded, 1 if a shape's codegen changed, 2 if it could not
check (no llc, a shape missing, or the positive control not firing).

## Why this exists

Until now the only test of a C-32 fix was a ~90-minute SQLite domain image build,
on a host whose glibc can compile the domain TU — which the apollo host cannot
(`docs/design/hosted-libc-os-analysis.md:24-34`). Every design-A validation that
was done on the wrong image came from that cost. These five shapes reproduce the
live site from the Sublet cell ⑥ `-O2` image (sha256 `113221f93b0ac994…`),
function `setupLookaside`, site `0x267a0`. The same sequence — not byte-identical: the image
interleaves two `ldc` between the `beqz` and the `mv`, and lays the arms out the other way round:

    image     26790 beqz a0 · 2679c mv s3,a0 · 267a0 movc a0,s3 · 267a4 jalr a1
              267c4 movc s3,zero · 267f4 mv a0,s3
    shape 4   beqz a0 · mv s0,a0 · movc a0,s0 · cjalr ra,0(a1)
              movc s0,zero · mv a0,s0

## What it shows

Design A is **not** rematerialisation in effect. RA-side remat never runs for
`PseudoBRIDGE_CAP`: `TargetInstrInfo::isReallyTriviallyReMaterializable`
(`llvm/lib/CodeGen/TargetInstrInfo.cpp`, "Don't allow any virtual-register
uses") refuses any instruction with a vreg use and this pseudo has one, and
Capstone's override (`CapstoneInstrInfo.cpp:244`) is the inherited RVV one, which
falls through to it for this opcode.

What removes the `movc` is `MachineSinking::PerformSinkAndFold`
(`llvm/lib/CodeGen/MachineSink.cpp:405`), pre-RA: it rewrites ISel's
`$c10 = COPY %bridged` into `$c10 = PseudoBRIDGE_CAP %int`, so the bridge is
duplicated into each use and no GPCR vreg survives for RA to copy.

**That fold is all-or-nothing per def.** It walks every use of the def, and the
first one that is not a copy chaining to a physreg of the *same* register class
(GPCR) and not a foldable load/store address makes it decline for the whole def.
So a single non-conforming use — a PHI at a join, or reading the address half
back as an integer — leaves the *other* uses' copies as `movc` too. That is the
whole difference between the lit test's shape and `setupLookaside`.

In `setupLookaside` the declining use is the **PHI**. The integer read-backs there
are uses of the *joined* value, so they are not what declined the fold — they are
what makes the nulling observable, by reading the source after the `movc` wrote
`cnull` over it. Shape 2 is the other route: a read-back of the bridge's *own*
def, which declines the fold by itself.

**Shape 4 is not redundant with the lit test.** `c32-movc-untagged-live.ll`'s
`bridged_phi_residue` pins a different thing: there each bridge's *only* use is the
PHI, so no conforming copy is lost to the decline and its `movc` are copies of the
PHI result. The live site is a bridge with a conforming call-argument copy *and* a
PHI use — which is where all-or-nothing actually bites. A fix that removed the live
site's copy would once have left the lit suite green. That is no longer true:
`c32-movc-untagged-live.ll` now carries `bridged_callarg_plus_phi`, the same shape,
negative-tested two-sided. **The lit arm is the guard; this directory is not.**
Nothing runs `check.py` automatically — it is a development instrument for
iterating on a candidate fix in under a second, not part of any suite.

Shape 1 vs shape 2 is that difference as one variable: identical but for a
`ptrtoint`, and `mv` becomes `movc`.

## The positive control is not optional

Every `0` here asserts a copy is absent, and a blind instrument reports the same
thing. `check.py` therefore rebuilds the shapes with
`-capstone-enable-sink-fold=false` — the single switch design A's protection
actually rides on — and requires shape 1 to go 0 → non-zero. It does (0 → 2),
which also reproduces the pre-design-A defect on demand. If it ever stops, the
script exits 2 rather than reporting a clean run.

There is a second control, and it is the one that pins the STORY rather than the
instrument: shapes 2-5 are recorded as ones the fold declines outright, so turning
the fold off must leave them unchanged. It does. If one ever moves, the fold is
contributing to a shape reported as fold-declined and these numbers cannot be read
the way the summary reads them — so that exits 2 as well.

All six exit paths were negative-tested: normal (0), a mutated expectation (1), a
missing llc (2), a probe shape that cannot fire the control (2), a deleted shape
(2), and an orphan EXPECT-MOVC line (2). The last two were added after an audit
found that deleting a `define` previously exited 0 having checked one shape fewer,
with a summary line that still read like a full run.

## What it does not show

It does not measure the SQLite image; a codegen change still has to be scanned
with `movc-cfg-scan.py` over a real Sublet build before any claim about the
workload. It says nothing about the RTL — that `movc` nulls a non-NONLIN source
is C-32's other half and is unchanged here. And `real_cap_copy_control`'s three
`movc` are **correct**: a genuine capability must be copied with `movc`, and a
change that removes those has broken something.
