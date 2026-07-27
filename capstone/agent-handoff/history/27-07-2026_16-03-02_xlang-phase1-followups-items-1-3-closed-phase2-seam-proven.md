# xlang Phase-1 follow-ups: items 1–3 closed, Phase-2 seam designed and proven

**Date:** 2026-07-27
**Lane:** A (Opus), branch `capstone-bootstrap-xlang-repro`
**Base at time of writing:** `03e2ced6cf86` (in sync with `origin/capstone-bootstrap`)
**Driving doc:** `capstone/agent-handoff/plans/xlang-phase1-followups-TODO.md`
**Board:** not used. Nothing here touches the FPGA, our compiler, or our QEMU fork.

> **State: the work described below is UNCOMMITTED.** It is in the working tree
> only. See "Uncommitted inventory" at the end before doing anything else.

---

## Summary

Worked the four-item xlang follow-up TODO in its stated priority order. Items 1–3
are closed. Item 4 (the Phase-2 harness) is designed and demonstrated on 2 of 14
rows; the rest is replication, not design.

The one result that changes a plan rather than executing it is **item 1**: the
preferred option turned out to be achievable *and wrong*, so the existing patch
stays. Detail below, because the reasoning is the deliverable.

---

## Item 1 — rows 1 & 2 corpus fidelity: preferred option tried, refuted

The TODO offered two routes: find a patch leaving row 1's defect path untouched,
or document precisely what the current patch changes. The first was attempted,
since "unpatched upstream source" is a claim worth keeping.

**It is reachable.** rlua instantiates the userdata destructor as
`destructor::<RefCell<T>>` (`rlua/src/lua.rs:1607`). The rustc ≥1.48 abort inside
`mem::uninitialized::<T>()` therefore fires only when `T`'s all-uninitialised
bit-pattern is invalid. `String` holds a `NonNull` and so is invalid; a raw pointer
plus a `usize` is not. Making the *harness* payload an all-bits-valid heap owner
(`xlang/1/src/main.rs` — ours, never upstream) lets the upstream destructor compile
and run **completely unmodified**, shrinking the patch to its one off-path hunk.

**It was built and run that way, and it does not reproduce the defect:**

| Destructor | Observed |
|---|---|
| `ptr::read` (current patch) | `heap-use-after-free`, READ of size 43, offset 0 of the freed region, free site attributed to `destructor::<RefCell<Userdata>>` |
| `mem::uninitialized` (pristine upstream) | `SEGV on unknown address`, "dereference of a high value address" |

Cause: `mem::replace(obj, mem::uninitialized())` *writes* undef bytes back into the
userdata slot. The resurrected handle then reads a garbage pointer rather than the
freed one, so the fault is a wild read ASan cannot attribute to any allocation — a
spatial-looking crash standing in for this row's temporal defect. That is the wrong
evidence for row 1, so `ptr::read` is kept and is the more faithful reproduction.

Row 1 was restored and **re-verified green** (`heap-use-after-free`, READ size 43,
freed by `destructor::<RefCell<Userdata>>` at `util.rs:279`, exit 1, PASS).

Row 2 needed no work: its patch (a trailing semicolon in `rlua_panic!`) is off the
defect path and `2/target.md` already said so.

**Where it is written down:** `xlang/1/target.md`, new section "Leaving the
destructor unpatched was tried, and it degrades the artifact".

**Consequence for the paper:** we cannot claim unpatched upstream source for row 1.
We can now state exactly why, with a tested alternative on the record.

---

## Item 2 — CVE-2026-1979 confirmed against NVD

Checked 2026-07-27 (needed network; previously unverifiable offline).

The NVD record references mruby issue **#6701** and fix commit
**`e50f15c1c6e131fa7934355eb02b8173b13df415`** directly, names the affected
component the **"JMPNOT-to-JMPIF Optimization"** in `mrb_vm_exec` / `src/vm.c`, and
lists affected versions **3.0–3.4.0**. Published 2026-02-06, modified 2026-06-17.
All of this matches row 6 exactly. The mapping no longer rests on a commit message.

Two consequences worth carrying into the paper:

1. **NVD assigns CWE-119 as well as CWE-416.** Row 6's spatial reclassification is
   therefore not a contradiction of the CVE — the spatial reading is already on the
   record. Only the CNA's (VulDB's) prose description says "use after free".
2. **Row 7's negative result is independently settled.** #6701 is *this* bug; the
   NVD record mentions no bigint, no `mrb_bint_reduce`, no rational path. Row 7's
   described defect has no issue backing it.

**Where:** `xlang/6/target.md`, `xlang/7/target.md`, `xlang/README.md`.

---

## Item 3 — rows 6 and 11 reclassification written

Two paragraphs each, paper-facing, in `xlang/6/README.md` and `xlang/11/README.md`
under "Classification: spatial, not temporal". The finding itself was already
accepted; this is the statement of it, not a re-argument.

Shared spine of both: revocation acts on references that outlive their referent,
and neither row has one — row 6 frees nothing at all, and row 11's temporal path is
closed by `envadjust()` rewriting `REnv::stack` on every realloc and by
environment-closing when a `Proc` escapes.

Each also uses its own "Tuning the trigger" facts, which turned out to be the
strongest argument available: **both rows read in bounds under some parameters and
fault under others** (row 6 at shallow recursion, row 11 with <~80 outer locals).
Under those parameters the defect is silent to ASan *and* to a plain run. A bound
on the index is the only mechanism that converts either into a deterministic trap
rather than a silent wrong answer. That is the sentence the paper wants.

---

## Item 4 — Phase-2 seam: designed, proven on 2 of 14 rows

**The blocker was structural, not conceptual.** Every mruby row (4–15) reproduces
by running the stock `bin/mruby` binary on its `trigger.rb`. The allocate → free →
use sequence under test happens entirely inside the VM, behind an allocator the row
cannot reach. Substituting a capability allocator would have meant editing twelve
vendored mruby trees.

mruby already exposes the seam — `mrb_open_allocf()` takes a custom allocator — and
only the stock `main()` never uses it. `xlang/shim/mruby_host.c` is that `main()`.
It yields `<row>/xlang-host`, a drop-in replacement for
`<row>/mruby/build/<build>/bin/mruby`, routing every VM allocation through three
replaceable functions:

```c
void *xlang_alloc(size_t size)            { return malloc(size); }
void *xlang_realloc(void *p, size_t size) { return realloc(p, size); }
void  xlang_free(void *p)                 { free(p); }
```

Three rather than one because a capability allocator must distinguish cases mruby's
single realloc-shaped hook conflates: mint a bounded capability; derive one for a
moved block and revoke the old; revoke outright. Row 4 is the worked example — its
UAF is a write through a register-stack pointer cached across exactly the
`xlang_realloc` call that frees the old stack.

**Verified byte-identical to stock on the two extremes of the corpus's mruby range:**

| Row | mruby | Stock | Via `xlang-host` |
|---|---|---|---|
| 4 | 3.x | `heap-use-after-free`, WRITE size 8, `vm.c:1426` in `mrb_vm_exec`, exit 1 | identical |
| 11 | 1.4.0 | `heap-buffer-overflow`, READ size 16, `vm.c:1208` in `mrb_vm_exec`, exit 1 | identical |

The same source compiles unmodified against both, so the API used
(`mrb_open_allocf`, `mrb_load_file`, `mrb_print_error`) is stable across the whole
range the corpus spans.

### Trap to carry forward

At `-O1` the three seam functions inline into the allocator callback, so they never
appear in an ASan backtrace — **a trace alone cannot distinguish "routed through the
seam" from "mruby used its default allocator"**. `XLANG_SEAM_STATS=1` prints live
counts (`alloc=1668 realloc=3 free=1883` on a trivial script) and is the check that
fails loudly if a capability allocator is ever silently bypassed. Keep it in Phase 2.
It cannot print on a row that aborts under ASan; verify on any non-crashing script,
since liveness is a property of the host, not the trigger.

### Incidental finding

Rows disagree on the ASan build name: rows 4, 5, 10 use `host-asan`; the other nine
use `host`. `build-mruby-host.sh` auto-detects.

### Remaining for item 4

- **Nine reproducing mruby rows** (5, 6, 8, 9, 10, 12–15): run the build script,
  repoint `run.sh` at `xlang-host`, re-confirm each documented verdict — that
  re-confirmation is the only thing proving the swap was behaviour-preserving. Row 7
  does not reproduce and needs no seam.
- **The riscv64 leg**: host is ASan-only; cross target needs
  `riscv64-linux-gnu-gcc` and no ASan flags.
- **Rows 1–2 (Rust)**: different seam, same idea — `#[global_allocator]`.
- **Row 3**: hardest. The allocation lives inside prebuilt `libpulse.so`, so there
  is no source-level seam; needs `LD_PRELOAD` interposition. Same structural
  blindness that already forces row 3 onto valgrind instead of ASan.

This remainder is stock-toolchain and decoupled from the compiler/ABI/board, so it
is a candidate for the external collaborator. `xlang/shim/README.md` was written to
be handed over as-is.

---

## Uncommitted inventory

Nothing below is committed. Branch `capstone-bootstrap-xlang-repro` is at
`03e2ced6cf86`, in sync with `origin/capstone-bootstrap`.

| Path | Change |
|---|---|
| `xlang/1/target.md` | Item 1: tested-and-refuted alternative |
| `xlang/6/target.md` | Item 2: NVD confirmation block |
| `xlang/7/target.md` | Item 2: NVD independently settles the negative result |
| `xlang/README.md` | Item 2 spin-offs; pointer to the new README sections |
| `xlang/6/README.md`, `xlang/11/README.md` | Item 3: reclassification paragraphs |
| `xlang/shim/` **(new)** | Item 4: `mruby_host.c`, `build-mruby-host.sh`, `README.md` |
| `xlang/.gitignore` | Ignores built `xlang-host` binaries |
| `capstone/agent-handoff/plans/xlang-phase1-followups-TODO.md` | Status annotations (additions only, 0 deletions) |

Note on that last file: another lane committed the same TODO in `d8bd4052f60e`
while this work was in flight. The committed text was taken as the base and the
status annotations re-applied on top, so the diff is pure addition and does not
churn the other lane's punctuation.

Built artifacts `xlang/4/xlang-host` and `xlang/11/xlang-host` exist locally and
are gitignored.

## Suggested next step

Commit this as one unit (name-scan the message and staged diff first, per the
project rule), then decide whether item 4's remaining rows stay in-lane or go to
the external collaborator.
