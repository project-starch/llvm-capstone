# xlang — next task set (Phase-2 seam)

**Written 2026-07-27.** Supersedes `xlang-phase1-followups-TODO.md`, whose items 1–3
are closed and whose item 4 is now the subject of this doc.

**Context.** The Phase-2 blocker turned out to be structural and it has been removed:
mruby already exposes `mrb_open_allocf`, only its stock `main()` never uses it, so
`xlang/shim/mruby_host.c` replaces `main()` instead of editing twelve vendored trees.
Three hooks rather than one, because a capability allocator has to distinguish cases
mruby's realloc-shaped hook conflates. Proven byte-identical to stock on rows 4 and 11
— the two extremes of the corpus's mruby version range. Read `xlang/shim/README.md`
first; it was written to be picked up cold.

**Board:** not needed for any item here. Everything is stock toolchain, decoupled from
our compiler, our QEMU fork and the FPGA — which is the point, since all three are in
flux. If a board session is ever wanted, it must be coordinated: the board is one
shared physical resource and sessions are serialized across lanes.

---

## 1. Replicate the seam across the nine remaining mruby rows

Rows 5, 6, 8, 9, 10, 12, 13, 14, 15. Run `build-mruby-host.sh`, repoint `run.sh` at
`xlang-host`, re-confirm each row's **already documented** verdict — same ASan error
class, same access size, same faulting line, same exit status.

That re-confirmation *is* the deliverable. It is the only evidence that swapping the
allocator was behaviour-preserving; without it, a Phase-2 result could be an artifact
of the harness rather than a property of the defect. Row 7 does not reproduce and needs
no seam.

Two traps already found, both in the shim README — carry them:
- At `-O1` the three seam functions inline into the allocator callback and vanish from
  ASan backtraces, so **a backtrace cannot distinguish "routed through the seam" from
  "mruby used its default allocator"**. `XLANG_SEAM_STATS=1` is the check that fails
  loudly. It cannot print on a row that aborts under ASan, so verify liveness on a
  non-crashing script — liveness is a property of the host, not the trigger.
- Rows disagree on the ASan build name (`host-asan` vs `host`); the build script
  auto-detects, but a hand-run will pick the wrong one.

## 2. Write the capability-allocator contract

A short spec — what `xlang_alloc` / `xlang_realloc` / `xlang_free` must do in Capstone
terms, not in prose about intent:

- `alloc` — mint a capability bounded to exactly the request.
- `realloc` that **moves** — derive a capability for the new block and **revoke** the
  old one; say explicitly what happens to the returned capability's bounds.
- `free` — revoke.
- And the part that actually decides the design: **what revocation does to a pointer
  the VM has already cached in a register or on its stack.** Row 4 is the worked
  example — its UAF is a write through a register-stack pointer cached across exactly
  the `xlang_realloc` call that frees the old stack. If a revoked capability in a
  register still faults on use, the row is caught; if revocation only invalidates
  memory-resident copies, it is not. That distinction is the whole benchmark.

This is the highest-leverage item for the project even though it is the smallest,
because it is the document the in-lane Phase-2 implementation will be written against,
and the person who just mapped where every row's stale pointer lives is the right
person to write it. Put it in `xlang/shim/` next to the seam it specifies.

## 3. The riscv64 leg

`xlang-host` is currently host-x86 and ASan-only. Our capability allocator exists only
on riscv64, so **Phase 2 cannot start on any row until that row reaches its faulting
site on riscv64** — this is the real gate, not a portability nicety. Needs
`riscv64-linux-gnu-gcc` and no ASan flags, which also means the fault presents as a
signal rather than an ASan report, so each row needs a new "what does success look
like" line written down.

Doing one row end-to-end and reporting what breaks is worth more here than doing all
of them; if the toolchain fights back, say so early rather than absorbing it.

## 4. Rows 1–2: the Rust seam

Same idea, different mechanism — `#[global_allocator]`. Lower priority than the mruby
rows only because it is 2 rows against 9, not because it matters less: it is what keeps
the "two subjects, genuinely cross-language" framing true at Phase 2 rather than only
at Phase 1.

Note row 1's constraint, established last week: the `ptr::read` patch stays. The
unpatched-upstream alternative was built and run and it degrades the artifact —
`mem::replace` writes undef bytes back into the userdata slot, so the fault becomes a
wild read ASan cannot attribute to any allocation, which is the wrong evidence for this
row. Don't re-litigate it; `xlang/1/target.md` has the reasoning.

## 5. Row 3 — leave for last

The allocation lives inside prebuilt `libpulse.so`, so there is no source-level seam;
it needs `LD_PRELOAD` interposition. One row, hardest mechanism, and the same
structural blindness that already forces it onto valgrind instead of ASan. Only worth
starting once 1–4 are done.

---

## Also worth having, if there is room

A paper-facing table of **what a bound catches that ASan does not**. The facts are
already in hand from the rows 6/11 reclassification: both rows read *in bounds* under
some parameters and fault under others, so under those parameters the defect is silent
to ASan *and* to a plain run. A bound on the index is the only mechanism that turns
either into a deterministic trap rather than a silent wrong answer. That is close to
the strongest single argument the corpus supports, and right now it exists only as two
paragraphs inside two row READMEs.

## Ground rules

- **No personal names** in anything committed or shared — commit subjects included.
  Use roles ("the collaborator", "the board owner", "the project lead").
- Never commit the FPGA console URL or token; the placeholder is `<FPGA-CONSOLE-URL>`.
- Manager-facing summaries go under `/tmp/capstone/`, never into the repo.
- Dated notes for investigations go in `capstone/agent-handoff/history/`
  (`DD-MM-YYYY_HH-MM-SS_name.md`); `design/` is for design decisions only.
- Built `xlang-host` binaries are gitignored — keep it that way.
