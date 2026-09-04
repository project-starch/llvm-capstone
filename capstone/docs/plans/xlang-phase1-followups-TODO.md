# xlang — short TODO to proceed (2026-07-27)

> **Status — updated 2026-07-27.** Items 1–3 are done. Item 4 is designed and
> proven on 2 of 14 rows; the remainder is mechanical but real. **The work is not
> committed yet** — it is sitting in the working tree. Per-row detail is in each
> row's `target.md` / `README.md`; the Phase-2 seam is documented in
> `xlang/shim/README.md`.

Phase-1 is in and merged to `capstone-bootstrap`. Nothing below needs our compiler,
our QEMU fork, or the board — all stock toolchain, so none of it can be blocked by
the churn on the silicon track.

**Priority order. 1–3 protect claims the paper will make; 4 is the one that saves us
real time later.**

1. **Rows 1 & 2 — corpus fidelity.** Both need a vendored patch to build on any
   post-2020 rustc, and row 1's patch **touches the destructor under test** (chosen to
   preserve the double-drop). Either find a patch that leaves the defect path
   untouched, or write two or three lines in `1/target.md` stating exactly what was
   changed and why the defect is still the upstream one. "Unpatched upstream source"
   is a claim we would like to keep making.

   > **DONE — the first option was tried and refuted, so the patch stays.** The
   > unpatched-destructor route does exist: rlua instantiates
   > `destructor::<RefCell<T>>`, so the rustc ≥1.48 abort fires only for a `T` that
   > forbids uninit. `String` holds a `NonNull`; a raw pointer + `usize` does not.
   > Changing the *harness* payload therefore lets the upstream destructor run
   > completely unmodified — and it was built and run that way. It does not reproduce
   > the defect: `mem::replace(obj, mem::uninitialized())` writes undef bytes back
   > into the userdata slot, so the resurrected handle reads a *garbage* pointer and
   > ASan reports `SEGV on unknown address` instead of `heap-use-after-free`. A
   > spatial-looking crash standing in for a temporal defect is the wrong evidence for
   > this row, so `ptr::read` is kept. Both traces are tabulated in `1/target.md`; row
   > 1 re-verified green (exit 1, PASS). Row 2 needed nothing — its patch is off the
   > defect path and `2/target.md` already said so.
   >
   > Net effect on the claim: we cannot say "unpatched upstream source" for row 1, but
   > we can now say *why*, with a tested alternative on the record rather than an
   > assertion.

2. **Confirm CVE-2026-1979 against NVD.** The #6701 mapping came from the upstream
   commit message and could not be checked offline. Needs network.

   > **DONE — confirmed, and it paid off twice.** The NVD record references issue
   > **#6701** and commit **`e50f15c1`** directly, names the affected component as the
   > **"JMPNOT-to-JMPIF Optimization"** in `mrb_vm_exec` / `src/vm.c`, and lists
   > versions 3.0–3.4.0. Two spin-offs: NVD assigns **CWE-119 as well as CWE-416**, so
   > row 6's spatial reclassification is *already on the record* — only the CNA's prose
   > calls it a use-after-free; and #6701 being spoken for independently settles row 7,
   > whose record mentions no bigint or `mrb_bint_reduce` anywhere. Written into
   > `6/target.md`, `7/target.md` and `xlang/README.md`.

3. **Rows 6 and 11 — write the reclassification, don't re-argue it.** The finding
   (both are spatial, not temporal) is accepted. What's needed is the paper-facing
   sentence for each: what the defect actually is, and why bounds rather than
   revocation are what stop it. Two short paragraphs in the row READMEs.

   > **DONE.** Two paragraphs each in `6/README.md` and `11/README.md`. The shared
   > spine of both: revocation acts on references that outlive their referent, and
   > neither row has one — row 6 frees nothing, row 11's temporal path is closed by
   > `envadjust()` and by environment-closing on escape. Each also leans on its own
   > "Tuning the trigger" facts: both rows read *in bounds* under some parameters and
   > fault under others, so a bound is the only mechanism that turns either into a
   > deterministic trap rather than a silent wrong answer.

4. **Phase-2 harness skeleton — the highest-value item.** For each reproducing row,
   factor its allocate → free → use sequence into a small, toolchain-agnostic shim so
   the capability version becomes a drop-in rather than a rewrite. Keep it building
   and passing under the stock toolchain exactly as it does today. This is the grindy
   half of Phase 2 and it is fully decoupled from our in-flux compiler/ABI, so it can
   proceed in parallel with the silicon work.

   > **PARTIAL — designed, and proven on 2 of 14 rows. See `xlang/shim/README.md`.**
   >
   > The blocker was structural, not conceptual: every mruby row runs the stock
   > `bin/mruby` binary, so there was nowhere to substitute an allocator without
   > editing twelve vendored mruby trees. mruby already has the seam —
   > `mrb_open_allocf()` takes a custom allocator — only its `main()` never uses it.
   > `xlang/shim/mruby_host.c` is that `main()`, yielding a drop-in
   > `<row>/xlang-host` with three replaceable bodies (`xlang_alloc` / `xlang_realloc`
   > / `xlang_free` — three rather than one, because a capability allocator must
   > distinguish mint, re-derive-and-revoke, and revoke).
   >
   > Verified byte-identical to stock on the two extremes of the corpus's mruby range:
   > row 4 (mruby 3.x, UAF WRITE, `vm.c:1426`) and row 11 (mruby 1.4.0, overflow READ,
   > `vm.c:1208`). The same source compiles unmodified against both.
   >
   > One trap worth carrying forward: at `-O1` the seam functions inline away, so an
   > ASan backtrace **cannot** distinguish "routed through the seam" from "mruby used
   > its default allocator". `XLANG_SEAM_STATS=1` prints live counts and is the check
   > that fails if a capability allocator is ever silently bypassed. Keep it.
   >
   > **Remaining:** nine more reproducing mruby rows (5, 6, 8, 9, 10, 12–15 — build
   > script, repoint `run.sh`, re-confirm each verdict); the riscv64 leg (cross
   > compiler, no ASan); rows 1–2 via Rust's `#[global_allocator]`; and row 3, which
   > has no source-level seam at all because the allocation lives inside prebuilt
   > `libpulse.so` — it needs `LD_PRELOAD` interposition, the same structural
   > blindness that already forces that row onto valgrind.

**Not now:** more corpus rows, and open-ended hunting for new cross-language defects.
Corpus size is not what the paper is short of — the capability half is.

**Board:** not needed for any of the above. The board is a single shared physical
resource and is serialized across everyone working on it, so if an xlang-on-silicon
step ever becomes worthwhile, coordinate a window first rather than assuming one.

**Reporting:** keep `xlang/README.md` and the dated state note in `history/` as the
single sources of truth; per-row detail stays in `<row>/target.md`.
