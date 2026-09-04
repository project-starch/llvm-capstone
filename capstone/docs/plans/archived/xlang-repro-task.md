# Task: reproduce the cross-language FFI memory-safety bugs (Phase 1 — QEMU / stock toolchain)

This is a **self-contained, tool-agnostic** task spec for reproducing a set of upstream
cross-language memory-safety bugs. It does **not** assume Claude Code, does not rely on any
auto-loaded project config, and does **not** require our custom LLVM-Capstone compiler, our
QEMU fork, or the FPGA. If you drive a coding agent (e.g. OpenAI Codex), give it this whole
file as context plus the specific bug you are working on. Read it top to bottom.

> If you are pasting this to a coding agent that does **not** auto-read repository config
> files (Codex, Cursor, etc.): everything the agent needs is in this one file. Give it this
> file plus the specific bug you are working on.

---

## 1. The goal (Phase 1 only)

Turn each cross-language memory-safety defect listed in §6 into a **reproducible artifact**:

> a **minimal, deterministic** program + trigger that, on a **stock toolchain**, crashes with
> a **memory-safety violation** (use-after-free or double-free) — first under **native
> AddressSanitizer** (fast, gives a clean trace), then under **plain RISC-V QEMU** — together
> with a short **boundary annotation** describing which pointer crosses the language boundary
> and where its lifetime is violated.

That is the entire Phase-1 deliverable. **You do not need to touch any special compiler, any
capability hardware, or any FPGA.** Those belong to a later phase owned by the core team; see
§9 "Out of scope."

**Why we want this.** These are *cross-language* bugs: two runtimes (a managed one — Lua or
Ruby — and a native one — Rust or C) share one address space through a foreign-function
interface (FFI). One side's garbage collector frees or moves an object the other side still
holds, and the unsafe FFI "shim" lets the stale reference be used. The reproducible artifacts
become the security benchmark for a hardware memory-safety mechanism (linear capabilities with
revocation) that the core team applies in a later phase. For Phase 1 you only need to make the
bug **fire deterministically** and **document the boundary** — you are not fixing it.

---

## 1a. Scope & autonomy — own this end to end (no per-row sign-off)

**This is a standing, autonomous task. Do the whole set; don't wait for us between rows.** The
mission is: **reproduce every bug in the §6 table, each with both a native-ASan trace and a
RISC-V QEMU run**, and land them all. Work at your own pace, in whatever order is efficient
(the mruby Tier-1 cluster shares one `3.1.0` checkout — batch it), and integrate as you go.

- **Definition of done for the whole task:** every row in §6 has its `<row>/` artifact directory
  with `target.md`, `boundary.md`, the trigger, `asan.txt` (native ASan heap-UAF / double-free),
  and `build.sh` + `run.sh` that reproduce **both** the native-ASan crash **and** the RISC-V
  QEMU run (`qemu-riscv64 -L /usr/riscv64-linux-gnu …`). Row 10 (already landed) is the template
  — match its shape.
- **Suggested flow:** finish the native-ASan repros across the cluster first (fast iteration,
  clean traces), then do the RISC-V QEMU pass across the same builds. Or interleave per row —
  your call. Either way both must be present at the end.
- **Don't ask us to pick the next row or approve each one.** If a specific bug turns out to be
  infeasible on a stock toolchain (needs a since-removed gem, won't build at the pinned version,
  no public trigger), **skip it, note why in that row's `target.md`, and move on** — a short
  written "skipped because X" is a perfectly good outcome for a hard row. Flag us only when you
  finish the set, or if you hit something that blocks *many* rows at once.
- **Commit + push to your branch as you complete rows** (a branch like `xlang-repro` or the
  existing CVE branch is fine); we integrate from there. Small, per-row-or-per-batch commits.
- **HARD RULE — no personal data in committed files.** ASan traces embed the absolute build
  path in every frame. **Scrub it before committing** (build under a neutral dir, or
  `sed -i "s#$HOME/<...>/#/path/to/#g" asan.txt`). No usernames, real names, or home paths in
  any committed file. Keep the file/line/offset info — only the leading path changes.

---

## 2. What a "reproducible artifact" must contain

For each bug, produce a self-contained directory `xlang/<id>/` with:

1. **`target.md`** — the exact upstream subject and **pinned version/commit** (e.g. `mruby
   3.1.0`, git tag or SHA). Never "latest".
2. **`build.sh`** — from a clean checkout to a runnable binary, **stock toolchain only**.
   Two build modes where possible:
   - **native + ASan** (`-fsanitize=address -g -O1 -fno-omit-frame-pointer`), and
   - **riscv64** (stock cross-toolchain), for the QEMU run.
3. **The trigger** — the minimal input that fires the bug: a `.rb` script for mruby, a `.lua`
   script + tiny Rust host for the Lua-Rust rows, or a small C harness. Keep it as small as
   possible (see §7 minimization).
4. **`run.sh`** — runs the trigger and **deterministically** shows the violation. Capture the
   crash: the ASan report for the native run, and the crash/behavior for the QEMU run.
5. **`asan.txt`** — the captured AddressSanitizer report (the ground-truth proof it is a UAF /
   double-free, with the freeing and using stack frames).
6. **`boundary.md`** — the annotation (see §8): the object that crosses the FFI, the "owner"
   side, the "borrower" side, the point where it is freed, and the point where the stale use
   happens. One paragraph + the two source locations is enough.
7. **`README.md`** — one screen: what the bug is, how to build and run, expected output, and a
   one-line "PASS = the sanitizer/QEMU shows the UAF at `<function>`".

Keep every artifact **deterministic** (fixed seeds, fixed versions, no network at run time)
and **hermetic** (build from the pinned source; vendor or script any dependency).

---

## 3. Environment (stock — no custom compiler)

You need ordinary tools only:

- A host C/C++ toolchain with AddressSanitizer: **clang** (or gcc) `-fsanitize=address`.
- **Ruby + Rake** (mruby builds with Rake) and a C toolchain — for the mruby subjects.
- A **stock RISC-V toolchain + QEMU** for the QEMU reproduction:
  - Cross-compiler: `riscv64-linux-gnu-gcc`/`clang --target=riscv64-linux-gnu` (any stock
    RISC-V GCC/LLVM; the distro package is fine).
  - QEMU: `qemu-riscv64` (user-mode, simplest — run a riscv64 ELF directly) or
    `qemu-system-riscv64` (full-system) if a bug needs a real kernel. **Start with user-mode
    `qemu-riscv64`** — it is the least friction and enough for a single-process crash.
- For the Lua-Rust rows only: a **Rust toolchain** (`rustup`, stable) and the `mlua`/`rlua`
  crates at the versions named in the tracker. (Do these last.)

Nothing here is our project's custom software. If your agent tries to fetch a "Capstone"
compiler or a special QEMU, **stop it** — Phase 1 is stock-toolchain only.

---

## 4. Reproduction strategy (native first, then QEMU)

For each bug:

1. **Confirm it natively under ASan first.** This is the fastest path to a clean, labeled
   trace and confirms you have the right version + trigger. The xlang defects all have public
   AddressSanitizer traces (§6 links), so your native trace should match the published one
   (same crashing function, e.g. `mrb_vm_exec`).
2. **Then reproduce under RISC-V QEMU** (the meeting's required target). Build the same target
   for `riscv64` with the stock cross-toolchain and run the trigger under `qemu-riscv64`.
   Notes:
   - ASan **on RISC-V under user-mode QEMU can be finicky**; if ASan-on-riscv is unavailable
     or unstable, a **plain (non-ASan) riscv64 build** that **crashes / behaves anomalously**
     on the trigger under `qemu-riscv64` is an acceptable QEMU reproduction — the native ASan
     run is the authoritative "it is a UAF" evidence, and the QEMU run shows it reproduces on
     the RISC-V target. Document exactly what you observed (segfault, corrupted output,
     assertion) in `run.sh`'s expected output.
   - Prefer a build with frame pointers and symbols so a QEMU+gdb backtrace localizes the
     crash if it is not an outright ASan report.
3. **Record both** in the artifact (native ASan report in `asan.txt`; QEMU behavior in the
   `README.md`/`run.sh`).

---

## 5. Worked example (the template) — mruby row 10, CVE-2022-1106

Do this one **first** and use it as the pattern for the rest. (Row 10 is a UAF in the mruby VM
loop `mrb_vm_exec`, mruby < 3.2, with fix commit `7f5a490d`.)

```bash
# 5.1 Get the pinned source (single build covers the whole pre-3.2 cluster).
git clone https://github.com/mruby/mruby
cd mruby
git checkout 3.1.0          # pins the pre-3.2 UAF cluster (rows 4,5,7,8,9,10,12,13,14,15)

# 5.2 Native + ASan build. mruby uses Rake + a build_config.rb; set sanitizer flags there
#     (CC/LD flags: -fsanitize=address -g -O1 -fno-omit-frame-pointer) or via a custom
#     build_config that adds them to conf.cc.flags / conf.linker.flags. Then:
MRUBY_CONFIG=<your-asan-config>.rb rake

# 5.3 Trigger. Fetch the minimal PoC script from the CVE's public tracker / NVD reference
#     (the mruby issue linked for this row carries the reproducing .rb and an ASan trace).
#     Save it as poc.rb. Run the built interpreter on it:
./build/host/bin/mruby poc.rb        # expect: AddressSanitizer: heap-use-after-free in mrb_vm_exec

# 5.4 Capture the ASan report -> asan.txt. Confirm the crashing frame matches the published
#     trace (mrb_vm_exec). That is your native reproduction.

# 5.5 RISC-V QEMU. Rebuild mruby for riscv64 with the stock cross toolchain (a build_config
#     with conf.toolchain :gcc and CC/AR/LD pointing at riscv64-linux-gnu-*), then:
qemu-riscv64 -L /usr/riscv64-linux-gnu ./build/riscv/bin/mruby poc.rb   # expect the same crash
```

Then write `target.md`, `build.sh`, `run.sh`, `asan.txt`, `boundary.md`, `README.md` per §2,
and the annotation per §8. That directory is the model every other bug copies.

> Getting the exact `build_config.rb` right (sanitizer flags for the native build; the
> riscv64 cross toolchain for the QEMU build) is the main fiddly part — this is a good thing
> to have your coding agent iterate on. mruby's `doc/guides/compile.md` documents build
> configs.

---

## 6. The target list (do them in this order)

From `xlang.tex` Table 2 (the temporal-borrow subset). **All are use-after-free or
double-free.** Start with the mruby cluster (pure C, standalone host, public ASan traces).

**Tier 1 — mruby, single `3.1.0` build (do these first, they share one checkout):**

| # | Ref | Site / class | Where to get the PoC |
|---|-----|--------------|----------------------|
| 10 | CVE-2022-1106 | UAF in `mrb_vm_exec` (fix `7f5a490d`) | github.com/mruby/mruby/commit/7f5a490d + NVD |
| 4 | CVE-2022-1071 | UAF in `mrb_vm_exec` | nvd.nist.gov/vuln/detail/CVE-2022-1071 |
| 5 | CVE-2022-1934 | UAF in mruby VM | nvd.nist.gov/vuln/detail/CVE-2022-1934 |
| 7 | mruby #6701 | UAF in `mrb_bint_reduce` (bigint gem), read by VM | github.com/mruby/mruby/issues/6701 |
| 8 | mruby #4926 | UAF in `hash_values_at` (mruby-hash-ext) | github.com/mruby/mruby/issues/4926 |
| 9 | mruby #3829 | UAF in `mrb_gc_mark` (marks freed object) | github.com/mruby/mruby/issues/3829 |
| 12 | mruby #4001 | dangling `DATA_PTR`, `File#initialize_copy` (mruby-io) | github.com/mruby/mruby/issues/4001 |
| 13 | mruby #4927 | UAF in `hash_slice` (mruby-hash-ext) | github.com/mruby/mruby/issues/4927 |
| 14 | mruby #3596 | UAF in `mark_context_stack` | github.com/mruby/mruby/issues/3596 |
| 15 | mruby #3722 | UAF in `mrb_str_format` via `sprintf` | github.com/mruby/mruby/issues/3722 |

(Some gem rows — 7,8,12,13 — need the relevant mrbgem enabled in the build config. Enable
`mruby-bigint`, `mruby-hash-ext`, `mruby-io` as needed.)

**Tier 2 — mruby, other builds:**

| # | Ref | Build | Site |
|---|-----|-------|------|
| 6 | CVE-2026-1979 | mruby ≤ 3.4.0 | UAF in `mrb_vm_exec` (codegen mislabels JMP as JMPNOT) |
| 11 | CVE-2018-10191 | mruby ≤ 1.4.0 | UAF in upvalue/env stack (`OP_GETUPVAR` int overflow) |

**Tier 3 — Lua-in-Rust (do last; needs a Rust toolchain):**

| # | Ref | Site |
|---|-----|------|
| 1 | rlua #19 | double-free/UAF: `__gc` runs the Rust destructor more than once |
| 2 | rlua #97 | UAF: understated callback lifetime; captured Rust ref outlives scope |
| 3 | GHSA-f56g-chqp-22m9 | Rust→C UAF: iterator outlives the C object (libpulse-binding) |

For Tier 3: a tiny Rust host that embeds Lua via `mlua`/`rlua` at the tracker-named version,
plus the minimal Lua script / callback that triggers the lifetime clash. Rows 1–2 are
reproduced from the rlua issues directly (rlua #19 was fetched and confirmed upstream).

---

## 7. Minimization

Strip each repro to the **smallest program that still crashes**:

- Remove everything the crash does not need (unrelated requires, gems, host code).
- Prefer a single script / single small C or Rust file.
- Keep it **deterministic**: no timing/network/random dependence; if GC timing matters, force
  collection explicitly (e.g. `GC.start` in Ruby) so the free happens at a fixed point.
- The goal (per the project lead) is "just enough for the crash" — a DST-style minimal case,
  not the whole application.

---

## 8. Boundary annotation (`boundary.md`)

For each bug, in a short paragraph + two code locations, state:

- **The object that crosses the FFI** (e.g. a wrapped C struct behind a Ruby `DATA_PTR`; a
  Rust value handed to Lua as userdata; a borrowed string pointer).
- **Owner vs. borrower:** which side allocates/frees, which side holds the stale reference.
- **The free site** (function + file:line) and **the stale-use site** (function + file:line).
- **The lifetime rule that is violated** in one sentence (e.g. "the C object is freed in
  `initialize_copy` on a bad argument, but the wrapper's `DATA_PTR` still points at it and is
  read in `fptr_finalize`").

This is what later lets the object be modeled as a capability that is **revoked** at the free
site, so the stale use **faults** instead of corrupting memory. You are only *documenting* the
boundary here, not implementing anything.

---

## 9. Out of scope for this task (owned by the core team, later)

Do **not** attempt any of these in Phase 1 — they belong to a later phase and depend on
in-flux, project-specific infrastructure:

- Compiling any target with the custom capability compiler.
- Running inside capability-protected domains, or applying revocation.
- Anything on the FPGA / capability hardware.
- Any change to the project's LLVM fork, QEMU fork, or runtime.

If a bug simply will not reproduce on stock RISC-V QEMU after reasonable effort, **note it and
move on** — the native ASan reproduction plus the boundary annotation is still a complete and
useful artifact. Flag anything ambiguous back to the core team rather than guessing.

---

## 10. Ground rules for artifacts

- **Pin every version** (git tag/SHA); no "latest"; hermetic builds.
- **Deterministic** repros only.
- **No real-person names** anywhere in the artifacts (files, commits, comments) — use neutral
  descriptions.
- Do **not** commit editor/agent scratch files, debug checkpoints, or session notes.
- One directory per bug (§2); a top-level `xlang/README.md` with a summary table of which rows
  reproduce natively and under QEMU.

---

## 11. Reporting back

When a bug's artifact is done, report: the row #, the pinned version, "native ASan:
reproduced (crashing frame `<fn>`)", "riscv64 QEMU: reproduced / behavior observed", and the
one-paragraph boundary annotation. The core team will integrate the artifacts and take them
into the capability phase.
