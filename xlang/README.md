# xlang — cross-language FFI memory-safety reproduction corpus

Phase-1 artifacts for `capstone/agent-handoff/plans/xlang-repro-task.md`. One
directory per row, each self-contained and buildable with a **stock toolchain** —
no Capstone compiler, no QEMU fork, no FPGA.

**14 of 15 rows reproduce.** Row 7 does not, and the evidence suggests the row as
specified does not exist.

## Status summary

| # | Boundary | Ref | Native | Class | RISC-V QEMU |
|---|---|---|---|---|---|
| 1 | Lua↔Rust | rlua #19 | ✅ heap-use-after-free | temporal | n/a — see below |
| 2 | Lua↔Rust | rlua #97 | ✅ stack-use-after-return | temporal | n/a — see below |
| 3 | Rust→C | GHSA-f56g-chqp-22m9 | ✅ invalid read (valgrind) | temporal | n/a — see below |
| 4 | Ruby↔C | CVE-2022-1071 | ✅ heap-use-after-free | temporal | ✅ SIGSEGV (139) |
| 5 | Ruby↔C | CVE-2022-1934 | ✅ heap-use-after-free | temporal | runs, exit 0 |
| 6 | Ruby↔C | CVE-2026-1979 | ✅ heap-buffer-overflow (WRITE) | **spatial** | ✅ SIGSEGV (139) |
| 7 | Ruby↔C gem | "mruby #6701 / bigint" | ❌ **not reproduced** | — | runs, exit 0 |
| 8 | Ruby↔C gem | mruby #4926 | ✅ heap-use-after-free | temporal | runs, exit 0 |
| 9 | Ruby↔C | mruby #3829 | ✅ heap-use-after-free | temporal | ✅ SIGSEGV (139) |
| 10 | Ruby↔C | CVE-2022-1106 | ✅ heap-use-after-free | temporal | runs, exit 0 |
| 11 | Ruby↔C | CVE-2018-10191 | ✅ heap-buffer-overflow | **spatial** | ✅ SIGSEGV (139) |
| 12 | Ruby↔C gem | mruby #4001 | ✅ heap-use-after-free | temporal | exit 1, caught `IOError` |
| 13 | Ruby↔C gem | mruby #4927 | ✅ heap-use-after-free | temporal | runs, exit 0 |
| 14 | Ruby↔C | mruby #3596 | ✅ heap-use-after-free | temporal | runs, exit 0 |
| 15 | Ruby↔C | mruby #3722 | ✅ heap-use-after-free | temporal | ✅ SIGSEGV (139)¹ |

¹ Row 15's QEMU fault is **heap-layout sensitive**: it segfaults when invoked with
absolute paths (as `run.sh` does) and exits 0 with short relative paths. Both are
individually deterministic. See `15/`.

**Crash sites** are recorded per row in `<row>/target.md`; `<row>/asan.txt` holds
the captured trace.

## Three findings that affect how this corpus should be described

### 1. Two rows are spatial, not temporal (rows 6 and 11)

Both the task spec and the benchmark table classify every row as a temporal borrow
(use-after-free), and the companion note states "every defect here is a temporal
borrow" and "There is no spatial or single-domain row, by selection". That does not
hold:

- **Row 11** (`OP_GETUPVAR` scope-level truncation) reproduces as
  `heap-buffer-overflow`. The plausible temporal path is closed at this mruby
  version by `envadjust()`, which rewrites `REnv::stack` on every stack realloc.
  Two attempts to force a dangling environment both failed. See `11/target.md`.
- **Row 6** (pattern-matching bytecode corruption) reproduces as
  `heap-buffer-overflow` **WRITE**. A corrupted register operand stores past the
  end of the VM stack; nothing is freed and reused. See `6/target.md`.

Neither is addressed by revocation — bounds are what stop them. Each row's
`target.md` lays out the options (reclassify vs. drop from the temporal subset),
and each row's `README.md` now carries the paper-facing statement of what the
defect actually is and why a bound rather than revocation is what catches it.

For row 6 the spatial reading is not merely our own: NVD assigns **CWE-119 as well
as CWE-416** to CVE-2026-1979. Only the CNA's prose description calls it a
use-after-free.

### 2. Row 7 appears not to exist as specified

Three independent problems, each verified against the source:

- Its issue number (**#6701**) belongs to **row 6** — the upstream fix `e50f15c1`
  says "Fixes #6701" and changes the pattern-matching peephole. Confirmed against
  NVD on 2026-07-27: the CVE-2026-1979 record references issue #6701 and commit
  `e50f15c1` directly and names the "JMPNOT-to-JMPIF Optimization" component, with
  no mention of bigint or `mrb_bint_reduce`.
- `mrb_bint_reduce` does **not exist** in mruby 3.1.0/3.2.0/3.3.0, contradicting
  the spec's placement of this row in the "single 3.1.0 build" Tier-1 cluster.
- The GC hazard the row describes is closed by mruby's allocation arena
  (`mrb_obj_alloc` → `gc_protect` roots every new object; nothing in the bigint or
  rational path saves/restores the arena).

The build works and is kept, so only a trigger would be missing if the row is real.
Full argument in `7/target.md`.

### 3. Row 3 needs valgrind, not AddressSanitizer

Row 3's stale dereference executes inside prebuilt `libpulse.so`, which carries no
sanitizer instrumentation. ASan poisons the region on free but never sees the read,
so the ASan build exits 0; valgrind instruments at machine level and catches it.
`3/asan.txt` therefore holds a valgrind report, and `3/run.sh` runs both legs so the
clean ASan pass is not mistaken for a failed reproduction.

This generalises: **for Rust→C rows where the stale dereference lands in a prebuilt
C library, ASan is structurally blind.**

## On the QEMU leg

Five rows fault under `qemu-riscv64`; six run to completion. That is expected and
permitted — task spec §4.2 accepts a plain non-ASan rv64 build that crashes *or
behaves anomalously*, and §9 states that where QEMU is impractical "the native ASan
reproduction plus the boundary annotation is still a complete and useful artifact".

Where a row exits 0, the reason is the same: without sanitizer instrumentation the
stale read or write lands on memory the allocator has not reused, so nothing traps.
Each `run.sh` documents its own observed behaviour per §4.2.

**Rows 1–3 have no rv64 build.** All three are Rust; the rv64 leg would need a
cross-compiled Rust toolchain plus cross-built C dependencies (rlua's vendored Lua,
or PulseAudio) — well beyond the stock-toolchain bar, and it would not change what
the defects are. The mruby rows carry the RISC-V evidence.

## Two rows require a toolchain patch

Rows 1 and 2 vendor a small patch, applied by `build.sh`, because the pinned rlua
predates changes that modern rustc enforces as hard errors. Each patch file carries
a header arguing why it does not mask the defect:

- **Row 1** — `mem::uninitialized()` → `std::ptr::read` at two sites. One of them
  *is* the userdata destructor under test, so the substitution was chosen to
  preserve the double-drop.
- **Row 2** — removes a trailing semicolon from the `rlua_panic!` macro body, far
  from the code under test.

Without these patches neither row builds at all on any post-2020 toolchain.

## Layout

Every row follows task spec §2:

```
<row>/
├── target.md      pinned commit(s), crash sites, verdict, caveats
├── build.sh       clean checkout -> runnable binary (stock toolchain)
├── <trigger>      trigger.rb, or trigger.lua + src/main.rs for Rust rows
├── run.sh         runs it and shows the violation; asserts PASS/FAIL
├── asan.txt       captured sanitizer trace (valgrind for row 3)
├── boundary.md    boundary annotation per §8
└── README.md      one screen: what it is, how to run, expected output
```

Rows 4–15 also carry `build_config.rb` (mruby host+ASan and riscv64 targets).
Rows 6 additionally has `bytecode-diff.txt`, a differential disassembly against a
fixed compiler.

## Running a row

```bash
cd <row> && ./build.sh && ./run.sh
```

Each `run.sh` exits non-zero if the expected violation does not appear, so the
corpus can be driven from a script.

### Host requirements

| For | Need |
|---|---|
| all mruby rows (4–15) | clang or gcc with `-fsanitize=address`, Ruby + Rake |
| RISC-V leg | `riscv64-linux-gnu-gcc`, `qemu-riscv64`, `/usr/riscv64-linux-gnu` sysroot |
| rows 1–3 | Rust **nightly** (`-Zsanitizer=address`), `llvm-symbolizer` |
| row 3 | `valgrind`, `libpulse-dev` |

Verified with clang 21.1.8, gcc riscv64 cross, qemu-riscv64 10.2.1, rake 13.3.1,
rustc/cargo 1.96.1 + nightly, valgrind, libpulse 17.0.

Build trees (`mruby/`, `rlua/`, `pulse-binding-rust/`, `target/`) are gitignored and
created on demand; the full corpus is roughly 3 GB built.

## Out of scope here

Per task spec §9: nothing in this directory uses the Capstone capability compiler,
capability-protected domains, revocation, or the FPGA. Each `boundary.md` ends with
a note on what a capability mechanism *would* do at that row's free site, but
nothing is implemented.
