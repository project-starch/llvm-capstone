# SQLite gap 7 — interior-pointer capability-global not tag-initialized

*Status: FIXED 2026-07-03 (compiler, `CapstoneCapGlobalInit`). Surfaced after the
gap-6 `saveBuf` alignment fix. Same cap-global-init family as gaps 1–2, a new
variant. Past gap 7, SQLite advances to gap 8 (unaligned cap access), tracked
separately (task #82).*

## Symptom

Past gap 6, SQLite aborted QEMU with
`helper_cscincoffset: Assertion 'rs1_v->tag' failed` (`op_helper.c`): a
`cscincoffset` used an **untagged capability as its base**.

## Localization (QEMU diagnostic, since reverted; submodule clean)

Threaded the translate-time pc through `helper_cscincoffset` and printed the
untagged base. One event:
- guest pc `0x102042e4c` = **`sqlite3VdbeExec+0x3b08`** (the VDBE bytecode
  interpreter).
- op: a byte-table lookup `table[index]` (`cscincoffset x10,x10,x11`, index
  `x11 = 0x35 = OP_Eq`), base `x10` loaded via `ldc` (52e3c) from a
  GCT-materialized global, coming back **untagged** with cursor `0x15c9e5`.
- `0x15c9e5` = `sqlite3UpperToLower + 0xd1` (rodata) — a genuine pointer into
  read-only data that lost/never-had its tag.

## Root cause

The base is one of SQLite's comparison-result tables, declared as **global
pointer variables initialized to an interior address of another global**:
```c
SQLITE_PRIVATE const unsigned char *sqlite3aLTb = &sqlite3UpperToLower[256-OP_Ne];
SQLITE_PRIVATE const unsigned char *sqlite3aEQb = &sqlite3UpperToLower[256+6-OP_Ne];
SQLITE_PRIVATE const unsigned char *sqlite3aGTb = &sqlite3UpperToLower[256+12-OP_Ne];
```
(A SQLite space trick: overlay a 6-entry compare table onto 18 bytes appended to
the 256-byte `sqlite3UpperToLower`. `OP_Ne=53`, so the indices are 203/209/215,
in bounds only because of the appended tail — SQLite's own comment flags that they
"would be out-of-bounds and thus be undefined behavior".)

On a capability machine each such pointer global must be materialized as a
**tagged** capability at startup (tags can't live in a static ELF image). The
`CapstoneCapGlobalInit` pass collected globals whose initializer references a
`GlobalVariable`/`Function`, but tested that with `needsMaterialization` via
`stripPointerCasts()` — which strips **only zero-index GEPs**. An interior
initializer `&global[N]` (N≠0) is a ConstantExpr GEP with non-zero indices, so
`stripPointerCasts()` stopped at the GEP, `isa<GlobalVariable>` was false, and the
slot was never tag-initialized → loaded untagged → `cscincoffset` fault.

Note `stripInBoundsConstantOffsets()` would *also* miss it: because the index is
only in-bounds via the appended tail, clang does **not** mark the GEP `inbounds`.

## Fix

`llvm/lib/Target/Capstone/CapstoneCapGlobalInit.cpp`, `needsMaterialization`: peel
the ConstantExpr GEP / bitcast / addrspacecast chain directly to the underlying
target, **inbounds-agnostic** (a global-initializer GEP always has constant
indices, so peeling operand 0 is safe). The stored value keeps the full interior
offset, so the synthesized `__capstone_cap_init` emits
`cincoffsetimm <cap>, <base>, <N>; stc` and the tagged capability lands at the
correct cursor. One-line change of the strip call + comment.

## Validation

- Lit: new `CodeGen/Capstone/static-cap-global-init-interior.ll` (a **non-inbounds**
  interior pointer is materialized); full Capstone lit dir **35/35**.
- Authority suite **25/0** (rebuilt with the new clang).
- SQLite: the `cscincoffset` untagged-base assert is **gone**; execution advances
  to gap 8 (`Unaligned cap access` at `0x102237588`, 8-aligned not 16).
- Shared-pass regression gate (the pass runs for every domain): CoreMark PASS,
  RV8 7/7 PASS, BEEBS 82/82 (all failures observed during the gate were pre-boot
  infra flakes — `<no serial output>`, reproducibly passing standalone — never
  domain faults).

## Why this is a distinct gap (not gaps 1–2)

Gaps 1–2 were nested aggregate cap-globals and a clang aggregate-copy template.
Gap 7 is a **scalar pointer global whose initializer is an interior offset into
another global** — a detection blind spot in `needsMaterialization`, orthogonal to
the aggregate-walking logic (which was already correct).
