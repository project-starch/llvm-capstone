# CHERI baseline — xlang corpus catch/no-catch results

**This is the empirical CHERI column for the xlang cross-language FFI corpus.**
For each in-scope defect it records whether CHERI-RISC-V **purecap** catches
the defect's offending access, and *when*, under three revocation
configurations. Measurement + classification: the **CHERI baseline, not our
system**.

Method, taxonomy and config definitions are the sqlite corpus's
(`capstone/tests/cheri-baseline/README.md`) applied unchanged, so the two columns are
comparable. Plan and design decisions:
`agent-handoff/plans/cheri-baseline-xlang.md`.

## Scope: 15 measured rows

The xlang corpus has 15 rows. **Row 7 is DROPPED**, leaving **14 measured** —
every remaining row.

**Row 7 is dropped because the defect it describes does not exist** — a defect
in the corpus specification, not a measurement gap, and a substantive negative
result: the trigger was built and driven under maximum GC pressure and
completes cleanly, natively and under QEMU. Three independent problems, each
verified against source (`xlang/repro/7/target.md`):

1. **Its issue number belongs to row 6.** mruby #6701 is closed by `e50f15c1`,
   which touches only the `JMPNOT`→`JMPIF` peephole. NVD confirms it: the
   CVE-2026-1979 record names that optimization and mentions no bigint.
2. **The function does not exist in the assigned version.** `mrb_bint_reduce`
   occurs 0 times in mruby 3.1.0/3.2.0/3.3.0; it first appears in the 3.4.0
   line and only under `MRB_USE_RATIONAL`. The spec groups row 7 into the
   "single 3.1.0 build" tier, so it was never buildable as specified.
3. **The described GC hazard is closed.** The code has the shape the spec
   describes, but `mrb_obj_alloc` roots every fresh allocation in the GC arena,
   so nothing dangles.

**Consequence to carry into the paper:** the companion note's claim that the
corpus spans "the bigint gem" rested entirely on row 7 and must be amended.
`xlang/repro/7/` is kept as the evidence trail for the negative result.

**All three Rust rows (1, 2, 3) are measured.** No purecap Rust toolchain
exists — cheribuild ships no Rust target — but none of these rows needs one,
because in each case the behaviour CHERI observes is language-independent.
Row 1's free runs inside Rust's drop glue, but the CHERI-relevant event is
just `malloc -> free -> read`; the double-drop is the *cause* of the early
free, not the behaviour under measurement, and Rust's allocator bottoms out
in malloc/free like everything else here. Row 2 is a
**stack**-use-after-return: no allocator is involved at all, so there is no
allocator route for a shim author to choose, and the machine-level event
(take a pointer into a frame, return, reuse the frame, read) is
language-independent (no allocator is involved at all). Row 3's memory
lifecycle is
entirely C-side (`pa_proplist_free` frees the block; `pa_proplist_iterate` ->
`pa_hashmap_iterate` reads through the iterator's raw pointer copy), so the
Rust binding only triggers it. Its shim reproduces valgrind's geometry
exactly — "32 bytes inside of 1072-byte", matching *Invalid read of size 8,
32 bytes inside a freed 1,072-byte block*. Rows 1 and 2 have no such route.

**Row 1 duplicates the allocator-visible UAF shape by construction** and is
included for completeness of the corpus table, not as independent evidence.
Say so when quoting it.

State the denominator as **14**, not 15, wherever these numbers appear.

## Toolchain / vehicle (config reality — state this in the paper)

| Component | Version |
|---|---|
| CHERI-LLVM (clang) | 17.0.0, CTSRD-CHERI `7e122876ee01` |
| CheriBSD purecap | 26.07 riscv64c distribution sets, FreeBSD 15.0-CURRENT |
| Emulator | `qemu-system-riscv64cheri` 7.1.0 (CTSRD-CHERI) |
| Firmware | `bbl-baremetal-riscv64-purecap` |
| cheribuild | `8e10ca19` |
| Build | shims compiled purecap `-O0` against the distribution rootfs as sysroot |

CHERI-LLVM is the **same commit** the sqlite column used, so the two columns
share a compiler. The vehicle is provisioned by
`capstone/tests/cheri-baseline/provision-cheri-vehicle.sh`; CheriBSD world is downloaded
rather than built (see that script's header for why building it is blocked on
a modern Linux host).

**Revocation is present and confirmed live.** The kernel exports
`security.cheri.runtime_revocation_default`,
`..._every_free_default` and `..._async`, and each pass records what the
runtime actually applied rather than what was requested:

| Config | `runtime_revocation_default` | `..._every_free_default` | `REVOKE_ENABLED` | meaning |
|---|:--:|:--:|:--:|---|
| **spatial** | 0 | 0 | 0 | CHERI spatial safety only (bounds + tags) |
| **temporal** | 1 | 0 | 1 | revocation ON, async quarantine — **realistic** default |
| **eager** | 1 | 1 | 1 | revoke on **every** free — expensive upper bound |

## Sanity (must hold or the row data is invalid)

`sanity_mock` **rc=0** — the lifecycle harness itself runs clean purecap, so a
fault in a row is the defect and not the scaffolding.

## Results (15 rows; rows 1-6 and 8-15 independently reproduced from a clean provision, row 7 measured twice identically on an existing vehicle)

**Reproduction status: verified end-to-end.** The whole vehicle was rebuilt
from an empty `CHERI_ROOT` — CHERI-LLVM, CHERI-QEMU, bbl firmware, downloaded
distribution sets, image — and the measurement re-run against it. All 14 row
verdicts are **byte-identical** to the original run, with `sanity_mock` rc=0
and `REVOKE_ENABLED` 0/1/1 in both. The 11-row subset additionally reproduced
across two earlier boots.

Verdicts: **BLOCKED-SYNC** = faults under `spatial` (caught at the access, no
revocation); **BLOCKED-SWEEP** = no spatial fault but a revocation config
faults (caught only by a sweep, *not* at the contract point);
**MISS** = survives every config. Exit codes: `SIGPROT`(162) = capability
fault, `exit0` = ran to completion.

| Row | Defect (class) | spatial | temporal (async, default) | eager (revoke/free) | verdict |
|----|----------------|:------:|:------:|:------:|----|
| 1 | rlua #19, Rust userdata freed by Lua `__gc` (UAF) | MISS | MISS | SIGPROT | blocked only *post-free* |
| 2 | rlua #97, escaped `Table` handle — **stack**-use-after-return | MISS | MISS | **MISS** | **never blocked in any config** |
| 3 | libpulse-binding GHSA-f56g-chqp-22m9, `Proplist::Iterator` UAF | MISS | MISS | SIGPROT | blocked only *post-free* |
| 4 | mruby CVE-2022-1071, VM-stack UAF (WRITE) | MISS | MISS | SIGPROT | blocked only *post-free* |
| 5 | mruby CVE-2022-1934, `hash_new_from_values` UAF | MISS | MISS | SIGPROT | blocked only post-free |
| 6 | mruby CVE-2026-1979, bytecode-corruption overflow | **SIGPROT** | SIGPROT | SIGPROT | **BLOCKED-SYNC** (bounds) |
| 7 | RUSTSEC-2022-0070, secp256k1 preallocated context UAF | MISS | MISS | SIGPROT | blocked only *post-free* |
| 8 | mruby#4926 / CVE-2020-6838, `hash_values_at` UAF | MISS | MISS | SIGPROT | blocked only post-free |
| 9 | mruby#3829, irep-pool string UAF (GC sweep) | MISS | MISS | SIGPROT | blocked only post-free |
| 10 | mruby CVE-2022-1106, `OP_RANGE_INC` UAF (WRITE) | MISS | MISS | SIGPROT | blocked only post-free |
| 11 | mruby CVE-2018-10191, `OP_GETUPVAR` truncation | **SIGPROT** | SIGPROT | SIGPROT | **BLOCKED-SYNC** (bounds) |
| 12 | mruby#4001, `File#initialize_copy` dangling DATA_PTR | MISS | MISS | SIGPROT | blocked only post-free |
| 13 | mruby#4927, `hash_slice` UAF | MISS | MISS | SIGPROT | blocked only post-free |
| 14 | mruby#3596, GC stack-root scanner UAF | MISS | MISS | SIGPROT | blocked only post-free |
| 15 | mruby#3722, `mrb_str_format` argv UAF | MISS | MISS | SIGPROT | blocked only post-free |

**Tally:** spatial-only blocks **2/15** (both genuine bounds catches).
Async-default temporal blocks **2/15** and **0/13 temporal defects** at the
contract point. Revoke-every-free blocks **14/15** — it cannot reach row 2.

### vs the predictions

**All 14 predictions in `rows.tsv` are confirmed**, and they were committed
before the first boot — so this is a test, not a description. Rows 6 and 11
were predicted BLOCKED-SYNC on the strength of the corpus's spatial
reclassification and their pinned trigger parameters; both faulted under
`spatial` as predicted.

## What the data says

1. **Base CHERI purecap (spatial only) is blind to the entire temporal
   class.** All 11 heap use-after-free rows and the stack row run to
   completion. Only the 2 spatial rows trap. This reproduces the sqlite column's central finding on a second,
   independent, cross-language corpus.

2. **Realistic CHERI temporal safety (async revocation, the default) still
   catches none of them at the contract point.** With revocation ON but
   sweeping on quarantine pressure, the short-lived reproducers free and reuse
   before any sweep runs, so every UAF row still MISSes. The dangling
   capability is revoked only by a later stop-the-world sweep, never
   synchronously at the lifecycle boundary.

3. **Rows 6 and 11 are real CHERI wins and should be reported as such.** They
   are spatial (heap-buffer-overflow), so bounds catch them with no revocation
   at all. They are also precisely the rows revocation does *not* address.

4. **Revocation is a HEAP mechanism, and row 2 is where that bites.** The
   stack-use-after-return survives **every** configuration, including
   revoke-on-every-free. A returned stack frame is never handed to the
   allocator, so quarantine-and-sweep has nothing to act on; and it is not a
   bounds violation either, since purecap bounds an address-taken local to
   that object and the capability stays exactly in bounds of storage that was
   merely reused. This is the corpus's clean **"CHERI cannot"** case, and
   unlike the allocator-hidden worst case it required no simulation to
   demonstrate — the analogue of the sqlite corpus's `3r` row.

5. **The corpus is less diverse than 15 rows suggests.** Six of the twelve
   temporal rows are one defect shape — a raw pointer into the VM register
   stack held across a callback that reallocs it. Report shapes, not only row
   counts.

## Caveat that must travel with these numbers: this is an UPPER BOUND

The shims link a lifecycle harness (`mock-mruby/`) that allocates and frees
with plain `malloc`/`free`, so **every object death is visible to
revocation**. Real runtimes are not like that: mruby's ordinary object death
is free-list reuse inside a still-mapped GC page, and FFI runtimes generally
use internal pools, arenas, slabs or lookaside allocators. None of those
frees reach the allocator, so revocation cannot see them.

The 14/15 that `eager` blocks is therefore **the most favourable possible
reading for CHERI**. The corresponding worst case is not measured here, and
was deliberately **not simulated**: hand-writing a pool allocator into the
mock would have manufactured the answer just as surely as modelling every
death as `free()` manufactures the opposite. A mock stays at malloc/free; a
richer allocator has to come from real software.

## Real mruby: builds purecap, does not run purecap

The honest route to the worst case is the real interpreter, so it was tried.

**It builds**, with three ABI-level one-line changes, none of which touch the
allocator, the GC, or object lifetime:

| Change | Why |
|---|---|
| `MRB_STR_EMBED_LEN_BIT` 5 → 6 | `RSTRING_EMBED_LEN_MAX` is `4*sizeof(void*) - 5`: 27 at 8-byte pointers, **59** at 16-byte capabilities, which no longer fits a 5-bit length field |
| `-ftls-model=initial-exec` | purecap CheriBSD requires it |
| `-cheri-tgot-tls` | purecap uses capability TGOT TLS; without it the binary keeps `R_RISCV_TLS_TPREL64` relocations and `ld-elf.so.1` rejects it with "Traditional TLS not supported" |

mruby's default value representation needed nothing — it already holds a real
`void *` rather than boxing pointers into words.

The resulting binary is genuinely capability-mode (`cheriabi, capability
mode`; ~32.6k capability instructions, 0 TPREL relocations).

**It does not run.** On `puts 1` it takes a capability fault, **SIGPROT
(rc=162)**, before reaching any defect. Control: in the same image, on the
same kernel, the purecap C shims run clean (`sanity_mock` rc=0), so the
vehicle is sound and what faults is mruby itself.

This is the direct analogue of the sqlite corpus's finding that upstream
SQLite dies with `SIGBUS` inside `sqlite3_open` before reaching its injected
bug. Both corpora independently hit the same wall: **the real runtime needs a
capability-cleanliness port, not a build fix.** Quantifying that gap — builds
in three one-liners, faults immediately at runtime — is itself a result worth
reporting.

**Consequence:** any worst-case (allocator-hidden) CHERI number for this
corpus is blocked on porting mruby to be capability-clean, a project on the
scale of the SQLite purecap port. Until then, report the upper bound above
with this caveat attached.

## Reproduce

```bash
capstone/tests/cheri-baseline/provision-cheri-vehicle.sh   # build the vehicle
xlang/cheri/run-cheri-baseline-xlang.sh
```

Before trusting any row: `sanity_mock` must be rc=0, and `REVOKE_ENABLED`
must read 0/1/1 across the three passes.
