# Capstone column — xlang corpus catch/no-catch results

**The Capstone half of the xlang comparison.** For each defect it records whether
Capstone's capability mechanism catches the offending access, and by which
mechanism. Read alongside `../cheri/RESULTS.md`, which is the same corpus, the
same shims and the same taxonomy measured under CHERI.

Predictions were committed to `rows.tsv` **before any of these ran** (commit
`91df3f87b72f`). 12 of 14 held; the two that did not are reported as misses
below rather than quietly rewritten.

## Vehicle

| Component | Version |
|---|---|
| Compiler | our capstone64-unknown-elf clang, `llvm/cmake-build-debug` |
| Emulator | our Capstone QEMU fork (`capstone/capstone-qemu`) |
| Harness | `capstone/tests/runtime-qemu/run-domain-smoke.py`, one boot per run |
| Build | shims compiled `-O0`, `ROF_MAX_SLOTS=64`, 64 KiB linear arena |
| Allocator | `revoke_on_free_alloc.h` — every allocation a SPLIT sub-capability with its own revocation node; `free` REVOKEs |

`-O0` is required, not preferred: at `-O1`+ the compiler hoists the cached
pointer's load above the free or elides the dangling access entirely, so the
access the mechanism must police is never emitted. The CHERI column builds `-O0`
for the same reason.

## Configs

| Config | Meaning | CHERI counterpart |
|---|---|---|
| **revoke** | revoke-on-free, the design point | CHERI **eager** (revoke on every `free`) |
| **control** | identical program, allocator and free path, **minus the one REVOKE** | — |

The control is not a config; it is the attribution test. A row's FAULT counts
only if its control MISSes, because otherwise the fault is not attributable to
the revoke. Two "controls" earlier in this work turned out not to control
anything (see the 2026-08-01 history note), which is why this is enforced per
row rather than assumed.

**This column uses PERVASIVE revocation, not the boundary-only scheme** the
design describes (`plans/compatibility-eval-silicon-app.md`: "the revoke-at-free
machinery is applied only to the pointers that cross the host↔engine boundary").
`rof_malloc`/`rof_free` make **every** allocation independently revocable, so
these numbers are an **upper bound** on what boundary-only protection achieves.
Row 7 is the corpus's cleanest cross-domain lend and would qualify under either;
the six VM-register-stack rows (4, 5, 8, 10, 13, 15) would **not**, because
their stale pointer is engine-internal and never crosses a domain line.

**There is no Capstone analogue of CHERI's async-quarantine default**, and no
bounds-only config is implemented yet. So this table cannot separate Capstone's
bounds contribution from its revocation contribution except through rows 6 and
11, which do it incidentally.

## Results (15 rows)

| Row | Defect (class) | revoke | control | Verdict |
|----|----------------|:------:|:-------:|----|
| 1 | rlua #19, Rust userdata freed by Lua `__gc` (UAF) | FAULT | MISS | **BLOCKED** by revocation |
| 2 | rlua #97, escaped `Table` handle — **stack**-use-after-return | MISS | MISS | **MISS** — no allocator involved |
| 3 | libpulse GHSA-f56g-chqp-22m9, `Proplist::Iterator` (UAF) | FAULT | MISS | **BLOCKED** by revocation |
| 4 | mruby CVE-2022-1071, VM-stack UAF (WRITE) | FAULT | MISS | **BLOCKED** by revocation |
| 5 | mruby CVE-2022-1934, `hash_new_from_values` (UAF) | FAULT | MISS | **BLOCKED** by revocation |
| 6 | mruby CVE-2026-1979, bytecode-corruption overflow | FAULT `c7` | FAULT `c7` | **BLOCKED by BOUNDS** (store fault) |
| 7 | RUSTSEC-2022-0070, secp256k1 preallocated context (UAF) | FAULT | MISS | **BLOCKED** by revocation |
| 8 | mruby#4926 / CVE-2020-6838, `hash_values_at` (UAF) | FAULT | MISS | **BLOCKED** by revocation |
| 9 | mruby#3829, irep-pool string UAF (GC sweep) | FAULT `c24` | MISS | **BLOCKED** by revocation |
| 10 | mruby CVE-2022-1106, `OP_RANGE_INC` UAF (WRITE) | FAULT | MISS | **BLOCKED** by revocation |
| 11 | mruby CVE-2018-10191, `OP_GETUPVAR` truncation | FAULT `c5` | FAULT `c5` | **BLOCKED by BOUNDS** (load fault) |
| 12 | mruby#4001, `File#initialize_copy` dangling `DATA_PTR` | FAULT `c24` | MISS | **BLOCKED** by revocation |
| 13 | mruby#4927, `hash_slice` (UAF) | FAULT | MISS | **BLOCKED** by revocation |
| 14 | mruby#3596, GC stack-root scanner (UAF) | FAULT `c24` | MISS | **BLOCKED** by revocation |
| 15 | mruby#3722, `mrb_str_format` argv (UAF) | FAULT | MISS | **BLOCKED** by revocation |

**Tally: 14/15 blocked** — 12 temporal by revocation, 2 spatial by bounds. The
single miss is row 2.

## Against CHERI, same corpus and same shims

| | spatial | async (default) | eager | Capstone revoke |
|---|:---:|:---:|:---:|:---:|
| rows blocked | 2/15 | 2/15 | 14/15 | **14/15** |
| temporal blocked at the contract point | 0/13 | **0/13** | 12/13 | **12/13** |

Capstone's revoke-on-free blocks exactly the rows CHERI's `eager` blocks, and
misses the same one. The difference the table does **not** show is cost and
deployability: CHERI's `eager` is explicitly an aggressive, non-default upper
bound, whereas revoke-on-free is Capstone's design point. CHERI's *deployed*
default blocks 0 of 12 temporal defects at the contract point.

That cost question is not answered here. It belongs to the QEMU-to-QEMU
performance comparison (`capstone/tests/cheri-perf/`), under the PI's rule of
QEMU-to-QEMU for comparison and RTL only for the Capstone absolute.

## Row 2 is the honest floor for both systems

A stack-use-after-return involves no allocator at all, so no allocator-mediated
mechanism — revocation on either system — can observe it. CHERI misses it in all
three configs; Capstone misses it. It is in the corpus precisely because it
marks the boundary of what this class of mechanism can do.

## Two predictions were wrong, both spatial

Rows 6 and 11 were predicted MISS on the reasoning that "revocation is
irrelevant to a spatial defect". True but one step short: `rof_malloc` hands out
SPLIT capabilities with **exact bounds**, so an out-of-bounds access faults
regardless of revocation. Both fault identically with and without the revoke,
which is the signature of a bounds catch. CHERI caught both in all three configs
for the same reason.

The fault causes sort by defect class, which is a stronger check than was
designed in: row 6 is a WRITE overflow and faults `cause = 7` (store), row 11 is
a READ overflow and faults `cause = 5` (load).

## Two fault manifestations, and what each proves

| shape | rows | manifestation |
|---|---|---|
| dereferences the stale pointer directly | 9, 12, 14 | **`cause = 24`, domain halted by the monitor** — a delivered capability fault |
| computes `regs + ACCESS_OFF` first | 1, 3, 4, 5, 8, 10, 13, 15 | QEMU `assert(rs1_v->tag)` in `cscincoffsetimm` |

The second group hits an **emulator gap, not a mechanism failure**:
`op_helper.c` has 13 bare tag asserts against 46 real `riscv_raise_exception`
calls, so QEMU has no model for arithmetic on an untagged capability. That the
revoke itself reaches the delivered-fault path is shown independently by the
`DELIVERY=1` probe (allocate / revoke / dereference at offset 0, with its own
control) and by rows 9, 12 and 14, which take that path naturally.

**What this means for the verdicts.** For all 11 temporal rows the stale access
was *prevented* and the control proves the revoke is why. For the 8 rows in the
second group, whether real hardware delivers a fault on the arithmetic or yields
an untagged capability whose subsequent store faults `cause = 24` is unresolved —
but the row is blocked either way, since the only escape would be arithmetic
that *restored* the tag. Settling which requires the RTL (submodule 404s here)
or silicon.

## Caveats

- **Shims, not real software — on both columns.** Real engines recycle objects
  on internal free lists that no revocation scheme observes, so both columns are
  upper bounds. The bias is symmetric because both compile the identical shim
  against the identical mock allocator, and that symmetry is what makes the
  comparison fair even though neither absolute is realistic.
- **QEMU, not silicon.** Nothing here has run on the FPGA. The shims are small
  bare-metal C, the same size class as the beebs rungs that do run there, so a
  silicon pass is plausible — but R-1 and R-5 may bite.
- **First-pass flake rate was 7/28 (25%)**, all boot failures rather than row
  properties; every one resolved within 3 retries. Individual runs, never
  batched: a faulting domain halts QEMU and every row after it in that boot
  silently never runs.
- **Reproduced from a clean rebuild (2026-08-02), with one caveat worth stating
  precisely.** `./reproduce.sh` wiped the build directory, rebuilt all 15 rows
  and both variants plus the host, re-ran all 30 boots, and reported
  **`REPRODUCED 15/15 rows identical to expected-results.tsv`** — green in a
  single pass. Row 7 is included, and was measured against a prediction
  committed before that run.

  An earlier 14-row attempt (2026-08-02, before row 7 existed) failed at 13/14:
  `mruby_values_at_uaf` hung in the guest bootloader on all 3 attempts, before
  any domain was loaded, and the check correctly FAILED rather than dropping the
  row. That is why the retry budget is now 5 — and why it is capped rather than
  unlimited, and only ever applied to runs that produced NO evidence.

## Reproduce

```bash
cd xlang/capstone
./reproduce.sh                                             # clean rebuild + all 28 + VERIFY
ROW=mruby_range_uaf ./build-xlang-capstone.sh              # fault build
ROW=mruby_range_uaf NO_REVOKE=1 ./build-xlang-capstone.sh  # its control
./run-xlang-capstone.sh                                    # all 14, both variants
```

`DELIVERY=1` builds the fault-delivery probe; `PROBE=1` builds the
share-plumbing probe. Both are diagnostics, not rows.
