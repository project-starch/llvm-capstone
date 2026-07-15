# LINEAR (row 11) and UNINIT (row 14) closed — 17/17 in-scope corpus rows validated

**Date:** 2026-07-10 (compiler-lane, task 009)
**Supersedes:** `08-07-2026_13-01-23_linear-uninit-rows-blocked-intra-domain.md`
**Probe:** `capstone/tests/runtime-qemu/linear-uninit-corpus-probe/`
+ `build-`/`run-linear-uninit-corpus-probe.sh`

## Bottom line

Both deferred rows are **validated on RTL**, at `-O0`, `-O1` and `-O2`, 21/21 probe
runs green. The 2026-07-08 deferral was correct when it was written and is now
stale: each of its two blockers was removed by later compiler-lane work.

| Blocker (2026-07-08) | Removed by |
|---|---|
| row 11: `csdrop` unimplemented in the emulator | task 002 — `helper_csdrop` |
| row 14: no intra-domain linear authority (the #78 wall) | task 007 — the `domain_main` receive protocol |

Neither row needed the escape hatches the deferral note scoped out. There is **no**
`share_uninit_region` monitor op, **no** `start.S` change, and **no** firmware
rebuild. One small emulator fix was required, described below.

## Row 14 — UNINIT, use-before-init

### The derivation

The deferral note said "a domain cannot mint an UNINIT cap", because minting one
needs a revoke, a revoke needs `mrev`, and `csmrev` asserts `CAP_TYPE_LIN` — which a
domain had no way to obtain. Task 007 removed exactly that premise: a
`REV_TRANSFERRED` region arrives at `domain_main` as a real LINEAR capability.

What remained was to find which revoke yields UNINIT rather than LIN. It is decided
by one bit, and the naming hides it. `cap_rev_tree_revoke()` computes
`retain_data = AND(!node.linear)` over the junior run it invalidates, and
`helper_csrevoke` (op_helper.c:709) stores that into a local called `is_linear`:

```c
bool is_linear = cap_rev_tree_revoke(&env->cr_tree, rs1_v->val.cap.rev_node_id);
rs1_v->val.cap.type = is_linear ? CAP_TYPE_LIN : CAP_TYPE_UNINIT;
```

So: revoke a lineage whose nodes were **all delinearised** and the retained handle
comes back LIN (data retained, cursor at base). Revoke a lineage that is **still
linear** and it comes back UNINIT (data not retained, cursor at end — the canonical
form `csinit` asserts on).

The held-cap probe always `delin`s the arena before using it, which is why its
revoke yields LIN. Dropping that one `delin` is the whole derivation:

```c
void *arena = probe_arena;                          /* LINEAR grant, node N */
void *rev   = __builtin_capstone_cap_mrev(arena);   /* node M, senior to N  */
return __builtin_capstone_cap_revoke(rev);          /* N was linear -> UNINIT */
```

This is a faithful model of the row: a freshly `calloc`'d connection wrapper whose
`sqlite3 *db` names storage it has no authority to read yet.

### The emulator fix (required, in-lane)

`_helper_access_with_cap` never looked at the capability's **type**. An UNINIT
capability was denied only incidentally, by the bounds check: `csrevoke` parks its
cursor at `end`, so `*db` is out of bounds and raised cause 5. A load at a
**negative** offset was inside `[base, end)`, cleared the bounds check, and returned
the stale byte — the exact disclosure UNINIT exists to prevent.

The spec is unambiguous and already had the exception code we needed. Every load
form — `lb/lbu/lh/lhu/lw/lwu/ld` (`spec/parts/existing-insn.adoc`, "Load
Instructions") and `LDC` (`spec/parts/mem-access-insn.adoc`, `[#load-cap]`) — lists
the permitted `x[rs1].type` as `0` (linear), `1` (non-linear), `5` (sealed-return),
and raises **`Unexpected capability type (26)`** otherwise. `RISCV_EXCP_UNEXP_CAP_TYPE
= 0x1a` was defined in `cpu_bits.h` and never raised anywhere.

So `_helper_access_with_cap` now denies loads through an UNINIT capability with
cause 26, before the bounds check. Stores are deliberately untouched: the spec
*permits* type 3 as the address operand of `sb/sh/sw/sd`/`STC` (that is how an owner
initialises memory, `imm` must be 0 and the cursor advances). This emulator
implements neither the `imm != 0` rejection nor the cursor advance; with cursor at
`end` an uninit store is out of bounds anyway, so nothing in this probe depends on
it. Left as a known, spec-visible partial implementation.

`uninit_negative_offset_fault` is the regression test for exactly this: it is an
in-bounds load, so it **returns instead of faulting** on an emulator without the
check. It was written to fail, and it did.

### Results

| Probe | `-O0` | `-O1` | `-O2` |
|---|---|---|---|
| `uninit_use_before_init_fault` (the mechanism) | FAULT **26** | FAULT **26** | FAULT **26** |
| `uninit_negative_offset_fault` (type, not bounds) | FAULT **26** | FAULT **26** | FAULT **26** |
| `uninit_init_then_use_ok` (csinit reclaims) | `0x1412005e` | `0x1412005e` | `0x1412005e` |

Cause 26 is **self-proving**: only an UNINIT-typed capability raises it, so unlike a
cause-24 expectation it needs no companion control to rule out an unrelated cause.
`uninit_init_then_use_ok` is still run, and carries the row's second half — after
`csinit` (the model of a successful `sqlite3_open`) the *same handle* reads and
writes the *same bytes*. It also proves the trap was not dead memory or a botched
grant. `csinit` is a required, explicit reclaim step: it asserts UNINIT with
cursor == end, and consumes its input (UNINIT is not copyable).

The cause does **not** move with the optimisation level, unlike the held-cap probe's.
The UNINIT handle keeps its tag and its own (valid) revocation node, so a spill and
reload changes nothing — `-O2` gives `revoke a2; lbu a2, 0(a2)`, `-O0` gives the same
load off a stack-reloaded register.

## Row 11 — LINEAR, double-free

`csdrop` (task 002) clears the register's tag: a consumed capability is the canonical
null, and any later use raises cause 24 rather than executing as an illegal
instruction. The statement handle is carved out of the arena with `cssplit`, so it is
a LINEAR capability with a revocation node of its own.

```c
void *stmt = corpus_carve_stmt();                 /* split(arena, mid): LINEAR */
void *gone = __builtin_capstone_cap_drop(stmt);   /* sqlite3_finalize() #1     */
volatile char v = *(volatile char *)gone;         /* #2: nothing left to use   */
```

| Probe | `-O0` | `-O1` | `-O2` |
|---|---|---|---|
| `linear_drop_use_fault` (use after consume) | FAULT 24 | FAULT 24 | FAULT 24 |
| `linear_double_drop_fault` (the literal shape) | FAULT 24 | FAULT 24 | FAULT 24 |
| `linear_no_drop_ok` (control) | `0x11120033` | `0x11120033` | `0x11120033` |
| `linear_drop_sibling_ok` (drop ≠ free) | `0x11130044` | `0x11130044` | `0x11130044` |

Cause 24 only says "no capability in this register", which a consumed linear
capability (task-005, C3) and a reloaded revoked capability both produce. So
`linear_no_drop_ok` runs the identical carve and deref with the `csdrop` removed and
must reach its return.

The two fault probes are distinguished in the log, not just by cause.
`linear_drop_use_fault` traps in `_helper_access_with_cap` ("Cap mem access requires
capability"); `linear_double_drop_fault` traps inside `helper_csdrop` ("DROP requires
capability"), which is what the Go binding actually does — it never dereferences the
statement, it hands it back to SQLite a second time. A `CAPSTONE_DEBUG_PRINT` was
added to `helper_csdrop`'s untagged branch so the two are tellable apart.

`linear_drop_sibling_ok` guards against overclaiming. **`csdrop` consumes a handle;
it does not revoke a lineage or free memory.** It clears one register. Row 11 is not
"the allocator refuses a second free" — it is "linearity leaves no second capability,
so the second `finalize` has nothing to hand back". The connection the statement was
carved from keeps working after the drop, and the host's ordinary Linux `mmap` of the
region independently reads back `0x44` at `arena[8]` — a byte written through the
surviving low half *after* the drop.

## Fidelity bar

Mechanism probes on RTL, with the exact fault cause asserted and a control for every
cause-24 expectation — the same bar rows 3/13/18/19 (R), 4–12 (H) and 1/2/6/16 (S)
were validated at. Not real-SQLite matched pairs; those remain optional for every row.

## Scope

- `capstone-qemu` (the compiler lane): the UNINIT-load check in `_helper_access_with_cap`, and
  the `DROP requires capability` diagnostic in `helper_csdrop`. Submodule bump.
- Superproject: the new probe dir + its two top-level scripts, this note, and the
  `stage2-mapping.md` validation table.
- Untouched: `start.S`, the monitor (`sbi_capstone.c`), `capstone-c`, `caplifive-buildroot`,
  A's existing runtime-qemu probes, the LLVM tree.

QEMU boots were serialised under the `rootfs.ext2` lock announced in `COORDINATION.md`.

## Note for A — the revoked-region host landmine still applies

Unchanged from task 007, and it constrains these probes too: after a domain revokes a
`REV_TRANSFERRED` region, the **host must not touch it**. `swap_cpmp()` reloads the
monitor's stale `regions[]` duplicate untagged and `cap_base()`'s `lcc` aborts QEMU.
Every row-14 probe revokes the arena by construction, so the controller's post-call
read of its Linux mapping is opt-in and used only by `linear_drop_sibling_ok`, which
revokes nothing. Fixes are still unapplied and still need A's call (monitor:
skip untagged `regions[]`, clear `regions[id]` on TRANSFERRED; emulator: make the
`lcc` assert a clean cause-24 fault).
