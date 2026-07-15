# Task 007 — single-domain held-cap Option B probe (steps 1–3)

*2026-07-09, compiler-lane. Probe at `capstone/tests/runtime-qemu/intra-domain-mrev-revoke-probe/`.
Plan: `plans/sqlite-row3-option-b-held-cap-probe-plan.md` (A). Depends on task-005
(`history/09-07-2026_20-42-10_...`) and task-006 (C1/C2 codegen fixes).*

## Bottom line

**The literal single-domain Option B works, on the real monitor-delivered
capability, at `-O0`, `-O1` and `-O2`.** 24/24 probe runs green. One domain
receives a `REV_TRANSFERRED` linear region, `MREV`s it, uses a `delin`'d alias,
`REVOKE`s, and the cached alias faults — with the arena provably reachable *only*
through the delivered capability.

Two things came out of it that A needs before integrating:

1. **The `.smode` scaffold cannot do this**; the `domain_main` `.dom` path can, and
   needs no glue at all. (Step 0 answer — see "receive protocol" below.)
2. **`MREV`-ing SQLite's own `memsys5` heap pointer is not reachable**, for
   capability-semantics reasons, not effort reasons. (Step 3 verdict.)

## Step 0 — the receive protocol (the plan's open question)

The plan asked which `lcc` index the probe's entry stub must use to capture the
delivered `REGION_SHARE` capability. **There is no index and no entry stub.**

`shared_region_annotated()` ends with `__domcallsaves(d, CAPSTONE_DPI_REGION_SHARE, r)`,
which *enters* the domain with the linear region cap `r` in the capability-argument
register. `my_first_domain/start.S` already saves it (`stc a1, sp, 80`) and reloads
it as `domain_main`'s **first argument** (`ldc a0, sp, 80`); the DPI func code
arrives as the second, scalar argument. So the delivered capability **is**
`domain_main`'s `arg` on the entry whose `func == CAPSTONE_DPI_REGION_SHARE (1)`.

The reason existing runtime probes discard it is structural, and it is worth
recording because the plan's step-0 framing assumed otherwise: those probes are
**`.smode` payloads under the `sbi.dom` scaffold**. There, `handle_dpi()` →
`dpi_share_region()` parks the capability in the *scaffold's* `regions[]`, and the
S-mode payload reaches the bytes through ambient `cpmp` authority after a
`swap_cpmp()` miss. A `.smode` payload never sees a capability at all — and it is
built with Buildroot gcc, which has no capability builtins. Hence this probe's
domain payloads are `domain_main` `.dom` images (Capstone clang, `build-domain.sh`),
loaded with `create_dom(path, NULL)`. **No shared file was touched**: not `start.S`,
not the monitor, not `capstone-c`.

Two grant annotations are forced, not chosen:

- `REV_TRANSFERRED` — hands the region over linear with no monitor-retained
  revocation handle, so the revoke is genuinely intra-domain.
- `PERM_INOUT` (RW), **never** `PERM_IN` (RO) — `helper_cstighten` silently
  delinearises a `LIN` capability whose perms disallow write, and `helper_csmrev`
  *asserts* `CAP_TYPE_LIN`. A read-only grant would abort the emulator rather than
  fault cleanly.

The arena is parked in a `.bss` capability slot across the two domain entries
(`REGION_SHARE`, then `CALL`). Safe: `stc`/`ldc` duplicate rather than move, and
`cap_compress`/`cap_uncompress` round-trip the type field, so it comes back `LIN`.

## Steps 1 + 2 — results (8 probes × `-O0`/`-O1`/`-O2`, one boot each)

| Probe | `-O0` | `-O1` | `-O2` |
|---|---|---|---|
| `held_revoke_fault` (the mechanism) | FAULT **24** | FAULT **25** | FAULT **25** |
| `held_mem_alias_fault` | FAULT 24 | FAULT 24 | FAULT 24 |
| `held_no_revoke_ok` (control) | `0x2230005e` | `0x2230005e` | `0x2230005e` |
| `held_unrelated_ok` | `0x22310033` | `0x22310033` | `0x22310033` |
| `held_ambient_miss` (provenance) | FAULT 24 | FAULT 24 | FAULT 24 |
| `held_split_sibling_ok` | `0x22350044` | `0x22350044` | `0x22350044` |
| `held_protected_value_lifecycle` | FAULT 24 | FAULT 25 | FAULT 25 |
| `held_arena_survives_revoke` | `0x22370077` | `0x22370077` | `0x22370077` |

**The expected cause is asserted, not reported**, and it is opt-level dependent for
the two "held alias" probes. At `-O0` clang spills the alias, so the post-revoke
deref reloads it (`ldc`) and the reload clears the tag → cause 24. At `-O1`/`-O2`
the alias stays in a register across the revoke → cause 25, which is self-proving
(tag intact, node revoked). Both are real faults; only `-O1+` proves it was the
revoke and not a consumed capability. Every cause-24 expectation carries
`held_no_revoke_ok` as its control.

`-O2` asm of the mechanism, the direct evidence:

```
ldc a2, 0(a2)   # the parked grant
mrev a4, a2     # revocation handle, retains the arena
delin a2        # working alias, in place
sb a3, 8(a2)    # live use through the held capability
revoke a4       # lifecycle point
lbu a2, 8(a2)   # USE-AFTER-REVOKE -- no reload, no auipc/gp re-materialisation
```

Independent host-side evidence that the delivery lands where we think: the
controller's ordinary Linux `mmap` of the region reads back `0x5e` / `0x44` at
`arena[8]` — the bytes the domain wrote *through the granted capability*.

Two task-006 fixes are load-bearing here and are confirmed in situ:

- **C1** — the `-O1`/`-O2` payloads would not compile at all without the
  `CC_Capstone_FastCC` capability-argument case (`probe_receive` takes a cap).
- **C2** — `held_no_revoke_ok` at `-O2` never uses its `MREV` result, and the
  `mrev` instruction still survives (`rd != x0`). Under the old `IntrNoMem`
  modelling it would have been DCE'd and the probe would have tested nothing.

## Gap found — a revoked region is a landmine for the host mapping

`REV_TRANSFERRED` leaves a **stale duplicate** of the region capability in the
monitor's `regions[]` (`sbi_capstone.c` says so itself: `// TODO: regions[region_id]
should be added to a free list`) and drops the region's `cpmp` entry. After the
domain revokes that lineage, the next **host** access to the region takes a `cpmp`
miss, and `swap_cpmp()` calls `cap_base(regions[id])` on a capability that now
reloads **untagged**:

```
qemu-system-riscv64: ../target/riscv/op_helper.c:666:
    helper_cslcc: Assertion `rs1_v->tag' failed.
```

QEMU aborts. Found by `held_unrelated_ok`, whose controller read `arena[8]` after
the domain had revoked the whole arena. Triangulated: probes that leave the arena's
node live (`held_no_revoke_ok`, `held_split_sibling_ok`) read the same byte through
the same path with no trouble.

This is a **robustness gap, not a mechanism failure** — nothing about the revoke is
wrong. But it is a real constraint on the SQLite integration: **after a domain
revokes a granted region, the host must not touch that region.** The fixes are
both outside the B lane and need A's call:

- monitor: `swap_cpmp()` should skip untagged `regions[]` entries, and
  `shared_region_annotated()`'s `REV_TRANSFERRED` branch should clear
  `regions[region_id]` (its own TODO);
- emulator: `helper_cslcc`'s `assert(rs1_v->tag)` could be a clean
  `RISCV_EXCP_UNEXP_OP_TYPE` instead of an abort.

Neither was changed. The probes work around it: the controller's post-call read of
its Linux mapping is opt-in and used only where the arena's node stays live.

## Step 3 — memsys5 linear-heap feasibility (spike; A finishes)

**Pointing memsys5 at a monitor-granted linear arena: works.** memsys5 never does
`inttoptr`; allocations are `&mem5.zPool[i * szAtom]` and free/realloc recover the
index with a pointer *difference*. `szAtom` is 64 here, comfortably past the
16-byte alignment `store_capregval` needs. `mem5.aCtrl` lives inside the arena past
the block area, so the grant must cover both. One `delin` plus a changed
`sqlite3_config(SQLITE_CONFIG_HEAP, …)` argument.

**`MREV`-ing a memsys5 allocation: not reachable.** Four independent reasons:

1. `&zPool[k]` lowers to `cincoffset`; `helper_cscincoffset` does `*rd_v = *rs1_v`
   and bumps only `bounds.cursor`. The derived capability inherits the pool's
   `rev_node_id`, type **and bounds** — an allocation is not a distinct capability
   to the revocation tree, and is not even bounds-narrowed to its own block.
2. So `MREV` of an allocation mints a node senior to the **pool's** node: the
   revoke would sweep the entire heap.
3. It cannot get that far regardless. `mem5.zPool = zByte` is a `movc`, and a
   LINEAR cap is consumed by copy (task-005, C3), so the pool must be `delin`'d
   before SQLite may touch it; allocations then come out `NONLIN`, and
   `helper_csmrev` **asserts** `CAP_TYPE_LIN` — an emulator abort.
4. `SPLIT` is the only derivation minting a fresh node (`cap_rev_tree_split`);
   `SHRINK`/`SHRINKTO` copy `rev_node_id` unchanged. But there is **no
   merge/unsplit op**, so splitting is one-way, and memsys5 is a buddy allocator
   that coalesces freed neighbours.

**Scaffold delivered:** `probe_linear_arena.h` — a separate small linear arena from
which the domain carves one `SPLIT` sub-capability per protected value:

```c
probe_arena_init(grant);                         // the granted arena
probe_protected_buf b = probe_arena_carve(256);  // this statement's column-name buffer
… hand b.alias out; it is an ordinary pointer …
probe_arena_revoke(&b);                          // sqlite3_finalize()
… every cached copy of b.alias now faults …
```

`held_protected_value_lifecycle` proves the fault; `held_arena_survives_revoke`
proves a second live value and the un-carved remainder survive. Carving is one-way,
so this suits a bounded number of protected values per grant — one per live
statement — not a general heap.

**Stopped here, per the task.** The remaining decision is corpus fidelity, A's
lane: does row3's "after" copy the protected value into a carved buffer (pragmatic
Option B, works today), or does the PI's bar demand `MREV` of SQLite's own
`memsys5` pointer? The latter needs either per-allocation `SPLIT` plus a new
emulator merge op, or an allocator that never coalesces.

## Scope

Additive only, all under the new probe dir plus its two top-level scripts. No
`start.S`, no monitor, no `capstone-c`, no submodule, no gitlink bump. QEMU boots
were serialised under the `rootfs.ext2` lock announced in `COORDINATION.md`.
